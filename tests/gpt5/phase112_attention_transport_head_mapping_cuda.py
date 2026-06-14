#!/usr/bin/env python3
"""
Phase 112: attention transport head mapping.

Pipeline:
  1. Build the same answer-site transport direction T_c used in Phase111.
  2. Scan answer_last attention mass from peak-3...peak layers to source groups.
  3. Select top heads by object/source mass.
  4. Ablate selected heads at answer_last via o_proj input head slices.
  5. Measure final DCF logits and answer_last T_c projection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase106_multitemplate_residual_cuda import TEMPLATES, find_subsequence, object_last_position  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, capture_centers, score_logits, summarize_delta  # noqa: E402
from phase109_support_suppressor_decomposition_cuda import build_readout_directions  # noqa: E402
from phase111_transport_path_causal_mapping_cuda import build_transport_components  # noqa: E402
from phase110_orthogonal_subspace_split_cuda import capture_transport_dirs  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase112_attention_transport_head_mapping")
TEST_CATEGORIES = ["number", "time", "container", "clothing", "furniture", "plant"]
SOURCE_GROUPS = ["object_span", "object_last", "pre_object", "post_object", "self"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_attention_module(layer: Any) -> Any:
    for name in ["self_attn", "self_attention", "attention", "attn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise TypeError(f"Cannot find attention module for {type(layer).__name__}")


def get_o_proj(attn: Any) -> Any:
    for name in ["o_proj", "dense", "out_proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise TypeError(f"Cannot find attention output projection for {type(attn).__name__}")


def get_num_heads(model: Any, attn: Any) -> int:
    for obj in [attn, getattr(model, "config", None)]:
        if obj is None:
            continue
        for name in ["num_heads", "num_attention_heads", "n_head"]:
            value = getattr(obj, name, None)
            if value:
                return int(value)
    raise TypeError("Cannot infer number of attention heads")


def object_span_positions(tokenizer: Any, prompt: str, obj: str, fallback: int) -> list[int]:
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    obj_ids = tokenizer(obj, add_special_tokens=False)["input_ids"]
    start = find_subsequence(full_ids, obj_ids)
    if start is None:
        obj_ids = tokenizer(" " + obj, add_special_tokens=False)["input_ids"]
        start = find_subsequence(full_ids, obj_ids)
    if start is None:
        return [fallback]
    return [p for p in range(start, min(start + len(obj_ids), fallback + 1))]


def group_indices(obj_span: list[int], answer_pos: int) -> dict[str, list[int]]:
    obj_last = obj_span[-1]
    obj_set = set(obj_span)
    pre = [i for i in range(0, answer_pos) if i < obj_span[0]]
    post = [i for i in range(0, answer_pos) if i not in obj_set and i >= obj_span[0]]
    return {
        "object_span": obj_span,
        "object_last": [obj_last],
        "pre_object": pre,
        "post_object": post,
        "self": [answer_pos],
    }


def build_prompts(cat: str, train_n: int, test_n: int) -> list[dict[str, str]]:
    prompts = []
    for tpl in TEMPLATES:
        for obj in CATEGORY_OBJECTS[cat][train_n:train_n + test_n]:
            prompts.append({"obj": obj, "prompt": tpl["text"].format(obj=obj)})
    return prompts


def scan_attention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[dict[str, str]],
    patch_layers: list[int],
    num_heads: int,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    sums = {g: np.zeros((len(patch_layers), num_heads), dtype=np.float64) for g in SOURCE_GROUPS}
    counts = 0
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            items = prompts[start:start + batch_size]
            texts = [x["prompt"] for x in items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            spans = [object_span_positions(tokenizer, item["prompt"], item["obj"], answer_pos[bi]) for bi, item in enumerate(items)]
            out = model(**batch, output_attentions=True, use_cache=False)
            if out.attentions is None:
                raise RuntimeError("Model did not return attentions")
            for li, layer_id in enumerate(patch_layers):
                attn = out.attentions[layer_id - 1].detach().float().cpu().numpy()
                for bi, ans in enumerate(answer_pos):
                    groups = group_indices(spans[bi], ans)
                    row = attn[bi, :, ans, :]
                    for g, idxs in groups.items():
                        if idxs:
                            sums[g][li] += row[:, idxs].sum(axis=1)
                counts += len(items)
            del out, batch
            torch.cuda.empty_cache()
    return {g: (v / max(counts, 1)).astype(np.float32) for g, v in sums.items()}


def make_head_ablation_pre_hook(num_heads: int, head_id: int, positions: torch.Tensor):
    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        head_dim = x.shape[-1] // num_heads
        y = x.clone()
        batch_idx = torch.arange(y.shape[0], device=y.device)
        pos = positions.to(y.device)
        y_view = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        y_view[batch_idx, pos, head_id, :] = 0
        return (y,) + inputs[1:]

    return hook


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, str]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    monitor_layer: int,
    monitor_direction: np.ndarray,
    batch_size: int,
    max_length: int,
    patch_layer: int | None = None,
    head_id: int | None = None,
    num_heads: int | None = None,
) -> dict[str, np.ndarray]:
    score_chunks = []
    proj_chunks = []
    d = torch.tensor(monitor_direction, device=device, dtype=torch.float32)
    d = d / (d.norm() + 1e-8)
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        handle = None
        if patch_layer is not None and head_id is not None and num_heads is not None:
            attn = get_attention_module(layers[patch_layer - 1])
            o_proj = get_o_proj(attn)
            handle = o_proj.register_forward_pre_hook(
                make_head_ablation_pre_hook(num_heads, head_id, answer_pos)
            )
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        if handle is not None:
            handle.remove()
        pos_gpu = answer_pos.to(out.logits.device)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        score_chunks.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        bidx = torch.arange(hs.shape[0], device=hs.device)
        ans = hs[bidx, answer_pos.to(hs.device), :].float()
        proj_chunks.append((ans @ d.to(hs.device)).detach().float().cpu().numpy())
        del out, batch
        torch.cuda.empty_cache()
    return {
        "scores": np.concatenate(score_chunks, axis=0),
        "answer_proj": np.concatenate(proj_chunks, axis=0),
    }


def select_heads(attn_scan: dict[str, np.ndarray], patch_layers: list[int], top_k: int) -> list[dict[str, Any]]:
    object_mass = attn_scan["object_span"] + attn_scan["object_last"]
    flat = []
    for li, layer_id in enumerate(patch_layers):
        for head_id in range(object_mass.shape[1]):
            flat.append({
                "patch_layer": layer_id,
                "head_id": head_id,
                "object_mass": float(object_mass[li, head_id]),
                "object_span_mass": float(attn_scan["object_span"][li, head_id]),
                "object_last_mass": float(attn_scan["object_last"][li, head_id]),
                "pre_object_mass": float(attn_scan["pre_object"][li, head_id]),
                "post_object_mass": float(attn_scan["post_object"][li, head_id]),
                "self_mass": float(attn_scan["self"][li, head_id]),
            })
    flat.sort(key=lambda x: x["object_mass"], reverse=True)
    return flat[:top_k]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        cat_local_ids, readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        readout_dirs = build_readout_directions(readout_rows.astype(np.float32), cat_local_ids, categories)
        monitor_layer = args.monitor_layer if args.monitor_layer is not None else BOUNDARY_LAYER[args.model]
        patch_layers = list(range(max(1, monitor_layer - args.layer_back), monitor_layer + 1))
        num_heads = get_num_heads(model, get_attention_module(layers[monitor_layer - 1]))
        alloc, reserved = vram_gb()
        log(f"{args.model}: monitor=L{monitor_layer}, patch_layers={patch_layers}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")

        centers = capture_centers(model, tokenizer, device, categories, monitor_layer, args.train_objects, args.batch_size, args.max_length)
        transport_dirs = capture_transport_dirs(model, tokenizer, device, categories, monitor_layer, args.train_objects, args.batch_size, args.max_length)
        components = build_transport_components(centers, transport_dirs, readout_dirs, categories)

        result = {
            "phase": 112,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "monitor_layer": monitor_layer,
            "patch_layers": patch_layers,
            "num_heads": num_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "top_k_heads": args.top_k_heads,
            "test_categories": test_categories,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for idx, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {idx}/{len(test_categories)} {cat}")
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            target_idx = categories.index(cat)
            monitor_dir = components[cat]["transport"]
            if np.linalg.norm(monitor_dir) < 1e-8:
                monitor_dir = components[cat]["raw_transport"]
            attn_scan = scan_attention(
                model, tokenizer, device, prompts, patch_layers, num_heads,
                args.batch_size, args.max_length
            )
            selected = select_heads(attn_scan, patch_layers, args.top_k_heads)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                monitor_layer, monitor_dir, args.batch_size, args.max_length
            )
            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_transport_proj": float(baseline["answer_proj"].mean()),
                "attention_scan": {g: attn_scan[g].tolist() for g in SOURCE_GROUPS},
                "selected_heads": selected,
                "conditions": [],
            }
            for item in selected:
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    monitor_layer, monitor_dir, args.batch_size, args.max_length,
                    patch_layer=item["patch_layer"], head_id=item["head_id"], num_heads=num_heads
                )
                summary = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
                cat_out["conditions"].append({
                    **item,
                    **summary,
                    "answer_transport_proj_delta": float((patched["answer_proj"] - baseline["answer_proj"]).mean()),
                })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 112 Attention Transport Head Mapping: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; heads: {result['num_heads']}")
    lines.append("")
    lines.append("| category | top attention head | strongest target-down ablation | strongest projection-down ablation |")
    lines.append("|---|---|---|---|")
    for cat, item in result["category_results"].items():
        selected = item["selected_heads"]
        conds = item["conditions"]
        top = selected[0] if selected else None
        best_t = min(conds, key=lambda c: c["target_delta"]) if conds else None
        best_p = min(conds, key=lambda c: c["answer_transport_proj_delta"]) if conds else None

        def fmt(c):
            if c is None:
                return "NA"
            return (
                f"L{c['patch_layer']} H{c['head_id']} obj{c['object_mass']:.3f} "
                f"T{c.get('target_delta', 0.0):+.2f} A{c.get('answer_transport_proj_delta', 0.0):+.2f}"
            )

        lines.append(f"| {cat} | {fmt(top)} | {fmt(best_t)} | {fmt(best_p)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=12)
    parser.add_argument("--test-objects", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--monitor-layer", type=int, default=None)
    parser.add_argument("--layer-back", type=int, default=3)
    parser.add_argument("--top-k-heads", type=int, default=8)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase112_{args.model}_attention_transport_head_mapping.json"
    md_path = out_dir / f"phase112_{args.model}_attention_transport_head_mapping.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
