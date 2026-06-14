#!/usr/bin/env python3
"""
Phase 113: head-set and MLP relay closure.

This phase asks whether cumulative attention head sets, MLP output ablation, or
their combination can approach the strong answer-site T_c removal effects from
Phase111.
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
from phase106_multitemplate_residual_cuda import TEMPLATES, object_last_position  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, capture_centers, score_logits, summarize_delta  # noqa: E402
from phase109_support_suppressor_decomposition_cuda import build_readout_directions  # noqa: E402
from phase110_orthogonal_subspace_split_cuda import capture_transport_dirs  # noqa: E402
from phase111_transport_path_causal_mapping_cuda import build_transport_components, make_transport_hook  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import (  # noqa: E402
    build_prompts,
    get_attention_module,
    get_num_heads,
    get_o_proj,
    scan_attention,
    select_heads,
)


OUT_ROOT = Path("results/gpt5_phase113_head_set_mlp_relay_closure")
TEST_CATEGORIES = ["number", "container", "clothing", "plant"]
SET_SIZES = [1, 2, 4, 8, 16]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_mlp_module(layer: Any) -> Any:
    for name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise TypeError(f"Cannot find MLP module for {type(layer).__name__}")


def deterministic_heads(num_heads: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.permutation(num_heads)[:n]]


def make_head_set_pre_hook(num_heads: int, head_ids: list[int], positions: torch.Tensor):
    head_ids = sorted(set(int(h) for h in head_ids))

    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        head_dim = x.shape[-1] // num_heads
        y = x.clone()
        batch_idx = torch.arange(y.shape[0], device=y.device)
        pos = positions.to(y.device)
        y_view = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        y_view[batch_idx[:, None], pos[:, None], torch.tensor(head_ids, device=y.device), :] = 0
        return (y,) + inputs[1:]

    return hook


def make_mlp_zero_hook(positions: torch.Tensor):
    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        batch_idx = torch.arange(out.shape[0], device=out.device)
        pos = positions.to(out.device)
        out[batch_idx, pos, :] = 0
        if rest is not None:
            return (out,) + rest
        return out

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
    num_heads: int,
    head_set: list[dict[str, Any]] | None = None,
    mlp_layers: list[int] | None = None,
    tc_remove: bool = False,
    tc_scale: float = 1.5,
) -> dict[str, np.ndarray]:
    score_chunks = []
    proj_chunks = []
    d = torch.tensor(monitor_direction, device=device, dtype=torch.float32)
    d = d / (d.norm() + 1e-8)
    head_set = head_set or []
    mlp_layers = mlp_layers or []
    by_layer: dict[int, list[int]] = {}
    for item in head_set:
        by_layer.setdefault(int(item["patch_layer"]), []).append(int(item["head_id"]))

    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu()
        object_pos = torch.tensor([
            object_last_position(tokenizer, item["prompt"], item["obj"], int(answer_pos[bi]))
            for bi, item in enumerate(items)
        ], dtype=torch.long)
        handles = []
        for layer_id, heads in by_layer.items():
            attn = get_attention_module(layers[layer_id - 1])
            handles.append(get_o_proj(attn).register_forward_pre_hook(
                make_head_set_pre_hook(num_heads, heads, answer_pos)
            ))
        for layer_id in mlp_layers:
            handles.append(get_mlp_module(layers[layer_id - 1]).register_forward_hook(
                make_mlp_zero_hook(answer_pos)
            ))
        if tc_remove:
            handles.append(layers[monitor_layer - 1].register_forward_hook(
                make_transport_hook(
                    d,
                    answer_pos,
                    "remove_target",
                    tc_scale,
                )
            ))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()

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


def unique_head_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        key = (int(item["patch_layer"]), int(item["head_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
        set_sizes = [int(x) for x in args.set_sizes.split(",") if x.strip()]
        alloc, reserved = vram_gb()
        log(f"{args.model}: monitor=L{monitor_layer}, layers={patch_layers}, heads={num_heads}, vram={alloc:.2f}/{reserved:.2f}GB")

        centers = capture_centers(model, tokenizer, device, categories, monitor_layer, args.train_objects, args.batch_size, args.max_length)
        transport_dirs = capture_transport_dirs(model, tokenizer, device, categories, monitor_layer, args.train_objects, args.batch_size, args.max_length)
        components = build_transport_components(centers, transport_dirs, readout_dirs, categories)

        result = {
            "phase": 113,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "monitor_layer": monitor_layer,
            "patch_layers": patch_layers,
            "num_heads": num_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "set_sizes": set_sizes,
            "candidate_heads": args.candidate_heads,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            target_idx = categories.index(cat)
            monitor_dir = components[cat]["transport"]
            if np.linalg.norm(monitor_dir) < 1e-8:
                monitor_dir = components[cat]["raw_transport"]

            attn_scan = scan_attention(
                model, tokenizer, device, prompts, patch_layers, num_heads,
                args.batch_size, args.max_length
            )
            source_heads = select_heads(attn_scan, patch_layers, args.candidate_heads)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                monitor_layer, monitor_dir, args.batch_size, args.max_length, num_heads
            )
            # Re-score source candidates singly to pick projection-down heads within an expanded source pool.
            single_conditions = []
            for item in source_heads:
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    monitor_layer, monitor_dir, args.batch_size, args.max_length, num_heads,
                    head_set=[item]
                )
                summary = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
                single_conditions.append({
                    **item,
                    **summary,
                    "answer_transport_proj_delta": float((patched["answer_proj"] - baseline["answer_proj"]).mean()),
                })
            projection_heads = sorted(single_conditions, key=lambda x: x["answer_transport_proj_delta"])
            target_heads = sorted(single_conditions, key=lambda x: x["target_delta"])
            mixed_heads = unique_head_items(source_heads[: args.candidate_heads // 2] + projection_heads[: args.candidate_heads // 2])

            tc_ref = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                monitor_layer, monitor_dir, args.batch_size, args.max_length, num_heads,
                tc_remove=True, tc_scale=args.tc_scale
            )
            tc_summary = summarize_delta(tc_ref["scores"] - baseline["scores"], target_idx, categories)

            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_transport_proj": float(baseline["answer_proj"].mean()),
                "tc_remove_reference": {
                    **tc_summary,
                    "answer_transport_proj_delta": float((tc_ref["answer_proj"] - baseline["answer_proj"]).mean()),
                    "scale": args.tc_scale,
                },
                "source_heads": source_heads,
                "single_head_conditions": single_conditions,
                "conditions": [],
            }

            head_sets = {
                "source": source_heads,
                "projection": projection_heads,
                "target": target_heads,
                "mixed": mixed_heads,
            }
            for size in set_sizes:
                random_heads = [
                    {"patch_layer": patch_layers[i % len(patch_layers)], "head_id": h, "object_mass": 0.0}
                    for i, h in enumerate(deterministic_heads(num_heads, size, 9100 + target_idx + size))
                ]
                for set_name, candidates in {**head_sets, "random": random_heads}.items():
                    selected = unique_head_items(candidates)[:size]
                    if not selected:
                        continue
                    for relay_name, mlp_layers in [
                        ("heads_only", []),
                        ("mlp_only", patch_layers if set_name == "source" else []),
                        ("heads_plus_mlp", patch_layers),
                    ]:
                        if relay_name == "mlp_only" and set_name != "source":
                            continue
                        if relay_name == "heads_only":
                            hs = selected
                        elif relay_name == "mlp_only":
                            hs = []
                        else:
                            hs = selected
                        patched = run_condition(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            monitor_layer, monitor_dir, args.batch_size, args.max_length, num_heads,
                            head_set=hs, mlp_layers=mlp_layers
                        )
                        summary = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
                        denom = tc_summary["target_delta"]
                        ratio = float(summary["target_delta"] / denom) if abs(denom) > 1e-6 else 0.0
                        cat_out["conditions"].append({
                            "set_name": set_name,
                            "relay": relay_name,
                            "set_size": size,
                            "heads": [{"patch_layer": int(x["patch_layer"]), "head_id": int(x["head_id"])} for x in hs],
                            "mlp_layers": mlp_layers,
                            **summary,
                            "answer_transport_proj_delta": float((patched["answer_proj"] - baseline["answer_proj"]).mean()),
                            "effect_ratio_vs_tc_remove": ratio,
                        })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 113 Head Set MLP Relay Closure: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; patch layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | T_c reference | best heads only | best heads+MLP | best MLP only | best random |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]

        def pick(fn):
            xs = [c for c in conds if fn(c)]
            return min(xs, key=lambda c: c["target_delta"]) if xs else None

        def fmt(c):
            if c is None:
                return "NA"
            return f"{c.get('set_name','tc')} {c.get('relay','')} k{c.get('set_size','')} T{c['target_delta']:+.2f} R{c.get('effect_ratio_vs_tc_remove',1.0):+.2f} A{c.get('answer_transport_proj_delta',0.0):+.2f}"

        tc = item["tc_remove_reference"]
        lines.append(
            f"| {cat} | {fmt(tc)} | "
            f"{fmt(pick(lambda c: c['relay'] == 'heads_only' and c['set_name'] != 'random'))} | "
            f"{fmt(pick(lambda c: c['relay'] == 'heads_plus_mlp' and c['set_name'] != 'random'))} | "
            f"{fmt(pick(lambda c: c['relay'] == 'mlp_only'))} | "
            f"{fmt(pick(lambda c: c['set_name'] == 'random'))} |"
        )
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
    parser.add_argument("--candidate-heads", type=int, default=16)
    parser.add_argument("--set-sizes", default="1,2,4,8,16")
    parser.add_argument("--tc-scale", type=float, default=1.5)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase113_{args.model}_head_set_mlp_relay_closure.json"
    md_path = out_dir / f"phase113_{args.model}_head_set_mlp_relay_closure.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
