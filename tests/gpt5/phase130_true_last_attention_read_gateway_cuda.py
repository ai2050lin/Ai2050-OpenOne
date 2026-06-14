#!/usr/bin/env python3
"""
Phase 130: true-last attention read gateway mapping.

Use corrected token positions to test whether the true last layer transfers the
pre-answer residual causal field into the answer token through attention and
answer-site updates.
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
from phase106_multitemplate_residual_cuda import TEMPLATES  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, score_logits, summarize_delta  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase126_residual_gap_decomposition_cuda import replace_output_tensor, tensor_from_output  # noqa: E402
from phase128_final_block_gateway_cuda import get_final_norm  # noqa: E402
from phase129_position_corrected_gateway_audit_cuda import (  # noqa: E402
    corrected_positions_for_site,
    first_nonpad_positions,
    last_nonpad_positions,
)


OUT_ROOT = Path("results/gpt5_phase130_true_last_attention_read_gateway")
TEST_CATEGORIES = ["number", "container", "plant"]
ANSWER_COMPONENTS = [
    "last_attention_output_answer",
    "last_mlp_input_answer",
    "last_mlp_output_answer",
    "last_block_output_answer",
    "final_norm_output_answer",
]
REFERENCE_COMPONENT = "last_input_pre_answer"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_answer_component_module(model: Any, layers: list[Any], last_layer: int, component: str) -> tuple[Any, str]:
    layer = layers[last_layer - 1]
    if component == "last_attention_output_answer":
        return get_attention_module(layer), "post"
    if component == "last_mlp_input_answer":
        return get_mlp_module(layer), "pre"
    if component == "last_mlp_output_answer":
        return get_mlp_module(layer), "post"
    if component == "last_block_output_answer":
        return layer, "post"
    if component == "final_norm_output_answer":
        module = get_final_norm(model)
        if module is None:
            raise RuntimeError("final norm unavailable")
        return module, "post"
    raise ValueError(component)


def make_capture_hook(kind: str, store: dict[str, torch.Tensor]):
    if kind == "pre":
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            store["value"] = inputs[0].detach()
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        store["value"] = tensor_from_output(output).detach()
    return hook, False


def make_subspace_patch_hook(kind: str, basis: torch.Tensor, batch_positions: list[list[int]], scale: float):
    basis = basis / (basis.norm(dim=1, keepdim=True) + 1e-8)

    def patch_tensor(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        b = basis.to(out.device).float()
        for bi, positions in enumerate(batch_positions):
            if not positions:
                continue
            pos = torch.tensor(positions, device=out.device, dtype=torch.long)
            vecs = out[bi, pos, :].float()
            proj = (vecs @ b.T) @ b
            out[bi, pos, :] = out[bi, pos, :] - scale * proj.to(out.dtype)
        return out

    if kind == "pre":
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            return (patch_tensor(inputs[0]),) + inputs[1:]
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        return replace_output_tensor(output, patch_tensor(tensor_from_output(output)))
    return hook, False


def make_head_ablation_pre_hook(num_heads: int, head_id: int, answer_positions: torch.Tensor):
    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        head_dim = x.shape[-1] // num_heads
        y = x.clone()
        y_view = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        bidx = torch.arange(y.shape[0], device=y.device)
        pos = answer_positions.to(y.device)
        y_view[bidx, pos, head_id, :] = 0
        return (y,) + inputs[1:]
    return hook


def batch_position_context(tokenizer: Any, batch: dict[str, torch.Tensor], items: list[dict[str, Any]]) -> tuple[list[int], list[int], list[list[int]], list[list[int]]]:
    first_pos = first_nonpad_positions(batch["attention_mask"])
    last_pos = last_nonpad_positions(batch["attention_mask"])
    token_rows = batch["input_ids"].detach().cpu().tolist()
    pre_positions = [
        corrected_positions_for_site(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi], "pre_answer")
        for bi, item in enumerate(items)
    ]
    answer_positions = [[p] for p in last_pos]
    return first_pos, last_pos, pre_positions, answer_positions


def capture_answer_component_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    categories: list[str],
    last_layer: int,
    component: str,
    train_objects: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    centers = np.zeros((len(TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(TEMPLATES), len(categories)), dtype=np.int64)
    items = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_objects]:
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})
    module, kind = get_answer_component_module(model, layers, last_layer, component)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            _first, last_pos, _pre, _ans = batch_position_context(tokenizer, batch, batch_items)
            store: dict[str, torch.Tensor] = {}
            hook_fn, is_pre = make_capture_hook(kind, store)
            handle = module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn)
            _ = model(**batch, use_cache=False)
            handle.remove()
            value = store["value"]
            for bi, item in enumerate(batch_items):
                vec = value[bi, last_pos[bi], :].float().detach().cpu().numpy()
                centers[item["ti"], item["ci"]] += vec
                counts[item["ti"], item["ci"]] += 1
            del batch
            torch.cuda.empty_cache()
    return (centers / counts[:, :, None]).astype(np.float32)


def capture_last_input_pre_answer_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    categories: list[str],
    last_layer: int,
    train_objects: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    centers = np.zeros((len(TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(TEMPLATES), len(categories)), dtype=np.int64)
    items = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_objects]:
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})
    module = layers[last_layer - 1]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in batch_items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            _first, _last, pre_positions, _ans = batch_position_context(tokenizer, batch, batch_items)
            store: dict[str, torch.Tensor] = {}
            handle = module.register_forward_pre_hook(lambda _m, inputs: store.setdefault("value", inputs[0].detach()))
            _ = model(**batch, use_cache=False)
            handle.remove()
            value = store["value"]
            for bi, item in enumerate(batch_items):
                pos = torch.tensor(pre_positions[bi], device=value.device, dtype=torch.long)
                vec = value[bi, pos, :].float().mean(dim=0).detach().cpu().numpy()
                centers[item["ti"], item["ci"]] += vec
                counts[item["ti"], item["ci"]] += 1
            del batch
            torch.cuda.empty_cache()
    return (centers / counts[:, :, None]).astype(np.float32)


def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    cat_local_ids: dict[str, list[int]],
    categories: list[str],
    batch_size: int,
    max_length: int,
    monitor_layer: int,
    monitor_basis: np.ndarray,
    last_layer: int,
    patch_component: str | None = None,
    patch_basis: np.ndarray | None = None,
    scale: float = 1.5,
    ablate_head: int | None = None,
    num_heads: int | None = None,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        _first, last_pos, pre_positions, answer_positions = batch_position_context(tokenizer, batch, items)
        handles = []
        if patch_component == REFERENCE_COMPONENT and patch_basis is not None:
            basis = torch.tensor(patch_basis, device=device, dtype=torch.float32)
            hook_fn, is_pre = make_subspace_patch_hook("pre", basis, pre_positions, scale)
            handles.append(layers[last_layer - 1].register_forward_pre_hook(hook_fn))
        elif patch_component is not None and patch_basis is not None:
            module, kind = get_answer_component_module(model, layers, last_layer, patch_component)
            basis = torch.tensor(patch_basis, device=device, dtype=torch.float32)
            hook_fn, is_pre = make_subspace_patch_hook(kind, basis, answer_positions, scale)
            handles.append(module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn))
        if ablate_head is not None:
            if num_heads is None:
                raise RuntimeError("num_heads required for head ablation")
            attn = get_attention_module(layers[last_layer - 1])
            o_proj = get_o_proj(attn)
            handles.append(o_proj.register_forward_pre_hook(
                make_head_ablation_pre_hook(num_heads, ablate_head, torch.tensor(last_pos, dtype=torch.long))
            ))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for handle in handles:
            handle.remove()
        pos_gpu = torch.tensor(last_pos, device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def scan_last_attention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[dict[str, Any]],
    last_layer: int,
    num_heads: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    pre_mass = np.zeros(num_heads, dtype=np.float64)
    self_mass = np.zeros(num_heads, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            items = prompts[start:start + batch_size]
            batch = tokenizer([x["prompt"] for x in items], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            _first, last_pos, pre_positions, _ans = batch_position_context(tokenizer, batch, items)
            out = model(**batch, output_attentions=True, use_cache=False)
            if out.attentions is None:
                raise RuntimeError("Model did not return attentions")
            attn = out.attentions[last_layer - 1].detach().float().cpu().numpy()
            for bi, ans in enumerate(last_pos):
                row = attn[bi, :, ans, :]
                pre = pre_positions[bi]
                if pre:
                    pre_mass += row[:, pre].sum(axis=1)
                self_mass += row[:, ans]
                count += 1
            del out, batch
            torch.cuda.empty_cache()
    return {
        "pre_answer_mass": (pre_mass / max(count, 1)).astype(np.float32),
        "self_mass": (self_mass / max(count, 1)).astype(np.float32),
    }


def position_audit(tokenizer: Any, prompts: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    batch = tokenizer([x["prompt"] for x in prompts], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    first_pos = first_nonpad_positions(batch["attention_mask"])
    last_pos = last_nonpad_positions(batch["attention_mask"])
    old_answer = (batch["attention_mask"].sum(dim=1) - 1).tolist()
    token_rows = batch["input_ids"].tolist()
    answer_in_pre = 0
    lengths = []
    for bi, item in enumerate(prompts):
        pre = corrected_positions_for_site(tokenizer, item, token_rows[bi], first_pos[bi], last_pos[bi], "pre_answer")
        answer_in_pre += int(last_pos[bi] in pre)
        lengths.append(len(pre))
    return {
        "n_prompts": len(prompts),
        "answer_in_pre_count": int(answer_in_pre),
        "old_answer_pos_mismatch_count": int(sum(int(old_answer[i] != last_pos[i]) for i in range(len(last_pos)))),
        "mean_pre_len": float(np.mean(lengths)) if lengths else 0.0,
    }


def summarize_condition(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
    summary = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    summary["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return summary


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        last_layer = len(layers)
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        num_heads = get_num_heads(model, get_attention_module(layers[last_layer - 1]))
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, heads={num_heads}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

        result: dict[str, Any] = {
            "phase": 130,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "num_heads": num_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "top_k_heads": args.top_k_heads,
            "answer_components": ANSWER_COMPONENTS,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log("Capturing last input pre-answer reference centers")
        reference_centers = capture_last_input_pre_answer_centers(
            model, tokenizer, device, layers, categories, last_layer,
            args.train_objects, args.batch_size, args.max_length,
        )
        component_centers = {}
        for component in ANSWER_COMPONENTS:
            log(f"Capturing answer centers {component}")
            component_centers[component] = capture_answer_component_centers(
                model, tokenizer, device, layers, categories, last_layer, component,
                args.train_objects, args.batch_size, args.max_length,
            )

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            monitor_basis, monitor_sv = svd_basis(build_category_contrast_matrix(component_centers["last_block_output_answer"], categories, cat), args.rank)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
            )
            ref_basis, ref_sv = svd_basis(build_category_contrast_matrix(reference_centers, categories, cat), args.rank)
            ref_patched = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                patch_component=REFERENCE_COMPONENT, patch_basis=ref_basis, scale=args.scale,
            )
            attn_scan = scan_last_attention(
                model, tokenizer, device, layers, prompts, last_layer, num_heads,
                args.batch_size, args.max_length,
            )
            selected_heads = [
                {
                    "head_id": int(h),
                    "pre_answer_mass": float(attn_scan["pre_answer_mass"][h]),
                    "self_mass": float(attn_scan["self_mass"][h]),
                }
                for h in np.argsort(-attn_scan["pre_answer_mass"])[:args.top_k_heads]
            ]
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": position_audit(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "monitor_singular_values": [float(x) for x in monitor_sv],
                "reference_pre_answer_singular_values": [float(x) for x in ref_sv],
                "reference_condition": {
                    "component": REFERENCE_COMPONENT,
                    **summarize_condition(ref_patched, baseline, target_idx, categories),
                },
                "attention_scan": {
                    "pre_answer_mass": attn_scan["pre_answer_mass"].tolist(),
                    "self_mass": attn_scan["self_mass"].tolist(),
                },
                "selected_heads": selected_heads,
                "answer_component_conditions": [],
                "head_ablation_conditions": [],
            }
            for component in ANSWER_COMPONENTS:
                basis, sv = svd_basis(build_category_contrast_matrix(component_centers[component], categories, cat), args.rank)
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                    patch_component=component, patch_basis=basis, scale=args.scale,
                )
                cat_out["answer_component_conditions"].append({
                    "component": component,
                    "singular_values": [float(x) for x in sv],
                    **summarize_condition(patched, baseline, target_idx, categories),
                })
            for head in selected_heads:
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                    ablate_head=head["head_id"], num_heads=num_heads,
                )
                cat_out["head_ablation_conditions"].append({
                    **head,
                    **summarize_condition(patched, baseline, target_idx, categories),
                })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    name = row.get("component", f"H{row.get('head_id')}")
    return f"{name} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 130 True-last Attention Read Gateway: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; heads: {result['num_heads']}")
    lines.append("")
    lines.append("| category | audit | reference pre-answer | best answer component | top head by mass | best head ablation |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        audit = item["position_audit"]
        audit_text = f"answer_in_pre={audit['answer_in_pre_count']}, old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        comp_best = min(item["answer_component_conditions"], key=lambda x: x["target_delta"]) if item["answer_component_conditions"] else None
        head_top = item["selected_heads"][0] if item["selected_heads"] else None
        head_best = min(item["head_ablation_conditions"], key=lambda x: x["target_delta"]) if item["head_ablation_conditions"] else None
        top_text = "NA" if head_top is None else f"H{head_top['head_id']} pre{head_top['pre_answer_mass']:.3f} self{head_top['self_mass']:.3f}"
        lines.append(
            f"| {cat} | {audit_text} | {_fmt(item['reference_condition'])} | "
            f"{_fmt(comp_best)} | {top_text} | {_fmt(head_best)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--top-k-heads", type=int, default=8)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase130_{args.model}_true_last_attention_read_gateway.json"
    md_path = out_dir / f"phase130_{args.model}_true_last_attention_read_gateway.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
