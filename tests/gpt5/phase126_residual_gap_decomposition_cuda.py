#!/usr/bin/env python3
"""
Phase 126: residual gap decomposition.

Decompose the pre-answer residual gap into layer input, attention output, MLP
output, and layer output category subspace interventions.
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
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase120_post_object_token_localization_cuda import item_positions, select_local_varimax_axis  # noqa: E402
from phase122_pre_answer_to_answer_projection_closure_cuda import projection_values  # noqa: E402
from phase123_attention_mlp_writer_localization_cuda import get_mlp_module  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module  # noqa: E402


OUT_ROOT = Path("results/gpt5_phase126_residual_gap_decomposition")
TEST_CATEGORIES = ["number", "container", "plant"]
COMPONENTS = ["layer_input", "attention_output", "mlp_output", "layer_output"]
PRE_SITE = "pre_answer"
ANSWER_SITE = "answer_last"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def positions_for_site(tokenizer: Any, item: dict[str, Any], answer_pos: int, site: str) -> list[int]:
    pos = item_positions(tokenizer, item, answer_pos)
    if site == PRE_SITE:
        return pos["post_object_excluding_answer"]
    if site == ANSWER_SITE:
        return pos["answer_last"]
    raise ValueError(site)


def get_component_module(layers: list[Any], layer_id: int, component: str) -> Any:
    layer = layers[layer_id - 1]
    if component in {"layer_input", "layer_output"}:
        return layer
    if component == "attention_output":
        return get_attention_module(layer)
    if component == "mlp_output":
        return get_mlp_module(layer)
    raise ValueError(component)


def tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output_tensor(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def make_capture_hook(component: str, store: dict[str, torch.Tensor]):
    if component == "layer_input":
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            store["value"] = inputs[0].detach()
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        store["value"] = tensor_from_output(output).detach()
    return hook, False


def make_subspace_patch_hook(component: str, basis: torch.Tensor, batch_positions: list[list[int]], scale: float):
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

    if component == "layer_input":
        def pre_hook(_module: Any, inputs: tuple[Any, ...]):
            return (patch_tensor(inputs[0]),) + inputs[1:]
        return pre_hook, True

    def hook(_module: Any, _inputs: Any, output: Any):
        return replace_output_tensor(output, patch_tensor(tensor_from_output(output)))
    return hook, False


def capture_component_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    categories: list[str],
    layer_id: int,
    component: str,
    site: str,
    train_objects: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    centers = np.zeros((len(TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(TEMPLATES), len(categories)), dtype=np.int64)
    items: list[dict[str, Any]] = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_objects]:
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})

    module = get_component_module(layers, layer_id, component)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [x["prompt"] for x in batch_items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            store: dict[str, torch.Tensor] = {}
            hook_fn, is_pre = make_capture_hook(component, store)
            handle = module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn)
            _ = model(**batch, use_cache=False)
            handle.remove()
            value = store["value"]
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            for bi, item in enumerate(batch_items):
                positions = positions_for_site(tokenizer, item, answer_pos[bi], site)
                pos = torch.tensor(positions, device=value.device, dtype=torch.long)
                vec = value[bi, pos, :].float().mean(dim=0).detach().cpu().numpy()
                centers[item["ti"], item["ci"]] += vec
                counts[item["ti"], item["ci"]] += 1
            del batch
            torch.cuda.empty_cache()
    return (centers / counts[:, :, None]).astype(np.float32)


def capture_answer_centers(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    categories: list[str],
    layer_id: int,
    train_objects: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    d_model = int(model.get_input_embeddings().weight.shape[1])
    centers = np.zeros((len(TEMPLATES), len(categories), d_model), dtype=np.float64)
    counts = np.zeros((len(TEMPLATES), len(categories)), dtype=np.int64)
    items: list[dict[str, Any]] = []
    for ti, tpl in enumerate(TEMPLATES):
        for ci, cat in enumerate(categories):
            for obj in CATEGORY_OBJECTS[cat][:train_objects]:
                items.append({"ti": ti, "ci": ci, "cat": cat, "obj": obj, "prompt": tpl["text"].format(obj=obj)})
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            texts = [x["prompt"] for x in batch_items]
            batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[layer_id]
            answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
            for bi, item in enumerate(batch_items):
                vec = hs[bi, answer_pos[bi], :].float().detach().cpu().numpy()
                centers[item["ti"], item["ci"]] += vec
                counts[item["ti"], item["ci"]] += 1
            del out, batch
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
    patch_layer: int | None = None,
    component_patches: list[tuple[str, np.ndarray]] | None = None,
    scale: float = 1.5,
) -> dict[str, np.ndarray]:
    scores = []
    answer_proj = []
    component_patches = component_patches or []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
        handles = []
        if patch_layer is not None:
            for component, basis_np in component_patches:
                positions = [positions_for_site(tokenizer, item, answer_pos[bi], PRE_SITE) for bi, item in enumerate(items)]
                basis = torch.tensor(basis_np, device=device, dtype=torch.float32)
                hook_fn, is_pre = make_subspace_patch_hook(component, basis, positions, scale)
                module = get_component_module(layers, patch_layer, component)
                handles.append(module.register_forward_pre_hook(hook_fn) if is_pre else module.register_forward_hook(hook_fn))
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        pos_gpu = torch.tensor(answer_pos, device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        hs = out.hidden_states[monitor_layer]
        ans = hs[torch.arange(hs.shape[0], device=hs.device), pos_gpu.to(hs.device), :].float()
        answer_proj.append(projection_values(ans, monitor_basis))
        del out, batch
        torch.cuda.empty_cache()
    return {"scores": np.concatenate(scores, axis=0), "answer_proj": np.concatenate(answer_proj, axis=0)}


def summarize_condition(patched: dict[str, np.ndarray], baseline: dict[str, np.ndarray], target_idx: int, categories: list[str]) -> dict[str, Any]:
    out = summarize_delta(patched["scores"] - baseline["scores"], target_idx, categories)
    out["answer_proj_delta"] = float((patched["answer_proj"] - baseline["answer_proj"]).mean())
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = [x.strip() for x in args.categories.split(",") if x.strip()] or TEST_CATEGORIES
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        patch_layers = list(range(max(1, peak_layer - args.layer_back), peak_layer + 1))
        components = [x.strip() for x in args.components.split(",") if x.strip()]
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, layers={patch_layers}, components={components}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

        result: dict[str, Any] = {
            "phase": 126,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "monitor_layer": peak_layer,
            "patch_layers": patch_layers,
            "components": components,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        log("Building answer monitor centers")
        answer_centers = capture_answer_centers(model, tokenizer, device, categories, peak_layer, args.train_objects, args.batch_size, args.max_length)
        center_cache: dict[tuple[int, str], np.ndarray] = {}
        for layer_id in patch_layers:
            for component in components:
                log(f"Capturing centers L{layer_id} {component}")
                center_cache[(layer_id, component)] = capture_component_centers(
                    model, tokenizer, device, layers, categories, layer_id, component, PRE_SITE,
                    args.train_objects, args.batch_size, args.max_length,
                )

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            answer_basis, answer_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            baseline_for_selection = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, answer_basis,
            )
            answer_choice = select_local_varimax_axis(
                model, tokenizer, device, layers, prompts, baseline_for_selection["scores"],
                peak_layer, ANSWER_SITE, cat_local_ids, categories, target_idx,
                args.batch_size, args.max_length, args.scale, answer_basis,
            )
            monitor_basis = answer_choice["axis"] if args.monitor_axis == "varimax" else answer_basis
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, peak_layer, monitor_basis,
            )
            basis_cache: dict[tuple[int, str], tuple[np.ndarray, list[float]]] = {}
            for layer_id in patch_layers:
                for component in components:
                    basis, sv = svd_basis(build_category_contrast_matrix(center_cache[(layer_id, component)], categories, cat), args.rank)
                    basis_cache[(layer_id, component)] = (basis, [float(x) for x in sv])

            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in answer_sv],
                "answer_varimax_selection": {
                    "basis_index": int(answer_choice["basis_index"]),
                    "selection_target_delta": float(answer_choice["selection_target_delta"]),
                },
                "conditions": [],
            }
            for layer_id in patch_layers:
                layer_component_summaries = {}
                for component in components:
                    basis, sv = basis_cache[(layer_id, component)]
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, peak_layer, monitor_basis,
                        patch_layer=layer_id,
                        component_patches=[(component, basis)],
                        scale=args.scale,
                    )
                    summary = summarize_condition(patched, baseline, target_idx, categories)
                    row = {
                        "patch_layer": int(layer_id),
                        "component": component,
                        "condition": f"{component}_only",
                        "singular_values": sv,
                        **summary,
                    }
                    layer_component_summaries[component] = row
                    cat_out["conditions"].append(row)
                if "attention_output" in components and "mlp_output" in components:
                    combo_specs = [
                        ("attention_output", basis_cache[(layer_id, "attention_output")][0]),
                        ("mlp_output", basis_cache[(layer_id, "mlp_output")][0]),
                    ]
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, peak_layer, monitor_basis,
                        patch_layer=layer_id,
                        component_patches=combo_specs,
                        scale=args.scale,
                    )
                    summary = summarize_condition(patched, baseline, target_idx, categories)
                    residual_ref = layer_component_summaries.get("layer_output")
                    ratio = 0.0
                    if residual_ref and abs(residual_ref["target_delta"]) > 1e-6:
                        ratio = float(summary["target_delta"] / residual_ref["target_delta"])
                    cat_out["conditions"].append({
                        "patch_layer": int(layer_id),
                        "component": "attention_plus_mlp",
                        "condition": "attention_plus_mlp",
                        **summary,
                        "effect_ratio_vs_layer_output": ratio,
                    })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"L{row['patch_layer']} {row['component']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 126 Residual Gap Decomposition: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; patch layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | layer input | attention output | MLP output | layer output | attn+MLP |")
    lines.append("|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        conds = item["conditions"]

        def best(component: str) -> dict[str, Any] | None:
            xs = [x for x in conds if x["component"] == component]
            return min(xs, key=lambda x: x["target_delta"]) if xs else None

        lines.append(
            f"| {cat} | {_fmt(best('layer_input'))} | {_fmt(best('attention_output'))} | "
            f"{_fmt(best('mlp_output'))} | {_fmt(best('layer_output'))} | {_fmt(best('attention_plus_mlp'))} |"
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
    parser.add_argument("--layer-back", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--components", default="layer_input,attention_output,mlp_output,layer_output")
    parser.add_argument("--monitor-axis", choices=["varimax", "subspace"], default="varimax")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase126_{args.model}_residual_gap_decomposition.json"
    md_path = out_dir / f"phase126_{args.model}_residual_gap_decomposition.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
