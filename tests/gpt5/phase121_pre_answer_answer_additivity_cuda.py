#!/usr/bin/env python3
"""
Phase 121: pre-answer interface and answer-site additivity.

Test whether pre-answer excluding-answer local field adds to, is redundant
with, or is absorbed by the answer_last readout field.
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
from phase118_causal_axis_transport_closure_cuda import random_in_subspace  # noqa: E402
from phase120_post_object_token_localization_cuda import (  # noqa: E402
    capture_local_centers,
    item_positions,
    select_local_varimax_axis,
)


OUT_ROOT = Path("results/gpt5_phase121_pre_answer_answer_additivity")
TEST_CATEGORIES = ["number", "container", "plant"]
SITES = ["post_object_excluding_answer", "answer_last"]
AXIS_TYPES = ["local_varimax_best", "local_svd_subspace", "random_in_local_subspace"]
PATCH_MODES = ["pre_only", "answer_only", "pre_plus_answer"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_multi_site_hook(specs: list[tuple[torch.Tensor, list[list[int]]]], scale: float):
    prepared = [(basis / (basis.norm(dim=1, keepdim=True) + 1e-8), positions) for basis, positions in specs]

    def hook(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            out = output[0].clone()
            rest = output[1:]
        else:
            out = output.clone()
            rest = None
        for basis, batch_positions in prepared:
            b = basis.to(out.device).float()
            for bi, positions in enumerate(batch_positions):
                pos = torch.tensor(positions, device=out.device, dtype=torch.long)
                vecs = out[bi, pos, :].float()
                proj = (vecs @ b.T) @ b
                out[bi, pos, :] = out[bi, pos, :] - scale * proj.to(out.dtype)
        if rest is not None:
            return (out,) + rest
        return out

    return hook


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
    patch_layer: int | None = None,
    patches: list[tuple[str, np.ndarray]] | None = None,
    scale: float = 1.5,
) -> np.ndarray:
    scores = []
    module_index = None if patch_layer is None else patch_layer - 1
    patches = patches or []
    for start in range(0, len(prompts), batch_size):
        items = prompts[start:start + batch_size]
        texts = [x["prompt"] for x in items]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_pos = (batch["attention_mask"].sum(dim=1) - 1).detach().cpu().tolist()
        specs = []
        for site, basis in patches:
            batch_positions = [
                item_positions(tokenizer, item, answer_pos[bi])[site]
                for bi, item in enumerate(items)
            ]
            specs.append((torch.tensor(basis, device=device, dtype=torch.float32), batch_positions))
        handle = None
        if specs and module_index is not None:
            handle = layers[module_index].register_forward_hook(make_multi_site_hook(specs, scale))
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        if handle is not None:
            handle.remove()
        pos_gpu = torch.tensor(answer_pos, device=out.logits.device, dtype=torch.long)
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos_gpu]
        scores.append(score_logits(logits, cat_local_ids, categories))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0)


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_probe_model(args.model)
    try:
        model = loaded.model
        tokenizer = loaded.tokenizer
        device = loaded.input_device
        layers = get_layers(model)
        categories = list(CATEGORY_OBJECTS.keys())
        test_categories = args.categories.split(",") if args.categories else TEST_CATEGORIES
        peak_layer = args.peak_layer if args.peak_layer is not None else BOUNDARY_LAYER[args.model]
        patch_layers = list(range(max(1, peak_layer - args.layer_back), peak_layer + 1))
        axis_types = [x.strip() for x in args.axis_types.split(",") if x.strip()]
        cat_local_ids, _readout_rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, layers={patch_layers}, axis_types={axis_types}, "
            f"rank={args.rank}, train/test={args.train_objects}/{args.test_objects}, "
            f"vram={alloc:.2f}/{reserved:.2f}GB"
        )

        result: dict[str, Any] = {
            "phase": 121,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "patch_layers": patch_layers,
            "sites": SITES,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "templates": [t["name"] for t in TEMPLATES],
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "axis_types": axis_types,
            "patch_modes": PATCH_MODES,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        basis_cache: dict[tuple[int, str, str], tuple[np.ndarray, list[float]]] = {}
        for layer_id in patch_layers:
            for site in SITES:
                log(f"Building local centers L{layer_id} {site}")
                centers = capture_local_centers(
                    model, tokenizer, device, categories, layer_id, site,
                    args.train_objects, args.batch_size, args.max_length
                )
                for cat in test_categories:
                    contrast = build_category_contrast_matrix(centers, categories, cat)
                    basis, singular_values = svd_basis(contrast, args.rank)
                    basis_cache[(layer_id, site, cat)] = (basis, [float(x) for x in singular_values])

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length
            )
            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline[:, target_idx].mean()),
                "conditions": [],
            }
            for layer_id in patch_layers:
                log(f"  {cat}: L{layer_id}")
                pre_basis, pre_sv = basis_cache[(layer_id, "post_object_excluding_answer", cat)]
                ans_basis, ans_sv = basis_cache[(layer_id, "answer_last", cat)]
                pre_choice = select_local_varimax_axis(
                    model, tokenizer, device, layers, prompts, baseline,
                    layer_id, "post_object_excluding_answer", cat_local_ids, categories,
                    target_idx, args.batch_size, args.max_length, args.scale, pre_basis
                )
                ans_choice = select_local_varimax_axis(
                    model, tokenizer, device, layers, prompts, baseline,
                    layer_id, "answer_last", cat_local_ids, categories,
                    target_idx, args.batch_size, args.max_length, args.scale, ans_basis
                )
                seed = 21000 + categories.index(cat) * 997 + layer_id * 31
                axes = {
                    "local_varimax_best": (
                        pre_choice["axis"],
                        ans_choice["axis"],
                        pre_choice,
                        ans_choice,
                    ),
                    "local_svd_subspace": (
                        pre_basis,
                        ans_basis,
                        {"basis_index": -1, "selection_target_delta": 0.0, "selection_max_other_delta": 0.0},
                        {"basis_index": -1, "selection_target_delta": 0.0, "selection_max_other_delta": 0.0},
                    ),
                    "random_in_local_subspace": (
                        random_in_subspace(pre_basis, seed),
                        random_in_subspace(ans_basis, seed + 1),
                        {"basis_index": -1, "selection_target_delta": 0.0, "selection_max_other_delta": 0.0},
                        {"basis_index": -1, "selection_target_delta": 0.0, "selection_max_other_delta": 0.0},
                    ),
                }
                for axis_type in axis_types:
                    pre_axis, ans_axis, pre_meta, ans_meta = axes[axis_type]
                    mode_patches = {
                        "pre_only": [("post_object_excluding_answer", pre_axis)],
                        "answer_only": [("answer_last", ans_axis)],
                        "pre_plus_answer": [
                            ("post_object_excluding_answer", pre_axis),
                            ("answer_last", ans_axis),
                        ],
                    }
                    for mode, patches in mode_patches.items():
                        patched = run_condition(
                            model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                            args.batch_size, args.max_length, layer_id, patches, args.scale
                        )
                        summary = summarize_delta(patched - baseline, target_idx, categories)
                        cat_out["conditions"].append({
                            "layer": layer_id,
                            "axis_type": axis_type,
                            "patch_mode": mode,
                            "pre_axis_rank": int(pre_axis.shape[0]),
                            "answer_axis_rank": int(ans_axis.shape[0]),
                            "pre_varimax_basis_index": int(pre_meta["basis_index"]),
                            "answer_varimax_basis_index": int(ans_meta["basis_index"]),
                            "pre_varimax_selection_target_delta": float(pre_meta["selection_target_delta"]),
                            "answer_varimax_selection_target_delta": float(ans_meta["selection_target_delta"]),
                            "pre_singular_values": pre_sv,
                            "answer_singular_values": ans_sv,
                            **summary,
                        })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 121 Pre-answer Answer Additivity: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Layers: {result['patch_layers']}")
    lines.append("")
    lines.append("| category | axis | best pre | best answer | best combined |")
    lines.append("|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        for axis in result["axis_types"]:
            conds = [c for c in item["conditions"] if c["axis_type"] == axis]

            def fmt(mode: str) -> str:
                rows = [c for c in conds if c["patch_mode"] == mode]
                if not rows:
                    return "NA"
                r = min(rows, key=lambda x: x["target_delta"])
                return f"L{r['layer']} T{r['target_delta']:+.2f} R{r['max_other_delta']:+.2f}"

            lines.append(f"| {cat} | {axis} | {fmt('pre_only')} | {fmt('answer_only')} | {fmt('pre_plus_answer')} |")
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
    parser.add_argument("--axis-types", default="local_varimax_best,local_svd_subspace")
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase121_{args.model}_pre_answer_answer_additivity.json"
    md_path = out_dir / f"phase121_{args.model}_pre_answer_answer_additivity.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
