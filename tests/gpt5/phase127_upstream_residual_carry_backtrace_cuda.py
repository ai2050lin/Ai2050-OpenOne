#!/usr/bin/env python3
"""
Phase 127: upstream residual carry backtrace.

Scan wider layer ranges for pre-answer layer_input/layer_output causal subspaces
to locate onset, peaks, decay, and final re-emergence of the residual field.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase105_global_category_atlas_cuda import CATEGORY_OBJECTS, collect_readout_rows  # noqa: E402
from phase106_multitemplate_residual_cuda import TEMPLATES  # noqa: E402
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase120_post_object_token_localization_cuda import select_local_varimax_axis  # noqa: E402
from phase126_residual_gap_decomposition_cuda import (  # noqa: E402
    ANSWER_SITE,
    PRE_SITE,
    capture_answer_centers,
    capture_component_centers,
    run_condition,
    summarize_condition,
)


OUT_ROOT = Path("results/gpt5_phase127_upstream_residual_carry_backtrace")
TEST_CATEGORIES = ["number", "container", "plant"]
COMPONENTS = ["layer_input", "layer_output"]
DEFAULT_LAYER_FROM = {"qwen3": 20, "glm4": 8, "deepseek7b": 12}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def curve_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda x: x["patch_layer"])
    active = [r for r in ordered if r["target_delta"] <= threshold]
    best = min(ordered, key=lambda x: x["target_delta"]) if ordered else None
    first = active[0] if active else None
    last = active[-1] if active else None
    return {
        "threshold": float(threshold),
        "first_active_layer": None if first is None else int(first["patch_layer"]),
        "last_active_layer": None if last is None else int(last["patch_layer"]),
        "best_layer": None if best is None else int(best["patch_layer"]),
        "best_target_delta": None if best is None else float(best["target_delta"]),
        "active_layers": [int(r["patch_layer"]) for r in active],
    }


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
        layer_from = args.layer_from if args.layer_from is not None else DEFAULT_LAYER_FROM[args.model]
        layer_to = args.layer_to if args.layer_to is not None else peak_layer
        patch_layers = list(range(max(1, layer_from), min(len(layers), layer_to) + 1))
        cat_local_ids, _rows, token_labels = collect_readout_rows(model, tokenizer, categories)
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, scan=L{patch_layers[0]}-L{patch_layers[-1]}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

        result: dict[str, Any] = {
            "phase": 127,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "monitor_layer": peak_layer,
            "patch_layers": patch_layers,
            "components": COMPONENTS,
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
        answer_centers = capture_answer_centers(
            model, tokenizer, device, categories, peak_layer,
            args.train_objects, args.batch_size, args.max_length,
        )
        center_cache: dict[tuple[int, str], np.ndarray] = {}
        for layer_id in patch_layers:
            for component in COMPONENTS:
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
            conditions = []
            for layer_id in patch_layers:
                for component in COMPONENTS:
                    basis, sv = svd_basis(build_category_contrast_matrix(center_cache[(layer_id, component)], categories, cat), args.rank)
                    patched = run_condition(
                        model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                        args.batch_size, args.max_length, peak_layer, monitor_basis,
                        patch_layer=layer_id,
                        component_patches=[(component, basis)],
                        scale=args.scale,
                    )
                    conditions.append({
                        "patch_layer": int(layer_id),
                        "component": component,
                        "singular_values": [float(x) for x in sv],
                        **summarize_condition(patched, baseline, target_idx, categories),
                    })
            metrics = {
                component: curve_metrics([r for r in conditions if r["component"] == component], args.onset_threshold)
                for component in COMPONENTS
            }
            cat_out = {
                "n_prompts": len(prompts),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "answer_singular_values": [float(x) for x in answer_sv],
                "answer_varimax_selection": {
                    "basis_index": int(answer_choice["basis_index"]),
                    "selection_target_delta": float(answer_choice["selection_target_delta"]),
                },
                "curve_metrics": metrics,
                "conditions": conditions,
            }
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"L{row['patch_layer']} T{row['target_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 127 Upstream Residual Carry Backtrace: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Monitor layer: L{result['monitor_layer']}; scan layers: {result['patch_layers'][0]}-{result['patch_layers'][-1]}")
    lines.append("")
    lines.append("| category | input onset | input best | output onset | output best | final input | final output |")
    lines.append("|---|---|---|---|---|---|---|")
    final_layer = result["patch_layers"][-1]
    for cat, item in result["category_results"].items():
        conds = item["conditions"]
        inp = [r for r in conds if r["component"] == "layer_input"]
        out = [r for r in conds if r["component"] == "layer_output"]
        best_inp = min(inp, key=lambda x: x["target_delta"]) if inp else None
        best_out = min(out, key=lambda x: x["target_delta"]) if out else None
        final_inp = next((r for r in inp if r["patch_layer"] == final_layer), None)
        final_out = next((r for r in out if r["patch_layer"] == final_layer), None)
        mi = item["curve_metrics"]["layer_input"]
        mo = item["curve_metrics"]["layer_output"]
        lines.append(
            f"| {cat} | L{mi['first_active_layer']} | {_fmt(best_inp)} | "
            f"L{mo['first_active_layer']} | {_fmt(best_out)} | {_fmt(final_inp)} | {_fmt(final_out)} |"
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
    parser.add_argument("--layer-from", type=int, default=None)
    parser.add_argument("--layer-to", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--monitor-axis", choices=["varimax", "subspace"], default="varimax")
    parser.add_argument("--onset-threshold", type=float, default=-0.5)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase127_{args.model}_upstream_residual_carry_backtrace.json"
    md_path = out_dir / f"phase127_{args.model}_upstream_residual_carry_backtrace.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
