#!/usr/bin/env python3
"""
Phase 131: cross-site basis transfer.

Apply the true-last input pre-answer category basis directly to answer-site
components. This separates local answer-site category axes from same-basis
transfer of the pre-answer residual field.
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
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER, summarize_delta  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import build_prompts, svd_basis  # noqa: E402
from phase130_true_last_attention_read_gateway_cuda import (  # noqa: E402
    ANSWER_COMPONENTS,
    REFERENCE_COMPONENT,
    capture_last_input_pre_answer_centers,
    position_audit,
    run_condition,
)


OUT_ROOT = Path("results/gpt5_phase131_cross_site_basis_transfer")
TEST_CATEGORIES = ["number", "container", "plant"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        alloc, reserved = vram_gb()
        log(f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, train/test={args.train_objects}/{args.test_objects}, vram={alloc:.2f}/{reserved:.2f}GB")

        log("Capturing true-last input pre-answer centers")
        reference_centers = capture_last_input_pre_answer_centers(
            model, tokenizer, device, layers, categories, last_layer,
            args.train_objects, args.batch_size, args.max_length,
        )

        result: dict[str, Any] = {
            "phase": 131,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "rank": args.rank,
            "scale": args.scale,
            "answer_components": ANSWER_COMPONENTS,
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_prompts(cat, args.train_objects, args.test_objects)
            pre_basis, pre_sv = svd_basis(build_category_contrast_matrix(reference_centers, categories, cat), args.rank)
            baseline = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, pre_basis, last_layer,
            )
            ref_patched = run_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, pre_basis, last_layer,
                patch_component=REFERENCE_COMPONENT, patch_basis=pre_basis, scale=args.scale,
            )
            cat_out = {
                "n_prompts": len(prompts),
                "position_audit": position_audit(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "pre_answer_singular_values": [float(x) for x in pre_sv],
                "reference_condition": {
                    "component": REFERENCE_COMPONENT,
                    **summarize_condition(ref_patched["scores"] - baseline["scores"], ref_patched["answer_proj"] - baseline["answer_proj"], target_idx, categories),
                },
                "cross_site_conditions": [],
            }
            for component in ANSWER_COMPONENTS:
                patched = run_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, pre_basis, last_layer,
                    patch_component=component, patch_basis=pre_basis, scale=args.scale,
                )
                cat_out["cross_site_conditions"].append({
                    "component": component,
                    **summarize_condition(patched["scores"] - baseline["scores"], patched["answer_proj"] - baseline["answer_proj"], target_idx, categories),
                })
            result["category_results"][cat] = cat_out
        return result
    finally:
        release_loaded(loaded)


def summarize_condition(score_delta: np.ndarray, proj_delta: np.ndarray, target_idx: int, categories: list[str]) -> dict[str, Any]:
    out = summarize_delta(score_delta, target_idx, categories)
    out["answer_proj_delta"] = float(proj_delta.mean())
    return out


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{row['component']} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 131 Cross-site Basis Transfer: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}")
    lines.append("")
    lines.append("| category | audit | reference | best same-basis answer component | attention answer | block output answer | final norm answer |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        audit = item["position_audit"]
        audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        by_comp = {x["component"]: x for x in item["cross_site_conditions"]}
        best = min(item["cross_site_conditions"], key=lambda x: x["target_delta"]) if item["cross_site_conditions"] else None
        lines.append(
            f"| {cat} | {audit_text} | {_fmt(item['reference_condition'])} | {_fmt(best)} | "
            f"{_fmt(by_comp.get('last_attention_output_answer'))} | "
            f"{_fmt(by_comp.get('last_block_output_answer'))} | "
            f"{_fmt(by_comp.get('final_norm_output_answer'))} |"
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
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase131_{args.model}_cross_site_basis_transfer.json"
    md_path = out_dir / f"phase131_{args.model}_cross_site_basis_transfer.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
