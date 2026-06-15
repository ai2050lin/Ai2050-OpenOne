#!/usr/bin/env python3
"""
Phase 136: long-template head re-ranking and path closure.

Re-rank true-last attention heads under the Phase135 long-template setting by
removing each head's all-pre-answer value contribution, then compare cumulative
long-template top-k heads against the previous short-template causal head set.
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
from phase107_causal_boundary_removal_cuda import BOUNDARY_LAYER  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads  # noqa: E402
from phase114_answer_site_causal_subspace_cuda import build_category_contrast_matrix  # noqa: E402
from phase116_subspace_basis_component_audit_cuda import svd_basis  # noqa: E402
from phase130_true_last_attention_read_gateway_cuda import REFERENCE_COMPONENT, summarize_condition  # noqa: E402
from phase132_source_value_contribution_cuda import get_num_kv_heads  # noqa: E402
from phase135_long_template_source_field_cuda import (  # noqa: E402
    LONG_TEMPLATES,
    build_long_prompts,
    capture_long_centers,
    position_audit_long,
    run_baseline_or_reference,
    run_source_condition,
    source_audit,
)


OUT_ROOT = Path("results/gpt5_phase136_long_template_head_reranking")
TEST_CATEGORIES = ["number", "container", "plant", "time", "clothing", "furniture"]
TOP_KS = [1, 2, 4, 8]
SOURCE_GROUP = "all_pre_answer"
SHORT_TEMPLATE_CORE_HEADS = {
    "qwen3": [11, 10, 28, 3, 31, 2, 5, 20],
    "glm4": [1, 28, 0, 18, 11, 27, 23, 4],
    "deepseek7b": [13, 12, 11, 8, 25, 10, 26, 24],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _valid_heads(heads: list[int], num_heads: int) -> list[int]:
    seen = set()
    out = []
    for head in heads:
        if 0 <= head < num_heads and head not in seen:
            out.append(head)
            seen.add(head)
    return out


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
        attn = get_attention_module(layers[last_layer - 1])
        num_heads = get_num_heads(model, attn)
        num_kv_heads = get_num_kv_heads(model, attn, num_heads)
        top_ks = [k for k in TOP_KS if k <= num_heads]
        short_core = _valid_heads(SHORT_TEMPLATE_CORE_HEADS.get(args.model, []), num_heads)
        alloc, reserved = vram_gb()
        log(
            f"{args.model}: peak=L{peak_layer}, true_last=L{last_layer}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, train/test={args.train_objects}/{args.test_objects}, "
            f"categories={test_categories}, vram={alloc:.2f}/{reserved:.2f}GB"
        )

        log("Capturing long-template monitor/reference centers")
        answer_centers = capture_long_centers(
            model, tokenizer, device, layers, categories, last_layer, "last_block_output_answer",
            args.train_objects, args.batch_size, args.max_length,
        )
        reference_centers = capture_long_centers(
            model, tokenizer, device, layers, categories, last_layer, "last_input_pre_answer",
            args.train_objects, args.batch_size, args.max_length,
        )

        result: dict[str, Any] = {
            "phase": 136,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_layer": peak_layer,
            "true_last_layer": last_layer,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "train_objects_per_category": args.train_objects,
            "test_objects_per_category": args.test_objects,
            "test_categories": test_categories,
            "rank": args.rank,
            "reference_scale": args.reference_scale,
            "contribution_scale": args.contribution_scale,
            "source_group": SOURCE_GROUP,
            "top_ks": top_ks,
            "short_template_core_heads": short_core,
            "templates": [x["name"] for x in LONG_TEMPLATES],
            "readout_token_labels": token_labels,
            "category_results": {},
        }

        for ci, cat in enumerate(test_categories, 1):
            log(f"Testing {args.model} {ci}/{len(test_categories)} {cat}")
            target_idx = categories.index(cat)
            prompts = build_long_prompts(cat, args.train_objects, args.test_objects)
            monitor_basis, monitor_sv = svd_basis(build_category_contrast_matrix(answer_centers, categories, cat), args.rank)
            ref_basis, ref_sv = svd_basis(build_category_contrast_matrix(reference_centers, categories, cat), args.rank)
            baseline = run_baseline_or_reference(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
            )
            ref_patched = run_baseline_or_reference(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                patch_basis=ref_basis, scale=args.reference_scale,
            )

            head_rows = []
            for head_id in range(num_heads):
                patched = run_source_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                    num_heads, num_kv_heads, SOURCE_GROUP, [head_id], args.contribution_scale,
                )
                head_rows.append({
                    "head_id": int(head_id),
                    **summarize_condition(patched, baseline, target_idx, categories),
                })
            ranked = sorted(head_rows, key=lambda x: x["target_delta"])

            aggregates = []
            for k in top_ks:
                head_ids = [int(x["head_id"]) for x in ranked[:k]]
                patched = run_source_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                    num_heads, num_kv_heads, SOURCE_GROUP, head_ids, args.contribution_scale,
                )
                aggregates.append({
                    "mode": f"long_top_{k}",
                    "head_ids": head_ids,
                    **summarize_condition(patched, baseline, target_idx, categories),
                })
            if short_core:
                patched = run_source_condition(
                    model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                    args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                    num_heads, num_kv_heads, SOURCE_GROUP, short_core, args.contribution_scale,
                )
                aggregates.append({
                    "mode": "short_template_core",
                    "head_ids": short_core,
                    **summarize_condition(patched, baseline, target_idx, categories),
                })
            patched = run_source_condition(
                model, tokenizer, device, layers, prompts, cat_local_ids, categories,
                args.batch_size, args.max_length, last_layer, monitor_basis, last_layer,
                num_heads, num_kv_heads, SOURCE_GROUP, list(range(num_heads)), args.contribution_scale,
            )
            aggregates.append({
                "mode": "all_heads",
                "head_ids": list(range(num_heads)),
                **summarize_condition(patched, baseline, target_idx, categories),
            })

            result["category_results"][cat] = {
                "n_prompts": len(prompts),
                "position_audit": position_audit_long(tokenizer, prompts, args.max_length),
                "source_audit": source_audit(tokenizer, prompts, args.max_length),
                "baseline_target_mean": float(baseline["scores"][:, target_idx].mean()),
                "baseline_answer_proj_mean": float(baseline["answer_proj"].mean()),
                "monitor_singular_values": [float(x) for x in monitor_sv],
                "reference_singular_values": [float(x) for x in ref_sv],
                "reference_condition": {
                    "component": REFERENCE_COMPONENT,
                    **summarize_condition(ref_patched, baseline, target_idx, categories),
                },
                "head_ranking": ranked,
                "aggregate_conditions": aggregates,
            }
        return result
    finally:
        release_loaded(loaded)


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    label = row.get("mode", f"H{row.get('head_id')}")
    heads = row.get("head_ids")
    suffix = f" {heads}" if heads is not None and row.get("mode") in {"long_top_4", "long_top_8", "short_template_core"} else ""
    return f"{label}{suffix} T{row['target_delta']:+.2f} R{row['max_other_delta']:+.2f} A{row['answer_proj_delta']:+.2f}"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [f"# Phase 136 Long-template Head Re-ranking: {result['model']}", ""]
    lines.append(f"Generated: {result['timestamp']}")
    lines.append(
        f"Peak layer: L{result['peak_layer']}; true last layer: L{result['true_last_layer']}; "
        f"heads: {result['num_heads']}; kv_heads: {result['num_kv_heads']}; "
        f"short core: {result['short_template_core_heads']}"
    )
    lines.append("")
    lines.append("| category | audit | reference | best head | top1 | top2 | top4 | top8 | short core | all heads |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cat, item in result["category_results"].items():
        audit = item["position_audit"]
        audit_text = f"old_mismatch={audit['old_answer_pos_mismatch_count']}, mean_pre={audit['mean_pre_len']:.1f}"
        ref = item["reference_condition"]
        ref_text = f"{ref['component']} T{ref['target_delta']:+.2f} R{ref['max_other_delta']:+.2f} A{ref['answer_proj_delta']:+.2f}"
        best_head = item["head_ranking"][0] if item["head_ranking"] else None
        by_mode = {x["mode"]: x for x in item["aggregate_conditions"]}
        lines.append(
            f"| {cat} | {audit_text} | {ref_text} | {_fmt(best_head)} | "
            f"{_fmt(by_mode.get('long_top_1'))} | {_fmt(by_mode.get('long_top_2'))} | "
            f"{_fmt(by_mode.get('long_top_4'))} | {_fmt(by_mode.get('long_top_8'))} | "
            f"{_fmt(by_mode.get('short_template_core'))} | {_fmt(by_mode.get('all_heads'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--reference-scale", type=float, default=1.5)
    parser.add_argument("--contribution-scale", type=float, default=1.0)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    json_path = out_dir / f"phase136_{args.model}_long_template_head_reranking.json"
    md_path = out_dir / f"phase136_{args.model}_long_template_head_reranking.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    log(f"Wrote {json_path}")
    log(f"Wrote {md_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
