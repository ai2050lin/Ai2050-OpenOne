#!/usr/bin/env python3
"""
Phase 696: Value Token Hidden-State Time-of-Entry Audit.

Phase 695 showed that target-value token visibility is causally important, even
though Phase 694 found low answer-last attention mass to target-value tokens.
This phase measures where the value signal appears as a hidden-state trajectory:

For paired short_only failures and terse_no_explain successes, capture every
layer_out and project selected token positions onto the case-specific
value-minus-prose readout direction:
  - target_value token positions
  - relation token positions
  - record_line positions
  - record_without_target_value positions
  - answer_last position

This is a diagnostic trajectory audit, not a causal patch.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase584_gate_repair import load_model_flash  # noqa: E402
from phase599_final_layer_washout_decomposition import extract_tensor  # noqa: E402
from phase683_prose_route_bias_source_decomposition import (  # noqa: E402
    expected_first_ids,
    expected_for,
    prompt_for,
    route_id_sets,
    select_base_cases,
    value_phrase,
)
from phase685_natural_value_readout_writer_localization import (  # noqa: E402
    SHORT_VARIANT,
    TERSE_VARIANT,
    projection,
    select_paired_cases,
    value_minus_prose_direction,
)
from phase687_l26_l27_value_support_state_decomposition import classify  # noqa: E402
from phase694_boundary_head_source_token_attention_audit import token_groups  # noqa: E402


OUT_ROOT = Path("results/glm5_phase696_value_token_hidden_state_time_of_entry")
GROUPS = [
    "target_value",
    "relation",
    "object_name",
    "record_line",
    "record_without_target_value",
    "record_value_object_relation",
    "instruction_line",
    "answer_last",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_positions(groups: dict[str, list[int]], n: int) -> dict[str, list[int]]:
    record = {i for i in groups.get("record_line", []) if 0 <= i < n}
    target = {i for i in groups.get("target_value", []) if 0 <= i < n}
    obj = {i for i in groups.get("object_name", []) if 0 <= i < n}
    relation = {i for i in groups.get("relation", []) if 0 <= i < n}
    record_value_object_relation = sorted((target | obj | relation) & record)
    out = {
        "target_value": sorted(target),
        "relation": sorted(relation),
        "object_name": sorted(obj),
        "record_line": sorted(record),
        "record_without_target_value": sorted(record - target),
        "record_value_object_relation": record_value_object_relation,
        "instruction_line": [i for i in groups.get("instruction_line", []) if 0 <= i < n],
        "answer_last": [n - 1],
    }
    return out


def capture_layer_group_projections(
    model,
    tokenizer,
    device,
    prompt: str,
    case: dict[str, Any],
    direction: torch.Tensor,
    routes,
    expected_ids,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    groups = group_positions(token_groups(tokenizer, prompt, case, ids), len(ids))
    captured: list[dict[str, Any]] = []
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            y = extract_tensor(output)[0].detach()
            for group, idxs in groups.items():
                valid = [i for i in idxs if 0 <= i < y.shape[0]]
                if not valid:
                    continue
                vec = y[valid].mean(dim=0)
                captured.append({
                    "layer": layer_idx,
                    "group": group,
                    "n_tokens": len(valid),
                    "projection": projection(vec, direction),
                    "norm": float(vec.float().norm().detach().cpu().item()),
                })
        return hook

    for li, layer in enumerate(get_layers(model)):
        handles.append(layer.register_forward_hook(make_hook(li)))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        diag = classify(out.logits[0, -1].detach(), routes, expected_ids)
    finally:
        for h in handles:
            h.remove()
    diag["seq_len"] = len(ids)
    return diag, captured


def summarize_model(model_name: str, paired_ids: list[str], rows: list[dict[str, Any]], finals: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["variant"], r["group"], r["layer"])].append(r)
    by_variant_group_layer = {}
    for (variant, group, layer), vals in grouped.items():
        by_variant_group_layer[f"{variant}|{group}|L{layer}"] = {
            "n": len(vals),
            "mean_projection": sum(v["projection"] for v in vals) / len(vals),
            "mean_norm": sum(v["norm"] for v in vals) / len(vals),
            "mean_n_tokens": sum(v["n_tokens"] for v in vals) / len(vals),
        }

    peaks = []
    for variant in ["short_only", "terse_no_explain"]:
        for group in GROUPS:
            vals = [
                {"variant": variant, "group": group, "layer": layer, **summary}
                for key, summary in by_variant_group_layer.items()
                for v2, g2, lstr in [key.split("|")]
                for layer in [int(lstr[1:])]
                if v2 == variant and g2 == group
            ]
            if not vals:
                continue
            peak = max(vals, key=lambda x: x["mean_projection"])
            first_positive = next((v for v in sorted(vals, key=lambda x: x["layer"]) if v["mean_projection"] > 0), None)
            final = max(vals, key=lambda x: x["layer"])
            peaks.append({
                "variant": variant,
                "group": group,
                "peak_layer": peak["layer"],
                "peak_projection": peak["mean_projection"],
                "first_positive_layer": None if first_positive is None else first_positive["layer"],
                "first_positive_projection": None if first_positive is None else first_positive["mean_projection"],
                "final_layer": final["layer"],
                "final_projection": final["mean_projection"],
            })

    variant_final = defaultdict(list)
    for f in finals:
        variant_final[f["variant"]].append(f)
    final_summary = {}
    for variant, vals in variant_final.items():
        final_summary[variant] = {
            "n": len(vals),
            "top1_rate": sum(1 for v in vals if v["expected_top1"]) / len(vals),
            "mean_expected_rank": sum(v["expected_rank"] for v in vals) / len(vals),
            "mean_pmv": sum(v["prose_minus_value"] for v in vals) / len(vals),
            "best_other_route_counts": {k: sum(1 for v in vals if v["best_other_route"] == k) for k in sorted({v["best_other_route"] for v in vals})},
        }

    return {
        "model": model_name,
        "n_paired_cases": len(paired_ids),
        "n_rows": len(rows),
        "final_summary": final_summary,
        "peaks": peaks,
        "by_variant_group_layer": by_variant_group_layer,
    }


def run_model(args) -> dict[str, Any]:
    paired_ids = select_paired_cases(args.model, args.limit)
    case_map = {c["case_id"]: c for c in select_base_cases()}
    model, tokenizer, device = load_model_flash(args.model)
    rows: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    try:
        dtype = next(model.parameters()).dtype
        for idx, case_id in enumerate(paired_ids, 1):
            case = case_map[case_id]
            expected_text = expected_for(case, SHORT_VARIANT)
            expected_ids = expected_first_ids(tokenizer, expected_text)
            routes = route_id_sets(tokenizer, case, expected_text)
            direction = value_minus_prose_direction(model, routes, expected_ids, device, dtype)
            for variant_name, variant in [("short_only", SHORT_VARIANT), ("terse_no_explain", TERSE_VARIANT)]:
                prompt = prompt_for(case, variant)
                diag, captured = capture_layer_group_projections(
                    model, tokenizer, device, prompt, case, direction, routes, expected_ids
                )
                finals.append({
                    "case_id": case_id,
                    "variant": variant_name,
                    **diag,
                })
                for rec in captured:
                    rows.append({
                        "case_id": case_id,
                        "family": case["family"],
                        "object_name": case.get("object_name"),
                        "relation": case.get("relation"),
                        "value": value_phrase(case),
                        "variant": variant_name,
                        "expected_top1": diag["expected_top1"],
                        "expected_rank": diag["expected_rank"],
                        "prose_minus_value": diag["prose_minus_value"],
                        **rec,
                    })
            if idx % args.log_every == 0 or idx == len(paired_ids):
                log(f"{args.model}: captured value-token trajectories {idx}/{len(paired_ids)} paired cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_model(args.model, paired_ids, rows, finals)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"phase696_{args.model}_trajectory_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / f"phase696_{args.model}_final_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in finals) + "\n",
        encoding="utf-8",
    )
    payload = {
        "phase": 696,
        "title": "Value Token Hidden-State Time-of-Entry Audit",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "n_paired_cases": len(paired_ids),
        "summary": summary,
    }
    (OUT_ROOT / f"phase696_{args.model}_trajectory_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def write_cross_summary() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = []
    for path in sorted(OUT_ROOT.glob("phase696_*_trajectory_summary.json")):
        models.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 696,
        "title": "Value Token Hidden-State Time-of-Entry Audit Cross-Model Summary",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase696_cross_model_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 696 Value Token Hidden-State Time-of-Entry Audit",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "| model | pairs | short_top1 | terse_top1 | short_rank | terse_rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        fs = item["summary"]["final_summary"]
        s = fs.get("short_only", {})
        t = fs.get("terse_no_explain", {})
        lines.append(
            f"| {item['model']} | {item['n_paired_cases']} | {s.get('top1_rate', 0.0):.3f} | "
            f"{t.get('top1_rate', 0.0):.3f} | {s.get('mean_expected_rank', 0.0):.2f} | {t.get('mean_expected_rank', 0.0):.2f} |"
        )
    lines.extend(["", "## Peak / First Positive Layers", ""])
    for item in models:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("| variant | group | first_pos_layer | first_pos_proj | peak_layer | peak_proj | final_layer | final_proj |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in item["summary"]["peaks"]:
            lines.append(
                f"| {row['variant']} | {row['group']} | "
                f"{'' if row['first_positive_layer'] is None else row['first_positive_layer']} | "
                f"{0.0 if row['first_positive_projection'] is None else row['first_positive_projection']:.3f} | "
                f"{row['peak_layer']} | {row['peak_projection']:.3f} | {row['final_layer']} | {row['final_projection']:.3f} |"
            )
        lines.append("")
    (OUT_ROOT / "phase696_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
