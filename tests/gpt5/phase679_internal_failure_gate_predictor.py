#!/usr/bin/env python3
"""
Phase 679: Internal Failure-Gate Predictor for Selective Readout Repair.

This is a post-processing audit. It uses Phase 674/675 internal readout and
trajectory rows, plus Phase 677 intervention rows, to test simple gate rules.
No learned classifier and no model forward pass are used.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT674 = Path("results/glm5_phase674_synthetic_value_readout_competitor_source_localization")
ROOT675 = Path("results/glm5_phase675_final_readout_direction_field_component_attribution")
ROOT677 = Path("results/glm5_phase677_readout_intervention_strength_scan")
OUT_ROOT = Path("results/glm5_phase679_internal_failure_gate_predictor")

REPAIR_CONDITIONS = [
    "final_cancel_gap_a1p25",
    "final_cancel_gap_a1p5",
    "final_cancel_gap_a2p0",
    "final_remove_comp_a2p0",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def component(row: dict, name: str) -> dict | None:
    for item in row["component_deltas"]:
        if item["component"].endswith(name) or item["component"] == name:
            return item
    return None


def last_component(row: dict, suffix: str) -> dict | None:
    items = [x for x in row["component_deltas"] if x["component"].endswith(suffix)]
    if not items:
        return None
    return max(items, key=lambda x: x["layer"])


def prev_component(row: dict, suffix: str) -> dict | None:
    items = sorted([x for x in row["component_deltas"] if x["component"].endswith(suffix)], key=lambda x: x["layer"])
    if len(items) < 2:
        return None
    return items[-2]


def build_case_table(model: str) -> dict[str, dict]:
    rows674 = {r["case_id"]: r for r in read_jsonl(ROOT674 / f"phase674_{model}_synthetic_value_readout_source_rows.jsonl")}
    rows675 = {r["case_id"]: r for r in read_jsonl(ROOT675 / f"phase675_{model}_component_attribution_rows.jsonl")}
    rows677 = read_jsonl(ROOT677 / f"phase677_{model}_strength_scan_rows.jsonl")
    by_case_cond: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows677:
        by_case_cond[row["case_id"]][row["condition"]] = row

    table = {}
    for case_id, r674 in rows674.items():
        r675 = rows675[case_id]
        final_norm = component(r675, "final_norm") or {}
        last_attn = last_component(r675, "attn_plus_residual") or {}
        prev_attn = prev_component(r675, "attn_plus_residual") or {}
        last_mlp = last_component(r675, "mlp_plus_residual") or {}
        attn_deltas = [x["delta_gap"] for x in r675["component_deltas"] if x["component"].endswith("attn_plus_residual")]
        mlp_deltas = [x["delta_gap"] for x in r675["component_deltas"] if x["component"].endswith("mlp_plus_residual")]
        pre_diag = r674.get("pre_diag") or {}
        post_diag = r674.get("post_diag") or {}
        baseline = by_case_cond[case_id]["baseline"]
        table[case_id] = {
            "case_id": case_id,
            "relation": r674["relation"],
            "baseline_success": bool(baseline["expected_top1"]),
            "baseline_rank": baseline["expected_rank"],
            "baseline_gap": baseline["gap"],
            "top1_category": r674["top1_category"],
            "top1_not_expected": r674["top1_category"] != "expected",
            "expected_rank": r674["expected_rank"],
            "final_gap": r675["final_gap"],
            "pre_gap": pre_diag.get("competitor_minus_expected"),
            "post_gap": post_diag.get("competitor_minus_expected"),
            "post_unit_gap": post_diag.get("unit_gap"),
            "post_cos_advantage": post_diag.get("competitor_cos_advantage"),
            "final_norm_before_gap": final_norm.get("before_gap"),
            "final_norm_delta": final_norm.get("delta_gap"),
            "last_attn_delta": last_attn.get("delta_gap"),
            "prev_attn_delta": prev_attn.get("delta_gap"),
            "last_mlp_delta": last_mlp.get("delta_gap"),
            "max_attn_delta": max(attn_deltas) if attn_deltas else None,
            "min_mlp_delta": min(mlp_deltas) if mlp_deltas else None,
            "conditions": by_case_cond[case_id],
        }
    return table


def numeric_gate_values(cases: list[dict], feature: str) -> list[float]:
    vals = sorted({float(c[feature]) for c in cases if c.get(feature) is not None and math.isfinite(float(c[feature]))})
    if not vals:
        return []
    # Keep thresholds small and interpretable: observed values around zero plus
    # quartile-like positions from the sorted unique list.
    idxs = {0, len(vals) // 4, len(vals) // 2, (3 * len(vals)) // 4, len(vals) - 1}
    thresholds = {0.0}
    for i in idxs:
        thresholds.add(vals[i])
    for i in range(len(vals) - 1):
        if vals[i] <= 0 <= vals[i + 1]:
            thresholds.add((vals[i] + vals[i + 1]) / 2.0)
            break
    return sorted(thresholds)


def make_gates(cases: list[dict]) -> list[dict]:
    gates = [
        {
            "name": "top1_category_not_expected",
            "kind": "near_readout",
            "fn": lambda c: c["top1_not_expected"],
        },
        {
            "name": "top1_category_word_or_newline_or_other",
            "kind": "near_readout",
            "fn": lambda c: c["top1_category"] in {"word_or_explanation", "newline", "other"},
        },
        {
            "name": "expected_rank_gt_1",
            "kind": "near_readout_upper_bound",
            "fn": lambda c: c["expected_rank"] > 1,
        },
        {
            "name": "expected_rank_gt_10",
            "kind": "near_readout",
            "fn": lambda c: c["expected_rank"] > 10,
        },
    ]
    features = [
        ("final_gap", "readout_gap"),
        ("pre_gap", "pre_final_gap"),
        ("post_unit_gap", "readout_geometry"),
        ("post_cos_advantage", "readout_geometry"),
        ("final_norm_before_gap", "pre_final_gap"),
        ("final_norm_delta", "trajectory"),
        ("last_attn_delta", "trajectory"),
        ("prev_attn_delta", "trajectory"),
        ("last_mlp_delta", "trajectory"),
        ("max_attn_delta", "trajectory"),
        ("min_mlp_delta", "trajectory"),
    ]
    for feature, kind in features:
        for threshold in numeric_gate_values(cases, feature):
            gates.append({
                "name": f"{feature}_gt_{threshold:.4g}",
                "kind": kind,
                "feature": feature,
                "op": ">",
                "threshold": threshold,
                "fn": lambda c, f=feature, t=threshold: c.get(f) is not None and float(c[f]) > t,
            })
            gates.append({
                "name": f"{feature}_lt_{threshold:.4g}",
                "kind": kind,
                "feature": feature,
                "op": "<",
                "threshold": threshold,
                "fn": lambda c, f=feature, t=threshold: c.get(f) is not None and float(c[f]) < t,
            })
    return gates


def evaluate_gate(cases: list[dict], gate: dict, repair_condition: str) -> dict:
    n = len(cases)
    baseline_failures = sum(1 for c in cases if not c["baseline_success"])
    baseline_successes = n - baseline_failures
    predicted = 0
    predicted_failures = 0
    predicted_successes = 0
    repaired_failures = 0
    damaged_successes = 0
    final_top1 = 0
    final_rank_sum = 0.0
    final_gap_sum = 0.0
    for c in cases:
        use_repair = bool(gate["fn"](c))
        predicted += int(use_repair)
        predicted_failures += int(use_repair and not c["baseline_success"])
        predicted_successes += int(use_repair and c["baseline_success"])
        selected = c["conditions"][repair_condition] if use_repair else c["conditions"]["baseline"]
        final_top1 += int(selected["expected_top1"])
        final_rank_sum += selected["expected_rank"]
        final_gap_sum += selected["gap"]
        repaired_failures += int((not c["baseline_success"]) and selected["expected_top1"])
        damaged_successes += int(c["baseline_success"] and not selected["expected_top1"])
    return {
        "gate": gate["name"],
        "kind": gate["kind"],
        "repair_condition": repair_condition,
        "n": n,
        "predicted_rate": predicted / max(1, n),
        "predicted_count": predicted,
        "failure_capture_rate": predicted_failures / max(1, baseline_failures),
        "success_false_positive_rate": predicted_successes / max(1, baseline_successes),
        "selective_top1_rate": final_top1 / max(1, n),
        "selective_mean_rank": final_rank_sum / max(1, n),
        "selective_mean_gap": final_gap_sum / max(1, n),
        "failure_repair_rate": repaired_failures / max(1, baseline_failures),
        "success_damage_rate": damaged_successes / max(1, baseline_successes),
    }


def summarize_model(model: str) -> dict:
    case_table = build_case_table(model)
    cases = list(case_table.values())
    gates = make_gates(cases)
    evaluations = []
    for repair in REPAIR_CONDITIONS:
        for gate in gates:
            evaluations.append(evaluate_gate(cases, gate, repair))

    # Sort by useful selective repair, then low damage, then high failure capture.
    ranked = sorted(
        evaluations,
        key=lambda r: (
            -r["selective_top1_rate"],
            r["success_damage_rate"],
            -r["failure_repair_rate"],
            r["success_false_positive_rate"],
            r["predicted_rate"],
        ),
    )
    by_kind = {}
    for kind in sorted({r["kind"] for r in evaluations}):
        items = [r for r in evaluations if r["kind"] == kind]
        by_kind[kind] = sorted(
            items,
            key=lambda r: (
                -r["selective_top1_rate"],
                r["success_damage_rate"],
                -r["failure_repair_rate"],
                r["success_false_positive_rate"],
            ),
        )[:10]
    return {
        "model": model,
        "n_cases": len(cases),
        "baseline_successes": sum(1 for c in cases if c["baseline_success"]),
        "baseline_failures": sum(1 for c in cases if not c["baseline_success"]),
        "top_overall": ranked[:25],
        "top_by_kind": by_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    models = [summarize_model(m) for m in args.models]
    result = {
        "phase": 679,
        "title": "Internal Failure-Gate Predictor for Selective Readout Repair",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": models,
    }
    (OUT_ROOT / "phase679_failure_gate_predictor.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Phase 679 Internal Failure-Gate Predictor",
        "",
        f"- generated: `{result['timestamp']}`",
        "",
        "| model | gate | kind | repair | pred_rate | fail_capture | false_pos | selective_top1 | fail_repair | damage |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        for row in item["top_overall"][:12]:
            lines.append(
                f"| {item['model']} | {row['gate']} | {row['kind']} | {row['repair_condition']} | "
                f"{row['predicted_rate']:.3f} | {row['failure_capture_rate']:.3f} | "
                f"{row['success_false_positive_rate']:.3f} | {row['selective_top1_rate']:.3f} | "
                f"{row['failure_repair_rate']:.3f} | {row['success_damage_rate']:.3f} |"
            )
    (OUT_ROOT / "phase679_failure_gate_predictor.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
