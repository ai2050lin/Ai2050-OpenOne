#!/usr/bin/env python3
"""Analyze Phase557 natural-color source recompute interventions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
MODELS = ("qwen3", "glm4")
CONFIRMATION_SPLIT = "behavior_confirmation"
UNSEEN_SPLIT = "unseen_recombination"
SPLITS = (CONFIRMATION_SPLIT, UNSEEN_SPLIT)
WRONG_CONTROLS = (
    "wrong_depth_donor_replace",
    "relation_position_donor_replace",
    "channel_roll_donor_replace",
)
GATES = {
    "same_case_max_abs_candidate_logit_delta": 0.05,
    "correct_donor_switch_effect_median_min": 0.50,
    "correct_donor_win_rate_min": 0.50,
    "correct_minus_best_wrong_mean_effect_min": 0.25,
    "delete_recipient_retention_rate_max": 0.75,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def condition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [float(row["donor_switch_effect"]) for row in rows]
    score_deltas = [
        max(
            abs(float(value) - float(row["baseline_scores"][word]))
            for word, value in row["intervention_scores"].items()
        )
        for row in rows
    ]
    return {
        "row_count": len(rows),
        "donor_switch_effect_mean": safe_mean(effects),
        "donor_switch_effect_median": float(statistics.median(effects)) if effects else 0.0,
        "donor_switch_effect_positive_rate": safe_mean([float(value > 0.0) for value in effects]),
        "donor_win_rate": safe_mean([float(row["intervention_donor_wins"]) for row in rows]),
        "recipient_retention_rate": safe_mean([
            float(row["intervention_recipient_retained"]) for row in rows
        ]),
        "candidate_logit_delta_mean_max": safe_mean(score_deltas),
        "candidate_logit_delta_max": max(score_deltas, default=0.0),
    }


def source_rows_path(model: str, split: str) -> Path:
    if split == CONFIRMATION_SPLIT:
        return OUT_DIR / "natural_color_source" / model / "phase557_natural_color_source_rows.jsonl"
    return (
        OUT_DIR / "natural_color_source" / model / split
        / "phase557_natural_color_source_rows.jsonl"
    )


def analyze_model(model: str, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = source_rows_path(model, split)
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"Missing Phase557 natural-color source rows for {model}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    candidate_reports = []
    qualified_edges = []
    for candidate_id, candidate_rows in sorted(grouped.items()):
        conditions = {
            condition: condition_report([
                row for row in candidate_rows if row["condition"] == condition
            ])
            for condition in sorted({row["condition"] for row in candidate_rows})
        }
        expected_count = int(conditions["correct_donor_replace"]["row_count"])
        if any(report["row_count"] != expected_count for report in conditions.values()):
            raise RuntimeError(f"Condition denominator drift for {candidate_id}")
        correct = conditions["correct_donor_replace"]
        best_wrong_mean = max(
            conditions[condition]["donor_switch_effect_mean"] for condition in WRONG_CONTROLS
        )
        specificity_gap = correct["donor_switch_effect_mean"] - best_wrong_mean
        implementation_valid = (
            conditions["same_case_restore"]["candidate_logit_delta_max"]
            <= GATES["same_case_max_abs_candidate_logit_delta"]
        )
        donor_specificity_pass = bool(
            correct["donor_switch_effect_median"]
            >= GATES["correct_donor_switch_effect_median_min"]
            and correct["donor_win_rate"] >= GATES["correct_donor_win_rate_min"]
            and specificity_gap >= GATES["correct_minus_best_wrong_mean_effect_min"]
        )
        necessity_pass = (
            conditions["object_specific_delete"]["recipient_retention_rate"]
            <= GATES["delete_recipient_retention_rate_max"]
        )
        qualified = bool(implementation_valid and donor_specificity_pass and necessity_pass)
        report = {
            "candidate_id": candidate_id,
            "model": model,
            "layer": int(candidate_rows[0]["layer"]),
            "zone": candidate_rows[0]["candidate_zone"],
            "pair_count_per_condition": expected_count,
            "conditions": conditions,
            "correct_minus_best_wrong_mean_effect": specificity_gap,
            "implementation_valid": implementation_valid,
            "donor_specificity_pass": donor_specificity_pass,
            "necessity_pass": necessity_pass,
            "qualified_compute_edge": qualified,
        }
        candidate_reports.append(report)
        if qualified:
            qualified_edges.append({
                "schema_version": "phase557_qualified_natural_color_compute_edge.v1",
                "phase_id": "Phase557",
                "created_at": now(),
                "model": model,
                "candidate_id": candidate_id,
                "layer": report["layer"],
                "source_position": "object_source_end",
                "component": "layer_output",
                "target_relation": "natural_color",
                "evidence_split": split,
                "specificity_gap": specificity_gap,
                "necessary_under_object_centroid_deletion": True,
                "sealed": False,
            })
    return ({
        "model": model,
        "split": split,
        "intervention_row_count": len(rows),
        "candidate_count": len(candidate_reports),
        "qualified_compute_edge_count": len(qualified_edges),
        "candidate_reports": candidate_reports,
    }, qualified_edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default=CONFIRMATION_SPLIT)
    args = parser.parse_args()
    split = args.split
    model_reports = []
    qualified_edges = []
    for model in MODELS:
        report, edges = analyze_model(model, split)
        model_reports.append(report)
        qualified_edges.extend(edges)
    summary = {
        "schema_version": "phase557_natural_color_source_analysis.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "split": split,
        "frozen_gates": GATES,
        "model_reports": model_reports,
        "qualified_compute_edge_count": len(qualified_edges),
        "qualified_models": sorted({row["model"] for row in qualified_edges}),
        "parameter_or_neuron_scan_authorized": bool(qualified_edges),
        "sealed_split_read": False,
        "closure_claim": False,
    }
    analysis_name = (
        "phase557_natural_color_source_analysis.json"
        if split == CONFIRMATION_SPLIT
        else "phase557_natural_color_unseen_source_analysis.json"
    )
    edge_name = (
        "phase557_qualified_natural_color_compute_edges.jsonl"
        if split == CONFIRMATION_SPLIT
        else "phase557_replicated_natural_color_compute_edges.jsonl"
    )
    write_json(OUT_DIR / analysis_name, summary)
    edge_path = OUT_DIR / edge_name
    edge_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in qualified_edges),
        encoding="utf-8",
    )
    print(json.dumps({
        "qualified_compute_edge_count": len(qualified_edges),
        "qualified_models": summary["qualified_models"],
        "candidate_results": {
            report["model"]: [
                {
                    "layer": row["layer"],
                    "qualified": row["qualified_compute_edge"],
                    "specificity_gap": row["correct_minus_best_wrong_mean_effect"],
                }
                for row in report["candidate_reports"]
            ]
            for report in model_reports
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
