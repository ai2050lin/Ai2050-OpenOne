#!/usr/bin/env python3
"""Validate frozen Phase398 order-conditioned ROQ candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = ("calibration", "physical_holdout")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(stage: str) -> None:
    freeze = read_json(OUT / "phase398_order_conditioned_candidate_freeze.json")
    if stage == "calibration" and not freeze["authorization"]["run_calibration_trace"]:
        raise RuntimeError("Phase398 calibration was not authorized")
    if stage == "physical_holdout":
        calibration = read_json(OUT / "phase398_order_conditioned_calibration_validation.json")
        if not calibration["authorization"]["open_physical_holdout"]:
            raise RuntimeError("Phase398 physical holdout was not authorized")
    gates = freeze["frozen_gates"]
    cells = []
    complete_rows = []
    for model in MODELS:
        root = OUT / f"query_trace/{stage}/private/models/{model}"
        complete = read_json(root / "complete.json")
        complete_rows.append(complete)
        effect_rows = read_jsonl(root / "factorial_effect_rows.jsonl")
        audits = read_jsonl(root / "group_audit_rows.jsonl")
        audit_by_group = {row["public_parallel_group_id"]: row for row in audits}
        for frozen in [row for row in freeze["frozen_cells"] if row["model"] == model]:
            items = [
                row for row in effect_rows
                if row["layer_index"] == frozen["candidate_layer"]
                and row["coordinate"] == freeze["candidate_coordinate"]
                and row["component"] == freeze["candidate_component"]
                and audit_by_group[row["public_parallel_group_id"]]["surface_private"] == frozen["task_surface"]
            ]
            valid_items = [
                row for row in items
                if audit_by_group[row["public_parallel_group_id"]]["prefix_transition_match_count"] == 16
                and audit_by_group[row["public_parallel_group_id"]]["target_completion_argmax_match_count"] == 16
                and audit_by_group[row["public_parallel_group_id"]]["block_conservation_pass"]
            ]
            roq_norms = [row["factorial_effect_metrics"]["ROQ"]["min_axis_normalized_norm"] for row in valid_items]
            roq_cosines = [row["factorial_effect_metrics"]["ROQ"]["cross_axis_cosine"] for row in valid_items]
            ratios = [
                row["factorial_effect_metrics"]["ROQ"]["min_axis_normalized_norm"]
                / max(row["factorial_effect_metrics"]["RQ"]["min_axis_normalized_norm"], 1e-12)
                for row in valid_items
            ]
            valid_group_count = len(valid_items)
            if not valid_items:
                med_norm = med_cos = med_ratio = 0.0
            else:
                med_norm, med_cos, med_ratio = median(roq_norms), median(roq_cosines), median(ratios)
            support = sum(value >= 0.02 for value in roq_norms)
            passed = bool(
                valid_group_count >= 3
                and med_norm >= gates["minimum_median_min_axis_normalized_roq_norm"]
                and support >= gates["minimum_calibration_or_physical_support_groups_at_0_02"]
                and med_cos >= gates["minimum_median_roq_cross_axis_cosine"]
                and med_ratio >= gates["minimum_median_roq_to_rq_norm_ratio"]
            )
            cells.append({
                "model": model,
                "task_surface": frozen["task_surface"],
                "candidate_layer": frozen["candidate_layer"],
                "registered_group_count": len(items),
                "valid_exact_replay_group_count": valid_group_count,
                "excluded_whole_group_count": len(items) - valid_group_count,
                "median_min_axis_normalized_roq_norm": round(med_norm, 9),
                "roq_support_group_count_at_0_02": support,
                "median_roq_cross_axis_cosine": round(med_cos, 9),
                "median_roq_to_rq_norm_ratio": round(med_ratio, 9),
                "validation_gate_pass": passed,
            })
    pass_count = sum(row["validation_gate_pass"] for row in cells)
    all_pass = pass_count == 9
    result = {
        "schema_version": "72.9.0",
        "phase_id": f"Phase398-OrderConditioned-{stage.title()}Validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "frozen_gates": gates,
        "denominator": {
            "registered_case_count": sum(row["case_count"] for row in complete_rows),
            "registered_group_count": sum(row["group_count"] for row in complete_rows),
            "model_surface_cell_count": 9,
            "groups_per_model_surface": 4,
            "valid_exact_replay_group_count": sum(row["valid_exact_replay_group_count"] for row in cells),
            "excluded_whole_group_count": sum(row["excluded_whole_group_count"] for row in cells),
        },
        "results": {
            "passing_model_surface_cell_count": pass_count,
            "all_registered_replays_exact": all(row["target_completion_argmax_match_count"] == row["case_count"] for row in complete_rows),
            "all_component_conservation_pass": all(row["all_block_conservation_pass"] for row in complete_rows),
        },
        "cells": cells,
        "claim_boundary": {
            "independent_observational_replication": all_pass,
            "roq_is_causal_binding_state": False,
            "roq_is_portable_across_order": False,
            "full_language_path_established": False,
            "single_neuron_mechanism_established": False,
        },
        "authorization": {
            "open_physical_holdout": stage == "calibration" and all_pass,
            "run_causal_intervention": False,
            "single_neuron_scan": False,
        },
    }
    name = f"phase398_order_conditioned_{'calibration' if stage == 'calibration' else 'physical'}_validation.json"
    (OUT / name).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    main(args.stage)
