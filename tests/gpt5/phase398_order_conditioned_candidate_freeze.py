#!/usr/bin/env python3
"""Freeze the exploratory Phase398 ROQ candidate before calibration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling", "coreference_resolution")
GATES = {
    "minimum_median_min_axis_normalized_roq_norm": 0.03,
    "minimum_discovery_support_groups_at_0_02": 6,
    "minimum_calibration_or_physical_support_groups_at_0_02": 3,
    "minimum_median_roq_cross_axis_cosine": 0.75,
    "minimum_median_roq_to_rq_norm_ratio": 2.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    path = OUT / "phase398_order_conditioned_candidate_freeze.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
        return
    profile = read_jsonl(OUT / "phase398_discovery_physical_map.jsonl")
    frozen = []
    for model in MODELS:
        for surface in SURFACES:
            rows = [
                row for row in profile
                if row["model"] == model and row["task_surface"] == surface
                and row["coordinate"] == "query_end" and row["component"] == "layer_output"
            ]
            assessed = []
            for row in rows:
                roq = row["effect_summary"]["ROQ"]
                rq = row["effect_summary"]["RQ"]
                ratio = roq["median_min_axis_normalized_norm"] / max(rq["median_min_axis_normalized_norm"], 1e-12)
                passes = bool(
                    roq["median_min_axis_normalized_norm"] >= GATES["minimum_median_min_axis_normalized_roq_norm"]
                    and roq["support_at_0_02"] >= GATES["minimum_discovery_support_groups_at_0_02"]
                    and roq["median_cross_axis_cosine"] >= GATES["minimum_median_roq_cross_axis_cosine"]
                    and ratio >= GATES["minimum_median_roq_to_rq_norm_ratio"]
                )
                assessed.append((row, ratio, passes))
            qualified = [item for item in assessed if item[2]]
            pool = qualified or assessed
            selected, ratio, passes = max(
                pool,
                key=lambda item: item[0]["effect_summary"]["ROQ"]["median_min_axis_normalized_norm"]
                * max(item[0]["effect_summary"]["ROQ"]["median_cross_axis_cosine"], 0.0),
            )
            frozen.append({
                "model": model,
                "task_surface": surface,
                "candidate_layer": selected["layer_index"],
                "candidate_relative_depth": selected["relative_depth"],
                "discovery_median_min_axis_normalized_roq_norm": selected["effect_summary"]["ROQ"]["median_min_axis_normalized_norm"],
                "discovery_roq_support_group_count_at_0_02": selected["effect_summary"]["ROQ"]["support_at_0_02"],
                "discovery_median_roq_cross_axis_cosine": selected["effect_summary"]["ROQ"]["median_cross_axis_cosine"],
                "discovery_median_roq_to_rq_norm_ratio": round(ratio, 9),
                "discovery_candidate_gate_pass": passes,
            })
    all_pass = all(row["discovery_candidate_gate_pass"] for row in frozen)
    result = {
        "schema_version": "72.8.0",
        "phase_id": "Phase398-OrderConditionedCandidateFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_discovery_candidate_frozen_before_independent_calibration",
        "hypothesis": "the_relation_query_operation_is_order_conditioned_and_appears_as_a_lexically_reusable_ROQ_trajectory_rather_than_an_order_invariant_RQ_vector",
        "candidate_coordinate": "query_end",
        "candidate_component": "layer_output",
        "frozen_gates": GATES,
        "frozen_cells": frozen,
        "claim_boundary": {
            "discovery_roq_is_independently_validated": False,
            "roq_is_a_causal_binding_state": False,
            "roq_is_a_portable_vector": False,
            "order_is_semantically_necessary_for_correct_answer": False,
        },
        "authorization": {
            "run_calibration_trace": all_pass,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
            "single_neuron_scan": False,
        },
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
