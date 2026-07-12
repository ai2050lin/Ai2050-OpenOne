#!/usr/bin/env python3
"""Freeze Phase369 raw-relation discovery gates before reading discovery outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/discovery_gate_freeze"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    payload = {
        "schema_version": "46.2.0",
        "phase_id": "Phase369",
        "created_at": now(),
        "frozen_before_discovery_evaluation": True,
        "unit_of_replication": "independent_anonymous_group_not_route_layer_or_condition",
        "future_prediction_gates_per_model": {
            "raw_mean_future_flow_error_strictly_below_low_ten_descriptor": True,
            "raw_case_win_fraction_over_low_strictly_above": 0.5,
            "raw_mean_vocab_error_not_above_low": True,
            "raw_mean_future_flow_error_below_random": True,
            "raw_mean_future_flow_error_below_time_shuffle": True,
            "raw_mean_future_flow_error_below_role_permutation": True,
            "raw_mean_future_flow_error_below_equal_energy_wrong_flow": True,
            "raw_mean_future_flow_error_below_public_backbone": True,
        },
        "cross_model_gates_per_pair": {
            "raw_residual_matched_separation_ratio_below_one": True,
            "raw_residual_matched_separation_ratio_below_low_residual_ratio": True,
            "raw_residual_top5_retrieval_rate_above_low_residual": True,
            "raw_residual_top5_retrieval_rate_above_random_rate": True,
        },
        "evidence_levels": {
            "level_1": "one_model_passes_all_future_prediction_components",
            "level_2": "glm4_qwen3_or_glm4_deepseek7b_pair_passes_all_cross_model_components_and_both_models_are_level_1",
            "architecture_family_only": "qwen3_deepseek7b_pair_without_glm4",
            "level_3": "all_three_pairs_pass_and_all_three_models_are_level_1",
        },
        "calibration_entry_gate": "at_least_one_level_2_pair",
        "weighted_scalar_score_used": False,
        "calibration_may_retune_gate": False,
        "semantic_labels_or_target_rank_used": False,
        "physical_holdout_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "phase369_discovery_gate_freeze.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
