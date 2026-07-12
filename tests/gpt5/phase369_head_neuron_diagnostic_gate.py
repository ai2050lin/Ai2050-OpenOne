#!/usr/bin/env python3
"""Freeze exploratory head/neuron diagnostic gates before evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/head_neuron_topology_diagnostic"


def main() -> None:
    payload = {
        "schema_version": "46.4.0",
        "phase_id": "Phase369-Diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_diagnostic_evaluation": True,
        "combination_rule": "minimize_worst_component_candidate_rank_then_rank_sum",
        "weighted_scalar_distance_used": False,
        "future_components_per_model_resolution": {
            "composite_mean_future_flow_error_below_raw_relation": True,
            "composite_case_win_fraction_over_raw_above": 0.5,
            "composite_mean_vocab_error_not_above_raw_relation": True,
        },
        "cross_model_components_per_pair_resolution": {
            "composite_matched_separation_ratio_below_raw_relation": True,
            "composite_top5_rate_above_raw_relation": True,
            "composite_top5_rate_above_random": True,
        },
        "new_cycle_entry": {
            "minimum_passing_hash_resolutions": 2,
            "requires_glm4_plus_qwen3_or_deepseek7b": True,
            "both_models_must_pass_future_components": True,
        },
        "multiple_hash_resolutions_or_seeds_are_independent_replications": False,
        "success_cannot_reopen_phase369_calibration": True,
        "success_only_authorizes_new_independent_cycle": True,
        "physical_holdout_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "phase369_head_neuron_diagnostic_gate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
