#!/usr/bin/env python3
"""Freeze Phase398 discovery gates before opening factorial trace effects."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"


def main() -> None:
    protocol = {
        "schema_version": "72.6.0",
        "phase_id": "Phase398-DiscoveryAnalysisProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_candidate_space": {
            "coordinate": "query_end",
            "component": "layer_output",
            "effect": "RQ",
            "models": ["qwen3", "glm4", "deepseek7b"],
            "eligible_surfaces": ["possession_relation", "role_filling", "coreference_resolution"],
            "groups_per_model_surface": 8,
            "depths": "all_layers_with_no_depth_exclusion",
        },
        "frozen_gates": {
            "minimum_median_min_axis_normalized_rq_norm": 0.005,
            "minimum_support_groups_at_0_005": 6,
            "minimum_median_rq_to_max_nuisance_interaction_ratio": 1.25,
            "nuisance_interactions": ["RO", "OQ", "ROQ"],
            "minimum_shared_operation_cross_axis_cosine": 0.1,
        },
        "classification": {
            "shared_operation_candidate": "magnitude_and_specificity_gates_pass_and_median_cross_axis_cosine_ge_0.1",
            "content_conditioned_interaction_candidate": "magnitude_and_specificity_gates_pass_but_direction_gate_fails",
            "not_qualified": "magnitude_or_specificity_gate_fails",
        },
        "calibration_authorization_rule": "all_nine_model_surface_cells_must_pass_magnitude_and_specificity_at_some_layer",
        "causal_authorization_rule": "calibration_must_reproduce_each_frozen_cell_before_any_intervention",
        "controls": {
            "analyze_all_coordinates_components_and_effects_descriptively": True,
            "answer_anchor_is_not_a_negative_control": True,
            "target_completion_margin_is_confidence_only_not_answer_identity": True,
            "single_neuron_scan": False,
        },
        "authorization": {
            "open_discovery_effects": True,
            "run_calibration_trace": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
        },
    }
    path = OUT / "phase398_discovery_analysis_protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["frozen_gates"] != protocol["frozen_gates"]:
            raise RuntimeError("Phase398 discovery gates were already frozen differently")
        protocol = existing
    else:
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
