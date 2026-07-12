#!/usr/bin/env python3
"""Freeze signed Phase383 event descriptors before discovery values are analyzed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    audit = read_json(OUT / "phase383_instrument_audit_summary.json")
    manifests = [
        read_json(OUT / "collection/discovery/models" / model / "manifest.json")
        for model in MODELS
    ]
    discovery_valid = (
        audit["authorization"]["discovery_collection"]
        and all(row["valid"] for row in manifests)
        and all(row["case_count"] == 48 for row in manifests)
        and all(row["baseline_replay_match_count"] == 48 for row in manifests)
    )
    if not discovery_valid:
        raise RuntimeError("Phase383 discovery collection is not valid")
    contract = {
        "schema_version": "57.3.0",
        "phase_id": "Phase383-SignedEventContract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_event_value_analysis": True,
        "factorial_effects": {
            "content": "((A-C)+(B-D))/2",
            "operation": "((A-B)+(C-D))/2",
            "interaction": "((A-B)-(C-D))/2",
            "linearity_claimed": False,
            "interpretation": "contrast coordinates only",
        },
        "event_families": [
            "layer_input_state",
            "attention_source_write",
            "attention_other_sources_write",
            "attention_output_write",
            "post_attention_state",
            "mlp_output_write",
            "layer_output_state",
        ],
        "descriptors_kept_separate": {
            "relative_amplitude": "norm(delta_event)/(norm(delta_terminal)+epsilon)",
            "signed_alignment": (
                "dot(delta_event,delta_terminal)/(norm(delta_event)*"
                "norm(delta_terminal)+epsilon)"
            ),
            "signed_projection": (
                "dot(delta_event,delta_terminal)/(norm(delta_terminal)^2+epsilon)"
            ),
            "absolute_cosine_used": False,
            "amplitude_clipped": False,
            "composite_score_used": False,
        },
        "normalization": {
            "relative_depth_bin_count": 8,
            "layer_is_independent_sample": False,
            "group_is_independent_unit": True,
            "discovery_groups_per_mechanism": 3,
        },
        "candidate_gates": {
            "minimum_group_count": 3,
            "all_groups_positive_direction": True,
            "minimum_group_median_signed_alignment": 0.2,
            "minimum_group_median_relative_amplitude": 0.02,
            "minimum_wrong_depth_alignment_margin": 0.05,
            "minimum_wrong_receiver_alignment_margin": 0.05,
            "all_gates_separate": True,
            "threshold_retuning_on_discovery": False,
        },
        "crossmodel_gate": {
            "level2": "GLM4 and at least one of qwen3/deepseek7b pass",
            "level3": "all three models pass",
            "same_layer_head_or_channel_required": False,
            "same_functional_signature_required": True,
        },
        "controls": {
            "wrong_depth": "depth bin shifted by four modulo eight",
            "wrong_receiver": "best other receiver role in the same cell family",
            "time_shuffle": "not identifiable with one target-decision snapshot",
            "equal_energy_intervention": "not run in descriptive discovery",
        },
        "claim_boundary": {
            "three_discovery_groups_establish_stability": False,
            "signed_event_candidate_is_causal_path": False,
            "source_role_aggregation_is_single_head_localization": False,
            "lazy_exact_head_and_channel_families_remain_replayable": True,
            "terminal_prediction_gain_computed": False,
        },
        "authorization": {
            "signed_discovery_map_extraction": discovery_valid,
            "calibration_collection": False,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
    }
    write_json(OUT / "phase383_signed_event_contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
