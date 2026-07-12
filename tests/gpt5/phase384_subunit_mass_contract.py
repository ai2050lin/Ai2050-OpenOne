#!/usr/bin/env python3
"""Freeze all-subunit projection-mass analysis before Phase384 extraction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase384_exact_subunit_mass_map"
SOURCE = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    calibration = read_json(SOURCE / "phase383_calibration_summary.json")
    authorized = calibration["authorization"]["exact_subunit_coordinate_expansion"]
    if not authorized:
        raise RuntimeError("Phase384 exact subunit expansion is not authorized")
    contract = {
        "schema_version": "58.0.0",
        "phase_id": "Phase384-SubunitMassContract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_before_subunit_descriptor_extraction": True,
        "source_phase": "Phase383",
        "objective": (
            "Distinguish absent parent signal from opposing exact head/channel writes "
            "without selecting top units."
        ),
        "exact_subunit_families": {
            "attention": (
                "receiver role x all source-position partitions x every attention head"
            ),
            "mlp": "receiver role x every MLP product channel",
            "source_partitions": [
                "source",
                "query",
                "answer_start",
                "current_generation",
                "other_sources",
            ],
            "alias_partition_priority": [
                "source",
                "query",
                "current_generation",
                "answer_start",
            ],
            "aliased_lower_priority_partition_is_zero": True,
            "top_k_used": False,
            "all_units_in_denominator": True,
        },
        "projection_mass": {
            "subunit_projection": (
                "dot(delta_subunit,delta_terminal)/(norm(delta_terminal)^2+epsilon)"
            ),
            "positive_mass": "sum(max(projection_i,0))",
            "negative_mass": "sum(max(-projection_i,0))",
            "absolute_mass": "positive_mass+negative_mass",
            "net_projection": "sum(projection_i)",
            "cancellation_fraction": (
                "1-abs(net_projection)/(absolute_mass+epsilon)"
            ),
            "parent_projection_conservation_required": True,
            "maximum_parent_projection_absolute_error": 0.02,
            "composite_language_score_used": False,
        },
        "frozen_pattern_gates": {
            "minimum_absolute_projection_mass": 0.1,
            "coherent_maximum_cancellation_fraction": 0.5,
            "coherent_minimum_absolute_net_projection": 0.05,
            "opposing_minimum_cancellation_fraction": 0.8,
            "all_discovery_groups_must_pass": True,
            "all_calibration_groups_must_pass": True,
            "upstream_depth_bins": [0, 1, 2, 3, 4, 5],
            "threshold_retuning_allowed": False,
        },
        "crossmodel_gate": {
            "level2": "GLM4 plus qwen3 or deepseek7b",
            "level3": "all three models",
            "same_head_or_channel_index_required": False,
            "same_role_depth_family_pattern_required": True,
        },
        "claim_boundary": {
            "opposing_mass_is_language_path": False,
            "coherent_mass_is_causal_path": False,
            "projection_mass_is_neuron_identity_equivalence": False,
            "one_semantic_time_covers_generation_dynamics": False,
        },
        "authorization": {
            "discovery_subunit_mass_extraction": authorized,
            "calibration_subunit_mass_extraction": False,
            "physical_holdout": False,
            "causal_intervention": False,
        },
    }
    write_json(OUT / "phase384_subunit_mass_contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
