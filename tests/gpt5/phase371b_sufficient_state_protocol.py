#!/usr/bin/env python3
"""Freeze the lossless sufficient-state repair after the materialization budget failure."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
AUDIT = PHASE371 / "phase371b_engineering_summary.json"
OUT = PHASE371 / "phase371b_sufficient_state_protocol.json"


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "47.4.0",
        "phase_id": "Phase371B-R",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "remove_only_deterministic_materialized_derivatives_and_validate_exact_on_demand_replay",
        "trigger": {
            "three_model_numeric_gate_pass": audit["results"]["three_model_numeric_gate_pass"],
            "materialized_derivative_storage_gate_pass": audit["results"]["materialized_derivative_storage_gate_pass"],
            "sufficient_state_storage_projection_pass": audit["results"]["sufficient_state_storage_projection_pass"],
        },
        "retained_state": [
            "all_component_vectors",
            "actual_rotary_query_key_value_states",
            "all_receiver_all_source_attention_probabilities",
            "mlp_channel_product",
            "deterministic_head_and_channel_partitions",
            "weight_reference_ids",
            "quality_and_claim_boundaries",
        ],
        "removed_materializations": [
            "attention_head_writes_all_receivers",
            "attention_head_partition_writes_all_receivers",
            "mlp_partition_writes_all_receivers",
        ],
        "lossless_replay_gates": {
            "query_key_probability_relative_error_max": 0.01,
            "attention_output_relative_error_max": 0.01,
            "attention_tree_conservation_relative_error_max": 1e-5,
            "mlp_output_relative_error_max": 0.01,
            "mlp_tree_conservation_relative_error_max": 1e-5,
            "removed_materialization_reconstruction_relative_error_max": 0.01,
            "block_output_relative_error_max": 0.01,
            "all_27_rows_must_pass": True,
        },
        "storage_gate": {
            "full_discovery_budget_bytes": audit["storage"]["frozen_budget_bytes"],
            "minimum_free_reserve_bytes": audit["storage"]["minimum_free_reserve_bytes"],
            "projection_uses_actual_compacted_file_sizes": True,
        },
        "claim_boundary": {
            "engineering_format_repair_only": True,
            "new_model_execution": False,
            "language_path_claim": False,
            "calibration_or_physical_holdout": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
