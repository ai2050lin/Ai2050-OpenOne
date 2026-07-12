#!/usr/bin/env python3
"""Freeze Phase371B anchor collection, numeric, storage, and claim gates."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PHASE371A = PHASE371 / "engineering_feasibility/phase371a_existing_ledger_tree_feasibility_summary.json"
OUT = PHASE371 / "phase371b_anchor_qk_protocol.json"


def main() -> None:
    phase371a = json.loads(PHASE371A.read_text(encoding="utf-8"))
    estimated_all_layer_bytes = int(phase371a["storage"]["estimated_additional_all_token_qk_bytes"])
    average_layer_count = (36 + 40 + 28) / 3
    estimated_three_anchor_bytes = round(estimated_all_layer_bytes * 3 / average_layer_count)
    free_bytes = int(shutil.disk_usage(ROOT).free)
    payload = {
        "schema_version": "47.2.0",
        "phase_id": "Phase371B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "freeze_three_anchor_all_token_qk_collection_and_exact_replay_gates_before_any_new_case_cycle",
        "evidence_basis": {
            "phase371a_role_limited_exact_tree_pass": bool(
                phase371a["results"]["role_limited_exact_tree_numeric_gate_pass"]
            ),
            "phase371a_all_token_states_available": bool(
                phase371a["results"]["all_token_receiver_states_available"]
            ),
            "phase371a_query_key_states_available": bool(
                phase371a["results"]["query_key_states_available"]
            ),
        },
        "collection_contract": {
            "case_source": "phase369_frozen_fresh_discovery_execution_cases",
            "engineering_case_count_per_model": 1,
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "generation_time_count": 3,
            "anchor_rule": ["first_layer", "floor(layer_count/2)", "last_layer"],
            "receiver_positions": "all_sequence_tokens",
            "query_key_state": "actual_rotary_transformed_tensors_entering_eager_attention",
            "required_private_tensors": [
                "layer_input_all_tokens",
                "input_normalized_state_all_tokens",
                "query_states_all_tokens",
                "key_states_all_tokens",
                "value_states_all_tokens",
                "attention_probabilities_all_receivers_all_sources",
                "attention_head_writes_all_receivers",
                "post_attention_normalized_state_all_tokens",
                "mlp_channel_product_all_tokens",
                "mlp_partition_writes_all_receivers",
                "layer_output_all_tokens",
            ],
            "target_rank_or_margin_available": False,
            "semantic_labels_available": False,
            "physical_holdout_opened": False,
        },
        "numeric_gates": {
            "query_key_probability_relative_error_max": 0.01,
            "attention_head_replay_relative_error_max": 0.01,
            "attention_tree_conservation_relative_error_max": 1e-5,
            "mlp_direct_replay_relative_error_max": 0.01,
            "mlp_tree_conservation_relative_error_max": 1e-5,
            "block_replay_relative_error_max": 0.01,
            "all_rows_must_pass": True,
        },
        "storage_gates": {
            "estimated_all_layer_additional_bytes": estimated_all_layer_bytes,
            "estimated_three_anchor_additional_bytes": estimated_three_anchor_bytes,
            "full_discovery_additional_budget_bytes": 64 * 1024**3,
            "minimum_free_disk_reserve_bytes": 200 * 1024**3,
            "free_disk_bytes_at_freeze": free_bytes,
            "engineering_measurement_required_before_expansion": True,
        },
        "promotion_gates": {
            "371b_engineering_pass_requires": [
                "three_models_all_numeric_gates_pass",
                "actual_measured_full_discovery_projection_within_budget",
                "free_disk_reserve_preserved",
                "private_file_hash_and_shape_contract_valid",
            ],
            "371c_new_cycle_authorized_by_protocol_alone": False,
            "language_mechanism_claim_authorized": False,
            "atlas_language_path_promotion_authorized": False,
        },
        "stop_rules": [
            "stop_if_actual_qk_cannot_reproduce_attention_probabilities",
            "stop_if_exact_children_do_not_reconstruct_their_parent",
            "stop_if_three_anchor_projection_exceeds_storage_budget",
            "stop_if_only_a_scalar_summary_improves",
            "do_not_open_calibration_or_physical_holdout_during_engineering",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
