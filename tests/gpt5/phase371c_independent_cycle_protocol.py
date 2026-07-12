#!/usr/bin/env python3
"""Freeze the independent exact-vector discovery/calibration cycle without opening cases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
B_REPAIR = PHASE371 / "phase371b_sufficient_state_summary.json"
OUT = PHASE371 / "phase371c_independent_cycle_protocol.json"


def main() -> None:
    phase371b = json.loads(B_REPAIR.read_text(encoding="utf-8"))
    mechanisms = [
        "relation_binding",
        "target_competition",
        "entity_recency",
        "number_agreement",
    ]
    discovery_per_mechanism = 12
    calibration_per_mechanism = 6
    physical_per_mechanism = 4
    condition_count = 4
    model_count = 3
    discovery_groups = discovery_per_mechanism * len(mechanisms)
    calibration_groups = calibration_per_mechanism * len(mechanisms)
    physical_groups = physical_per_mechanism * len(mechanisms)
    payload = {
        "schema_version": "47.6.0",
        "phase_id": "Phase371C",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "independently_test_exact_vector_conservation_paths_without_scalar_or_hash_terminal_states",
        "authorization_basis": {
            "phase371b_repaired_full_gate_pass": phase371b["results"]["phase371b_repaired_full_gate_pass"],
            "protocol_design_authorized": phase371b["results"]["phase371c_protocol_design_authorized"],
            "model_execution_authorized_before_case_and_code_audit": False,
        },
        "frozen_denominator": {
            "mechanisms": mechanisms,
            "model_order": ["qwen3", "glm4", "deepseek7b"],
            "condition_slots_per_parallel_group": condition_count,
            "fresh_discovery_groups_per_mechanism": discovery_per_mechanism,
            "sealed_calibration_groups_per_mechanism": calibration_per_mechanism,
            "sealed_physical_groups_per_mechanism": physical_per_mechanism,
            "fresh_discovery_parallel_groups": discovery_groups,
            "sealed_calibration_parallel_groups": calibration_groups,
            "sealed_physical_parallel_groups": physical_groups,
            "behavior_case_count": (discovery_groups + calibration_groups + physical_groups) * condition_count * model_count,
            "maximum_internal_collection_cases_before_physical": (discovery_groups + calibration_groups) * condition_count * model_count,
            "physical_case_count": physical_groups * condition_count * model_count,
            "prior_prompt_overlap_required": 0,
        },
        "qualification": {
            "all_four_conditions_required_per_model_group": True,
            "common_cross_model_groups_require_all_three_models": True,
            "behavior_qualification_precedes_internal_collection": True,
            "mixed_token_length_batching_forbidden": True,
            "minimum_generation_budget_tokens": 24,
            "failed_groups_are_recorded_not_replaced_after_internal_unsealing": True,
        },
        "private_collection": {
            "anchor_layers": ["first", "floor_half", "last"],
            "generation_time_count": 3,
            "storage_mode": "lossless_sufficient_state",
            "actual_rotary_query_key_value_required": True,
            "all_token_receiver_states_required": True,
            "deterministic_tree_partitions": 8,
            "materialized_head_or_partition_caches_required": False,
            "semantic_label_target_rank_and_margin_available": False,
            "full_discovery_budget_bytes": 64 * 1024**3,
            "minimum_free_disk_reserve_bytes": 200 * 1024**3,
        },
        "discovery_object": {
            "node_identity": [
                "generation_time", "relative_layer_anchor", "token_position_role",
                "event_type", "source_receiver_relation", "exact_vector_reference",
            ],
            "event_types": [
                "query_key_score", "attention_head_write", "mlp_neuron_write",
                "residual_merge", "generation_feedback", "label_free_vocab_state",
            ],
            "branch_and_merge_order_preserved": True,
            "exact_vector_cross_terms_reconstructable": True,
            "scalar_norm_or_hash_energy_as_terminal_state": False,
            "top_k_selection": False,
        },
        "same_graph_replay_gates": {
            "next_layer_vector_replay_required": True,
            "next_generation_role_state_replay_required": True,
            "label_free_vocab_distribution_replay_required": True,
            "history_residual_must_not_improve_replay_beyond_repeat_noise": True,
            "every_gate_pass_required": True,
            "weighted_scalar_score": False,
            "case_nearest_neighbor_future_prediction": False,
        },
        "cross_model_equivalence": {
            "unrestricted_learned_coordinate_rotation": False,
            "same_absolute_layer_head_or_neuron_index_required": False,
            "preserve": [
                "source_receiver_roles", "event_partial_order", "branch_merge_structure",
                "conservation", "same_graph_replay_direction", "compensation_response",
            ],
            "heterogeneous_level2_requires": ["glm4", "qwen3_or_deepseek7b"],
            "qwen3_plus_deepseek7b_only_is_architecture_family_evidence": True,
        },
        "sealed_stage_rules": {
            "discovery_tunes_no_threshold_after_open": True,
            "calibration_open_only_after_discovery_candidate_and_code_hash_freeze": True,
            "physical_open_only_after_heterogeneous_level2_calibration_pass": True,
            "negative_results_are_published_to_atlas": True,
            "failed_candidates_are_not_language_paths": True,
        },
        "execution_authorization": {
            "fresh_case_bank_generation": True,
            "static_case_contract_audit": True,
            "behavior_model_execution": False,
            "internal_model_collection": False,
            "calibration_or_physical_open": False,
            "next_gate": "generate_and_audit_fresh_case_bank_then_freeze_execution_hashes",
        },
        "claim_boundary": {
            "phase371b_proves_measurement_engineering_only": True,
            "phase371c_protocol_proves_no_language_mechanism": True,
            "global_atlas_completion_percentage_valid": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
