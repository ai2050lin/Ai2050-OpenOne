#!/usr/bin/env python3
"""Freeze the next exact-vector coactivity path-object protocol without model execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"


def main() -> None:
    payload = {
        "schema_version": "47.0.0",
        "phase_id": "Phase371",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "replace_lossy_gram_and_hash_energy_summaries_with_a_conservation_preserving_exact_vector_coactivity_tree",
        "evidence_basis": {
            "phase369_raw_relation_full_gate_pass": False,
            "phase370_hash_energy_new_cycle_gate_pass": False,
            "dynamic_state_existence_refuted": False,
            "current_projection_family_declared_sufficient": False,
        },
        "path_object": {
            "node": [
                "generation_time", "layer", "token_position", "event_type",
                "source_role", "receiver_role", "exact_vector_reference",
                "parent_node", "child_nodes", "conservation_residual",
            ],
            "required_event_types": [
                "query_key_score", "attention_head_write", "mlp_single_neuron_write",
                "residual_merge", "generation_feedback", "label_free_vocab_state",
            ],
            "all_token_positions_required_for_feasibility": True,
            "four_role_view_retained_only_as_display_projection": True,
        },
        "conservation_tree": {
            "root_is_exact_component_sum": True,
            "children_are_deterministic_index_partitions": True,
            "parent_equals_sum_of_children_within_numeric_gate": True,
            "refinement_triggers": [
                "high_child_cancellation_ratio", "high_child_direction_diversity",
                "downstream_replay_error_above_repeat_noise",
            ],
            "task_label_or_target_score_top_k_allowed": False,
            "hash_energy_share_used_as_terminal_state": False,
            "exact_vector_cross_terms_retained": True,
        },
        "future_test": {
            "case_nearest_neighbor_used": False,
            "same_forward_graph_replay_used": True,
            "tests": [
                "next_layer_vector_replay", "next_generation_role_state_replay",
                "label_free_vocab_distribution_replay", "history_residual_information",
            ],
            "all_components_must_pass": True,
            "weighted_scalar_score_used": False,
        },
        "cross_model_equivalence": {
            "same_layer_head_or_neuron_number_required": False,
            "unrestricted_coordinate_rotation_fitted": False,
            "preserved": [
                "source_receiver_roles", "partial_order", "branch_merge_structure",
                "conservation", "replay_response_direction", "compensation_response",
            ],
            "standardized_response_fingerprint": [
                "zero_parent", "half_parent", "child_swap_within_parent",
                "wrong_depth_parent", "wrong_position_parent",
            ],
        },
        "staged_execution": [
            {
                "stage": "371A",
                "scope": "one_existing_case_per_model_engineering_replay_only",
                "model_order": ["qwen3", "glm4", "deepseek7b"],
                "claim_authorized": False,
            },
            {
                "stage": "371B",
                "scope": "freeze_storage_numeric_and_replay_gates_before_new_cases",
                "claim_authorized": False,
            },
            {
                "stage": "371C",
                "scope": "new_independent_discovery_and_calibration_cycle_only_if_371A_371B_pass",
                "claim_authorized": False,
            },
        ],
        "authorization": {
            "existing_ledger_engineering_feasibility": True,
            "new_model_generation": False,
            "reuse_phase369_sealed_calibration_for_tuning": False,
            "physical_holdout": False,
            "causal_language_mechanism_claim": False,
        },
        "stop_rules": [
            "stop_if_exact_tree_cannot_reconstruct_parent_with_repeat_noise_gate",
            "stop_if_storage_exceeds_frozen_budget_before_full_case_expansion",
            "stop_if_replay_does_not_improve_over_exact_unsplit_parent",
            "do_not_open_new_data_if_only_energy_or_norm_summaries_improve",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "phase371_exact_vector_coactivity_protocol.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
