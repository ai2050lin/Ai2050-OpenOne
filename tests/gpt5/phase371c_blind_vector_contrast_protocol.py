#!/usr/bin/env python3
"""Freeze blind all-pair exact-vector contrast before any condition unblinding."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PATH_SUMMARY = PHASE371 / "phase371c_lazy_exact_paths/phase371c_lazy_exact_path_summary.json"
OUT = PHASE371 / "phase371c_blind_vector_contrast_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = json.loads(PATH_SUMMARY.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "47.16.0",
        "phase_id": "Phase371C-Contrast",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "compare_all_anonymous_condition_pairs_on_exact_path_vectors_before_semantic_unblinding",
        "authorization_basis": {
            "lazy_path_object_valid": paths["valid"],
            "candidate_language_path_count_before_contrast": paths["results"]["candidate_language_path_count"],
            "path_summary_hash": sha256_file(PATH_SUMMARY),
        },
        "frozen_denominator": {
            "model_count": 3,
            "parallel_group_count": 22,
            "model_group_count": 66,
            "anonymous_conditions_per_model_group": 4,
            "unordered_condition_pairs_per_model_group": 6,
            "pair_rows_before_route_expansion": 396,
            "generation_time_count": 3,
            "local_layer_pair_count": 3,
            "roles": ["source_end", "query_end", "answer_start", "current_generation"],
        },
        "blind_extraction": {
            "all_six_unordered_pairs_retained": True,
            "all_three_perfect_matchings_retained": True,
            "condition_semantics_available": False,
            "family_or_mechanism_labels_available": False,
            "target_rank_or_margin_available": False,
            "top_k_route_selection": False,
            "weighted_scalar_score": False,
        },
        "exact_route_families": [
            "layer_input_difference",
            "attention_head_partition_difference",
            "attention_merge_difference",
            "post_attention_difference",
            "mlp_neuron_partition_difference",
            "mlp_merge_difference",
            "layer_output_difference",
            "label_free_vocab_distribution_difference",
        ],
        "navigation_indices_not_terminal_states": [
            "exact_difference_norm",
            "signed_cosine_to_parent_difference",
            "child_parent_inner_product_share",
            "child_cancellation_with_siblings",
            "adjacent_output_direction_persistence",
        ],
        "same_graph_gates": {
            "component_difference_sum_reconstructs_layer_output_difference": True,
            "head_children_reconstruct_attention_difference": True,
            "neuron_children_reconstruct_mlp_difference": True,
            "source_layer_output_equals_receiver_layer_input": True,
            "next_layer_output_relation_beats_wrong_depth_control": True,
            "next_generation_relation_beats_time_shuffle_control": True,
            "history_residual_gate_required": True,
            "all_gates_must_pass_separately": True,
        },
        "controls": [
            "all_unordered_condition_pairs",
            "wrong_depth_receiver",
            "wrong_token_role",
            "generation_time_shuffle",
            "head_partition_permutation",
            "mlp_partition_permutation",
        ],
        "candidate_freeze": {
            "no_candidate_is_selected_by_a_single_navigation_index": True,
            "candidate_requires_exact_vector_and_same_graph_gate_vector": True,
            "candidate_code_and_rows_hashed_before_condition_unblinding": True,
            "calibration_remains_sealed_during_discovery": True,
            "failed_routes_remain_in_negative_atlas": True,
        },
        "cross_model": {
            "absolute_layer_head_neuron_ids_compared": False,
            "unrestricted_coordinate_rotation_fitted": False,
            "functional_partial_order_and_response_required": True,
            "glm4_required_for_heterogeneous_level2": True,
        },
        "authorization": {
            "implement_and_hash_blind_contrast_extractor": True,
            "execute_contrast_before_extractor_hash_freeze": False,
            "open_semantic_condition_key": False,
            "open_calibration": False,
            "open_physical": False,
        },
        "next_decision": "implement_streaming_exact_contrast_extractor_then_audit_cost_before_execution",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
