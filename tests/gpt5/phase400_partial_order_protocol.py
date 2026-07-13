#!/usr/bin/env python3
"""Freeze Phase400 dynamic partial-order and prediction gates before new data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase400_partial_order"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "possession_relation",
    "role_filling",
    "coreference_resolution",
    "field_extraction",
)
SPLIT_CANDIDATE_COUNTS = {"discovery": 12, "calibration": 6, "physical_holdout": 6}
SPLIT_SELECTED_COUNTS = {"discovery": 8, "calibration": 4, "physical_holdout": 4}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def protocol() -> dict[str, Any]:
    event_classes = {
        "source_content": {
            "prefixes": [
                "state:source_entity_a:layer_output",
                "state:source_entity_b:layer_output",
                "state:source_value_a:layer_output",
                "state:source_value_b:layer_output",
            ],
            "required": False,
            "semantic_time": "source_encoded",
        },
        "source_structure": {
            "prefixes": [
                "state:clause_end_0:layer_output",
                "state:clause_end_1:layer_output",
            ],
            "required": False,
            "semantic_time": "source_encoded",
        },
        "source_to_query_route": {
            "prefixes": [
                "route:source_entity_a->query_end:attention_write",
                "route:source_entity_b->query_end:attention_write",
                "route:source_value_a->query_end:attention_write",
                "route:source_value_b->query_end:attention_write",
                "route:source_structure->query_end:attention_write",
            ],
            "required": True,
            "semantic_time": "query_integrated",
        },
        "query_attention": {
            "prefixes": ["state:query_end:attention_output"],
            "required": True,
            "semantic_time": "query_integrated",
        },
        "query_mlp": {
            "prefixes": ["state:query_end:mlp_output"],
            "required": True,
            "semantic_time": "query_integrated",
        },
        "query_residual": {
            "prefixes": ["state:query_end:layer_output"],
            "required": True,
            "semantic_time": "query_integrated",
        },
        "terminal_route": {
            "prefixes": [
                "route:source_entity_a->first_answer:attention_write",
                "route:source_entity_b->first_answer:attention_write",
                "route:source_value_a->first_answer:attention_write",
                "route:source_value_b->first_answer:attention_write",
                "route:source_structure->first_answer:attention_write",
                "route:query_entity->first_answer:attention_write",
                "route:query_context->first_answer:attention_write",
            ],
            "required": True,
            "semantic_time": "first_answer",
        },
        "terminal_content": {
            "prefixes": ["state:first_answer:layer_output"],
            "required": True,
            "semantic_time": "first_answer",
        },
        "completion_continuation": {
            "prefixes": [
                "state:target_completion:layer_output",
                "state:post_target:layer_output",
            ],
            "required": False,
            "semantic_time": "completion_or_post_target",
        },
    }
    return {
        "schema_version": "74.0.0",
        "phase_id": "Phase400-PartialOrderProtocol",
        "created_at": now(),
        "objective": "test_frozen_dynamic_partial_order_graphs_on_a_fresh_crossmodel_denominator",
        "prior_information_boundary": {
            "phase399_used_only_for_protocol_development": True,
            "phase399_cases_used_for_phase400_validation": False,
            "phase399_physical_holdout_reopened": False,
            "phase400_thresholds_frozen_before_phase400_behavior_execution": True,
        },
        "models_in_execution_order": list(MODELS),
        "behavior_denominator": {
            "surfaces": list(SURFACES),
            "candidate_groups_per_surface": 24,
            "conditions_per_group": 16,
            "candidate_case_count": 4 * 24 * 16 * 3,
            "candidate_split_group_counts": SPLIT_CANDIDATE_COUNTS,
            "selected_split_group_counts": SPLIT_SELECTED_COUNTS,
            "surface_gate": "at_least_8_discovery_4_calibration_4_physical_groups_with_all_16_conditions_correct_in_all_three_models",
            "selected_group_priority_frozen_before_behavior": True,
            "failed_groups_recorded_and_never_replaced_beyond_frozen_priority_selection": True,
        },
        "semantic_times": [
            "source_encoded",
            "query_integrated",
            "first_answer",
            "target_completion",
            "post_target",
        ],
        "event_classes": event_classes,
        "per_group_layer_gate": {
            "roq_min_axis_normalized_norm_min": 0.01,
            "roq_cross_axis_cosine_min": 0.75,
            "roq_to_competing_interaction_min": 1.25,
            "inherited_unchanged_from_phase399": True,
        },
        "interval_contract": {
            "single_internal_gap_fill_max": 1,
            "minimum_consecutive_qualified_layers": 2,
            "onset_definition": "first_layer_of_first_qualified_interval",
            "offset_definition": "last_layer_of_last_qualified_interval",
            "duration_definition": "qualified_layer_count_after_single_gap_fill",
            "amplification_absolute_delta_min": 0.005,
            "amplification_relative_delta_min": 0.25,
            "flip_transition_cosine_max": -0.25,
            "transition_requires_both_layers_qualified": True,
            "all_intervals_persisted": True,
            "single_peak_is_not_used_as_event_order": True,
        },
        "discovery_node_gate": {
            "group_interval_pass_rate_min": 0.75,
            "median_interval_duration_layers_min": 2.0,
            "median_interval_roq_norm_min": 0.015,
            "median_interval_cross_axis_cosine_min": 0.85,
            "median_interval_specificity_ratio_min": 1.50,
        },
        "validation_node_gate": {
            "group_interval_pass_rate_min": 0.75,
            "median_interval_duration_layers_min": 2.0,
            "median_interval_roq_norm_min": 0.012,
            "median_interval_cross_axis_cosine_min": 0.80,
            "median_interval_specificity_ratio_min": 1.25,
        },
        "required_nodes": [
            "source_to_query_route",
            "query_attention",
            "query_mlp",
            "query_residual",
            "terminal_route",
            "terminal_content",
        ],
        "required_edges": [
            ["source_to_query_route", "query_attention", "same_layer_parent_sum"],
            ["query_attention", "query_mlp", "same_layer_compute_order"],
            ["query_mlp", "query_residual", "same_layer_parent_sum"],
            ["query_residual", "terminal_route", "next_semantic_time"],
            ["terminal_route", "terminal_content", "same_layer_parent_sum"],
        ],
        "edge_gate": {
            "same_time_interval_distance_layers_max": 1,
            "group_edge_pass_rate_min": 0.75,
            "next_time_requires_both_nodes_not_layer_monotonicity": True,
            "edge_must_match_real_compute_graph": True,
        },
        "crossmodel_isomorphism_gate": {
            "all_three_models_required": True,
            "required_node_type_coverage_min": 1.0,
            "required_edge_type_coverage_min": 1.0,
            "pairwise_onset_order_agreement_min": 0.70,
            "relative_onset_tie_tolerance": 0.08,
            "pairwise_normalized_duration_difference_max": 0.35,
            "relative_depth_shift_allowed": True,
            "peak_layer_identity_required": False,
            "arbitrary_coordinate_rotation_allowed": False,
        },
        "prediction_contract": {
            "graph_query_layers": "intersection_of_one_layer_dilated_source_route_query_attention_query_mlp_query_residual_intervals",
            "graph_vote": "median_target_minus_distractor_logit_lens_margin_over_graph_query_layers",
            "correct_answer_case_accuracy_min": 0.75,
            "group_accuracy_min": 0.75,
            "next_time_node_recovery_min": 0.80,
            "next_time_edge_recovery_min": 0.80,
            "improvement_over_discovery_frozen_best_single_layer_min": 0.02,
            "improvement_over_wrong_depth_min": 0.10,
            "improvement_over_depth_reversal_min": 0.10,
            "improvement_over_deterministic_random_graph_min": 0.10,
            "wrong_query_accuracy_max": 0.35,
            "single_layer_selected_on_discovery_only": True,
            "validation_thresholds_reselected": False,
        },
        "controls": [
            "best_single_layer_frozen_on_discovery",
            "single_peak_layer",
            "wrong_query_label",
            "wrong_depth_half_model_shift",
            "depth_reversal",
            "deterministic_hash_random_graph_same_layer_count",
            "same_nodes_shuffled_event_order",
        ],
        "raw_anchor_contract": {
            "one_discovery_group_per_model_surface": True,
            "signed_roq_vectors_for_all_predeclared_events_and_layers": True,
            "stored_dtype": "float16",
            "private_only": True,
            "published_to_client": False,
        },
        "causal_authorization": {
            "requires_same_surface_three_model_discovery_pass": True,
            "requires_same_surface_three_model_calibration_pass": True,
            "requires_same_surface_three_model_physical_pass": True,
            "requires_crossmodel_isomorphism_pass": True,
            "requires_prediction_pass": True,
            "head_channel_neuron_scan_before_joint_gate": False,
        },
        "stopping_rules": {
            "partial_order_not_better_than_peak_chain": "close_partial_order_algorithm",
            "events_replicate_but_prediction_fails": "register_observational_stage_skeleton_only",
            "crossmodel_requires_posthoc_mapping": "reject_functional_isomorphism",
            "wrong_controls_match_true_graph": "register_sequence_covariation_only",
            "joint_damage_fails": "do_not_scan_heads_channels_or_neurons",
            "single_model_only": "register_model_specific_schedule_only",
        },
        "claim_boundary": {
            "partial_order_observation_is_causal": False,
            "prediction_is_natural_necessity": False,
            "aggregate_event_is_a_head_channel_or_neuron": False,
            "phase400_success_is_complete_language_closure": False,
        },
    }


def main() -> None:
    payload = protocol()
    write_json(OUT / "phase400_partial_order_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
