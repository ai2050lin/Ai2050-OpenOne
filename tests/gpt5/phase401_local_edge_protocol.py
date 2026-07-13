#!/usr/bin/env python3
"""Freeze Phase401 execution, semantic-span, and local-edge gates.

This file must be executed before any Phase401 model output is generated.  The
thresholds deliberately keep engineering conservation, local causal response,
and language-level prediction as separate claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "possession_relation",
    "role_filling",
    "coreference_resolution",
    "field_extraction",
)
SPLIT_CANDIDATE_COUNTS = {
    "discovery": 12,
    "calibration": 6,
    "physical_holdout": 6,
}
SPLIT_SELECTED_COUNTS = {
    "discovery": 8,
    "calibration": 4,
    "physical_holdout": 4,
}
FROZEN_DTYPES = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def protocol() -> dict[str, Any]:
    return {
        "schema_version": "75.0.1",
        "phase_id": "Phase401-LocalEdgeProtocol",
        "created_at": now(),
        "objective": (
            "separate_semantic_format_and_stop_spans_under_one_execution_shape_"
            "then_test_real_compute_graph_parent_child_responses"
        ),
        "prior_information_boundary": {
            "phase400_results_used_for_protocol_design_only": True,
            "phase400_cases_reused": False,
            "phase400_discovery_or_calibration_reopened": False,
            "all_phase401_thresholds_frozen_before_phase401_behavior": True,
        },
        "protocol_amendment_001": {
            "trigger": (
                "independent_batch_pilot_exposed_that_repeated_target_text_was_"
                "incorrectly_counted_as_semantic_failure"
            ),
            "scientific_reason": (
                "target_repetition_belongs_to_format_suffix_or_continuation_and_"
                "must_not_be_confounded_with_semantic_content_correctness"
            ),
            "changed_field": "output_span_contract.semantic_match",
            "old_rule": "target_present_once_and_no_distractor",
            "new_rule": "target_present_at_least_once_and_no_distractor",
            "unchanged": [
                "all_samples",
                "all_split_assignments",
                "all_numeric_gates",
                "all_edge_classes",
                "all_controls",
            ],
            "pre_amendment_outputs_are_instrument_only": True,
            "all_formal_models_reexecuted_after_amendment": True,
        },
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "batch_size": 1,
            "padding": "none_for_single_unpadded_prompt",
            "attention_implementation": "eager",
            "use_cache": True,
            "do_sample": False,
            "max_new_tokens": 12,
            "runtime_dtype_by_model": FROZEN_DTYPES,
            "behavior_and_trace_execution_shape_identical": True,
            "execution_shape_is_measurement_machine_state_not_semantic_state": True,
            "formal_denominator_may_not_use_batch_sensitivity_rows": True,
        },
        "batch_sensitivity_audit": {
            "independent_pilot_groups_per_surface": 1,
            "conditions_per_group": 16,
            "models": list(MODELS),
            "batch_sizes": [1, 8],
            "pilot_case_count_per_batch_shape": 4 * 16 * 3,
            "formal_denominator_overlap": False,
            "comparison_fields": [
                "generated_token_ids",
                "semantic_correct",
                "semantic_span",
                "format_prefix",
                "format_suffix",
                "stop_step",
            ],
        },
        "behavior_denominator": {
            "surfaces": list(SURFACES),
            "candidate_groups_per_surface": 24,
            "conditions_per_group": 16,
            "candidate_case_count": 4 * 24 * 16 * 3,
            "candidate_split_group_counts": SPLIT_CANDIDATE_COUNTS,
            "selected_split_group_counts": SPLIT_SELECTED_COUNTS,
            "surface_gate": (
                "at_least_8_discovery_4_calibration_4_physical_groups_with_all_"
                "16_conditions_semantically_correct_and_span_resolved_in_all_models"
            ),
            "failed_groups_retained_and_never_backfilled": True,
        },
        "output_span_contract": {
            "decomposition": [
                "format_prefix",
                "semantic_answer",
                "format_suffix",
                "stop",
            ],
            "semantic_match": (
                "unicode_nfkc_casefolded_whole_word_target_present_at_least_once_"
                "and_no_whole_word_distractor_before_stop"
            ),
            "semantic_start": (
                "latest_token_start_of_the_shortest_completion_prefix_whose_"
                "decoded_segment_contains_the_target_alias"
            ),
            "semantic_completion": (
                "first_generated_token_step_whose_decoded_prefix_contains_target_alias"
            ),
            "stop": "first_eos_token_or_generation_limit_if_eos_absent",
            "format_exactness_is_recorded_but_not_a_semantic_behavior_gate": True,
            "target_repetition_is_recorded_as_format_or_continuation_not_semantic_failure": True,
            "same_shape_trace_requires_exact_generated_token_replay": True,
        },
        "instrument_ledger_gates": {
            "attention_probability_replay_relative_error_max": 0.01,
            "attention_output_replay_relative_error_max": 0.01,
            "block_output_replay_relative_error_max": 0.01,
            "mlp_output_replay_relative_error_max": 0.01,
            "layer_output_to_next_input_relative_error_max": 1e-6,
            "all_cases_and_layers_required": True,
        },
        "edge_classes": {
            "source_kv_to_query_attention": {
                "parent": "source_role_key_and_value_states",
                "child": "query_attention_output",
                "operation": (
                    "replace_role_aligned_source_KV_from_matched_donor_and_recompute_"
                    "attention_with_recipient_query_mask_and_other_source_states"
                ),
                "language_candidate": True,
            },
            "query_attention_to_post_attention": {
                "parent": "query_attention_output",
                "child": "query_post_attention_residual",
                "operation": "recipient_layer_input_plus_donor_attention_output",
                "language_candidate": False,
                "reason": "architectural_addition_ledger_edge",
            },
            "post_attention_to_mlp": {
                "parent": "query_post_attention_residual",
                "child": "query_mlp_output",
                "operation": "recipient_layer_norm_and_mlp_recomputed_on_donor_parent",
                "language_candidate": True,
            },
            "mlp_to_layer_output": {
                "parent": "query_mlp_output",
                "child": "query_layer_output",
                "operation": "recipient_post_attention_plus_donor_mlp_output",
                "language_candidate": False,
                "reason": "architectural_addition_ledger_edge",
            },
            "layer_output_to_next_input": {
                "parent": "query_layer_output",
                "child": "next_layer_query_input",
                "operation": "identity_continuity_check",
                "language_candidate": False,
                "reason": "architectural_identity_ledger_edge",
            },
        },
        "intervention_pair_contract": {
            "true_pair": "same_axis_order_query_with_relation_level_toggled",
            "directions": "both_R0_to_R1_and_R1_to_R0",
            "role_alignment": "semantic_role_positions_not_absolute_token_indices",
            "recipient_other_parents_held_natural": True,
            "natural_donor_child_is_prediction_target": True,
            "minimum_informative_baseline_relative_norm": 0.01,
        },
        "edge_response_fields": {
            "positive_mass": "sum(max(delta_i,0))",
            "negative_mass": "sum(max(-delta_i,0))",
            "net_mass": "sum(delta_i)",
            "absolute_mass": "positive_mass_plus_negative_mass",
            "direction_cosine": "cosine(counterfactual_minus_recipient,donor_minus_recipient)",
            "state_recovery": (
                "1_minus_norm(counterfactual_minus_donor)_over_"
                "norm(recipient_minus_donor)"
            ),
        },
        "discovery_local_edge_gate": {
            "informative_pair_rate_min": 0.75,
            "pair_direction_cosine_min": 0.60,
            "pair_state_recovery_min": 0.20,
            "pair_pass_rate_min": 0.75,
            "median_state_recovery_min": 0.25,
            "qualified_discovery_group_rate_min": 0.75,
            "layer_selected_on_discovery_only": True,
            "separate_gate_fields_not_weighted_sum": True,
        },
        "validation_local_edge_gate": {
            "informative_pair_rate_min": 0.75,
            "pair_direction_cosine_min": 0.55,
            "pair_state_recovery_min": 0.15,
            "pair_pass_rate_min": 0.75,
            "median_state_recovery_min": 0.20,
            "qualified_group_rate_min": 0.75,
            "discovery_layer_and_role_map_frozen": True,
            "thresholds_reselected": False,
        },
        "counterfactual_controls": {
            "required_separately": [
                "wrong_source_order_matched_same_target",
                "wrong_receiver_role",
                "wrong_semantic_time",
                "wrong_depth_quarter_model_shift",
                "source_role_permutation",
                "same_content_wrong_structure",
                "deterministic_random_natural_donor",
                "same_absolute_mass_sign_permuted",
            ],
            "true_minus_each_control_median_recovery_min": 0.10,
            "true_minus_each_control_pair_pass_rate_min": 0.20,
            "control_failure_cannot_be_averaged_away": True,
        },
        "semantic_prediction_gate": {
            "target": "donor_target_minus_recipient_target_logit_lens_competition",
            "pair_positive_shift_rate_min": 0.65,
            "median_normalized_competition_recovery_min": 0.10,
            "improvement_over_each_control_min": 0.10,
            "direct_child_only_without_semantic_prediction": (
                "register_local_physical_edge_not_language_path"
            ),
        },
        "crossmodel_functional_equivalence": {
            "all_three_models_required": True,
            "glm4_required": True,
            "same_source_receiver_roles_required": True,
            "same_component_edge_class_required": True,
            "same_semantic_time_required": True,
            "same_sign_class_required": True,
            "same_control_outcome_required": True,
            "relative_depth_zone_agreement_required": True,
            "relative_depth_shift_allowed": True,
            "absolute_layer_head_or_neuron_identity_required": False,
            "posthoc_rotation_or_mapping_allowed": False,
        },
        "stage_gate": (
            "execution_and_semantic_and_ledger_and_local_edge_and_prediction_and_"
            "calibration_must_all_pass_before_physical_holdout"
        ),
        "authorization": {
            "run_behavior_after_protocol_freeze": True,
            "run_instrument_after_behavior_surface_gate": True,
            "run_discovery_after_instrument_ledger_gate": True,
            "run_calibration_after_discovery_local_and_prediction_gates": True,
            "open_physical_after_calibration_joint_gate": True,
            "head_channel_or_neuron_scan": False,
        },
        "stopping_rules": {
            "same_shape_trace_unstable": "repair_instrument_and_stop_mechanism_analysis",
            "ledger_fails": "stop_before_local_edge_analysis",
            "direct_child_passes_terminal_prediction_fails": (
                "register_local_physical_edge_only"
            ),
            "any_required_control_matches_true_edge": (
                "register_common_numeric_response_not_functional_edge"
            ),
            "single_model_only": "register_model_specific_edge_only",
            "no_calibration_surface": "keep_physical_and_causal_stages_closed",
        },
        "claim_boundary": {
            "execution_shape_is_a_language_latent_variable": False,
            "architecture_ledger_edge_is_a_language_mechanism": False,
            "local_parent_response_is_a_complete_language_path": False,
            "observational_recovery_is_natural_necessity": False,
            "phase401_success_is_language_encoding_closure": False,
        },
    }


def main() -> None:
    payload = protocol()
    write_json(OUT / "phase401_local_edge_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
