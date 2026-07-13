#!/usr/bin/env python3
"""Freeze Phase402's executable multi-parent K/V experiment.

The protocol deliberately maps every parent category to a disjoint set of
actual attention K/V positions.  It does not treat semantic labels as tensors
and does not call the remaining prompt prefix a generated-history state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase402_multiparent_graph"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "entity_attribute_binding",
    "role_filling",
    "coreference_resolution",
    "two_step_composition",
    "conditional_presence",
    "number_agreement",
)
SPLIT_CANDIDATE_COUNTS = {
    "discovery": 12,
    "calibration": 6,
    "physical_holdout": 6,
}
SPLIT_SELECTED_COUNTS = {
    "discovery": 6,
    "calibration": 3,
    "physical_holdout": 3,
}
FROZEN_DTYPES = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}
PARENT_CATEGORIES = (
    "source_content",
    "source_structure",
    "query_local",
    "remaining_prefix",
)
CONTROL_NAMES = (
    "true_relation",
    "same_target_wrong_order",
    "wrong_receiver_role",
    "wrong_semantic_time",
    "wrong_depth_quarter_shift",
    "source_content_role_permutation",
    "same_content_wrong_structure",
    "deterministic_random_natural_donor",
    "same_absolute_mass_sign_permuted",
)


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
        "schema_version": "76.0.0",
        "phase_id": "Phase402-MultiParentProtocol",
        "created_at": now(),
        "objective": (
            "test_disjoint_real_attention_kv_parent_partitions_before_any_"
            "causal_state_or_operator_claim"
        ),
        "phase401_evidence_boundary": {
            "formal_behavior_cases": 4608,
            "semantic_correct_cases": 4557,
            "complete_crossmodel_groups": 66,
            "eligible_surfaces": ["possession_relation", "role_filling"],
            "strict_local_edge_layers": 0,
            "strict_local_edge_layer_denominator": 208,
            "function_specific_direct_cells": 0,
            "function_specific_direct_cell_denominator": 6,
            "calibration_consumed": False,
            "physical_holdout_consumed": False,
        },
        "hypothesis_audit": {
            "some_computation_implements_observed_behavior": "necessary",
            "implementation_is_compact_or_human_readable": "unproven",
            "implementation_restores_brain_language_structure": "unproven",
            "causal_state_quotient_exists_at_current_measurement_resolution": "unproven",
            "operator_algebra_exists": "unproven",
            "single_neurons_heads_or_channels_are_never_joint_mechanism_parts": "not_established",
        },
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "batch_size": 1,
            "padding": "none",
            "attention_implementation": "eager",
            "use_cache": True,
            "do_sample": False,
            "max_new_tokens": 10,
            "runtime_dtype_by_model": FROZEN_DTYPES,
            "behavior_trace_and_intervention_shape_identical": True,
        },
        "fresh_behavior_denominator": {
            "surfaces": list(SURFACES),
            "conditions_per_group": 16,
            "candidate_groups_per_surface": sum(SPLIT_CANDIDATE_COUNTS.values()),
            "candidate_split_group_counts": SPLIT_CANDIDATE_COUNTS,
            "selected_split_group_counts": SPLIT_SELECTED_COUNTS,
            "candidate_case_count": (
                len(SURFACES)
                * sum(SPLIT_CANDIDATE_COUNTS.values())
                * 16
                * len(MODELS)
            ),
            "complete_group_gate": (
                "all_16_conditions_semantically_correct_and_span_resolved_in_all_models"
            ),
            "failed_groups_retained": True,
            "no_cross_split_backfill": True,
        },
        "factorial_contract": {
            "levels": ["lexical_replica", "binding", "clause_order", "query_entity"],
            "full_conditions": 16,
            "claims_are_surface_specific": True,
            "two_step_surface_contains_an_explicit_intermediate_key": True,
            "conditional_surface_targets_yes_or_no": True,
            "agreement_surface_targets_is_or_are_from_source_quantity": True,
        },
        "direct_child": "query_position_attention_output",
        "executable_parent_partition": {
            "source_content": [
                "source_entity_a",
                "source_entity_b",
                "source_value_a",
                "source_value_b",
            ],
            "source_structure": ["all_other_source_tokens"],
            "query_local": ["query_entity", "other_query_tokens", "receiver_self"],
            "remaining_prefix": ["all_unclaimed_prior_prompt_tokens"],
            "partition_is_disjoint": True,
            "partition_conserves_every_position_up_to_receiver": True,
            "remaining_prefix_is_generated_history": False,
            "semantic_parent_labels_are_directly_intervened": False,
        },
        "subset_contract": {
            "categories": list(PARENT_CATEGORIES),
            "subset_count_including_empty": 16,
            "operation": (
                "replace_selected_donor_KV_partition_positions_then_recompute_"
                "recipient_query_attention_with_recipient_query_and_mask"
            ),
            "empty_subset_is_exact_replay_control": True,
            "all_subsets_are_real_compute_graph_parent_interventions": True,
        },
        "control_applicability": {
            "true_relation": "physical",
            "same_target_wrong_order": "physical",
            "wrong_receiver_role": "physical",
            "wrong_semantic_time": "physical",
            "wrong_depth_quarter_shift": "physical",
            "source_content_role_permutation": "physical",
            "same_content_wrong_structure": "physical",
            "deterministic_random_natural_donor": "physical",
            "same_absolute_mass_sign_permuted": "physical_measurement_only",
            "semantic_axis_for_same_target_control": "not_applicable",
            "a_not_applicable_semantic_control_is_not_counted_as_failure": True,
        },
        "instrument_gate": {
            "exact_first_token_replay_required": True,
            "empty_subset_attention_relative_error_max": 0.01,
            "partition_conservation_required": True,
            "all_instrument_cases_and_layers_required": True,
        },
        "pair_gate": {
            "minimum_informative_baseline_relative_norm": 0.01,
            "direction_cosine_min": 0.60,
            "state_recovery_min": 0.20,
        },
        "group_layer_subset_gate": {
            "informative_pair_rate_min": 0.75,
            "pair_pass_rate_min": 0.75,
            "median_state_recovery_min": 0.25,
            "joint_subset_min_size": 2,
            "joint_minus_best_contained_singleton_median_recovery_min": 0.10,
            "joint_minus_best_contained_singleton_pair_pass_rate_min": 0.20,
            "true_minus_each_control_median_recovery_min": 0.10,
            "true_minus_each_control_pair_pass_rate_min": 0.20,
            "each_control_required_separately": True,
            "controls_cannot_be_averaged": True,
        },
        "discovery_candidate_gate": {
            "qualified_discovery_group_rate_min": 0.666666667,
            "same_surface_subset_and_relative_depth_zone_required": True,
            "crossmodel_public_candidate_requires_all_three_models": True,
            "two_model_result_may_only_be_registered_as_partial_replication": True,
            "candidate_selection_uses_discovery_only": True,
        },
        "stage_separation": {
            "discovery_target": "direct_child_specific_joint_parent_candidate",
            "terminal_prediction_measured_in_discovery": False,
            "natural_generation_intervened_in_discovery": False,
            "calibration_open_condition": (
                "at_least_one_frozen_crossmodel_direct_child_candidate"
            ),
            "physical_holdout_open_condition": "candidate_passes_frozen_calibration",
            "propagation_and_terminal_test_open_condition": (
                "candidate_passes_frozen_calibration"
            ),
        },
        "stopping_rules": {
            "no_surface_behavior_eligible": "close_phase_before_internal_collection",
            "instrument_failure": "repair_instrument_and_do_not_run_discovery",
            "no_joint_subset_exceeds_best_singleton": (
                "close_current_four_partition_joint_parent_hypothesis"
            ),
            "any_control_matches_true": "register_non_specific_numeric_response_only",
            "direct_candidate_without_terminal_test": (
                "register_direct_child_candidate_not_language_path"
            ),
            "no_crossmodel_candidate": (
                "keep_calibration_physical_holdout_and_neuron_scan_closed"
            ),
        },
        "authorization": {
            "run_behavior_after_protocol_freeze": True,
            "run_instrument_after_behavior_freeze": True,
            "run_discovery_after_instrument_pass": True,
            "run_calibration_before_discovery_candidate_freeze": False,
            "run_physical_holdout": False,
            "run_propagation_terminal": False,
            "run_head_channel_or_neuron_scan": False,
        },
        "claim_boundary": {
            "successful_behavior_proves_brain_structure_restoration": False,
            "direct_joint_parent_candidate_is_a_minimal_sufficient_language_state": False,
            "direct_joint_parent_candidate_is_a_language_operator": False,
            "negative_result_excludes_finer_joint_mechanisms": False,
            "phase402_is_language_encoding_closure": False,
        },
    }


def main() -> None:
    payload = protocol()
    write_json(OUT / "phase402_multiparent_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
