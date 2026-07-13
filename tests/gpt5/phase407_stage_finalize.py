#!/usr/bin/env python3
"""Freeze the Phase407 stage decision from formal and diagnostic artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase407_event_horizon_kernel"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read(OUT / "phase407_event_horizon_protocol.json")
    discovery = read(OUT / "phase407_discovery_analysis.json")
    diagnostic = read(OUT / "phase407_failure_diagnostic.json")
    partition = read(OUT / "phase407_response_partition_diagnostic.json")
    qualification = {
        model: read(OUT / "qualification" / f"{model}_complete.json")
        for model in MODELS
    }
    collections = {
        model: read(OUT / "collection/discovery" / model / "complete.json")
        for model in MODELS
    }
    calibration = read(OUT / "phase407_calibration_analysis.json")
    behavioral = read(OUT / "phase407_behavioral_holdout_analysis.json")

    model_family = {
        f"{row['model']}:{row['family_id']}": {
            "case_count": row["case_count"],
            "semantic_correct_count": row["semantic_correct_count"],
            "complete_response_count": row["complete_response_count"],
            "eos_observed_count": row["eos_observed_count"],
            "stop_right_censored_count": row["stop_right_censored_count"],
            "gate_group_pass_counts": row["gate_group_pass_counts"],
            "gate_model_family_pass": row["gate_model_family_pass"],
            "semantic_state_candidate": row["semantic_state_candidate"],
        }
        for row in discovery["model_family_rows"]
    }
    payload = {
        "schema_version": "81.6.0",
        "phase_id": "Phase407-EventHorizonConditionResponseStage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "separate_semantic_boundary_and_stop_events_then_test_surface_"
            "interface_and_history_transfer_before_any_internal_mapping"
        ),
        "source_audit": {
            "phase406_audit_direction_correct": True,
            "accepted_changes": [
                "replace_single_next_token_with_conditioned_response_sequence",
                "use_family_specific_interfaces",
                "compare_inline_and_prior_turn_carried_equivalent_state",
                "separate_surface_interface_history_sequence_and_completion_gates",
                "freeze_fresh_discovery_calibration_behavioral_and_physical_groups",
                "require_all_three_models_before_downstream_mapping",
            ],
            "required_corrections": [
                "generate_to_model_eos_or_H48_instead_of_minimum_event_time",
                "record_tau_semantic_tau_boundary_and_tau_stop_independently",
                "treat_H48_missing_events_as_right_censored",
                "call_target_and_foil_scores_compressed_evidence_not_full_kernel",
                "do_not_call_predeclared_endpoint_transition_a_model_executed_operator",
                "do_not_treat_two_history_conditions_as_all_generation_history",
            ],
        },
        "assessment": {
            "frozen_protocol_complete": True,
            "all_three_execution_qualifications_valid": all(
                row["valid"] for row in qualification.values()
            ),
            "all_three_discovery_collections_valid": all(
                row["valid"] for row in collections.values()
            ),
            "semantic_boundary_and_stop_events_separated": True,
            "compressed_probability_ledger_complete": True,
            "some_registered_state_information_observed": True,
            "stable_single_model_conditioned_state_observed": False,
            "stable_crossmodel_conditioned_state_observed": False,
            "direct_endpoint_operator_validated": False,
            "physical_mapping_authorized": False,
            "causal_or_neuron_work_authorized": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "frozen_all_split_case_count": protocol["denominator"][
                "case_count_all_models_all_splits"
            ],
            "formal_discovery_case_count": discovery["case_count"],
            "formal_condition_cell_count": partition["condition_cell_count"],
            "formal_group_count": diagnostic["formal_group_count"],
            "model_family_cell_count": len(discovery["model_family_rows"]),
            "execution_qualification_unique_case_count": sum(
                row["case_count"] for row in qualification.values()
            ),
            "execution_qualification_total_repeated_runs": sum(
                row["case_count"] * row["repeat_count_per_case"]
                for row in qualification.values()
            ),
            "calibration_case_count_consumed": calibration["case_count"],
            "behavioral_holdout_case_count_consumed": behavioral["case_count"],
            "physical_holdout_case_count_consumed": 0,
        },
        "results": {
            "semantic_correct_count": discovery["semantic_correct_count"],
            "complete_response_count": discovery["complete_response_count"],
            "semantic_reversal_count": discovery["semantic_reversal_count"],
            "eos_observed_count": discovery["eos_observed_count"],
            "semantic_right_censored_count": discovery[
                "semantic_right_censored_count"
            ],
            "stop_right_censored_count": discovery["stop_right_censored_count"],
            "canonical_target_preferred_count": discovery[
                "canonical_target_preferred_count"
            ],
            "target_preferred_but_natural_semantic_wrong_count": diagnostic[
                "target_preferred_but_natural_semantic_wrong_count"
            ],
            "natural_semantic_correct_but_target_not_preferred_count": diagnostic[
                "natural_semantic_correct_but_target_not_preferred_count"
            ],
            "fully_semantic_gated_group_count": diagnostic[
                "fully_semantic_gated_group_count"
            ],
            "formal_group_count": diagnostic["formal_group_count"],
            "single_model_candidate_family_count": sum(
                len(values)
                for values in discovery[
                    "single_model_semantic_candidate_families"
                ].values()
            ),
            "strict_crossmodel_candidate_family_count": len(
                discovery["strict_crossmodel_semantic_candidate_families"]
            ),
            "strict_crossmodel_candidate_families": discovery[
                "strict_crossmodel_semantic_candidate_families"
            ],
            "response_mapping_class_counts": partition["mapping_class_counts"],
            "surface_stable_mapping_count": partition[
                "surface_stable_mapping_count"
            ],
            "surface_mapping_group_count": partition[
                "surface_mapping_group_count"
            ],
            "surface_stable_mapping_class_counts": partition[
                "surface_stable_mapping_class_counts"
            ],
            "nonfinite_generation_path_count": diagnostic[
                "nonfinite_generation_path"
            ]["case_count"],
            "nonfinite_all_reached_H48": diagnostic[
                "nonfinite_generation_path"
            ]["all_nonfinite_cases_reached_H48"],
            "grammar_unparsed_count": diagnostic["grammar_contract_diagnostic"][
                "unparsed_case_count"
            ],
            "grammar_unparsed_perfect_or_progressive_count": diagnostic[
                "grammar_contract_diagnostic"
            ]["unparsed_with_perfect_or_progressive_be_count"],
            "model_family_results": model_family,
            "model_collection_quality": {
                model: {
                    "case_count": row["case_count"],
                    "eos_observed_count": row["eos_observed_count"],
                    "H48_right_edge_count": row["H48_right_edge_count"],
                    "nonfinite_any_step_case_count": row[
                        "nonfinite_any_step_case_count"
                    ],
                    "canonical_target_preferred_count": row[
                        "canonical_target_preferred_count"
                    ],
                }
                for model, row in collections.items()
            },
            "validated_direct_operator_count": 0,
            "new_physical_path_count": 0,
            "new_head_channel_or_neuron_count": 0,
        },
        "hard_limits": [
            "two_interfaces_per_family_are_not_all_legal_future_conditions",
            "two_history_modes_are_not_all_generation_histories",
            "H48_is_an_observation_budget_and_missing_events_are_right_censored",
            "the_external_semantic_parser_is_not_an_internal_state_readout",
            "grammar_prompts_admit_legal_unregistered_continuations",
            "target_foil_preference_and_natural_generation_disagree_in_both_directions",
            "glm4_has_129_fp16_nonfinite_exclamation_degeneration_paths",
            "only_10_of_108_groups_pass_all_four_semantic_gates",
            "zero_of_nine_model_family_cells_pass_the_registered_state_gate",
            "zero_of_three_families_pass_the_strict_crossmodel_gate",
            "the_43_bijective_nonidentity_cells_are_not_surface_stable",
            "the_endpoint_transition_graph_was_not_executed_by_the_model",
            "no_internal_activation_or_physical_path_was_collected",
        ],
        "authorization": {
            "show_event_response_ledger": True,
            "show_response_partition_as_observational": True,
            "show_nonidentity_mapping_as_validated_operator": False,
            "show_conditioned_state_as_validated": False,
            "show_physical_state_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_calibration": False,
            "run_behavioral_holdout": False,
            "run_direct_operator": False,
            "run_physical_mapping": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "next_stage": {
            "phase_id": "Phase408",
            "objective": (
                "validate_query_contract_exclusivity_and_separate_state_"
                "information_from_condition_specific_response_mapping"
            ),
            "automatic_model_execution_authorized": False,
            "reason": (
                "Phase407_closes_its_downstream_gate_and_Phase408_requires_a_"
                "new_task_contract_new_denominator_and_new_execution_qualification"
            ),
            "required_changes": [
                "use_semantically_exclusive_response_slots_for_each_family",
                "separate_state_distinguishability_from_answer_correctness",
                "freeze_identity_permutation_collapse_and_missing_mapping_cells",
                "include_history_content_controls_not_only_turn_placement",
                "add_a_runtime_rejection_gate_for_model_specific_nonfinite_paths",
                "require_surface_stability_before_treating_any_permutation_as_a_candidate",
                "consume_no_physical_holdout_until_crossmodel_functional_calibration_passes",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    path = OUT / "phase407_event_horizon_stage_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
