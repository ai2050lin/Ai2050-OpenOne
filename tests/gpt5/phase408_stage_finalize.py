#!/usr/bin/env python3
"""Freeze the Phase408 decision without promoting response maps to internal states."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase408_partition_interface"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_binding", "rule_reasoning", "grammar_constraint")
STAGES = ("discovery", "calibration", "behavioral_holdout")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_candidates(payload: dict[str, Any]) -> list[str]:
    return list(payload.get("strict_crossmodel_partition_candidate_families", []))


def main() -> None:
    protocol = read(OUT / "phase408_partition_interface_protocol.json")
    qualification = {
        model: read(OUT / "qualification" / f"{model}_complete.json")
        for model in MODELS
    }
    analyses = {
        stage: read(OUT / f"phase408_{stage}_analysis.json") for stage in STAGES
    }
    collections = {
        stage: {
            model: read(OUT / "collection" / stage / model / "complete.json")
            for model in MODELS
        }
        for stage in STAGES
    }

    discovery = analyses["discovery"]
    calibration = analyses["calibration"]
    behavioral = analyses["behavioral_holdout"]
    diagnostic = read(OUT / "phase408_failure_diagnostic.json")
    recovery_audit = read(OUT / "phase408_execution_recovery_audit.json")
    discovery_candidates = stage_candidates(discovery)
    calibration_candidates = stage_candidates(calibration)
    behavioral_candidates = stage_candidates(behavioral)
    all_qualification_valid = all(row.get("valid") for row in qualification.values())
    all_collection_markers_valid = all(
        row.get("valid")
        for stage_rows in collections.values()
        for row in stage_rows.values()
    )
    history_replication_authorized = bool(behavioral_candidates)
    physical_authorized = False

    model_family_results = {
        stage: {
            f"{row['model']}:{row['family_id']}": {
                "group_count": row["group_count"],
                "functional_group_pass_count": row["functional_group_pass_count"],
                "label_aligned_group_count": row["label_aligned_group_count"],
                "candidate_signature_group_count": row[
                    "candidate_signature_group_count"
                ],
                "required_candidate_signature_group_count": row[
                    "required_candidate_signature_group_count"
                ],
                "model_family_partition_candidate": row[
                    "model_family_partition_candidate"
                ],
            }
            for row in analyses[stage].get("model_family_audits", [])
        }
        for stage in STAGES
    }
    semantic_counts = {
        key: int(discovery.get("semantic_class_counts", {}).get(key, 0))
        for key in ("allowed", "rejected", "ambiguous", "unparsed")
    }
    runtime_counts = {
        key: int(discovery.get("runtime_numeric_status_counts", {}).get(key, 0))
        for key in ("valid", "invalid")
    }
    payload = {
        "schema_version": "82.6.0",
        "phase_id": "Phase408-ExclusiveResponsePartitionInterfaceStage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test_exclusive_response_partitions_and_interface_covariance_before_"
            "any_internal_physical_or_neuron_mapping"
        ),
        "source_audit": {
            "phase407_followup_direction_correct": True,
            "accepted_changes": [
                "separate_state_distinguishability_from_label_alignment",
                "freeze_finite_exclusive_machine_response_registries",
                "use_three_interfaces_and_three_slot_identifiable_permutations",
                "separate_semantic_numeric_event_and_censoring_axes",
                "require_surface_and_lexical_replica_stability",
                "gate_calibration_behavioral_and_physical_splits_sequentially",
            ],
            "required_corrections": [
                "stepwise_greedy_is_not_global_sequence_MAP",
                "semantic_runtime_and_event_axes_are_not_six_exclusive_classes",
                "response_separation_is_not_proof_of_internal_information_retention",
                "cycle_consistency_from_shared_rows_is_not_independent_evidence",
                "task_coordinate_maps_are_external_protocol_objects_not_internal_operators",
                "crossmodel_behavioral_partitions_are_not_physical_invariants",
                "single_global_progress_percentage_is_invalid",
            ],
        },
        "assessment": {
            "machine_exclusive_registry_frozen": True,
            "independent_rule_reviewer_agreement_recorded": False,
            "original_preregistration_fully_satisfied": False,
            "all_three_execution_qualifications_valid": all_qualification_valid,
            "all_required_collection_markers_valid": all_collection_markers_valid,
            "semantic_numeric_event_and_censoring_axes_separated": True,
            "functional_response_partition_observed": discovery[
                "functional_group_pass_count"
            ]
            > 0,
            "strict_crossmodel_discovery_partition_observed": bool(
                discovery_candidates
            ),
            "strict_crossmodel_calibration_partition_observed": bool(
                calibration_candidates
            ),
            "strict_crossmodel_behavioral_partition_observed": bool(
                behavioral_candidates
            ),
            "independent_history_controls_executed": False,
            "validated_internal_state_operator": False,
            "physical_mapping_executed": False,
            "causal_or_neuron_work_executed": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "all_registered_case_count": protocol["denominator"][
                "case_count_all_models_all_registered_splits"
            ],
            "qualification_unique_case_count": sum(
                row["case_count"] for row in qualification.values()
            ),
            "discovery_case_count": discovery["case_count"],
            "discovery_group_count": discovery["group_count"],
            "calibration_case_count_consumed": calibration["case_count"],
            "calibration_group_count_consumed": calibration["group_count"],
            "behavioral_holdout_case_count_consumed": behavioral["case_count"],
            "behavioral_holdout_group_count_consumed": behavioral["group_count"],
            "physical_holdout_case_count_consumed": 0,
            "model_family_cell_count": len(MODELS) * len(FAMILIES),
        },
        "results": {
            "semantic_class_counts": semantic_counts,
            "runtime_numeric_status_counts": runtime_counts,
            "registered_response_observed_count": discovery[
                "registered_response_observed_count"
            ],
            "allowed_response_observed_count": discovery[
                "allowed_response_observed_count"
            ],
            "boundary_observed_count": discovery["boundary_observed_count"],
            "stop_observed_count": discovery["stop_observed_count"],
            "H48_right_edge_count": discovery["H48_right_edge_count"],
            "condition_separation_pass_group_count": discovery[
                "condition_separation_pass_count"
            ],
            "surface_lexical_stability_pass_group_count": discovery[
                "surface_lexical_stability_pass_count"
            ],
            "task_coordinate_covariance_pass_group_count": discovery[
                "task_coordinate_covariance_pass_count"
            ],
            "functional_group_pass_count": discovery[
                "functional_group_pass_count"
            ],
            "condition_status_counts": diagnostic["condition_status_counts"],
            "condition_cell_count": diagnostic["condition_cell_count"],
            "condition_separating_count": diagnostic[
                "condition_separating_count"
            ],
            "condition_label_aligned_count": diagnostic[
                "condition_label_aligned_count"
            ],
            "native_runtime_recovery_count": recovery_audit[
                "native_crash_exit_count"
            ],
            "discovery_crossmodel_candidate_families": discovery_candidates,
            "calibration_crossmodel_candidate_families": calibration_candidates,
            "behavioral_crossmodel_candidate_families": behavioral_candidates,
            "glm_inclusive_pair_candidate_families": discovery[
                "glm_inclusive_pair_candidate_families"
            ],
            "model_family_results": model_family_results,
            "collection_quality": {
                stage: {
                    model: {
                        "case_count": row.get("case_count", 0),
                        "eos_observed_count": row.get("eos_observed_count", 0),
                        "H48_right_edge_count": row.get("H48_right_edge_count", 0),
                        "nonfinite_raw_case_count": row.get(
                            "nonfinite_raw_case_count", 0
                        ),
                        "nonfinite_processed_case_count": row.get(
                            "nonfinite_processed_case_count", 0
                        ),
                        "stopped_by_prior_gate": row.get(
                            "stopped_by_prior_gate", False
                        ),
                    }
                    for model, row in stage_rows.items()
                }
                for stage, stage_rows in collections.items()
            },
            "validated_direct_internal_operator_count": 0,
            "new_physical_path_count": 0,
            "new_head_channel_or_neuron_count": 0,
        },
        "hard_limits": [
            "finite_response_contracts_cover_only_the_registered_task_interfaces",
            "two_independent_rule_reviewer_agreement_was_not_recorded",
            "history_is_fixed_empty_and_not_yet_an_independent_factor",
            "grammar_be_form_and_sentence_completion_registries_share_plain_be_form_aliases",
            "functional_response_separation_is_a_behavioral_property_not_an_internal_readout",
            "task_coordinate_covariance_is_test_defined_and_not_a_discovered_operator",
            "cycle_consistency_reuses_the_same_state_rows_and_is_not_independent",
            "greedy_generation_observes_one_path_and_not_semantic_set_probability",
            "H48_is_an_observation_budget_and_missing_stop_is_right_censored",
            "small_models_may_implement_coarser_or_model_specific_response_structures",
            "no_activation_head_channel_or_neuron_state_was_collected",
        ],
        "authorization": {
            "show_exclusive_response_ledger": True,
            "show_functional_response_partition_as_observational": True,
            "show_task_coordinate_covariance_as_internal_operator": False,
            "show_crossmodel_response_partition_as_physical_invariant": False,
            "show_specific_physical_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_physical_mapping_next": physical_authorized,
            "run_history_conditioned_replication_next": (
                history_replication_authorized
            ),
            "run_causal_intervention_next": False,
            "run_neuron_scan_next": False,
        },
        "next_stage": {
            "phase_id": "Phase409",
            "objective": (
                "freeze_and_execute_independent_history_content_placement_conflict_"
                "and_override_replication_for_behaviorally_heldout_partitions"
                if history_replication_authorized
                else "diagnose_partition_failure_by_family_interface_surface_and_lexical_"
                "factor_then_freeze_a_new_dynamic_history_conditioned_response_object"
            ),
            "automatic_model_execution_authorized": False,
            "reason": (
                "behavioral_candidates_still_require_the_pre_registered_independent_"
                "history_controls_before_any_physical_protocol"
                if history_replication_authorized
                else "Phase408_downstream_gate_is_closed_and_any_new_response_object_"
                "requires_a_new_denominator_and_execution_qualification"
            ),
        },
        "single_global_progress_percentage_valid": False,
    }
    path = OUT / "phase408_partition_interface_stage_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
