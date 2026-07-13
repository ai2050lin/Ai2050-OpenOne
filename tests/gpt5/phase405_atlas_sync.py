#!/usr/bin/env python3
"""Publish Phase403-405 predictive-state evidence to both atlas mirrors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase380_atlas_sync import (
    CLIENT,
    NEURON_CLIENT,
    NEURON_TARGET,
    TARGET,
    public_manifest,
    read_json,
    sha256,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
P403 = ROOT / "tests/gpt5/result/phase403_predictive_state"
P404 = ROOT / "tests/gpt5/result/phase404_direct_predictive_state"
P405 = ROOT / "tests/gpt5/result/phase405_natural_future_state"
JSON_SOURCES = {
    "phase403_predictive_state_protocol.json": P403
    / "phase403_predictive_state_protocol.json",
    "phase403_discovery_analysis.json": P403 / "phase403_discovery_analysis.json",
    "phase403_failure_diagnostic.json": P403 / "phase403_failure_diagnostic.json",
    "phase404_direct_state_protocol.json": P404 / "phase404_direct_state_protocol.json",
    "phase404_discovery_analysis.json": P404 / "phase404_discovery_analysis.json",
    "phase404_failure_diagnostic.json": P404 / "phase404_failure_diagnostic.json",
    "phase405_natural_future_protocol.json": P405
    / "phase405_natural_future_protocol.json",
    "phase405_discovery_analysis.json": P405 / "phase405_discovery_analysis.json",
    "phase405_failure_diagnostic.json": P405 / "phase405_failure_diagnostic.json",
}
JSONL_SOURCES = {
    "phase404_failure_axes.jsonl": P404 / "analysis/phase404_failure_axes.jsonl",
    "phase405_failure_axes.jsonl": P405 / "analysis/phase405_failure_axes.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def update_checksums(root: Path) -> None:
    path = root / "checksums.json"
    if not path.is_file():
        return
    write_json(
        path,
        {
            "schema_version": "artifact_checksums.v1",
            "files": [
                {"path": str(item.relative_to(root)), "sha256": sha256(item)}
                for item in sorted(root.rglob("*"))
                if item.is_file() and item != path
            ],
        },
    )


def stage_summary(updated_at: str, payloads: dict[str, Any]) -> dict[str, Any]:
    p403 = payloads["phase403_discovery_analysis.json"]
    p403_diag = payloads["phase403_failure_diagnostic.json"]
    p404 = payloads["phase404_discovery_analysis.json"]
    p404_diag = payloads["phase404_failure_diagnostic.json"]
    p405 = payloads["phase405_discovery_analysis.json"]
    p405_diag = payloads["phase405_failure_diagnostic.json"]
    return {
        "schema_version": "79.9.0",
        "phase_id": "Phase405-PredictiveStateParadigmStage",
        "created_at": updated_at,
        "objective": (
            "test_whether_truth_enumerated_language_states_form_stable_future_"
            "response_equivalence_classes_before_any_physical_mapping"
        ),
        "assessment": {
            "finite_world_state_tables_frozen": True,
            "integer_group_gates_frozen": True,
            "imperative_update_execution_separated": True,
            "direct_endpoint_finite_response_measured": True,
            "natural_unfinished_future_response_measured": True,
            "stable_crossmodel_predictive_state_observed": False,
            "causal_state_equivalence_established": False,
            "semantic_transition_is_internal_operator": False,
            "physical_mapping_opened": False,
            "causal_or_neuron_work_opened": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "phase403_formal_discovery_case_count": p403["case_count"],
            "phase403_semantic_correct_case_count": p403[
                "semantic_correct_count"
            ],
            "phase404_direct_endpoint_case_count": p404["case_count"],
            "phase404_finite_candidate_correct_count": p404[
                "finite_candidate_correct_count"
            ],
            "phase404_global_top_target_count": p404[
                "global_top_is_target_count"
            ],
            "phase405_natural_future_case_count": p405["case_count"],
            "phase405_finite_candidate_correct_count": p405[
                "finite_candidate_correct_count"
            ],
            "phase405_natural_top_target_count": p405[
                "global_top_is_target_count"
            ],
            "phase405_natural_top_in_candidate_set_count": p405[
                "global_top_in_candidate_set_count"
            ],
            "calibration_case_count_consumed": 0,
            "behavioral_holdout_case_count_consumed": 0,
            "physical_holdout_case_count_consumed": 0,
        },
        "results": {
            "phase403_crossmodel_candidate_family_count": len(
                p403["crossmodel_candidate_families"]
            ),
            "phase403_crossmodel_strict_base_group_count": sum(
                p403_diag["crossmodel_base_group_count_by_family"].values()
            ),
            "phase404_crossmodel_candidate_family_count": len(
                p404["crossmodel_candidate_families"]
            ),
            "phase404_candidate_correct_but_global_top_wrong_count": p404_diag[
                "finite_candidate_correct_but_global_top_wrong_count"
            ],
            "phase405_crossmodel_candidate_family_count": len(
                p405["crossmodel_candidate_families"]
            ),
            "phase405_candidate_correct_but_natural_top_wrong_count": p405_diag[
                "candidate_correct_but_natural_top_wrong_count"
            ],
            "phase405_nonfinite_global_logit_case_count": p405_diag[
                "nonfinite_global_logit_case_count"
            ],
            "phase405_model_family_natural_group_pass_counts": p405_diag[
                "model_family_group_pass_count"
            ],
            "validated_predictive_state_family_count": 0,
            "validated_transition_operator_count": 0,
            "new_physical_path_count": 0,
            "new_neuron_node_count": 0,
        },
        "hard_limits": [
            "the_finite_future_panels_are_not_exhaustive_over_all_legal_future_branches",
            "phase403_conflates_state_representation_with_imperative_update_execution",
            "phase404_finite_candidate_winners_often_differ_from_full_vocabulary_natural_top_tokens",
            "phase405_raw_completion_is_not_a_semantically_neutral_measurement_interface",
            "all_nine_phase405_model_family_cells_have_zero_strict_natural_group_passes",
            "small_model_surface_sensitivity_can_hide_or_destroy_functional_equivalence",
            "truth_enumerated_semantic_edges_are_generator_rules_not_observed_internal_operators",
            "no_behaviorally_valid_crossmodel_state_exists_for_physical_or_neuron_promotion",
        ],
        "authorization": {
            "show_protocol_and_response_ledger": True,
            "show_predictive_state_as_validated": False,
            "show_semantic_transition_as_internal_operator": False,
            "show_physical_state_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_phase403_or_phase404_calibration": False,
            "run_phase405_calibration": False,
            "run_behavioral_or_physical_holdout": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "next_stage": {
            "objective": (
                "represent_each_future_branch_as_a_condition_response_pair_and_"
                "test_exact_context_conditioned_transition_tables_without_"
                "assuming_interface_invariant_single_token_answers"
            ),
            "automatic_model_execution_authorized": False,
            "reason": (
                "the_next_protocol_changes_the_state_definition_again_and_must_"
                "first_freeze_branch_coverage_and_non_single_token_outputs"
            ),
            "required_changes": [
                "treat_query_and_output_interface_as_part_of_the_future_condition",
                "record_short_natural_continuation_sequences_instead_of_only_one_token",
                "use_exact_response_tables_before_any_similarity_or_clustering",
                "require_unseen_branch_composition_before_calling_a_state_sufficient",
                "open_physical_mapping_only_after_crossmodel_calibration_and_behavioral_holdout",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }


def evidence_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p403_imperative_predictive_state_negative",
            "node_type": "finite_state_plus_imperative_operation_response_audit",
            "phase_id": "Phase403",
            "case_count": 5184,
            "semantic_correct_count": 3411,
            "crossmodel_candidate_family_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p404_direct_endpoint_response_split",
            "node_type": "direct_endpoint_finite_candidate_vs_full_vocabulary_split",
            "phase_id": "Phase404",
            "case_count": 2880,
            "finite_candidate_correct_count": 2597,
            "global_top_target_count": 1529,
            "candidate_only_correct_count": 1068,
            "crossmodel_candidate_family_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p405_natural_future_response_negative",
            "node_type": "natural_unfinished_future_branch_response_audit",
            "phase_id": "Phase405",
            "case_count": 2880,
            "finite_candidate_correct_count": 2076,
            "natural_top_target_count": 1266,
            "candidate_only_correct_count": 810,
            "crossmodel_candidate_family_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p405_predictive_physical_neuron_gates_closed",
            "node_type": "predictive_state_validation_physical_causal_neuron_boundary",
            "phase_id": "Phase405",
            "calibration_cases_consumed": 0,
            "behavioral_holdout_cases_consumed": 0,
            "physical_holdout_cases_consumed": 0,
            "neuron_nodes_promoted": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p402_to_p403_function_first",
            "source_node_id": "p402_validation_causal_neuron_gates_closed",
            "target_node_id": "p403_imperative_predictive_state_negative",
            "edge_type": "replaces_static_parent_transfer_with_finite_future_response_protocol",
            "phase_id": "Phase403",
            "causal_path": False,
        },
        {
            "edge_id": "p403_to_p404_remove_update_execution",
            "source_node_id": "p403_imperative_predictive_state_negative",
            "target_node_id": "p404_direct_endpoint_response_split",
            "edge_type": "removes_imperative_update_execution_confound",
            "phase_id": "Phase404",
            "causal_path": False,
        },
        {
            "edge_id": "p404_to_p405_remove_explicit_choice",
            "source_node_id": "p404_direct_endpoint_response_split",
            "target_node_id": "p405_natural_future_response_negative",
            "edge_type": "replaces_explicit_choice_with_unfinished_natural_future_branch",
            "phase_id": "Phase405",
            "causal_path": False,
        },
        {
            "edge_id": "p405_negative_closes_physical",
            "source_node_id": "p405_natural_future_response_negative",
            "target_node_id": "p405_predictive_physical_neuron_gates_closed",
            "edge_type": "zero_crossmodel_candidates_stop_all_downstream_mapping",
            "phase_id": "Phase405",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_progress(root: Path, updated_at: str) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    progress["last_phase"] = "Phase405-PredictiveStateParadigmStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["predictive_state_stage"] = {
        "phase403_semantic_correct_cases": {"numerator": 3411, "denominator": 5184},
        "phase403_crossmodel_candidate_families": {"numerator": 0, "denominator": 3},
        "phase404_finite_candidate_correct_cases": {"numerator": 2597, "denominator": 2880},
        "phase404_natural_top_target_cases": {"numerator": 1529, "denominator": 2880},
        "phase404_crossmodel_candidate_families": {"numerator": 0, "denominator": 3},
        "phase405_finite_candidate_correct_cases": {"numerator": 2076, "denominator": 2880},
        "phase405_natural_top_target_cases": {"numerator": 1266, "denominator": 2880},
        "phase405_model_family_natural_group_pass_cells": {"numerator": 0, "denominator": 9},
        "phase405_crossmodel_candidate_families": {"numerator": 0, "denominator": 3},
        "calibration_cases_consumed": {"numerator": 0, "denominator": 1},
        "behavioral_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase405_decision"] = (
        "retain_response_ledgers_reject_current_predictive_equivalence_"
        "and_keep_physical_causal_neuron_gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [
        str(path)
        for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase403-405 artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    nodes, edges = evidence_graph()
    payloads["phase405_predictive_state_stage_summary.json"] = stage
    published = [
        *payloads,
        *row_payloads,
        "phase405_evidence_nodes.jsonl",
        "phase405_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        write_jsonl(root / "phase405_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase405_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase405-PredictiveStateParadigmStage"
        manifest["phase405"] = {
            "status": "three_predictive_state_protocols_complete_zero_crossmodel_candidates_all_physical_gates_closed",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase405-PredictiveStateParadigmStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        update_checksums(root)
        public_manifest(root, updated_at)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase405_predictive_state_stage_summary.json", stage)
        write_jsonl(root / "phase405_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase405_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 405
        manifest["generated_at"] = updated_at
        manifest["phase405_audit"] = {
            "status": "predictive_state_candidates_zero_no_physical_or_neuron_nodes_promoted",
            "phase403_discovery_case_count": 5184,
            "phase404_discovery_case_count": 2880,
            "phase405_discovery_case_count": 2880,
            "crossmodel_predictive_state_family_count": 0,
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase405_predictive_state_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase403_predictive_state_case_count": 5184,
                "phase403_predictive_state_correct_count": 3411,
                "phase403_crossmodel_state_family_count": 0,
                "phase404_direct_state_case_count": 2880,
                "phase404_finite_candidate_correct_count": 2597,
                "phase404_global_top_target_count": 1529,
                "phase404_crossmodel_state_family_count": 0,
                "phase405_natural_future_case_count": 2880,
                "phase405_finite_candidate_correct_count": 2076,
                "phase405_natural_top_target_count": 1266,
                "phase405_model_family_natural_group_pass_count": 0,
                "phase405_crossmodel_state_family_count": 0,
                "phase405_physical_holdout_case_count": 0,
                "phase405_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase405_predictive_state_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase405-PredictiveStateParadigmStage",
            "statement": (
                "Phase403-405 tested imperative transitions, direct endpoints, "
                "and unfinished natural future branches over three frozen "
                "language families. None produced a calibrated crossmodel "
                "predictive-state family. No physical path, component, head, "
                "channel, or neuron is promoted from these response ledgers."
            ),
            "predictive_response_ledger_available": True,
            "validated_predictive_state_available": False,
            "validated_internal_operator_available": False,
            "physical_predictive_state_path_available": False,
            "single_unit_causal_closure": False,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        update_checksums(root)
        public_manifest(root, updated_at)

    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
