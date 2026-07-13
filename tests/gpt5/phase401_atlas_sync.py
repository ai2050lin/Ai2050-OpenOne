#!/usr/bin/env python3
"""Publish Phase401 execution, local-edge, and stopping boundaries to the atlas."""

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
P401 = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
JSON_SOURCES = {
    "phase401_local_edge_protocol.json": P401 / "phase401_local_edge_protocol.json",
    "phase401_behavior_protocol.json": P401 / "phase401_behavior_protocol.json",
    "phase401_batch_sensitivity_audit.json": P401
    / "phase401_batch_sensitivity_audit.json",
    "phase401_behavior_freeze_summary.json": P401
    / "phase401_behavior_freeze_summary.json",
    "phase401_trace_protocol.json": P401 / "phase401_trace_protocol.json",
    "phase401_instrument_audit.json": P401 / "phase401_instrument_audit.json",
    "phase401_local_edge_execution_freeze.json": P401
    / "phase401_local_edge_execution_freeze.json",
    "phase401_local_edge_discovery_audit.json": P401
    / "phase401_local_edge_discovery_audit.json",
    "phase401_local_edge_stage_profile.json": P401
    / "phase401_local_edge_stage_profile.json",
}


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
    behavior = payloads["phase401_behavior_freeze_summary.json"]
    batch = payloads["phase401_batch_sensitivity_audit.json"]
    instrument = payloads["phase401_instrument_audit.json"]
    discovery = payloads["phase401_local_edge_discovery_audit.json"]
    stages = payloads["phase401_local_edge_stage_profile.json"]
    semantic_correct = sum(
        row["semantic_correct_count"] for row in behavior["model_results"].values()
    )
    instrument_cases = sum(
        row["case_count"] for row in instrument["models"].values()
    )
    instrument_pass = sum(
        row["quality_pass_case_count"] for row in instrument["models"].values()
    )
    layer_denominator = sum(
        item["layer_count"] * 2
        for item in discovery["source_completeness"].values()
    )
    strict_layers = sum(
        surface["strict_passing_layer_count"]
        for model in discovery["model_surface_summary"].values()
        for surface in model.values()
    )
    sensitivity_layers = sum(
        surface["sensitivity_passing_layer_count"]
        for model in discovery["model_surface_summary"].values()
        for surface in model.values()
    )
    return {
        "schema_version": "75.12.0",
        "phase_id": "Phase401-ExecutionSemanticLocalEdgeStage",
        "created_at": updated_at,
        "objective": (
            "match_execution_shape_separate_semantic_span_and_test_actual_"
            "source_KV_to_query_attention_responses_with_eight_controls"
        ),
        "assessment": {
            "fresh_four_surface_denominator": True,
            "formal_execution_batch_size_one": True,
            "semantic_format_suffix_and_stop_separated": True,
            "batch_shape_empirically_invariant": False,
            "same_shape_instrument_ledger_pass": instrument["joint_gate"]["pass"],
            "direct_local_response_observed": True,
            "function_specific_direct_local_edge_observed": False,
            "crossmodel_local_edge_candidate_observed": False,
            "calibration_opened": False,
            "physical_holdout_opened": False,
            "causal_or_neuron_search_authorized": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "behavior_candidate_case_count": behavior["denominator"][
                "candidate_case_count"
            ],
            "behavior_semantic_correct_case_count": semantic_correct,
            "behavior_parallel_group_count": behavior["denominator"][
                "candidate_parallel_group_count"
            ],
            "behavior_complete_three_model_group_count": behavior["denominator"][
                "qualified_parallel_group_count"
            ],
            "registered_surface_count": 4,
            "eligible_surface_count": len(behavior["eligible_surfaces"]),
            "selected_trace_case_count": behavior["denominator"][
                "selected_case_count"
            ],
            "batch_pilot_case_count": batch["case_count"],
            "batch_pilot_all_field_match_count": batch[
                "all_observed_fields_match_count"
            ],
            "instrument_case_count": instrument_cases,
            "instrument_quality_pass_case_count": instrument_pass,
            "discovery_model_case_count": sum(
                item["case_count"]
                for item in discovery["source_completeness"].values()
            ),
            "discovery_pair_row_count": stages["total_pair_row_count"],
            "discovery_group_stage_row_count": stages["group_stage_row_count"],
            "model_surface_layer_count": layer_denominator,
            "model_surface_cell_count": 6,
            "crossmodel_surface_count": 2,
            "calibration_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "batch_sensitive_case_count": batch["batch_sensitive_case_count"],
            "batch_semantic_correctness_difference_count": batch[
                "semantic_correctness_difference_count"
            ],
            "strict_local_edge_passing_layer_count": strict_layers,
            "sensitivity_local_edge_passing_layer_count": sensitivity_layers,
            "function_specific_direct_attention_model_surface_count": stages[
                "registered_direct_attention_local_physical_candidate_count"
            ],
            "strict_crossmodel_local_edge_surface_count": len(
                discovery["strict_crossmodel_candidates"]
            ),
            "protocol_same_target_semantic_contradiction_count": 1,
            "validated_crossmodel_local_edge_surface_count": 0,
            "joint_causal_intervention_count": 0,
            "new_head_channel_or_neuron_node_count": 0,
            "complete_language_path_count": 0,
        },
        "hard_limits": [
            "only_possession_relation_and_role_filling_met_the_frozen_three_split_behavior_gate",
            "seven_of_192_independent_pilot_cases_changed_at_least_one_observed_field_between_batch_size_one_and_eight",
            "batch_shape_is_part_of_the_measurement_contract_not_a_demonstrated_semantic_latent_state",
            "the_target_repetition_parser_was_amended_before_full_formal_execution_and_pre_amendment_outputs_were_quarantined",
            "the_same_target_control_has_no_defined_donor_minus_recipient_semantic_competition_despite_the_original_all_eight_semantic_control_rule",
            "the_strict_primary_audit_and_the_non_authorizing_not_applicable_sensitivity_audit_both_found_zero_passing_layers",
            "true_relation_replacement_often_restored_a_direct_attention_state_but_never_separated_from_all_eight_controls",
            "post_attention_MLP_and_layer_output_responses_are_propagation_profiles_not_additional_direct_edges",
            "no_calibration_or_physical_case_was_consumed_and_no_head_channel_or_neuron_was_selected",
        ],
        "authorization": {
            "show_exact_architecture_ledger": True,
            "show_batch_sensitive_execution_boundary": True,
            "show_direct_response_profile": True,
            "show_function_specific_local_edge": False,
            "show_language_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_calibration": False,
            "run_physical_holdout": False,
            "run_joint_causal_intervention": False,
            "run_single_neuron_scan": False,
        },
        "next_stage": {
            "objective": (
                "freeze_a_new_group_level_multi_parent_intervention_contract_that_"
                "tests_source_structure_and_query_state_jointly_without_reusing_"
                "Phase401_calibration_or_physical_groups"
            ),
            "automatic_continuation_authorized": False,
            "reason": (
                "Phase401_stopping_rule_closes_calibration_after_zero_strict_"
                "discovery_candidates;_a_new_contract_and_fresh_denominator_are_required"
            ),
            "required_changes": [
                "make_control_applicability_explicit_before_data_collection",
                "separate_direct_child_physical_gates_from_terminal_semantic_gates",
                "test_joint_multi_position_or_multi_parent_state_instead_of_another_single_source_KV_patch",
                "preserve_batch_size_one_as_the_formal_measurement_shape",
                "retain_group_first_independence_and_all_eight_unweighted_controls",
            ],
            "causal_or_neuron_work_authorized": False,
        },
        "single_global_progress_percentage_valid": False,
    }


def evidence_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p401_matched_execution_semantic_denominator",
            "node_type": "matched_execution_and_semantic_span_denominator",
            "phase_id": "Phase401",
            "behavior_case_count": 4608,
            "eligible_surface_count": 2,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p401_batch_shape_sensitivity",
            "node_type": "independent_batch_shape_sensitivity_observation",
            "phase_id": "Phase401",
            "different_case_count": 7,
            "case_denominator": 192,
            "semantic_difference_count": 1,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p401_exact_component_ledger",
            "node_type": "same_shape_exact_component_ledger",
            "phase_id": "Phase401",
            "passing_cases": 96,
            "case_denominator": 96,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p401_nonspecific_local_response",
            "node_type": "direct_source_KV_response_without_control_specificity",
            "phase_id": "Phase401",
            "pair_row_count": 239616,
            "passing_model_surface_cells": 0,
            "model_surface_denominator": 6,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p401_protocol_control_contradiction",
            "node_type": "same_target_control_semantic_gate_contradiction",
            "phase_id": "Phase401",
            "strict_result_changed_by_sensitivity": False,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p401_validation_causal_neuron_gates_closed",
            "node_type": "validation_physical_causal_and_neuron_gates_closed",
            "phase_id": "Phase401",
            "calibration_cases_consumed": 0,
            "physical_cases_consumed": 0,
            "neuron_nodes_promoted": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p400_to_p401_execution_contract",
            "source_node_id": "p400_batch_shape_replay_failure",
            "target_node_id": "p401_matched_execution_semantic_denominator",
            "edge_type": "repairs_execution_and_semantic_span_contract",
            "phase_id": "Phase401",
            "causal_path": False,
        },
        {
            "edge_id": "p401_denominator_to_batch_audit",
            "source_node_id": "p401_matched_execution_semantic_denominator",
            "target_node_id": "p401_batch_shape_sensitivity",
            "edge_type": "separates_independent_execution_sensitivity_pilot",
            "phase_id": "Phase401",
            "causal_path": False,
        },
        {
            "edge_id": "p401_denominator_to_ledger",
            "source_node_id": "p401_matched_execution_semantic_denominator",
            "target_node_id": "p401_exact_component_ledger",
            "edge_type": "qualifies_same_shape_internal_measurement",
            "phase_id": "Phase401",
            "causal_path": False,
        },
        {
            "edge_id": "p401_ledger_to_local_response",
            "source_node_id": "p401_exact_component_ledger",
            "target_node_id": "p401_nonspecific_local_response",
            "edge_type": "enables_actual_parent_recomputation",
            "phase_id": "Phase401",
            "causal_path": False,
        },
        {
            "edge_id": "p401_control_audit_to_boundary",
            "source_node_id": "p401_protocol_control_contradiction",
            "target_node_id": "p401_nonspecific_local_response",
            "edge_type": "strict_and_sensitivity_audits_both_reject_candidate",
            "phase_id": "Phase401",
            "causal_path": False,
        },
        {
            "edge_id": "p401_local_response_closes_gates",
            "source_node_id": "p401_nonspecific_local_response",
            "target_node_id": "p401_validation_causal_neuron_gates_closed",
            "edge_type": "fails_all_control_specificity_and_stops_validation",
            "phase_id": "Phase401",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_progress(root: Path, updated_at: str) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    progress["last_phase"] = "Phase401-ExecutionSemanticLocalEdgeStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["local_edge_stage"] = {
        "behavior_candidate_cases": {"numerator": 4608, "denominator": 4608},
        "semantic_correct_cases": {"numerator": 4557, "denominator": 4608},
        "complete_three_model_groups": {"numerator": 66, "denominator": 96},
        "eligible_surfaces": {"numerator": 2, "denominator": 4},
        "batch_shape_all_field_matches": {"numerator": 185, "denominator": 192},
        "instrument_quality_cases": {"numerator": 96, "denominator": 96},
        "discovery_model_cases": {"numerator": 768, "denominator": 768},
        "strict_local_edge_layers": {"numerator": 0, "denominator": 208},
        "direct_local_physical_model_surface_cells": {
            "numerator": 0,
            "denominator": 6,
        },
        "crossmodel_local_edge_surfaces": {"numerator": 0, "denominator": 2},
        "calibration_cases_consumed": {"numerator": 0, "denominator": 384},
        "physical_holdout_cases_consumed": {
            "numerator": 0,
            "denominator": 384,
        },
        "joint_causal_intervention_authorized": {
            "numerator": 0,
            "denominator": 1,
        },
        "complete_language_paths": {"numerator": 0, "denominator": 72},
        "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase401_decision"] = (
        "retain_exact_execution_ledger_and_nonspecific_response_profile_close_"
        "calibration_physical_causal_and_neuron_gates_require_new_multi_parent_contract"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase401 artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    payloads["phase401_execution_semantic_local_edge_stage_summary.json"] = stage
    nodes, edges = evidence_graph()
    published = [
        *payloads,
        "phase401_evidence_nodes.jsonl",
        "phase401_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase401_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase401_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase401-ExecutionSemanticLocalEdgeStage"
        manifest["phase401"] = {
            "status": (
                "matched_execution_and_semantic_span_complete_local_response_"
                "nonspecific_no_validation_physical_causal_or_neuron_authorization"
            ),
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase401-ExecutionSemanticLocalEdgeStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        update_checksums(root)
        public_manifest(root, updated_at)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase401_execution_semantic_local_edge_stage_summary.json", stage)
        write_jsonl(root / "phase401_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase401_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 401
        manifest["generated_at"] = updated_at
        manifest["phase401_audit"] = {
            "status": "direct_response_observed_but_zero_function_specific_local_edges_and_zero_neuron_nodes_promoted",
            "behavior_candidate_case_count": 4608,
            "behavior_semantic_correct_case_count": 4557,
            "batch_sensitive_pilot_case_count": 7,
            "batch_pilot_case_count": 192,
            "instrument_quality_pass_case_count": 96,
            "instrument_case_count": 96,
            "strict_local_edge_passing_layer_count": 0,
            "model_surface_layer_count": 208,
            "direct_local_physical_model_surface_count": 0,
            "model_surface_count": 6,
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase401_execution_semantic_local_edge_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase401_behavior_candidate_case_count": 4608,
                "phase401_behavior_semantic_correct_case_count": 4557,
                "phase401_batch_sensitive_pilot_case_count": 7,
                "phase401_batch_pilot_case_count": 192,
                "phase401_instrument_quality_pass_case_count": 96,
                "phase401_instrument_case_count": 96,
                "phase401_discovery_pair_row_count": 239616,
                "phase401_strict_local_edge_passing_layer_count": 0,
                "phase401_model_surface_layer_count": 208,
                "phase401_direct_local_physical_model_surface_count": 0,
                "phase401_model_surface_count": 6,
                "phase401_crossmodel_local_edge_surface_count": 0,
                "phase401_physical_holdout_case_count": 0,
                "phase401_joint_causal_intervention_count": 0,
                "phase401_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase401_execution_semantic_local_edge_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase401-ExecutionSemanticLocalEdgeStage",
            "statement": (
                "Phase401 matched formal behavior and internal tracing at batch size one, separated semantic answers from format and stop, and passed 96/96 same-shape component ledgers. Across 239,616 directed intervention-control rows, true relation replacement often restored an attention state but never separated from all eight controls; strict and non-authorizing sensitivity audits both found 0/208 passing model-surface layers and 0/6 function-specific local physical edges. Calibration and physical groups remain unused, and no head, channel, or neuron node was promoted."
            ),
            "aggregate_execution_ledger_available": True,
            "batch_sensitivity_observed": True,
            "direct_response_profile_available": True,
            "function_specific_local_edge_available": False,
            "validated_local_edge_rule_available": False,
            "joint_causal_mediation_available": False,
            "upstream_language_path_available": False,
            "single_unit_causal_closure": False,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        update_checksums(root)
        public_manifest(root, updated_at)

    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
