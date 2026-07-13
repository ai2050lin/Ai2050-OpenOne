#!/usr/bin/env python3
"""Publish Phase399 aggregate dynamic-event evidence to both atlases."""

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
P399 = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
JSON_SOURCES = {
    "phase399_protocol.json": P399 / "phase399_protocol.json",
    "phase399_behavior_freeze_summary.json": P399 / "phase399_behavior_freeze_summary.json",
    "phase399_dynamic_trace_protocol.json": P399 / "phase399_dynamic_trace_protocol.json",
    "phase399_instrument_audit.json": P399 / "phase399_instrument_audit.json",
    "phase399_dynamic_candidate_protocol.json": P399 / "phase399_dynamic_candidate_protocol.json",
    "phase399_dynamic_candidate_freeze.json": P399 / "phase399_dynamic_candidate_freeze.json",
    "phase399_dynamic_discovery_analysis.json": P399 / "phase399_dynamic_discovery_analysis.json",
    "phase399_dynamic_calibration_validation.json": P399 / "phase399_dynamic_calibration_validation.json",
    "phase399_dynamic_physical_validation.json": P399 / "phase399_dynamic_physical_validation.json",
}


def passing_cells(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in payload["cells"] if row["dynamic_chain_validation_pass"]]


def required_event_pass_count(payload: dict[str, Any]) -> int:
    return sum(
        all(
            event["validation_pass"]
            for event in cell["event_classes"].values()
            if event["required_for_chain"]
        )
        for cell in payload["cells"]
    )


def stage_summary(updated_at: str, payloads: dict[str, Any]) -> dict[str, Any]:
    behavior = payloads["phase399_behavior_freeze_summary.json"]
    discovery = payloads["phase399_dynamic_discovery_analysis.json"]
    calibration = payloads["phase399_dynamic_calibration_validation.json"]
    physical = payloads["phase399_dynamic_physical_validation.json"]
    instrument = payloads["phase399_instrument_audit.json"]
    physical_pass = passing_cells(physical)
    return {
        "schema_version": "73.10.0",
        "phase_id": "Phase399-MultiPositionDynamicBindingStage",
        "created_at": updated_at,
        "objective": "trace_role_partitioned_multi_position_dynamic_events_and_gate_joint_causal_mediation_without_head_channel_or_neuron_search",
        "assessment": {
            "fresh_four_surface_behavior_denominator": True,
            "field_extraction_behavior_gate": False,
            "three_surface_trace_denominator": True,
            "parent_component_and_attention_route_conservation": True,
            "required_dynamic_events_three_split_replication": True,
            "ordered_dynamic_chain_model_specific_replication": True,
            "ordered_dynamic_chain_crossmodel_replication": False,
            "joint_causal_intervention_authorized": False,
            "natural_necessity_established": False,
            "complete_binding_algorithm_established": False,
            "complete_language_path_established": False,
            "single_neuron_mechanism_established": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "behavior_candidate_case_count": behavior["denominator"]["candidate_case_count"],
            "behavior_candidate_parallel_group_count": behavior["denominator"]["candidate_parallel_group_count"],
            "behavior_qualified_parallel_group_count": behavior["denominator"]["qualified_parallel_group_count"],
            "eligible_surface_count": behavior["denominator"]["eligible_surface_count"],
            "registered_surface_count": 4,
            "instrument_case_count": behavior["denominator"]["instrument_case_count"],
            "selected_trace_case_count": behavior["denominator"]["selected_case_count"],
            "discovery_trace_case_count": 1440,
            "calibration_trace_case_count": 720,
            "physical_trace_case_count": 720,
            "discovery_group_model_cell_count": 90,
            "calibration_group_model_cell_count": 45,
            "physical_group_model_cell_count": 45,
            "event_layer_search_count": discovery["search_denominator"]["event_layer_candidate_count"],
            "candidate_model_surface_cell_count": 9,
            "eligible_crossmodel_surface_count": 3,
        },
        "results": {
            "instrument_quality_cell_count": instrument["results"]["quality_group_model_cell_count"],
            "instrument_quality_cell_denominator": instrument["denominator"]["group_model_cell_count"],
            "required_event_discovery_cell_count": sum(
                cell["required_class_gate_pass"] for cell in discovery["cells"]
            ),
            "required_event_calibration_cell_count": required_event_pass_count(calibration),
            "required_event_physical_cell_count": required_event_pass_count(physical),
            "ordered_chain_discovery_cell_count": discovery["results"]["dynamic_chain_discovery_cell_count"],
            "ordered_chain_calibration_cell_count": calibration["results"]["dynamic_chain_validation_cell_count"],
            "ordered_chain_physical_cell_count": physical["results"]["dynamic_chain_validation_cell_count"],
            "ordered_chain_crossmodel_surface_count": physical["results"]["crossmodel_surface_count"],
            "model_specific_chain_model": physical_pass[0]["model"] if physical_pass else None,
            "model_specific_chain_surface": physical_pass[0]["surface"] if physical_pass else None,
            "model_specific_chain_layers": [
                physical_pass[0]["event_classes"][name]["layer_index"]
                for name in ("source_to_query_route", "query_integration", "terminal_integration")
            ] if physical_pass else [],
            "joint_causal_intervention_count": 0,
            "abstract_binding_algorithm_count": 0,
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        },
        "hard_limits": [
            "field_extraction_had_only_one_of_28_complete_three_model_groups_and_was_excluded_before_trace_selection",
            "nine_model_surface_cells_are_not_nine_independent_language_mechanisms",
            "all_nine_cells_replicated_required_event_classes_but_only_one_preserved_the_frozen_peak_order",
            "the_only_three_split_ordered_chain_was_DS7B_role_filling_and_is_model_specific",
            "an_event_peak_order_is_a_search_coordinate_not_a_verified_execution_formula",
            "the_role_partitioned_attention_write_is_aggregate_and_contains_no_head_or_neuron_identity",
            "no_joint_damage_layered_restore_or_natural_necessity_test_was_authorized",
            "only_three_behavior_eligible_surfaces_and_three_small_models_were_traced",
        ],
        "authorization": {
            "show_model_specific_dynamic_event_chain": True,
            "show_chain_as_aggregate_observation": True,
            "show_chain_as_crossmodel_rule": False,
            "show_chain_as_causal_binding_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_joint_causal_intervention": False,
            "run_single_neuron_scan": False,
            "claim_language_encoding_closure": False,
        },
        "next_stage": {
            "objective": "freeze_a_new_partial_order_event_graph_protocol_that_tests_full_trajectory_onset_merge_and_persistence_without_reusing_the_exhausted_Phase399_holdout",
            "automatic_continuation_authorized": False,
            "reason": "Phase399 exhausted its discovery, calibration, and physical holdout and closed the causal gate. Reusing those data to redesign the peak-order rule would be post-hoc overfitting.",
        },
        "single_global_progress_percentage_valid": False,
    }


def graph(stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {"node_id": "p399_fresh_behavior_denominator", "node_type": "fresh_four_surface_factorial_denominator", "phase_id": "Phase399", "case_count": 5376, "eligible_surface_count": 3, "registered_surface_count": 4, "causal": False, "language_path": False},
        {"node_id": "p399_role_partitioned_event_ledger", "node_type": "multi_position_role_partitioned_dynamic_event_ledger", "phase_id": "Phase399", "trace_case_count": 2880, "event_layer_search_count": 8112, "causal": False, "language_path": False},
        {"node_id": "p399_model_specific_role_chain", "node_type": "model_specific_ordered_dynamic_event_chain", "phase_id": "Phase399", "model": "deepseek7b", "surface": "role_filling", "layers": [10, 10, 20], "discovery_pass": True, "calibration_pass": True, "physical_pass": True, "causal": False, "language_path": False},
        {"node_id": "p399_crossmodel_chain_negative", "node_type": "crossmodel_ordered_dynamic_chain_negative", "phase_id": "Phase399", "passing_surface_count": 0, "surface_denominator": 3, "causal": False, "language_path": False},
        {"node_id": "p399_causal_gate_closed", "node_type": "joint_causal_mediation_gate_closed", "phase_id": "Phase399", "joint_intervention_count": 0, "head_channel_neuron_scan_authorized": False, "causal": False, "language_path": False},
    ]
    edges = [
        {"edge_id": "p398_to_p399_dynamic_ledger", "source_node_id": "p398_dynamic_path_required", "target_node_id": "p399_role_partitioned_event_ledger", "edge_type": "implements_multi_position_aggregate_event_trace", "phase_id": "Phase399", "causal_path": False},
        {"edge_id": "p399_behavior_to_ledger", "source_node_id": "p399_fresh_behavior_denominator", "target_node_id": "p399_role_partitioned_event_ledger", "edge_type": "qualifies_frozen_trace_denominator", "phase_id": "Phase399", "causal_path": False},
        {"edge_id": "p399_ledger_to_model_chain", "source_node_id": "p399_role_partitioned_event_ledger", "target_node_id": "p399_model_specific_role_chain", "edge_type": "three_split_observational_selection", "phase_id": "Phase399", "causal_path": False},
        {"edge_id": "p399_model_to_crossmodel_negative", "source_node_id": "p399_model_specific_role_chain", "target_node_id": "p399_crossmodel_chain_negative", "edge_type": "fails_crossmodel_equivalence_gate", "phase_id": "Phase399", "causal_path": False},
        {"edge_id": "p399_crossmodel_to_gate", "source_node_id": "p399_crossmodel_chain_negative", "target_node_id": "p399_causal_gate_closed", "edge_type": "closes_joint_damage_and_fine_resolution_search", "phase_id": "Phase399", "causal_path": False},
    ]
    return nodes, edges


def aggregate_event_nodes(physical: dict[str, Any], updated_at: str) -> list[dict[str, Any]]:
    cell = next(
        row for row in physical["cells"]
        if row["model"] == "deepseek7b" and row["surface"] == "role_filling"
    )
    definitions = [
        ("source_to_query_route", "aggregate_attention_source_route_event", "query_end", 31),
        ("query_integration", "aggregate_query_integration_event", "query_end", 32),
        ("terminal_integration", "aggregate_terminal_route_event", "first_answer", 33),
    ]
    nodes = []
    for event_name, unit_kind, token_position, unit_index in definitions:
        event = cell["event_classes"][event_name]
        metrics = event["metrics"]
        nodes.append({
            "schema_version": "aggregate_dynamic_event_anchor.v1",
            "node_id": f"language_action:deepseek7b:L{event['layer_index']}:phase399:role_filling:{event_name}",
            "node_type": "aggregate_dynamic_route_event",
            "family_id": "language_action",
            "family_name": "语言行为模式族",
            "relation": "role_filling",
            "model": "deepseek7b",
            "layer": event["layer_index"],
            "component": event["event_id"],
            "unit_kind": unit_kind,
            "unit_index": unit_index,
            "token_position": token_position,
            "candidate_score": metrics["median_roq_min_axis_normalized_norm"],
            "case_count": 320,
            "natural_observed": True,
            "group_intervention_supported": False,
            "expanded_confirmation_pass": True,
            "causal_scope": "three_split_model_specific_aggregate_dynamic_event_no_causal_intervention",
            "evidence_level": "L3-three-split-model-specific-observation",
            "evidence_status": "replicated_model_specific_dynamic_event_causal_gate_closed",
            "evidence_boundary": "This aggregate role-partitioned event replicated in DS7B role filling over discovery, calibration, and physical holdout. The complete ordered chain was 1/9 model-surface cells and 0/3 crossmodel surfaces; it is not a head, neuron, or causal binding path.",
            "display_priority": 18 + metrics["median_roq_min_axis_normalized_norm"],
            "phase399_tested": True,
            "phase399_event_class": event_name,
            "phase399_event_id": event["event_id"],
            "phase399_physical_observational_pass": event["validation_pass"],
            "phase399_roq_norm": metrics["median_roq_min_axis_normalized_norm"],
            "phase399_roq_cross_axis_cosine": metrics["median_roq_cross_axis_cosine"],
            "phase399_roq_to_competitor_ratio": metrics["median_roq_to_competing_interaction"],
            "phase399_ordered_chain_pass": cell["dynamic_chain_validation_pass"],
            "phase399_crossmodel_chain_pass": False,
            "phase399_causal_gate_open": False,
            "is_real_unit": False,
            "single_neuron_claim": False,
            "generated_at": updated_at,
            "source_artifacts": ["phase399_dynamic_physical_validation.json", "phase399_dynamic_binding_stage_summary.json"],
        })
    return nodes


def update_checksums(root: Path) -> None:
    path = root / "checksums.json"
    if path.is_file():
        write_json(path, {"schema_version": "artifact_checksums.v1", "files": [
            {"path": str(item.relative_to(root)), "sha256": sha256(item)}
            for item in sorted(root.rglob("*")) if item.is_file() and item != path
        ]})


def update_progress(root: Path, updated_at: str) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    progress["last_phase"] = "Phase399-MultiPositionDynamicBindingStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["multi_position_dynamic_binding_stage"] = {
        "behavior_candidate_cases": {"numerator": 5376, "denominator": 5376},
        "qualified_parallel_groups": {"numerator": 82, "denominator": 112},
        "eligible_surfaces": {"numerator": 3, "denominator": 4},
        "selected_trace_cases": {"numerator": 2880, "denominator": 2880},
        "quality_trace_group_model_cells": {"numerator": 180, "denominator": 180},
        "required_event_discovery_cells": {"numerator": 9, "denominator": 9},
        "required_event_calibration_cells": {"numerator": 9, "denominator": 9},
        "required_event_physical_cells": {"numerator": 9, "denominator": 9},
        "ordered_chain_discovery_cells": {"numerator": 1, "denominator": 9},
        "ordered_chain_calibration_cells": {"numerator": 1, "denominator": 9},
        "ordered_chain_physical_cells": {"numerator": 1, "denominator": 9},
        "crossmodel_ordered_chain_surfaces": {"numerator": 0, "denominator": 3},
        "joint_causal_intervention_authorized": {"numerator": 0, "denominator": 1},
        "complete_language_paths": {"numerator": 0, "denominator": 72},
        "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase399_decision"] = "retain_DS7B_role_filling_aggregate_dynamic_event_chain_close_crossmodel_causal_and_fine_resolution_gates"
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase399 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    evidence_nodes, evidence_edges = graph(stage)
    payloads["phase399_dynamic_binding_stage_summary.json"] = stage
    published = [*payloads, "phase399_evidence_nodes.jsonl", "phase399_evidence_edges.jsonl"]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase399_evidence_nodes.jsonl", evidence_nodes)
        write_jsonl(root / "phase399_evidence_edges.jsonl", evidence_edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase399-MultiPositionDynamicBindingStage"
        manifest["phase399"] = {
            "status": "required_events_replicated_one_model_specific_ordered_chain_crossmodel_and_causal_gates_closed",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase399-MultiPositionDynamicBindingStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        update_checksums(root)
        public_manifest(root, updated_at)

    dynamic_nodes = aggregate_event_nodes(payloads["phase399_dynamic_physical_validation.json"], updated_at)
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase399_dynamic_binding_stage_summary.json", stage)
        write_jsonl(root / "phase399_evidence_nodes.jsonl", evidence_nodes)
        write_jsonl(root / "phase399_evidence_edges.jsonl", evidence_edges)
        partition_path = root / "partitions/language_action/deepseek7b.json"
        partition = read_json(partition_path)
        retained = [node for node in partition.get("nodes", []) if not node.get("phase399_tested")]
        partition["nodes"] = [*retained, *dynamic_nodes]
        partition["mapping_status"] = "phase399_model_specific_dynamic_event_chain_replicated_crossmodel_and_causal_gates_closed"
        partition["generated_at"] = updated_at
        partition.setdefault("metrics", {}).update({
            "phase399_aggregate_dynamic_event_count": 3,
            "phase399_ordered_chain_physical_pass_count": 1,
            "phase399_crossmodel_chain_count": 0,
            "phase399_joint_causal_intervention_count": 0,
            "phase399_single_neuron_causal_count": 0,
        })
        partition["evidence_boundary"] = "Phase399 adds three aggregate DS7B role-filling dynamic event anchors at layers 10, 10, and 20. They replicated over three splits but are model-specific observations, not heads, neurons, causal mediators, or a complete binding path."
        write_json(partition_path, partition)

        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 399
        manifest["generated_at"] = updated_at
        manifest["phase399_audit"] = {
            "status": "one_model_specific_ordered_dynamic_chain_replicated_crossmodel_and_causal_gates_closed",
            "behavior_candidate_case_count": 5376,
            "qualified_parallel_group_count": 82,
            "eligible_surface_count": 3,
            "selected_trace_case_count": 2880,
            "required_event_physical_cell_count": 9,
            "ordered_chain_physical_cell_count": 1,
            "crossmodel_chain_surface_count": 0,
            "new_aggregate_dynamic_event_count": 3,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase399_dynamic_binding_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update({
            "phase399_behavior_candidate_case_count": 5376,
            "phase399_qualified_parallel_group_count": 82,
            "phase399_eligible_surface_count": 3,
            "phase399_registered_surface_count": 4,
            "phase399_selected_trace_case_count": 2880,
            "phase399_quality_trace_group_model_cell_count": 180,
            "phase399_required_event_physical_cell_count": 9,
            "phase399_ordered_chain_discovery_cell_count": 1,
            "phase399_ordered_chain_calibration_cell_count": 1,
            "phase399_ordered_chain_physical_cell_count": 1,
            "phase399_crossmodel_chain_surface_count": 0,
            "phase399_aggregate_dynamic_event_count": 3,
            "phase399_joint_causal_intervention_count": 0,
            "phase399_single_neuron_causal_count": 0,
        })
        manifest.setdefault("files", {})["latest_evidence_summary"] = "phase399_dynamic_binding_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase399-MultiPositionDynamicBindingStage",
            "statement": "Role-partitioned source-to-query, query-integration, and terminal events replicated in all 9 model-surface cells, but the frozen peak-order chain held only for DS7B role filling across three splits. Crossmodel chains are 0/3; no causal intervention or fine-resolution scan was authorized.",
            "aggregate_dynamic_events_available": True,
            "model_specific_ordered_chain_available": True,
            "crossmodel_dynamic_chain_available": False,
            "joint_causal_mediation_available": False,
            "upstream_language_path_available": False,
            "single_unit_causal_closure": False,
            "candidate_depth_specificity_available": False,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        update_checksums(root)
        public_manifest(root, updated_at)

    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
