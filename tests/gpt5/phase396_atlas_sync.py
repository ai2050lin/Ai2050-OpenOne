#!/usr/bin/env python3
"""Publish Phase394-396 binding separation evidence to both atlas mirrors."""

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
P394 = ROOT / "tests/gpt5/result/phase394_binding_separation"
P395 = ROOT / "tests/gpt5/result/phase395_natural_binding"
P396 = ROOT / "tests/gpt5/result/phase396_field_binding_physical"
MODELS = ("qwen3", "glm4", "deepseek7b")

JSON_SOURCES = {
    "phase394_protocol.json": P394 / "phase394_protocol.json",
    "phase394_behavior_freeze_summary.json": P394 / "phase394_behavior_freeze_summary.json",
    "phase395_protocol.json": P395 / "phase395_protocol.json",
    "phase395_behavior_freeze_summary.json": P395 / "phase395_behavior_freeze_summary.json",
    "phase395_discovery_candidate_freeze.json": P395 / "phase395_discovery_candidate_freeze.json",
    "phase395_calibration_replication.json": P395 / "calibration_analysis/phase395_calibration_replication.json",
    "phase395_causal_calibration_protocol.json": P395 / "phase395_causal_calibration_protocol.json",
    "phase395_causal_instrument_audit.json": P395 / "phase395_causal_instrument_audit.json",
    "phase395_causal_calibration_analysis.json": P395 / "phase395_causal_calibration_analysis.json",
    "phase396_protocol.json": P396 / "phase396_protocol.json",
    "phase396_physical_analysis.json": P396 / "phase396_physical_analysis.json",
}


def physical_cell_by_model(physical: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {cell["model"]: cell for cell in physical["cells"]}


def stage_summary(updated_at: str, payloads: dict[str, Any]) -> dict[str, Any]:
    p394 = payloads["phase394_behavior_freeze_summary.json"]
    p395_behavior = payloads["phase395_behavior_freeze_summary.json"]
    p395 = payloads["phase395_causal_calibration_analysis.json"]
    p396 = payloads["phase396_physical_analysis.json"]
    physical_switches = sum(
        cell["scenario_summaries"]["donor_same_literal_candidate"]["answer_switch_count"]
        for cell in p396["cells"]
    )
    content_switches = sum(
        cell["scenario_summaries"]["donor_same_position_candidate"]["answer_switch_count"]
        for cell in p396["cells"]
    )
    return {
        "schema_version": "70.3.0",
        "phase_id": "Phase396-BindingSeparationStage",
        "created_at": updated_at,
        "objective": "separate_formal_interface_ability_literal_content_and_contextual_relation_state",
        "assessment": {
            "formal_pointer_interface_shared": False,
            "natural_same_token_multiset_behavior_available": True,
            "crossmodel_crosssurface_static_context_state_established": False,
            "crossmodel_field_specific_context_carrier_physically_replicated": True,
            "abstract_binding_algorithm_established": False,
            "natural_necessity_established": False,
            "single_neuron_mechanism_established": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "phase394_formal_behavior_cases": p394["denominator"]["candidate_case_count"],
            "phase394_formal_parallel_groups": p394["denominator"]["candidate_parallel_group_count"],
            "phase395_natural_behavior_cases": p395_behavior["denominator"]["candidate_case_count"],
            "phase395_natural_parallel_groups": p395_behavior["denominator"]["candidate_parallel_group_count"],
            "phase395_calibration_causal_directions": p395["denominator"]["direction_count"],
            "phase396_physical_directions": p396["denominator"]["direction_count"],
        },
        "results": {
            "phase394_qualified_formal_groups": p394["denominator"]["qualified_parallel_group_count"],
            "phase395_qualified_natural_groups": p395_behavior["denominator"]["qualified_parallel_group_count"],
            "phase395_eligible_surfaces": p395_behavior["denominator"]["eligible_surface_count"],
            "phase395_local_context_transport_cells": p395["results"]["local_static_same_literal_context_transport_cell_count"],
            "phase395_context_transport_cell_denominator": p395["denominator"]["cell_count"],
            "phase395_crosssurface_shared_state_count": int(p395["results"]["crossmodel_crosssurface_shared_state_gate_pass"]),
            "phase396_physical_model_cells": p396["results"]["physical_model_cell_pass_count"],
            "phase396_physical_model_cell_denominator": 3,
            "phase396_same_literal_answer_switches": physical_switches,
            "phase396_same_position_content_switches": content_switches,
            "phase396_direction_count": p396["denominator"]["direction_count"],
            "abstract_binding_algorithm_count": 0,
            "natural_necessity_count": 0,
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        },
        "candidate_layers": {
            model: payloads["phase395_discovery_candidate_freeze.json"]["frozen_candidate"]["model_layers"][model]
            for model in MODELS
        },
        "hard_limits": [
            "formal_pointer_failure_measures_artificial_interface_ability_not_natural_binding_absence",
            "same_token_multiset_keeps_token_order_and_contextual_history_as_available_information",
            "phase395_crosssurface_gate_failed_in_glm4_and_deepseek7b_entity_recency_cells",
            "same_position_content_transport_remains_stronger_than_same_literal_context_transport",
            "phase396_replication_covers_only_field_extraction",
            "donor_state_sufficiency_does_not_establish_natural_necessity",
            "aggregate_token_state_interventions_do_not_identify_attention_heads_or_mlp_neurons",
            "three_small_models_do_not_establish_large_model_equivalence",
        ],
        "authorization": {
            "show_field_context_carrier_anchor": True,
            "show_same_position_content_control": True,
            "show_crosssurface_binding_rule": False,
            "show_complete_language_path": False,
            "show_specific_neuron_path": False,
            "run_unbounded_neuron_scan": False,
        },
        "next_stage": {
            "objective": "freeze_broader_natural_relation_surfaces_and_separate_order_local_context_and_binding",
            "automatic_continuation_authorized": False,
            "reason": "Phase396 completes the field-specific physical replication; broader surfaces require a new untouched denominator.",
        },
        "single_global_progress_percentage_valid": False,
    }


def evidence_graph(stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = stage["results"]
    nodes = [
        {
            "node_id": "p394_formal_pointer_interface_negative",
            "node_type": "formal_pointer_interface_crossmodel_negative",
            "phase_id": "Phase394",
            "qualified_group_count": results["phase394_qualified_formal_groups"],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p395_natural_same_multiset_denominator",
            "node_type": "natural_same_token_multiset_binding_denominator",
            "phase_id": "Phase395",
            "qualified_group_count": results["phase395_qualified_natural_groups"],
            "eligible_surface_count": results["phase395_eligible_surfaces"],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p395_query_contrast_candidate",
            "node_type": "crossmodel_crosssurface_query_contrast_candidate",
            "phase_id": "Phase395",
            "relative_depth": 4 / 7,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p395_local_context_transport",
            "node_type": "same_literal_context_state_transport_calibration",
            "phase_id": "Phase395",
            "passing_cell_count": results["phase395_local_context_transport_cells"],
            "cell_denominator": results["phase395_context_transport_cell_denominator"],
            "causal": True,
            "causal_scope": "controlled_same_literal_context_state_sufficiency",
            "language_path": False,
        },
        {
            "node_id": "p395_crosssurface_shared_state_negative",
            "node_type": "crosssurface_static_binding_state_negative",
            "phase_id": "Phase395",
            "passing_shared_state_count": results["phase395_crosssurface_shared_state_count"],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p396_field_context_carrier_physical",
            "node_type": "field_specific_same_literal_context_carrier_physical_replication",
            "phase_id": "Phase396",
            "models_passing": results["phase396_physical_model_cells"],
            "model_denominator": results["phase396_physical_model_cell_denominator"],
            "answer_switch_count": results["phase396_same_literal_answer_switches"],
            "direction_count": results["phase396_direction_count"],
            "causal": True,
            "causal_scope": "field_specific_controlled_context_state_sufficiency",
            "language_path": False,
        },
        {
            "node_id": "p396_same_position_content_control",
            "node_type": "same_position_content_transport_control",
            "phase_id": "Phase396",
            "answer_switch_count": results["phase396_same_position_content_switches"],
            "direction_count": results["phase396_direction_count"],
            "causal": True,
            "causal_scope": "content_transport_control",
            "language_path": False,
        },
        {
            "node_id": "p396_neuron_boundary",
            "node_type": "strict_neuron_promotion_boundary",
            "phase_id": "Phase396",
            "single_neuron_causal_path_count": 0,
            "complete_language_path_count": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {"edge_id": "p393_to_p394", "source_node_id": "p393_attribute_content_transport", "target_node_id": "p394_formal_pointer_interface_negative", "edge_type": "separates_content_transport_from_binding_interface", "phase_id": "Phase394", "causal_path": False},
        {"edge_id": "p394_to_p395", "source_node_id": "p394_formal_pointer_interface_negative", "target_node_id": "p395_natural_same_multiset_denominator", "edge_type": "abandons_formal_pointer_prerequisite_and_freezes_natural_pairs", "phase_id": "Phase395", "causal_path": False},
        {"edge_id": "p395_denominator_to_contrast", "source_node_id": "p395_natural_same_multiset_denominator", "target_node_id": "p395_query_contrast_candidate", "edge_type": "discovers_and_calibrates_query_position_contrast", "phase_id": "Phase395", "causal_path": False},
        {"edge_id": "p395_contrast_to_transport", "source_node_id": "p395_query_contrast_candidate", "target_node_id": "p395_local_context_transport", "edge_type": "same_literal_parent_boundary_intervention", "phase_id": "Phase395", "causal_path": True, "complete_language_path": False},
        {"edge_id": "p395_transport_to_shared_negative", "source_node_id": "p395_local_context_transport", "target_node_id": "p395_crosssurface_shared_state_negative", "edge_type": "fails_all_model_all_surface_gate", "phase_id": "Phase395", "causal_path": False},
        {"edge_id": "p395_local_to_p396_physical", "source_node_id": "p395_local_context_transport", "target_node_id": "p396_field_context_carrier_physical", "edge_type": "independent_field_specific_physical_replication", "phase_id": "Phase396", "causal_path": True, "complete_language_path": False},
        {"edge_id": "p396_control_boundary", "source_node_id": "p396_same_position_content_control", "target_node_id": "p396_neuron_boundary", "edge_type": "content_control_and_missing_crosssurface_rule_block_neuron_promotion", "phase_id": "Phase396", "causal_path": False},
    ]
    return nodes, edges


def aggregate_nodes(model: str, cell: dict[str, Any], updated_at: str) -> list[dict[str, Any]]:
    layer = {"qwen3": 20, "glm4": 22, "deepseek7b": 15}[model]
    model_revision = None
    partition_path = NEURON_TARGET / f"partitions/content_knowledge/{model}.json"
    if partition_path.is_file():
        partition = read_json(partition_path)
        model_revision = partition.get("model_snapshot", {}).get("revision")
    definitions = (
        ("same_literal_context", "donor_same_literal_candidate", "context_carrier", 0),
        ("same_position_content", "donor_same_position_candidate", "content_control", 1),
    )
    rows = []
    for suffix, scenario, cohort, unit_index in definitions:
        summary = cell["scenario_summaries"][scenario]
        rows.append({
            "schema_version": "aggregate_state_anchor.v1",
            "node_id": f"content_knowledge:{model}:L{layer}:phase396:{suffix}",
            "node_type": "aggregate_token_state_anchor",
            "family_id": "content_knowledge",
            "family_name": "内容知识模式族",
            "relation": "field_extraction",
            "model": model,
            "model_revision": model_revision,
            "layer": layer,
            "component": "residual_stream",
            "unit_kind": "token_state_aggregate",
            "unit_index": unit_index,
            "token_position": "literal_value_positions",
            "candidate_score": summary["median_normalized_margin_mediation"],
            "case_count": 24,
            "natural_observed": False,
            "group_intervention_supported": True,
            "expanded_confirmation_pass": True,
            "causal_scope": (
                "field_specific_same_literal_context_state_sufficiency"
                if cohort == "context_carrier" else "same_position_content_transport_control"
            ),
            "evidence_level": "L5-controlled-sufficiency",
            "evidence_status": "independent_physical_replication_not_single_neuron",
            "evidence_boundary": (
                "Aggregate literal-token states, not an attention head or MLP neuron; "
                "field-specific sufficiency does not establish crosssurface binding or natural necessity."
            ),
            "display_priority": 12 + summary["median_normalized_margin_mediation"],
            "phase396_tested": True,
            "phase396_cohort": cohort,
            "phase396_physical_replication_pass": cell["physical_static_same_literal_context_transport_gate_pass"],
            "phase396_normalized_margin_mediation": summary["median_normalized_margin_mediation"],
            "phase396_positive_direction_rate": summary["positive_direction_rate"],
            "phase396_answer_switch_rate": summary["answer_switch_rate"],
            "is_real_unit": False,
            "single_neuron_claim": False,
            "generated_at": updated_at,
            "source_artifacts": [
                "phase396_physical_analysis.json",
                "phase396_protocol.json",
            ],
        })
    return rows


def update_neuron_atlas(stage: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], physical: dict[str, Any], updated_at: str) -> None:
    cell_map = physical_cell_by_model(physical)
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase396_binding_separation_stage_summary.json", stage)
        write_jsonl(root / "phase396_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase396_evidence_edges.jsonl", edges)
        for model in MODELS:
            path = root / f"partitions/content_knowledge/{model}.json"
            partition = read_json(path)
            partition["nodes"] = [
                node for node in partition.get("nodes", []) if not node.get("phase396_tested")
            ] + aggregate_nodes(model, cell_map[model], updated_at)
            partition["mapping_status"] = "phase396_field_context_carrier_physical_not_single_neuron"
            partition["generated_at"] = updated_at
            partition["metrics"]["phase396_aggregate_state_anchor_count"] = 2
            partition["metrics"]["phase396_context_carrier_anchor_count"] = 1
            partition["metrics"]["phase396_content_control_anchor_count"] = 1
            partition["metrics"]["phase396_single_neuron_causal_count"] = 0
            partition["evidence_boundary"] = (
                "Phase396 physically replicates a field-extraction same-literal contextual "
                "state carrier at an aggregate token-state boundary. It is not a neuron, "
                "not crosssurface binding, and not natural necessity."
            )
            partition.setdefault("source_artifacts", [])
            for source in ("phase396_physical_analysis.json", "phase396_protocol.json"):
                if source not in partition["source_artifacts"]:
                    partition["source_artifacts"].append(source)
            write_json(path, partition)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 396
        manifest["generated_at"] = updated_at
        manifest["phase396_audit"] = {
            "status": "field_context_carrier_physical_crosssurface_binding_absent",
            "physical_direction_count": 72,
            "same_literal_answer_switch_count": stage["results"]["phase396_same_literal_answer_switches"],
            "same_position_content_switch_count": stage["results"]["phase396_same_position_content_switches"],
            "models_passing_field_context_transport": 3,
            "crosssurface_shared_state_count": 0,
            "new_aggregate_state_anchor_count": 6,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase396_binding_separation_stage_summary.json",
        }
        metrics = manifest.setdefault("metrics", {})
        metrics.update({
            "phase394_formal_qualified_group_count": 0,
            "phase395_natural_qualified_group_count": stage["results"]["phase395_qualified_natural_groups"],
            "phase395_local_context_transport_cell_count": stage["results"]["phase395_local_context_transport_cells"],
            "phase395_context_transport_cell_denominator": stage["results"]["phase395_context_transport_cell_denominator"],
            "phase395_crosssurface_shared_state_count": 0,
            "phase396_physical_direction_count": 72,
            "phase396_same_literal_answer_switch_count": stage["results"]["phase396_same_literal_answer_switches"],
            "phase396_same_position_content_switch_count": stage["results"]["phase396_same_position_content_switches"],
            "phase396_field_context_transport_pass_model_count": 3,
            "phase396_aggregate_state_anchor_count": 6,
            "phase396_single_neuron_causal_count": 0,
        })
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase396_binding_separation_stage_summary.json"
        boundary = manifest.setdefault("evidence_boundary", {})
        boundary.update({
            "latest_phase": "Phase396-FieldPhysicalReplication",
            "statement": (
                "Same-literal contextual token states causally shifted field-extraction "
                "answers in all three models on independent physical groups (46/72 strict "
                "switches). The crosssurface Phase395 gate failed; aggregate anchors are "
                "not neurons and no complete language path is promoted."
            ),
            "field_specific_context_carrier_available": True,
            "crosssurface_binding_rule_available": False,
            "natural_necessity_available": False,
            "upstream_language_path_available": False,
            "single_unit_causal_closure": False,
        })
        write_json(root / "manifest.json", manifest)
        checksum_path = root / "checksums.json"
        if checksum_path.is_file():
            write_json(checksum_path, {
                "schema_version": "artifact_checksums.v1",
                "files": [
                    {"path": str(item.relative_to(root)), "sha256": sha256(item)}
                    for item in sorted(root.rglob("*"))
                    if item.is_file() and item != checksum_path
                ],
            })
        public_manifest(root, updated_at)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase394-396 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    nodes, edges = evidence_graph(stage)
    payloads["phase396_binding_separation_stage_summary.json"] = stage
    published = [*payloads, "phase396_evidence_nodes.jsonl", "phase396_evidence_edges.jsonl"]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase396_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase396_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase396-BindingSeparationStage"
        manifest["phase394_396"] = {
            "status": "field_context_carrier_physical_crosssurface_binding_absent",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase396-BindingSeparationStage"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["binding_separation_stage"] = {
            "formal_pointer_qualified_groups": {"numerator": 0, "denominator": 72},
            "natural_behavior_qualified_groups": {"numerator": stage["results"]["phase395_qualified_natural_groups"], "denominator": 72},
            "eligible_natural_surfaces": {"numerator": stage["results"]["phase395_eligible_surfaces"], "denominator": 3},
            "local_context_transport_cells": {"numerator": stage["results"]["phase395_local_context_transport_cells"], "denominator": stage["results"]["phase395_context_transport_cell_denominator"]},
            "crosssurface_shared_states": {"numerator": 0, "denominator": 1},
            "field_physical_model_cells": {"numerator": 3, "denominator": 3},
            "field_same_literal_answer_switches": {"numerator": stage["results"]["phase396_same_literal_answer_switches"], "denominator": 72},
            "same_position_content_switches": {"numerator": stage["results"]["phase396_same_position_content_switches"], "denominator": 72},
            "abstract_binding_algorithms": {"numerator": 0, "denominator": 1},
            "complete_language_paths": {"numerator": 0, "denominator": 72},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
        }
        progress["phase396_decision"] = "record_field_specific_context_carrier_keep_crosssurface_binding_and_neuron_gates_closed"
        write_json(root / "progress.json", progress)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase396-BindingSeparationStage"
            index["latest_stage_files"] = [
                "phase396_binding_separation_stage_summary.json",
                "phase396_evidence_nodes.jsonl",
                "phase396_evidence_edges.jsonl",
                "phase396_physical_analysis.json",
                "phase395_causal_calibration_analysis.json",
                "phase395_discovery_candidate_freeze.json",
                "phase394_behavior_freeze_summary.json",
            ]
            initial = index.setdefault("initial_files", [])
            for name in index["latest_stage_files"]:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
    update_neuron_atlas(stage, nodes, edges, payloads["phase396_physical_analysis.json"], updated_at)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
