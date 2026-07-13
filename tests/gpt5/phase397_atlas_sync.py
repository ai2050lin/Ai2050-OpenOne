#!/usr/bin/env python3
"""Publish Phase397 factor-separated binding evidence to both atlas mirrors."""

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
P397 = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LAYERS = {"qwen3": 20, "glm4": 22, "deepseek7b": 15}
SURFACE_FAMILIES = {
    "possession_relation": ("content_knowledge", "内容知识模式族"),
    "role_filling": ("language_action", "语言行为模式族"),
    "coreference_resolution": ("reasoning_constraint", "推理约束模式族"),
}
JSON_SOURCES = {
    "phase397_protocol.json": P397 / "phase397_protocol.json",
    "phase397_behavior_freeze_summary.json": P397 / "phase397_behavior_freeze_summary.json",
    "phase397_factor_trace_protocol.json": P397 / "phase397_factor_trace_protocol.json",
    "phase397_factor_trace_instrument_audit.json": P397 / "phase397_factor_trace_instrument_audit.json",
    "phase397_factor_discovery_analysis.json": P397 / "phase397_factor_discovery_analysis.json",
    "phase397_factor_calibration_analysis.json": P397 / "phase397_factor_calibration_analysis.json",
    "phase397_factor_physical_analysis.json": P397 / "phase397_factor_physical_analysis.json",
    "phase397_causal_protocol.json": P397 / "phase397_causal_protocol.json",
    "phase397_causal_analysis.json": P397 / "phase397_causal_analysis.json",
}


def stage_summary(updated_at: str, payloads: dict[str, Any]) -> dict[str, Any]:
    behavior = payloads["phase397_behavior_freeze_summary.json"]
    discovery = payloads["phase397_factor_discovery_analysis.json"]
    calibration = payloads["phase397_factor_calibration_analysis.json"]
    physical = payloads["phase397_factor_physical_analysis.json"]
    causal = payloads["phase397_causal_analysis.json"]
    relation_values = [
        cell["scenario_summaries"]["donor_relation_candidate"]["median_normalized_relation_margin_mediation"]
        for cell in causal["cells"]
    ]
    content_values = [
        cell["scenario_summaries"]["donor_content_candidate"]["median_normalized_relation_margin_mediation"]
        for cell in causal["cells"]
    ]
    order_values = [
        cell["scenario_summaries"]["donor_order_candidate"]["median_normalized_relation_margin_mediation"]
        for cell in causal["cells"]
    ]
    return {
        "schema_version": "71.12.0",
        "phase_id": "Phase397-FactorSeparatedBindingStage",
        "created_at": updated_at,
        "objective": "separate_relation_context_signature_from_content_position_order_syntax_query_and_task_factors",
        "assessment": {
            "multitask_behavior_denominator_frozen": True,
            "three_surface_behavior_gate_pass": True,
            "three_split_relation_signature_replication": True,
            "same_position_relation_context_causal_carrier": False,
            "crosssurface_binding_rule_established": False,
            "natural_necessity_established": False,
            "complete_physical_path_established": False,
            "single_neuron_mechanism_established": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "behavior_case_count": behavior["denominator"]["candidate_case_count"],
            "behavior_parallel_group_count": behavior["denominator"]["candidate_parallel_group_count"],
            "qualified_parallel_group_count": behavior["denominator"]["qualified_parallel_group_count"],
            "eligible_surface_count": behavior["denominator"]["eligible_surface_count"],
            "registered_surface_count": 6,
            "discovery_factor_pair_count": discovery["denominator"]["raw_factor_pair_count"],
            "calibration_factor_pair_count": calibration["denominator"]["raw_factor_pair_count"],
            "physical_factor_pair_count": physical["denominator"]["raw_factor_pair_count"],
            "causal_direction_count": causal["denominator"]["direction_count"],
            "causal_scenario_count": causal["denominator"]["scenario_count"],
            "causal_generation_count": causal["denominator"]["generation_count"],
        },
        "surface_gates": behavior["surface_gates"],
        "results": {
            "discovery_observational_cells": discovery["results"]["passing_model_surface_cell_count"],
            "calibration_observational_cells": calibration["results"]["passing_model_surface_cell_count"],
            "physical_observational_cells": physical["results"]["passing_model_surface_cell_count"],
            "observational_cell_denominator_per_split": 9,
            "causal_relation_context_cells": causal["results"]["passing_model_surface_cell_count"],
            "causal_relation_context_cell_denominator": 9,
            "relation_answer_switch_count": sum(cell["relation_answer_switch_count"] for cell in causal["cells"]),
            "relation_direction_count": causal["denominator"]["direction_count"],
            "relation_median_mediation_range": [min(relation_values), max(relation_values)],
            "content_median_mediation_range": [min(content_values), max(content_values)],
            "order_median_mediation_range": [min(order_values), max(order_values)],
            "query_source_maximum_effect": causal["results"]["maximum_query_source_control_effect"],
            "abstract_binding_algorithm_count": 0,
            "natural_necessity_count": 0,
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        },
        "phase396_reinterpretation": {
            "field_specific_old_contrast_result_retained": True,
            "old_contrast_moved_literal_value_positions": True,
            "old_result_generalizes_to_pure_same_position_relation_context": False,
            "reason": "Phase397 fixed value identity and absolute positions while changing only preceding entity slots; the replicated signature produced 0/144 answer switches and 0/9 causal cells.",
        },
        "hard_limits": [
            "three_of_six_surfaces_failed_the_strict_three_model_ten_condition_behavior_gate",
            "relation_signatures_are_observationally_stable_but_not_independently_sufficient",
            "content_and_clause_order_state_transfers_are_larger_than_pure_relation_context_transfer",
            "two_frozen_depths_do_not_map_the_complete_formation_curve",
            "aggregate_value_token_states_do_not_identify_attention_heads_or_mlp_neurons",
            "causal_sufficiency_failure_does_not_prove_relation_information_is_unused_in_joint_computation",
            "three_small_models_do_not_establish_large_model_equivalence",
        ],
        "authorization": {
            "show_relation_signature_anchors": True,
            "show_relation_signature_as_causal_carrier": False,
            "show_crosssurface_binding_rule": False,
            "show_complete_language_path": False,
            "show_specific_neuron_path": False,
            "run_calibration_or_physical_causal_intervention": False,
            "run_unbounded_neuron_scan": False,
        },
        "next_stage": {
            "objective": "localize_joint_structure_content_integration_after_value_context_without_assuming_a_portable_relation_vector",
            "automatic_continuation_authorized": False,
            "reason": "Phase397 closes the isolated same-position relation-context carrier route; the next stage requires a new joint-state intervention contract rather than more groups or layer tuning.",
        },
        "single_global_progress_percentage_valid": False,
    }


def evidence_graph(stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = stage["results"]
    nodes = [
        {
            "node_id": "p397_multitask_behavior_denominator",
            "node_type": "six_surface_ten_condition_behavior_denominator",
            "phase_id": "Phase397",
            "case_count": stage["denominators"]["behavior_case_count"],
            "qualified_group_count": stage["denominators"]["qualified_parallel_group_count"],
            "eligible_surface_count": stage["denominators"]["eligible_surface_count"],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p397_relation_signature_three_split",
            "node_type": "same_position_relation_context_signature",
            "phase_id": "Phase397",
            "discovery_cells": results["discovery_observational_cells"],
            "calibration_cells": results["calibration_observational_cells"],
            "physical_cells": results["physical_observational_cells"],
            "cell_denominator_per_split": results["observational_cell_denominator_per_split"],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p397_relation_context_causal_negative",
            "node_type": "isolated_same_position_relation_context_causal_negative",
            "phase_id": "Phase397",
            "passing_cell_count": results["causal_relation_context_cells"],
            "cell_denominator": results["causal_relation_context_cell_denominator"],
            "answer_switch_count": results["relation_answer_switch_count"],
            "direction_count": results["relation_direction_count"],
            "causal": True,
            "causal_effect_established": False,
            "language_path": False,
        },
        {
            "node_id": "p397_content_order_dominance_control",
            "node_type": "content_and_order_transport_control",
            "phase_id": "Phase397",
            "content_mediation_range": results["content_median_mediation_range"],
            "order_mediation_range": results["order_median_mediation_range"],
            "causal": True,
            "causal_scope": "control_transport_not_binding_rule",
            "language_path": False,
        },
        {
            "node_id": "p397_neuron_promotion_closed",
            "node_type": "strict_neuron_promotion_boundary",
            "phase_id": "Phase397",
            "single_neuron_causal_path_count": 0,
            "complete_language_path_count": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {"edge_id": "p396_to_p397_denominator", "source_node_id": "p396_field_context_carrier_physical", "target_node_id": "p397_multitask_behavior_denominator", "edge_type": "freezes_stricter_multitask_same_position_factor_denominator", "phase_id": "Phase397", "causal_path": False},
        {"edge_id": "p397_denominator_to_signature", "source_node_id": "p397_multitask_behavior_denominator", "target_node_id": "p397_relation_signature_three_split", "edge_type": "three_split_observational_replication", "phase_id": "Phase397", "causal_path": False},
        {"edge_id": "p397_signature_to_causal_negative", "source_node_id": "p397_relation_signature_three_split", "target_node_id": "p397_relation_context_causal_negative", "edge_type": "same_literal_same_position_parent_boundary_intervention", "phase_id": "Phase397", "causal_path": False},
        {"edge_id": "p397_controls_to_negative", "source_node_id": "p397_content_order_dominance_control", "target_node_id": "p397_relation_context_causal_negative", "edge_type": "factor_specificity_controls_dominate_relation_patch", "phase_id": "Phase397", "causal_path": False},
        {"edge_id": "p397_negative_to_neuron_boundary", "source_node_id": "p397_relation_context_causal_negative", "target_node_id": "p397_neuron_promotion_closed", "edge_type": "causal_gate_failure_blocks_neuron_scan", "phase_id": "Phase397", "causal_path": False},
    ]
    return nodes, edges


def cell_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(cell["model"], cell["task_surface"]): cell for cell in payload["cells"]}


def phase397_node(model: str, surface: str, physical: dict[str, Any], causal: dict[str, Any], updated_at: str) -> dict[str, Any]:
    family_id, family_name = SURFACE_FAMILIES[surface]
    layer = MODEL_LAYERS[model]
    return {
        "schema_version": "aggregate_state_anchor.v1",
        "node_id": f"{family_id}:{model}:L{layer}:phase397:{surface}:relation_signature",
        "node_type": "aggregate_token_state_anchor",
        "family_id": family_id,
        "family_name": family_name,
        "relation": surface,
        "model": model,
        "layer": layer,
        "component": "residual_stream_layer_input",
        "unit_kind": "token_state_aggregate",
        "unit_index": 3,
        "token_position": "literal_value_positions",
        "candidate_score": physical["median_minimum_two_axis_relation_candidate_delta"],
        "case_count": 120,
        "natural_observed": True,
        "group_intervention_supported": False,
        "expanded_confirmation_pass": True,
        "causal_scope": "observational_relation_context_signature_not_portable_carrier",
        "evidence_level": "L3-three-split-observational",
        "evidence_status": "replicated_signature_causal_sufficiency_rejected",
        "evidence_boundary": "Aggregate value-token relation signature. Three independent splits replicated, but isolated transport passed 0/9 cells and switched 0/144 answers; this is not a neuron or a binding rule.",
        "display_priority": 12 + physical["median_minimum_two_axis_relation_candidate_delta"],
        "phase397_tested": True,
        "phase397_cohort": "relation_signature_observational",
        "phase397_physical_observational_pass": physical["physical_observational_candidate_gate_pass"],
        "phase397_relation_candidate_delta": physical["median_minimum_two_axis_relation_candidate_delta"],
        "phase397_relation_wrong_depth_delta": physical["median_minimum_two_axis_relation_wrong_depth_delta"],
        "phase397_causal_gate_pass": causal["causal_relation_context_specificity_gate_pass"],
        "phase397_relation_mediation": causal["scenario_summaries"]["donor_relation_candidate"]["median_normalized_relation_margin_mediation"],
        "phase397_relation_answer_switch_rate": causal["relation_answer_switch_rate"],
        "is_real_unit": False,
        "single_neuron_claim": False,
        "generated_at": updated_at,
        "source_artifacts": ["phase397_factor_physical_analysis.json", "phase397_causal_analysis.json"],
    }


def update_neuron_atlas(stage: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], payloads: dict[str, Any], updated_at: str) -> None:
    physical = cell_map(payloads["phase397_factor_physical_analysis.json"])
    causal = cell_map(payloads["phase397_causal_analysis.json"])
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase397_factor_separated_binding_stage_summary.json", stage)
        write_jsonl(root / "phase397_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase397_evidence_edges.jsonl", edges)
        for surface, (family_id, _family_name) in SURFACE_FAMILIES.items():
            for model in MODELS:
                path = root / f"partitions/{family_id}/{model}.json"
                partition = read_json(path)
                retained = [node for node in partition.get("nodes", []) if not node.get("phase397_tested")]
                for node in retained:
                    if node.get("phase396_tested"):
                        node["phase397_followup_scope_limit"] = "Phase396 moved literal value positions; Phase397 same-position relation-only transport failed."
                        node["phase397_superseded_as_crosssurface_binding_evidence"] = True
                partition["nodes"] = retained + [phase397_node(model, surface, physical[(model, surface)], causal[(model, surface)], updated_at)]
                partition["mapping_status"] = "phase397_relation_signature_replicated_causal_carrier_rejected"
                partition["generated_at"] = updated_at
                metrics = partition.setdefault("metrics", {})
                metrics.update({
                    "phase397_relation_signature_anchor_count": 1,
                    "phase397_observational_physical_pass_count": 1,
                    "phase397_causal_relation_carrier_count": 0,
                    "phase397_single_neuron_causal_count": 0,
                })
                partition["evidence_boundary"] = "Phase397 displays one aggregate relation-signature anchor for this model/task. It is observationally replicated, causally nonportable, and not a neuron."
                sources = partition.setdefault("source_artifacts", [])
                for source in ("phase397_factor_physical_analysis.json", "phase397_causal_analysis.json"):
                    if source not in sources:
                        sources.append(source)
                write_json(path, partition)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 397
        manifest["generated_at"] = updated_at
        manifest["phase397_audit"] = {
            "status": "three_split_relation_signature_replicated_causal_carrier_rejected",
            "behavior_case_count": 4320,
            "qualified_group_count": 79,
            "eligible_surface_count": 3,
            "registered_surface_count": 6,
            "observational_pass_cells_per_split": 9,
            "causal_direction_count": 144,
            "causal_pass_cell_count": 0,
            "relation_answer_switch_count": 0,
            "new_aggregate_state_anchor_count": 9,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase397_factor_separated_binding_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update({
            "phase397_behavior_case_count": 4320,
            "phase397_behavior_qualified_group_count": 79,
            "phase397_eligible_surface_count": 3,
            "phase397_registered_surface_count": 6,
            "phase397_discovery_observational_pass_cell_count": 9,
            "phase397_calibration_observational_pass_cell_count": 9,
            "phase397_physical_observational_pass_cell_count": 9,
            "phase397_causal_direction_count": 144,
            "phase397_causal_relation_pass_cell_count": 0,
            "phase397_relation_answer_switch_count": 0,
            "phase397_aggregate_state_anchor_count": 9,
            "phase397_single_neuron_causal_count": 0,
        })
        manifest.setdefault("files", {})["latest_evidence_summary"] = "phase397_factor_separated_binding_stage_summary.json"
        boundary = manifest.setdefault("evidence_boundary", {})
        boundary.update({
            "latest_phase": "Phase397-FactorSeparatedBindingStage",
            "statement": "Same-position value states carry stable relation-correlated signatures over three tasks and three models, but isolated transport passed 0/9 causal cells and switched 0/144 answers. Displayed Phase397 anchors are aggregate observations, not neurons or binding rules.",
            "relation_signature_available": True,
            "portable_relation_carrier_available": False,
            "crosssurface_binding_rule_available": False,
            "natural_necessity_available": False,
            "upstream_language_path_available": False,
            "single_unit_causal_closure": False,
            "candidate_depth_specificity_available": False,
        })
        write_json(root / "manifest.json", manifest)
        checksum_path = root / "checksums.json"
        if checksum_path.is_file():
            write_json(checksum_path, {
                "schema_version": "artifact_checksums.v1",
                "files": [
                    {"path": str(item.relative_to(root)), "sha256": sha256(item)}
                    for item in sorted(root.rglob("*")) if item.is_file() and item != checksum_path
                ],
            })
        public_manifest(root, updated_at)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase397 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    nodes, edges = evidence_graph(stage)
    payloads["phase397_factor_separated_binding_stage_summary.json"] = stage
    published = [*payloads, "phase397_evidence_nodes.jsonl", "phase397_evidence_edges.jsonl"]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase397_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase397_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase397-FactorSeparatedBindingStage"
        manifest["phase397"] = {
            "status": "relation_signature_replicated_causal_carrier_rejected",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase397-FactorSeparatedBindingStage"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["factor_separated_binding_stage"] = {
            "behavior_cases": {"numerator": 4320, "denominator": 4320},
            "qualified_parallel_groups": {"numerator": 79, "denominator": 144},
            "eligible_surfaces": {"numerator": 3, "denominator": 6},
            "discovery_observational_cells": {"numerator": 9, "denominator": 9},
            "calibration_observational_cells": {"numerator": 9, "denominator": 9},
            "physical_observational_cells": {"numerator": 9, "denominator": 9},
            "causal_relation_context_cells": {"numerator": 0, "denominator": 9},
            "relation_answer_switches": {"numerator": 0, "denominator": 144},
            "abstract_binding_algorithms": {"numerator": 0, "denominator": 1},
            "complete_language_paths": {"numerator": 0, "denominator": 72},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
        }
        progress["phase397_decision"] = "record_relation_signatures_reject_isolated_portable_carrier_close_neuron_gate"
        write_json(root / "progress.json", progress)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase397-FactorSeparatedBindingStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
    update_neuron_atlas(stage, nodes, edges, payloads, updated_at)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
