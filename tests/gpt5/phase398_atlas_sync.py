#!/usr/bin/env python3
"""Publish Phase398 order-conditioned joint-interaction evidence to both atlases."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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
P398 = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACE_FAMILIES = {
    "possession_relation": ("content_knowledge", "内容知识模式族"),
    "role_filling": ("language_action", "语言行为模式族"),
    "coreference_resolution": ("reasoning_constraint", "推理约束模式族"),
}
JSON_SOURCES = {
    "phase398_protocol.json": P398 / "phase398_protocol.json",
    "phase398_query_trace_protocol.json": P398 / "phase398_query_trace_protocol.json",
    "phase398_instrument_audit.json": P398 / "phase398_instrument_audit.json",
    "phase398_discovery_analysis_protocol.json": P398 / "phase398_discovery_analysis_protocol.json",
    "phase398_discovery_analysis.json": P398 / "phase398_discovery_analysis.json",
    "phase398_order_conditioned_candidate_freeze.json": P398 / "phase398_order_conditioned_candidate_freeze.json",
    "phase398_order_conditioned_calibration_validation.json": P398 / "phase398_order_conditioned_calibration_validation.json",
    "phase398_order_conditioned_physical_validation.json": P398 / "phase398_order_conditioned_physical_validation.json",
    "phase398_order_conditioned_causal_protocol.json": P398 / "phase398_order_conditioned_causal_protocol.json",
    "phase398_order_conditioned_causal_instrument_audit.json": P398 / "phase398_order_conditioned_causal_instrument_audit.json",
    "phase398_order_conditioned_causal_analysis.json": P398 / "phase398_order_conditioned_causal_analysis.json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_summary(updated_at: str, payloads: dict[str, Any]) -> dict[str, Any]:
    discovery = payloads["phase398_discovery_analysis.json"]
    calibration = payloads["phase398_order_conditioned_calibration_validation.json"]
    physical = payloads["phase398_order_conditioned_physical_validation.json"]
    causal = payloads["phase398_order_conditioned_causal_analysis.json"]
    return {
        "schema_version": "72.14.0",
        "phase_id": "Phase398-OrderConditionedJointBindingStage",
        "created_at": updated_at,
        "objective": "map_relation_order_query_joint_trajectories_at_query_integration_and_test_single_position_causal_sufficiency",
        "assessment": {
            "complete_sixteen_condition_factorial_denominator": True,
            "three_model_three_surface_behavior_gate": True,
            "order_invariant_rq_candidate": False,
            "order_conditioned_roq_three_split_replication": True,
            "order_conditioned_roq_crosslexical_reuse": True,
            "single_query_position_causal_sufficiency": False,
            "natural_necessity_established": False,
            "complete_binding_algorithm_established": False,
            "complete_language_path_established": False,
            "single_neuron_mechanism_established": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "behavior_case_count": 3456,
            "candidate_parallel_group_count": 72,
            "qualified_parallel_group_count": 68,
            "frozen_trace_case_count": 2304,
            "instrument_trace_case_count": 144,
            "discovery_trace_case_count": 1152,
            "calibration_trace_case_count": 576,
            "physical_trace_case_count": 576,
            "model_surface_cell_count": 9,
            "causal_direction_count": causal["denominator"]["direction_count"],
            "causal_scenario_count": causal["denominator"]["scenario_count"],
        },
        "results": {
            "order_invariant_rq_discovery_cells": discovery["results"]["qualified_model_surface_cell_count"],
            "order_conditioned_roq_discovery_cells": 9,
            "order_conditioned_roq_calibration_cells": calibration["results"]["passing_model_surface_cell_count"],
            "order_conditioned_roq_physical_cells": physical["results"]["passing_model_surface_cell_count"],
            "order_conditioned_single_position_causal_cells": causal["results"]["passing_causal_cell_count"],
            "same_order_query_state_answer_switch_count": causal["results"]["same_order_total_answer_switch_count"],
            "causal_direction_count": causal["results"]["direction_count"],
            "observational_candidate_layer_relative_depth_range": [0.282051282, 0.62962963],
            "physical_roq_norm_range": [
                min(row["median_min_axis_normalized_roq_norm"] for row in physical["cells"]),
                max(row["median_min_axis_normalized_roq_norm"] for row in physical["cells"]),
            ],
            "physical_roq_cross_axis_cosine_range": [
                min(row["median_roq_cross_axis_cosine"] for row in physical["cells"]),
                max(row["median_roq_cross_axis_cosine"] for row in physical["cells"]),
            ],
            "abstract_binding_algorithm_count": 0,
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        },
        "hard_limits": [
            "the_original_order_invariant_RQ_candidate_passed_zero_of_nine_specificity_gates",
            "ROQ_is_a_factorial_contrast_not_a_standalone_state_or_neuron",
            "whole_clause_order_changes_literal_absolute_positions_and_route_geometry",
            "single_query_position_transport_passed_zero_of_nine_causal_cells",
            "ten_of_432_switches_do_not_establish_crossmodel_crosssurface_sufficiency",
            "one_DS7B_calibration_group_was_excluded_whole_after_one_completion_replay_mismatch",
            "only_three_behavior_eligible_surfaces_and_three_small_models_were_tested",
            "no_attention_head_MLP_channel_or_single_neuron_path_was_identified",
        ],
        "authorization": {
            "show_order_conditioned_joint_anchors": True,
            "show_anchors_as_aggregate_observations": True,
            "show_anchors_as_causal_binding_path": False,
            "show_specific_neuron_path": False,
            "run_single_neuron_scan": False,
            "claim_language_encoding_closure": False,
        },
        "next_stage": {
            "objective": "trace_multi_position_multi_component_query_integration_with_attention_source_edges_and_MLP_writes_without_assuming_a_portable_single_position_state",
            "automatic_continuation_authorized": False,
            "reason": "Phase398 completed the frozen factorial, three-split observation, and finite causal test. The next stage needs a new multi-position dynamic intervention denominator rather than layer tuning on the closed single-position object.",
        },
        "single_global_progress_percentage_valid": False,
    }


def graph(stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {"node_id": "p398_complete_factorial_denominator", "node_type": "sixteen_condition_joint_factorial_denominator", "phase_id": "Phase398", "case_count": 3456, "qualified_parallel_group_count": 68, "causal": False, "language_path": False},
        {"node_id": "p398_order_invariant_rq_negative", "node_type": "order_invariant_relation_query_candidate_negative", "phase_id": "Phase398", "passing_cell_count": 0, "cell_denominator": 9, "causal": False, "language_path": False},
        {"node_id": "p398_order_conditioned_roq_replication", "node_type": "order_conditioned_relation_query_interaction_trajectory", "phase_id": "Phase398", "discovery_cells": 9, "calibration_cells": 9, "physical_cells": 9, "cell_denominator_per_split": 9, "causal": False, "language_path": False},
        {"node_id": "p398_single_query_position_causal_negative", "node_type": "single_query_position_joint_state_causal_negative", "phase_id": "Phase398", "passing_cell_count": 0, "cell_denominator": 9, "answer_switch_count": 10, "direction_count": 432, "causal": True, "causal_effect_established": False, "language_path": False},
        {"node_id": "p398_dynamic_path_required", "node_type": "multi_position_multi_component_dynamic_path_requirement", "phase_id": "Phase398", "causal": False, "language_path": False, "single_neuron_scan_authorized": False},
    ]
    edges = [
        {"edge_id": "p397_to_p398_factorial", "source_node_id": "p397_relation_context_causal_negative", "target_node_id": "p398_complete_factorial_denominator", "edge_type": "requires_joint_factorial_query_integration_test", "phase_id": "Phase398", "causal_path": False},
        {"edge_id": "p398_factorial_to_rq_negative", "source_node_id": "p398_complete_factorial_denominator", "target_node_id": "p398_order_invariant_rq_negative", "edge_type": "frozen_primary_candidate_test", "phase_id": "Phase398", "causal_path": False},
        {"edge_id": "p398_rq_to_roq", "source_node_id": "p398_order_invariant_rq_negative", "target_node_id": "p398_order_conditioned_roq_replication", "edge_type": "reveals_order_conditioned_three_factor_trajectory", "phase_id": "Phase398", "causal_path": False},
        {"edge_id": "p398_roq_to_causal", "source_node_id": "p398_order_conditioned_roq_replication", "target_node_id": "p398_single_query_position_causal_negative", "edge_type": "frozen_parent_boundary_state_transport", "phase_id": "Phase398", "causal_path": False},
        {"edge_id": "p398_causal_to_dynamic", "source_node_id": "p398_single_query_position_causal_negative", "target_node_id": "p398_dynamic_path_required", "edge_type": "rejects_single_position_sufficiency_requires_dynamic_joint_path", "phase_id": "Phase398", "causal_path": False},
    ]
    return nodes, edges


def aggregate_node(cell: dict[str, Any], causal_cell: dict[str, Any], updated_at: str) -> dict[str, Any]:
    surface = cell["task_surface"]
    family_id, family_name = SURFACE_FAMILIES[surface]
    model = cell["model"]
    layer = cell["candidate_layer"]
    return {
        "schema_version": "aggregate_state_anchor.v1",
        "node_id": f"{family_id}:{model}:L{layer}:phase398:{surface}:order_conditioned_joint_interaction",
        "node_type": "aggregate_interaction_trajectory_anchor",
        "family_id": family_id,
        "family_name": family_name,
        "relation": surface,
        "model": model,
        "layer": layer,
        "component": "residual_stream_layer_output",
        "unit_kind": "query_end_factorial_interaction_aggregate",
        "unit_index": 4,
        "token_position": "query_end",
        "candidate_score": cell["median_min_axis_normalized_roq_norm"],
        "case_count": 4 * 16,
        "natural_observed": True,
        "group_intervention_supported": False,
        "expanded_confirmation_pass": True,
        "causal_scope": "three_split_order_conditioned_ROQ_observation_single_position_sufficiency_rejected",
        "evidence_level": "L3-three-split-observational-with-L4-negative-causal-test",
        "evidence_status": "replicated_order_conditioned_trajectory_causal_sufficiency_rejected",
        "evidence_boundary": "A query-end aggregate ROQ trajectory replicated across lexical axes, discovery, calibration, and physical holdout. Query-end single-position transport passed 0/9 causal cells; this is not a neuron or complete binding path.",
        "display_priority": 14 + cell["median_min_axis_normalized_roq_norm"],
        "phase398_tested": True,
        "phase398_physical_observational_pass": cell["validation_gate_pass"],
        "phase398_roq_norm": cell["median_min_axis_normalized_roq_norm"],
        "phase398_roq_cross_axis_cosine": cell["median_roq_cross_axis_cosine"],
        "phase398_roq_to_rq_ratio": cell["median_roq_to_rq_norm_ratio"],
        "phase398_causal_gate_pass": causal_cell["causal_cell_gate_pass"],
        "phase398_same_order_answer_switch_rate": causal_cell["same_order_donor_answer_switch_rate"],
        "is_real_unit": False,
        "single_neuron_claim": False,
        "generated_at": updated_at,
        "source_artifacts": ["phase398_order_conditioned_physical_validation.json", "phase398_order_conditioned_causal_analysis.json"],
    }


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
    progress["last_phase"] = "Phase398-OrderConditionedJointBindingStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["order_conditioned_joint_binding_stage"] = {
        "behavior_cases": {"numerator": 3456, "denominator": 3456},
        "qualified_parallel_groups": {"numerator": 68, "denominator": 72},
        "frozen_trace_cases": {"numerator": 2304, "denominator": 2304},
        "order_invariant_RQ_cells": {"numerator": 0, "denominator": 9},
        "ROQ_discovery_cells": {"numerator": 9, "denominator": 9},
        "ROQ_calibration_cells": {"numerator": 9, "denominator": 9},
        "ROQ_physical_cells": {"numerator": 9, "denominator": 9},
        "single_query_position_causal_cells": {"numerator": 0, "denominator": 9},
        "same_order_answer_switches": {"numerator": 10, "denominator": 432},
        "abstract_binding_algorithms": {"numerator": 0, "denominator": 1},
        "complete_language_paths": {"numerator": 0, "denominator": 72},
        "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase398_decision"] = "retain_three_split_order_conditioned_ROQ_anchors_reject_single_query_position_causal_sufficiency"
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase398 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    nodes, edges = graph(stage)
    payloads["phase398_order_conditioned_joint_binding_stage_summary.json"] = stage
    physical_map = read_jsonl(P398 / "phase398_discovery_physical_map.jsonl")
    published = [*payloads, "phase398_discovery_physical_map.jsonl", "phase398_evidence_nodes.jsonl", "phase398_evidence_edges.jsonl"]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase398_discovery_physical_map.jsonl", physical_map)
        write_jsonl(root / "phase398_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase398_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase398-OrderConditionedJointBindingStage"
        manifest["phase398"] = {"status": "ROQ_three_split_replicated_single_query_position_causal_sufficiency_rejected", **stage["results"], "files": published}
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase398-OrderConditionedJointBindingStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        update_checksums(root)
        public_manifest(root, updated_at)

    physical_cells = {(row["model"], row["task_surface"]): row for row in payloads["phase398_order_conditioned_physical_validation.json"]["cells"]}
    causal_cells = {(row["model"], row["task_surface"]): row for row in payloads["phase398_order_conditioned_causal_analysis.json"]["cells"]}
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase398_order_conditioned_joint_binding_stage_summary.json", stage)
        write_jsonl(root / "phase398_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase398_evidence_edges.jsonl", edges)
        for surface, (family_id, _name) in SURFACE_FAMILIES.items():
            for model in MODELS:
                path = root / f"partitions/{family_id}/{model}.json"
                partition = read_json(path)
                retained = [node for node in partition.get("nodes", []) if not node.get("phase398_tested")]
                retained.append(aggregate_node(physical_cells[(model, surface)], causal_cells[(model, surface)], updated_at))
                partition["nodes"] = retained
                partition["mapping_status"] = "phase398_order_conditioned_ROQ_replicated_single_position_sufficiency_rejected"
                partition["generated_at"] = updated_at
                partition.setdefault("metrics", {}).update({
                    "phase398_order_conditioned_anchor_count": 1,
                    "phase398_observational_physical_pass_count": 1,
                    "phase398_single_position_causal_count": 0,
                    "phase398_single_neuron_causal_count": 0,
                })
                partition["evidence_boundary"] = "Phase398 adds one aggregate query-end ROQ anchor. It replicated observationally over three splits but failed single-position causal sufficiency and is not a neuron."
                write_json(path, partition)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 398
        manifest["generated_at"] = updated_at
        manifest["phase398_audit"] = {
            "status": "ROQ_three_split_replicated_single_query_position_causal_sufficiency_rejected",
            "behavior_case_count": 3456,
            "qualified_group_count": 68,
            "observational_pass_cells_per_split": 9,
            "causal_direction_count": 432,
            "causal_pass_cell_count": 0,
            "same_order_answer_switch_count": 10,
            "new_aggregate_state_anchor_count": 9,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase398_order_conditioned_joint_binding_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update({
            "phase398_behavior_case_count": 3456,
            "phase398_qualified_group_count": 68,
            "phase398_ROQ_physical_pass_cell_count": 9,
            "phase398_causal_direction_count": 432,
            "phase398_causal_pass_cell_count": 0,
            "phase398_same_order_answer_switch_count": 10,
            "phase398_aggregate_state_anchor_count": 9,
            "phase398_single_neuron_causal_count": 0,
        })
        manifest.setdefault("files", {})["latest_evidence_summary"] = "phase398_order_conditioned_joint_binding_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase398-OrderConditionedJointBindingStage",
            "statement": "Query-end ROQ trajectories replicated across lexical axes and three splits, but single-position transport passed 0/9 causal cells and switched only 10/432 answers. Phase398 anchors are aggregate dynamic-route observations, not neurons or complete binding paths.",
            "order_conditioned_interaction_available": True,
            "single_position_causal_carrier_available": False,
            "crosssurface_binding_rule_available": False,
            "natural_necessity_available": False,
            "upstream_language_path_available": False,
            "single_unit_causal_closure": False,
            "candidate_depth_specificity_available": True,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        update_checksums(root)
        public_manifest(root, updated_at)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
