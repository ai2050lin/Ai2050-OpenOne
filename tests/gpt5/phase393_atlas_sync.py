#!/usr/bin/env python3
"""Publish Phase390-393 joint-formation and attribute-transport evidence."""

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
P390 = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
P391 = ROOT / "tests/gpt5/result/phase391_local_parent_graph"
P392 = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
P393 = ROOT / "tests/gpt5/result/phase393_attribute_content_holdout"

JSON_SOURCES = {
    "phase390_protocol.json": P390 / "phase390_protocol.json",
    "phase390_behavior_freeze_summary.json": P390
    / "phase390_behavior_freeze_summary.json",
    "phase390_instrument_audit_summary.json": P390
    / "phase390_instrument_audit_summary.json",
    "phase390_joint_contract_amendment.json": P390
    / "phase390_joint_contract_amendment.json",
    "phase390_discovery_candidate_freeze.json": P390
    / "phase390_discovery_candidate_freeze.json",
    "phase391_protocol.json": P391 / "phase391_protocol.json",
    "phase391_discovery_candidate_freeze.json": P391
    / "phase391_discovery_candidate_freeze.json",
    "phase391_calibration_summary.json": P391 / "phase391_calibration_summary.json",
    "phase391_physical_summary.json": P391 / "phase391_physical_summary.json",
    "phase392_protocol.json": P392 / "phase392_protocol.json",
    "phase392_intervention_freeze.json": P392
    / "phase392_intervention_freeze.json",
    "phase392_instrument_audit_summary.json": P392
    / "phase392_instrument_audit_summary.json",
    "phase392_causal_summary.json": P392 / "phase392_causal_summary.json",
    "phase393_protocol.json": P393 / "phase393_protocol.json",
    "phase393_summary.json": P393 / "phase393_summary.json",
}


def evidence_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p390_fresh_joint_denominator",
            "node_type": "fresh_crossmodel_joint_formation_denominator",
            "phase_id": "Phase390",
            "behavior_case_count": 1728,
            "mechanism_count": 6,
            "eligible_mechanism_count": 1,
            "eligible_mechanism": "field_extraction",
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p390_global_terminal_alignment_negative",
            "node_type": "global_linear_terminal_alignment_negative",
            "phase_id": "Phase390",
            "model_candidate_count": 144,
            "crossmodel_candidate_count": 48,
            "passing_crossmodel_candidate_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p391_local_parent_layout",
            "node_type": "replicated_graph_local_parent_layout",
            "phase_id": "Phase391",
            "receiver_coordinate": "query_integrated",
            "anchor_fraction": 0.5714285714285714,
            "discovery_models_passing": 3,
            "calibration_models_passing": 3,
            "physical_models_passing": 3,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p392_joint_parent_boundary_replay",
            "node_type": "graph_consistent_multi_source_parent_boundary_replay",
            "phase_id": "Phase392",
            "causal_direction_count": 144,
            "joint_answer_switch_count": 139,
            "models_passing_joint_specificity": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p392_multi_source_joint_path_negative",
            "node_type": "multi_source_joint_specificity_negative",
            "phase_id": "Phase392",
            "attribute_only_explains_joint_effect": True,
            "wrong_depth_often_matches_candidate_depth": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p393_independent_attribute_holdout",
            "node_type": "independent_attribute_transport_holdout",
            "phase_id": "Phase393",
            "group_count": 12,
            "direction_count": 72,
            "phase392_group_overlap": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p393_attribute_content_transport",
            "node_type": "crossmodel_graph_consistent_attribute_content_transport",
            "phase_id": "Phase393",
            "attribute_answer_switch_count": 71,
            "direction_count": 72,
            "structure_answer_switch_count": 0,
            "random_answer_switch_count": 0,
            "models_passing": 3,
            "causal": True,
            "causal_scope": "controlled_attribute_state_transport",
            "language_path": False,
        },
        {
            "node_id": "p393_depth_specificity_negative",
            "node_type": "candidate_depth_specificity_negative",
            "phase_id": "Phase393",
            "wrong_depth_attribute_switch_count": 72,
            "direction_count": 72,
            "models_passing_depth_specificity": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p393_neuron_scan_boundary",
            "node_type": "strict_neuron_promotion_boundary",
            "phase_id": "Phase393",
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p389_negative_to_p390_joint_denominator",
            "source_node_id": "p389_crossmodel_source_specificity_negative",
            "target_node_id": "p390_fresh_joint_denominator",
            "edge_type": "replaces_single_source_patch_with_fresh_joint_contract",
            "phase_id": "Phase390",
            "causal_path": False,
        },
        {
            "edge_id": "p390_denominator_to_global_negative",
            "source_node_id": "p390_fresh_joint_denominator",
            "target_node_id": "p390_global_terminal_alignment_negative",
            "edge_type": "tests_and_rejects_crosslayer_direction_conservation",
            "phase_id": "Phase390",
            "causal_path": False,
        },
        {
            "edge_id": "p390_global_to_p391_local_parent",
            "source_node_id": "p390_global_terminal_alignment_negative",
            "target_node_id": "p391_local_parent_layout",
            "edge_type": "replaces_terminal_alignment_with_direct_parent_conservation",
            "phase_id": "Phase391",
            "causal_path": False,
        },
        {
            "edge_id": "p391_local_parent_to_p392_replay",
            "source_node_id": "p391_local_parent_layout",
            "target_node_id": "p392_joint_parent_boundary_replay",
            "edge_type": "authorizes_one_boundary_replay_with_natural_recomputation",
            "phase_id": "Phase392",
            "causal_path": False,
        },
        {
            "edge_id": "p392_replay_to_joint_negative",
            "source_node_id": "p392_joint_parent_boundary_replay",
            "target_node_id": "p392_multi_source_joint_path_negative",
            "edge_type": "fails_attribute_only_and_wrong_depth_specificity_controls",
            "phase_id": "Phase392",
            "causal_path": False,
        },
        {
            "edge_id": "p392_negative_to_p393_holdout",
            "source_node_id": "p392_multi_source_joint_path_negative",
            "target_node_id": "p393_independent_attribute_holdout",
            "edge_type": "freezes_attribute_only_hypothesis_on_unused_groups",
            "phase_id": "Phase393",
            "causal_path": False,
        },
        {
            "edge_id": "p393_holdout_to_attribute_transport",
            "source_node_id": "p393_independent_attribute_holdout",
            "target_node_id": "p393_attribute_content_transport",
            "edge_type": "controlled_attribute_state_transport",
            "phase_id": "Phase393",
            "causal_path": True,
            "complete_language_path": False,
        },
        {
            "edge_id": "p393_transport_to_depth_negative",
            "source_node_id": "p393_attribute_content_transport",
            "target_node_id": "p393_depth_specificity_negative",
            "edge_type": "transport_replication_does_not_identify_specialized_depth",
            "phase_id": "Phase393",
            "causal_path": False,
        },
        {
            "edge_id": "p393_depth_negative_to_neuron_boundary",
            "source_node_id": "p393_depth_specificity_negative",
            "target_node_id": "p393_neuron_scan_boundary",
            "edge_type": "blocks_single_neuron_promotion_before_binding_specificity",
            "phase_id": "Phase393",
            "causal_path": False,
        },
    ]
    return nodes, edges


def stage_summary(updated_at: str) -> dict[str, Any]:
    return {
        "schema_version": "67.3.0",
        "phase_id": "Phase393-StageMerge",
        "created_at": updated_at,
        "objective": "separate_joint_formation_structure_from_transportable_attribute_content",
        "assessment": {
            "global_terminal_direction_conservation_supported": False,
            "graph_local_parent_layout_physically_replicated": True,
            "multi_source_joint_path_established": False,
            "attribute_content_transport_established": True,
            "candidate_depth_specificity_established": False,
            "field_extraction_algorithm_closed": False,
            "complete_language_path_available": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "phase390_behavior_cases": 1728,
            "phase390_registered_mechanisms": 6,
            "phase390_behavior_eligible_mechanisms": 1,
            "phase390_crossmodel_global_candidates": 48,
            "phase391_physical_local_parent_candidates": 1,
            "phase392_causal_directions": 144,
            "phase393_independent_directions": 72,
        },
        "results": {
            "phase390_passing_global_candidates": 0,
            "phase391_physical_local_parent_candidates": 1,
            "phase392_joint_answer_switch_count": 139,
            "phase392_models_passing_joint_specificity": 0,
            "phase393_attribute_answer_switch_count": 71,
            "phase393_structure_answer_switch_count": 0,
            "phase393_random_answer_switch_count": 0,
            "phase393_wrong_depth_attribute_switch_count": 72,
            "models_passing_attribute_transport": 3,
            "models_passing_depth_specificity": 0,
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
        },
        "hard_limits": [
            "only_field_extraction_passed_the_six_mechanism_behavior_denominator",
            "crosslayer_terminal_direction_conservation_is_not_supported",
            "local_parent_layout_is_predictive_structure_not_a_complete_causal_path",
            "joint_replay_is_explained_by_attribute_positions_without_multi_source_advantage",
            "attribute_transport_also_switches_at_wrong_depth_in_all_seventy_two_directions",
            "all_attribute_tokens_are_a_large_distributed_intervention_not_a_minimal_unit",
            "the_paired_prompt_exposes_both_target_and_rejected_attribute_values",
            "small_model_results_do_not_establish_large_model_structure",
        ],
        "authorization": {
            "show_local_parent_layout": True,
            "show_attribute_content_transport_edge": True,
            "show_depth_specialized_path": False,
            "show_multi_source_joint_path": False,
            "show_complete_language_path": False,
            "show_specific_neuron_path": False,
            "run_unbounded_neuron_scan": False,
        },
        "next_stage": {
            "objective": "separate_generic_content_overwrite_from_relation_binding_and_natural_routing",
            "automatic_continuation_authorized": False,
            "reason": (
                "Phase393 completes the attribute-transport calibration stage. A new "
                "token-identity-held-fixed binding denominator is required before "
                "another causal or neuron-level experiment."
            ),
        },
        "single_global_progress_percentage_valid": False,
    }


def update_neuron_atlas(
    stage: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    updated_at: str,
) -> None:
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase393_joint_formation_stage_summary.json", stage)
        write_jsonl(root / "phase393_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase393_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 393
        manifest["generated_at"] = updated_at
        manifest["phase393_audit"] = {
            "status": "attribute_transport_positive_depth_specificity_negative",
            "attribute_answer_switch_count": 71,
            "attribute_direction_count": 72,
            "wrong_depth_attribute_switch_count": 72,
            "models_passing_attribute_transport": 3,
            "models_passing_depth_specificity": 0,
            "new_neuron_path_nodes_promoted": 0,
            "single_unit_causal_count": 0,
            "language_path_count": 0,
            "source": "phase393_joint_formation_stage_summary.json",
        }
        metrics = manifest.setdefault("metrics", {})
        metrics["phase390_behavior_case_count"] = 1728
        metrics["phase390_behavior_eligible_mechanism_count"] = 1
        metrics["phase391_physical_local_parent_layout_count"] = 1
        metrics["phase392_joint_answer_switch_count"] = 139
        metrics["phase392_joint_specificity_pass_model_count"] = 0
        metrics["phase393_attribute_answer_switch_count"] = 71
        metrics["phase393_attribute_direction_count"] = 72
        metrics["phase393_wrong_depth_attribute_switch_count"] = 72
        metrics["phase393_attribute_transport_pass_model_count"] = 3
        metrics["phase393_depth_specificity_pass_model_count"] = 0
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase393_joint_formation_stage_summary.json"
        boundary = manifest.setdefault("evidence_boundary", {})
        boundary["latest_phase"] = "Phase393-AttributeContentHoldout"
        boundary["statement"] = (
            "A graph-consistent attribute-state intervention switched 71 of 72 "
            "independent directions while structure and random controls switched "
            "zero. The same intervention at a wrong depth switched all 72, so this "
            "is content transport, not a specialized natural path. No neuron path "
            "is promoted."
        )
        boundary["graph_local_parent_layout_available"] = True
        boundary["attribute_content_transport_available"] = True
        boundary["candidate_depth_specificity_available"] = False
        boundary["multi_source_joint_path_available"] = False
        boundary["upstream_language_path_available"] = False
        boundary["single_unit_causal_closure"] = False
        write_json(root / "manifest.json", manifest)
        checksum_path = root / "checksums.json"
        if checksum_path.is_file():
            write_json(
                checksum_path,
                {
                    "schema_version": "artifact_checksums.v1",
                    "files": [
                        {"path": str(item.relative_to(root)), "sha256": sha256(item)}
                        for item in sorted(root.rglob("*"))
                        if item.is_file() and item != checksum_path
                    ],
                },
            )
        public_manifest(root, updated_at)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase390-393 public artifacts: {missing}")

    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at)
    nodes, edges = evidence_graph()
    payloads["phase393_joint_formation_stage_summary.json"] = stage
    published = [*payloads.keys(), "phase393_evidence_nodes.jsonl", "phase393_evidence_edges.jsonl"]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase393_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase393_evidence_edges.jsonl", edges)

        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase393-StageMerge"
        manifest["phase390_393"] = {
            "status": "local_parent_layout_and_attribute_transport_separated",
            "behavior_case_count": 1728,
            "behavior_eligible_mechanism_count": 1,
            "global_terminal_alignment_candidate_count": 0,
            "physical_local_parent_layout_count": 1,
            "joint_answer_switch_count": 139,
            "models_passing_joint_specificity": 0,
            "attribute_answer_switch_count": 71,
            "attribute_direction_count": 72,
            "wrong_depth_attribute_switch_count": 72,
            "models_passing_attribute_transport": 3,
            "models_passing_depth_specificity": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published,
        }
        write_json(root / "manifest.json", manifest)

        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase393-StageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["joint_formation_stage"] = {
            "behavior_cases": {"numerator": 1728, "denominator": 1728},
            "behavior_eligible_mechanisms": {"numerator": 1, "denominator": 6},
            "global_terminal_alignment_candidates": {"numerator": 0, "denominator": 48},
            "physical_local_parent_layouts": {"numerator": 1, "denominator": 1},
            "joint_causal_directions": {"numerator": 144, "denominator": 144},
            "joint_answer_switches": {"numerator": 139, "denominator": 144},
            "models_passing_joint_specificity": {"numerator": 0, "denominator": 3},
            "independent_attribute_directions": {"numerator": 72, "denominator": 72},
            "attribute_answer_switches": {"numerator": 71, "denominator": 72},
            "structure_answer_switches": {"numerator": 0, "denominator": 72},
            "random_answer_switches": {"numerator": 0, "denominator": 72},
            "wrong_depth_attribute_switches": {"numerator": 72, "denominator": 72},
            "models_passing_attribute_transport": {"numerator": 3, "denominator": 3},
            "models_passing_depth_specificity": {"numerator": 0, "denominator": 3},
            "complete_language_paths": {"numerator": 0, "denominator": 72},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
        }
        progress["phase393_decision"] = (
            "freeze_attribute_transport_as_depth_nonspecific_and_design_binding_specificity"
        )
        write_json(root / "progress.json", progress)

        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase393-StageMerge"
            client_index["latest_stage_files"] = [
                "phase393_joint_formation_stage_summary.json",
                "phase393_evidence_nodes.jsonl",
                "phase393_evidence_edges.jsonl",
                "phase393_summary.json",
                "phase392_causal_summary.json",
                "phase391_physical_summary.json",
                "phase390_discovery_candidate_freeze.json",
            ]
            initial = client_index.setdefault("initial_files", [])
            for name in client_index["latest_stage_files"]:
                if name not in initial:
                    initial.append(name)
            write_json(client_index_path, client_index)
        public_manifest(root, updated_at)

    update_neuron_atlas(stage, nodes, edges, updated_at)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
