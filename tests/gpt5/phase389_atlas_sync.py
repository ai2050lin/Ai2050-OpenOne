#!/usr/bin/env python3
"""Publish Phase387-389 order, intervention, and source-specificity evidence."""

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
from phase381_joint_state_case_bank import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
P387 = ROOT / "tests/gpt5/result/phase387_computational_order_audit"
P388 = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
P389 = ROOT / "tests/gpt5/result/phase389_head_source_decomposition"

JSON_SOURCES = {
    "phase387_computational_order_contract.json": P387
    / "phase387_computational_order_contract.json",
    "phase387_summary.json": P387 / "phase387_summary.json",
    "phase388_interface_amendment.json": P388 / "phase388_interface_amendment.json",
    "phase388_runtime_amendment.json": P388 / "phase388_runtime_amendment.json",
    "phase388_intervention_freeze.json": P388 / "phase388_intervention_freeze.json",
    "phase388_instrument_audit_summary.json": P388
    / "phase388_instrument_audit_summary.json",
    "phase388_causal_summary.json": P388 / "phase388_causal_summary.json",
    "phase389_summary.json": P389 / "phase389_summary.json",
}

JSONL_SOURCES = {
    "phase387_candidate_order_rows.jsonl": P387 / "phase387_candidate_order_rows.jsonl",
    "phase388_model_causal_rows.jsonl": P388 / "phase388_model_causal_rows.jsonl",
    "phase389_head_candidate_rows.jsonl": P389 / "phase389_head_candidate_rows.jsonl",
    "phase389_role_candidate_rows.jsonl": P389 / "phase389_role_candidate_rows.jsonl",
    "phase389_source_anchor_specificity_rows.jsonl": P389
    / "phase389_source_anchor_specificity_rows.jsonl",
}


def evidence_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p387_computational_order_audit",
            "node_type": "computational_partial_order_audit",
            "phase_id": "Phase387",
            "predictive_trajectory_count": 10,
            "direct_computational_edge_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p388_runtime_contract_correction",
            "node_type": "single_sample_native_interface_correction",
            "phase_id": "Phase388",
            "glm4_batched_exact_count": 18,
            "glm4_single_sample_exact_count": 48,
            "denominator": 48,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p388_source_kv_intervention_denominator",
            "node_type": "fresh_bidirectional_source_kv_intervention",
            "phase_id": "Phase388",
            "group_count": 16,
            "direction_count": 96,
            "scenario_count": 672,
            "holdout_reuse_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p388_local_query_shift",
            "node_type": "local_receiver_perturbation_without_specific_transport",
            "phase_id": "Phase388",
            "query_gate_models": 0,
            "margin_gate_models": 0,
            "behavior_gate_models": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p388_coarse_kv_path_negative",
            "node_type": "strict_causal_negative",
            "phase_id": "Phase388",
            "strict_donor_answer_switch_count": 0,
            "direction_count": 96,
            "source_kv_causal_path_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p389_distributed_head_source_activity",
            "node_type": "broad_descriptive_head_source_distribution",
            "phase_id": "Phase389",
            "all_head_count": 92,
            "replicated_head_count": 56,
            "replicated_role_count": 92,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p389_crossmodel_source_specificity_negative",
            "node_type": "strict_crossmodel_source_anchor_negative",
            "phase_id": "Phase389",
            "qwen3_specific_head_count": 0,
            "glm4_specific_head_count": 2,
            "deepseek7b_specific_head_count": 0,
            "crossmodel_specific_route_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p389_neuron_scan_boundary",
            "node_type": "strict_negative_boundary",
            "phase_id": "Phase389",
            "complete_language_path_count": 0,
            "single_neuron_causal_path_count": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p386_predictive_to_p387_order_audit",
            "source_node_id": "p386_physical_predictive_relations",
            "target_node_id": "p387_computational_order_audit",
            "edge_type": "predictive_relation_requires_computational_orientation",
            "phase_id": "Phase387",
            "causal_path": False,
        },
        {
            "edge_id": "p387_order_to_p388_runtime",
            "source_node_id": "p387_computational_order_audit",
            "target_node_id": "p388_runtime_contract_correction",
            "edge_type": "requires_fresh_causally_admissible_single_sample_path",
            "phase_id": "Phase388",
            "causal_path": False,
        },
        {
            "edge_id": "p388_runtime_to_kv_denominator",
            "source_node_id": "p388_runtime_contract_correction",
            "target_node_id": "p388_source_kv_intervention_denominator",
            "edge_type": "qualifies_fresh_intervention_denominator",
            "phase_id": "Phase388",
            "causal_path": False,
        },
        {
            "edge_id": "p388_kv_to_local_shift",
            "source_node_id": "p388_source_kv_intervention_denominator",
            "target_node_id": "p388_local_query_shift",
            "edge_type": "measures_small_receiver_shift_without_control_specificity",
            "phase_id": "Phase388",
            "causal_path": False,
        },
        {
            "edge_id": "p388_local_to_causal_negative",
            "source_node_id": "p388_local_query_shift",
            "target_node_id": "p388_coarse_kv_path_negative",
            "edge_type": "fails_margin_behavior_and_crossmodel_control_gates",
            "phase_id": "Phase388",
            "causal_path": False,
        },
        {
            "edge_id": "p388_negative_to_p389_decomposition",
            "source_node_id": "p388_coarse_kv_path_negative",
            "target_node_id": "p389_distributed_head_source_activity",
            "edge_type": "decomposes_existing_exact_events_without_new_model_run",
            "phase_id": "Phase389",
            "causal_path": False,
        },
        {
            "edge_id": "p389_distribution_to_specificity_negative",
            "source_node_id": "p389_distributed_head_source_activity",
            "target_node_id": "p389_crossmodel_source_specificity_negative",
            "edge_type": "matched_source_role_specificity_rejects_crossmodel_route",
            "phase_id": "Phase389",
            "causal_path": False,
        },
        {
            "edge_id": "p389_specificity_to_neuron_boundary",
            "source_node_id": "p389_crossmodel_source_specificity_negative",
            "target_node_id": "p389_neuron_scan_boundary",
            "edge_type": "does_not_authorize_head_or_neuron_scan",
            "phase_id": "Phase389",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_neuron_atlas(
    stage: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    updated_at: str,
) -> None:
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase389_causal_order_stage_summary.json", stage)
        write_jsonl(root / "phase389_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase389_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 389
        manifest["generated_at"] = updated_at
        manifest["phase389_audit"] = {
            "status": "coarse_source_kv_path_and_crossmodel_source_specific_head_rejected",
            "predictive_trajectory_count": 10,
            "direct_computational_edge_count": 0,
            "causal_direction_count": 96,
            "strict_answer_switch_count": 0,
            "crossmodel_source_specific_head_route_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "single_unit_causal_count": 0,
            "language_path_count": 0,
            "source": "phase389_causal_order_stage_summary.json",
        }
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase389_causal_order_stage_summary.json"
        boundary = manifest.setdefault("evidence_boundary", {})
        boundary["latest_phase"] = "Phase389-HeadSourceDecomposition"
        boundary["statement"] = (
            "Phase386 predictive trajectories remain descriptive. None is a direct "
            "same-layer computational edge. A fresh 96-direction source K/V test "
            "produced zero answer switches, and no source-anchor-specific head route "
            "replicated across all three models. No neuron path is promoted."
        )
        boundary["direct_computational_edge_count"] = 0
        boundary["source_kv_causal_path_count"] = 0
        boundary["crossmodel_source_specific_head_route_count"] = 0
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
    missing = [
        str(path)
        for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase387-389 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = {
        "schema_version": "63.2.0",
        "phase_id": "Phase389-StageMerge",
        "created_at": updated_at,
        "objective": "orient_predictive_relations_and_test_causally_admissible_source_transport",
        "assessment": {
            "phase386_predictive_relations_retained": True,
            "predictive_relations_are_direct_computational_edges": False,
            "fresh_source_kv_intervention_completed": True,
            "local_receiver_shift_detected": True,
            "source_specific_transport_detected": False,
            "behavior_mediation_detected": False,
            "crossmodel_source_specific_head_route_detected": False,
            "causal_language_path_available": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "phase386_physical_predictive_trajectories": 10,
            "phase388_candidate_behavior_cases": 144,
            "phase388_qualified_groups": 22,
            "phase388_causal_test_groups": 16,
            "phase388_causal_directions": 96,
            "phase388_intervention_scenarios": 672,
            "phase389_all_heads": 92,
        },
        "results": {
            "direct_computational_edge_count": 0,
            "strict_donor_answer_switch_count": 0,
            "models_passing_query_gate": 0,
            "models_passing_margin_gate": 0,
            "models_passing_behavior_gate": 0,
            "replicated_descriptive_head_count": 56,
            "crossmodel_source_specific_head_route_count": 0,
            "single_neuron_causal_path_count": 0,
            "complete_language_path_count": 0,
        },
        "hard_limits": [
            "semantic_coordinate_adjacency_is_not_computational_adjacency",
            "batched_left_padding_is_not_a_qualified_glm4_runtime_path",
            "single_source_position_kv_transfer_does_not_beat_controls_consistently",
            "zero_of_ninety_six_main_patches_switch_the_answer",
            "head_relations_are_broadly_distributed_across_heads_and_source_roles",
            "source_anchor_specificity_does_not_replicate_across_models",
            "phase389_head_source_analysis_has_no_fresh_independent_holdout",
            "small_model_results_do_not_establish_large_model_structure",
        ],
        "authorization": {
            "show_phase386_relations_as_predictive_trajectories": True,
            "show_direct_causal_edge": False,
            "show_specific_crossmodel_head_route": False,
            "show_specific_neuron_path": False,
            "run_unbounded_neuron_scan": False,
            "reuse_phase386_or_phase388_denominators": False,
        },
        "next_stage": {
            "objective": "freeze_a_multi_position_multi_head_cross_layer_joint_state_contract",
            "automatic_continuation_authorized": False,
            "reason": (
                "The current single-edge hypothesis chain is exhausted. A new joint "
                "state denominator and conservation contract must be designed before "
                "additional model execution."
            ),
        },
        "single_global_progress_percentage_valid": False,
    }
    nodes, edges = evidence_graph()
    payloads["phase389_causal_order_stage_summary.json"] = stage
    row_payloads["phase389_evidence_nodes.jsonl"] = nodes
    row_payloads["phase389_evidence_edges.jsonl"] = edges
    published = [*payloads.keys(), *row_payloads.keys()]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase389-StageMerge"
        manifest["phase389"] = {
            "status": "predictive_trajectories_oriented_coarse_kv_path_rejected",
            "predictive_trajectory_count": 10,
            "direct_computational_edge_count": 0,
            "causal_direction_count": 96,
            "strict_answer_switch_count": 0,
            "replicated_descriptive_head_count": 56,
            "crossmodel_source_specific_head_route_count": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published,
        }
        write_json(root / "manifest.json", manifest)

        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase389-StageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["causal_order_kv_stage"] = {
            "physical_predictive_trajectories": {"numerator": 10, "denominator": 10},
            "direct_computational_edges": {"numerator": 0, "denominator": 10},
            "causal_intervention_directions": {"numerator": 96, "denominator": 96},
            "strict_answer_switches": {"numerator": 0, "denominator": 96},
            "models_passing_query_gate": {"numerator": 0, "denominator": 3},
            "models_passing_margin_gate": {"numerator": 0, "denominator": 3},
            "models_passing_behavior_gate": {"numerator": 0, "denominator": 3},
            "crossmodel_source_specific_head_routes": {"numerator": 0, "denominator": 3},
            "complete_language_paths": {"numerator": 0, "denominator": 72},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
        }
        progress["phase389_decision"] = (
            "stop_single_edge_patching_and_design_joint_multi_position_multi_head_state"
        )
        write_json(root / "progress.json", progress)

        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase389-StageMerge"
            client_index["latest_stage_files"] = [
                "phase389_causal_order_stage_summary.json",
                "phase389_evidence_nodes.jsonl",
                "phase389_evidence_edges.jsonl",
                "phase388_causal_summary.json",
                "phase389_summary.json",
                "phase387_summary.json",
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
