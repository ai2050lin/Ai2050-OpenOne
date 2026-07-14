#!/usr/bin/env python3
"""Publish Phase412 protocol evidence without promoting physical nodes."""

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
P412 = ROOT / "tests/gpt5/result/phase412_typed_quotient_preflight"
JSON_SOURCES = {
    "phase412_typed_observer_covariance_audit.json": P412
    / "phase412_typed_observer_covariance_audit.json",
    "phase412_nontrivial_quotient_audit.json": P412
    / "phase412_nontrivial_quotient_audit.json",
    "phase412_irreversible_operation_readiness.json": P412
    / "phase412_irreversible_operation_readiness.json",
    "phase412_typed_composition_readiness.json": P412
    / "phase412_typed_composition_readiness.json",
    "phase412_qualification.json": P412 / "phase412_qualification.json",
    "phase412_stage_summary.json": P412 / "phase412_stage_summary.json",
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


def evidence_graph(stage: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    denominator = stage["denominators"]
    results = stage["results"]
    nodes = [
        {
            "node_id": "p412_fixed_observer_counterexample_reaudit",
            "node_type": "fixed_observer_role_transport_mismatch_reaudit",
            "phase_id": "Phase412",
            "fixed_observer_unstable_cell_count": results[
                "fixed_observer_unstable_cell_count"
            ],
            "role_moved_cell_count": results["role_moved_cell_count"],
            "explained_by_role_transport_count": results[
                "fixed_instability_explained_by_role_transport_count"
            ],
            "refutes_all_role_conditioned_states": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p412_typed_observer_covariance",
            "node_type": "state_observer_response_typed_covariance_contract",
            "phase_id": "Phase412",
            "observer_operation_cell_count": denominator[
                "registered_query_observer_operation_cell_count"
            ],
            "typed_unstable_cell_count": results[
                "typed_observer_unstable_cell_count"
            ],
            "observer_action_composition_case_count": denominator[
                "observer_action_composition_case_count"
            ],
            "observer_action_composition_failure_count": results[
                "observer_action_composition_failure_count"
            ],
            "model_covariance": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p412_exhaustive_nontrivial_quotient_null",
            "node_type": "exhaustive_global_nontrivial_finite_quotient_audit",
            "phase_id": "Phase412",
            "partition_count": denominator["finite_partition_count"],
            "nontrivial_partition_count": denominator[
                "nontrivial_partition_count"
            ],
            "full_operation_congruent_nontrivial_count": results[
                "full_operation_congruent_nontrivial_partition_count"
            ],
            "global_qualifying_nontrivial_count": results[
                "global_nontrivial_qualifying_partition_count"
            ],
            "model_derived": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p412_role_indexed_external_partition_bundle",
            "node_type": "knowledge_role_indexed_external_observation_partition_bundle",
            "phase_id": "Phase412",
            "role_conditioned_quotient_count": results[
                "external_role_conditioned_quotient_count"
            ],
            "role_indexed_bundle_count": results[
                "external_role_indexed_partition_bundle_count"
            ],
            "global_state_quotient": False,
            "model_derived": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p412_irreversible_and_typed_bridge_gate",
            "node_type": "missing_state_universe_and_cross_family_bridge_readiness_gate",
            "phase_id": "Phase412",
            "proposed_irreversible_operation_count": denominator[
                "proposed_irreversible_operation_count"
            ],
            "registered_irreversible_operation_count": results[
                "registered_executable_irreversible_operation_count"
            ],
            "proposed_cross_family_bridge_count": denominator[
                "proposed_cross_family_bridge_count"
            ],
            "registered_cross_family_bridge_count": results[
                "registered_executable_cross_family_bridge_count"
            ],
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p412_execution_boundary",
            "node_type": "external_review_collector_and_semantic_universe_execution_boundary",
            "phase_id": "Phase412",
            "completed_external_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "required_external_reviewer_count": denominator[
                "required_independent_reviewer_count"
            ],
            "sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "sealed_model_collector_denominator": denominator[
                "future_sealed_model_collector_case_count"
            ],
            "model_execution_authorized": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p411_partition_stability_to_p412_reaudit",
            "source_node_id": "p411_partition_operation_stability",
            "target_node_id": "p412_fixed_observer_counterexample_reaudit",
            "edge_type": "retests_fixed_observer_failures_with_explicit_query_role_transport",
            "phase_id": "Phase412",
            "causal_path": False,
        },
        {
            "edge_id": "p412_reaudit_to_typed_covariance",
            "source_node_id": "p412_fixed_observer_counterexample_reaudit",
            "target_node_id": "p412_typed_observer_covariance",
            "edge_type": "replaces_untyped_fixed_observer_test_with_state_observer_response_covariance",
            "phase_id": "Phase412",
            "causal_path": False,
        },
        {
            "edge_id": "p412_covariance_to_global_quotient_null",
            "source_node_id": "p412_typed_observer_covariance",
            "target_node_id": "p412_exhaustive_nontrivial_quotient_null",
            "edge_type": "exhausts_global_partitions_after_typed_covariance_correction",
            "phase_id": "Phase412",
            "causal_path": False,
        },
        {
            "edge_id": "p412_global_null_to_role_bundle",
            "source_node_id": "p412_exhaustive_nontrivial_quotient_null",
            "target_node_id": "p412_role_indexed_external_partition_bundle",
            "edge_type": "separates_missing_global_quotient_from_valid_role_conditioned_external_bundle",
            "phase_id": "Phase412",
            "causal_path": False,
        },
        {
            "edge_id": "p412_role_bundle_to_irreversible_gate",
            "source_node_id": "p412_role_indexed_external_partition_bundle",
            "target_node_id": "p412_irreversible_and_typed_bridge_gate",
            "edge_type": "blocks_unreviewed_irreversible_and_cross_family_structure_invention",
            "phase_id": "Phase412",
            "causal_path": False,
        },
        {
            "edge_id": "p412_readiness_to_execution_boundary",
            "source_node_id": "p412_irreversible_and_typed_bridge_gate",
            "target_node_id": "p412_execution_boundary",
            "edge_type": "keeps_cuda_physical_causal_and_neuron_execution_closed",
            "phase_id": "Phase412",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_metrics(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    denominator = stage["denominators"]
    results = stage["results"]
    manifest.setdefault("metrics", {}).update(
        {
            "phase412_observer_operation_cell_count": denominator[
                "registered_query_observer_operation_cell_count"
            ],
            "phase412_fixed_observer_unstable_cell_count": results[
                "fixed_observer_unstable_cell_count"
            ],
            "phase412_role_moved_cell_count": results["role_moved_cell_count"],
            "phase412_role_transport_explained_cell_count": results[
                "fixed_instability_explained_by_role_transport_count"
            ],
            "phase412_typed_observer_unstable_cell_count": results[
                "typed_observer_unstable_cell_count"
            ],
            "phase412_observer_action_composition_case_count": denominator[
                "observer_action_composition_case_count"
            ],
            "phase412_observer_action_composition_failure_count": results[
                "observer_action_composition_failure_count"
            ],
            "phase412_finite_partition_count": denominator["finite_partition_count"],
            "phase412_nontrivial_partition_count": denominator[
                "nontrivial_partition_count"
            ],
            "phase412_global_qualifying_nontrivial_partition_count": results[
                "global_nontrivial_qualifying_partition_count"
            ],
            "phase412_role_conditioned_quotient_count": results[
                "external_role_conditioned_quotient_count"
            ],
            "phase412_role_indexed_partition_bundle_count": results[
                "external_role_indexed_partition_bundle_count"
            ],
            "phase412_registered_irreversible_operation_count": results[
                "registered_executable_irreversible_operation_count"
            ],
            "phase412_registered_cross_family_bridge_count": results[
                "registered_executable_cross_family_bridge_count"
            ],
            "phase412_completed_external_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "phase412_required_external_reviewer_count": denominator[
                "required_independent_reviewer_count"
            ],
            "phase412_sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "phase412_model_case_count": denominator["model_case_count_consumed"],
            "phase412_physical_case_count": denominator[
                "physical_case_count_consumed"
            ],
            "phase412_new_neuron_node_count": results["new_neuron_path_count"],
        }
    )


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    denominator = stage["denominators"]
    results = stage["results"]
    progress["last_phase"] = "Phase412-TypedObserverQuotientPreflightStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["typed_observer_quotient_preflight_stage"] = {
        "typed_observer_covariance_cells": {
            "numerator": denominator["registered_query_observer_operation_cell_count"]
            - results["typed_observer_unstable_cell_count"],
            "denominator": denominator[
                "registered_query_observer_operation_cell_count"
            ],
        },
        "fixed_failures_explained_by_role_transport": {
            "numerator": results[
                "fixed_instability_explained_by_role_transport_count"
            ],
            "denominator": results["fixed_observer_unstable_cell_count"],
        },
        "observer_action_compositions": {
            "numerator": denominator["observer_action_composition_case_count"]
            - results["observer_action_composition_failure_count"],
            "denominator": denominator["observer_action_composition_case_count"],
        },
        "finite_partitions_exhausted": {
            "numerator": denominator["finite_partition_count"],
            "denominator": denominator["finite_partition_count"],
        },
        "global_nontrivial_qualifying_partitions": {
            "numerator": results["global_nontrivial_qualifying_partition_count"],
            "denominator": denominator["nontrivial_partition_count"],
        },
        "external_role_indexed_partition_bundles": {
            "numerator": results["external_role_indexed_partition_bundle_count"],
            "denominator": 1,
        },
        "registered_irreversible_operations": {
            "numerator": results[
                "registered_executable_irreversible_operation_count"
            ],
            "denominator": denominator["proposed_irreversible_operation_count"],
        },
        "registered_cross_family_bridges": {
            "numerator": results[
                "registered_executable_cross_family_bridge_count"
            ],
            "denominator": denominator["proposed_cross_family_bridge_count"],
        },
        "independent_external_reviewers": {
            "numerator": results["completed_external_reviewer_count"],
            "denominator": denominator["required_independent_reviewer_count"],
        },
        "sealed_model_collector_equivalence": {
            "numerator": results["sealed_model_collector_equivalence_case_count"],
            "denominator": denominator["future_sealed_model_collector_case_count"],
        },
        "model_cases_consumed": {"numerator": 0, "denominator": 165},
        "physical_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase412_decision"] = (
        "publish_typed_covariance_and_exhaustive_finite_quotient_protocol_only;_"
        "keep_model_physical_causal_and_neuron_gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase412 preflight artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    stage = payloads["phase412_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        "phase412_evidence_nodes.jsonl",
        "phase412_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase412_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase412_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase412-TypedObserverQuotientPreflightStage"
        manifest["phase412"] = {
            "status": "typed_finite_protocol_pass_external_and_model_gates_closed",
            "files": published,
            "machine_preflight_pass": stage["assessment"]["machine_preflight_pass"],
            "global_nontrivial_quotient_count": stage["results"][
                "global_nontrivial_qualifying_partition_count"
            ],
            "external_role_indexed_bundle_count": stage["results"][
                "external_role_indexed_partition_bundle_count"
            ],
            "model_case_count": 0,
            "physical_case_count": 0,
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase412-TypedObserverQuotientPreflightStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase412_stage_summary.json", stage)
        write_jsonl(root / "phase412_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase412_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 412
        manifest["generated_at"] = updated_at
        manifest["phase412_audit"] = {
            "status": "typed_protocol_only_no_model_physical_or_neuron_evidence",
            "observer_operation_cell_count": stage["denominators"][
                "registered_query_observer_operation_cell_count"
            ],
            "fixed_observer_unstable_cell_count": stage["results"][
                "fixed_observer_unstable_cell_count"
            ],
            "role_transport_explained_cell_count": stage["results"][
                "fixed_instability_explained_by_role_transport_count"
            ],
            "typed_observer_unstable_cell_count": stage["results"][
                "typed_observer_unstable_cell_count"
            ],
            "global_nontrivial_quotient_count": stage["results"][
                "global_nontrivial_qualifying_partition_count"
            ],
            "external_role_indexed_bundle_count": stage["results"][
                "external_role_indexed_partition_bundle_count"
            ],
            "model_case_count": 0,
            "physical_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase412_stage_summary.json",
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        public_manifest(root, updated_at)
        update_checksums(root)

    print(
        json.dumps(
            {
                "valid": True,
                "phase_id": stage["phase_id"],
                "published_files": published,
                "evidence_node_count": len(nodes),
                "evidence_edge_count": len(edges),
                "model_case_count": 0,
                "physical_or_neuron_nodes_promoted": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
