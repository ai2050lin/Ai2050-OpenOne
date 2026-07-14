#!/usr/bin/env python3
"""Publish Phase411 protocol evidence without promoting physical nodes."""

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
P411 = ROOT / "tests/gpt5/result/phase411_finite_operation_preflight"
JSON_SOURCES = {
    "phase411_registered_semantic_dual_channel_audit.json": P411
    / "phase411_registered_semantic_dual_channel_audit.json",
    "phase411_finite_operation_closure_audit.json": P411
    / "phase411_finite_operation_closure_audit.json",
    "phase411_external_review_v2_status.json": P411
    / "phase411_external_review_v2_status.json",
    "phase411_qualification.json": P411 / "phase411_qualification.json",
    "phase411_stage_summary.json": P411 / "phase411_stage_summary.json",
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
            "node_id": "p411_registered_semantic_dual_channel",
            "node_type": "strict_exact_plus_finite_registered_semantic_template_contract",
            "phase_id": "Phase411",
            "contract_context_count": denominator[
                "finite_semantic_contract_context_count"
            ],
            "finite_response_case_count": denominator[
                "finite_semantic_response_case_count"
            ],
            "failure_count": results["finite_semantic_failure_count"],
            "open_language_semantics_solved": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p411_finite_operation_closure",
            "node_type": "registered_external_world_operation_transition_algebra",
            "phase_id": "Phase411",
            "operation_count": denominator["registered_operation_count"],
            "state_transition_count": denominator[
                "registered_state_transition_count"
            ],
            "composition_case_count": denominator[
                "operation_composition_case_count"
            ],
            "composition_failure_count": results[
                "operation_composition_failure_count"
            ],
            "model_internal_operator": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p411_partition_operation_stability",
            "node_type": "coarse_and_joint_observation_partition_operation_stability_audit",
            "phase_id": "Phase411",
            "history_covariance_case_count": denominator[
                "history_rule_covariance_case_count"
            ],
            "history_covariance_failure_count": results[
                "history_rule_covariance_failure_count"
            ],
            "coarse_unstable_operation_cell_count": results[
                "coarse_observer_unstable_operation_cell_count"
            ],
            "joint_unstable_operation_cell_count": results[
                "joint_observer_unstable_operation_cell_count"
            ],
            "model_functional_bisimulation": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p411_independent_review_adjudication",
            "node_type": "reviewer_first_machine_registry_second_adjudication_gate",
            "phase_id": "Phase411",
            "required_reviewer_count": denominator[
                "required_independent_reviewer_count"
            ],
            "completed_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "scenario_count_per_reviewer": denominator[
                "review_scenario_count_per_reviewer"
            ],
            "accepted_item_count": results[
                "external_review_accepted_item_count"
            ],
            "machine_registry_privileged_in_disagreement": False,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p411_execution_boundary",
            "node_type": "external_review_and_sealed_collector_hard_execution_boundary",
            "phase_id": "Phase411",
            "sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "sealed_model_collector_denominator": denominator[
                "future_sealed_model_collector_case_count"
            ],
            "model_case_count": denominator["model_case_count_consumed"],
            "physical_case_count": denominator["physical_case_count_consumed"],
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
            "edge_id": "p410_orthogonal_to_p411_dual_channel",
            "source_node_id": "p410_orthogonal_dynamic_measurement",
            "target_node_id": "p411_registered_semantic_dual_channel",
            "edge_type": "preserves_strict_channel_and_adds_finite_registered_semantic_channel",
            "phase_id": "Phase411",
            "causal_path": False,
        },
        {
            "edge_id": "p411_dual_channel_to_operation_closure",
            "source_node_id": "p411_registered_semantic_dual_channel",
            "target_node_id": "p411_finite_operation_closure",
            "edge_type": "defines_external_state_observations_before_operation_composition",
            "phase_id": "Phase411",
            "causal_path": False,
        },
        {
            "edge_id": "p411_operation_to_partition_stability",
            "source_node_id": "p411_finite_operation_closure",
            "target_node_id": "p411_partition_operation_stability",
            "edge_type": "tests_whether_registered_observation_partitions_survive_operations",
            "phase_id": "Phase411",
            "causal_path": False,
        },
        {
            "edge_id": "p411_partition_to_review_gate",
            "source_node_id": "p411_partition_operation_stability",
            "target_node_id": "p411_independent_review_adjudication",
            "edge_type": "machine_contracts_require_independent_human_semantic_review",
            "phase_id": "Phase411",
            "causal_path": False,
        },
        {
            "edge_id": "p411_review_to_execution_boundary",
            "source_node_id": "p411_independent_review_adjudication",
            "target_node_id": "p411_execution_boundary",
            "edge_type": "keeps_cuda_physical_causal_and_neuron_execution_closed",
            "phase_id": "Phase411",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_metrics(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    denominator = stage["denominators"]
    results = stage["results"]
    manifest.setdefault("metrics", {}).update(
        {
            "phase411_semantic_contract_context_count": denominator[
                "finite_semantic_contract_context_count"
            ],
            "phase411_finite_semantic_case_count": denominator[
                "finite_semantic_response_case_count"
            ],
            "phase411_finite_semantic_failure_count": results[
                "finite_semantic_failure_count"
            ],
            "phase411_strict_resolved_case_count": results[
                "strict_resolved_case_count"
            ],
            "phase411_registered_semantic_resolved_case_count": results[
                "registered_semantic_resolved_case_count"
            ],
            "phase411_semantic_only_resolved_case_count": results[
                "semantic_only_resolved_case_count"
            ],
            "phase411_registered_operation_count": denominator[
                "registered_operation_count"
            ],
            "phase411_state_transition_count": denominator[
                "registered_state_transition_count"
            ],
            "phase411_operation_composition_case_count": denominator[
                "operation_composition_case_count"
            ],
            "phase411_operation_composition_failure_count": results[
                "operation_composition_failure_count"
            ],
            "phase411_history_covariance_case_count": denominator[
                "history_rule_covariance_case_count"
            ],
            "phase411_history_covariance_failure_count": results[
                "history_rule_covariance_failure_count"
            ],
            "phase411_coarse_unstable_operation_cell_count": results[
                "coarse_observer_unstable_operation_cell_count"
            ],
            "phase411_joint_unstable_operation_cell_count": results[
                "joint_observer_unstable_operation_cell_count"
            ],
            "phase411_required_external_reviewer_count": denominator[
                "required_independent_reviewer_count"
            ],
            "phase411_completed_external_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "phase411_review_accepted_item_count": results[
                "external_review_accepted_item_count"
            ],
            "phase411_sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "phase411_model_case_count": denominator["model_case_count_consumed"],
            "phase411_physical_case_count": denominator[
                "physical_case_count_consumed"
            ],
            "phase411_new_neuron_node_count": results["new_neuron_path_count"],
        }
    )


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    denominator = stage["denominators"]
    results = stage["results"]
    progress["last_phase"] = "Phase411-FiniteSemanticOperationPreflightStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["finite_semantic_operation_preflight_stage"] = {
        "finite_semantic_cases_machine_audited": {
            "numerator": denominator["finite_semantic_response_case_count"]
            - results["finite_semantic_failure_count"],
            "denominator": denominator["finite_semantic_response_case_count"],
        },
        "semantic_only_registered_resolutions": {
            "numerator": results["semantic_only_resolved_case_count"],
            "denominator": denominator["finite_semantic_response_case_count"],
        },
        "registered_operations_with_inverse": {
            "numerator": denominator["registered_operation_count"],
            "denominator": denominator["registered_operation_count"],
        },
        "operation_compositions_machine_audited": {
            "numerator": denominator["operation_composition_case_count"]
            - results["operation_composition_failure_count"],
            "denominator": denominator["operation_composition_case_count"],
        },
        "history_covariance_cases_machine_audited": {
            "numerator": denominator["history_rule_covariance_case_count"]
            - results["history_rule_covariance_failure_count"],
            "denominator": denominator["history_rule_covariance_case_count"],
        },
        "independent_external_reviewers": {
            "numerator": results["completed_external_reviewer_count"],
            "denominator": denominator["required_independent_reviewer_count"],
        },
        "accepted_external_review_items": {
            "numerator": results["external_review_accepted_item_count"],
            "denominator": denominator["review_scenario_count_per_reviewer"],
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
    progress["phase411_decision"] = (
        "publish_finite_protocol_evidence_only_and_keep_cuda_physical_causal_"
        "and_neuron_gates_closed_until_external_review_and_real_collector_equivalence"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase411 preflight artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    stage = payloads["phase411_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        "phase411_evidence_nodes.jsonl",
        "phase411_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase411_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase411_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase411-FiniteSemanticOperationPreflightStage"
        manifest["phase411"] = {
            "status": "finite_machine_preflight_pass_external_and_model_gates_closed",
            "files": published,
            "machine_preflight_pass": stage["assessment"]["machine_preflight_pass"],
            "model_case_count": 0,
            "physical_case_count": 0,
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase411-FiniteSemanticOperationPreflightStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase411_stage_summary.json", stage)
        write_jsonl(root / "phase411_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase411_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 411
        manifest["generated_at"] = updated_at
        manifest["phase411_audit"] = {
            "status": "finite_protocol_only_no_model_physical_or_neuron_evidence",
            "finite_semantic_case_count": stage["denominators"][
                "finite_semantic_response_case_count"
            ],
            "registered_operation_count": stage["denominators"][
                "registered_operation_count"
            ],
            "operation_composition_case_count": stage["denominators"][
                "operation_composition_case_count"
            ],
            "coarse_unstable_operation_cell_count": stage["results"][
                "coarse_observer_unstable_operation_cell_count"
            ],
            "completed_external_reviewer_count": stage["results"][
                "completed_external_reviewer_count"
            ],
            "sealed_model_collector_equivalence_case_count": stage["results"][
                "sealed_model_collector_equivalence_case_count"
            ],
            "model_case_count": 0,
            "physical_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase411_stage_summary.json",
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
