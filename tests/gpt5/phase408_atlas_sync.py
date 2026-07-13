#!/usr/bin/env python3
"""Publish Phase408 response-partition evidence without fabricating physical nodes."""

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
P408 = ROOT / "tests/gpt5/result/phase408_partition_interface"
JSON_SOURCES = {
    "phase408_partition_interface_protocol.json": P408
    / "phase408_partition_interface_protocol.json",
    "phase408_discovery_analysis.json": P408 / "phase408_discovery_analysis.json",
    "phase408_calibration_analysis.json": P408
    / "phase408_calibration_analysis.json",
    "phase408_behavioral_holdout_analysis.json": P408
    / "phase408_behavioral_holdout_analysis.json",
    "phase408_failure_diagnostic.json": P408
    / "phase408_failure_diagnostic.json",
    "phase408_execution_recovery_audit.json": P408
    / "phase408_execution_recovery_audit.json",
    "phase408_partition_interface_stage_summary.json": P408
    / "phase408_partition_interface_stage_summary.json",
}
JSONL_SOURCES = {
    "phase408_discovery_group_audits.jsonl": P408
    / "analysis/discovery/phase408_group_audits.jsonl",
    "phase408_calibration_group_audits.jsonl": P408
    / "analysis/calibration/phase408_group_audits.jsonl",
    "phase408_behavioral_holdout_group_audits.jsonl": P408
    / "analysis/behavioral_holdout/phase408_group_audits.jsonl",
    "phase408_failure_axes.jsonl": P408
    / "analysis/phase408_failure_axes.jsonl",
    "phase408_interface_failure_axes.jsonl": P408
    / "analysis/phase408_interface_failure_axes.jsonl",
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


def evidence_graph(stage: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    results = stage["results"]
    denominator = stage["denominators"]["discovery_case_count"]
    candidates = results["behavioral_crossmodel_candidate_families"]
    nodes = [
        {
            "node_id": "p408_exclusive_response_registry",
            "node_type": "finite_exclusive_machine_response_contract",
            "phase_id": "Phase408",
            "case_count": denominator,
            "semantic_class_counts": results["semantic_class_counts"],
            "runtime_numeric_status_counts": results[
                "runtime_numeric_status_counts"
            ],
            "semantic_numeric_event_axes_orthogonal": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p408_functional_response_partition",
            "node_type": "surface_lexical_stable_interface_response_partition_audit",
            "phase_id": "Phase408",
            "group_count": stage["denominators"]["discovery_group_count"],
            "condition_separation_pass_group_count": results[
                "condition_separation_pass_group_count"
            ],
            "surface_lexical_stability_pass_group_count": results[
                "surface_lexical_stability_pass_group_count"
            ],
            "task_coordinate_covariance_pass_group_count": results[
                "task_coordinate_covariance_pass_group_count"
            ],
            "functional_group_pass_count": results[
                "functional_group_pass_count"
            ],
            "observational": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p408_crossmodel_partition_gate",
            "node_type": "discovery_calibration_behavioral_crossmodel_partition_gate",
            "phase_id": "Phase408",
            "discovery_candidate_families": results[
                "discovery_crossmodel_candidate_families"
            ],
            "calibration_candidate_families": results[
                "calibration_crossmodel_candidate_families"
            ],
            "behavioral_candidate_families": candidates,
            "history_replication_protocol_authorized": bool(candidates),
            "physical_protocol_authorized": False,
            "physical_protocol_executed": False,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p408_internal_physical_neuron_boundary",
            "node_type": "behavioral_partition_internal_operator_physical_neuron_boundary",
            "phase_id": "Phase408",
            "validated_internal_operator_count": 0,
            "physical_holdout_cases_consumed": 0,
            "physical_path_nodes_promoted": 0,
            "head_channel_neuron_nodes_promoted": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p407_to_p408_exclusive_registry",
            "source_node_id": "p407_registered_response_partition",
            "target_node_id": "p408_exclusive_response_registry",
            "edge_type": "replaces_overlapping_contracts_with_orthogonal_response_runtime_event_axes",
            "phase_id": "Phase408",
            "causal_path": False,
        },
        {
            "edge_id": "p408_registry_to_functional_partition",
            "source_node_id": "p408_exclusive_response_registry",
            "target_node_id": "p408_functional_response_partition",
            "edge_type": "tests_distinguishability_stability_and_external_coordinate_covariance",
            "phase_id": "Phase408",
            "causal_path": False,
        },
        {
            "edge_id": "p408_partition_to_crossmodel_gate",
            "source_node_id": "p408_functional_response_partition",
            "target_node_id": "p408_crossmodel_partition_gate",
            "edge_type": "requires_frozen_signature_replication_across_models_and_splits",
            "phase_id": "Phase408",
            "causal_path": False,
        },
        {
            "edge_id": "p408_gate_to_internal_boundary",
            "source_node_id": "p408_crossmodel_partition_gate",
            "target_node_id": "p408_internal_physical_neuron_boundary",
            "edge_type": (
                "authorizes_new_blind_history_replication_but_not_physical_mapping"
                if candidates
                else "closes_internal_physical_causal_and_neuron_work"
            ),
            "phase_id": "Phase408",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    results = stage["results"]
    denominator = stage["denominators"]["discovery_case_count"]
    group_denominator = stage["denominators"]["discovery_group_count"]
    progress["last_phase"] = "Phase408-ExclusiveResponsePartitionInterfaceStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["partition_interface_stage"] = {
        "formal_discovery_cases": {
            "numerator": denominator,
            "denominator": denominator,
        },
        "registered_response_cases": {
            "numerator": results["registered_response_observed_count"],
            "denominator": denominator,
        },
        "allowed_response_cases": {
            "numerator": results["allowed_response_observed_count"],
            "denominator": denominator,
        },
        "condition_separation_groups": {
            "numerator": results["condition_separation_pass_group_count"],
            "denominator": group_denominator,
        },
        "surface_lexical_stability_groups": {
            "numerator": results[
                "surface_lexical_stability_pass_group_count"
            ],
            "denominator": group_denominator,
        },
        "functional_partition_groups": {
            "numerator": results["functional_group_pass_count"],
            "denominator": group_denominator,
        },
        "discovery_crossmodel_candidate_families": {
            "numerator": len(results["discovery_crossmodel_candidate_families"]),
            "denominator": 3,
        },
        "calibration_crossmodel_candidate_families": {
            "numerator": len(results["calibration_crossmodel_candidate_families"]),
            "denominator": 3,
        },
        "behavioral_crossmodel_candidate_families": {
            "numerator": len(results["behavioral_crossmodel_candidate_families"]),
            "denominator": 3,
        },
        "calibration_cases_consumed": {
            "numerator": stage["denominators"]["calibration_case_count_consumed"],
            "denominator": 1,
        },
        "behavioral_holdout_cases_consumed": {
            "numerator": stage["denominators"][
                "behavioral_holdout_case_count_consumed"
            ],
            "denominator": 1,
        },
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase408_decision"] = (
        "retain_observational_response_partitions_and_require_blind_history_"
        "replication_before_any_physical_protocol_or_internal_node_promotion"
        if results["behavioral_crossmodel_candidate_families"]
        else "retain_failure_ledger_and_keep_internal_physical_causal_and_neuron_gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [
        str(path)
        for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase408 artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    stage = payloads["phase408_partition_interface_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        *row_payloads,
        "phase408_evidence_nodes.jsonl",
        "phase408_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        write_jsonl(root / "phase408_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase408_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase408-ExclusiveResponsePartitionInterfaceStage"
        manifest["phase408"] = {
            "status": "exclusive_response_partition_audit_complete",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase408-ExclusiveResponsePartitionInterfaceStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    results = stage["results"]
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase408_partition_interface_stage_summary.json", stage)
        write_jsonl(root / "phase408_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase408_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 408
        manifest["generated_at"] = updated_at
        manifest["phase408_audit"] = {
            "status": "response_partition_observation_only_no_physical_or_neuron_nodes_promoted",
            "formal_discovery_case_count": stage["denominators"][
                "discovery_case_count"
            ],
            "formal_group_count": stage["denominators"]["discovery_group_count"],
            "functional_group_pass_count": results["functional_group_pass_count"],
            "behavioral_crossmodel_partition_family_count": len(
                results["behavioral_crossmodel_candidate_families"]
            ),
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase408_partition_interface_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase408_formal_discovery_case_count": stage["denominators"][
                    "discovery_case_count"
                ],
                "phase408_registered_response_observed_count": results[
                    "registered_response_observed_count"
                ],
                "phase408_allowed_response_observed_count": results[
                    "allowed_response_observed_count"
                ],
                "phase408_condition_separation_pass_group_count": results[
                    "condition_separation_pass_group_count"
                ],
                "phase408_surface_lexical_stability_pass_group_count": results[
                    "surface_lexical_stability_pass_group_count"
                ],
                "phase408_functional_group_pass_count": results[
                    "functional_group_pass_count"
                ],
                "phase408_discovery_crossmodel_partition_family_count": len(
                    results["discovery_crossmodel_candidate_families"]
                ),
                "phase408_calibration_crossmodel_partition_family_count": len(
                    results["calibration_crossmodel_candidate_families"]
                ),
                "phase408_behavioral_crossmodel_partition_family_count": len(
                    results["behavioral_crossmodel_candidate_families"]
                ),
                "phase408_physical_holdout_case_count": 0,
                "phase408_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase408_partition_interface_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase408-ExclusiveResponsePartitionInterfaceStage",
            "statement": (
                "Phase408 separates response distinguishability, label alignment, "
                "runtime validity, events, and censoring. Published partitions are "
                "behavioral observations; no internal operator, physical path, head, "
                "channel, or neuron is promoted."
            ),
            "exclusive_response_ledger_available": True,
            "functional_response_partition_observation_available": True,
            "validated_internal_operator_available": False,
            "physical_conditioned_state_path_available": False,
            "single_unit_causal_closure": False,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        public_manifest(root, updated_at)
        update_checksums(root)

    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
