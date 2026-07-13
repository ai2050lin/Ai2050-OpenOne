#!/usr/bin/env python3
"""Publish Phase407 event-horizon evidence without promoting physical nodes."""

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
P407 = ROOT / "tests/gpt5/result/phase407_event_horizon_kernel"
JSON_SOURCES = {
    "phase407_event_horizon_protocol.json": P407
    / "phase407_event_horizon_protocol.json",
    "phase407_discovery_analysis.json": P407 / "phase407_discovery_analysis.json",
    "phase407_calibration_analysis.json": P407
    / "phase407_calibration_analysis.json",
    "phase407_behavioral_holdout_analysis.json": P407
    / "phase407_behavioral_holdout_analysis.json",
    "phase407_failure_diagnostic.json": P407 / "phase407_failure_diagnostic.json",
    "phase407_response_partition_diagnostic.json": P407
    / "phase407_response_partition_diagnostic.json",
    "phase407_event_horizon_stage_summary.json": P407
    / "phase407_event_horizon_stage_summary.json",
}
JSONL_SOURCES = {
    "phase407_failure_axes.jsonl": P407 / "analysis/phase407_failure_axes.jsonl",
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
    nodes = [
        {
            "node_id": "p407_event_horizon_response_ledger",
            "node_type": "condition_response_semantic_boundary_stop_event_ledger",
            "phase_id": "Phase407",
            "case_count": stage["denominators"]["formal_discovery_case_count"],
            "semantic_correct_count": results["semantic_correct_count"],
            "complete_response_count": results["complete_response_count"],
            "eos_observed_count": results["eos_observed_count"],
            "semantic_right_censored_count": results[
                "semantic_right_censored_count"
            ],
            "stop_right_censored_count": results["stop_right_censored_count"],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p407_registered_response_partition",
            "node_type": "observed_state_condition_registered_response_mapping_partition",
            "phase_id": "Phase407",
            "condition_cell_count": stage["denominators"][
                "formal_condition_cell_count"
            ],
            **results["response_mapping_class_counts"],
            "surface_stable_mapping_count": results[
                "surface_stable_mapping_count"
            ],
            "surface_mapping_group_count": results[
                "surface_mapping_group_count"
            ],
            "diagnostic_only": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p407_transfer_gate_negative",
            "node_type": "independent_surface_interface_history_sequence_transfer_gate",
            "phase_id": "Phase407",
            "fully_semantic_gated_group_count": results[
                "fully_semantic_gated_group_count"
            ],
            "formal_group_count": results["formal_group_count"],
            "single_model_candidate_family_count": results[
                "single_model_candidate_family_count"
            ],
            "strict_crossmodel_candidate_family_count": results[
                "strict_crossmodel_candidate_family_count"
            ],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p407_glm_nonfinite_runtime_warning",
            "node_type": "model_specific_fp16_nonfinite_H48_generation_path",
            "phase_id": "Phase407",
            "case_count": results["nonfinite_generation_path_count"],
            "all_reached_H48": results["nonfinite_all_reached_H48"],
            "model": "glm4",
            "runtime_quality_warning": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p407_operator_physical_neuron_gates_closed",
            "node_type": "conditioned_state_operator_physical_causal_neuron_boundary",
            "phase_id": "Phase407",
            "calibration_cases_consumed": 0,
            "behavioral_holdout_cases_consumed": 0,
            "physical_holdout_cases_consumed": 0,
            "neuron_nodes_promoted": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p406_to_p407_event_response_object",
            "source_node_id": "p406_conditioned_short_sequence_observation",
            "target_node_id": "p407_event_horizon_response_ledger",
            "edge_type": "separates_semantic_boundary_and_stop_events_under_H48",
            "phase_id": "Phase407",
            "causal_path": False,
        },
        {
            "edge_id": "p407_event_to_response_partition",
            "source_node_id": "p407_event_horizon_response_ledger",
            "target_node_id": "p407_registered_response_partition",
            "edge_type": "classifies_identity_permutation_collapse_and_missing_maps",
            "phase_id": "Phase407",
            "causal_path": False,
        },
        {
            "edge_id": "p407_event_to_transfer_gate",
            "source_node_id": "p407_event_horizon_response_ledger",
            "target_node_id": "p407_transfer_gate_negative",
            "edge_type": "tests_independent_surface_interface_history_and_sequence_gates",
            "phase_id": "Phase407",
            "causal_path": False,
        },
        {
            "edge_id": "p407_runtime_warning_to_transfer_gate",
            "source_node_id": "p407_glm_nonfinite_runtime_warning",
            "target_node_id": "p407_transfer_gate_negative",
            "edge_type": "marks_nonfinite_paths_as_failed_without_semantic_interpretation",
            "phase_id": "Phase407",
            "causal_path": False,
        },
        {
            "edge_id": "p407_negative_closes_downstream",
            "source_node_id": "p407_transfer_gate_negative",
            "target_node_id": "p407_operator_physical_neuron_gates_closed",
            "edge_type": "zero_crossmodel_candidates_stop_all_downstream_mapping",
            "phase_id": "Phase407",
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
    denominator = stage["denominators"]["formal_discovery_case_count"]
    progress["last_phase"] = "Phase407-EventHorizonConditionResponseStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["event_horizon_condition_response_stage"] = {
        "formal_discovery_cases": {
            "numerator": denominator,
            "denominator": denominator,
        },
        "semantic_correct_cases": {
            "numerator": results["semantic_correct_count"],
            "denominator": denominator,
        },
        "complete_response_cases": {
            "numerator": results["complete_response_count"],
            "denominator": denominator,
        },
        "fully_semantic_gated_groups": {
            "numerator": results["fully_semantic_gated_group_count"],
            "denominator": results["formal_group_count"],
        },
        "crossmodel_candidate_families": {"numerator": 0, "denominator": 3},
        "calibration_cases_consumed": {"numerator": 0, "denominator": 1},
        "behavioral_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase407_decision"] = (
        "retain_event_and_response_partition_observations_reject_current_"
        "crossmodel_state_and_keep_operator_physical_neuron_gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [
        str(path)
        for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase407 artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    stage = payloads["phase407_event_horizon_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        *row_payloads,
        "phase407_evidence_nodes.jsonl",
        "phase407_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        write_jsonl(root / "phase407_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase407_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase407-EventHorizonConditionResponseStage"
        manifest["phase407"] = {
            "status": "event_ledger_complete_zero_crossmodel_states_all_downstream_gates_closed",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase407-EventHorizonConditionResponseStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase407_event_horizon_stage_summary.json", stage)
        write_jsonl(root / "phase407_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase407_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 407
        manifest["generated_at"] = updated_at
        manifest["phase407_audit"] = {
            "status": "event_response_observation_only_no_operator_physical_or_neuron_nodes_promoted",
            "formal_discovery_case_count": 5760,
            "semantic_correct_count": stage["results"]["semantic_correct_count"],
            "fully_semantic_gated_group_count": stage["results"][
                "fully_semantic_gated_group_count"
            ],
            "formal_group_count": stage["results"]["formal_group_count"],
            "crossmodel_conditioned_state_family_count": 0,
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase407_event_horizon_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase407_formal_discovery_case_count": 5760,
                "phase407_semantic_correct_count": stage["results"][
                    "semantic_correct_count"
                ],
                "phase407_complete_response_count": stage["results"][
                    "complete_response_count"
                ],
                "phase407_fully_semantic_gated_group_count": stage["results"][
                    "fully_semantic_gated_group_count"
                ],
                "phase407_formal_group_count": stage["results"][
                    "formal_group_count"
                ],
                "phase407_nonfinite_generation_path_count": stage["results"][
                    "nonfinite_generation_path_count"
                ],
                "phase407_crossmodel_state_family_count": 0,
                "phase407_physical_holdout_case_count": 0,
                "phase407_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase407_event_horizon_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase407-EventHorizonConditionResponseStage",
            "statement": (
                "Phase407 separates semantic, sentence-boundary, and stop events "
                "and records observational response partitions. No model-family "
                "cell passes the registered state gate, so no operator, physical "
                "path, head, channel, or neuron is promoted."
            ),
            "condition_response_event_ledger_available": True,
            "response_partition_observation_available": True,
            "validated_conditioned_state_available": False,
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
