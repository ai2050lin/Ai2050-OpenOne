#!/usr/bin/env python3
"""Publish Phase406 conditioned-sequence evidence to both atlas mirrors."""

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
P406 = ROOT / "tests/gpt5/result/phase406_conditioned_sequence_state"
JSON_SOURCES = {
    "phase406_conditioned_sequence_protocol.json": P406
    / "phase406_conditioned_sequence_protocol.json",
    "phase406_discovery_analysis.json": P406 / "phase406_discovery_analysis.json",
    "phase406_failure_diagnostic.json": P406
    / "phase406_discovery_failure_diagnostic.json",
    "phase406_lexical_upper_bound_audit.json": P406
    / "phase406_lexical_upper_bound_audit.json",
    "phase406_horizon_extension_diagnostic.json": P406
    / "phase406_horizon_extension_diagnostic.json",
    "phase406_conditioned_sequence_stage_summary.json": P406
    / "phase406_conditioned_sequence_stage_summary.json",
}
JSONL_SOURCES = {
    "phase406_failure_axes.jsonl": P406 / "analysis/phase406_failure_axes.jsonl",
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
            "node_id": "p406_conditioned_short_sequence_observation",
            "node_type": "state_condition_H12_short_sequence_response_ledger",
            "phase_id": "Phase406",
            "case_count": 5760,
            "first_step_candidate_correct_count": results[
                "first_step_candidate_correct_count"
            ],
            "first_step_global_top_target_count": results[
                "first_step_global_top_target_count"
            ],
            "H12_sequence_semantic_correct_count": results[
                "H12_short_sequence_semantic_correct_count"
            ],
            "first_vocab_wrong_sequence_correct_count": results[
                "first_vocab_wrong_H12_sequence_correct_count"
            ],
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p406_interface_transfer_gate_negative",
            "node_type": "paired_interface_leave_one_transfer_gate",
            "phase_id": "Phase406",
            "formal_group_pass_count": results["formal_group_pass_count"],
            "formal_group_count": results["formal_group_count"],
            "crossmodel_candidate_family_count": 0,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p406_horizon_extension_diagnostic",
            "node_type": "post_discovery_H12_H24_H36_H48_failure_extension",
            "phase_id": "Phase406",
            "selected_failure_case_count": 2248,
            "newly_recovered_at_H48_count": results[
                "horizon_newly_recovered_after_H12"
            ]["48"],
            "candidate_gate": False,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p406_operator_physical_neuron_gates_closed",
            "node_type": "conditioned_state_operator_physical_causal_neuron_boundary",
            "phase_id": "Phase406",
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
            "edge_id": "p405_to_p406_condition_response_object",
            "source_node_id": "p405_predictive_physical_neuron_gates_closed",
            "target_node_id": "p406_conditioned_short_sequence_observation",
            "edge_type": "replaces_single_token_state_with_state_condition_short_sequence",
            "phase_id": "Phase406",
            "causal_path": False,
        },
        {
            "edge_id": "p406_sequence_to_interface_gate",
            "source_node_id": "p406_conditioned_short_sequence_observation",
            "target_node_id": "p406_interface_transfer_gate_negative",
            "edge_type": "tests_surface_stability_and_paired_interface_leave_one_transfer",
            "phase_id": "Phase406",
            "causal_path": False,
        },
        {
            "edge_id": "p406_sequence_to_horizon_diagnostic",
            "source_node_id": "p406_conditioned_short_sequence_observation",
            "target_node_id": "p406_horizon_extension_diagnostic",
            "edge_type": "tests_whether_H12_failure_is_response_truncation",
            "phase_id": "Phase406",
            "causal_path": False,
        },
        {
            "edge_id": "p406_negative_closes_downstream",
            "source_node_id": "p406_interface_transfer_gate_negative",
            "target_node_id": "p406_operator_physical_neuron_gates_closed",
            "edge_type": "zero_crossmodel_candidates_stop_operator_and_physical_mapping",
            "phase_id": "Phase406",
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
    progress["last_phase"] = "Phase406-ConditionedSequenceStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["conditioned_sequence_stage"] = {
        "formal_discovery_cases": {"numerator": 5760, "denominator": 5760},
        "first_step_candidate_correct_cases": {
            "numerator": results["first_step_candidate_correct_count"],
            "denominator": 5760,
        },
        "first_step_global_top_target_cases": {
            "numerator": results["first_step_global_top_target_count"],
            "denominator": 5760,
        },
        "H12_short_sequence_semantic_correct_cases": {
            "numerator": results["H12_short_sequence_semantic_correct_count"],
            "denominator": 5760,
        },
        "formal_group_passes": {
            "numerator": results["formal_group_pass_count"],
            "denominator": results["formal_group_count"],
        },
        "crossmodel_candidate_families": {"numerator": 0, "denominator": 3},
        "calibration_cases_consumed": {"numerator": 0, "denominator": 1},
        "behavioral_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase406_decision"] = (
        "retain_short_sequence_recovery_observation_reject_current_"
        "conditioned_state_and_keep_operator_physical_neuron_gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [
        str(path)
        for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase406 artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    stage = payloads["phase406_conditioned_sequence_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        *row_payloads,
        "phase406_evidence_nodes.jsonl",
        "phase406_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        write_jsonl(root / "phase406_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase406_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase406-ConditionedSequenceStage"
        manifest["phase406"] = {
            "status": "short_sequence_recovery_observed_zero_crossmodel_conditioned_states_all_downstream_gates_closed",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase406-ConditionedSequenceStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase406_conditioned_sequence_stage_summary.json", stage)
        write_jsonl(root / "phase406_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase406_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 406
        manifest["generated_at"] = updated_at
        manifest["phase406_audit"] = {
            "status": "conditioned_sequence_candidates_zero_no_operator_physical_or_neuron_nodes_promoted",
            "formal_discovery_case_count": 5760,
            "H12_sequence_semantic_correct_count": stage["results"][
                "H12_short_sequence_semantic_correct_count"
            ],
            "crossmodel_conditioned_state_family_count": 0,
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase406_conditioned_sequence_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase406_formal_discovery_case_count": 5760,
                "phase406_first_step_candidate_correct_count": stage["results"][
                    "first_step_candidate_correct_count"
                ],
                "phase406_first_step_global_top_target_count": stage["results"][
                    "first_step_global_top_target_count"
                ],
                "phase406_H12_sequence_semantic_correct_count": stage["results"][
                    "H12_short_sequence_semantic_correct_count"
                ],
                "phase406_formal_group_pass_count": stage["results"][
                    "formal_group_pass_count"
                ],
                "phase406_crossmodel_state_family_count": 0,
                "phase406_physical_holdout_case_count": 0,
                "phase406_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase406_conditioned_sequence_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase406-ConditionedSequenceStage",
            "statement": (
                "Phase406 records finite state-condition short sequences and "
                "shows substantial semantic recovery beyond the first token. "
                "No language family passes the registered crossmodel state "
                "gate, so no operator, physical path, head, channel, or neuron "
                "is promoted."
            ),
            "condition_response_ledger_available": True,
            "short_sequence_recovery_observed": True,
            "validated_conditioned_state_available": False,
            "validated_internal_operator_available": False,
            "physical_conditioned_state_path_available": False,
            "single_unit_causal_closure": False,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        public_manifest(root, updated_at)
        update_checksums(root)

    print(json.dumps(stage, indent=2))


if __name__ == "__main__":
    main()
