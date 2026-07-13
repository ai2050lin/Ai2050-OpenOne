#!/usr/bin/env python3
"""Publish Phase409 protocol evidence without promoting behavioral/physical nodes."""

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
P409 = ROOT / "tests/gpt5/result/phase409_dynamic_response_protocol"
JSON_SOURCES = {
    "phase409_dynamic_response_protocol.json": P409
    / "phase409_dynamic_response_protocol.json",
    "phase409_protocol_qualification.json": P409
    / "phase409_protocol_qualification.json",
    "phase409_rule_engine_agreement.json": P409
    / "phase409_rule_engine_agreement.json",
    "phase409_prompt_hash_audit.json": P409
    / "phase409_prompt_hash_audit.json",
    "phase409_protocol_stage_summary.json": P409
    / "phase409_protocol_stage_summary.json",
}
JSONL_SOURCES = {
    "phase409_query_contract_registry.jsonl": P409
    / "phase409_query_contract_registry.jsonl",
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
    denominator = stage["denominators"]
    results = stage["results"]
    nodes = [
        {
            "node_id": "p409_dynamic_history_protocol",
            "node_type": "registered_dynamic_response_and_history_protocol",
            "phase_id": "Phase409",
            "abstract_case_count": denominator["abstract_case_count"],
            "future_model_rendered_prompt_count": denominator[
                "future_model_rendered_prompt_count"
            ],
            "history_mode_count": 5,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p409_dual_solver_machine_audit",
            "node_type": "closed_form_and_finite_world_machine_rule_agreement",
            "phase_id": "Phase409",
            "scenario_count": denominator["rule_engine_scenario_count"],
            "agreement_count": results["dual_rule_engine_agreement_count"],
            "disagreement_count": results["dual_rule_engine_disagreement_count"],
            "independent_human_rule_review_completed": False,
            "protocol_only": True,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p409_interface_contract_correction",
            "node_type": "joint_query_identifiability_and_full_sentence_contract",
            "phase_id": "Phase409",
            "knowledge_single_query_individually_injective": False,
            "knowledge_three_query_joint_signature_injective": True,
            "grammar_sentence_bare_be_alias_count": results[
                "grammar_sentence_bare_be_alias_count"
            ],
            "protocol_only": True,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p409_execution_boundary",
            "node_type": "external_review_collector_model_physical_neuron_gate",
            "phase_id": "Phase409",
            "model_case_count_consumed": denominator["model_case_count_consumed"],
            "physical_case_count_consumed": denominator[
                "physical_case_count_consumed"
            ],
            "new_physical_path_count": results["new_physical_path_count"],
            "new_head_channel_or_neuron_count": results[
                "new_head_channel_or_neuron_count"
            ],
            "model_execution_authorized": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p408_gate_to_p409_dynamic_protocol",
            "source_node_id": "p408_crossmodel_partition_gate",
            "target_node_id": "p409_dynamic_history_protocol",
            "edge_type": "replaces_failed_static_endpoint_object_with_preregistered_dynamic_history_object",
            "phase_id": "Phase409",
            "causal_path": False,
        },
        {
            "edge_id": "p409_protocol_to_dual_solver_audit",
            "source_node_id": "p409_dynamic_history_protocol",
            "target_node_id": "p409_dual_solver_machine_audit",
            "edge_type": "requires_two_machine_derivations_before_future_model_execution",
            "phase_id": "Phase409",
            "causal_path": False,
        },
        {
            "edge_id": "p409_protocol_to_interface_correction",
            "source_node_id": "p409_dynamic_history_protocol",
            "target_node_id": "p409_interface_contract_correction",
            "edge_type": "registers_joint_query_roles_and_nonoverlapping_full_sentence_aliases",
            "phase_id": "Phase409",
            "causal_path": False,
        },
        {
            "edge_id": "p409_machine_audit_to_execution_boundary",
            "source_node_id": "p409_dual_solver_machine_audit",
            "target_node_id": "p409_execution_boundary",
            "edge_type": "machine_agreement_does_not_replace_external_review_or_collector_equivalence",
            "phase_id": "Phase409",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    denominator = stage["denominators"]
    results = stage["results"]
    progress["last_phase"] = "Phase409-DynamicResponseProtocolStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["dynamic_response_protocol_stage"] = {
        "registered_abstract_cases": {
            "numerator": denominator["abstract_case_count"],
            "denominator": denominator["abstract_case_count"],
        },
        "future_model_prompt_hashes": {
            "numerator": denominator["future_model_rendered_prompt_count"],
            "denominator": denominator["future_model_rendered_prompt_count"],
        },
        "dual_rule_engine_scenarios": {
            "numerator": results["dual_rule_engine_agreement_count"],
            "denominator": denominator["rule_engine_scenario_count"],
        },
        "query_contracts": {
            "numerator": denominator["query_contract_count"],
            "denominator": denominator["query_contract_count"],
        },
        "independent_human_rule_review": {"numerator": 0, "denominator": 1},
        "incremental_collector_token_equivalence": {
            "numerator": 0,
            "denominator": 1,
        },
        "model_qualification_cases_consumed": {"numerator": 0, "denominator": 165},
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase409_decision"] = (
        "publish_protocol_only_and_keep_model_physical_causal_and_neuron_gates_"
        "closed_until_external_review_and_collector_equivalence"
    )
    write_json(path, progress)


def update_metrics(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    denominator = stage["denominators"]
    results = stage["results"]
    manifest.setdefault("metrics", {}).update(
        {
            "phase409_registered_abstract_case_count": denominator[
                "abstract_case_count"
            ],
            "phase409_future_model_prompt_hash_count": denominator[
                "future_model_rendered_prompt_count"
            ],
            "phase409_query_contract_count": denominator["query_contract_count"],
            "phase409_rule_engine_scenario_count": denominator[
                "rule_engine_scenario_count"
            ],
            "phase409_dual_rule_agreement_count": results[
                "dual_rule_engine_agreement_count"
            ],
            "phase409_external_rule_review_count": 0,
            "phase409_collector_equivalence_count": 0,
            "phase409_model_case_count": denominator["model_case_count_consumed"],
            "phase409_physical_case_count": denominator[
                "physical_case_count_consumed"
            ],
            "phase409_new_neuron_node_count": results[
                "new_head_channel_or_neuron_count"
            ],
        }
    )


def main() -> None:
    missing = [
        str(path)
        for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase409 protocol artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    stage = payloads["phase409_protocol_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        *row_payloads,
        "phase409_evidence_nodes.jsonl",
        "phase409_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        write_jsonl(root / "phase409_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase409_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase409-DynamicResponseProtocolStage"
        manifest["phase409"] = {
            "status": "dynamic_response_history_protocol_frozen_model_execution_closed",
            "files": published,
            "abstract_case_count": stage["denominators"]["abstract_case_count"],
            "model_case_count": 0,
            "physical_case_count": 0,
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase409-DynamicResponseProtocolStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase409_protocol_stage_summary.json", stage)
        write_jsonl(root / "phase409_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase409_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 409
        manifest["generated_at"] = updated_at
        manifest["phase409_audit"] = {
            "status": "protocol_only_no_model_physical_or_neuron_evidence",
            "abstract_case_count": stage["denominators"]["abstract_case_count"],
            "future_model_prompt_count": stage["denominators"][
                "future_model_rendered_prompt_count"
            ],
            "machine_rule_agreement_count": stage["results"][
                "dual_rule_engine_agreement_count"
            ],
            "external_rule_review_count": 0,
            "model_case_count": 0,
            "physical_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase409_protocol_stage_summary.json",
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
