#!/usr/bin/env python3
"""Publish the Phase382 transition-operator identifiability audit."""

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
P382 = ROOT / "tests/gpt5/result/phase382_transition_event_audit"

JSON_SOURCES = {
    "phase382_transition_protocol.json": P382 / "phase382_transition_protocol.json",
    "phase382_transition_summary.json": P382 / "phase382_transition_summary.json",
}
JSONL_SOURCES = {
    "phase382_profiles.jsonl": P382 / "phase382_profiles.jsonl",
    "phase382_residual_profiles.jsonl": P382 / "phase382_residual_profiles.jsonl",
    "phase382_replication_rows.jsonl": P382 / "phase382_replication_rows.jsonl",
    "phase382_crossmodel_rows.jsonl": P382 / "phase382_crossmodel_rows.jsonl",
}


def graph(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p382_static_state_baseline",
            "node_type": "offline_identifiability_baseline",
            "phase_id": "Phase382-TransitionAnalysis",
            "status": "13_of_27_own_profile_wins",
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p382_total_layer_update",
            "node_type": "candidate_transition_operator",
            "phase_id": "Phase382-TransitionAnalysis",
            "formula": "layer_output-layer_input",
            "status": "12_of_27_own_profile_wins",
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p382_transition_operator_negative",
            "node_type": "strong_offline_negative_result",
            "phase_id": "Phase382-TransitionAnalysis",
            "status": "all_three_parameter_free_improvement_gates_failed",
            "causal": False,
            "language_path": False,
            "single_unit_causal": False,
        },
        {
            "node_id": "p383_component_event_ledger_unknown",
            "node_type": "unresolved_algorithmic_object",
            "phase_id": "Phase382-StageMerge",
            "status": "attention_source_contributions_and_mlp_writes_not_yet_conserved",
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p382_static_state_baseline->p382_transition_operator_negative",
            "source_node_id": "p382_static_state_baseline",
            "target_node_id": "p382_transition_operator_negative",
            "edge_type": "baseline_outperforms_candidate_on_frozen_vector",
            "phase_id": "Phase382-TransitionAnalysis",
            "causal_path": False,
        },
        {
            "edge_id": "p382_total_layer_update->p382_transition_operator_negative",
            "source_node_id": "p382_total_layer_update",
            "target_node_id": "p382_transition_operator_negative",
            "edge_type": "candidate_rejected_by_all_parameter_free_gates",
            "phase_id": "Phase382-TransitionAnalysis",
            "causal_path": False,
        },
        {
            "edge_id": "p382_transition_operator_negative->p383_component_event_ledger_unknown",
            "source_node_id": "p382_transition_operator_negative",
            "target_node_id": "p383_component_event_ledger_unknown",
            "edge_type": "requires_finer_conserved_event_decomposition",
            "phase_id": "Phase382-StageMerge",
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
        write_json(root / "phase382_transition_stage_summary.json", stage)
        write_jsonl(root / "phase382_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase382_evidence_edges.jsonl", edges)
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest["phase"] = 382
            manifest["generated_at"] = updated_at
            manifest["phase382_audit"] = {
                "status": "total_layer_update_operator_rejected_no_neuron_promotion",
                "transition_event_row_count": 20592,
                "transition_own_profile_win_count": 12,
                "static_own_profile_win_count": 13,
                "identifiability_gate_pass": False,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "language_path_count": 0,
                "source": "phase382_transition_stage_summary.json",
            }
            manifest.setdefault("files", {})[
                "latest_evidence_summary"
            ] = "phase382_transition_stage_summary.json"
            boundary = manifest.setdefault("evidence_boundary", {})
            boundary["statement"] = (
                "An offline frozen-split audit finds that total layer updates are less identifiable than static "
                "layer inputs on all three parameter-free comparisons. The candidate operator is rejected and no "
                "component or neuron path is promoted."
            )
            boundary["latest_phase"] = "Phase382-TransitionAnalysis"
            boundary["total_layer_update_operator_supported"] = False
            boundary["upstream_language_path_available"] = False
            boundary["single_unit_causal_closure"] = False
            write_json(manifest_path, manifest)
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
        raise FileNotFoundError(f"Missing Phase382 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    summary = payloads["phase382_transition_summary.json"]
    transition = summary["results"]["metrics"]["transition_update"]
    static = summary["results"]["metrics"]["static_layer_input"]
    stage = {
        "schema_version": "55.2.0",
        "phase_id": "Phase382-StageMerge",
        "created_at": updated_at,
        "objective": "audit_a_basic_dynamic_operator_before_spending_new_cuda_budget",
        "assessment": {
            "static_full_state_is_sufficient_operator": False,
            "joint_semantic_position_full_state_is_sufficient_operator": False,
            "total_layer_update_is_more_identifiable_than_static_state": False,
            "component_event_conservation_needed": True,
            "new_cuda_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
            "nine_family_layout_completed": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": summary["denominator"],
        "results": {
            "transition_update": transition,
            "static_layer_input": static,
            "parameter_free_gate_vector": summary["results"][
                "parameter_free_gate_vector"
            ],
            "transition_update_more_identifiable_than_static_state": False,
        },
        "hard_limits": [
            "the_audit_reuses_phase381_traces_and_is_not_independent_causal_confirmation",
            "layer_output_minus_layer_input_collapses_attention_mlp_and_residual_interactions",
            "three_mechanisms_do_not_cover_the_nine_family_registry",
            "profile_identifiability_does_not_establish_natural_necessity_or_sufficiency",
            "small_models_may_expose_coarser_or_architecture_specific_events",
        ],
        "authorization": {
            "show_total_layer_update_as_rejected_operator": True,
            "show_any_phase382_profile_as_language_path": False,
            "show_any_phase382_neuron": False,
            "run_new_cuda_intervention_from_phase382": False,
            "claim_global_layout_complete": False,
        },
        "next_stage": {
            "phase": 383,
            "objective": "establish_an_exact_component_event_conservation_ledger_before_new_mechanism_search",
            "required_order": [
                "instrument_attention_source_token_contributions_and_mlp_write_events",
                "verify_per_layer_reconstruction_error_on_all_three_models",
                "retain_token_role_and_generation_time_without_pooling",
                "only_then_define_a_four_condition_transition_operator",
                "require_fresh_heldout_prediction_before_any_causal_intervention",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    nodes, edges = graph(summary)
    payloads["phase382_transition_stage_summary.json"] = stage
    row_payloads["phase382_evidence_nodes.jsonl"] = nodes
    row_payloads["phase382_evidence_edges.jsonl"] = edges
    published_files = [*payloads.keys(), *row_payloads.keys()]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase382-StageMerge"
        manifest["phase382"] = {
            "status": "total_layer_update_operator_rejected",
            "transition_event_row_count": 20592,
            "profile_count": 108,
            "transition_own_profile_win_count": 12,
            "static_own_profile_win_count": 13,
            "parameter_free_gate_pass_count": 0,
            "causal_intervention_authorized": False,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published_files,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase382-StageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["transition_operator_stage"] = {
            "replay_qualified_groups": {"numerator": 22, "denominator": 22},
            "transition_event_rows": {"numerator": 20592, "denominator": 20592},
            "parameter_free_improvement_gates": {"numerator": 0, "denominator": 3},
            "causal_intervention_authorized": {"numerator": 0, "denominator": 1},
            "complete_language_paths": {"numerator": 0, "denominator": 18},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 18},
        }
        progress["phase382_decision"] = (
            "reject_total_layer_update_and_build_exact_component_event_conservation_ledger"
        )
        write_json(root / "progress.json", progress)
        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase382-StageMerge"
            client_index["latest_stage_files"] = [
                "phase382_transition_stage_summary.json",
                "phase382_evidence_nodes.jsonl",
                "phase382_evidence_edges.jsonl",
                "phase382_transition_summary.json",
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
