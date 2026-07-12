#!/usr/bin/env python3
"""Publish Phase386 incremental relation evidence with strict claim boundaries."""

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
P386 = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"

JSON_SOURCES = {
    name: P386 / name
    for name in (
        "phase386_incremental_contract_amendment.json",
        "phase386_instrument_audit_summary.json",
        "phase386_discovery_collection_summary.json",
        "phase386_relation_contract.json",
        "phase386_discovery_relation_summary.json",
        "phase386_calibration_collection_summary.json",
        "phase386_calibration_summary.json",
        "phase386_physical_holdout_protocol.json",
        "phase386_physical_collection_summary.json",
        "phase386_physical_summary.json",
    )
}
JSONL_SOURCES = {
    "phase386_frozen_relation_candidates.jsonl": P386
    / "discovery_relations/phase386_frozen_relation_candidates.jsonl",
    "phase386_calibrated_relation_candidates.jsonl": P386
    / "phase386_calibrated_relation_candidates.jsonl",
    "phase386_frozen_physical_candidates.jsonl": P386
    / "phase386_frozen_physical_candidates.jsonl",
    "phase386_physical_candidate_rows.jsonl": P386
    / "phase386_physical_candidate_rows.jsonl",
    "phase386_physical_model_rows.jsonl": P386 / "phase386_physical_model_rows.jsonl",
}


def graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p386_behavior_denominator",
            "node_type": "fresh_behavior_denominator",
            "phase_id": "Phase386",
            "candidate_cases": 2880,
            "eligible_mechanisms": 3,
            "attempted_mechanisms": 6,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_cache_path_correction",
            "node_type": "runtime_path_contract_correction",
            "phase_id": "Phase386",
            "status": "teacher_forced_full_replay_retired_actual_incremental_cache_used",
            "teacher_forced_deepseek_transition_mismatches": 2,
            "incremental_probe_matches": 2,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_incremental_event_ledger",
            "node_type": "five_coordinate_exact_incremental_event_ledger",
            "phase_id": "Phase386",
            "discovery_cases": 288,
            "discovery_model_calls": 1850,
            "semantic_coordinate_count": 5,
            "top_k_used": False,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_discovery_relations",
            "node_type": "descriptive_relation_candidates",
            "phase_id": "Phase386",
            "candidate_count": 135,
            "mlp_channel_candidate_count": 5,
            "attention_head_candidate_count": 47,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_calibrated_predictive_relations",
            "node_type": "calibrated_prediction_control_survivors",
            "phase_id": "Phase386",
            "relation_replication_count": 117,
            "predictive_relation_count": 12,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_physical_predictive_relations",
            "node_type": "physical_holdout_predictive_relations",
            "phase_id": "Phase386",
            "frozen_candidate_count": 12,
            "relation_replication_count": 11,
            "predictive_relation_count": 10,
            "status": "descriptive_predictive_only",
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_terminal_relation_cluster",
            "node_type": "terminal_answer_continuation_relation_cluster",
            "phase_id": "Phase386",
            "relation_count": 9,
            "transition": "target_encoded_to_post_decision_next_token",
            "upstream": False,
            "terminal_interface": True,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_upstream_relation_candidate",
            "node_type": "upstream_descriptive_predictive_relation",
            "phase_id": "Phase386",
            "mechanism_id": "relation_binding",
            "vector_family": "attention_head_state",
            "transition": "source_encoded_to_query_integrated",
            "depth_bin": 5,
            "relation_count": 1,
            "upstream": True,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p386_neuron_causal_boundary",
            "node_type": "strict_negative_boundary",
            "phase_id": "Phase386",
            "physical_mlp_channel_relation_count": 0,
            "single_neuron_causal_path_count": 0,
            "complete_language_path_count": 0,
            "language_path": False,
            "causal": False,
        },
    ]
    edges = [
        {
            "edge_id": "p385_negative_to_p386_denominator",
            "source_node_id": "p385_specificity_control_negative",
            "target_node_id": "p386_behavior_denominator",
            "edge_type": "requires_fresh_relation_aware_denominator",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_denominator_to_cache_correction",
            "source_node_id": "p386_behavior_denominator",
            "target_node_id": "p386_cache_path_correction",
            "edge_type": "runtime_replay_audit_rejects_teacher_forced_equivalence",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_cache_to_incremental_ledger",
            "source_node_id": "p386_cache_path_correction",
            "target_node_id": "p386_incremental_event_ledger",
            "edge_type": "qualifies_actual_generation_path_measurement",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_ledger_to_discovery_relations",
            "source_node_id": "p386_incremental_event_ledger",
            "target_node_id": "p386_discovery_relations",
            "edge_type": "extracts_four_condition_adjacent_coordinate_relations",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_discovery_to_calibration",
            "source_node_id": "p386_discovery_relations",
            "target_node_id": "p386_calibrated_predictive_relations",
            "edge_type": "frozen_nearest_neighbor_prediction_with_three_controls",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_calibration_to_physical",
            "source_node_id": "p386_calibrated_predictive_relations",
            "target_node_id": "p386_physical_predictive_relations",
            "edge_type": "one_time_physical_holdout_validation",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_physical_to_terminal",
            "source_node_id": "p386_physical_predictive_relations",
            "target_node_id": "p386_terminal_relation_cluster",
            "edge_type": "nine_survivors_are_terminal_answer_continuation",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_physical_to_upstream",
            "source_node_id": "p386_physical_predictive_relations",
            "target_node_id": "p386_upstream_relation_candidate",
            "edge_type": "one_survivor_precedes_answer_generation",
            "phase_id": "Phase386",
            "causal_path": False,
        },
        {
            "edge_id": "p386_physical_to_neuron_boundary",
            "source_node_id": "p386_physical_predictive_relations",
            "target_node_id": "p386_neuron_causal_boundary",
            "edge_type": "no_mlp_channel_or_causal_survivor",
            "phase_id": "Phase386",
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
        write_json(root / "phase386_relation_stage_summary.json", stage)
        write_jsonl(root / "phase386_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase386_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 386
        manifest["generated_at"] = updated_at
        manifest["phase386_audit"] = {
            "status": "physical_predictive_relations_mapped_no_neuron_or_causal_path",
            "physical_predictive_relation_count": 10,
            "terminal_relation_count": 9,
            "upstream_descriptive_relation_count": 1,
            "physical_mlp_channel_relation_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "single_unit_causal_count": 0,
            "language_path_count": 0,
            "source": "phase386_relation_stage_summary.json",
        }
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase386_relation_stage_summary.json"
        boundary = manifest.setdefault("evidence_boundary", {})
        boundary["latest_phase"] = "Phase386-PhysicalRelations"
        boundary["statement"] = (
            "Ten cross-model predictive relations survive a one-time physical holdout. "
            "Nine are terminal target-to-post continuations and one is a late-depth "
            "relation-binding source-to-query attention-head relation. None identifies "
            "an MLP neuron channel or establishes causal necessity."
        )
        boundary["multitime_incremental_event_ledger_available"] = True
        boundary["physical_predictive_relation_count"] = 10
        boundary["upstream_descriptive_predictive_relation_count"] = 1
        boundary["physical_mlp_channel_relation_count"] = 0
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
        raise FileNotFoundError(f"Missing Phase386 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    physical = payloads["phase386_physical_summary.json"]
    discovery = payloads["phase386_discovery_relation_summary.json"]
    calibration = payloads["phase386_calibration_summary.json"]
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = {
        "schema_version": "60.14.0",
        "phase_id": "Phase386-StageMerge",
        "created_at": updated_at,
        "objective": "map_actual_incremental_multitime_relations_before_causal_testing",
        "assessment": {
            "actual_incremental_kv_cache_path_used": True,
            "five_semantic_coordinates_available": True,
            "exact_attention_head_source_events_replayable": True,
            "exact_mlp_channel_events_replayable": True,
            "one_time_physical_holdout_completed": True,
            "descriptive_predictive_relations_available": True,
            "upstream_descriptive_predictive_relation_available": True,
            "mlp_neuron_relation_available": False,
            "causal_language_path_available": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "candidate_behavior_cases": 2880,
            "behavior_eligible_mechanisms": {"numerator": 3, "denominator": 6},
            "incremental_discovery_cases": 288,
            "incremental_discovery_model_calls": 1850,
            "incremental_calibration_cases": 144,
            "incremental_physical_cases": 144,
            "physical_candidate_count": 12,
        },
        "results": {
            "discovery_relation_candidate_count": discovery["denominator"][
                "crossmodel_frozen_candidate_count"
            ],
            "calibration_relation_replication_count": calibration["results"][
                "crossmodel_relation_replication_count"
            ],
            "calibration_predictive_relation_count": calibration["results"][
                "crossmodel_predictive_relation_path_count"
            ],
            "physical_relation_replication_count": physical["results"][
                "physical_relation_replication_count"
            ],
            "physical_predictive_relation_count": physical["results"][
                "physical_predictive_relation_path_count"
            ],
            "terminal_predictive_relation_count": 9,
            "upstream_descriptive_predictive_relation_count": 1,
            "physical_mlp_channel_relation_count": 0,
            "single_neuron_causal_path_count": 0,
            "complete_language_path_count": 0,
        },
        "hard_limits": [
            "only_three_of_six_fresh_mechanisms_passed_the_behavior_denominator",
            "nine_of_ten_physical_survivors_are_terminal_answer_continuations",
            "the_only_upstream_survivor_is_late_depth_and_attention_head_aggregated",
            "attention_head_state_sums_all_source_positions_before_output_projection",
            "nearest_neighbor_prediction_can_capture_token_identity_without_causal_transport",
            "wrong_time_and_wrong_depth_controls_do_not_equal_an_intervention",
            "no_mlp_channel_relation_survived_physical_prediction_controls",
            "no_single_neuron_or_causal_necessity_test_passed",
            "small_model_architecture_and_generation_length_differ_substantially",
        ],
        "authorization": {
            "show_physical_predictive_relations_as_descriptive": True,
            "show_ten_relations_as_complete_language_paths": False,
            "show_specific_neuron_path": False,
            "show_causal_necessity": False,
            "reuse_physical_holdout": False,
        },
        "next_stage": {
            "phase": 387,
            "objective": (
                "causally_test_the_single_upstream_relation_on_a_fresh_denominator_"
                "without_reusing_the_physical_holdout"
            ),
            "required_order": [
                "freeze_fresh_relation_binding_cases_and_position_contract",
                "separate_source_state_key_value_and_query_receiver_interventions",
                "require_target_event_mediation_and_behavior_effect",
                "use_wrong_layer_wrong_position_and_terminal_relation_controls",
                "do_not_promote_attention_head_aggregate_to_neuron_identity",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    nodes, edges = graph()
    payloads["phase386_relation_stage_summary.json"] = stage
    row_payloads["phase386_evidence_nodes.jsonl"] = nodes
    row_payloads["phase386_evidence_edges.jsonl"] = edges
    published_files = [*payloads.keys(), *row_payloads.keys()]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase386-StageMerge"
        manifest["phase386"] = {
            "status": "incremental_multitime_relations_physically_validated_not_causal",
            "behavior_candidate_case_count": 2880,
            "eligible_mechanism_count": 3,
            "incremental_discovery_case_count": 288,
            "discovery_relation_candidate_count": 135,
            "calibration_predictive_relation_count": 12,
            "physical_predictive_relation_count": 10,
            "terminal_predictive_relation_count": 9,
            "upstream_descriptive_predictive_relation_count": 1,
            "physical_mlp_channel_relation_count": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published_files,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase386-StageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["multitime_relation_stage"] = {
            "behavior_eligible_mechanisms": {"numerator": 3, "denominator": 6},
            "incremental_discovery_cases": {"numerator": 288, "denominator": 288},
            "incremental_event_ledger_models": {"numerator": 3, "denominator": 3},
            "physical_holdout_cases": {"numerator": 144, "denominator": 144},
            "descriptive_discovery_relations": {"numerator": 135, "denominator": 135},
            "calibrated_predictive_relations": {"numerator": 12, "denominator": 135},
            "physical_predictive_relations": {"numerator": 10, "denominator": 12},
            "upstream_descriptive_predictive_relations": {"numerator": 1, "denominator": 10},
            "physical_mlp_channel_relations": {"numerator": 0, "denominator": 5},
            "complete_language_paths": {"numerator": 0, "denominator": 72},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
        }
        objective = progress.get("objective_denominator_progress", {})
        if "physical_heldout_mechanism_coverage" in objective:
            objective["physical_heldout_mechanism_coverage"]["numerator"] = 2
        progress["phase386_decision"] = (
            "freeze_ten_descriptive_predictive_relations_and_test_one_upstream_"
            "candidate_causally_on_fresh_cases"
        )
        write_json(root / "progress.json", progress)
        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase386-StageMerge"
            client_index["latest_stage_files"] = [
                "phase386_relation_stage_summary.json",
                "phase386_evidence_nodes.jsonl",
                "phase386_evidence_edges.jsonl",
                "phase386_physical_summary.json",
                "phase386_physical_candidate_rows.jsonl",
                "phase386_relation_contract.json",
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
