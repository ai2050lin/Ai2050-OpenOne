#!/usr/bin/env python3
"""Publish Phase381 joint-state negative evidence to both atlas mirrors."""

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
P381 = ROOT / "tests/gpt5/result/phase381_joint_state_formation"

JSON_SOURCES = {
    "phase381_joint_state_protocol.json": P381 / "phase381_protocol.json",
    "phase381_case_bank_summary.json": P381 / "phase381_case_bank_summary.json",
    "phase381_target_expansion_summary.json": P381
    / "target_expansion/phase381x_protocol.json",
    "phase381_behavior_analysis_final_summary.json": P381
    / "phase381_behavior_analysis_final_summary.json",
    "phase381_joint_scan_freeze.json": P381 / "phase381_joint_scan_freeze.json",
    "phase381_joint_state_summary.json": P381 / "phase381_joint_state_summary.json",
    "phase381_qwen3_trace_summary.json": P381 / "trace/models/qwen3/complete.json",
    "phase381_glm4_trace_summary.json": P381 / "trace/models/glm4/complete.json",
    "phase381_deepseek7b_trace_summary.json": P381
    / "trace/models/deepseek7b/complete.json",
}

JSONL_SOURCES = {
    "phase381_joint_model_cells.jsonl": P381
    / "causal/phase381_joint_model_cells.jsonl",
    "phase381_joint_crossmodel_cells.jsonl": P381
    / "causal/phase381_joint_crossmodel_cells.jsonl",
    "phase381_shared_upstream_territories.jsonl": P381
    / "causal/phase381_shared_upstream_territories.jsonl",
}


def build_graph(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p381_fresh_behavior_denominator",
            "node_type": "fresh_joint_state_denominator",
            "phase_id": "Phase381-BehaviorAnalysisFinal",
            "status": "24_three_model_groups_frozen_before_trace",
            "language_path": False,
        },
        {
            "node_id": "p381_exact_trace",
            "node_type": "decision_aligned_exact_trace",
            "phase_id": "Phase381-ExactTrace",
            "status": "285_of_288_replay_matches_22_groups_qualified",
            "language_path": False,
        },
        {
            "node_id": "p381_joint_role_state_hypothesis",
            "node_type": "joint_state_causal_hypothesis",
            "phase_id": "Phase381-JointScanFreeze",
            "role_set": ["source", "query", "current"],
            "status": "tested_with_single_position_and_control_baselines",
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p381_joint_role_state_negative",
            "node_type": "strong_negative_result",
            "phase_id": "Phase381-JointStateAnalysis",
            "status": "zero_replicated_model_cells",
            "direction_pass_count": summary["results"]["joint_direction_gate_pass_count"],
            "maximum_repeated_group_count": summary["results"][
                "maximum_all_four_direction_group_pass_count_in_any_model_cell"
            ],
            "causal": False,
            "language_path": False,
            "single_unit_causal": False,
        },
        {
            "node_id": "p382_transition_operator_unknown",
            "node_type": "unresolved_algorithmic_object",
            "phase_id": "Phase381-StageMerge",
            "status": "static_state_replacement_insufficient_transition_operator_unresolved",
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p381_fresh_behavior_denominator->p381_exact_trace",
            "source_node_id": "p381_fresh_behavior_denominator",
            "target_node_id": "p381_exact_trace",
            "edge_type": "authorizes_exact_trace",
            "phase_id": "Phase381-ExactTrace",
            "causal_path": False,
        },
        {
            "edge_id": "p381_exact_trace->p381_joint_role_state_hypothesis",
            "source_node_id": "p381_exact_trace",
            "target_node_id": "p381_joint_role_state_hypothesis",
            "edge_type": "authorizes_replay_qualified_joint_scan",
            "phase_id": "Phase381-JointScanFreeze",
            "causal_path": False,
        },
        {
            "edge_id": "p381_joint_role_state_hypothesis->p381_joint_role_state_negative",
            "source_node_id": "p381_joint_role_state_hypothesis",
            "target_node_id": "p381_joint_role_state_negative",
            "edge_type": "rejected_by_frozen_repetition_gate",
            "phase_id": "Phase381-JointStateAnalysis",
            "causal_path": False,
        },
        {
            "edge_id": "p381_joint_role_state_negative->p382_transition_operator_unknown",
            "source_node_id": "p381_joint_role_state_negative",
            "target_node_id": "p382_transition_operator_unknown",
            "edge_type": "requires_dynamic_transition_object_not_broader_static_swap",
            "phase_id": "Phase381-StageMerge",
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
        write_json(root / "phase381_joint_state_stage_summary.json", stage)
        write_jsonl(root / "phase381_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase381_evidence_edges.jsonl", edges)
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest["phase"] = 381
            manifest["generated_at"] = updated_at
            manifest["phase381_audit"] = {
                "status": "joint_semantic_position_state_rejected_no_neuron_promotion",
                "causal_condition_row_count": 69120,
                "joint_direction_pass_count": 563,
                "model_cell_pass_count": 0,
                "crossmodel_upstream_cell_count": 0,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "language_path_count": 0,
                "source": "phase381_joint_state_stage_summary.json",
            }
            manifest.setdefault("files", {})[
                "latest_evidence_summary"
            ] = "phase381_joint_state_stage_summary.json"
            boundary = manifest.setdefault("evidence_boundary", {})
            boundary["statement"] = (
                "A fresh three-model intervention denominator rejects the hypothesis that replacing the combined "
                "source, query, and current vectors is a repeatable upstream state operator. No model cell reaches "
                "the six-group gate, so no component or neuron path is promoted."
            )
            boundary["latest_phase"] = "Phase381-JointStateAnalysis"
            boundary["joint_semantic_position_state_supported"] = False
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
        raise FileNotFoundError(f"Missing Phase381 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    behavior = payloads["phase381_behavior_analysis_final_summary.json"]
    freeze = payloads["phase381_joint_scan_freeze.json"]
    result = payloads["phase381_joint_state_summary.json"]
    trace_summaries = [
        payloads["phase381_qwen3_trace_summary.json"],
        payloads["phase381_glm4_trace_summary.json"],
        payloads["phase381_deepseek7b_trace_summary.json"],
    ]
    stage = {
        "schema_version": "54.7.0",
        "phase_id": "Phase381-StageMerge",
        "created_at": updated_at,
        "objective": "test_static_joint_semantic_position_state_as_an_upstream_global_layout_object",
        "assessment": {
            "fresh_behavior_and_trace_denominator_valid": True,
            "single_position_failure_explained_by_joint_position_state": False,
            "joint_state_has_positive_raw_transfer": True,
            "joint_state_has_repeatable_synergy_over_best_single": False,
            "crossmodel_upstream_joint_state_cell_found": False,
            "broader_static_state_swap_should_continue": False,
            "transition_level_operator_needed": True,
            "single_neuron_scan_authorized": False,
            "nine_family_layout_completed": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "base_behavior_cases": behavior["denominator"]["original_behavior_case_count"],
            "behavior_expansion_cases": behavior["denominator"]["expansion_behavior_case_count"],
            "selected_parallel_groups": behavior["denominator"]["selected_parallel_group_count"],
            "selected_trace_cases": behavior["denominator"]["selected_trace_case_count"],
            "exact_event_vectors": sum(row["exact_event_vector_count"] for row in trace_summaries),
            "replay_match_cases": freeze["denominator"]["replay_match_case_count"],
            "replay_qualified_groups": freeze["denominator"]["replay_qualified_group_count"],
            "causal_condition_rows": result["denominator"]["condition_row_count"],
            "joint_direction_gate_rows": result["denominator"]["joint_direction_gate_row_count"],
            "model_cells": result["denominator"]["model_cell_count"],
            "model_cell_passes": result["results"]["model_cell_pass_count"],
            "crossmodel_upstream_cells": result["results"][
                "heterogeneous_upstream_joint_state_cell_count"
            ],
            "complete_language_paths": 0,
            "single_neuron_causal_paths": 0,
            "strict_closure_cells": 0,
            "registered_closure_cells": 72,
        },
        "results": {
            "joint_transfer_gain_median": result["descriptive"][
                "joint_transfer_gain_median"
            ],
            "joint_synergy_gain_median": result["descriptive"][
                "joint_synergy_gain_median"
            ],
            "direction_gate_pass_counts": result["results"][
                "direction_gate_pass_counts"
            ],
            "joint_direction_gate_pass_count": result["results"][
                "joint_direction_gate_pass_count"
            ],
            "maximum_repeated_complete_group_count": result["results"][
                "maximum_all_four_direction_group_pass_count_in_any_model_cell"
            ],
            "model_cell_pass_count": 0,
            "heterogeneous_level2_cell_count": 0,
            "joint_distributed_upstream_state_established": False,
        },
        "hard_limits": [
            "three_mechanisms_not_all_nine_families_are_in_the_joint_scan",
            "raw_joint_transfer_is_large_but_does_not_repeat_as_synergy_over_the_best_single_position",
            "the_best_model_cell_contains_only_one_complete_group_against_a_six_group_gate",
            "full_vector_replacement_can_transfer_generic_content_and_architecture_state_together",
            "the_test_rejects_one_static_joint_state_operator_not_all_possible_dynamic_operators",
            "small_models_may_use_coarse_or_architecture_specific_routes",
        ],
        "authorization": {
            "show_joint_state_as_rejected_hypothesis": True,
            "show_any_phase381_component_as_language_path": False,
            "show_any_phase381_neuron": False,
            "reuse_phase381_groups_to_select_a_new_operator": False,
            "continue_broader_static_role_bundle_scan": False,
            "claim_global_layout_complete": False,
        },
        "next_stage": {
            "phase": 382,
            "objective": "replace_static_state_copy_with_frozen_transition_event_contrasts",
            "first_step": "offline_identifiability_and_conservation_audit_before_new_cuda_data",
            "candidate_basic_operator": "recipient_state_plus_frozen_four_condition_component_update_not_full_donor_state",
            "required_controls": [
                "preserve_recipient_common_backbone",
                "separate_content_main_effect_operation_main_effect_and_interaction_update",
                "predict_heldout_terminal_interface_change_before_intervention",
                "freeze_wrong_depth_wrong_component_equal_energy_and_side_effect_controls",
                "keep_single_neuron_scan_closed_until_a_transition_path_replicates",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    nodes, edges = build_graph(result)
    payloads["phase381_joint_state_stage_summary.json"] = stage
    row_payloads["phase381_evidence_nodes.jsonl"] = nodes
    row_payloads["phase381_evidence_edges.jsonl"] = edges
    published_files = [*payloads.keys(), *row_payloads.keys()]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase381-StageMerge"
        manifest["phase381"] = {
            "status": "static_joint_semantic_position_state_rejected",
            "behavior_case_count": 1152,
            "selected_trace_case_count": 288,
            "exact_event_vector_count": 119808,
            "causal_condition_row_count": 69120,
            "joint_direction_pass_count": 563,
            "model_cell_pass_count": 0,
            "crossmodel_upstream_cell_count": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published_files,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase381-StageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["global_research_estimates"] = {
            "status": "invalid_for_scientific_completion",
            "reason": "a_strong_negative_on_three_joint_role_mechanisms_does_not_measure_nine_family_layout_completion",
            "single_scalar_estimate_valid": False,
        }
        progress["joint_state_stage"] = {
            "selected_behavior_groups": {"numerator": 24, "denominator": 24},
            "replay_qualified_groups": {"numerator": 22, "denominator": 24},
            "causal_condition_rows": {"numerator": 69120, "denominator": 69120},
            "joint_direction_passes": {"numerator": 563, "denominator": 8640},
            "model_cell_passes": {"numerator": 0, "denominator": 300},
            "crossmodel_upstream_cells": {"numerator": 0, "denominator": 1},
            "complete_language_paths": {"numerator": 0, "denominator": 18},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 18},
        }
        progress["phase381_decision"] = (
            "close_static_joint_role_state_and_audit_transition_event_operator_offline"
        )
        write_json(root / "progress.json", progress)
        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase381-StageMerge"
            client_index["latest_stage_files"] = [
                "phase381_joint_state_stage_summary.json",
                "phase381_evidence_nodes.jsonl",
                "phase381_evidence_edges.jsonl",
                "phase381_joint_state_summary.json",
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
