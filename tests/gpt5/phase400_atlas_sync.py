#!/usr/bin/env python3
"""Publish Phase400 partial-order discovery and calibration failure boundaries."""

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
P400 = ROOT / "tests/gpt5/result/phase400_partial_order"
JSON_SOURCES = {
    "phase400_protocol.json": P400 / "phase400_protocol.json",
    "phase400_behavior_freeze_summary.json": P400
    / "phase400_behavior_freeze_summary.json",
    "phase400_partial_order_protocol.json": P400
    / "phase400_partial_order_protocol.json",
    "phase400_dynamic_trace_protocol.json": P400
    / "phase400_dynamic_trace_protocol.json",
    "phase400_instrument_audit.json": P400 / "phase400_instrument_audit.json",
    "phase400_partial_order_discovery.json": P400
    / "phase400_partial_order_discovery.json",
    "phase400_partial_order_candidate_freeze.json": P400
    / "phase400_partial_order_candidate_freeze.json",
    "phase400_calibration_collection_quality_audit.json": P400
    / "phase400_calibration_collection_quality_audit.json",
    "phase400_partial_order_calibration.json": P400
    / "phase400_partial_order_calibration.json",
    "phase400_partial_order_physical.json": P400
    / "phase400_partial_order_physical.json",
}


def stage_summary(updated_at: str, payloads: dict[str, Any]) -> dict[str, Any]:
    behavior = payloads["phase400_behavior_freeze_summary.json"]
    discovery = payloads["phase400_partial_order_discovery.json"]
    quality = payloads["phase400_calibration_collection_quality_audit.json"]
    field_pilot = [
        read_json(P400 / f"field_contract_pilot/{model}/complete.json")
        for model in ("qwen3", "glm4", "deepseek7b")
    ]
    return {
        "schema_version": "74.9.0",
        "phase_id": "Phase400-DynamicPartialOrderStage",
        "created_at": updated_at,
        "objective": "replace_single_peak_order_with_frozen_interval_partial_order_graphs_then_require_prediction_and_independent_validation_before_causal_or_neuron_search",
        "assessment": {
            "fresh_four_surface_behavior_denominator": True,
            "field_prompt_contract_pilot_three_model_pass": all(
                item["valid"] for item in field_pilot
            ),
            "behavior_eligible_surface_count": len(behavior["eligible_surfaces"]),
            "discovery_collection_quality_pass": discovery["denominator"][
                "all_collection_quality_gates_pass"
            ],
            "interval_partial_order_graph_observed": True,
            "crossmodel_discovery_isomorphism_observed": True,
            "terminal_answer_prediction_pass": False,
            "calibration_collection_quality_pass": False,
            "physical_holdout_opened": False,
            "joint_causal_intervention_authorized": False,
            "single_neuron_search_authorized": False,
            "complete_binding_algorithm_established": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "field_contract_pilot_case_count": sum(
                item["case_count"] for item in field_pilot
            ),
            "behavior_candidate_case_count": behavior["denominator"][
                "candidate_case_count"
            ],
            "behavior_three_model_complete_group_count": behavior["denominator"][
                "qualified_parallel_group_count"
            ],
            "registered_surface_count": 4,
            "eligible_surface_count": len(behavior["eligible_surfaces"]),
            "selected_internal_case_count": behavior["denominator"][
                "selected_case_count"
            ],
            "instrument_case_count": behavior["denominator"]["instrument_case_count"],
            "discovery_case_count": discovery["denominator"]["case_count"],
            "discovery_group_model_cell_count": discovery["denominator"][
                "group_model_cell_count"
            ],
            "discovery_event_trajectory_row_count": discovery["denominator"][
                "event_trajectory_row_count"
            ],
            "calibration_case_count": quality["denominator"]["case_count"],
            "calibration_group_model_cell_count": quality["denominator"][
                "group_model_cell_count"
            ],
            "physical_case_count_consumed": 0,
        },
        "results": {
            "discovery_partial_order_graph_cell_count": discovery["results"][
                "partial_order_graph_cell_count"
            ],
            "discovery_model_surface_cell_count": discovery["results"][
                "model_surface_cell_count"
            ],
            "discovery_crossmodel_isomorphism_surface_count": discovery["results"][
                "crossmodel_isomorphism_surface_count"
            ],
            "discovery_prediction_pass_cell_count": discovery["results"][
                "prediction_pass_cell_count"
            ],
            "calibration_quality_group_model_cell_count": quality["denominator"][
                "quality_group_model_cell_count"
            ],
            "calibration_group_model_cell_count": quality["denominator"][
                "group_model_cell_count"
            ],
            "calibration_first_answer_replay_match_count": quality["denominator"][
                "first_answer_replay_match_count"
            ],
            "calibration_target_completion_replay_match_count": quality["denominator"][
                "target_completion_replay_match_count"
            ],
            "batch_size_invariant_first_format_token": quality["diagnosis"][
                "batch_size_1_vs_8_first_token_invariance"
            ],
            "parent_capture_hook_changed_single_case_top1": quality["diagnosis"][
                "parent_capture_hooks_changed_current_single_case_top1"
            ],
            "validated_crossmodel_partial_order_surface_count": 0,
            "joint_causal_intervention_count": 0,
            "new_neuron_path_node_count": 0,
            "complete_language_path_count": 0,
        },
        "hard_limits": [
            "only_two_of_four_registered_surfaces_met_the_frozen_three_split_behavior_gate",
            "five_of_six_discovery_model_surface_cells_passed_the_interval_graph_gate_but_zero_of_six_passed_the_answer_prediction_gate",
            "the_single_discovery_crossmodel_isomorphism_is_an_observational_functional_type_match_not_identical_layers_heads_or_neurons",
            "the_graph_readout_was_between_48_4_and_59_4_percent_and_did_not_beat_the_frozen_controls",
            "one_DS7B_calibration_case_changed_its_first_format_token_between_batch_size_eight_and_one",
            "the_frozen_exact_first_answer_replay_gate_failed_even_though_semantic_target_completion_and_post_target_replay_were_384_of_384",
            "calibration_event_metrics_were_not_analyzed_after_collection_invalidity",
            "the_physical_holdout_remains_unopened_and_no_causal_or_fine_resolution_search_was_run",
            "the_next_time_recovery_fields_are_structural_recovery_checks_not_a_learned_forecast_of_future_events",
        ],
        "authorization": {
            "show_discovery_partial_order_candidate": True,
            "show_candidate_as_observational_only": True,
            "show_candidate_as_validated_rule": False,
            "show_candidate_as_causal_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_joint_causal_intervention": False,
            "run_single_neuron_scan": False,
            "claim_language_encoding_closure": False,
        },
        "next_stage": {
            "objective": "freeze_a_batch_invariant_semantic_decision_coordinate_and_match_behavior_and_trace_execution_shapes_before_restarting_partial_order_validation_on_a_fresh_denominator",
            "automatic_continuation_authorized": True,
            "required_changes": [
                "separate_format_prefix_tokens_from_the_semantic_answer_decision",
                "run_behavior_and_trace_qualification_at_matched_batch_shape_or_predeclare_batch_sensitivity",
                "compare_plain_attention_output_and_hooked_topk_logits_before_event_collection",
                "use_a_fresh_calibration_denominator_after_the_contract_change",
                "keep_the_current_Phase400_physical_holdout_sealed_for_Phase400_claims",
            ],
            "causal_or_neuron_work_authorized": False,
        },
        "single_global_progress_percentage_valid": False,
    }


def evidence_graph(stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p400_fresh_partial_order_denominator",
            "node_type": "fresh_dynamic_partial_order_denominator",
            "phase_id": "Phase400",
            "behavior_case_count": 4608,
            "eligible_surface_count": 2,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p400_interval_event_graph_discovery",
            "node_type": "aggregate_interval_partial_order_graph_discovery",
            "phase_id": "Phase400",
            "trace_case_count": 768,
            "passing_model_surface_cells": 5,
            "cell_denominator": 6,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p400_possession_crossmodel_observation",
            "node_type": "crossmodel_functional_partial_order_candidate",
            "phase_id": "Phase400",
            "surface": "possession_relation",
            "model_count": 3,
            "discovery_only": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p400_prediction_gate_negative",
            "node_type": "partial_order_terminal_prediction_negative",
            "phase_id": "Phase400",
            "passing_cells": 0,
            "cell_denominator": 6,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p400_batch_shape_replay_failure",
            "node_type": "batch_shape_sensitive_format_token_replay_failure",
            "phase_id": "Phase400",
            "model": "deepseek7b",
            "failed_cases": 1,
            "case_denominator": 384,
            "parent_capture_hook_changed_top1": False,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p400_validation_and_causal_gate_closed",
            "node_type": "validation_physical_causal_and_neuron_gates_closed",
            "phase_id": "Phase400",
            "physical_case_count": 0,
            "joint_intervention_count": 0,
            "neuron_scan_count": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p399_to_p400_interval_graph",
            "source_node_id": "p399_role_partitioned_event_ledger",
            "target_node_id": "p400_interval_event_graph_discovery",
            "edge_type": "replaces_peak_order_with_interval_partial_order",
            "phase_id": "Phase400",
            "causal_path": False,
        },
        {
            "edge_id": "p400_denominator_to_discovery",
            "source_node_id": "p400_fresh_partial_order_denominator",
            "target_node_id": "p400_interval_event_graph_discovery",
            "edge_type": "qualifies_discovery_trace",
            "phase_id": "Phase400",
            "causal_path": False,
        },
        {
            "edge_id": "p400_discovery_to_crossmodel_candidate",
            "source_node_id": "p400_interval_event_graph_discovery",
            "target_node_id": "p400_possession_crossmodel_observation",
            "edge_type": "relative_onset_and_duration_isomorphism",
            "phase_id": "Phase400",
            "causal_path": False,
        },
        {
            "edge_id": "p400_candidate_to_prediction_negative",
            "source_node_id": "p400_possession_crossmodel_observation",
            "target_node_id": "p400_prediction_gate_negative",
            "edge_type": "fails_terminal_answer_and_control_prediction",
            "phase_id": "Phase400",
            "causal_path": False,
        },
        {
            "edge_id": "p400_calibration_shape_failure",
            "source_node_id": "p400_batch_shape_replay_failure",
            "target_node_id": "p400_validation_and_causal_gate_closed",
            "edge_type": "invalidates_calibration_denominator_and_keeps_holdout_sealed",
            "phase_id": "Phase400",
            "causal_path": False,
        },
        {
            "edge_id": "p400_prediction_closes_causal_gate",
            "source_node_id": "p400_prediction_gate_negative",
            "target_node_id": "p400_validation_and_causal_gate_closed",
            "edge_type": "independently_blocks_causal_and_neuron_search",
            "phase_id": "Phase400",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_checksums(root: Path) -> None:
    path = root / "checksums.json"
    if path.is_file():
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


def update_progress(root: Path, updated_at: str) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    progress["last_phase"] = "Phase400-DynamicPartialOrderStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["dynamic_partial_order_stage"] = {
        "behavior_candidate_cases": {"numerator": 4608, "denominator": 4608},
        "eligible_surfaces": {"numerator": 2, "denominator": 4},
        "discovery_quality_group_model_cells": {"numerator": 48, "denominator": 48},
        "discovery_partial_order_graph_cells": {"numerator": 5, "denominator": 6},
        "discovery_crossmodel_isomorphism_surfaces": {"numerator": 1, "denominator": 2},
        "discovery_prediction_cells": {"numerator": 0, "denominator": 6},
        "calibration_quality_group_model_cells": {"numerator": 23, "denominator": 24},
        "validated_crossmodel_partial_order_surfaces": {"numerator": 0, "denominator": 2},
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 384},
        "joint_causal_intervention_authorized": {"numerator": 0, "denominator": 1},
        "complete_language_paths": {"numerator": 0, "denominator": 72},
        "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase400_decision"] = (
        "retain_discovery_only_possession_partial_order_candidate_close_prediction_"
        "validation_physical_causal_and_neuron_gates_fix_batch_semantic_contract_next"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase400 artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads)
    payloads["phase400_dynamic_partial_order_stage_summary.json"] = stage
    nodes, edges = evidence_graph(stage)
    published = [
        *payloads,
        "phase400_evidence_nodes.jsonl",
        "phase400_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase400_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase400_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase400-DynamicPartialOrderStage"
        manifest["phase400"] = {
            "status": "discovery_partial_order_candidate_prediction_negative_calibration_batch_contract_failure_physical_and_causal_gates_closed",
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase400-DynamicPartialOrderStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        update_checksums(root)
        public_manifest(root, updated_at)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase400_dynamic_partial_order_stage_summary.json", stage)
        write_jsonl(root / "phase400_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase400_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 400
        manifest["generated_at"] = updated_at
        manifest["phase400_audit"] = {
            "status": "no_neuron_nodes_promoted_discovery_graph_unvalidated_and_nonpredictive",
            "discovery_partial_order_graph_cell_count": 5,
            "discovery_crossmodel_isomorphism_surface_count": 1,
            "discovery_prediction_pass_cell_count": 0,
            "calibration_quality_group_model_cell_count": 23,
            "calibration_group_model_cell_count": 24,
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase400_dynamic_partial_order_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase400_behavior_candidate_case_count": 4608,
                "phase400_discovery_trace_case_count": 768,
                "phase400_discovery_partial_order_graph_cell_count": 5,
                "phase400_discovery_crossmodel_isomorphism_surface_count": 1,
                "phase400_discovery_prediction_pass_cell_count": 0,
                "phase400_calibration_quality_group_model_cell_count": 23,
                "phase400_calibration_group_model_cell_count": 24,
                "phase400_physical_holdout_case_count": 0,
                "phase400_joint_causal_intervention_count": 0,
                "phase400_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase400_dynamic_partial_order_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase400-DynamicPartialOrderStage",
            "statement": "Phase400 found one discovery-only crossmodel interval partial-order candidate, but 0/6 graph readouts passed prediction and one DS7B calibration case failed the frozen exact first-format-token replay gate because batch size 8 and 1 selected different semantically correct answer forms. Calibration was not analyzed, the physical holdout remains sealed, and no head, channel, or neuron node was promoted.",
            "aggregate_partial_order_candidate_available": True,
            "validated_partial_order_rule_available": False,
            "terminal_prediction_available": False,
            "joint_causal_mediation_available": False,
            "upstream_language_path_available": False,
            "single_unit_causal_closure": False,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        update_checksums(root)
        public_manifest(root, updated_at)

    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
