#!/usr/bin/env python3
"""Publish Phase402 multi-parent direct-child evidence to both atlas mirrors."""

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
P402 = ROOT / "tests/gpt5/result/phase402_multiparent_graph"
JSON_SOURCES = {
    "phase402_multiparent_protocol.json": P402 / "phase402_multiparent_protocol.json",
    "phase402_behavior_protocol.json": P402 / "phase402_behavior_protocol.json",
    "phase402_behavior_freeze_summary.json": P402 / "phase402_behavior_freeze_summary.json",
    "phase402_trace_protocol.json": P402 / "phase402_trace_protocol.json",
    "phase402_instrument_audit.json": P402 / "phase402_instrument_audit.json",
    "phase402_discovery_execution_freeze.json": P402 / "phase402_discovery_execution_freeze.json",
    "phase402_discovery_audit.json": P402 / "phase402_discovery_audit.json",
}
CANDIDATE_ROWS = P402 / "discovery_analysis/phase402_group_layer_candidate_rows.jsonl"


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


def strict_local_cells() -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(CANDIDATE_ROWS)
        if row["strict_group_layer_candidate"]
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["surface"],
            row["layer_index"],
            row["public_parallel_group_id"],
        ),
    )


def stage_summary(
    updated_at: str,
    payloads: dict[str, Any],
    strict_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    behavior = payloads["phase402_behavior_freeze_summary.json"]
    instrument = payloads["phase402_instrument_audit.json"]
    discovery = payloads["phase402_discovery_audit.json"]
    semantic_correct = sum(
        row["semantic_correct_count"] for row in behavior["model_results"].values()
    )
    instrument_rows = sum(row["row_count"] for row in instrument["models"].values())
    instrument_pass = sum(
        row["passing_row_count"] for row in instrument["models"].values()
    )
    strict_by_model = {
        model: sum(row["model"] == model for row in strict_rows)
        for model in ("qwen3", "glm4", "deepseek7b")
    }
    return {
        "schema_version": "76.9.0",
        "phase_id": "Phase402-MultiParentDirectChildStage",
        "created_at": updated_at,
        "objective": (
            "test_four_disjoint_attention_KV_parent_partitions_against_best_"
            "singletons_and_each_registered_control_at_the_direct_query_child"
        ),
        "assessment": {
            "six_surface_behavior_denominator_frozen": True,
            "four_parent_partition_disjoint_and_prefix_conserving": True,
            "remaining_prefix_is_generated_history": False,
            "same_shape_instrument_pass": instrument["valid"],
            "strict_local_joint_cells_observed": bool(strict_rows),
            "model_level_joint_parent_candidate_observed": False,
            "crossmodel_joint_parent_candidate_observed": False,
            "calibration_opened": False,
            "physical_holdout_opened": False,
            "propagation_terminal_or_neuron_search_authorized": False,
            "causal_state_quotient_or_operator_algebra_established": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "behavior_candidate_case_count": behavior["denominator"]["candidate_case_count"],
            "behavior_semantic_correct_case_count": semantic_correct,
            "behavior_parallel_group_count": behavior["denominator"]["candidate_parallel_group_count"],
            "behavior_complete_three_model_group_count": behavior["denominator"]["qualified_parallel_group_count"],
            "registered_surface_count": 6,
            "eligible_surface_count": behavior["denominator"]["eligible_surface_count"],
            "selected_trace_case_count": behavior["denominator"]["selected_case_count"],
            "instrument_row_count": instrument_rows,
            "instrument_pass_row_count": instrument_pass,
            "discovery_model_case_count": discovery["denominator"]["collection_case_count"],
            "discovery_pair_metric_count": discovery["denominator"]["pair_metric_count"],
            "discovery_group_layer_subset_metric_count": discovery["denominator"]["group_layer_subset_metric_count"],
            "joint_group_layer_subset_count": discovery["denominator"]["joint_group_layer_subset_count"],
            "calibration_case_count_consumed": 0,
            "calibration_case_count_reserved": 288,
            "physical_holdout_case_count_consumed": 0,
            "physical_holdout_case_count_reserved": 288,
        },
        "results": {
            **discovery["gate_flow"],
            "strict_local_cell_count_by_model": strict_by_model,
            "strict_local_subset_ids": sorted({row["subset_id"] for row in strict_rows}),
            "strict_local_depth_zones": sorted({row["depth_zone"] for row in strict_rows}),
            "validated_crossmodel_joint_parent_surface_count": 0,
            "joint_causal_intervention_count": 0,
            "new_head_channel_or_neuron_node_count": 0,
            "complete_language_path_count": 0,
        },
        "protocol_precision_audit": discovery["protocol_precision_audit"],
        "hard_limits": [
            "only_role_filling_and_conditional_presence_met_the_frozen_three_split_behavior_gate",
            "two_step_composition_and_number_agreement_had_zero_complete_crossmodel_groups",
            "the_four_parent_categories_are_broad_token_partitions_not_discovered_semantic_variables",
            "only_the_direct_query_attention_child_was_recomputed",
            "no_terminal_generation_natural_necessity_propagation_or_neuron_effect_was_measured",
            "the_sign_permutation_is_a_measurement_control_not_a_legal_compute_graph_parent_patch",
            "all_eight_strict_local_cells_use_source_structure_plus_query_local_in_early_layers_but_none_reaches_a_model_level_candidate",
            "the_stored_decimal_group_rate_threshold_requires_five_of_six_groups_and_was_not_relaxed_after_results",
            "the_negative_result_does_not_exclude_finer_dynamic_cross_layer_or_multi_component_parent_states",
        ],
        "authorization": {
            "show_four_parent_partition_ledger": True,
            "show_strict_local_joint_response_cells": True,
            "show_model_level_joint_parent_candidate": False,
            "show_crossmodel_language_path": False,
            "show_specific_head_channel_or_neuron": False,
            "run_calibration": False,
            "run_physical_holdout": False,
            "run_propagation_terminal": False,
            "run_single_neuron_scan": False,
        },
        "next_stage": {
            "objective": (
                "repair_exact_count_thresholds_then_freeze_a_dynamic_cross_layer_"
                "multi_component_parent_chain_with_fresh_behavior_groups"
            ),
            "automatic_model_execution_authorized": False,
            "reason": (
                "the_current_four_partition_direct_child_hypothesis_is_closed_"
                "and_the_next_experiment_changes_the_causal_object_and_protocol"
            ),
            "required_changes": [
                "store_integer_required_group_counts_instead_of_decimal_rates",
                "separate_prompt_prefix_context_from_generated_history",
                "replace_broad_parent_partitions_with_stage_specific_attention_and_MLP_parent_events",
                "freeze_direct_child_propagation_and_terminal_gates_before_collection",
                "use_fresh_discovery_calibration_and_physical_groups",
            ],
            "causal_or_neuron_work_authorized": False,
        },
        "single_global_progress_percentage_valid": False,
    }


def evidence_graph(
    strict_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p402_six_surface_behavior_denominator",
            "node_type": "fresh_six_surface_factorial_behavior_denominator",
            "phase_id": "Phase402",
            "case_count": 6912,
            "semantic_correct_case_count": 5585,
            "eligible_surface_count": 2,
            "registered_surface_count": 6,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p402_four_parent_partition_ledger",
            "node_type": "disjoint_attention_KV_parent_partition_ledger",
            "phase_id": "Phase402",
            "passing_row_count": 3328,
            "row_count": 3328,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p402_multiparent_direct_child_discovery",
            "node_type": "four_partition_direct_query_attention_intervention",
            "phase_id": "Phase402",
            "case_count": 576,
            "pair_metric_count": 2875392,
            "joint_group_layer_subset_count": 13728,
            "interventional": True,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p402_S0110_early_local_hint",
            "node_type": "local_source_structure_plus_query_context_response_hint",
            "phase_id": "Phase402",
            "subset_id": "S0110",
            "strict_local_cell_count": len(strict_rows),
            "qwen3_cell_count": 0,
            "glm4_cell_count": 1,
            "deepseek7b_cell_count": 7,
            "model_level_candidate": False,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p402_decimal_threshold_precision_audit",
            "node_type": "frozen_decimal_threshold_effective_count_audit",
            "phase_id": "Phase402",
            "stored_rate": 0.666666667,
            "group_denominator": 6,
            "effective_required_count": 5,
            "posthoc_relaxation_applied": False,
            "causal": False,
            "language_path": False,
        },
        {
            "node_id": "p402_validation_causal_neuron_gates_closed",
            "node_type": "validation_physical_causal_and_neuron_gates_closed",
            "phase_id": "Phase402",
            "model_candidate_count": 0,
            "crossmodel_candidate_count": 0,
            "calibration_cases_consumed": 0,
            "physical_cases_consumed": 0,
            "neuron_nodes_promoted": 0,
            "causal": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p401_to_p402_multi_parent_contract",
            "source_node_id": "p401_nonspecific_local_response",
            "target_node_id": "p402_six_surface_behavior_denominator",
            "edge_type": "motivates_fresh_multi_parent_contract",
            "phase_id": "Phase402",
            "causal_path": False,
        },
        {
            "edge_id": "p402_behavior_to_parent_ledger",
            "source_node_id": "p402_six_surface_behavior_denominator",
            "target_node_id": "p402_four_parent_partition_ledger",
            "edge_type": "qualifies_selected_surface_parent_partition_measurement",
            "phase_id": "Phase402",
            "causal_path": False,
        },
        {
            "edge_id": "p402_ledger_to_discovery",
            "source_node_id": "p402_four_parent_partition_ledger",
            "target_node_id": "p402_multiparent_direct_child_discovery",
            "edge_type": "enables_direct_child_subset_recomputation",
            "phase_id": "Phase402",
            "causal_path": False,
        },
        {
            "edge_id": "p402_discovery_to_S0110_hint",
            "source_node_id": "p402_multiparent_direct_child_discovery",
            "target_node_id": "p402_S0110_early_local_hint",
            "edge_type": "contains_eight_strict_local_cells_without_model_replication",
            "phase_id": "Phase402",
            "causal_path": False,
        },
        {
            "edge_id": "p402_precision_to_gate_boundary",
            "source_node_id": "p402_decimal_threshold_precision_audit",
            "target_node_id": "p402_validation_causal_neuron_gates_closed",
            "edge_type": "records_frozen_effective_five_of_six_requirement",
            "phase_id": "Phase402",
            "causal_path": False,
        },
        {
            "edge_id": "p402_discovery_closes_gates",
            "source_node_id": "p402_multiparent_direct_child_discovery",
            "target_node_id": "p402_validation_causal_neuron_gates_closed",
            "edge_type": "zero_model_and_crossmodel_candidates_stop_validation",
            "phase_id": "Phase402",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_progress(root: Path, updated_at: str) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    progress["last_phase"] = "Phase402-MultiParentDirectChildStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["multiparent_direct_child_stage"] = {
        "behavior_candidate_cases": {"numerator": 6912, "denominator": 6912},
        "semantic_correct_cases": {"numerator": 5585, "denominator": 6912},
        "complete_three_model_groups": {"numerator": 68, "denominator": 144},
        "eligible_surfaces": {"numerator": 2, "denominator": 6},
        "instrument_rows": {"numerator": 3328, "denominator": 3328},
        "discovery_model_cases": {"numerator": 576, "denominator": 576},
        "true_base_joint_cells": {"numerator": 5909, "denominator": 13728},
        "above_best_singleton_cells": {"numerator": 641, "denominator": 13728},
        "all_control_specific_cells": {"numerator": 18, "denominator": 13728},
        "strict_local_joint_cells": {"numerator": 8, "denominator": 13728},
        "model_level_joint_parent_candidates": {"numerator": 0, "denominator": 12},
        "crossmodel_joint_parent_surfaces": {"numerator": 0, "denominator": 2},
        "calibration_cases_consumed": {"numerator": 0, "denominator": 288},
        "physical_holdout_cases_consumed": {"numerator": 0, "denominator": 288},
        "joint_causal_intervention_authorized": {"numerator": 0, "denominator": 1},
        "complete_language_paths": {"numerator": 0, "denominator": 72},
        "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase402_decision"] = (
        "retain_eight_S0110_early_local_cells_as_nonregistered_hints_close_"
        "four_partition_direct_child_hypothesis_and_all_downstream_gates"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if not CANDIDATE_ROWS.is_file():
        missing.append(str(CANDIDATE_ROWS))
    if missing:
        raise FileNotFoundError(f"Missing Phase402 artifacts: {missing}")

    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    strict_rows = strict_local_cells()
    if len(strict_rows) != 8:
        raise RuntimeError(f"Expected 8 strict Phase402 local cells, got {len(strict_rows)}")
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = stage_summary(updated_at, payloads, strict_rows)
    payloads["phase402_multiparent_direct_child_stage_summary.json"] = stage
    nodes, edges = evidence_graph(strict_rows)
    published = [
        *payloads,
        "phase402_strict_local_cells.jsonl",
        "phase402_evidence_nodes.jsonl",
        "phase402_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase402_strict_local_cells.jsonl", strict_rows)
        write_jsonl(root / "phase402_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase402_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase402-MultiParentDirectChildStage"
        manifest["phase402"] = {
            "status": (
                "four_parent_direct_child_discovery_complete_eight_local_hints_"
                "zero_model_or_crossmodel_candidates_all_downstream_gates_closed"
            ),
            **stage["results"],
            "files": published,
        }
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase402-MultiParentDirectChildStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        update_checksums(root)
        public_manifest(root, updated_at)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase402_multiparent_direct_child_stage_summary.json", stage)
        write_jsonl(root / "phase402_strict_local_cells.jsonl", strict_rows)
        write_jsonl(root / "phase402_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase402_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 402
        manifest["generated_at"] = updated_at
        manifest["phase402_audit"] = {
            "status": (
                "eight_local_joint_cells_but_zero_model_level_or_crossmodel_"
                "joint_parent_candidates_and_zero_neuron_nodes_promoted"
            ),
            "behavior_candidate_case_count": 6912,
            "behavior_semantic_correct_case_count": 5585,
            "behavior_eligible_surface_count": 2,
            "behavior_registered_surface_count": 6,
            "instrument_pass_row_count": 3328,
            "instrument_row_count": 3328,
            "discovery_pair_metric_count": 2875392,
            "joint_group_layer_subset_count": 13728,
            "strict_local_joint_cell_count": 8,
            "model_level_joint_parent_candidate_count": 0,
            "crossmodel_joint_parent_surface_count": 0,
            "physical_holdout_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase402_multiparent_direct_child_stage_summary.json",
        }
        manifest.setdefault("metrics", {}).update(
            {
                "phase402_behavior_candidate_case_count": 6912,
                "phase402_behavior_semantic_correct_case_count": 5585,
                "phase402_eligible_surface_count": 2,
                "phase402_registered_surface_count": 6,
                "phase402_instrument_pass_row_count": 3328,
                "phase402_instrument_row_count": 3328,
                "phase402_discovery_pair_metric_count": 2875392,
                "phase402_joint_group_layer_subset_count": 13728,
                "phase402_true_base_joint_cell_count": 5909,
                "phase402_above_best_singleton_cell_count": 641,
                "phase402_all_control_specific_cell_count": 18,
                "phase402_strict_local_joint_cell_count": 8,
                "phase402_model_level_joint_parent_candidate_count": 0,
                "phase402_crossmodel_joint_parent_surface_count": 0,
                "phase402_calibration_case_count": 0,
                "phase402_physical_holdout_case_count": 0,
                "phase402_joint_causal_intervention_count": 0,
                "phase402_new_neuron_node_count": 0,
            }
        )
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase402_multiparent_direct_child_stage_summary.json"
        manifest["evidence_boundary"] = {
            "latest_phase": "Phase402-MultiParentDirectChildStage",
            "statement": (
                "Phase402 froze 6,912 behavior cases over six task surfaces and "
                "qualified two surfaces. All 3,328 parent-partition replay rows "
                "passed. Discovery retained 2,875,392 private pair metrics and "
                "tested 13,728 joint group-layer subsets against the best "
                "contained singleton and every applicable control. Eight early "
                "source-structure plus query-local cells passed locally, but "
                "there were zero model-level and zero crossmodel candidates. "
                "Calibration, physical holdout, terminal, causal, and neuron "
                "gates remain closed."
            ),
            "aggregate_execution_ledger_available": True,
            "multi_parent_direct_response_profile_available": True,
            "strict_local_joint_hint_available": True,
            "model_level_joint_parent_candidate_available": False,
            "crossmodel_joint_parent_candidate_available": False,
            "validated_local_edge_rule_available": False,
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
