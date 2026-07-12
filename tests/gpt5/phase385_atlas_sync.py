#!/usr/bin/env python3
"""Publish strict Phase383-385 exact-event evidence to both atlas mirrors."""

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
P383 = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
P384 = ROOT / "tests/gpt5/result/phase384_exact_subunit_mass_map"
P385 = ROOT / "tests/gpt5/result/phase385_opposing_mass_specificity"

JSON_SOURCES = {
    "phase383_protocol.json": P383 / "phase383_protocol.json",
    "phase383_single_path_qualification_summary.json": P383
    / "phase383_single_path_qualification_summary.json",
    "phase383_instrument_audit_summary.json": P383
    / "phase383_instrument_audit_summary.json",
    "phase383_signed_event_contract.json": P383 / "phase383_signed_event_contract.json",
    "phase383_signed_event_map_summary.json": P383
    / "phase383_signed_event_map_summary.json",
    "phase383_calibration_summary.json": P383 / "phase383_calibration_summary.json",
    "phase384_subunit_mass_contract.json": P384 / "phase384_subunit_mass_contract.json",
    "phase384_discovery_summary.json": P384 / "phase384_discovery_summary.json",
    "phase384_calibration_summary.json": P384 / "phase384_calibration_summary.json",
    "phase385_specificity_contract.json": P385 / "phase385_specificity_contract.json",
    "phase385_specificity_summary.json": P385 / "phase385_specificity_summary.json",
}

JSONL_SOURCES = {
    "phase383_model_event_cells.jsonl": P383 / "phase383_model_event_cells.jsonl",
    "phase383_crossmodel_event_cells.jsonl": P383
    / "phase383_crossmodel_event_cells.jsonl",
    "phase383_reuse_difference_matrix.jsonl": P383
    / "phase383_reuse_difference_matrix.jsonl",
    "phase383_calibration_replication_rows.jsonl": P383
    / "phase383_calibration_replication_rows.jsonl",
    "phase384_discovery_crossmodel_patterns.jsonl": P384
    / "phase384_discovery_crossmodel_patterns.jsonl",
    "phase384_calibration_crossmodel_patterns.jsonl": P384
    / "phase384_calibration_crossmodel_patterns.jsonl",
    "phase384_calibration_replication_rows.jsonl": P384
    / "phase384_calibration_replication_rows.jsonl",
    "phase385_candidate_specificity_rows.jsonl": P385
    / "phase385_candidate_specificity_rows.jsonl",
    "phase385_specificity_control_rows.jsonl": P385
    / "phase385_specificity_control_rows.jsonl",
}


def graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {
            "node_id": "p383_single_sample_runtime_contract",
            "node_type": "execution_contract_repair",
            "phase_id": "Phase383",
            "status": "batch_qualified_denominator_rebuilt_on_single_sample_path",
            "models": ["qwen3", "glm4", "deepseek7b"],
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p383_exact_component_event_ledger",
            "node_type": "exact_replayable_event_ledger",
            "phase_id": "Phase383",
            "status": "attention_source_and_mlp_channel_families_conserved",
            "top_k_used": False,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p383_late_current_event_layout",
            "node_type": "calibrated_descriptive_layout",
            "phase_id": "Phase383",
            "status": "24_of_32_late_current_candidates_replicated_no_upstream_candidate",
            "upstream": False,
            "terminal_interface": True,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p384_all_head_channel_mass",
            "node_type": "all_subunit_projection_mass_map",
            "phase_id": "Phase384",
            "status": "all_heads_and_channels_included_without_top_k",
            "top_k_used": False,
            "single_neuron_identity": False,
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p384_entity_recency_opposing_mass",
            "node_type": "replicated_opposing_channel_mass",
            "phase_id": "Phase384",
            "mechanism_id": "entity_recency",
            "contrast_axis": "operation",
            "receiver_role": "source",
            "depth_bin": 5,
            "status": "level3_replication_net_projection_near_zero",
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p384_number_agreement_opposing_mass",
            "node_type": "replicated_opposing_channel_mass",
            "phase_id": "Phase384",
            "mechanism_id": "number_agreement",
            "contrast_axis": "operation",
            "receiver_role": "source",
            "depth_bin": 5,
            "status": "heterogeneous_level2_replication_net_projection_near_zero",
            "language_path": False,
            "causal": False,
        },
        {
            "node_id": "p385_specificity_control_negative",
            "node_type": "matched_control_negative_result",
            "phase_id": "Phase385",
            "status": "zero_of_two_opposing_mass_patterns_function_specific",
            "language_path": False,
            "causal": False,
            "single_unit_causal": False,
        },
        {
            "node_id": "p386_multi_time_relation_graph_unresolved",
            "node_type": "unresolved_algorithmic_object",
            "phase_id": "Phase385-StageMerge",
            "status": "requires_new_registered_multi_time_relation_protocol",
            "language_path": False,
            "causal": False,
        },
    ]
    edges = [
        {
            "edge_id": "p383_runtime->p383_ledger",
            "source_node_id": "p383_single_sample_runtime_contract",
            "target_node_id": "p383_exact_component_event_ledger",
            "edge_type": "qualifies_exact_measurement_denominator",
            "phase_id": "Phase383",
            "causal_path": False,
        },
        {
            "edge_id": "p383_ledger->p383_late_layout",
            "source_node_id": "p383_exact_component_event_ledger",
            "target_node_id": "p383_late_current_event_layout",
            "edge_type": "supports_signed_event_mapping",
            "phase_id": "Phase383",
            "causal_path": False,
        },
        {
            "edge_id": "p383_ledger->p384_mass",
            "source_node_id": "p383_exact_component_event_ledger",
            "target_node_id": "p384_all_head_channel_mass",
            "edge_type": "lazy_exact_subunits_expanded_without_top_k",
            "phase_id": "Phase384",
            "causal_path": False,
        },
        {
            "edge_id": "p384_mass->p384_entity",
            "source_node_id": "p384_all_head_channel_mass",
            "target_node_id": "p384_entity_recency_opposing_mass",
            "edge_type": "discovers_opposing_projection_mass",
            "phase_id": "Phase384",
            "causal_path": False,
        },
        {
            "edge_id": "p384_mass->p384_number",
            "source_node_id": "p384_all_head_channel_mass",
            "target_node_id": "p384_number_agreement_opposing_mass",
            "edge_type": "discovers_opposing_projection_mass",
            "phase_id": "Phase384",
            "causal_path": False,
        },
        {
            "edge_id": "p384_entity->p385_negative",
            "source_node_id": "p384_entity_recency_opposing_mass",
            "target_node_id": "p385_specificity_control_negative",
            "edge_type": "fails_function_role_or_depth_specificity_controls",
            "phase_id": "Phase385",
            "causal_path": False,
        },
        {
            "edge_id": "p384_number->p385_negative",
            "source_node_id": "p384_number_agreement_opposing_mass",
            "target_node_id": "p385_specificity_control_negative",
            "edge_type": "fails_function_role_or_depth_specificity_controls",
            "phase_id": "Phase385",
            "causal_path": False,
        },
        {
            "edge_id": "p385_negative->p386_unresolved",
            "source_node_id": "p385_specificity_control_negative",
            "target_node_id": "p386_multi_time_relation_graph_unresolved",
            "edge_type": "requires_relation_aware_temporal_coordinates",
            "phase_id": "Phase385-StageMerge",
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
    retained_json = (
        "phase371_exact_vector_stage_summary.json",
        "phase378_decision_aligned_stage_summary.json",
        "phase380_global_layout_stage_summary.json",
        "phase381_joint_state_stage_summary.json",
        "phase382_transition_stage_summary.json",
    )
    retained_jsonl = (
        "phase371_evidence_nodes.jsonl",
        "phase371_evidence_edges.jsonl",
        "phase378_evidence_nodes.jsonl",
        "phase378_evidence_edges.jsonl",
        "phase380_evidence_nodes.jsonl",
        "phase380_evidence_edges.jsonl",
        "phase381_evidence_nodes.jsonl",
        "phase381_evidence_edges.jsonl",
        "phase382_evidence_nodes.jsonl",
        "phase382_evidence_edges.jsonl",
    )
    for root in (NEURON_TARGET, NEURON_CLIENT):
        for name in retained_json:
            source = TARGET / name
            if source.is_file():
                write_json(root / name, read_json(source))
        for name in retained_jsonl:
            source = TARGET / name
            if source.is_file():
                write_jsonl(root / name, read_jsonl(source))
        write_json(root / "phase385_exact_event_stage_summary.json", stage)
        write_jsonl(root / "phase385_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase385_evidence_edges.jsonl", edges)
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["phase"] = 385
        manifest["generated_at"] = updated_at
        manifest.setdefault(
            "phase380_audit",
            {
                "status": "terminal_interface_only_no_upstream_or_neuron_path",
                "replicated_residual_object_count": 5,
                "terminal_interface_boundary_count": 2,
                "terminal_interface_mechanism_count": 3,
                "upstream_crossmodel_cell_count": 0,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "language_path_count": 0,
                "source": "phase380_global_layout_stage_summary.json",
            },
        )
        manifest.setdefault(
            "phase381_audit",
            {
                "status": "joint_semantic_position_state_rejected_no_neuron_promotion",
                "causal_condition_row_count": 69120,
                "joint_direction_pass_count": 563,
                "model_cell_pass_count": 0,
                "crossmodel_upstream_cell_count": 0,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "language_path_count": 0,
                "source": "phase381_joint_state_stage_summary.json",
            },
        )
        manifest.setdefault(
            "phase382_audit",
            {
                "status": "total_layer_update_operator_rejected_no_neuron_promotion",
                "transition_event_row_count": 20592,
                "transition_own_profile_win_count": 12,
                "static_own_profile_win_count": 13,
                "identifiability_gate_pass": False,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "language_path_count": 0,
                "source": "phase382_transition_stage_summary.json",
            },
        )
        manifest["phase383_385_audit"] = {
            "status": "exact_subunit_mass_measured_no_neuron_path_promoted",
            "exact_attention_head_events_discovery": 2315520,
            "exact_mlp_channel_events_discovery": 205701120,
            "replicated_upstream_opposing_mass_patterns": 2,
            "function_specific_opposing_mass_patterns": 0,
            "new_neuron_path_nodes_promoted": 0,
            "single_unit_causal_count": 0,
            "language_path_count": 0,
            "source": "phase385_exact_event_stage_summary.json",
        }
        manifest.setdefault("files", {})[
            "latest_evidence_summary"
        ] = "phase385_exact_event_stage_summary.json"
        boundary = manifest.setdefault("evidence_boundary", {})
        boundary["mapped_scope"] = (
            "nine_family_engineering_registry_with_exact_event_audit_for_four_mechanisms"
        )
        boundary["exact_event_audited_families"] = [
            "content_knowledge",
            "readout_competition",
            "state_drift",
            "syntax_structure",
        ]
        boundary["exact_event_audited_mechanisms"] = [
            "relation_binding",
            "target_vs_wrong",
            "entity_recency",
            "number_agreement",
        ]
        boundary["exact_event_unaudited_families"] = [
            "output_protocol",
            "reasoning_constraint",
            "language_action",
            "cross_lingual",
            "closure",
        ]
        boundary["statement"] = (
            "Exact attention-head and MLP-channel event families are replayable. Two "
            "upstream opposing MLP mass patterns replicate but fail matched functional, "
            "receiver, or depth controls; no neuron or language path is promoted."
        )
        boundary["latest_phase"] = "Phase385-OpposingMassSpecificity"
        boundary["exact_subunit_mass_available"] = True
        boundary["function_specific_upstream_subunit_pattern_available"] = False
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
        raise FileNotFoundError(f"Missing Phase383-385 public artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    row_payloads = {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()}
    p383 = payloads["phase383_calibration_summary.json"]
    p384 = payloads["phase384_calibration_summary.json"]
    p385 = payloads["phase385_specificity_summary.json"]
    updated_at = datetime.now(timezone.utc).isoformat()
    stage = {
        "schema_version": "59.1.0",
        "phase_id": "Phase385-StageMerge",
        "created_at": updated_at,
        "objective": "map_exact_component_and_subunit_events_then_reject_non_specific_mass",
        "assessment": {
            "single_sample_runtime_contract_frozen": True,
            "exact_attention_source_events_replayable": True,
            "exact_mlp_channel_events_replayable": True,
            "signed_component_event_map_available": True,
            "all_subunit_projection_mass_map_available": True,
            "function_specific_upstream_event_available": False,
            "complete_upstream_language_path_available": False,
            "physical_holdout_opened": False,
            "causal_intervention_authorized": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "single_path_qualified_groups_by_mechanism": payloads[
                "phase383_protocol.json"
            ]["source_denominator"]["single_path_qualified_groups_by_mechanism"],
            "phase383_discovery_event_rows": payloads[
                "phase383_signed_event_map_summary.json"
            ]["denominator"]["event_row_count"],
            "phase383_calibration_event_rows": p383["denominator"]["event_row_count"],
            "phase384_discovery_exact_attention_head_events": payloads[
                "phase384_discovery_summary.json"
            ]["denominator"]["exact_attention_head_event_count"],
            "phase384_discovery_exact_mlp_channel_events": payloads[
                "phase384_discovery_summary.json"
            ]["denominator"]["exact_mlp_channel_event_count"],
        },
        "results": {
            "phase383_frozen_candidates": p383["denominator"]["frozen_candidate_count"],
            "phase383_level2_replications": p383["results"][
                "calibration_level2_replication_count"
            ],
            "phase383_upstream_level2_replications": p383["results"][
                "upstream_level2_replication_count"
            ],
            "phase384_level2_mass_replications": p384["results"][
                "level2_replication_count"
            ],
            "phase384_upstream_opposing_replications": p384["results"][
                "upstream_opposing_replication_count"
            ],
            "phase385_function_specific_replications": p385["results"][
                "replicated_function_specific_candidate_count"
            ],
            "new_neuron_path_nodes_promoted": 0,
            "language_path_count": 0,
        },
        "hard_limits": [
            "phase383_reuses_a_retrospective_phase380_case_bank_after_single_path_requalification",
            "three_discovery_and_two_calibration_groups_per_mechanism_are_small",
            "target_decision_is_only_one_semantic_time",
            "source_role_attention_events_are_aggregated_across_heads_in_the_parent_map",
            "projection_mass_patterns_do_not_identify_equal_neurons_across_models",
            "opposing_mass_can_reflect_general_mlp_architecture",
            "no_terminal_prediction_gain_or_causal_mediation_gate_passed",
        ],
        "authorization": {
            "show_exact_event_ledger": True,
            "show_late_current_layout_as_descriptive": True,
            "show_opposing_mass_as_language_path": False,
            "show_specific_neuron_path": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
        },
        "next_stage": {
            "phase": 386,
            "objective": "register_a_new_relation_aware_multi_time_event_graph",
            "required_order": [
                "freeze_semantic_times_and_single_sample_runtime_before_case_generation",
                "collect_source_query_predecision_and_decision_snapshots",
                "track_parent_child_event_relations_instead_of_static_mass",
                "require_function_role_depth_controls_before_calibration",
                "require_terminal_prediction_gain_before_physical_holdout",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    nodes, edges = graph()
    payloads["phase385_exact_event_stage_summary.json"] = stage
    row_payloads["phase385_evidence_nodes.jsonl"] = nodes
    row_payloads["phase385_evidence_edges.jsonl"] = edges
    published_files = [*payloads.keys(), *row_payloads.keys()]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase385-StageMerge"
        manifest["phase383_385"] = {
            "status": "exact_event_map_complete_for_four_mechanisms_no_upstream_path",
            "single_path_case_count": 759,
            "signed_discovery_event_row_count": 149760,
            "signed_calibration_event_row_count": 99840,
            "exact_attention_head_event_count_discovery": 2315520,
            "exact_mlp_channel_event_count_discovery": 205701120,
            "replicated_upstream_opposing_pattern_count": 2,
            "function_specific_upstream_pattern_count": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published_files,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase385-StageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["exact_event_stage"] = {
            "audited_families": {"numerator": 4, "denominator": 9},
            "audited_registered_mechanisms": {"numerator": 4, "denominator": 72},
            "single_path_qualified_models": {"numerator": 3, "denominator": 3},
            "instrument_conservation_models": {"numerator": 3, "denominator": 3},
            "signed_event_calibration_candidates": {"numerator": 24, "denominator": 32},
            "upstream_signed_event_candidates": {"numerator": 0, "denominator": 32},
            "upstream_opposing_mass_patterns": {"numerator": 2, "denominator": 11},
            "function_specific_upstream_patterns": {"numerator": 0, "denominator": 2},
            "complete_language_paths": {"numerator": 0, "denominator": 72},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
        }
        progress["phase385_decision"] = (
            "keep_physical_holdout_sealed_and_register_multi_time_relation_graph"
        )
        write_json(root / "progress.json", progress)
        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase385-StageMerge"
            client_index["latest_stage_files"] = [
                "phase385_exact_event_stage_summary.json",
                "phase385_evidence_nodes.jsonl",
                "phase385_evidence_edges.jsonl",
                "phase385_specificity_summary.json",
                "phase384_calibration_summary.json",
                "phase383_calibration_summary.json",
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
