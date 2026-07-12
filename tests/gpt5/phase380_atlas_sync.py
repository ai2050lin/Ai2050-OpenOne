#!/usr/bin/env python3
"""Publish compact Phase379-380 global-layout evidence to both atlas mirrors."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
P379 = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
P380 = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
TARGET = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON_TARGET = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"

JSON_SOURCES = {
    "phase379_global_layout_protocol.json": P379 / "phase379_protocol.json",
    "phase379_case_bank_summary.json": P379 / "phase379_case_bank_summary.json",
    "phase379_phase330_decision_audit.json": P379
    / "phase379_phase330_decision_audit.json",
    "phase379_discovery_mapping_summary.json": P379
    / "phase379_discovery_mapping_summary.json",
    "phase379_calibration_summary.json": P379 / "phase379_calibration_summary.json",
    "phase379_backbone_confound_audit.json": P379
    / "phase379_backbone_confound_audit.json",
    "phase380_independent_protocol.json": P380 / "phase380_protocol.json",
    "phase380_case_bank_summary.json": P380 / "phase380_case_bank_summary.json",
    "phase380_behavior_analysis_final_summary.json": P380
    / "phase380_behavior_analysis_final_summary.json",
    "phase380_residual_validation_summary.json": P380
    / "phase380_residual_validation_summary.json",
    "phase380_causal_scan_freeze.json": P380 / "phase380_causal_scan_freeze.json",
    "phase380_causal_layout_summary.json": P380
    / "phase380_causal_layout_summary.json",
}

JSONL_SOURCES = {
    "phase380_stable_layout_objects.jsonl": P380
    / "validation/phase380_stable_layout_objects.jsonl",
    "phase380_crossmodel_causal_cells.jsonl": P380
    / "causal/phase380_crossmodel_cell_rows.jsonl",
    "phase380_shared_terminal_interfaces.jsonl": P380
    / "causal/phase380_shared_terminal_interface_rows.jsonl",
}

FAMILY_BY_MECHANISM = {
    "entity_recency": "content_knowledge",
    "relation_binding": "reasoning_constraint",
    "number_agreement": "syntax_structure",
    "target_vs_wrong": "readout_competition",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_manifest(root: Path, updated_at: str) -> None:
    path = root / "public_manifest.json"
    payload = read_json(path) if path.is_file() else {"schema_version": "public_manifest.v1"}
    payload["generated_at"] = updated_at
    payload["files"] = sorted(
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file()
        and item != path
        and item.suffix.lower() in {".json", ".jsonl", ".md"}
    )
    write_json(path, payload)


def public_payloads() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    missing = [str(path) for path in (*JSON_SOURCES.values(), *JSONL_SOURCES.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase379-380 artifacts: {missing}")
    return (
        {name: read_json(path) for name, path in JSON_SOURCES.items()},
        {name: read_jsonl(path) for name, path in JSONL_SOURCES.items()},
    )


def evidence_graph(
    residual_summary: dict[str, Any],
    causal_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = [
        {
            "node_id": "p379_phase330_denominator_failure",
            "node_type": "denominator_audit",
            "phase_id": "Phase379",
            "status": "old_crossmodel_layout_denominator_rejected",
            "language_path": False,
        },
        {
            "node_id": "p379_common_residual_backbone",
            "node_type": "common_architecture_backbone",
            "phase_id": "Phase379-BackboneConfoundAudit",
            "status": "confound_confirmed",
            "function_specific": False,
            "language_path": False,
        },
        {
            "node_id": "p380_independent_factorial_denominator",
            "node_type": "independent_validation_denominator",
            "phase_id": "Phase380-BehaviorAnalysisFinal",
            "status": "behavior_qualified_before_trace",
            "case_count": 780,
            "language_path": False,
        },
    ]
    edges: list[dict[str, Any]] = [
        {
            "edge_id": "p379_phase330_denominator_failure->p379_common_residual_backbone",
            "source_node_id": "p379_phase330_denominator_failure",
            "target_node_id": "p379_common_residual_backbone",
            "edge_type": "triggers_backbone_confound_audit",
            "phase_id": "Phase379-BackboneConfoundAudit",
            "causal_path": False,
        },
        {
            "edge_id": "p379_common_residual_backbone->p380_independent_factorial_denominator",
            "source_node_id": "p379_common_residual_backbone",
            "target_node_id": "p380_independent_factorial_denominator",
            "edge_type": "requires_fresh_metric_frozen_validation",
            "phase_id": "Phase380-Protocol",
            "causal_path": False,
        },
    ]
    residual_nodes: dict[str, list[str]] = {}
    for row in residual_summary["results"]["stable_objects"]:
        mechanism = row["mechanism_id"]
        axis = row["contrast_axis"]
        node_id = f"p380_residual:{mechanism}:{axis}"
        residual_nodes.setdefault(mechanism, []).append(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "replicated_function_residual_profile",
                "phase_id": "Phase380-ResidualValidation",
                "family_id": FAMILY_BY_MECHANISM[mechanism],
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "models": row["individually_stable_models"],
                "status": "heterogeneous_crossmodel_replication",
                "causal": False,
                "single_unit_causal": False,
                "language_path": False,
            }
        )
        edges.append(
            {
                "edge_id": f"p380_independent_factorial_denominator->{node_id}",
                "source_node_id": "p380_independent_factorial_denominator",
                "target_node_id": node_id,
                "edge_type": "supports_replicated_descriptive_profile",
                "phase_id": "Phase380-ResidualValidation",
                "causal_path": False,
            }
        )
    terminal_mechanisms: set[str] = set()
    for territory in causal_summary["results"]["shared_terminal_interface_territories"]:
        component = territory["component_type"]
        for mechanism in territory["mechanisms"]:
            terminal_mechanisms.add(mechanism)
            node_id = f"p380_terminal:{mechanism}:{component}:current"
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": "terminal_interface_causal_cell",
                    "phase_id": "Phase380-CausalLayoutAnalysis",
                    "family_id": FAMILY_BY_MECHANISM[mechanism],
                    "mechanism_id": mechanism,
                    "relative_depth": "late",
                    "component_type": component,
                    "position_role": "current",
                    "models": ["qwen3", "glm4", "deepseek7b"],
                    "status": "level3_terminal_interface_only",
                    "causal": True,
                    "causal_scope": "terminal_content_transfer_interface",
                    "upstream_encoding_rule": False,
                    "single_unit_causal": False,
                    "language_path": False,
                }
            )
            for residual_id in residual_nodes.get(mechanism, []):
                edges.append(
                    {
                        "edge_id": f"{residual_id}->{node_id}",
                        "source_node_id": residual_id,
                        "target_node_id": node_id,
                        "edge_type": "causal_scan_localizes_to_terminal_interface",
                        "phase_id": "Phase380-CausalLayoutAnalysis",
                        "causal_path": False,
                    }
                )
    for mechanism in sorted(terminal_mechanisms):
        source = f"p380_terminal:{mechanism}:layer_input:current"
        target = f"p380_terminal:{mechanism}:layer_output:current"
        edges.append(
            {
                "edge_id": f"{source}->{target}",
                "source_node_id": source,
                "target_node_id": target,
                "edge_type": "adjacent_terminal_interface_boundaries",
                "phase_id": "Phase380-CausalLayoutAnalysis",
                "causal_path": False,
                "mediation_between_boundaries_tested": False,
            }
        )
    nodes.append(
        {
            "node_id": "p380_upstream_global_layout_unknown",
            "node_type": "unresolved_global_mechanism",
            "phase_id": "Phase380-CausalLayoutAnalysis",
            "status": "no_crossmodel_upstream_cell_passed",
            "causal": False,
            "single_unit_causal": False,
            "language_path": False,
        }
    )
    for mechanism in sorted(terminal_mechanisms):
        edges.append(
            {
                "edge_id": f"p380_upstream_global_layout_unknown->p380_terminal:{mechanism}:layer_input:current",
                "source_node_id": "p380_upstream_global_layout_unknown",
                "target_node_id": f"p380_terminal:{mechanism}:layer_input:current",
                "edge_type": "unresolved_predecessor_of_terminal_interface",
                "phase_id": "Phase380-CausalLayoutAnalysis",
                "causal_path": False,
            }
        )
    return nodes, edges


def update_neuron_atlas(
    stage: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    updated_at: str,
) -> None:
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase380_global_layout_stage_summary.json", stage)
        write_jsonl(root / "phase380_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase380_evidence_edges.jsonl", edges)
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest["phase"] = 380
            manifest["generated_at"] = updated_at
            manifest["phase380_audit"] = {
                "status": "terminal_interface_only_no_upstream_or_neuron_path",
                "replicated_residual_object_count": 5,
                "terminal_interface_boundary_count": 2,
                "terminal_interface_mechanism_count": 3,
                "upstream_crossmodel_cell_count": 0,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "language_path_count": 0,
                "source": "phase380_global_layout_stage_summary.json",
            }
            manifest.setdefault("files", {})[
                "latest_evidence_summary"
            ] = "phase380_global_layout_stage_summary.json"
            boundary = manifest.setdefault("evidence_boundary", {})
            boundary["statement"] = (
                "Independent all-layer validation separates a common architecture backbone from five replicated "
                "function-residual objects. Controlled swaps localize all heterogeneous causal cells to the late "
                "current-position input/output terminal interface; no upstream cross-model cell or neuron path passes."
            )
            boundary["latest_phase"] = "Phase380-CausalLayoutAnalysis"
            boundary["replicated_function_residual_profiles_available"] = True
            boundary["terminal_interface_causal_cells_available"] = True
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
    payloads, row_payloads = public_payloads()
    updated_at = datetime.now(timezone.utc).isoformat()
    phase379 = payloads["phase379_backbone_confound_audit.json"]
    behavior = payloads["phase380_behavior_analysis_final_summary.json"]
    residual = payloads["phase380_residual_validation_summary.json"]
    causal = payloads["phase380_causal_layout_summary.json"]
    nodes, edges = evidence_graph(residual, causal)
    stage = {
        "schema_version": "53.11.0",
        "phase_id": "Phase380-GlobalLayoutStageMerge",
        "created_at": updated_at,
        "objective": "map_global_reuse_and_differentiation_before_local_language_family_closure",
        "assessment": {
            "global_layout_is_higher_priority_than_single_family_closure": True,
            "phase330_is_valid_crossmodel_scientific_layout": False,
            "raw_profile_replication_is_function_specific": False,
            "common_architecture_backbone_confirmed": True,
            "independent_function_residual_replication_observed": True,
            "crossmodel_causal_effect_is_upstream": False,
            "crossmodel_causal_effect_is_terminal_interface_only": True,
            "same_physical_neurons_across_models_established": False,
            "complete_language_path_established": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "registered_language_families": 9,
            "registered_representative_mechanisms": 18,
            "phase379_behavior_qualified_cases": 516,
            "phase380_initial_behavior_cases": behavior["denominator"]["initial_behavior_case_count"],
            "phase380_expansion_behavior_cases": behavior["denominator"]["expansion_behavior_case_count"],
            "phase380_qualified_trace_cases": behavior["denominator"]["qualified_trace_case_count"],
            "phase380_replay_qualified_groups": residual["denominator"]["replay_qualified_parallel_group_count"],
            "phase380_exact_event_vectors": residual["denominator"]["exact_event_vector_count"],
            "phase380_all_pair_event_rows": residual["denominator"]["all_pair_event_row_count"],
            "phase380_causal_condition_rows": causal["denominator"]["condition_row_count"],
            "replicated_function_residual_objects": residual["results"]["heterogeneous_level2_stable_object_count"],
            "heterogeneous_terminal_interface_cells": causal["results"]["heterogeneous_terminal_interface_cell_count"],
            "heterogeneous_upstream_cells": causal["results"]["heterogeneous_upstream_cell_count"],
            "complete_upstream_language_paths": 0,
            "single_neuron_causal_paths": 0,
            "strictly_closed_registered_cells": 0,
            "registered_closure_cells": 72,
        },
        "results": {
            "phase379_raw_profile_minimum_replication_cosine": phase379["results"]["raw_discovery_calibration_cosine"]["minimum"],
            "phase379_residual_crossmodel_median_cosine": phase379["results"]["heterogeneous_crossmodel_residual_cosine"]["median"],
            "phase380_stable_residual_objects": residual["results"]["stable_objects"],
            "terminal_interface_mechanisms": ["entity_recency", "relation_binding", "target_vs_wrong"],
            "terminal_interface_boundaries": ["late_layer_input_current", "late_layer_output_current"],
            "syntax_causal_scan_opened": False,
            "upstream_crossmodel_cell_count": 0,
            "global_reuse_layout_completed": False,
        },
        "hard_limits": [
            "only_four_representative_mechanisms_were_independently_traced_not_all_nine_families",
            "all_crossmodel_causal_cells_are_inside_the_late_current_terminal_interface",
            "late_layer_input_is_not_an_upstream_rule_region",
            "normalized_cells_do_not_identify_the_same_neurons_across_architectures",
            "number_agreement_had_only_seven_replay_qualified_groups_and_did_not_enter_the_causal_scan",
            "single_boundary_swaps_may_miss_distributed_joint_state_but_that_explanation_is_not_yet_proven",
            "small_models_may_use_coarser_or_architecture_specific_internal_routes",
        ],
        "authorization": {
            "show_replicated_residual_profiles_as_descriptive": True,
            "show_late_current_boundaries_as_terminal_interface_causal_cells": True,
            "connect_terminal_boundaries_as_proven_mediated_path": False,
            "show_any_upstream_language_rule": False,
            "promote_any_single_neuron": False,
            "claim_global_layout_complete": False,
            "open_old_physical_holdout": False,
        },
        "next_stage": {
            "phase": 381,
            "objective": "test_joint_state_formation_fronts_upstream_of_the_frozen_terminal_interface",
            "required_design": [
                "freeze_fresh_behavior_qualified_groups_before_internal_measurement",
                "separate_single_event_failure_from_joint_distributed_state_failure",
                "use_terminal_interface_only_as_outcome_anchor_not_as_the_search_target",
                "require_wrong_depth_wrong_role_equal_energy_and_side_effect_controls",
                "keep_single_neuron_scan_closed_until_an_upstream_component_path_replicates",
            ],
        },
        "single_global_progress_percentage_valid": False,
    }
    payloads["phase380_global_layout_stage_summary.json"] = stage
    row_payloads["phase380_evidence_nodes.jsonl"] = nodes
    row_payloads["phase380_evidence_edges.jsonl"] = edges
    published_files = [*payloads.keys(), *row_payloads.keys()]
    if not (TARGET / "client_index.json").is_file() and (CLIENT / "client_index.json").is_file():
        write_json(TARGET / "client_index.json", read_json(CLIENT / "client_index.json"))
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        for name, rows in row_payloads.items():
            write_jsonl(root / name, rows)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase380-GlobalLayoutStageMerge"
        manifest["phase380"] = {
            "status": "independent_residual_replication_terminal_interface_only",
            "representative_mechanism_count": 4,
            "qualified_trace_case_count": 780,
            "exact_event_vector_count": 324480,
            "causal_condition_row_count": 57600,
            "replicated_residual_object_count": 5,
            "terminal_interface_mechanism_count": 3,
            "heterogeneous_upstream_cell_count": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": published_files,
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase380-GlobalLayoutStageMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["global_research_estimates"] = {
            "status": "invalid_for_scientific_completion",
            "reason": "four_mechanism_terminal_interface_evidence_does_not_measure_nine_family_upstream_layout_completion",
            "single_scalar_estimate_valid": False,
        }
        progress["global_layout_stage"] = {
            "registered_language_families": {"numerator": 9, "denominator": 9, "scope": "engineering_registry"},
            "independently_traced_representative_mechanisms": {"numerator": 4, "denominator": 18},
            "qualified_exact_trace_cases": {"numerator": 780, "denominator": 780},
            "replay_qualified_exact_trace_cases": {"numerator": 770, "denominator": 780},
            "replicated_residual_objects": {"numerator": 5, "denominator": 12},
            "terminal_interface_causal_mechanisms": {"numerator": 3, "denominator": 4},
            "upstream_causal_mechanisms": {"numerator": 0, "denominator": 4},
            "complete_language_paths": {"numerator": 0, "denominator": 18},
            "single_neuron_causal_paths": {"numerator": 0, "denominator": 18},
            "strict_closure_cells": {"numerator": 0, "denominator": 72},
        }
        progress["phase380_decision"] = (
            "freeze_terminal_interface_result_and_test_joint_upstream_state_formation_without_promoting_neurons"
        )
        write_json(root / "progress.json", progress)
        client_index_path = root / "client_index.json"
        if client_index_path.is_file():
            client_index = read_json(client_index_path)
            client_index["latest_phase"] = "Phase380-GlobalLayoutStageMerge"
            client_index["latest_stage_files"] = [
                "phase380_global_layout_stage_summary.json",
                "phase380_evidence_nodes.jsonl",
                "phase380_evidence_edges.jsonl",
                "phase380_causal_layout_summary.json",
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
