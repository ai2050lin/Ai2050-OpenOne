#!/usr/bin/env python3
"""Publish compact Phase375-378 evidence with strict terminal-carrier boundaries."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
P375 = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
P376 = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
P377 = ROOT / "tests/gpt5/result/phase377_decision_aligned_calibration"
P378 = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
TARGET = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON_TARGET = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"

SOURCES = {
    "phase375_protocol.json": P375 / "phase375_protocol.json",
    "phase375_blind_inventory_summary.json": P375 / "phase375_blind_inventory_summary.json",
    "phase375_discovery_summary.json": P375 / "phase375_discovery/phase375_discovery_summary.json",
    "phase375_negative_result_diagnostic.json": P375
    / "phase375_discovery/phase375_negative_result_diagnostic.json",
    "phase376_decision_time_alignment_summary.json": P376
    / "phase376_decision_time_alignment_summary.json",
    "phase376_intervention_summary.json": P376
    / "phase376_intervention/phase376_intervention_summary.json",
    "phase377_calibration_summary.json": P377
    / "phase377_intervention/phase377_calibration_summary.json",
    "phase378_physical_protocol.json": P378 / "phase378_physical_protocol.json",
    "phase378_physical_behavior_analysis_summary.json": P378
    / "phase378_physical_behavior_analysis_summary.json",
    "phase378_physical_summary.json": P378
    / "phase378_intervention/phase378_physical_summary.json",
    "phase378_terminal_carrier_minimality_summary.json": P378
    / "phase378_terminal_carrier_minimality_summary.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
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


def update_neuron_atlas(
    stage: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    updated_at: str,
) -> None:
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase378_decision_aligned_stage_summary.json", stage)
        write_jsonl(root / "phase378_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase378_evidence_edges.jsonl", edges)
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest["phase"] = 378
            manifest["generated_at"] = updated_at
            manifest["phase378_audit"] = {
                "status": "terminal_residual_carrier_physically_confirmed_no_neuron_path",
                "physically_confirmed_terminal_carrier_count": 2,
                "new_neuron_path_nodes_promoted": 0,
                "single_unit_causal_count": 0,
                "upstream_encoding_rule_count": 0,
                "language_path_count": 0,
                "source": "phase378_decision_aligned_stage_summary.json",
            }
            manifest.setdefault("files", {})[
                "latest_evidence_summary"
            ] = "phase378_decision_aligned_stage_summary.json"
            boundary = manifest.setdefault("evidence_boundary", {})
            boundary["statement"] = (
                "Decision-aligned discovery, calibration, and physical interventions confirm a late current-position "
                "residual content carrier for relation binding and entity recency. The carrier is an output endpoint, "
                "not an upstream rule or neuron-level mechanism."
            )
            boundary["latest_phase"] = "Phase378-PhysicalMerge"
            boundary["terminal_content_carrier_available"] = True
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
    payloads = {name: read_json(path) for name, path in SOURCES.items()}
    minimality_public = deepcopy(
        payloads["phase378_terminal_carrier_minimality_summary.json"]
    )
    for stage_row in minimality_public["stages"]:
        for model_row in stage_row["models"]:
            private_ids = model_row.pop("baseline_mismatch_case_ids", [])
            model_row["baseline_mismatch_case_count"] = len(private_ids)
    payloads["phase378_terminal_carrier_minimality_summary.json"] = minimality_public
    updated_at = datetime.now(timezone.utc).isoformat()
    inventory = payloads["phase375_blind_inventory_summary.json"]
    p375 = payloads["phase375_discovery_summary.json"]
    diagnostic = payloads["phase375_negative_result_diagnostic.json"]
    alignment = payloads["phase376_decision_time_alignment_summary.json"]
    discovery = payloads["phase376_intervention_summary.json"]
    calibration = payloads["phase377_calibration_summary.json"]
    behavior = payloads["phase378_physical_behavior_analysis_summary.json"]
    physical = payloads["phase378_physical_summary.json"]
    minimality = payloads["phase378_terminal_carrier_minimality_summary.json"]
    stage = {
        "schema_version": "51.6.0",
        "phase_id": "Phase378-PhysicalMerge",
        "created_at": updated_at,
        "objective": "replace_fixed_offset_linear_prediction_with_decision_aligned_direct_intervention",
        "assessment": {
            "phase375_finite_linear_future_state_crossmodel_sufficient": False,
            "phase371_fixed_t0_t2_crossmodel_semantically_aligned": False,
            "decision_aligned_late_current_residual_is_content_carrier": True,
            "carrier_physically_confirmed": True,
            "carrier_is_mechanism_specific": False,
            "source_query_additions_required": False,
            "upstream_encoding_rule_discovered": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "registered_language_families": 9,
            "registered_representative_mechanisms": 18,
            "blind_finite_subgraph_inventory_rows": inventory["denominator"][
                "total_inventory_row_count"
            ],
            "phase375_lexical_evaluations": p375["denominator"][
                "total_lexical_evaluation_count"
            ],
            "phase375_crossmodel_candidates": p375["results"][
                "heterogeneous_level2_count"
            ],
            "fixed_time_crossmodel_semantic_conditions": alignment["denominator"][
                "crossmodel_semantic_condition_count"
            ],
            "fixed_time_all_models_aligned_conditions": alignment["crossmodel"][
                "all_models_within_fixed_t0_t2_count"
            ],
            "decision_aligned_discovery_cases": discovery["denominator"]["case_count"],
            "decision_aligned_discovery_patched_forwards": discovery["denominator"][
                "patched_forward_condition_count"
            ],
            "calibration_cases": calibration["denominator"]["case_count"],
            "calibration_patched_forwards": calibration["denominator"][
                "patched_forward_condition_count"
            ],
            "physical_cases": physical["denominator"]["physical_case_count"],
            "physical_behavior_strict_correct": physical["quality"][
                "behavior_strict_correct_count"
            ],
            "physical_patched_forwards": physical["denominator"][
                "patched_forward_condition_count"
            ],
            "raw_physical_template_candidates": physical["results"][
                "physically_confirmed_terminal_carrier_count"
            ],
            "minimal_distinct_terminal_carriers": 2,
            "upstream_encoding_rules": 0,
            "language_path_candidates": 0,
            "strictly_closed_registered_cells": 0,
            "registered_closure_cells": 72,
        },
        "results": {
            "phase375_all_templates_rejected": True,
            "phase375_minimum_current_error": diagnostic["interpretation"][
                "minimum_current_error_observed"
            ],
            "decision_time_repair_materially_changed_result": True,
            "physically_confirmed_terminal_carrier_mechanisms": [
                "relation_binding",
                "entity_recency",
            ],
            "minimal_carrier_topology": "late_current_residual_output",
            "single_unit_causal_count": 0,
            "language_path_candidate_count": 0,
            "language_mechanism_claimed": False,
        },
        "hard_limits": [
            "the_positive_object_is_near_the_output_endpoint_and_does_not_explain_where_content_is_computed",
            "treatment_and_direct_route_control_both_transfer_content_so_mechanism_specificity_is_not_established",
            "source_query_additions_do_not_change_winner_in_sealed_calibration_or_physical_replication",
            "only_two_mechanisms_are_in_scope",
            "natural_necessity_and_full_generation_sufficiency_are_not_tested",
            "deepseek7b_entity_recency_winner_flip_did_not_replicate_at_the_registered_threshold",
            "current_models_are_small_and_qwen3_deepseek7b_are_architecture_related",
        ],
        "authorization": {
            "show_terminal_current_residual_carrier": True,
            "show_source_query_extension_as_independent_path": False,
            "show_as_upstream_encoding_rule": False,
            "show_as_language_mechanism": False,
            "show_as_neuron_path": False,
        },
        "next_stage": physical["next_stage"],
        "single_global_progress_percentage_valid": False,
    }
    nodes = [
        {
            "node_id": "p375_finite_linear_gate",
            "node_type": "state_sufficiency_gate",
            "phase_id": "Phase375-Discovery-Merge",
            "status": "all_templates_rejected",
            "lexical_evaluations": 1584,
            "language_path": False,
        },
        {
            "node_id": "p376_decision_alignment",
            "node_type": "semantic_time_alignment",
            "phase_id": "Phase376-DecisionAlignmentAudit",
            "status": "fixed_offset_invalid",
            "aligned_conditions": 16,
            "total_conditions": 88,
            "language_path": False,
        },
        {
            "node_id": "p376_direct_intervention",
            "node_type": "decision_aligned_intervention",
            "phase_id": "Phase376-Intervention-Merge",
            "status": "crossmodel_positive",
            "patched_forwards": 9504,
            "language_path": False,
        },
        {
            "node_id": "p377_calibration",
            "node_type": "independent_calibration",
            "phase_id": "Phase377-CalibrationMerge",
            "status": "replicated",
            "case_count": 132,
            "language_path": False,
        },
        {
            "node_id": "p378_physical_behavior",
            "node_type": "physical_behavior_gate",
            "phase_id": "Phase378-PhysicalBehaviorAnalysis",
            "status": "96_of_96_strict_correct",
            "language_path": False,
        },
        {
            "node_id": "p378_relation_terminal_carrier",
            "node_type": "terminal_residual_content_carrier",
            "phase_id": "Phase378-PhysicalMerge",
            "mechanism_id": "relation_binding",
            "relative_depth": "late",
            "position_role": "current_generation",
            "models": ["qwen3", "glm4", "deepseek7b"],
            "status": "physically_confirmed_level3_endpoint",
            "language_path": False,
            "single_unit_causal": False,
        },
        {
            "node_id": "p378_recency_terminal_carrier",
            "node_type": "terminal_residual_content_carrier",
            "phase_id": "Phase378-PhysicalMerge",
            "mechanism_id": "entity_recency",
            "relative_depth": "late",
            "position_role": "current_generation",
            "models": ["qwen3", "glm4"],
            "status": "physically_confirmed_heterogeneous_level2_endpoint",
            "language_path": False,
            "single_unit_causal": False,
        },
        {
            "node_id": "p378_upstream_formation_unknown",
            "node_type": "unresolved_mechanism",
            "phase_id": "Phase378-PhysicalMerge",
            "status": "unresolved",
            "language_path": False,
            "single_unit_causal": False,
        },
    ]
    edges = [
        {
            "edge_id": "p375_finite_linear_gate->p376_decision_alignment",
            "source_node_id": "p375_finite_linear_gate",
            "target_node_id": "p376_decision_alignment",
            "edge_type": "negative_result_triggers_time_audit",
            "phase_id": "Phase376-DecisionAlignmentAudit",
        },
        {
            "edge_id": "p376_decision_alignment->p376_direct_intervention",
            "source_node_id": "p376_decision_alignment",
            "target_node_id": "p376_direct_intervention",
            "edge_type": "authorizes_decision_aligned_readout",
            "phase_id": "Phase376-Intervention",
        },
        {
            "edge_id": "p376_direct_intervention->p377_calibration",
            "source_node_id": "p376_direct_intervention",
            "target_node_id": "p377_calibration",
            "edge_type": "authorizes_independent_replication",
            "phase_id": "Phase377-CalibrationMerge",
        },
        {
            "edge_id": "p377_calibration->p378_physical_behavior",
            "source_node_id": "p377_calibration",
            "target_node_id": "p378_physical_behavior",
            "edge_type": "authorizes_narrow_physical_opening",
            "phase_id": "Phase378-PhysicalBehaviorAnalysis",
        },
        {
            "edge_id": "p378_physical_behavior->p378_relation_terminal_carrier",
            "source_node_id": "p378_physical_behavior",
            "target_node_id": "p378_relation_terminal_carrier",
            "edge_type": "physically_confirms_terminal_transfer",
            "phase_id": "Phase378-PhysicalMerge",
        },
        {
            "edge_id": "p378_physical_behavior->p378_recency_terminal_carrier",
            "source_node_id": "p378_physical_behavior",
            "target_node_id": "p378_recency_terminal_carrier",
            "edge_type": "physically_confirms_terminal_transfer",
            "phase_id": "Phase378-PhysicalMerge",
        },
        {
            "edge_id": "p378_relation_terminal_carrier->p378_upstream_formation_unknown",
            "source_node_id": "p378_relation_terminal_carrier",
            "target_node_id": "p378_upstream_formation_unknown",
            "edge_type": "does_not_resolve_upstream_formation",
            "phase_id": "Phase378-TerminalCarrierMinimality",
        },
        {
            "edge_id": "p378_recency_terminal_carrier->p378_upstream_formation_unknown",
            "source_node_id": "p378_recency_terminal_carrier",
            "target_node_id": "p378_upstream_formation_unknown",
            "edge_type": "does_not_resolve_upstream_formation",
            "phase_id": "Phase378-TerminalCarrierMinimality",
        },
    ]
    payloads["phase378_decision_aligned_stage_summary.json"] = stage
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase378_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase378_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase378-PhysicalMerge"
        manifest["phase378"] = {
            "status": "terminal_residual_carrier_physically_confirmed_upstream_unknown",
            "decision_aligned_discovery_cases": 264,
            "calibration_cases": 132,
            "physical_cases": 96,
            "minimal_terminal_carrier_count": 2,
            "upstream_encoding_rule_count": 0,
            "language_path_count": 0,
            "single_unit_causal_count": 0,
            "files": [
                *SOURCES.keys(),
                "phase378_decision_aligned_stage_summary.json",
                "phase378_evidence_nodes.jsonl",
                "phase378_evidence_edges.jsonl",
            ],
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase378-PhysicalMerge"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["global_research_estimates"] = {
            "status": "invalid_for_scientific_completion",
            "reason": "terminal_carrier_evidence_does_not_measure_upstream_mechanism_completion",
            "single_scalar_estimate_valid": False,
        }
        progress["decision_aligned_stage"] = {
            "mechanisms_in_scope": {"numerator": 2, "denominator": 18},
            "physical_behavior_cases": {"numerator": 96, "denominator": 96},
            "minimal_terminal_content_carriers": {"numerator": 2, "denominator": 2},
            "upstream_encoding_rules": {"numerator": 0, "denominator": 18},
            "natural_necessity": {"numerator": 0, "denominator": 18},
            "strict_language_paths": {"numerator": 0, "denominator": 18},
            "strict_closure_cells": {"numerator": 0, "denominator": 72},
        }
        progress["phase378_decision"] = (
            "stop_terminal_endpoint_swaps_and_trace_earlier_formation_events"
        )
        write_json(root / "progress.json", progress)
        public_manifest(root, updated_at)
    update_neuron_atlas(stage, nodes, edges, updated_at)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
