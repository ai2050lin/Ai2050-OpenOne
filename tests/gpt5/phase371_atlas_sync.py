#!/usr/bin/env python3
"""Publish compact Phase371 exact-vector evidence without promoting language paths."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
TARGET = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON_TARGET = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


SOURCES = {
    "phase371a_exact_tree_feasibility_summary.json": PHASE371 / "engineering_feasibility/phase371a_existing_ledger_tree_feasibility_summary.json",
    "phase371b_engineering_summary.json": PHASE371 / "phase371b_engineering_summary.json",
    "phase371b_sufficient_state_summary.json": PHASE371 / "phase371b_sufficient_state_summary.json",
    "phase371c_case_bank_summary.json": PHASE371 / "phase371c_case_bank/phase371c_case_bank_summary.json",
    "phase371c_behavior_analysis_summary.json": PHASE371 / "phase371c_behavior_analysis/phase371c_behavior_analysis_summary.json",
    "phase371c_internal_collection_audit.json": PHASE371 / "phase371c_internal_collection_audit.json",
    "phase371c_adjacent_extension_audit.json": PHASE371 / "phase371c_adjacent_extension_audit.json",
    "phase371c_lazy_exact_path_summary.json": PHASE371 / "phase371c_lazy_exact_paths/phase371c_lazy_exact_path_summary.json",
    "phase371c_blind_vector_contrast_summary.json": PHASE371 / "phase371c_blind_vector_contrast/phase371c_blind_vector_contrast_summary.json",
    "phase371c_blind_contrast_audit.json": PHASE371 / "phase371c_blind_vector_contrast/phase371c_blind_contrast_audit.json",
    "phase371c_discovery_mapping_summary.json": PHASE371 / "phase371c_discovery_mapping/phase371c_discovery_mapping_summary.json",
    "phase371c_exact_history_residual_summary.json": PHASE371 / "phase371c_exact_history_residual/phase371c_exact_history_residual_summary.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
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
        if item.is_file() and item != path and item.suffix.lower() in {".json", ".jsonl", ".md"}
    )
    write_json(path, payload)


def sync_neuron_atlas(stage: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]], updated_at: str) -> None:
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase371_exact_vector_stage_summary.json", stage)
        write_jsonl(root / "phase371_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase371_evidence_edges.jsonl", edges)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        manifest["phase"] = 371
        manifest["generated_at"] = updated_at
        manifest["phase371_audit"] = {
            "status": "exact_measurement_complete_history_crossmodel_gate_negative",
            "measured_discovery_cases": 264,
            "measured_local_layer_pairs": 9,
            "new_neuron_path_nodes_promoted": 0,
            "single_unit_causal_count": 0,
            "history_heterogeneous_level2_count": 0,
            "calibration_opened": False,
            "physical_holdout_opened": False,
            "source": "phase371_exact_vector_stage_summary.json",
        }
        manifest.setdefault("files", {})["latest_evidence_summary"] = "phase371_exact_vector_stage_summary.json"
        evidence = manifest.setdefault("evidence_boundary", {})
        evidence["statement"] = (
            "Phase371 built all-token exact Q/K/V and deterministic head/neuron path references for 264 discovery cases. "
            "Blind discovery produced provisional routes, but the exact history gate left no heterogeneous cross-model route. "
            "No neuron or language-family path is promoted."
        )
        evidence["latest_phase"] = "Phase371C-History"
        evidence["exact_measurement_path_available"] = True
        evidence["global_all_layer_path_available"] = False
        evidence["language_encoding_mechanism_closed"] = False
        evidence["single_unit_causal_closure"] = False
        write_json(manifest_path, manifest)
        checksum_path = root / "checksums.json"
        if checksum_path.is_file():
            write_json(checksum_path, {
                "schema_version": "artifact_checksums.v1",
                "files": [
                    {"path": str(item.relative_to(root)), "sha256": file_sha256(item)}
                    for item in sorted(root.rglob("*"))
                    if item.is_file() and item != checksum_path
                ],
            })
        public_manifest(root, updated_at)


def main() -> None:
    payloads = {name: read_json(path) for name, path in SOURCES.items()}
    updated_at = datetime.now(timezone.utc).isoformat()
    behavior = payloads["phase371c_behavior_analysis_summary.json"]
    ledger = payloads["phase371c_internal_collection_audit.json"]
    adjacent = payloads["phase371c_adjacent_extension_audit.json"]
    paths = payloads["phase371c_lazy_exact_path_summary.json"]
    contrast = payloads["phase371c_blind_vector_contrast_summary.json"]
    mapping = payloads["phase371c_discovery_mapping_summary.json"]
    history = payloads["phase371c_exact_history_residual_summary.json"]
    stage = {
        "schema_version": "47.23.0",
        "phase_id": "Phase371C-History",
        "created_at": updated_at,
        "objective": "replace_lossy_scalar_topology_with_exact_all_token_conservation_paths_and_test_history_sufficiency",
        "assessment": {
            "phase364_368_lossy_projection_diagnosis_supported": True,
            "exact_all_token_qk_measurement_feasible": True,
            "local_adjacent_layer_continuity_verified": True,
            "single_route_history_state_crossmodel_sufficient": False,
            "dynamic_language_state_existence_refuted": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "registered_language_families": 9,
            "registered_representative_mechanisms": 18,
            "phase371c_preregistered_behavior_cases": 1056,
            "nonphysical_behavior_cases_executed": 864,
            "common_behavior_qualified_groups": behavior["behavior"]["common_qualified_group_count"],
            "behavior_eligible_mechanisms": behavior["results"]["eligible_mechanism_count"],
            "internal_discovery_cases": ledger["denominator"]["case_count"],
            "base_and_adjacent_files": ledger["denominator"]["file_count"] + adjacent["denominator"]["adjacent_file_count"],
            "base_and_adjacent_bytes": adjacent["storage"]["combined_bytes"],
            "local_continuity_rows": adjacent["denominator"]["continuity_row_count"],
            "lazy_explicit_nodes": paths["denominator"]["explicit_node_count"],
            "lazy_explicit_edges": paths["denominator"]["explicit_edge_count"],
            "implicit_query_key_events": paths["denominator"]["implicit_exact_event_counts"]["query_key_score_events"],
            "implicit_attention_head_writes": paths["denominator"]["implicit_exact_event_counts"]["attention_head_write_events"],
            "implicit_mlp_neuron_writes": paths["denominator"]["implicit_exact_event_counts"]["mlp_single_neuron_write_events"],
            "blind_route_contrast_rows": contrast["denominator"]["route_contrast_row_count"],
            "blind_vocab_contrast_rows": contrast["denominator"]["vocab_contrast_row_count"],
            "provisional_model_candidates": mapping["denominator"]["provisional_model_candidate_count"],
            "provisional_heterogeneous_level2": mapping["results"]["provisional_heterogeneous_level2_count"],
            "provisional_level3": mapping["results"]["provisional_level3_count"],
            "t1_history_candidates": history["denominator"]["t1_provisional_model_candidate_count"],
            "history_model_passes": history["denominator"]["history_model_pass_count"],
            "history_heterogeneous_level2": history["results"]["history_heterogeneous_level2_count"],
            "history_level3": history["results"]["history_level3_count"],
            "calibration_internal_cases_opened": 0,
            "physical_holdout_cases_opened": 0,
            "strictly_closed_registered_cells": 0,
            "registered_closure_cells": 72,
        },
        "results": {
            "measurement_ledger_valid": ledger["valid"] and adjacent["valid"],
            "lazy_exact_path_objects_valid": paths["valid"],
            "all_six_blind_pairs_complete": contrast["results"]["blind_index_extraction_complete"],
            "provisional_routes_are_language_paths": False,
            "history_crossmodel_gate_passed": False,
            "language_path_candidate_count": 0,
            "single_neuron_causal_count": 0,
        },
        "hard_limits": [
            "only_two_of_four_preregistered_mechanisms_passed_common_behavior_qualification",
            "only_three_local_layer_pairs_per_model_are_contiguous_not_the_global_all_layer_path",
            "navigation_cosines_and_inner_products_are_indices_not_sufficient_states",
            "history_projection_is_a_local_linear_diagnostic_not_causal_replay",
            "the_three_history_model_passes_are_different_routes_and_do_not_form_crossmodel_evidence",
            "current_models_are_small_and_qwen3_deepseek7b_are_architecture_related",
        ],
        "authorization": {
            "show_verified_exact_measurement_ledger_in_client": True,
            "show_measured_local_layer_pairs": True,
            "show_phase371_objects_as_language_family_paths": False,
            "show_raw_private_tensors_or_blind_rows": False,
            "open_internal_calibration": False,
            "physical_confirmation": False,
            "causal_intervention": False,
        },
        "next_stage": {
            "priority": "multi_route_exact_subgraph_state_with_preregistered_history_and_intervention_gate",
            "continue_single_route_search": False,
            "open_phase371c_calibration": False,
        },
        "single_global_progress_percentage_valid": False,
    }
    measured_pairs = []
    for row in adjacent["models"]:
        for name, source, receiver in zip(
            ("early", "middle", "late"),
            row["base_layers"][:2] + row["adjacent_layers"][2:3],
            row["adjacent_layers"][:2] + row["base_layers"][2:3],
            strict=True,
        ):
            measured_pairs.append({
                "schema_version": "47.23.0",
                "phase_id": "Phase371C-History",
                "model": row["model"],
                "depth_pair": name,
                "source_layer": source,
                "receiver_layer": receiver,
                "case_count": row["case_count"],
                "continuity_relative_error_max": row["max_layer_continuity_relative_error"],
                "status": "verified_measurement_not_language_path",
                "language_path": False,
                "single_unit_causal": False,
            })
    nodes = [
        {"node_id": "p371_exact_qk_engineering", "node_type": "measurement_gate", "phase_id": "Phase371B-R", "status": "passed", "language_path": False},
        {"node_id": "p371c_behavior_gate", "node_type": "behavior_gate", "phase_id": "Phase371C", "status": "two_of_four_mechanisms_admitted", "case_count": 864, "language_path": False},
        {"node_id": "p371c_exact_ledger", "node_type": "measurement_ledger", "phase_id": "Phase371C", "status": "passed", "case_count": 264, "language_path": False},
        {"node_id": "p371c_local_continuity", "node_type": "local_physical_path", "phase_id": "Phase371C-Adj", "status": "measurement_verified", "continuity_rows": 2376, "language_path": False},
        {"node_id": "p371c_blind_contrast", "node_type": "blind_candidate_index", "phase_id": "Phase371C-Contrast", "status": "complete", "row_count": 299376, "language_path": False},
        {"node_id": "p371c_provisional_crossmodel", "node_type": "provisional_gate", "phase_id": "Phase371C-Discovery", "status": "failed_history_gate", "provisional_level2": 39, "provisional_level3": 3, "language_path": False},
        {"node_id": "p371c_history_gate", "node_type": "history_sufficiency_gate", "phase_id": "Phase371C-History", "status": "crossmodel_zero", "level2": 0, "level3": 0, "language_path": False},
        {"node_id": "p371c_calibration", "node_type": "sealed_evidence", "phase_id": "Phase371C", "status": "sealed", "language_path": False},
    ]
    edges = [
        {"edge_id": "p371_exact_qk_engineering->p371c_exact_ledger", "source_node_id": "p371_exact_qk_engineering", "target_node_id": "p371c_exact_ledger", "edge_type": "authorizes_exact_collection", "phase_id": "Phase371C"},
        {"edge_id": "p371c_behavior_gate->p371c_exact_ledger", "source_node_id": "p371c_behavior_gate", "target_node_id": "p371c_exact_ledger", "edge_type": "freezes_discovery_denominator", "phase_id": "Phase371C"},
        {"edge_id": "p371c_exact_ledger->p371c_local_continuity", "source_node_id": "p371c_exact_ledger", "target_node_id": "p371c_local_continuity", "edge_type": "adds_adjacent_measurement", "phase_id": "Phase371C-Adj"},
        {"edge_id": "p371c_local_continuity->p371c_blind_contrast", "source_node_id": "p371c_local_continuity", "target_node_id": "p371c_blind_contrast", "edge_type": "authorizes_blind_index", "phase_id": "Phase371C-Contrast"},
        {"edge_id": "p371c_blind_contrast->p371c_provisional_crossmodel", "source_node_id": "p371c_blind_contrast", "target_node_id": "p371c_provisional_crossmodel", "edge_type": "semantic_discovery_only", "phase_id": "Phase371C-Discovery"},
        {"edge_id": "p371c_provisional_crossmodel->p371c_history_gate", "source_node_id": "p371c_provisional_crossmodel", "target_node_id": "p371c_history_gate", "edge_type": "fails_crossmodel_history", "phase_id": "Phase371C-History"},
        {"edge_id": "p371c_history_gate->p371c_calibration", "source_node_id": "p371c_history_gate", "target_node_id": "p371c_calibration", "edge_type": "keeps_sealed", "phase_id": "Phase371C-History"},
    ]
    payloads["phase371_exact_vector_stage_summary.json"] = stage
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase371_measured_layer_pairs.jsonl", measured_pairs)
        write_jsonl(root / "phase371_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase371_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase371C-History"
        manifest["phase371"] = {
            "status": "exact_measurement_complete_history_crossmodel_gate_negative",
            "behavior_case_count": 864,
            "internal_discovery_case_count": 264,
            "combined_exact_ledger_bytes": adjacent["storage"]["combined_bytes"],
            "lazy_explicit_node_count": paths["denominator"]["explicit_node_count"],
            "blind_route_row_count": contrast["denominator"]["route_contrast_row_count"],
            "history_level2_count": 0,
            "history_level3_count": 0,
            "language_path_discovered": False,
            "single_unit_causal_count": 0,
            "files": [*SOURCES.keys(), "phase371_exact_vector_stage_summary.json", "phase371_measured_layer_pairs.jsonl", "phase371_evidence_nodes.jsonl", "phase371_evidence_edges.jsonl"],
        }
        write_json(root / "manifest.json", manifest)
        progress = read_json(root / "progress.json")
        progress["last_phase"] = "Phase371C-History"
        progress["updated_at"] = updated_at
        progress["single_global_progress_percentage_valid"] = False
        progress["global_research_estimates"] = {
            "status": "invalid_for_scientific_completion",
            "reason": "heterogeneous_denominators_and_zero_strict_closure_cells",
            "single_scalar_estimate_valid": False,
        }
        progress["exact_vector_stage"] = {
            "behavior_eligible_mechanisms": {"numerator": 2, "denominator": 4},
            "exact_discovery_ledger": {"numerator": 264, "denominator": 264},
            "local_continuity": {"numerator": 2376, "denominator": 2376},
            "history_full_gate_models": {"numerator": 3, "denominator": 210},
            "history_heterogeneous_level_2": {"numerator": 0, "denominator": 39},
            "strict_language_paths": {"numerator": 0, "denominator": 18},
            "physical_confirmation_opened": False,
        }
        progress["phase371_decision"] = "stop_before_calibration_after_history_gate_removed_all_crossmodel_routes"
        write_json(root / "progress.json", progress)
        public_manifest(root, updated_at)
    sync_neuron_atlas(stage, nodes, edges, updated_at)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
