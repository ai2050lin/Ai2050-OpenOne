#!/usr/bin/env python3
"""Publish compact Phase369 evidence and synchronize both atlas clients."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
TARGET = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON_TARGET = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


SOURCES = {
    "phase369_case_bank_summary.json": PHASE_ROOT / "raw_topology_preregister/phase369_case_bank_summary.json",
    "phase369_behavior_qualification_summary.json": PHASE_ROOT / "behavior_qualification_final_v2/phase369_behavior_qualification_final_v2_summary.json",
    "phase369_bundle_contract_repair_summary.json": PHASE_ROOT / "dynamic_bundle_extraction/phase369_bundle_contract_repair_summary.json",
    "phase369_dynamic_bundle_summary.json": PHASE_ROOT / "dynamic_bundle_extraction/phase365_dynamic_bundle_summary.json",
    "phase369_raw_relation_schema.json": PHASE_ROOT / "raw_relation_features/phase369_raw_relation_schema.json",
    "phase369_raw_relation_summary.json": PHASE_ROOT / "raw_relation_features/phase369_raw_relation_summary.json",
    "phase369_blind_future_crossmodel_summary.json": PHASE_ROOT / "blind_future_and_crossmodel/phase369_blind_future_and_crossmodel_summary.json",
    "phase370_head_neuron_topology_summary.json": PHASE_ROOT / "head_neuron_topology_diagnostic/phase369_head_neuron_topology_summary.json",
    "phase370_head_neuron_diagnostic_summary.json": PHASE_ROOT / "head_neuron_topology_diagnostic_evaluation/phase369_head_neuron_diagnostic_evaluation_summary.json",
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


def compact_collection_summary() -> dict[str, Any]:
    manifests = [read_json(PHASE_ROOT / "raw_collection/models" / model / "manifest.json") for model in ("qwen3", "glm4", "deepseek7b")]
    return {
        "schema_version": "46.5.0",
        "phase_id": "Phase369",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": 3,
            "case_count": sum(item["case_count"] for item in manifests),
            "layer_file_count": sum(item["file_count"] for item in manifests),
            "total_byte_count": sum(item["total_byte_count"] for item in manifests),
            "generation_time_count": 3,
        },
        "results": {
            "valid_model_count": sum(bool(item["valid"]) for item in manifests),
            "all_case_gates_pass": all(item["all_case_gates_pass"] for item in manifests),
            "max_gate_errors": {
                key: max(float(item["gate_maxima"][key]) for item in manifests)
                for key in manifests[0]["gate_maxima"]
            },
            "semantic_or_target_labels_used": False,
            "calibration_or_physical_case_used": False,
            "raw_tensors_exported_to_frontend": False,
        },
        "models": [
            {
                "model": item["model"],
                "case_count": item["case_count"],
                "layer_count": item["layer_count"],
                "file_count": item["file_count"],
                "total_byte_count": item["total_byte_count"],
                "valid": item["valid"],
            }
            for item in manifests
        ],
    }


def sync_neuron_atlas(stage_summary: dict[str, Any], updated_at: str) -> None:
    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase369_raw_topology_stage_summary.json", stage_summary)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        manifest["phase"] = 370
        manifest["generated_at"] = updated_at
        manifest["phase369_370_audit"] = {
            "status": "raw_topology_and_hash_energy_diagnostic_strict_negative",
            "new_neuron_path_nodes_promoted": 0,
            "single_unit_causal_count": 0,
            "calibration_opened": False,
            "physical_holdout_opened": False,
            "source": "phase369_raw_topology_stage_summary.json",
        }
        manifest["files"]["latest_evidence_summary"] = "phase369_raw_topology_stage_summary.json"
        evidence = manifest.setdefault("evidence_boundary", {})
        evidence["statement"] = (
            "Phase369 completed a 336-case raw directed-flow ledger, but raw Gram relations failed the full future gate; "
            "Phase370 fixed-hash head/neuron energy topology also failed. No new neuron or language-family path is promoted."
        )
        evidence["latest_phase"] = "Phase370"
        evidence["raw_topology_path_closed"] = False
        evidence["single_unit_causal_closure"] = False
        evidence["language_encoding_mechanism_closed"] = False
        write_json(manifest_path, manifest)
        checksum_path = root / "checksums.json"
        if checksum_path.is_file():
            checksum = {
                "schema_version": "artifact_checksums.v1",
                "files": [
                    {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
                    for path in sorted(root.rglob("*"))
                    if path.is_file() and path != checksum_path
                ],
            }
            write_json(checksum_path, checksum)


def main() -> None:
    payloads = {name: read_json(path) for name, path in SOURCES.items()}
    collection = compact_collection_summary()
    payloads["phase369_raw_collection_summary.json"] = collection
    behavior = payloads["phase369_behavior_qualification_summary.json"]
    bundles = payloads["phase369_dynamic_bundle_summary.json"]
    raw_relations = payloads["phase369_raw_relation_summary.json"]
    discovery = payloads["phase369_blind_future_crossmodel_summary.json"]
    topology = payloads["phase370_head_neuron_topology_summary.json"]
    diagnostic = payloads["phase370_head_neuron_diagnostic_summary.json"]
    updated_at = datetime.now(timezone.utc).isoformat()

    stage_summary = {
        "schema_version": "46.5.0",
        "phase_id": "Phase370",
        "created_at": updated_at,
        "objective": "test_raw_coordinate_invariant_flow_relations_then_diagnose_fixed_hash_head_neuron_energy_topology",
        "assessment": {
            "phase364_368_projection_loss_assessment_supported": True,
            "phase369_raw_relation_state_sufficient": False,
            "phase370_hash_energy_topology_sufficient": False,
            "dynamic_language_state_existence_refuted": False,
            "language_encoding_mechanism_closed": False,
        },
        "objective_denominators": {
            "registered_language_families": 9,
            "registered_representative_mechanisms": 18,
            "admitted_mechanisms": 4,
            "preregistered_cases": 576,
            "behavior_qualified_discovery_cases": behavior["qualification"]["fresh_discovery_blind_case_count"],
            "behavior_qualified_calibration_cases_kept_sealed": behavior["qualification"]["fresh_calibration_blind_case_count"],
            "raw_collection_cases": collection["denominator"]["case_count"],
            "raw_layer_files": collection["denominator"]["layer_file_count"],
            "raw_bytes": collection["denominator"]["total_byte_count"],
            "valid_dynamic_bundles": bundles["results"]["valid_bundle_count"],
            "typed_events": bundles["denominator"]["event_count"],
            "typed_edges": bundles["denominator"]["edge_count"],
            "raw_relation_route_rows": raw_relations["denominator"]["route_row_count"],
            "head_route_rows": topology["denominator"]["head_route_count"],
            "mlp_role_rows": topology["denominator"]["mlp_role_count"],
            "level_1_predictive_models": discovery["evidence"]["level_1_model_count"],
            "level_2_heterogeneous_pairs": discovery["evidence"]["level_2_heterogeneous_pair_count"],
            "calibration_internal_cases_opened": 0,
            "physical_holdout_cases_opened": 0,
            "strictly_closed_registered_cells": 0,
            "registered_closure_cells": 72,
        },
        "results": {
            "measurement_ledger_valid": collection["results"]["valid_model_count"] == 3,
            "dynamic_bundles_valid": bundles["results"]["valid_bundle_count"] == 336,
            "raw_relations_beat_all_frozen_future_gates": False,
            "glm4_deepseek_pair_raw_crossmodel_components_passed_before_future_gate": True,
            "future_and_crossmodel_joint_level_2_passed": False,
            "hash_energy_diagnostic_new_cycle_authorized": diagnostic["authorization"]["new_independent_topology_cycle"],
            "language_path_candidate_count": 0,
            "single_neuron_causal_count": 0,
        },
        "hard_limits": [
            "four_role_scope_not_all_token_positions",
            "gram_and_norm_share_relations_still_discard_absolute_vector_coordinates_and_higher_order_interactions",
            "thirty_two_point_depth_resampling_is_a_comparison_view_not_the_original_trajectory",
            "nearest_neighbor_future_prediction_is_a_basic_probe_not_a_learned_transition_law",
            "hash_energy_topology_discards_within_shard_vector_direction_and_cross_terms",
            "hash_seeds_and_resolutions_are_sensitivity_checks_not_replications",
            "only_four_of_eighteen_representative_mechanisms_are_admitted",
            "small_model_paths_may_be_coarser_than_large_model_paths",
        ],
        "authorization": {
            "show_verified_measurement_ledger_in_client": True,
            "show_phase369_or_phase370_as_language_family_paths": False,
            "show_raw_private_tensors_in_client": False,
            "open_behavior_qualified_calibration_internal_traces": False,
            "physical_confirmation": False,
            "causal_intervention": False,
        },
        "next_stage": {
            "priority": "exact_vector_coactivity_branch_state_object_before_any_new_model_cycle",
            "reuse_phase369_discovery_or_sealed_calibration_for_tuning": False,
            "continue_hash_energy_expansion": False,
        },
        "single_global_progress_percentage_valid": False,
    }
    payloads["phase369_raw_topology_stage_summary.json"] = stage_summary

    nodes = [
        {"node_id": "p369_behavior_gate", "node_type": "evidence_gate", "phase_id": "Phase369", "status": "passed", "case_count": 516, "language_path": False},
        {"node_id": "p369_raw_ledger", "node_type": "measurement_ledger", "phase_id": "Phase369", "status": "passed", "case_count": 336, "event_count": bundles["denominator"]["event_count"], "language_path": False},
        {"node_id": "p369_raw_relation", "node_type": "state_candidate", "phase_id": "Phase369", "status": "failed_full_gate", "route_row_count": raw_relations["denominator"]["route_row_count"], "language_path": False},
        {"node_id": "p369_crossmodel", "node_type": "cross_model_gate", "phase_id": "Phase369", "status": "failed_joint_future_gate", "level_2_pair_count": 0, "language_path": False},
        {"node_id": "p370_hash_topology", "node_type": "state_candidate", "phase_id": "Phase370", "status": "failed_diagnostic_gate", "head_route_count": topology["denominator"]["head_route_count"], "mlp_role_count": topology["denominator"]["mlp_role_count"], "language_path": False},
        {"node_id": "p369_sealed_calibration", "node_type": "sealed_evidence", "phase_id": "Phase369", "status": "sealed", "case_count": 180, "language_path": False},
    ]
    edges = [
        {"edge_id": "p369_behavior_gate->p369_raw_ledger", "source_node_id": "p369_behavior_gate", "target_node_id": "p369_raw_ledger", "edge_type": "authorizes_measurement", "phase_id": "Phase369"},
        {"edge_id": "p369_raw_ledger->p369_raw_relation", "source_node_id": "p369_raw_ledger", "target_node_id": "p369_raw_relation", "edge_type": "derives_candidate", "phase_id": "Phase369"},
        {"edge_id": "p369_raw_relation->p369_crossmodel", "source_node_id": "p369_raw_relation", "target_node_id": "p369_crossmodel", "edge_type": "fails_joint_gate", "phase_id": "Phase369"},
        {"edge_id": "p369_raw_relation->p370_hash_topology", "source_node_id": "p369_raw_relation", "target_node_id": "p370_hash_topology", "edge_type": "motivates_diagnostic", "phase_id": "Phase370"},
        {"edge_id": "p369_crossmodel->p369_sealed_calibration", "source_node_id": "p369_crossmodel", "target_node_id": "p369_sealed_calibration", "edge_type": "keeps_sealed", "phase_id": "Phase369"},
    ]
    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase369_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase369_evidence_edges.jsonl", edges)

    manifest = read_json(TARGET / "manifest.json")
    manifest["updated_at"] = updated_at
    manifest["last_phase"] = "Phase370"
    manifest["phase369"] = {
        "status": "raw_relation_strict_negative_calibration_and_physical_holdout_sealed",
        "behavior_qualified_discovery_cases": 336,
        "behavior_qualified_calibration_cases": 180,
        "raw_collection_cases": 336,
        "raw_relation_route_rows": 605696,
        "level_1_model_count": 0,
        "level_2_heterogeneous_pair_count": 0,
        "language_path_discovered": False,
        "files": [
            "phase369_behavior_qualification_summary.json", "phase369_raw_collection_summary.json",
            "phase369_dynamic_bundle_summary.json", "phase369_raw_relation_summary.json",
            "phase369_blind_future_crossmodel_summary.json", "phase369_evidence_nodes.jsonl",
            "phase369_evidence_edges.jsonl", "phase369_raw_topology_stage_summary.json",
        ],
    }
    manifest["phase370"] = {
        "status": "fixed_hash_head_neuron_energy_topology_diagnostic_negative",
        "head_route_count": 605696,
        "mlp_role_count": 128128,
        "new_independent_cycle_authorized": False,
        "single_unit_causal_count": 0,
        "files": ["phase370_head_neuron_topology_summary.json", "phase370_head_neuron_diagnostic_summary.json"],
    }
    write_json(TARGET / "manifest.json", manifest)
    write_json(CLIENT / "manifest.json", manifest)

    progress = read_json(TARGET / "progress.json")
    progress["last_phase"] = "Phase370"
    progress["updated_at"] = updated_at
    progress["client_stage_coverage"] = {
        "pattern_family_atlas": 9 / 9,
        "physical_path_atlas": 4 / 18,
        "component_path_atlas": 0 / 18,
        "readout_competition_trace": 1 / 18,
        "stepwise_rollout_trace": 4 / 18,
        "causal_closure": 0 / 72,
    }
    progress["raw_topology_stage"] = {
        "behavior_qualified_discovery_cases": {"numerator": 336, "denominator": 336},
        "raw_collection_validity": {"numerator": 336, "denominator": 336},
        "dynamic_bundle_validity": {"numerator": 336, "denominator": 336},
        "raw_relation_full_gate_models": {"numerator": 0, "denominator": 3},
        "heterogeneous_level_2_pairs": {"numerator": 0, "denominator": 2},
        "physical_confirmation_opened": False,
    }
    progress["single_global_progress_percentage_valid"] = False
    progress["phase369_decision"] = "strict_negative_stop_before_calibration_and_physical_holdout"
    progress["phase370_decision"] = "hash_energy_topology_negative_design_exact_vector_coactivity_path_object"
    write_json(TARGET / "progress.json", progress)
    write_json(CLIENT / "progress.json", progress)

    public_manifest_path = CLIENT / "public_manifest.json"
    public_manifest = read_json(public_manifest_path)
    public_manifest["generated_at"] = updated_at
    public_manifest["files"] = sorted(
        str(path.relative_to(CLIENT))
        for path in CLIENT.rglob("*")
        if path.is_file() and path != public_manifest_path and path.suffix.lower() in {".json", ".jsonl", ".md"}
    )
    write_json(public_manifest_path, public_manifest)
    sync_neuron_atlas(stage_summary, updated_at)
    print(json.dumps(stage_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
