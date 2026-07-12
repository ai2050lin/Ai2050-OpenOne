#!/usr/bin/env python3
"""Publish compact Phase365-368 dynamic-path evidence without private tensors or labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation"
TARGET = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"

SOURCES = {
    "phase365_repeat_noise_summary.json": PHASE_ROOT / "repeat_noise_format_gate/phase365_repeat_noise_summary.json",
    "phase365_collection_freeze_summary.json": PHASE_ROOT / "engineering_collection_freeze/phase365_collection_freeze_summary.json",
    "phase366_full_collection_summary.json": PHASE_ROOT / "engineering_collection/phase366_full_collection_summary.json",
    "phase366_dynamic_bundle_summary.json": PHASE_ROOT / "dynamic_bundle_extraction/phase365_dynamic_bundle_summary.json",
    "phase366_bundle_split_summary.json": PHASE_ROOT / "dynamic_bundle_extraction/phase366_bundle_split_summary.json",
    "phase366_derived_artifact_integrity_summary.json": PHASE_ROOT / "dynamic_bundle_extraction/phase366_derived_artifact_integrity_summary.json",
    "phase366_descriptor_summary.json": PHASE_ROOT / "label_blind_flow_descriptors/phase366_descriptor_summary.json",
    "phase366_threshold_custodian_summary.json": PHASE_ROOT / "blind_threshold_custodian/phase366_threshold_custodian_summary.json",
    "phase367_blind_motif_discovery_summary.json": PHASE_ROOT / "blind_motif_discovery/phase367_blind_motif_discovery_summary.json",
    "phase368_blind_motif_calibration_summary.json": PHASE_ROOT / "blind_motif_calibration/phase368_blind_motif_calibration_summary.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    payloads = {name: read_json(path) for name, path in SOURCES.items()}
    for name, payload in payloads.items():
        write_json(TARGET / name, payload)

    collection = payloads["phase366_full_collection_summary.json"]
    bundles = payloads["phase366_dynamic_bundle_summary.json"]
    split = payloads["phase366_bundle_split_summary.json"]
    integrity = payloads["phase366_derived_artifact_integrity_summary.json"]
    descriptors = payloads["phase366_descriptor_summary.json"]
    thresholds = payloads["phase366_threshold_custodian_summary.json"]
    discovery = payloads["phase367_blind_motif_discovery_summary.json"]
    calibration = payloads["phase368_blind_motif_calibration_summary.json"]
    updated_at = datetime.now(timezone.utc).isoformat()

    stage_summary = {
        "schema_version": "45.1.0",
        "phase_id": "Phase368",
        "created_at": updated_at,
        "objective": "replace_lossy_static_norm_candidates_with_replayable_directed_dynamic_flow_objects",
        "assessment": {
            "phase363_static_norm_route_closed": True,
            "dynamic_state_existence_refuted": False,
            "old_projection_has_fundamental_information_loss": True,
            "new_dynamic_descriptor_algorithm_is_sufficient": False,
        },
        "objective_denominators": {
            "models": 3,
            "free_rollout_cases": collection["denominator"]["case_count"],
            "independent_groups": split["denominator"]["independent_group_count"],
            "dynamic_bundles": bundles["denominator"]["bundle_count"],
            "typed_events": bundles["denominator"]["event_count"],
            "typed_edges": bundles["denominator"]["edge_count"],
            "derived_role_edge_files": integrity["denominator"]["unique_derived_file_reference_count"],
            "directed_path_descriptors": descriptors["denominator"]["directed_path_descriptor_count"],
            "threshold_rows": thresholds["denominator"]["threshold_row_count"],
            "blind_discovery_candidates": discovery["results"]["frozen_discovery_candidate_count"],
            "blind_calibrated_local_motifs": calibration["results"]["calibrated_candidate_count"],
            "cross_model_calibrated_signatures": calibration["results"]["cross_model_three_of_three_signature_count"],
            "physical_confirmation_cases_opened": 0,
            "strict_mechanism_closure_numerator": 0,
            "strict_mechanism_closure_denominator": 72,
        },
        "results": {
            "collection_models_valid": collection["results"]["valid_model_count"],
            "dynamic_bundles_valid": bundles["results"]["valid_bundle_count"],
            "derived_files_valid": integrity["results"]["status_counts"]["valid"],
            "all_source_routes_retained": descriptors["results"]["all_source_routes_retained"],
            "condition_average_used": False,
            "semantic_or_target_labels_used_in_discovery_or_calibration": False,
            "calibrated_local_motif_count": calibration["results"]["calibrated_candidate_count"],
            "cross_model_language_path_candidate_count": 0,
            "language_path_discovered": False,
            "language_encoding_mechanism_closed": False,
        },
        "hard_limits": [
            "four_role_alias_scope_not_all_token_positions",
            "attention_heads_aggregated_after_output_projection_in_motif_descriptors",
            "mlp_neuron_writes_recoverable_but_not_enumerated_in_motif_scoring",
            "ten_scalar_descriptors_and_ten_depth_anchors_remain_lossy",
            "exact_cross_model_discrete_signature_equivalence_may_be_too_rigid",
            "candidate_gates_are_engineering_preregistration_not_a_mathematical_language_law",
            "only_four_of_eighteen_mechanisms_are_admitted_to_this_dynamic_denominator",
            "small_model_internal_paths_may_not_transfer_to_larger_models",
        ],
        "authorization": {
            "show_dynamic_measurement_ledger_in_client": True,
            "show_four_local_motifs_as_language_family_paths": False,
            "semantic_label_reveal": False,
            "physical_confirmation": False,
            "causal_intervention": False,
        },
        "next_stage": {
            "requires_new_independent_development_and_calibration_data": True,
            "reuse_phase368_calibration_to_tune_equivalence": False,
            "priority": "raw_vector_topology_preserving_cross_model_equivalence_before_new_confirmation",
        },
        "single_global_progress_percentage_valid": False,
    }
    write_json(TARGET / "phase368_dynamic_path_stage_summary.json", stage_summary)

    manifest_path = TARGET / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["updated_at"] = updated_at
    manifest["last_phase"] = "Phase368"
    manifest["phase365"] = {
        "status": "three_model_dynamic_instrumentation_and_free_rollout_collection_complete",
        "model_count": 3,
        "case_count": collection["denominator"]["case_count"],
        "layer_file_count": collection["denominator"]["layer_file_count"],
        "valid_model_count": collection["results"]["valid_model_count"],
        "physical_confirmation_opened": False,
        "raw_tensors_frontend_exported": False,
        "files": ["phase365_repeat_noise_summary.json", "phase365_collection_freeze_summary.json", "phase366_full_collection_summary.json"],
    }
    manifest["phase366"] = {
        "status": "full_288_role_dynamic_bundles_and_blind_descriptors_complete",
        "bundle_count": bundles["denominator"]["bundle_count"],
        "event_count": bundles["denominator"]["event_count"],
        "edge_count": bundles["denominator"]["edge_count"],
        "derived_file_count": integrity["denominator"]["unique_derived_file_reference_count"],
        "descriptor_count": descriptors["denominator"]["directed_path_descriptor_count"],
        "all_token_position_scope_complete": False,
        "physical_confirmation_opened": False,
        "raw_tensors_frontend_exported": False,
        "files": [
            "phase366_dynamic_bundle_summary.json", "phase366_bundle_split_summary.json",
            "phase366_derived_artifact_integrity_summary.json", "phase366_descriptor_summary.json",
            "phase366_threshold_custodian_summary.json",
        ],
    }
    manifest["phase367"] = {
        "status": "blind_discovery_motifs_frozen_not_language_paths",
        "discovery_case_count": discovery["denominator"]["discovery_case_count"],
        "enumerated_window_count": discovery["denominator"]["enumerated_window_occurrence_count"],
        "frozen_candidate_count": discovery["results"]["frozen_discovery_candidate_count"],
        "cross_model_signature_count": discovery["results"]["cross_model_three_of_three_signature_count"],
        "semantic_labels_used": False,
        "files": ["phase367_blind_motif_discovery_summary.json"],
    }
    manifest["phase368"] = {
        "status": "four_model_specific_calibrated_motifs_zero_cross_model_language_paths",
        "calibration_case_count": calibration["denominator"]["calibration_case_count"],
        "calibrated_local_motif_count": calibration["results"]["calibrated_candidate_count"],
        "cross_model_signature_count": calibration["results"]["cross_model_three_of_three_signature_count"],
        "language_path_discovered": False,
        "physical_confirmation_opened": False,
        "raw_tensors_frontend_exported": False,
        "files": ["phase368_blind_motif_calibration_summary.json", "phase368_dynamic_path_stage_summary.json"],
    }
    write_json(manifest_path, manifest)

    progress_path = TARGET / "progress.json"
    progress = read_json(progress_path)
    progress["last_phase"] = "Phase368"
    progress["updated_at"] = updated_at
    progress["dynamic_path_stage"] = {
        "free_rollout_case_coverage": {"numerator": 288, "denominator": 288},
        "dynamic_bundle_validity": {"numerator": 288, "denominator": 288},
        "derived_artifact_integrity": {"numerator": 29952, "denominator": 29952},
        "blind_discovery_candidate_count": discovery["results"]["frozen_discovery_candidate_count"],
        "blind_calibrated_local_motif_count": calibration["results"]["calibrated_candidate_count"],
        "cross_model_calibrated_language_path_count": 0,
        "physical_confirmation_opened": False,
    }
    progress["objective_denominator_progress"]["strict_mechanism_closure"] = {"numerator": 0, "denominator": 72}
    progress["single_global_progress_percentage_valid"] = False
    progress["phase368_decision"] = "stop_without_label_reveal_and_revise_cross_model_dynamic_equivalence_on_new_independent_data"
    write_json(progress_path, progress)

    for name, payload in payloads.items():
        write_json(CLIENT / name, payload)
    write_json(CLIENT / "phase368_dynamic_path_stage_summary.json", stage_summary)
    write_json(CLIENT / "manifest.json", manifest)
    write_json(CLIENT / "progress.json", progress)
    public_manifest_path = CLIENT / "public_manifest.json"
    public_manifest = read_json(public_manifest_path)
    public_manifest["generated_at"] = updated_at
    public_manifest["source"] = str(TARGET.relative_to(ROOT))
    public_manifest.pop("sources", None)
    public_manifest["files"] = sorted(
        str(path.relative_to(CLIENT))
        for path in CLIENT.rglob("*")
        if path.is_file() and path != public_manifest_path and path.suffix.lower() in {".json", ".jsonl", ".md"}
    )
    write_json(public_manifest_path, public_manifest)
    print(json.dumps(stage_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
