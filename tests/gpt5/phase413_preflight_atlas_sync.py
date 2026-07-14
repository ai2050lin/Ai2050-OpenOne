#!/usr/bin/env python3
"""Publish Phase413 measurement-boundary evidence without physical promotion."""

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
P413 = ROOT / "tests/gpt5/result/phase413_prediction_kernel_preflight"
JSON_SOURCES = {
    "phase413_source_claim_audit.json": P413 / "phase413_source_claim_audit.json",
    "phase413_candidate_panel_contract.json": P413
    / "phase413_candidate_panel_contract.json",
    "phase413_terminal_nonidentifiability_audit.json": P413
    / "phase413_terminal_nonidentifiability_audit.json",
    "phase413_future_equivalence_audit.json": P413
    / "phase413_future_equivalence_audit.json",
    "phase413_channel_permutation_audit.json": P413
    / "phase413_channel_permutation_audit.json",
    "phase413_readout_qualification.json": P413
    / "phase413_readout_qualification.json",
    "phase413_execution_qualification.json": P413
    / "phase413_execution_qualification.json",
    "phase413_stage_summary.json": P413 / "phase413_stage_summary.json",
}


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


def evidence_graph(stage: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    denominators = stage["denominators"]
    results = stage["results"]
    common = {
        "phase_id": "Phase413",
        "protocol_only": True,
        "behavior_observed": False,
        "causal": False,
        "physical": False,
        "language_path": False,
    }
    nodes = [
        {
            "node_id": "p413_supplied_claim_boundary_audit",
            "node_type": "source_claim_evidence_boundary_audit",
            "claim_count": denominators["source_claim_count"],
            "supported_claim_count": results["supported_claim_count"],
            "qualification_required_claim_count": results[
                "qualification_required_claim_count"
            ],
            "incorrect_as_stated_claim_count": results[
                "incorrect_as_stated_claim_count"
            ],
            **common,
        },
        {
            "node_id": "p413_terminal_kernel_readout_boundary",
            "node_type": "native_terminal_vs_intermediate_readout_qualification",
            "readout_mode_count": denominators["readout_mode_count"],
            "native_terminal_method_count": results["native_terminal_method_count"],
            "direct_layer_local_readout_mode_count": denominators[
                "direct_layer_local_readout_mode_count"
            ],
            "qualified_direct_layer_local_probability_readout_count": results[
                "qualified_direct_layer_local_probability_readout_count"
            ],
            **common,
        },
        {
            "node_id": "p413_terminal_trajectory_nonidentifiability",
            "node_type": "terminally_identical_internally_distinct_finite_trajectories",
            "synthetic_path_count": denominators["synthetic_path_count"],
            "same_terminal_distribution_path_count": results[
                "same_terminal_distribution_path_count"
            ],
            "path_pair_count": denominators["synthetic_path_pair_count"],
            "different_internal_pair_count": results[
                "same_endpoint_different_internal_pair_count"
            ],
            **common,
        },
        {
            "node_id": "p413_future_equivalence_counterexample",
            "node_type": "one_step_equal_future_different_finite_kernel_pair",
            "state_pair_count": denominators["future_state_pair_count"],
            "one_step_equal_but_future_different_pair_count": results[
                "one_step_equal_but_future_different_pair_count"
            ],
            **common,
        },
        {
            "node_id": "p413_channel_permutation_coordinate_boundary",
            "node_type": "mlp_channel_permutation_output_invariance_coordinate_probe_failure",
            "case_count": denominators["channel_permutation_case_count"],
            "native_output_invariant_case_count": results[
                "native_output_invariant_channel_case_count"
            ],
            "fixed_coordinate_probe_failure_count": results[
                "fixed_coordinate_probe_failure_count"
            ],
            "transported_probe_invariant_case_count": results[
                "transported_probe_invariant_case_count"
            ],
            **common,
        },
        {
            "node_id": "p413_equal_horizon_multiaxis_candidate_panel",
            "node_type": "synthetic_disjoint_equal_horizon_candidate_panel_contract",
            "candidate_count": denominators["candidate_panel_case_count"],
            "model_token_ids_registered": False,
            "real_model_panel_exhaustive": False,
            **common,
        },
        {
            "node_id": "p413_execution_boundary",
            "node_type": "external_review_collector_and_readout_execution_boundary",
            "completed_external_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "required_external_reviewer_count": denominators[
                "required_independent_reviewer_count"
            ],
            "sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "sealed_model_collector_denominator": denominators[
                "future_sealed_model_collector_case_count"
            ],
            "model_execution_authorized": False,
            **common,
        },
    ]
    edges = [
        {
            "edge_id": "p412_execution_boundary_to_p413_claim_audit",
            "source_node_id": "p412_execution_boundary",
            "target_node_id": "p413_supplied_claim_boundary_audit",
            "edge_type": "audits_new_theory_without_reopening_closed_execution_gates",
            "phase_id": "Phase413",
            "causal_path": False,
        },
        {
            "edge_id": "p413_claim_audit_to_readout_boundary",
            "source_node_id": "p413_supplied_claim_boundary_audit",
            "target_node_id": "p413_terminal_kernel_readout_boundary",
            "edge_type": "separates_native_terminal_kernel_from_intermediate_decoder_claims",
            "phase_id": "Phase413",
            "causal_path": False,
        },
        {
            "edge_id": "p413_readout_to_trajectory_nonidentifiability",
            "source_node_id": "p413_terminal_kernel_readout_boundary",
            "target_node_id": "p413_terminal_trajectory_nonidentifiability",
            "edge_type": "constructs_terminally_indistinguishable_internal_paths",
            "phase_id": "Phase413",
            "causal_path": False,
        },
        {
            "edge_id": "p413_trajectory_to_future_equivalence",
            "source_node_id": "p413_terminal_trajectory_nonidentifiability",
            "target_node_id": "p413_future_equivalence_counterexample",
            "edge_type": "separates_current_one_step_equality_from_full_future_equivalence",
            "phase_id": "Phase413",
            "causal_path": False,
        },
        {
            "edge_id": "p413_readout_to_channel_permutation",
            "source_node_id": "p413_terminal_kernel_readout_boundary",
            "target_node_id": "p413_channel_permutation_coordinate_boundary",
            "edge_type": "checks_fixed_coordinate_readout_under_exact_channel_relabeling",
            "phase_id": "Phase413",
            "causal_path": False,
        },
        {
            "edge_id": "p413_future_and_coordinate_to_candidate_panel",
            "source_node_id": "p413_future_equivalence_counterexample",
            "target_node_id": "p413_equal_horizon_multiaxis_candidate_panel",
            "edge_type": "freezes_disjoint_future_events_and_metric_scope",
            "phase_id": "Phase413",
            "causal_path": False,
        },
        {
            "edge_id": "p413_candidate_panel_to_execution_boundary",
            "source_node_id": "p413_equal_horizon_multiaxis_candidate_panel",
            "target_node_id": "p413_execution_boundary",
            "edge_type": "keeps_cuda_physical_causal_and_neuron_execution_closed",
            "phase_id": "Phase413",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_metrics(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    denominators = stage["denominators"]
    results = stage["results"]
    manifest.setdefault("metrics", {}).update(
        {
            "phase413_source_claim_count": denominators["source_claim_count"],
            "phase413_supported_claim_count": results["supported_claim_count"],
            "phase413_incorrect_claim_count": results[
                "incorrect_as_stated_claim_count"
            ],
            "phase413_readout_mode_count": denominators["readout_mode_count"],
            "phase413_native_terminal_method_count": results[
                "native_terminal_method_count"
            ],
            "phase413_direct_layer_local_readout_count": denominators[
                "direct_layer_local_readout_mode_count"
            ],
            "phase413_qualified_direct_layer_local_readout_count": results[
                "qualified_direct_layer_local_probability_readout_count"
            ],
            "phase413_synthetic_path_count": denominators["synthetic_path_count"],
            "phase413_same_terminal_path_count": results[
                "same_terminal_distribution_path_count"
            ],
            "phase413_synthetic_path_pair_count": denominators[
                "synthetic_path_pair_count"
            ],
            "phase413_internal_distinct_path_pair_count": results[
                "same_endpoint_different_internal_pair_count"
            ],
            "phase413_future_state_pair_count": denominators[
                "future_state_pair_count"
            ],
            "phase413_future_different_pair_count": results[
                "one_step_equal_but_future_different_pair_count"
            ],
            "phase413_channel_permutation_case_count": denominators[
                "channel_permutation_case_count"
            ],
            "phase413_native_output_invariant_channel_case_count": results[
                "native_output_invariant_channel_case_count"
            ],
            "phase413_fixed_coordinate_probe_failure_count": results[
                "fixed_coordinate_probe_failure_count"
            ],
            "phase413_candidate_panel_case_count": denominators[
                "candidate_panel_case_count"
            ],
            "phase413_completed_external_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "phase413_required_external_reviewer_count": denominators[
                "required_independent_reviewer_count"
            ],
            "phase413_sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "phase413_model_case_count": denominators["model_case_count_consumed"],
            "phase413_physical_case_count": denominators[
                "physical_case_count_consumed"
            ],
            "phase413_new_neuron_node_count": results["new_neuron_path_count"],
        }
    )


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    denominators = stage["denominators"]
    results = stage["results"]
    progress["last_phase"] = "Phase413-PredictionKernelMeasurementPreflightStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["prediction_kernel_measurement_preflight_stage"] = {
        "source_claims_audited": {
            "numerator": denominators["source_claim_count"],
            "denominator": denominators["source_claim_count"],
        },
        "terminal_identical_synthetic_paths": {
            "numerator": results["same_terminal_distribution_path_count"],
            "denominator": denominators["synthetic_path_count"],
        },
        "endpoint_identical_internal_distinct_pairs": {
            "numerator": results["same_endpoint_different_internal_pair_count"],
            "denominator": denominators["synthetic_path_pair_count"],
        },
        "one_step_equal_future_different_pairs": {
            "numerator": results[
                "one_step_equal_but_future_different_pair_count"
            ],
            "denominator": denominators["future_state_pair_count"],
        },
        "native_output_channel_permutation_invariance": {
            "numerator": results["native_output_invariant_channel_case_count"],
            "denominator": denominators["channel_permutation_case_count"],
        },
        "fixed_coordinate_probe_counterexamples": {
            "numerator": results["fixed_coordinate_probe_failure_count"],
            "denominator": denominators["channel_permutation_case_count"],
        },
        "candidate_panel_contract_cases": {
            "numerator": denominators["candidate_panel_case_count"],
            "denominator": denominators["candidate_panel_case_count"],
        },
        "qualified_direct_layer_local_probability_readouts": {
            "numerator": results[
                "qualified_direct_layer_local_probability_readout_count"
            ],
            "denominator": denominators["direct_layer_local_readout_mode_count"],
        },
        "independent_external_reviewers": {
            "numerator": results["completed_external_reviewer_count"],
            "denominator": denominators["required_independent_reviewer_count"],
        },
        "sealed_model_collector_equivalence": {
            "numerator": results["sealed_model_collector_equivalence_case_count"],
            "denominator": denominators["future_sealed_model_collector_case_count"],
        },
        "model_cases_consumed": {"numerator": 0, "denominator": 165},
        "physical_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase413_decision"] = (
        "publish_terminal_kernel_and_intermediate_readout_measurement_boundary_only;_"
        "keep_model_physical_causal_and_neuron_gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase413 preflight artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    stage = payloads["phase413_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        "phase413_evidence_nodes.jsonl",
        "phase413_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase413_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase413_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase413-PredictionKernelMeasurementPreflightStage"
        manifest["phase413"] = {
            "status": "measurement_boundary_pass_external_and_model_gates_closed",
            "files": published,
            "machine_preflight_pass": stage["assessment"]["machine_preflight_pass"],
            "same_terminal_synthetic_path_count": stage["results"][
                "same_terminal_distribution_path_count"
            ],
            "qualified_direct_layer_local_probability_readout_count": stage[
                "results"
            ]["qualified_direct_layer_local_probability_readout_count"],
            "model_case_count": 0,
            "physical_case_count": 0,
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase413-PredictionKernelMeasurementPreflightStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase413_stage_summary.json", stage)
        write_jsonl(root / "phase413_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase413_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 413
        manifest["generated_at"] = updated_at
        manifest["phase413_audit"] = {
            "status": "measurement_protocol_only_no_model_physical_or_neuron_evidence",
            "synthetic_path_count": stage["denominators"]["synthetic_path_count"],
            "same_terminal_distribution_path_count": stage["results"][
                "same_terminal_distribution_path_count"
            ],
            "qualified_direct_layer_local_probability_readout_count": stage[
                "results"
            ]["qualified_direct_layer_local_probability_readout_count"],
            "model_case_count": 0,
            "physical_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase413_stage_summary.json",
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        public_manifest(root, updated_at)
        update_checksums(root)

    print(
        json.dumps(
            {
                "valid": True,
                "phase_id": stage["phase_id"],
                "published_files": published,
                "evidence_node_count": len(nodes),
                "evidence_edge_count": len(edges),
                "model_case_count": 0,
                "physical_or_neuron_nodes_promoted": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
