#!/usr/bin/env python3
"""Publish Phase410 preflight evidence without promoting physical nodes."""

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
P410 = ROOT / "tests/gpt5/result/phase410_orthogonal_preflight"
JSON_SOURCES = {
    "phase410_orthogonal_state_contract.json": P410
    / "phase410_orthogonal_state_contract.json",
    "phase410_h3_order_symmetry_audit.json": P410
    / "phase410_h3_order_symmetry_audit.json",
    "phase410_grammar_finite_universe_audit.json": P410
    / "phase410_grammar_finite_universe_audit.json",
    "phase410_external_review_status.json": P410
    / "phase410_external_review_status.json",
    "phase410_collector_reducer_equivalence.json": P410
    / "phase410_collector_reducer_equivalence.json",
    "phase410_preflight_qualification.json": P410
    / "phase410_preflight_qualification.json",
    "phase410_preflight_stage_summary.json": P410
    / "phase410_preflight_stage_summary.json",
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
    denominator = stage["denominators"]
    results = stage["results"]
    nodes = [
        {
            "node_id": "p410_orthogonal_dynamic_measurement",
            "node_type": "orthogonal_external_dynamic_response_measurement_contract",
            "phase_id": "Phase410",
            "axis_count": denominator["orthogonal_axis_count"],
            "machine_valid": results["orthogonal_state_contract_failure_count"] == 0,
            "protocol_only": True,
            "behavior_observed": False,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p410_h3_order_symmetry",
            "node_type": "conflict_contract_order_reversal_machine_audit",
            "phase_id": "Phase410",
            "unordered_contract_pair_count": denominator[
                "h3_unordered_contract_pair_count"
            ],
            "order_variant_count": denominator["h3_order_variant_count"],
            "failure_count": results["h3_order_symmetry_failure_count"],
            "behavioral_order_invariance_observed": False,
            "protocol_only": True,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p410_grammar_finite_universe",
            "node_type": "frozen_grammar_response_contract_exhaustive_machine_audit",
            "phase_id": "Phase410",
            "finite_response_case_count": denominator[
                "grammar_finite_response_case_count"
            ],
            "failure_count": results["grammar_finite_universe_failure_count"],
            "general_grammar_closed": False,
            "protocol_only": True,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p410_external_review_gate",
            "node_type": "two_distinct_external_rule_reviewer_gate",
            "phase_id": "Phase410",
            "required_reviewer_count": denominator[
                "required_independent_reviewer_count"
            ],
            "completed_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "scenario_count_per_reviewer": denominator[
                "independent_review_scenario_count_per_reviewer"
            ],
            "machine_self_review_allowed": False,
            "protocol_only": True,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
        {
            "node_id": "p410_model_execution_boundary",
            "node_type": "sealed_collector_model_physical_causal_neuron_gate",
            "phase_id": "Phase410",
            "sealed_model_equivalence_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "sealed_model_equivalence_case_denominator": denominator[
                "future_sealed_model_qualification_case_count"
            ],
            "model_case_count": denominator["model_case_count_consumed"],
            "physical_case_count": denominator["physical_case_count_consumed"],
            "model_execution_authorized": False,
            "protocol_only": True,
            "causal": False,
            "physical": False,
            "language_path": False,
        },
    ]
    edges = [
        {
            "edge_id": "p409_protocol_to_p410_orthogonal_contract",
            "source_node_id": "p409_dynamic_history_protocol",
            "target_node_id": "p410_orthogonal_dynamic_measurement",
            "edge_type": "replaces_single_external_automaton_with_independent_measurement_axes",
            "phase_id": "Phase410",
            "causal_path": False,
        },
        {
            "edge_id": "p410_orthogonal_to_h3_symmetry",
            "source_node_id": "p410_orthogonal_dynamic_measurement",
            "target_node_id": "p410_h3_order_symmetry",
            "edge_type": "requires_order_reversal_before_conflict_behavior_measurement",
            "phase_id": "Phase410",
            "causal_path": False,
        },
        {
            "edge_id": "p410_orthogonal_to_grammar_universe",
            "source_node_id": "p410_orthogonal_dynamic_measurement",
            "target_node_id": "p410_grammar_finite_universe",
            "edge_type": "separates_semantic_format_boundary_stop_and_numeric_axes",
            "phase_id": "Phase410",
            "causal_path": False,
        },
        {
            "edge_id": "p410_machine_checks_to_external_review",
            "source_node_id": "p410_grammar_finite_universe",
            "target_node_id": "p410_external_review_gate",
            "edge_type": "machine_checks_do_not_replace_two_external_reviewers",
            "phase_id": "Phase410",
            "causal_path": False,
        },
        {
            "edge_id": "p410_external_review_to_execution_boundary",
            "source_node_id": "p410_external_review_gate",
            "target_node_id": "p410_model_execution_boundary",
            "edge_type": "keeps_cuda_physical_causal_and_neuron_execution_closed",
            "phase_id": "Phase410",
            "causal_path": False,
        },
    ]
    return nodes, edges


def update_metrics(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    denominator = stage["denominators"]
    results = stage["results"]
    manifest.setdefault("metrics", {}).update(
        {
            "phase410_orthogonal_axis_count": denominator[
                "orthogonal_axis_count"
            ],
            "phase410_h3_order_variant_count": denominator[
                "h3_order_variant_count"
            ],
            "phase410_h3_order_symmetry_failure_count": results[
                "h3_order_symmetry_failure_count"
            ],
            "phase410_grammar_finite_case_count": denominator[
                "grammar_finite_response_case_count"
            ],
            "phase410_grammar_failure_count": results[
                "grammar_finite_universe_failure_count"
            ],
            "phase410_required_external_reviewer_count": denominator[
                "required_independent_reviewer_count"
            ],
            "phase410_completed_external_reviewer_count": results[
                "completed_external_reviewer_count"
            ],
            "phase410_synthetic_collector_path_count": denominator[
                "synthetic_collector_path_count"
            ],
            "phase410_sealed_model_collector_case_count": results[
                "sealed_model_collector_equivalence_case_count"
            ],
            "phase410_model_case_count": denominator["model_case_count_consumed"],
            "phase410_physical_case_count": denominator[
                "physical_case_count_consumed"
            ],
            "phase410_new_neuron_node_count": results[
                "new_neuron_path_count"
            ],
        }
    )


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    denominator = stage["denominators"]
    results = stage["results"]
    progress["last_phase"] = "Phase410-OrthogonalDynamicPreflightStage"
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["orthogonal_dynamic_preflight_stage"] = {
        "orthogonal_state_axes": {
            "numerator": denominator["orthogonal_axis_count"],
            "denominator": denominator["orthogonal_axis_count"],
        },
        "h3_order_variants_machine_audited": {
            "numerator": denominator["h3_order_variant_count"]
            - results["h3_order_symmetry_failure_count"],
            "denominator": denominator["h3_order_variant_count"],
        },
        "finite_grammar_cases_machine_audited": {
            "numerator": denominator["grammar_finite_response_case_count"]
            - results["grammar_finite_universe_failure_count"],
            "denominator": denominator["grammar_finite_response_case_count"],
        },
        "independent_external_reviewers": {
            "numerator": results["completed_external_reviewer_count"],
            "denominator": denominator["required_independent_reviewer_count"],
        },
        "sealed_model_collector_equivalence": {
            "numerator": results["sealed_model_collector_equivalence_case_count"],
            "denominator": denominator[
                "future_sealed_model_qualification_case_count"
            ],
        },
        "model_cases_consumed": {"numerator": 0, "denominator": 165},
        "physical_cases_consumed": {"numerator": 0, "denominator": 1},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase410_decision"] = (
        "publish_machine_preflight_only_and_keep_cuda_physical_causal_and_neuron_"
        "gates_closed_until_two_external_reviews_and_sealed_model_collector_equivalence"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase410 preflight artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    stage = payloads["phase410_preflight_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        "phase410_evidence_nodes.jsonl",
        "phase410_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase410_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase410_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = "Phase410-OrthogonalDynamicPreflightStage"
        manifest["phase410"] = {
            "status": "machine_preflight_pass_external_and_model_gates_closed",
            "files": published,
            "machine_preflight_pass": stage["assessment"][
                "machine_preflight_pass"
            ],
            "model_case_count": 0,
            "physical_case_count": 0,
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = "Phase410-OrthogonalDynamicPreflightStage"
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase410_preflight_stage_summary.json", stage)
        write_jsonl(root / "phase410_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase410_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 410
        manifest["generated_at"] = updated_at
        manifest["phase410_audit"] = {
            "status": "protocol_preflight_only_no_model_physical_or_neuron_evidence",
            "orthogonal_axis_count": stage["denominators"][
                "orthogonal_axis_count"
            ],
            "h3_order_variant_count": stage["denominators"][
                "h3_order_variant_count"
            ],
            "grammar_finite_case_count": stage["denominators"][
                "grammar_finite_response_case_count"
            ],
            "completed_external_reviewer_count": stage["results"][
                "completed_external_reviewer_count"
            ],
            "sealed_model_collector_equivalence_case_count": stage["results"][
                "sealed_model_collector_equivalence_case_count"
            ],
            "model_case_count": 0,
            "physical_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase410_preflight_stage_summary.json",
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
