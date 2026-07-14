#!/usr/bin/env python3
"""Publish Phase414 observer/event contracts without physical promotion."""

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
P414 = ROOT / "tests/gpt5/result/phase414_observer_event_preflight"
JSON_SOURCES = {
    name: P414 / name
    for name in (
        "phase414_supplied_claim_audit.json",
        "phase414_natural_replay_identity_audit.json",
        "phase414_trajectory_ontology.json",
        "phase414_observer_readability_audit.json",
        "phase414_variable_length_event_contract.json",
        "phase414_cross_tokenizer_semantic_alignment.json",
        "phase414_observer_qualification_contract.json",
        "phase414_catalog_qualification.json",
        "phase414_execution_qualification.json",
        "phase414_stage_summary.json",
    )
}
LAST_PHASE = "Phase414-ObserverIndexedEventPreflightStage"


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
    den = stage["denominators"]
    result = stage["results"]
    common = {
        "phase_id": "Phase414",
        "protocol_only": True,
        "behavior_observed": False,
        "physical": False,
        "causal": False,
        "language_path": False,
    }
    nodes = [
        {
            "node_id": "p414_supplied_claim_and_catalog_audit",
            "node_type": "mixed_evidence_catalog_boundary_audit",
            "supplied_claim_count": den["supplied_claim_count"],
            "catalog_item_count": den["catalog_item_count"],
            "catalog_category_count": den["catalog_category_count"],
            "strict_model_mechanism_closed_catalog_item_count": result[
                "strict_model_mechanism_closed_catalog_item_count"
            ],
            "global_progress_catalog_item_count": result[
                "global_progress_catalog_item_count"
            ],
            **common,
        },
        {
            "node_id": "p414_natural_complete_state_replay_identity",
            "node_type": "finite_complete_state_terminal_replay_identity",
            "replay_cell_count": den["natural_replay_cell_count"],
            "exact_replay_count": result["complete_natural_replay_exact_count"],
            "layerwise_terminal_kernel_variation_case_count": result[
                "case_with_layerwise_terminal_kernel_variation_count"
            ],
            "incomplete_state_counterexample_count": result[
                "incomplete_local_state_replay_failure_count"
            ],
            **common,
        },
        {
            "node_id": "p414_typed_trajectory_ontology",
            "node_type": "five_way_measurement_object_separation",
            "trajectory_object_count": den["trajectory_object_count"],
            "generic_intermediate_candidate_trajectory_count": 0,
            **common,
        },
        {
            "node_id": "p414_observer_indexed_readability",
            "node_type": "synthetic_observer_indexed_layer_readability",
            "observer_cell_count": den["observer_readability_cell_count"],
            "varying_trajectory_count": result[
                "varying_case_observer_trajectory_count"
            ],
            "observer_disagreement_cell_count": result[
                "same_state_observer_disagreement_cell_count"
            ],
            "native_intermediate_probability_count": 0,
            **common,
        },
        {
            "node_id": "p414_variable_length_event_panel",
            "node_type": "prefix_free_eos_closed_variable_length_event_contract",
            "candidate_event_count": den["candidate_event_count"],
            "panel_mass": result["candidate_panel_mass"],
            "outside_mass": result["candidate_outside_mass"],
            "invalid_prefix_panel_rejected_count": result[
                "invalid_prefix_panel_rejected_count"
            ],
            **common,
        },
        {
            "node_id": "p414_cross_tokenizer_semantic_event_alignment",
            "node_type": "semantic_event_not_token_id_alignment_contract",
            "semantic_event_count": den["cross_tokenizer_semantic_event_count"],
            "semantic_alignment_count": result[
                "cross_tokenizer_semantic_event_alignment_count"
            ],
            "identical_token_id_sequence_count": result[
                "cross_tokenizer_identical_token_id_sequence_count"
            ],
            **common,
        },
        {
            "node_id": "p414_observer_qualification_boundary",
            "node_type": "diagnostic_and_learned_observer_qualification_gate",
            "observer_method_count": den["observer_method_count"],
            "qualified_observer_count": result["qualified_observer_count"],
            **common,
        },
        {
            "node_id": "p414_execution_boundary",
            "node_type": "external_review_collector_observer_execution_boundary",
            "completed_external_reviewer_count": result[
                "completed_external_reviewer_count"
            ],
            "required_external_reviewer_count": den[
                "required_independent_reviewer_count"
            ],
            "collector_equivalence_case_count": result[
                "sealed_model_collector_equivalence_case_count"
            ],
            "collector_equivalence_case_denominator": den[
                "future_sealed_model_collector_case_count"
            ],
            "model_execution_authorized": False,
            **common,
        },
    ]
    links = [
        (
            "p413_execution_boundary",
            "p414_supplied_claim_and_catalog_audit",
            "reaudits_claims_without_reopening_closed_execution_gates",
        ),
        (
            "p414_supplied_claim_and_catalog_audit",
            "p414_natural_complete_state_replay_identity",
            "reclassifies_complete_state_replay_as_instrument_identity",
        ),
        (
            "p414_natural_complete_state_replay_identity",
            "p414_typed_trajectory_ontology",
            "separates_identity_probability_readability_physical_and_causal_objects",
        ),
        (
            "p414_typed_trajectory_ontology",
            "p414_observer_indexed_readability",
            "requires_explicit_observer_index_for_layer_readability",
        ),
        (
            "p414_typed_trajectory_ontology",
            "p414_variable_length_event_panel",
            "generalizes_candidate_events_under_disjointness",
        ),
        (
            "p414_variable_length_event_panel",
            "p414_cross_tokenizer_semantic_event_alignment",
            "aligns_events_semantically_across_tokenizers",
        ),
        (
            "p414_observer_indexed_readability",
            "p414_observer_qualification_boundary",
            "keeps_synthetic_and_diagnostic_observers_unqualified",
        ),
        (
            "p414_observer_qualification_boundary",
            "p414_execution_boundary",
            "keeps_model_physical_causal_and_neuron_execution_closed",
        ),
    ]
    edges = [
        {
            "edge_id": f"p414_edge_{index:02d}",
            "source_node_id": source,
            "target_node_id": target,
            "edge_type": edge_type,
            "phase_id": "Phase414",
            "causal_path": False,
        }
        for index, (source, target, edge_type) in enumerate(links, start=1)
    ]
    return nodes, edges


def update_metrics(manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    den = stage["denominators"]
    result = stage["results"]
    manifest.setdefault("metrics", {}).update(
        {
            "phase414_supplied_claim_count": den["supplied_claim_count"],
            "phase414_catalog_item_count": den["catalog_item_count"],
            "phase414_catalog_category_count": den["catalog_category_count"],
            "phase414_catalog_mechanism_closed_count": result[
                "strict_model_mechanism_closed_catalog_item_count"
            ],
            "phase414_catalog_global_progress_unit_count": result[
                "global_progress_catalog_item_count"
            ],
            "phase414_trajectory_object_count": den["trajectory_object_count"],
            "phase414_natural_replay_case_count": den[
                "natural_replay_case_count"
            ],
            "phase414_natural_replay_cell_count": den[
                "natural_replay_cell_count"
            ],
            "phase414_natural_replay_exact_count": result[
                "complete_natural_replay_exact_count"
            ],
            "phase414_layerwise_terminal_kernel_variation_count": result[
                "case_with_layerwise_terminal_kernel_variation_count"
            ],
            "phase414_incomplete_state_counterexample_count": result[
                "incomplete_local_state_replay_failure_count"
            ],
            "phase414_observer_readability_cell_count": den[
                "observer_readability_cell_count"
            ],
            "phase414_observer_trajectory_count": den[
                "case_observer_trajectory_count"
            ],
            "phase414_varying_observer_trajectory_count": result[
                "varying_case_observer_trajectory_count"
            ],
            "phase414_observer_disagreement_cell_count": result[
                "same_state_observer_disagreement_cell_count"
            ],
            "phase414_candidate_event_count": den["candidate_event_count"],
            "phase414_candidate_panel_mass": result["candidate_panel_mass"],
            "phase414_candidate_outside_mass": result["candidate_outside_mass"],
            "phase414_invalid_prefix_panel_rejected_count": result[
                "invalid_prefix_panel_rejected_count"
            ],
            "phase414_cross_tokenizer_semantic_event_count": den[
                "cross_tokenizer_semantic_event_count"
            ],
            "phase414_cross_tokenizer_semantic_alignment_count": result[
                "cross_tokenizer_semantic_event_alignment_count"
            ],
            "phase414_qualified_observer_count": result[
                "qualified_observer_count"
            ],
            "phase414_observer_method_count": den["observer_method_count"],
            "phase414_completed_external_reviewer_count": result[
                "completed_external_reviewer_count"
            ],
            "phase414_required_external_reviewer_count": den[
                "required_independent_reviewer_count"
            ],
            "phase414_sealed_model_collector_case_count": result[
                "sealed_model_collector_equivalence_case_count"
            ],
            "phase414_model_case_count": den["model_case_count_consumed"],
            "phase414_physical_case_count": den["physical_case_count_consumed"],
            "phase414_new_neuron_node_count": result["new_neuron_path_count"],
        }
    )


def update_progress(root: Path, updated_at: str, stage: dict[str, Any]) -> None:
    path = root / "progress.json"
    if not path.is_file():
        return
    progress = read_json(path)
    den = stage["denominators"]
    result = stage["results"]
    progress["last_phase"] = LAST_PHASE
    progress["updated_at"] = updated_at
    progress["single_global_progress_percentage_valid"] = False
    progress["observer_indexed_event_preflight_stage"] = {
        "supplied_claims_audited": {
            "numerator": den["supplied_claim_count"],
            "denominator": den["supplied_claim_count"],
        },
        "mixed_catalog_items_classified": {
            "numerator": den["catalog_item_count"],
            "denominator": den["catalog_item_count"],
        },
        "natural_complete_state_replay_identity": {
            "numerator": result["complete_natural_replay_exact_count"],
            "denominator": den["natural_replay_cell_count"],
        },
        "layerwise_terminal_kernel_variation": {
            "numerator": result[
                "case_with_layerwise_terminal_kernel_variation_count"
            ],
            "denominator": den["natural_replay_case_count"],
        },
        "incomplete_state_counterexamples": {
            "numerator": result["incomplete_local_state_replay_failure_count"],
            "denominator": den["natural_replay_cell_count"],
        },
        "observer_indexed_cells": {
            "numerator": den["observer_readability_cell_count"],
            "denominator": den["observer_readability_cell_count"],
        },
        "variable_length_event_panel": {"numerator": 1, "denominator": 1},
        "invalid_prefix_panel_rejected": {
            "numerator": result["invalid_prefix_panel_rejected_count"],
            "denominator": den["invalid_prefix_panel_count"],
        },
        "cross_tokenizer_semantic_alignment": {
            "numerator": result[
                "cross_tokenizer_semantic_event_alignment_count"
            ],
            "denominator": den["cross_tokenizer_semantic_event_count"],
        },
        "qualified_observers": {
            "numerator": result["qualified_observer_count"],
            "denominator": den["observer_method_count"],
        },
        "independent_external_reviewers": {
            "numerator": result["completed_external_reviewer_count"],
            "denominator": den["required_independent_reviewer_count"],
        },
        "sealed_model_collector_equivalence": {
            "numerator": result[
                "sealed_model_collector_equivalence_case_count"
            ],
            "denominator": den["future_sealed_model_collector_case_count"],
        },
        "model_cases_consumed": {"numerator": 0, "denominator": 165},
        "new_physical_paths": {"numerator": 0, "denominator": 72},
        "new_single_neuron_causal_paths": {"numerator": 0, "denominator": 72},
    }
    progress["phase414_decision"] = (
        "publish_complete_state_replay_identity_typed_trajectory_observer_index_"
        "and_variable_event_contracts_only;_keep_model_physical_causal_and_neuron_"
        "gates_closed"
    )
    write_json(path, progress)


def main() -> None:
    missing = [str(path) for path in JSON_SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase414 preflight artifacts: {missing}")
    payloads = {name: read_json(path) for name, path in JSON_SOURCES.items()}
    stage = payloads["phase414_stage_summary.json"]
    nodes, edges = evidence_graph(stage)
    updated_at = datetime.now(timezone.utc).isoformat()
    published = [
        *payloads,
        "phase414_evidence_nodes.jsonl",
        "phase414_evidence_edges.jsonl",
    ]

    for root in (TARGET, CLIENT):
        for name, payload in payloads.items():
            write_json(root / name, payload)
        write_jsonl(root / "phase414_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase414_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["updated_at"] = updated_at
        manifest["last_phase"] = LAST_PHASE
        manifest["phase414"] = {
            "status": "observer_event_contract_pass_external_and_model_gates_closed",
            "files": published,
            "machine_preflight_pass": stage["assessment"]["machine_preflight_pass"],
            "complete_natural_replay_exact_count": stage["results"][
                "complete_natural_replay_exact_count"
            ],
            "qualified_observer_count": stage["results"][
                "qualified_observer_count"
            ],
            "catalog_item_count": stage["denominators"]["catalog_item_count"],
            "catalog_global_progress_unit_count": 0,
            "model_case_count": 0,
            "physical_case_count": 0,
        }
        update_metrics(manifest, stage)
        write_json(root / "manifest.json", manifest)
        update_progress(root, updated_at, stage)
        index_path = root / "client_index.json"
        if index_path.is_file():
            index = read_json(index_path)
            index["latest_phase"] = LAST_PHASE
            index["latest_stage_files"] = published
            initial = index.setdefault("initial_files", [])
            for name in published:
                if name not in initial:
                    initial.append(name)
            write_json(index_path, index)
        public_manifest(root, updated_at)
        update_checksums(root)

    for root in (NEURON_TARGET, NEURON_CLIENT):
        write_json(root / "phase414_stage_summary.json", stage)
        write_jsonl(root / "phase414_evidence_nodes.jsonl", nodes)
        write_jsonl(root / "phase414_evidence_edges.jsonl", edges)
        manifest = read_json(root / "manifest.json")
        manifest["phase"] = 414
        manifest["generated_at"] = updated_at
        manifest["phase414_audit"] = {
            "status": "protocol_only_no_model_physical_causal_or_neuron_evidence",
            "natural_replay_cell_count": stage["denominators"][
                "natural_replay_cell_count"
            ],
            "complete_natural_replay_exact_count": stage["results"][
                "complete_natural_replay_exact_count"
            ],
            "qualified_observer_count": stage["results"][
                "qualified_observer_count"
            ],
            "model_case_count": 0,
            "physical_case_count": 0,
            "new_neuron_path_nodes_promoted": 0,
            "source": "phase414_stage_summary.json",
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
