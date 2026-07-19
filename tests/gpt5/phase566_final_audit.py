#!/usr/bin/env python3
"""Final integrity and evidence-boundary audit for Phase564-565."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P564 = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
P565 = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator"
PUBLIC = ROOT / "frontend/public/vis_data/phase565_fixed_identity_color_residual_atlas"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
AUDIT_PATH = P565 / "phase566_final_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit() -> dict[str, Any]:
    static = read_json(P564 / "phase564_static_audit.json")
    behavior = read_json(P564 / "phase564_behavior_summary.json")
    edge_behavior = read_json(P564 / "phase564_edge_behavior_summary.json")
    numeric = read_json(P564 / "phase564_numeric_calibration.json")
    discovery_contract = read_json(P564 / "phase564_source_edge_discovery_frozen_contract.json")
    discovery_execution = read_json(P564 / "phase564_source_edge_discovery_execution_summary.json")
    discovery_analysis = read_json(P564 / "phase564_source_edge_discovery_analysis.json")
    confirmation_contract = read_json(P564 / "phase564_source_edge_confirmation_frozen_contract.json")
    confirmation_execution = read_json(P564 / "phase564_source_edge_confirmation_execution_summary.json")
    confirmation_analysis = read_json(P564 / "phase564_source_edge_confirmation_analysis.json")
    residual_contract = read_json(P565 / "phase565_residual_operator_frozen_contract.json")
    residual_execution = read_json(P565 / "phase565_residual_operator_execution_summary.json")
    residual_analysis = read_json(P565 / "phase565_residual_operator_analysis.json")
    atlas_summary = read_json(P565 / "phase565_atlas_publish_summary.json")
    manifest = read_json(PUBLIC / "manifest.json")
    registry = read_json(REGISTRY_PATH)

    checks = {
        "static_protocol_valid": static["valid"],
        "registered_50688": static["registered_case_count"] == 50_688,
        "open_41472": static["open_case_count"] == 41_472,
        "sealed_9216": static["sealed_case_count"] == 9_216,
        "prior_open_overlap_zero": static["prior_open_object_overlap_count"] == 0,
        "behavior_rows_24576": behavior["behavior_open_case_count"] == 24_576,
        "qwen_only_behavior_authorized": behavior["authorized_models"] == ["qwen3"],
        "qwen_only_internal_authorized": edge_behavior["authorized_models"] == ["qwen3"],
        "numeric_calibration_before_rows": (
            numeric["intervention_rows_written_before_calibration"] == 0
            and not numeric["candidate_scores_read_before_calibration"]
            and not numeric["causal_success_gate_changed"]
        ),
        "discovery_rows_complete": (
            line_count(P564 / "phase564_source_edge_discovery_rows.jsonl")
            == discovery_contract["expected_intervention_rows"] == 86_016
            and discovery_execution["status"] == "complete"
        ),
        "discovery_reconstruction_valid": (
            discovery_execution["maximum_reconstruction_relative_error"]
            <= discovery_contract["reconstruction_relative_error_max"]
        ),
        "confirmation_rows_complete": (
            line_count(P564 / "phase564_source_edge_confirmation_rows.jsonl")
            == confirmation_contract["expected_intervention_rows"] == 57_344
            and confirmation_execution["status"] == "complete"
        ),
        "source_edges_zero_of_four": (
            confirmation_analysis["confirmation_passing_candidate_count"] == 0
            and confirmation_analysis["candidate_count"] == 4
        ),
        "unseen_edge_correctly_stopped": (
            not confirmation_analysis["unseen_required"]
            and not (P564 / "phase564_source_edge_unseen_rows.jsonl").exists()
        ),
        "residual_rows_complete": (
            line_count(P565 / "phase565_residual_operator_rows.jsonl")
            == residual_contract["expected_intervention_rows"] == 71_424
            and residual_execution["status"] == "complete"
        ),
        "residual_six_of_six_qualified": (
            residual_analysis["qualified_operator_count"] == 6
            and residual_analysis["candidate_count"] == 6
        ),
        "no_compute_edge_claim": (
            discovery_analysis["compute_edge_count"] == 0
            and confirmation_analysis["compute_edge_count"] == 0
            and residual_analysis["compute_edge_count"] == 0
            and atlas_summary["compute_edge_count"] == 0
        ),
        "no_fine_scan": (
            not residual_analysis["head_channel_parameter_neuron_scan_executed"]
            and atlas_summary["single_neuron_node_count"] == 0
        ),
        "sealed_unread": (
            not static["sealed_rows_read_for_analysis"]
            and not behavior["sealed_split_read"]
            and not edge_behavior["sealed_split_read"]
            and not confirmation_analysis["sealed_split_read"]
            and not residual_analysis["sealed_split_read"]
        ),
        "atlas_three_models": len(manifest["items"]) == 3,
        "atlas_source_registered": any(
            row["id"] == "gpt5_phase565_fixed_identity_color_residual_atlas"
            for row in registry["sources"]
        ),
        "closure_zero_of_72": atlas_summary["strict_closed_mechanisms"] == 0,
    }
    graph_checks = []
    for item in manifest["items"]:
        payload = read_json(PUBLIC / item["path"])
        graph = payload["graph"]
        node_ids = [row["id"] for row in graph["nodes"]]
        edge_ids = [row["id"] for row in graph["edges"]]
        node_set = set(node_ids)
        graph_checks.append({
            "model": item["model"],
            "node_count": len(node_ids),
            "edge_count": len(edge_ids),
            "unique_node_ids": len(node_ids) == len(node_set),
            "unique_edge_ids": len(edge_ids) == len(set(edge_ids)),
            "all_edge_references_valid": all(
                edge["source"] in node_set and edge["target"] in node_set
                for edge in graph["edges"]
            ),
        })
    checks["atlas_graph_integrity"] = all(
        row["unique_node_ids"] and row["unique_edge_ids"] and row["all_edge_references_valid"]
        for row in graph_checks
    )
    payload = {
        "schema_version": "phase566_final_audit.v1",
        "phase_id": "Phase566",
        "created_at": now(),
        "valid": all(checks.values()),
        "checks": checks,
        "graph_checks": graph_checks,
        "objective_counts": {
            "registered_cases": 50_688,
            "three_model_behavior_rows": 24_576,
            "qwen_edge_behavior_rows": 5_632,
            "source_edge_discovery_rows": 86_016,
            "source_edge_confirmation_rows": 57_344,
            "residual_operator_rows": 71_424,
            "actual_model_generated_behavior_rows": 30_208,
            "actual_internal_intervention_rows": 214_784,
            "qualified_source_compute_edges": 0,
            "qualified_distributed_residual_operators": 6,
            "strict_closed_mechanisms": 0,
            "closure_denominator": 72,
        },
        "evidence_boundary": {
            "source_post_softmax_value_edge_route": "rejected_for_frozen_L4_to_L10_query_answer_candidates",
            "distributed_residual_state_sufficiency": "supported_at_L4_L7_L10_for_semantic7_and_full_sequence",
            "natural_necessity": "not_tested",
            "key_and_attention_weight_mechanism": "not_identified",
            "precise_compute_edge": "not_identified",
            "cross_model_internal_replication": "absent",
            "parameter_or_neuron_closure": "absent",
        },
    }
    write_json(AUDIT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise RuntimeError("Phase566 final audit failed")
    return payload


if __name__ == "__main__":
    audit()
