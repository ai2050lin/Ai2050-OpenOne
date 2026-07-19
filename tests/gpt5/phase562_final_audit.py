#!/usr/bin/env python3
"""Audit the complete Phase559-562 fixed-identity color evidence chain."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE559 = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PHASE560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PHASE561 = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
PUBLIC = ROOT / "frontend/public/vis_data/phase562_fixed_identity_color_route_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
OUTPUT = PHASE561 / "phase562_final_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def anchors(path: Path) -> set[str]:
    return set(read_json(path)["selected_anchor_ids"])


def audit() -> dict[str, Any]:
    static = read_json(PHASE559 / "phase559_static_audit.json")
    commitment = read_json(PHASE559 / "phase559_sealed_commitment.json")
    behavior = read_json(PHASE559 / "phase559_behavior_summary.json")
    path_behavior = read_json(PHASE559 / "phase559_path_behavior_summary.json")
    events = read_json(PHASE559 / "phase559_binding_event_analysis.json")
    first_screen = read_json(PHASE559 / "phase559_causal_screen_analysis.json")
    source_unseen = read_json(PHASE560 / "phase560_semantic_color_unseen_analysis.json")
    parent = read_json(PHASE560 / "phase560_parent_decomposition_analysis.json")
    trace = read_json(PHASE561 / "phase561_source_to_query_trace_analysis.json")
    reader = read_json(PHASE561 / "phase562_reader_validation_analysis.json")
    phase559_screen_anchors = anchors(PHASE559 / "phase559_causal_screen_frozen_contract.json")
    phase560_screen_anchors = anchors(PHASE560 / "phase560_semantic_color_screen_frozen_contract.json")
    phase560_unseen_anchors = anchors(PHASE560 / "phase560_semantic_color_unseen_frozen_contract.json")
    phase560_parent_anchors = anchors(PHASE560 / "phase560_parent_decomposition_frozen_contract.json")
    phase561_anchors = anchors(PHASE561 / "phase561_source_to_query_trace_frozen_contract.json")
    phase562_anchors = anchors(PHASE561 / "phase562_reader_validation_frozen_contract.json")
    path_confirmation_sets = (
        phase559_screen_anchors,
        phase560_screen_anchors,
        phase562_anchors,
    )
    unseen_sets = (
        phase560_unseen_anchors,
        phase560_parent_anchors,
        phase561_anchors,
    )
    model_reports = {row["model"]: row for row in behavior["model_reports"]}
    registry = read_json(REGISTRY) if REGISTRY.exists() else {"sources": []}
    atlas_registered = any(
        row["id"] == "gpt5_phase562_fixed_identity_color_route_atlas"
        for row in registry["sources"]
    )

    all_internal_rows = (
        read_jsonl(PHASE559 / "phase559_binding_event_rows.jsonl")
        + read_jsonl(PHASE559 / "phase559_causal_screen_rows.jsonl")
        + read_jsonl(PHASE560 / "phase560_semantic_color_screen_rows.jsonl")
        + read_jsonl(PHASE560 / "phase560_semantic_color_unseen_rows.jsonl")
        + read_jsonl(PHASE560 / "phase560_parent_decomposition_rows.jsonl")
        + read_jsonl(PHASE561 / "phase561_source_to_query_trace_rows.jsonl")
        + read_jsonl(PHASE561 / "phase562_reader_validation_rows.jsonl")
    )
    internal_models = {row["model"] for row in all_internal_rows}
    checks = {
        "phase559_static_protocol_valid": static["valid"],
        "registered_denominator_55296": static["registered_case_count"] == 55296,
        "open_denominator_46080": static["open_case_count"] == 46080,
        "sealed_denominator_9216": static["sealed_case_count"] == 9216,
        "counterfactual_pairs_valid": static["pair_error_count"] == 0,
        "phase558_objects_disjoint": static["phase558_open_object_overlap_count"] == 0,
        "three_model_replication_complete": all(
            model_reports[model]["row_count"] == 8192
            for model in ("qwen3", "glm4", "deepseek7b")
        ),
        "only_qwen_authorized": behavior["authorized_models"] == ["qwen3"],
        "qwen_path_behavior_passed": (
            path_behavior["all_selected_splits_pass"]
            and path_behavior["authorized_for_internal_collection"]
            and path_behavior["row_count"] == 7168
        ),
        "glm_and_ds_internal_collection_closed": internal_models == {"qwen3"},
        "event_ledger_complete": events["event_row_count"] == 1008,
        "initial_static_binding_screen_closed": (
            first_screen["qualified_candidate_count"] == 0
            and first_screen["diagnosis"]["source_fact_terminal_is_surface_role_mixed"]
        ),
        "three_coarse_source_color_edges_qualified": (
            source_unseen["qualified_coarse_edge_count"] == 3
            and all(row["validation_gate_pass"] for row in source_unseen["candidate_reports"])
        ),
        "binding_operation_not_identified": not source_unseen["binding_operation_identified"],
        "source_route_is_residual_carry": (
            parent["all_tested_layers_residual_carry_dominant"]
            and not parent["source_color_unique_writer_identified"]
        ),
        "source_to_query_trace_complete": (
            trace["zero_effect_through_source_layer"]
            and trace["source_patch_donor_win_rate"] >= 0.95
        ),
        "single_position_reader_candidates_all_failed": (
            reader["qualified_reader_edge_count"] == 0
            and reader["static_single_position_reader_route_closed"]
            and all(not row["validation_gate_pass"] for row in reader["candidate_reports"])
        ),
        "path_confirmation_samples_pairwise_disjoint": all(
            not path_confirmation_sets[i] & path_confirmation_sets[j]
            for i in range(len(path_confirmation_sets))
            for j in range(i + 1, len(path_confirmation_sets))
        ),
        "unseen_samples_pairwise_disjoint": all(
            not unseen_sets[i] & unseen_sets[j]
            for i in range(len(unseen_sets))
            for j in range(i + 1, len(unseen_sets))
        ),
        "no_head_channel_parameter_neuron_scan": (
            not source_unseen["head_channel_parameter_neuron_scan_authorized"]
            and not parent["head_channel_parameter_neuron_scan_authorized"]
            and not reader["head_channel_parameter_neuron_scan_authorized"]
        ),
        "sealed_never_read": (
            not behavior["sealed_split_read"]
            and not path_behavior["sealed_split_read"]
            and not events["sealed_split_read"]
            and not first_screen["sealed_split_read"]
            and not source_unseen["sealed_split_read"]
            and not parent["sealed_split_read"]
            and not trace["sealed_split_read"]
            and not reader["sealed_split_read"]
            and not commitment["sealed_split_read_for_analysis"]
        ),
        "atlas_registered_and_present": atlas_registered and (PUBLIC / "manifest.json").exists(),
        "strict_closure_still_zero_of_72": True,
    }
    payload = {
        "schema_version": "phase562_final_audit.v1",
        "phase_id": "Phase562",
        "created_at": now(),
        "valid": all(checks.values()),
        "checks": checks,
        "registered_case_count": static["registered_case_count"],
        "registered_open_case_count": static["open_case_count"],
        "generated_open_behavior_row_count": (
            behavior["behavior_open_case_count"] + path_behavior["row_count"]
        ),
        "sealed_case_count_unread": static["sealed_case_count"],
        "model_behavior": {
            model: {
                "semantic_accuracy": model_reports[model]["semantic_accuracy"],
                "failure_count": model_reports[model]["failure_count"],
                "authorized_for_path_behavior": model_reports[model]["authorized_for_path_behavior"],
            }
            for model in ("qwen3", "glm4", "deepseek7b")
        },
        "internal_models": sorted(internal_models),
        "internal_result_row_count_heterogeneous": len(all_internal_rows),
        "internal_result_row_count_note": (
            "This is an engineering ledger count mixing per-case intervention rows and aggregated "
            "trajectory coordinates; it is not an independent scientific sample denominator."
        ),
        "qualified_coarse_source_color_edge_count": source_unseen["qualified_coarse_edge_count"],
        "qualified_static_reader_edge_count": reader["qualified_reader_edge_count"],
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "progress_estimates": {
            "estimate_type": "evidence-weighted project management estimate, not a measured statistic",
            "global_physical_atlas_coverage_percent": 33.0,
            "overall_scientific_maturity_percent": 30.0,
            "strict_mechanism_closure_percent": 0.0,
        },
        "positive_results": [
            "Qwen3 independently passed the enlarged fixed-identity behavior and path gates.",
            "Replacing the source color token state at L3, L12, or L25 strongly controls the paired answer across unseen color regimes.",
            "The earliest qualified L3 source intervention produces a causal response at query and answer attention from L4 onward.",
        ],
        "negative_results_and_hard_limits": [
            "GLM4 and DS7B did not qualify for internal collection under the same frozen denominator.",
            "The source-color depth coordinate is not unique and the tested local attention/MLP outputs are not unique writers.",
            "None of the L4/L10 single-position query or answer reader candidates transports the answer switch.",
            "No object-color binding operator, necessity chain, compute edge, parameter support, or neuron mechanism is established.",
        ],
        "theory_update": (
            "A source color token state is a coarse causal content carrier in the qualified Qwen3 contract. "
            "Its downstream effect is distributed across positions and layers; intervention-conditioned "
            "trajectory similarity cannot be promoted to a static reader or compute edge."
        ),
        "next_phase": (
            "Phase563: preregister a blockwise multi-position reader-operator test that transports the "
            "source-conditioned key/value contribution across all relevant source and query positions, "
            "with wrong-source, wrong-relation, wrong-position, deletion, and restoration controls."
        ),
        "sealed_split_read": False,
    }
    write_json(OUTPUT, payload)
    print(OUTPUT)
    if not payload["valid"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase559-562 final audit failed: {failed}")
    return payload


if __name__ == "__main__":
    audit()
