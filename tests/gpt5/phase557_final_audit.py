#!/usr/bin/env python3
"""Build the strict final Phase557 evidence audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase557_fruit_composite"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def count_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    protocol = read_json(RESULT / "phase557_frozen_protocol.json")
    static = read_json(RESULT / "phase557_static_audit.json")
    behavior = read_json(RESULT / "phase557_behavior_summary.json")
    source = read_json(RESULT / "phase557_natural_color_source_analysis.json")
    unseen = read_json(RESULT / "phase557_natural_color_unseen_source_analysis.json")
    parent = read_json(RESULT / "phase557_natural_color_parent_analysis.json")
    upstream = read_json(RESULT / "phase557_natural_color_upstream_analysis.json")
    atlas = read_json(RESULT / "phase557_atlas_publish_summary.json")
    registry = read_json(REGISTRY)

    event_summaries = {
        model: read_json(
            RESULT / "natural_color_events" / model / "phase557_natural_color_event_summary.json"
        )
        for model in ("qwen3", "glm4")
    }
    source_paths = [
        RESULT / "natural_color_source" / model / suffix / "phase557_natural_color_source_rows.jsonl"
        if suffix else RESULT / "natural_color_source" / model / "phase557_natural_color_source_rows.jsonl"
        for model in ("qwen3", "glm4")
        for suffix in ("", "unseen_recombination")
    ]
    parent_paths = [
        RESULT / "natural_color_parent_blocks" / model / stage / "phase557_natural_color_parent_rows.jsonl"
        for model in ("qwen3", "glm4")
        for stage in ("parent_discovery", "parent_confirmation")
    ]
    upstream_paths = [
        RESULT / "natural_color_upstream_trace" / model / stage / "phase557_natural_color_upstream_rows.jsonl"
        for model in ("qwen3", "glm4")
        for stage in ("trace_discovery", "trace_confirmation")
    ]
    source_count = sum(count_rows(path) for path in source_paths)
    parent_count = sum(count_rows(path) for path in parent_paths)
    upstream_count = sum(count_rows(path) for path in upstream_paths)
    behavior_reports = {row["model"]: row for row in behavior["model_reports"]}
    registry_source = next(
        (row for row in registry["sources"] if row["id"] == "gpt5_phase557_fruit_composite_atlas"),
        None,
    )

    checks = {
        "static_protocol_valid": bool(static["valid"]),
        "registered_denominator_29184": protocol["registered_case_count"] == 29184,
        "open_denominator_24192": behavior["open_case_count"] == 24192,
        "sealed_denominator_4992": behavior["sealed_case_count_unread"] == 4992,
        "sealed_never_read": not any([
            behavior["sealed_split_read"], source["sealed_split_read"], unseen["sealed_split_read"],
            parent["sealed_split_read"], upstream["sealed_split_read"], atlas["sealed_split_read"],
        ]),
        "three_model_behavior_complete": all(
            behavior_reports[model]["open_case_count"] == 8064 for model in MODELS
        ),
        "contextual_gate_rejected_all_models": (
            behavior["models_authorized_for_contextual_internal_collection"] == []
        ),
        "natural_color_authorized_only_qwen_glm": behavior["natural_authorizations"] == {
            "qwen3": ["color"], "glm4": ["color"], "deepseek7b": []
        },
        "event_observers_bf16": all(
            row["torch_dtype"] == "torch.bfloat16" for row in event_summaries.values()
        ),
        "event_full_vectors_not_persisted": all(
            not row["full_vectors_persisted"] for row in event_summaries.values()
        ),
        "confirmation_coarse_edge_count_4": source["qualified_compute_edge_count"] == 4,
        "unseen_replicated_edge_count_3": unseen["qualified_compute_edge_count"] == 3,
        "replicated_parent_blocks_are_layer_input_only": (
            parent["replicated_parent_block_count"] == 3
            and parent["replicated_writer_parent_count"] == 0
            and parent["replicated_residual_carry_parent_count"] == 3
        ),
        "embedding_boundary_reached_two_models": set(
            upstream["embedding_boundary_reached_models"]
        ) == {"qwen3", "glm4"},
        "fine_scan_not_authorized": (
            not parent["fine_grained_parameter_scan_authorized"]
            and not upstream["fine_grained_parameter_scan_authorized"]
        ),
        "atlas_registered": registry_source is not None,
        "closure_still_zero_of_72": atlas["strict_closed_mechanisms"] == 0,
    }
    valid = all(checks.values())
    audit = {
        "schema_version": "phase557_final_audit.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "valid": valid,
        "checks": checks,
        "registered_case_count": 29184,
        "open_behavior_generation_count": 24192,
        "sealed_case_count_unread": 4992,
        "natural_color_observer_case_count": sum(
            row["case_count"] for row in event_summaries.values()
        ),
        "natural_color_event_row_count": sum(
            row["event_row_count"] for row in event_summaries.values()
        ),
        "source_recompute_intervention_row_count": source_count,
        "parent_block_intervention_row_count": parent_count,
        "upstream_trace_intervention_row_count": upstream_count,
        "total_causal_intervention_row_count": source_count + parent_count + upstream_count,
        "behavior_results": {
            model: {
                "semantic_accuracy": behavior_reports[model]["semantic_accuracy"],
                "strict_sequence_accuracy": behavior_reports[model]["strict_sequence_accuracy"],
                "contextual_authorized": behavior_reports[model][
                    "contextual_internal_collection_authorized"
                ],
                "authorized_natural_relations": behavior_reports[model][
                    "authorized_natural_relations"
                ],
                "behavior_discovery_world_all32_rate": behavior_reports[model][
                    "controlled_split_reports"
                ]["behavior_discovery"]["world_all_32_correct_rate"],
                "behavior_confirmation_world_all32_rate": behavior_reports[model][
                    "controlled_split_reports"
                ]["behavior_confirmation"]["world_all_32_correct_rate"],
            }
            for model in MODELS
        },
        "confirmed_coarse_object_source_edges": 4,
        "unseen_replicated_coarse_object_source_edges": 3,
        "replicated_edge_coordinates": [
            {"model": row["model"], "layer": row["layer"], "candidate_id": row["candidate_id"]}
            for row in read_jsonl(RESULT / "phase557_replicated_natural_color_compute_edges.jsonl")
        ],
        "replicated_writer_parent_count": 0,
        "replicated_layer_input_parent_count": 3,
        "lexical_input_boundary_models": upstream["embedding_boundary_reached_models"],
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "progress_estimates": {
            "strict_mechanism_closure_percent": 0.0,
            "global_physical_atlas_coverage_percent": 33.0,
            "overall_scientific_maturity_percent": 30.0,
            "estimate_type": "evidence-weighted project management estimate, not a measured statistic",
        },
        "positive_results": [
            "Natural color behavior qualified independently on Qwen3 and GLM4.",
            "Three complete object-token state edges replicated on unseen fruits: Qwen3 L2/L19 and GLM4 L22.",
            "Correct donor identity replacement controlled donor color while relation-position and channel-roll controls usually failed.",
            "All three replicated edges decomposed to layer-input residual carry and traced to L0.",
        ],
        "negative_results_and_hard_limits": [
            "No model passed the complete contextual object/category/attribute/binding behavior gate.",
            "DS7B passed neither contextual nor natural-color internal authorization.",
            "No attention or MLP writer parent replicated at the coarse candidate layers.",
            "L0 success imports complete lexical object identity and does not isolate color, category, or binding code.",
            "No head, channel, parameter, neuron, or sealed test is authorized by this result.",
        ],
        "theory_update": (
            "The dynamic-pattern-network hypothesis gains a concrete lexical-identity transport edge, "
            "but not a reusable fruit/color code. The next object must hold lexical identity fixed while "
            "changing only attribute content and binding in a color-only controlled contract."
        ),
        "next_phase": (
            "Phase558: preregister a color-only controlled counterfactual contract with fixed object tokens, "
            "balanced attribute/binding swaps, large independent splits, and conditional query integration tracing."
        ),
        "atlas_source_id": "gpt5_phase557_fruit_composite_atlas",
    }
    write_json(RESULT / "phase557_final_audit.json", audit)
    if not valid:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase557 final audit failed: {failed}")
    print(RESULT / "phase557_final_audit.json")


if __name__ == "__main__":
    main()
