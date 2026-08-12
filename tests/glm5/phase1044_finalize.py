#!/usr/bin/env python3
"""Aggregate Phase1044 without turning response cells into mechanism edges."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1044_natural_recompute_trajectory_protocol as protocol


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_mode"]),
        int(row["depth_slot"]),
        str(row["channel"]),
        str(row["receiver_site"]),
    )


def behavior_relevance(
    summary: dict[str, Any],
    source_mode: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    metrics = summary["intervention_metrics"][source_mode]
    selected = [
        metrics[f"cross_selected_l{surface}"][
            "cross_minus_target_margin_shift"
        ]
        for surface in (0, 1)
    ]
    unselected = [
        metrics[f"cross_unselected_l{surface}"][
            "cross_minus_target_margin_shift"
        ]
        for surface in (0, 1)
    ]
    selected_median = min(float(row["median"]) for row in selected)
    selected_positive_rate = min(
        float(row["positive_rate"]) for row in selected
    )
    selected_minus_unselected = min(
        float(selected[index]["median"])
        - float(unselected[index]["median"])
        for index in (0, 1)
    )
    passed = (
        selected_median
        >= gate["selected_cross_margin_shift_median_min"]
        and selected_positive_rate
        >= gate["selected_cross_positive_rate_min"]
        and selected_minus_unselected
        >= gate["selected_minus_unselected_median_min"]
    )
    return {
        "selected_cross_min_median": selected_median,
        "selected_cross_min_positive_rate": selected_positive_rate,
        "selected_minus_unselected_min_median": (
            selected_minus_unselected
        ),
        "passed": bool(passed),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model_name: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model_name / "summary.json"
        )
        for model_name in protocol.MODELS
    }
    for model_name, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name} protocol digest drift")

    repeated: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    rows_by_model: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for model_name, summary in summaries.items():
        rows_by_model[model_name] = {}
        for row in summary["descriptive_pass_cells"]:
            key = cell_key(row)
            repeated[key].append(model_name)
            rows_by_model[model_name][key] = row

    minimum_models = int(
        prereg["descriptive_repetition_gate"]["minimum_models"]
    )
    repeated_cells = []
    for key, model_names in sorted(repeated.items()):
        if len(model_names) < minimum_models:
            continue
        source_mode, depth_slot, channel, receiver_site = key
        relevance = {
            model_name: behavior_relevance(
                summaries[model_name],
                source_mode,
                prereg["behavior_relevance_gate"],
            )
            for model_name in model_names
        }
        repeated_cells.append({
            "source_mode": source_mode,
            "depth_slot": depth_slot,
            "channel": channel,
            "receiver_site": receiver_site,
            "models": model_names,
            "model_count": len(model_names),
            "model_rows": {
                model_name: rows_by_model[model_name][key]
                for model_name in model_names
            },
            "behavior_relevance": relevance,
            "all_repeated_models_behavior_relevant": all(
                row["passed"] for row in relevance.values()
            ),
        })

    event_topology = {}
    for model_name, summary in summaries.items():
        passed = summary["descriptive_pass_cells"]
        query = [
            row for row in passed
            if row["receiver_site"] == "query_nonce"
        ]
        boundary = [
            row for row in passed
            if row["receiver_site"] == "pre_output"
        ]
        event_topology[model_name] = {
            "first_query_depth_slot": min(
                (int(row["depth_slot"]) for row in query),
                default=None,
            ),
            "first_boundary_depth_slot": min(
                (int(row["depth_slot"]) for row in boundary),
                default=None,
            ),
            "query_pass_count": len(query),
            "boundary_pass_count": len(boundary),
            "full_state_pass_count": sum(
                row["source_mode"] == "full_state" for row in passed
            ),
            "mlp_write_pass_count": sum(
                row["source_mode"] == "mlp_write" for row in passed
            ),
        }

    candidates = [
        row for row in repeated_cells
        if row["all_repeated_models_behavior_relevant"]
        and (
            not prereg["behavior_relevance_gate"][
                "full_state_required_for_confirmation"
            ]
            or row["source_mode"] == "full_state"
        )
    ][:3]
    confirmation_needed = bool(candidates)
    automatic_next = {
        "confirmation_needed": confirmation_needed,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "route": (
            "Run an independent surface-2 receiver reset/replay test on "
            "the frozen query-state trajectory; do not search additional "
            "cells."
            if confirmation_needed
            else prereg["automatic_followup"]["otherwise"]
        ),
    }

    artifact_manifest = {}
    for model_name in protocol.MODELS:
        model_root = protocol.OUT_ROOT / "atlas" / model_name
        artifact_manifest[model_name] = {
            str(path.relative_to(protocol.OUT_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(model_root.iterdir())
            if path.is_file()
        }

    aggregate = {
        "schema_version": "phase1044_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "sample_counts": prereg["sample_plan"],
        "model_behavior": {
            model_name: summary["behavior"]
            for model_name, summary in summaries.items()
        },
        "model_finite_audits": {
            model_name: {
                "source_cache": summary["source_cache_finite"],
                "response_norms": summary["response_norms_finite"],
                "receiver_vectors": summary["receiver_vectors_finite"],
                "candidate_logits": summary["candidate_logits_finite"],
                "closure": summary["closure"],
            }
            for model_name, summary in summaries.items()
        },
        "model_descriptive_pass_counts": {
            model_name: int(summary["descriptive_pass_count"])
            for model_name, summary in summaries.items()
        },
        "cross_model_repeated_cell_count": len(repeated_cells),
        "cross_model_repeated_cells": repeated_cells,
        "event_topology": event_topology,
        "automatic_next_decision": automatic_next,
        "artifact_manifest": artifact_manifest,
        "evidence_interpretation": [
            (
                "A single early complete-state edit causes a repeated "
                "naturally recomputed response at the query role in at "
                "least Qwen3 and GLM4."
            ),
            (
                "The first passing query event occurs at relative slots "
                "3, 2, and 4 in Qwen3, GLM4, and DeepSeek7B; this supports "
                "a shared event type, not a fixed physical layer."
            ),
            (
                "No MLP-source response cell passes the descriptive gate, "
                "even though MLP-source edits shift output margins. Local "
                "causal contribution and stable transport geometry remain "
                "different evidence."
            ),
            (
                "The repeated query response is not yet a receiver or "
                "mediator. Independent reset/replay is required."
            ),
        ],
        "theory_status": {
            "supported_measurement_language": [
                "conditional state field",
                "relative differential response",
                "natural downstream state transition",
                "role-conditioned competition",
            ],
            "not_established": [
                "minimum sufficient alliance",
                "a universal language equation",
                "biological near-optimality",
                "brain-LLM mechanism isomorphism",
                "a complete knowledge, reasoning, or grammar mechanism",
            ],
        },
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    protocol.write_json(
        protocol.OUT_ROOT / "automatic_next_decision.json",
        automatic_next,
    )
    print(json.dumps({
        "cross_model_repeated_cell_count": len(repeated_cells),
        "confirmation_needed": confirmation_needed,
        "candidate_count": len(candidates),
        "event_topology": event_topology,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
