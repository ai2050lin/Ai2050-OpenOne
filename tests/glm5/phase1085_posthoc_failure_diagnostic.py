#!/usr/bin/env python3
"""Offline diagnosis of Phase1085 failed purity gates.

This script uses only frozen aggregate response files.  It does not load a
model, select a neuron, or upgrade prospective evidence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1085_direct_entity_attribute_protocol as protocol

sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol
import phase1082_semantic_output_operation_world_finalize as analysis


analysis.protocol = protocol
analysis.base.protocol = protocol
analysis.base.DEPTH_GRID = np.linspace(
    protocol.TARGET_RELATIVE_DEPTH_MIN,
    protocol.TARGET_RELATIVE_DEPTH_MAX,
    7,
)


def selected_rows(
    rows: list[dict[str, Any]], role: str, component: str | None
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["role"] == role
        and (component is None or row["component"] == component)
    ]


def bank(
    rows: list[dict[str, Any]],
    worlds: tuple[str, ...],
    split: str,
    field: str,
    role: str,
) -> np.ndarray:
    values = []
    for operation in protocol.OPERATIONS:
        profiles = [
            analysis.base.build_profile(
                rows,
                f"{operation}__{world}",
                split,
                field,
                roles=(role,),
            )
            for world in worlds
        ]
        values.append(np.mean(np.stack(profiles), axis=0))
    return analysis.row_normalize(np.stack(values), centered=True)


def identity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    matrix = source @ target.T
    diagonal = np.diag(matrix)
    return {
        "top1": int(np.sum(np.argmax(matrix, axis=1) == np.arange(len(matrix)))),
        "identity_mean": float(np.mean(diagonal)),
    }


def control_ratio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in rows:
        if row["conditioning"] != "all_finite":
            continue
        content = row["mean_content_route_relative_magnitude"]
        output = row["mean_label_swap"]
        shell = row["mean_shell"]
        if content is None or output is None or shell is None:
            continue
        if float(content) <= 1e-12:
            continue
        value = max(float(output), float(shell)) / float(content)
        if math.isfinite(value):
            values.append(value)
    return {
        "median_max_control_to_content": (
            float(np.median(values)) if values else None
        ),
        "observation_count": len(values),
    }


def scope_diagnostic(
    rows: list[dict[str, Any]], role: str, component: str | None
) -> dict[str, Any]:
    scoped = selected_rows(rows, role, component)
    within = {}
    for field in ("content_route", "duplicate_route"):
        within[field] = identity(
            bank(scoped, tuple(protocol.WORLDS), "discovery", field, role),
            bank(scoped, tuple(protocol.WORLDS), "confirmation", field, role),
        )
    pairs = []
    for source_world in protocol.WORLDS:
        for target_world in protocol.WORLDS:
            if source_world == target_world:
                continue
            content = identity(
                bank(scoped, (source_world,), "discovery", "content_route", role),
                bank(scoped, (target_world,), "confirmation", "content_route", role),
            )
            duplicate = identity(
                bank(scoped, (source_world,), "discovery", "duplicate_route", role),
                bank(scoped, (target_world,), "confirmation", "duplicate_route", role),
            )
            pairs.append({
                "source_world": source_world,
                "target_world": target_world,
                "content_top1": content["top1"],
                "duplicate_top1": duplicate["top1"],
                "content_advantage": (
                    content["identity_mean"] - duplicate["identity_mean"]
                ),
            })
    advantages = [row["content_advantage"] for row in pairs]
    return {
        "role": role,
        "component": component or "all_components",
        "within_item_split": within,
        "cross_world": {
            "mean_content_advantage": float(np.mean(advantages)),
            "positive_pair_count": sum(value > 0 for value in advantages),
            "threshold_pair_count": sum(value >= 0.05 for value in advantages),
            "pairs": pairs,
        },
        "control_ratio": control_ratio(scoped),
    }


def main() -> None:
    analysis_root = protocol.OUT_ROOT / "analysis"
    decomposition = protocol.read_json(
        analysis_root / "operation_world_decomposition.json"
    )["by_model"]
    heldout = protocol.read_json(
        analysis_root / "heldout_world_audit.json"
    )["by_model"]
    by_model = {}
    for model in protocol.MODELS:
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        scopes = []
        for role in ("query_entity", "answer_boundary"):
            scopes.append(scope_diagnostic(rows, role, None))
            for component in ("residual", "attention_output", "mlp_output"):
                scopes.append(scope_diagnostic(rows, role, component))
        ranked = sorted(
            scopes,
            key=lambda row: row["cross_world"]["mean_content_advantage"],
            reverse=True,
        )
        by_model[model] = {
            "scopes": scopes,
            "best_posthoc_scope": ranked[0],
            "operation_world_decomposition": decomposition[model],
            "heldout_world": heldout[model],
        }
    result = {
        "schema_version": "phase1085_posthoc_failure_diagnostic.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_descriptive_no_new_model_calls",
        "by_model": by_model,
        "diagnosis": {
            "behavior": (
                "Direct entity naming repaired the Phase1084 behavior failure "
                "in all three models."
            ),
            "repeatability": (
                "The middle-answer-boundary content profile repeats across fresh "
                "items and output words, but not across held-out lexical worlds."
            ),
            "carrier": (
                "Matched duplicate/entity and shell-output controls remain much "
                "larger or more transferable than the content interaction."
            ),
            "structure": (
                "The measured field is strongly model- and world-conditioned; "
                "a fixed context-free attribute vector is not supported."
            ),
        },
        "limits": [
            "Every scope comparison is post-hoc and cannot upgrade P4, P6, P7, or P8.",
            "A better sub-scope is a future hypothesis, not evidence for a component mechanism.",
            "No model inference, intervention, head search, or neuron localization was run.",
        ],
    }
    result["diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(
        analysis_root / "posthoc_failure_diagnostic.json", result
    )
    print({
        "phase": protocol.PHASE,
        "best_posthoc_scopes": {
            model: {
                "role": row["best_posthoc_scope"]["role"],
                "component": row["best_posthoc_scope"]["component"],
                "mean_advantage": row["best_posthoc_scope"]["cross_world"][
                    "mean_content_advantage"
                ],
                "threshold_pairs": row["best_posthoc_scope"]["cross_world"][
                    "threshold_pair_count"
                ],
                "control_ratio": row["best_posthoc_scope"]["control_ratio"][
                    "median_max_control_to_content"
                ],
            }
            for model, row in by_model.items()
        },
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
