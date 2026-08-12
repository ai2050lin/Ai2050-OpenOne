#!/usr/bin/env python3
"""Diagnose where Phase1083 content loses to carrier-only structure."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1083_same_carrier_attribute_protocol as protocol

sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol

import phase1082_semantic_output_operation_world_finalize as analysis


def scope_rows(rows: list[dict[str, Any]], scope: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["role"] not in scope["roles"]:
            continue
        if scope.get("components") and row["component"] not in scope["components"]:
            continue
        low, high = scope.get("depth_range", (0.0, 1.0))
        depth = float(row["relative_depth"])
        if not (low <= depth <= high):
            continue
        output.append(row)
    return output


def profile(
    rows: list[dict[str, Any]], operation: str, worlds: tuple[str, ...],
    split: str, field: str, roles: tuple[str, ...],
) -> np.ndarray:
    values = [
        analysis.base.build_profile(
            rows, f"{operation}__{world}", split, field, roles=roles
        )
        for world in worlds
    ]
    return np.mean(np.stack(values), axis=0)


def bank(
    rows: list[dict[str, Any]], worlds: tuple[str, ...], split: str,
    field: str, roles: tuple[str, ...],
) -> np.ndarray:
    values = np.stack([
        profile(rows, operation, worlds, split, field, roles)
        for operation in protocol.OPERATIONS
    ])
    return analysis.row_normalize(values, centered=True)


def identity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    scores = source @ target.T
    diagonal = np.diag(scores)
    top1 = int(np.sum(np.argmax(scores, axis=1) == np.arange(len(scores))))
    margins = []
    for index in range(len(scores)):
        other = np.delete(scores[index], index)
        margins.append(float(scores[index, index] - np.max(other)))
    return {
        "top1": top1,
        "identity_mean": float(np.mean(diagonal)),
        "identity_margin_mean": float(np.mean(margins)),
    }


def control_ratio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = []
    for row in rows:
        if row["conditioning"] != "all_finite":
            continue
        content = row["mean_content_route_relative_magnitude"]
        output = row["mean_label_swap"]
        shell = row["mean_shell"]
        if content is None or output is None or shell is None:
            continue
        if float(content) <= analysis.EPSILON:
            continue
        ratio = max(float(output), float(shell)) / float(content)
        if math.isfinite(ratio):
            ratios.append(ratio)
    return {
        "median_max_control_to_content": (
            float(np.median(ratios)) if ratios else None
        ),
        "observation_count": len(ratios),
    }


def main() -> None:
    scopes = []
    for role in protocol.PRIMARY_PROFILE_ROLES:
        scopes.append({"name": role, "roles": (role,)})
        for component in ("residual", "attention_output", "mlp_output"):
            scopes.append({
                "name": f"{role}.{component}",
                "roles": (role,),
                "components": (component,),
            })
        for label, bounds in (
            ("early", (0.0, 0.33)),
            ("middle", (0.34, 0.66)),
            ("late", (0.67, 1.0)),
        ):
            scopes.append({
                "name": f"{role}.{label}",
                "roles": (role,),
                "depth_range": bounds,
            })

    by_model: dict[str, Any] = {}
    for model in protocol.MODELS:
        raw = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        model_scopes = {}
        for scope in scopes:
            rows = scope_rows(raw, scope)
            roles = tuple(scope["roles"])
            within = {}
            for field in ("content_route", "duplicate_route"):
                within[field] = identity(
                    bank(rows, tuple(protocol.WORLDS), "discovery", field, roles),
                    bank(rows, tuple(protocol.WORLDS), "confirmation", field, roles),
                )
            pairs = []
            for source_world in protocol.WORLDS:
                for target_world in protocol.WORLDS:
                    if source_world == target_world:
                        continue
                    content = identity(
                        bank(rows, (source_world,), "discovery", "content_route", roles),
                        bank(rows, (target_world,), "confirmation", "content_route", roles),
                    )
                    duplicate = identity(
                        bank(rows, (source_world,), "discovery", "duplicate_route", roles),
                        bank(rows, (target_world,), "confirmation", "duplicate_route", roles),
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
            model_scopes[scope["name"]] = {
                "scope": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in scope.items()
                },
                "within_item_split": within,
                "cross_world": {
                    "mean_content_advantage": float(np.mean(advantages)),
                    "positive_pair_count": sum(value > 0 for value in advantages),
                    "preregistered_pass_pair_count": sum(
                        value >= protocol.EVIDENCE_THRESHOLDS[
                            "minimum_cross_world_content_advantage"
                        ]
                        for value in advantages
                    ),
                    "pairs": pairs,
                },
                "control_ratio": control_ratio(rows),
            }
        ranked = sorted(
            model_scopes,
            key=lambda name: model_scopes[name]["cross_world"][
                "mean_content_advantage"
            ],
            reverse=True,
        )
        by_model[model] = {
            "scopes": model_scopes,
            "ranked_scope_names": ranked,
            "best_scope": ranked[0],
        }

    result = {
        "schema_version": "phase1083_posthoc_scope_diagnostic.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_descriptive_no_new_model_calls",
        "by_model": by_model,
        "interpretation": (
            "Role/component/depth scopes are post-hoc diagnostics. They may guide "
            "a future preregistration but cannot upgrade Phase1083 evidence or "
            "authorize component/neuron localization."
        ),
    }
    result["diagnostic_digest"] = protocol.digest(result)
    path = protocol.OUT_ROOT / "analysis" / "posthoc_scope_diagnostic.json"
    protocol.write_json(path, result)
    print({
        "phase": protocol.PHASE,
        "best_scopes": {
            model: row["best_scope"] for model, row in by_model.items()
        },
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
