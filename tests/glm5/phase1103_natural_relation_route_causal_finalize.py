#!/usr/bin/env python3
"""Freeze independent causal confirmation for Phase1103."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any

import phase1103_natural_relation_route_protocol as protocol


CONTROL_KINDS = (
    "ordinal_control",
    "wrong_pair_control",
    "equal_norm_random_control",
)


def metric(records: list[dict[str, Any]]) -> dict[str, Any]:
    recoveries = [
        float(row["recovery"])
        for row in records
        if row.get("behavior_valid") and row.get("recovery") is not None
        and math.isfinite(float(row["recovery"]))
    ]
    return {
        "count": len(records),
        "finite_fraction": sum(bool(row["finite"]) for row in records)
        / max(len(records), 1),
        "behavior_valid_fraction": sum(
            bool(row["behavior_valid"]) for row in records
        ) / max(len(records), 1),
        "recovery_count": len(recoveries),
        "mean_recovery": (
            statistics.fmean(recoveries) if recoveries else None
        ),
        "median_recovery": (
            statistics.median(recoveries) if recoveries else None
        ),
        "positive_recovery_fraction": (
            sum(value > 0.0 for value in recoveries) / len(recoveries)
            if recoveries else None
        ),
        "flip_rate": sum(bool(row["flip"]) for row in records)
        / max(len(records), 1),
    }


def metrics_at(
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    model: str,
    pair: str,
    surface: str,
    split: str,
    depth: int,
) -> dict[str, Any]:
    result = {}
    for kind in ("same_pair", *CONTROL_KINDS, "congruent_collateral"):
        result[kind] = metric(grouped.get((
            model, pair, surface, split, depth, kind,
        ), []))
    directions = {}
    same_rows = grouped.get((
        model, pair, surface, split, depth, "same_pair",
    ), [])
    for source_route, target_route in (
        ("exact", "paraphrase"),
        ("paraphrase", "exact"),
    ):
        key = f"{source_route}_to_{target_route}"
        directions[key] = metric([
            row for row in same_rows
            if row["source_route"] == source_route
            and row["target_route"] == target_route
        ])
    result["directions"] = directions
    same_median = result["same_pair"]["median_recovery"]
    control_medians = [
        result[kind]["median_recovery"] for kind in CONTROL_KINDS
        if result[kind]["median_recovery"] is not None
    ]
    result["specificity_advantage"] = (
        float(same_median) - max(float(value) for value in control_medians)
        if same_median is not None and control_medians else None
    )
    return result


def gates(metrics: dict[str, Any]) -> dict[str, bool]:
    thresholds = protocol.THRESHOLDS
    same = metrics["same_pair"]
    congruent = metrics["congruent_collateral"]
    direction_gates = {}
    for key, row in metrics["directions"].items():
        direction_gates[f"{key}_median"] = (
            row["median_recovery"] is not None
            and row["median_recovery"]
            >= thresholds["minimum_each_direction_median_recovery"]
        )
        direction_gates[f"{key}_positive"] = (
            row["positive_recovery_fraction"] is not None
            and row["positive_recovery_fraction"]
            >= thresholds["minimum_each_direction_positive_fraction"]
        )
    result = {
        "finite": same["finite_fraction"]
        >= thresholds["minimum_causal_finite_fraction"],
        "behavior_valid": same["behavior_valid_fraction"]
        >= thresholds["minimum_causal_behavior_valid_fraction"],
        "median_recovery": (
            same["median_recovery"] is not None
            and same["median_recovery"]
            >= thresholds["minimum_causal_median_recovery"]
        ),
        "positive_fraction": (
            same["positive_recovery_fraction"] is not None
            and same["positive_recovery_fraction"]
            >= thresholds["minimum_causal_positive_fraction"]
        ),
        "flip_rate": same["flip_rate"]
        >= thresholds["minimum_causal_flip_rate"],
        "specificity": (
            metrics["specificity_advantage"] is not None
            and metrics["specificity_advantage"]
            >= thresholds["minimum_causal_specificity_advantage"]
        ),
        "congruent_collateral": congruent["flip_rate"]
        <= thresholds["maximum_congruent_collateral_flip_rate"],
        **direction_gates,
    }
    return result


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not behavior["causal_scan_authorized"]:
        result = {
            "schema_version": "phase1103_causal_authorization.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "behavior_authorization_digest": behavior[
                "authorization_digest"
            ],
            "causal_scan_authorized": False,
            "component_scan_authorized": False,
            "reason": "behavior gate stopped hidden-state access",
            "decision": "stop_at_behavior_gate",
            "model_cells": {},
            "shared_confirmed_cells": [],
        }
        result["causal_authorization_digest"] = protocol.digest(result)
        protocol.write_json(
            protocol.OUT_ROOT / "analysis" / "causal_authorization.json",
            result,
        )
        print(json.dumps(result), flush=True)
        return

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    summaries = {}
    all_records = []
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "causal" / model / "summary.json"
        )
        summaries[model] = summary
        if summary.get("skipped"):
            continue
        records = protocol.read_jsonl(
            protocol.OUT_ROOT / "causal" / model / "patch_detail.jsonl"
        )
        all_records.extend(records)
        for row in records:
            grouped[(
                model, str(row["relation_pair"]), str(row["surface"]),
                str(row["split"]), int(row["depth"]),
                str(row["patch_kind"]),
            )].append(row)

    model_cells = {}
    for model in protocol.MODELS:
        if summaries[model].get("skipped"):
            continue
        depths = [
            int(row["depth"])
            for row in summaries[model]["sampled_depths"]
        ]
        for pair in summaries[model]["eligible_pairs"]:
            for surface in protocol.SURFACES:
                qualification_candidates = []
                for depth in depths:
                    row = metrics_at(
                        grouped, model, pair, surface,
                        "qualification", depth,
                    )
                    score = row["specificity_advantage"]
                    qualification_candidates.append((
                        float(score) if score is not None else -math.inf,
                        -depth,
                        depth,
                        row,
                    ))
                selected = max(qualification_candidates)
                selected_depth = int(selected[2])
                qualification_metrics = selected[3]
                confirmation_metrics = metrics_at(
                    grouped, model, pair, surface,
                    "confirmation", selected_depth,
                )
                qualification_gates = gates(qualification_metrics)
                confirmation_gates = gates(confirmation_metrics)
                passed = (
                    all(qualification_gates.values())
                    and all(confirmation_gates.values())
                )
                cell_id = "|".join((model, pair, surface))
                model_cells[cell_id] = {
                    "model": model,
                    "relation_pair": pair,
                    "surface": surface,
                    "selected_depth": selected_depth,
                    "selected_relative_depth": (
                        selected_depth / int(summaries[model]["layer_count"])
                    ),
                    "selection_used_only_qualification": True,
                    "qualification_metrics": qualification_metrics,
                    "qualification_gates": qualification_gates,
                    "confirmation_metrics": confirmation_metrics,
                    "confirmation_gates": confirmation_gates,
                    "passed": passed,
                }

    shared_confirmed_cells = []
    for pair in behavior["causally_eligible_pairs"]:
        for surface in protocol.SURFACES:
            passing_models = [
                model for model in protocol.MODELS
                if model_cells.get("|".join((model, pair, surface)), {}).get(
                    "passed", False
                )
            ]
            if (
                len(passing_models)
                >= protocol.THRESHOLDS[
                    "minimum_models_per_confirmed_causal_cell"
                ]
            ):
                shared_confirmed_cells.append({
                    "relation_pair": pair,
                    "surface": surface,
                    "passing_models": passing_models,
                })
    component_authorized = bool(shared_confirmed_cells)
    result = {
        "schema_version": "phase1103_causal_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_authorization_digest": behavior[
            "authorization_digest"
        ],
        "causal_scan_authorized": True,
        "model_scan_summaries": {
            model: summary["summary_digest"]
            for model, summary in summaries.items()
        },
        "model_cells": model_cells,
        "passing_model_cell_count": sum(
            row["passed"] for row in model_cells.values()
        ),
        "shared_confirmed_cells": shared_confirmed_cells,
        "component_scan_authorized": component_authorized,
        "decision": (
            "authorize_phase1104_component_localization"
            if component_authorized
            else "retain_pair_specific_response_map_without_mechanism_closure"
        ),
        "interpretation": (
            "A shared confirmed cell is causal evidence for expression-robust "
            "relation selection at a residual interface. It is not yet a "
            "head, MLP, neuron, global coordinate, or full relation-family "
            "mechanism."
            if component_authorized
            else "No pair-surface cell met the frozen two-model causal and "
            "control gates; observed response structure cannot be upgraded "
            "to relation-semantic transport."
        ),
    }
    result["causal_authorization_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "causal_authorization.json",
        result,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_model_cell_count": result["passing_model_cell_count"],
        "shared_confirmed_cells": shared_confirmed_cells,
        "component_scan_authorized": component_authorized,
        "causal_authorization_digest": result[
            "causal_authorization_digest"
        ],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
