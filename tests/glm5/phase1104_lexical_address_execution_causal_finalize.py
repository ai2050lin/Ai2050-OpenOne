#!/usr/bin/env python3
"""Select Phase1104 depths on qualification and judge confirmation."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any

import phase1104_lexical_address_execution_protocol as protocol


CONTROL_KINDS = (
    "selector_null_control",
    "ordinal_control",
    "wrong_pair_control",
    "equal_norm_random_control",
)


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row for row in rows
        if row["behavior_valid"] and row["recovery"] is not None
    ]
    recoveries = [float(row["recovery"]) for row in valid]
    return {
        "count": len(rows),
        "finite_fraction": sum(row["finite"] for row in rows) / max(len(rows), 1),
        "behavior_valid_fraction": sum(row["behavior_valid"] for row in rows) / max(len(rows), 1),
        "valid_count": len(valid),
        "median_recovery": statistics.median(recoveries) if recoveries else None,
        "mean_recovery": statistics.fmean(recoveries) if recoveries else None,
        "positive_recovery_fraction": (
            sum(value > 0.0 for value in recoveries) / len(recoveries)
            if recoveries else None
        ),
        "flip_rate": sum(row["flip"] for row in rows) / max(len(rows), 1),
        "median_delta_norm": (
            statistics.median(
                float(row["delta_norm"]) for row in rows
                if row["delta_norm"] is not None
            )
            if rows else None
        ),
    }


def metrics_at(
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    model: str,
    pair: str,
    surface: str,
    source_regime: str,
    target_regime: str,
    split: str,
    depth: int,
) -> dict[str, Any]:
    result = {}
    kinds = (
        "within_regime_raw",
        "cross_regime_raw",
        "cross_regime_interaction",
        *CONTROL_KINDS,
        "congruent_collateral_interaction",
    )
    for kind in kinds:
        result[kind] = metric(grouped.get((
            model, pair, surface, source_regime, target_regime,
            split, depth, kind,
        ), []))
    interaction_rows = grouped.get((
        model, pair, surface, source_regime, target_regime,
        split, depth, "cross_regime_interaction",
    ), [])
    result["target_directions"] = {
        direction: metric([
            row for row in interaction_rows
            if row["target_direction"] == direction
        ])
        for direction in ("q0_to_q1", "q1_to_q0")
    }
    primary = result["cross_regime_interaction"]["median_recovery"]
    control_medians = [
        result[kind]["median_recovery"] for kind in CONTROL_KINDS
        if result[kind]["median_recovery"] is not None
    ]
    result["specificity_advantage"] = (
        float(primary) - max(float(value) for value in control_medians)
        if primary is not None and control_medians else None
    )
    return result


def gates(metrics: dict[str, Any]) -> dict[str, bool]:
    thresholds = protocol.THRESHOLDS
    primary = metrics["cross_regime_interaction"]
    raw = metrics["cross_regime_raw"]
    within = metrics["within_regime_raw"]
    collateral = metrics["congruent_collateral_interaction"]
    result = {
        "finite": primary["finite_fraction"]
        >= thresholds["minimum_causal_finite_fraction"],
        "behavior_valid": primary["behavior_valid_fraction"]
        >= thresholds["minimum_causal_behavior_valid_fraction"],
        "within_regime_instrument": (
            within["median_recovery"] is not None
            and within["median_recovery"]
            >= thresholds["minimum_raw_median_recovery"]
        ),
        "cross_regime_raw": (
            raw["median_recovery"] is not None
            and raw["median_recovery"]
            >= thresholds["minimum_raw_median_recovery"]
        ),
        "interaction_median": (
            primary["median_recovery"] is not None
            and primary["median_recovery"]
            >= thresholds["minimum_interaction_median_recovery"]
        ),
        "interaction_positive": (
            primary["positive_recovery_fraction"] is not None
            and primary["positive_recovery_fraction"]
            >= thresholds["minimum_positive_fraction"]
        ),
        "interaction_flip": primary["flip_rate"]
        >= thresholds["minimum_flip_rate"],
        "specificity": (
            metrics["specificity_advantage"] is not None
            and metrics["specificity_advantage"]
            >= thresholds["minimum_specificity_advantage"]
        ),
        "congruent_collateral": collateral["flip_rate"]
        <= thresholds["maximum_congruent_collateral_flip_rate"],
    }
    for direction, row in metrics["target_directions"].items():
        result[f"{direction}_median"] = (
            row["median_recovery"] is not None
            and row["median_recovery"]
            >= thresholds["minimum_each_direction_median_recovery"]
        )
        result[f"{direction}_positive"] = (
            row["positive_recovery_fraction"] is not None
            and row["positive_recovery_fraction"]
            >= thresholds["minimum_each_direction_positive_fraction"]
        )
    return result


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not behavior["model_specific_causal_scan_authorized"]:
        result = {
            "schema_version": "phase1104_causal_authorization.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "behavior_authorization_digest": behavior["authorization_digest"],
            "causal_scan_authorized": False,
            "component_localization_authorized": False,
            "decision": "stop_at_behavior_gate",
            "model_cells": {},
            "model_specific_confirmed_cells": [],
            "cross_model_confirmed_cells": [],
        }
        result["causal_authorization_digest"] = protocol.digest(result)
        protocol.write_json(
            protocol.OUT_ROOT / "analysis" / "causal_authorization.json", result
        )
        print(json.dumps(result), flush=True)
        return
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    summaries = {}
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "causal" / model / "summary.json"
        )
        summaries[model] = summary
        if summary.get("skipped"):
            continue
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "causal" / model / "patch_detail.jsonl"
        )
        for row in rows:
            grouped[(
                model, str(row["relation_pair"]), str(row["surface"]),
                str(row["source_regime"]), str(row["target_regime"]),
                str(row["split"]), int(row["depth"]), str(row["patch_kind"]),
            )].append(row)
    model_cells = {}
    model_specific_confirmed = []
    regime_directions = (
        ("relation_label", "neutral_label"),
        ("neutral_label", "relation_label"),
    )
    for model, summary in summaries.items():
        if summary.get("skipped"):
            continue
        depths = [int(row["depth"]) for row in summary["sampled_depths"]]
        for pair in summary["eligible_pairs"]:
            for surface in protocol.SURFACES:
                for source_regime, target_regime in regime_directions:
                    candidates = []
                    for depth in depths:
                        row = metrics_at(
                            grouped, model, pair, surface,
                            source_regime, target_regime,
                            "qualification", depth,
                        )
                        score = row["specificity_advantage"]
                        candidates.append((
                            float(score) if score is not None else -math.inf,
                            -depth,
                            depth,
                            row,
                        ))
                    selected = max(candidates)
                    selected_depth = int(selected[2])
                    qualification = selected[3]
                    confirmation = metrics_at(
                        grouped, model, pair, surface,
                        source_regime, target_regime,
                        "confirmation", selected_depth,
                    )
                    qualification_gates = gates(qualification)
                    confirmation_gates = gates(confirmation)
                    passed = all(qualification_gates.values()) and all(
                        confirmation_gates.values()
                    )
                    cell_id = "|".join((
                        model, pair, surface, source_regime, target_regime,
                    ))
                    model_cells[cell_id] = {
                        "model": model,
                        "relation_pair": pair,
                        "surface": surface,
                        "source_regime": source_regime,
                        "target_regime": target_regime,
                        "selected_depth": selected_depth,
                        "selected_relative_depth": (
                            selected_depth / int(summary["layer_count"])
                        ),
                        "selection_used_only_qualification": True,
                        "qualification_metrics": qualification,
                        "qualification_gates": qualification_gates,
                        "confirmation_metrics": confirmation,
                        "confirmation_gates": confirmation_gates,
                        "passed": passed,
                    }
                    if passed:
                        model_specific_confirmed.append(cell_id)
    cross_model_confirmed = []
    for pair in protocol.CANDIDATE_PAIRS:
        for surface in protocol.SURFACES:
            for source_regime, target_regime in regime_directions:
                passing_models = [
                    model for model in protocol.MODELS
                    if model_cells.get("|".join((
                        model, pair, surface, source_regime, target_regime,
                    )), {}).get("passed", False)
                ]
                if len(passing_models) >= protocol.THRESHOLDS[
                    "minimum_models_for_cross_model_upgrade"
                ]:
                    cross_model_confirmed.append({
                        "relation_pair": pair,
                        "surface": surface,
                        "source_regime": source_regime,
                        "target_regime": target_regime,
                        "passing_models": passing_models,
                    })
    component_authorized = bool(model_specific_confirmed)
    result = {
        "schema_version": "phase1104_causal_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_authorization_digest": behavior["authorization_digest"],
        "causal_scan_authorized": True,
        "model_scan_summary_digests": {
            model: summary["summary_digest"]
            for model, summary in summaries.items()
        },
        "model_cells": model_cells,
        "model_specific_confirmed_cells": model_specific_confirmed,
        "cross_model_confirmed_cells": cross_model_confirmed,
        "component_localization_authorized": component_authorized,
        "cross_model_mechanism_upgrade": bool(cross_model_confirmed),
        "decision": (
            "authorize_model_specific_component_localization"
            if component_authorized
            else "retain_causal_response_map_without_execution_closure"
        ),
        "interpretation": (
            "A passing one-model cell is evidence for a model-specific, "
            "content-conditioned lexical routing interface. It is not a "
            "paraphrase-semantic or cross-model mechanism."
            if component_authorized
            else "No cell passed the frozen interaction and control gates. "
            "Raw replacement effects remain instrument checks only."
        ),
    }
    result["causal_authorization_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "causal_authorization.json", result
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "model_specific_confirmed_cell_count": len(model_specific_confirmed),
        "cross_model_confirmed_cells": cross_model_confirmed,
        "component_localization_authorized": component_authorized,
        "causal_authorization_digest": result["causal_authorization_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
