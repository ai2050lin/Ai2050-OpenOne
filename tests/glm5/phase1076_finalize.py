#!/usr/bin/env python3
"""Aggregate Phase1076 causal effects and apply frozen gates."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1076_polarity_head_causal_protocol as protocol


EPSILON = 1e-12


def stats(values: list[float]) -> dict[str, Any]:
    finite = [
        float(value) for value in values if math.isfinite(value)
    ]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "positive_fraction": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "positive_fraction": sum(
            value > 0.0 for value in finite
        ) / len(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def aggregate(
    records: list[dict[str, Any]],
    conditioned_only: bool,
) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if row["all_finite"]
        and (
            not conditioned_only
            or row["pair_behavior_conditioned"]
        )
    ]
    by_contrast: dict[str, Any] = {}
    for contrast in protocol.CONTRASTS:
        contrast_rows = [
            row for row in selected if row["contrast"] == contrast
        ]
        interventions = {}
        for intervention in protocol.INTERVENTIONS:
            intervention_stats = stats([
                float(row["margin_drops"][intervention])
                for row in contrast_rows
            ])
            intervention_stats["by_task"] = {
                task: stats([
                    float(row["margin_drops"][intervention])
                    for row in contrast_rows
                    if row["task"] == task
                ])
                for task in protocol.TASKS_BY_CONTRAST[contrast]
            }
            intervention_stats["by_path"] = {
                path: stats([
                    float(row["margin_drops"][intervention])
                    for row in contrast_rows
                    if row["path"] == path
                ])
                for path in protocol.PATHS
            }
            intervention_stats["by_layout"] = {
                layout: stats([
                    float(row["margin_drops"][intervention])
                    for row in contrast_rows
                    if row["layout"] == layout
                ])
                for layout in protocol.LAYOUTS
            }
            interventions[intervention] = intervention_stats
        by_contrast[contrast] = {
            "case_count": len(contrast_rows),
            "baseline_margin": stats([
                float(row["baseline_margin"])
                for row in contrast_rows
            ]),
            "interventions": interventions,
        }
    return {
        "population": (
            "behavior_conditioned"
            if conditioned_only
            else "all"
        ),
        "finite_case_count": len(selected),
        "contrasts": by_contrast,
    }


def finite_mean(value: dict[str, Any]) -> float:
    result = value["mean"]
    return float(result) if result is not None else float("nan")


def model_gate(
    prereg: dict[str, Any],
    scan_summary: dict[str, Any],
    primary: dict[str, Any],
) -> dict[str, Any]:
    gates = prereg["gates"]
    polarity = primary["contrasts"]["polarity"][
        "interventions"
    ]
    surface = primary["contrasts"]["surface"][
        "interventions"
    ]
    selected_swap = finite_mean(polarity["selected_swap"])
    control_swap = finite_mean(polarity["control_swap"])
    selected_zero = finite_mean(polarity["selected_zero"])
    control_zero = finite_mean(polarity["control_zero"])
    surface_selected_swap = finite_mean(
        surface["selected_swap"]
    )
    positive_fraction = polarity["selected_swap"][
        "positive_fraction"
    ]
    selected_control_difference = (
        selected_swap - control_swap
    )
    selected_control_ratio = selected_swap / max(
        abs(control_swap), EPSILON
    )
    zero_control_ratio = selected_zero / max(
        abs(control_zero), EPSILON
    )
    task_specificity_ratio = selected_swap / max(
        abs(surface_selected_swap), EPSILON
    )
    path_drops = {
        path: finite_mean(
            polarity["selected_swap"]["by_path"][path]
        )
        for path in protocol.PATHS
    }
    checks = {
        "causal_finite_rate": (
            scan_summary["candidate_margin_finite_rate"]
            >= gates["causal_finite_rate_min"]
        ),
        "selected_swap_drop": (
            selected_swap
            >= gates["polarity_selected_swap_drop_min"]
        ),
        "selected_swap_positive_fraction": (
            positive_fraction is not None
            and positive_fraction
            >= gates[
                "polarity_selected_swap_positive_fraction_min"
            ]
        ),
        "selected_minus_control": (
            selected_control_difference
            >= gates["polarity_selected_minus_control_min"]
        ),
        "selected_to_control_ratio": (
            selected_control_ratio
            >= gates[
                "polarity_selected_to_control_ratio_min"
            ]
        ),
        "selected_zero_drop": (
            selected_zero
            >= gates["polarity_selected_zero_drop_min"]
        ),
        "zero_to_control_ratio": (
            zero_control_ratio
            >= gates["polarity_zero_to_control_ratio_min"]
        ),
        "each_path_swap_drop": (
            min(path_drops.values())
            >= gates["polarity_each_path_swap_drop_min"]
        ),
        "polarity_to_surface_ratio": (
            task_specificity_ratio
            >= gates["polarity_to_surface_swap_ratio_min"]
        ),
    }
    return {
        "selected_swap_drop": selected_swap,
        "selected_swap_positive_fraction": positive_fraction,
        "control_swap_drop": control_swap,
        "selected_minus_control": selected_control_difference,
        "selected_to_control_ratio": selected_control_ratio,
        "selected_zero_drop": selected_zero,
        "control_zero_drop": control_zero,
        "zero_to_control_ratio": zero_control_ratio,
        "surface_selected_swap_drop": surface_selected_swap,
        "polarity_to_surface_ratio": task_specificity_ratio,
        "selected_swap_drop_by_path": path_drops,
        "checks": checks,
        "model_causal_gate_passed": all(checks.values()),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    decision = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_decision.json"
    )
    model_results = {}
    passing_models = []
    if decision["should_run_causal_validation"]:
        for model in protocol.MODELS:
            scan_summary = protocol.read_json(
                protocol.OUT_ROOT
                / "causal"
                / model
                / "summary.json"
            )
            records = protocol.read_jsonl(
                protocol.OUT_ROOT
                / "causal"
                / model
                / "causal_records.jsonl"
            )
            primary = aggregate(records, conditioned_only=False)
            sensitivity = aggregate(
                records, conditioned_only=True
            )
            gate = model_gate(prereg, scan_summary, primary)
            if gate["model_causal_gate_passed"]:
                passing_models.append(model)
            model_results[model] = {
                "scan_summary": scan_summary,
                "primary_all_cases": primary,
                "sensitivity_behavior_conditioned": sensitivity,
                "causal_gate": gate,
            }
    required = int(prereg["gates"]["minimum_causal_models"])
    cross_model_passed = (
        decision["should_run_causal_validation"]
        and len(passing_models) >= required
        and set(passing_models) == set(protocol.MODELS)
    )
    payload = {
        "schema_version": "phase1076_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_decision_digest": decision[
            "decision_digest"
        ],
        "behavior_authorized": decision[
            "should_run_causal_validation"
        ],
        "primary_analysis_population": prereg[
            "primary_analysis_population"
        ],
        "models": model_results,
        "passing_causal_models": passing_models,
        "cross_model_causal_gate_passed": cross_model_passed,
        "claim_status": (
            "cross_model_local_causal_influence_supported"
            if cross_model_passed
            else (
                "targeted_local_causal_gate_not_met"
                if decision["should_run_causal_validation"]
                else "causal_test_not_authorized"
            )
        ),
        "scientific_scope": (
            "The result concerns the held-out height-polarity family "
            "and the frozen late Attention coalition only. It does not "
            "establish a universal relative code, a complete routing "
            "algorithm, or physical head homology."
        ),
    }
    payload["summary_digest"] = protocol.digest(payload)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json",
        payload,
    )
    automatic_next = {
        "schema_version": "phase1076_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "summary_digest": payload["summary_digest"],
        "should_continue_automatically": False,
        "route": "stop_after_authorized_causal_validation",
        "reason": (
            "Phase1076 exhausts the continuation authorized by the "
            "Phase1075 preregistration. A new family, larger replication, "
            "or finer causal decomposition requires a separately frozen "
            "protocol regardless of whether this local gate passed."
        ),
        "recommended_next_big_task": (
            "Freeze an independent multi-family relative-selection atlas "
            "that separates evidence representation, relation polarity, "
            "surface selection, and answer realization before any new "
            "causal localization."
        ),
    }
    automatic_next["decision_digest"] = protocol.digest(
        automatic_next
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic_next,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_causal_models": passing_models,
        "cross_model_causal_gate_passed": cross_model_passed,
        "claim_status": payload["claim_status"],
        "should_continue_automatically": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
