#!/usr/bin/env python3
"""Descriptive Phase1071 diagnostics that do not alter frozen gates."""

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

import phase1071_exposure_pattern_protocol as protocol


def finite(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def median(values: list[Any]) -> float | None:
    clean = finite(values)
    return statistics.median(clean) if clean else None


def aggregate_metrics(
    rows: list[dict[str, Any]],
    conditioning: str,
) -> dict[str, float | None]:
    selected = [
        row
        for row in rows
        if row["conditioning"] == conditioning
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    return {
        "process_did_relative_magnitude": median([
            row["mean_process_did_relative_magnitude"]
            for row in selected
        ]),
        "process_lexical_reuse_cosine": median([
            row["mean_process_lexical_reuse_cosine"]
            for row in selected
        ]),
        "process_answer_invariance_cosine": median([
            row["mean_process_answer_invariance_cosine"]
            for row in selected
        ]),
        "lexical_surface_relative_magnitude": median([
            row["mean_surface_relative_magnitude"]
            for row in selected
        ]),
        "process_answer_absolute_cosine": median([
            row["mean_process_answer_absolute_cosine"]
            for row in selected
        ]),
    }


def role_depth_diagnostics(
    rows: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["conditioning"] == "all" and row["role"] == role
    ]
    by_depth: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in selected:
        value = row["mean_process_did_relative_magnitude"]
        if value is not None and math.isfinite(float(value)):
            by_depth[(
                int(row["depth"]),
                float(row["relative_depth"]),
            )].append(float(value))
    profile = [
        {
            "depth": depth,
            "relative_depth": relative_depth,
            "median_process_did_relative_magnitude": (
                statistics.median(values)
            ),
        }
        for (depth, relative_depth), values in sorted(
            by_depth.items(), key=lambda item: item[0][1]
        )
    ]
    peak = max(
        profile,
        key=lambda row: row[
            "median_process_did_relative_magnitude"
        ],
        default=None,
    )
    threshold = protocol.GATES[
        "process_did_relative_magnitude_min"
    ]
    first_persistent = None
    for index in range(max(0, len(profile) - 1)):
        current = profile[index]
        following = profile[index + 1]
        if (
            current["median_process_did_relative_magnitude"]
            >= threshold
            and following[
                "median_process_did_relative_magnitude"
            ] >= threshold
        ):
            first_persistent = {
                "depth": current["depth"],
                "relative_depth": current["relative_depth"],
                "threshold": threshold,
            }
            break
    return {
        "peak": peak,
        "first_two_consecutive_depths_above_threshold": (
            first_persistent
        ),
        "profile": profile,
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    calibration_selection = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "analysis"
        / "prompt_selection.json"
    )
    selected_style = str(prereg["selected_prompt_style"])
    within_family = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "within_model_family_profiles.jsonl"
    )
    model_gates = {
        row["model"]: row
        for row in protocol.read_jsonl(
            protocol.OUT_ROOT / "analysis" / "model_gates.jsonl"
        )
    }

    model_rows = {}
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        response_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "response_metrics.jsonl"
        )
        readout_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "local_readout_metrics.jsonl"
        )
        calibration_summary = protocol.read_json(
            protocol.CALIBRATION_ROOT
            / "atlas"
            / model
            / "summary.json"
        )
        calibration_style = calibration_summary["styles"][
            selected_style
        ]
        total_cases = sum(
            row["case_count"]
            for row in summary["relations"].values()
        )
        total_hits = sum(
            row["candidate_hit_count"]
            for row in summary["relations"].values()
        )
        total_natural = sum(
            row["natural_audit_case_count"]
            for row in summary["relations"].values()
        )
        semantic_hits = sum(
            row["semantic_first_natural_rate"]
            * row["natural_audit_case_count"]
            for row in summary["relations"].values()
        )
        hard_negative = [
            row["mean_process_did_relative_magnitude"]
            for row in response_rows
            if row["conditioning"] == "all"
            and row["role"] in protocol.HARD_NEGATIVE_ROLES
            and row["mean_process_did_relative_magnitude"]
            is not None
        ]
        late_readout = [
            row
            for row in readout_rows
            if row["conditioning"] == "all"
            and float(row["relative_depth"])
            >= protocol.GATES["late_depth_start"]
        ]
        family_cosines = [
            row["normalized_profile_cosine"]
            for row in within_family
            if row["model"] == model
        ]
        model_rows[model] = {
            "behavior": {
                "candidate_accuracy": (
                    total_hits / total_cases if total_cases else 0.0
                ),
                "semantic_first_natural_rate": (
                    semantic_hits / total_natural
                    if total_natural else 0.0
                ),
                "strong_relations": [
                    relation
                    for relation, row in summary[
                        "relations"
                    ].items()
                    if row["strong_behavior_gate_passed"]
                ],
            },
            "calibration_to_mechanism_transfer": {
                "calibration_candidate_accuracy": (
                    calibration_style["candidate_accuracy"]
                ),
                "mechanism_candidate_accuracy": (
                    total_hits / total_cases if total_cases else 0.0
                ),
                "candidate_accuracy_change": (
                    total_hits / total_cases
                    - calibration_style["candidate_accuracy"]
                    if total_cases else None
                ),
                "calibration_semantic_first_rate": (
                    calibration_style["semantic_first_rate"]
                ),
                "mechanism_semantic_first_rate": (
                    semantic_hits / total_natural
                    if total_natural else 0.0
                ),
                "semantic_first_rate_change": (
                    semantic_hits / total_natural
                    - calibration_style["semantic_first_rate"]
                    if total_natural else None
                ),
            },
            "numerical": {
                "candidate_finite_rate": summary[
                    "candidate_finite_rate"
                ],
                "residual_metric_finite_rate": summary[
                    "residual_metric_finite_rate"
                ],
                "internal_readout_finite_rate": summary[
                    "internal_readout_finite_rate"
                ],
                "nonfinite_candidate_count": summary[
                    "nonfinite_candidate_count"
                ],
                "nonfinite_residual_metric_count": summary[
                    "nonfinite_residual_metric_count"
                ],
                "nonfinite_internal_readout_count": summary[
                    "nonfinite_internal_readout_count"
                ],
            },
            "all_pair_primary": aggregate_metrics(
                response_rows, "all"
            ),
            "behavior_conditioned_primary": aggregate_metrics(
                response_rows, "behavior_conditioned"
            ),
            "hard_negative_process_did_max": (
                max(finite(hard_negative))
                if finite(hard_negative) else None
            ),
            "role_depth": {
                role: role_depth_diagnostics(response_rows, role)
                for role in (
                    "evidence_probe",
                    "operator",
                    "query",
                    "answer_boundary",
                )
            },
            "late_local_answer_readout": {
                "matched_answer_positive_rate": median([
                    row["matched_answer_positive_rate"]
                    for row in late_readout
                ]),
                "mismatched_answer_positive_rate": median([
                    row["mismatched_answer_positive_rate"]
                    for row in late_readout
                ]),
                "positive_rate_gap": median([
                    row["positive_rate_gap"]
                    for row in late_readout
                ]),
                "process_to_answer_readout_ratio": median([
                    row[
                        "absolute_process_to_answer_readout_ratio"
                    ]
                    for row in late_readout
                ]),
            },
            "within_model_relation_profile_cosine": {
                "minimum": min(family_cosines),
                "median": statistics.median(family_cosines),
                "maximum": max(family_cosines),
                "warning": (
                    "Very high scalar-profile similarity may reflect "
                    "generic depth dynamics; it is not a family circuit."
                ),
            },
            "frozen_gate": model_gates[model],
        }

    result = {
        "schema_version": "phase1071_posthoc_diagnostics.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "frozen_automatic_decision_unchanged": automatic,
        "selected_prompt_style": prereg["selected_prompt_style"],
        "calibration_gate_passed": calibration_selection[
            "calibration_gate_passed"
        ],
        "models": model_rows,
        "interpretation_limits": [
            "These diagnostics are descriptive and cannot rescue a failed frozen gate.",
            "Peak depth is a residual-state location, not a component or causal edge.",
            "A shared direction plus a lexical-specific difference is compatible with differential reuse but does not prove optimal compression.",
            "High relation-profile similarity can be a generic depth-shape confound.",
        ],
    }
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "posthoc_diagnostics.json",
        result,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "automatic_next_unchanged": automatic[
            "should_continue_automatically"
        ],
        "models": {
            model: {
                "all_pair_primary": row[
                    "all_pair_primary"
                ],
                "hard_negative_max": row[
                    "hard_negative_process_did_max"
                ],
                "evidence_peak": row["role_depth"][
                    "evidence_probe"
                ]["peak"],
                "answer_peak": row["role_depth"][
                    "answer_boundary"
                ]["peak"],
            }
            for model, row in model_rows.items()
        },
    }), flush=True)


if __name__ == "__main__":
    main()
