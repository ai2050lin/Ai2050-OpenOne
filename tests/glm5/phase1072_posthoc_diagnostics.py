#!/usr/bin/env python3
"""Descriptive diagnostics for Phase1072; never changes frozen gates."""

from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1072_bidirectional_pattern_protocol as protocol
import phase1072_finalize as frozen


def finite(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def mean(values: list[Any]) -> float | None:
    clean = finite(values)
    return sum(clean) / len(clean) if clean else None


def peak(
    rows: list[dict[str, Any]],
    role: str,
) -> dict[str, Any] | None:
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in rows:
        parsed = protocol.parse_condition(row["relation"])
        if (
            parsed["task_family"] != "transitive"
            or row["conditioning"] != "all"
            or row["role"] != role
        ):
            continue
        value = row["mean_process_did_relative_magnitude"]
        if value is not None and math.isfinite(float(value)):
            grouped[
                (int(row["depth"]), float(row["relative_depth"]))
            ].append(float(value))
    if not grouped:
        return None
    values = {
        key: sum(items) / len(items)
        for key, items in grouped.items()
    }
    key = max(values, key=values.get)
    return {
        "depth": key[0],
        "relative_depth": key[1],
        "mean_process_did": values[key],
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    condition_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "condition_evidence.jsonl"
    )
    relation_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "relation_gates.jsonl"
    )

    model_diagnostics = {}
    within_model_rows = []
    for model in protocol.MODELS:
        response_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "response_metrics.jsonl"
        )
        selected = [
            row for row in condition_rows if row["model"] == model
        ]
        grouped = {}
        for family in protocol.TASK_FAMILIES:
            for prompt in protocol.PROMPT_BRANCHES:
                for order in protocol.EVIDENCE_ORDERS:
                    values = [
                        row
                        for row in selected
                        if row["task_family"] == family
                        and row["prompt_branch"] == prompt
                        and row["evidence_order"] == order
                    ]
                    grouped[
                        f"{family}::{prompt}::{order}"
                    ] = {
                        "condition_count": len(values),
                        "formal_candidate_accuracy_mean": mean([
                            row["formal_candidate_accuracy"]
                            for row in values
                        ]),
                        "formal_semantic_first_rate_mean": mean([
                            row["formal_semantic_first_rate"]
                            for row in values
                        ]),
                        "calibration_candidate_accuracy_mean": mean([
                            row["calibration_candidate_accuracy"]
                            for row in values
                        ]),
                        "calibration_semantic_first_rate_mean": mean([
                            row["calibration_semantic_first_rate"]
                            for row in values
                        ]),
                        "process_did_mean": mean([
                            row["post_evidence_metrics"][
                                "process_did_relative_magnitude"
                            ]
                            for row in values
                        ]),
                        "lexical_reuse_mean": mean([
                            row["post_evidence_metrics"][
                                "process_lexical_reuse_cosine"
                            ]
                            for row in values
                        ]),
                        "answer_invariance_mean": mean([
                            row["post_evidence_metrics"][
                                "process_answer_invariance_cosine"
                            ]
                            for row in values
                        ]),
                    }
        target = [
            row
            for row in selected
            if row["task_family"] == "transitive"
        ]
        model_diagnostics[model] = {
            "condition_groups": grouped,
            "maximum_candidate_transfer_gap": max(
                finite([
                    row["candidate_transfer_gap"]
                    for row in selected
                ]),
                default=None,
            ),
            "maximum_semantic_transfer_gap": max(
                finite([
                    row["semantic_transfer_gap"]
                    for row in selected
                ]),
                default=None,
            ),
            "hard_negative_max": max(
                finite([
                    row["hard_negative_process_did_max"]
                    for row in selected
                ]),
                default=None,
            ),
            "evidence_probe_peak": peak(
                response_rows, "evidence_probe"
            ),
            "answer_boundary_peak": peak(
                response_rows, "answer_boundary"
            ),
            "target_lexical_specificity_mean": mean([
                row["post_evidence_metrics"][
                    "surface_relative_magnitude"
                ]
                for row in target
            ]),
        }
        for left, right in itertools.combinations(
            protocol.BASE_RELATIONS, 2
        ):
            left_profiles = []
            right_profiles = []
            for prompt in protocol.PROMPT_BRANCHES:
                for order in protocol.EVIDENCE_ORDERS:
                    left_profiles.extend(frozen.profile(
                        response_rows,
                        protocol.condition_key(
                            left, "transitive", prompt, order
                        ),
                    ))
                    right_profiles.extend(frozen.profile(
                        response_rows,
                        protocol.condition_key(
                            right, "transitive", prompt, order
                        ),
                    ))
            within_model_rows.append({
                "model": model,
                "left_relation": left,
                "right_relation": right,
                "profile_cosine": frozen.cosine(
                    left_profiles, right_profiles
                ),
            })

    result = {
        "schema_version": "phase1072_posthoc_diagnostics.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "frozen_automatic_next": automatic,
        "model_diagnostics": model_diagnostics,
        "within_model_relation_profile_cosines": within_model_rows,
        "relation_gate_failure_inventory": [
            {
                "model": row["model"],
                "base_relation": row["base_relation"],
                "failed_checks": [
                    key
                    for key, value in row["checks"].items()
                    if not value
                ],
            }
            for row in relation_rows
        ],
        "status": (
            "Descriptive only. No threshold or automatic decision was "
            "changed after viewing results."
        ),
    }
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "posthoc_diagnostics.json",
        result,
    )
    print("Phase1072 posthoc diagnostics complete")


if __name__ == "__main__":
    main()
