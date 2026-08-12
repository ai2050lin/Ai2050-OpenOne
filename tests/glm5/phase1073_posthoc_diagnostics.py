#!/usr/bin/env python3
"""Descriptive diagnostics that cannot alter frozen Phase1073 gates."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1073_late_query_protocol as protocol


def finite(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def mean(values: list[Any]) -> float | None:
    clean = finite(values)
    return sum(clean) / len(clean) if clean else None


def top_rows(
    rows: list[dict[str, Any]],
    relation: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["base_relation"] == relation
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_OPERATION_ROLES
        and row["mean_operation_contrast_relative_magnitude"] is not None
    ]
    ranked = sorted(
        selected,
        key=lambda row: float(
            row["mean_operation_contrast_relative_magnitude"]
        ),
        reverse=True,
    )
    return [
        {
            "operation_condition": row["operation_condition"],
            "split": row["split"],
            "query_type": row["query_type"],
            "role": row["role"],
            "depth": int(row["depth"]),
            "relative_depth": float(row["relative_depth"]),
            "operation_contrast_relative_magnitude": float(
                row["mean_operation_contrast_relative_magnitude"]
            ),
            "transitive_did_relative_magnitude": row[
                "mean_transitive_did_relative_magnitude"
            ],
            "key_copy_did_relative_magnitude": row[
                "mean_key_copy_did_relative_magnitude"
            ],
        }
        for row in ranked[:limit]
    ]


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    condition_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "condition_evidence.jsonl"
    )
    operation_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "operation_evidence.jsonl"
    )
    relation_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "relation_gates.jsonl"
    )

    model_diagnostics = {}
    for model in protocol.MODELS:
        conditions = [
            row for row in condition_rows if row["model"] == model
        ]
        operations = [
            row for row in operation_rows if row["model"] == model
        ]
        response_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "response_metrics.jsonl"
        )
        behavior_groups = {}
        for task in protocol.TASK_FAMILIES:
            for alignment in protocol.KEY_ALIGNMENTS:
                values = [
                    row
                    for row in conditions
                    if row["task_family"] == task
                    and row["key_alignment"] == alignment
                ]
                behavior_groups[f"{task}::{alignment}"] = {
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
                }
        alignment_groups = {}
        for alignment in protocol.KEY_ALIGNMENTS:
            values = [
                row
                for row in operations
                if row["key_alignment"] == alignment
            ]
            alignment_groups[alignment] = {
                "operation_count": len(values),
                "operation_contrast_mean": mean([
                    row["operation_metrics"][
                        "operation_contrast_relative_magnitude"
                    ]
                    for row in values
                ]),
                "transitive_did_mean": mean([
                    row["operation_metrics"][
                        "transitive_did_relative_magnitude"
                    ]
                    for row in values
                ]),
                "key_copy_did_mean": mean([
                    row["operation_metrics"][
                        "key_copy_did_relative_magnitude"
                    ]
                    for row in values
                ]),
                "lexical_reuse_mean": mean([
                    row["operation_metrics"][
                        "operation_lexical_reuse_cosine"
                    ]
                    for row in values
                ]),
                "answer_invariance_mean": mean([
                    row["operation_metrics"][
                        "operation_answer_invariance_cosine"
                    ]
                    for row in values
                ]),
            }
        model_diagnostics[model] = {
            "behavior_groups": behavior_groups,
            "alignment_groups": alignment_groups,
            "maximum_candidate_transfer_gap": max(
                finite([
                    row["candidate_transfer_gap"]
                    for row in conditions
                ]),
                default=None,
            ),
            "maximum_semantic_transfer_gap": max(
                finite([
                    row["semantic_transfer_gap"]
                    for row in conditions
                ]),
                default=None,
            ),
            "pre_branch_operation_contrast_max": max(
                finite([
                    row["pre_branch_operation_contrast_max"]
                    for row in operations
                ]),
                default=None,
            ),
            "embedding_operation_contrast_max": max(
                finite([
                    row["embedding_operation_contrast_max"]
                    for row in operations
                ]),
                default=None,
            ),
            "top_operation_cells_by_relation": {
                relation: top_rows(response_rows, relation)
                for relation in protocol.BASE_RELATIONS
            },
        }
        del response_rows

    result = {
        "schema_version": "phase1073_posthoc_diagnostics.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "frozen_automatic_next": automatic,
        "model_diagnostics": model_diagnostics,
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
        protocol.OUT_ROOT / "analysis" / "posthoc_diagnostics.json",
        result,
    )
    print("Phase1073 posthoc diagnostics complete")


if __name__ == "__main__":
    main()
