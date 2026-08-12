#!/usr/bin/env python3
"""Aggregate BF16 held-out precision evidence without a mechanism score."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
OUT_ROOT = SOURCE_ROOT / "precision_bf16"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> None:
    protocol = read_json(
        SOURCE_ROOT / "precision_protocol" / "protocol.json"
    )
    rows = []
    model_summaries = {}
    for model in MODELS:
        summary = read_json(OUT_ROOT / model / "summary.json")
        model_rows = read_jsonl(
            OUT_ROOT / model / "candidate_summary.jsonl"
        )
        rows.extend(model_rows)
        cosines = [
            row["eight_bit_bf16_median_direction_cosine"]
            for row in model_rows
            if row["eight_bit_bf16_median_direction_cosine"] is not None
        ]
        model_summaries[model] = {
            "selected_event_count": len(model_rows),
            "singleton_forward_count": int(
                summary["singleton_forward_count"]
            ),
            "identity_maximum": float(summary["identity_maximum"]),
            "precision_supported_event_count": int(sum(
                row["precision_supported"] for row in model_rows
            )),
            "direction_four_panel_event_count": int(sum(
                row["bf16_direction_panel_count"] >= 4
                for row in model_rows
            )),
            "specificity_four_panel_event_count": int(sum(
                row["bf16_specificity_panel_count"] >= 4
                for row in model_rows
            )),
            "both_four_panel_event_count": int(sum(
                row["bf16_both_panel_count"] >= 4
                for row in model_rows
            )),
            "precision_cosine_median": (
                finite(np.median(cosines)) if cosines else None
            ),
            "precision_cosine_minimum": (
                finite(np.min(cosines)) if cosines else None
            ),
            "precision_cosine_maximum": (
                finite(np.max(cosines)) if cosines else None
            ),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
        }

    by_operation = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["operation"]].append(row)
    for operation, operation_rows in sorted(grouped.items()):
        cosines = [
            row["eight_bit_bf16_median_direction_cosine"]
            for row in operation_rows
            if row["eight_bit_bf16_median_direction_cosine"] is not None
        ]
        by_operation[operation] = {
            "selected_event_count": len(operation_rows),
            "precision_supported_event_count": int(sum(
                row["precision_supported"] for row in operation_rows
            )),
            "precision_cosine_median": finite(np.median(cosines)),
            "precision_cosine_range": [
                finite(np.min(cosines)),
                finite(np.max(cosines)),
            ],
            "direction_panel_count_median": finite(np.median([
                row["bf16_direction_panel_count"]
                for row in operation_rows
            ])),
            "cross_panel_direction_consistency_median": finite(
                np.median([
                    row["bf16_cross_panel_direction_consistency"]
                    for row in operation_rows
                    if row[
                        "bf16_cross_panel_direction_consistency"
                    ] is not None
                ])
            ),
        }

    sensitivity = []
    for cross_threshold in (0.10, 0.20, 0.30, 0.40, 0.50):
        for precision_threshold in (0.50, 0.70, 0.90):
            sensitivity.append({
                "cross_panel_direction_threshold": cross_threshold,
                "precision_cosine_threshold": precision_threshold,
                "event_count": int(sum(
                    row["bf16_both_panel_count"] >= 4
                    and row[
                        "bf16_cross_panel_direction_consistency"
                    ] is not None
                    and row[
                        "bf16_cross_panel_direction_consistency"
                    ] >= cross_threshold
                    and row[
                        "eight_bit_bf16_median_direction_cosine"
                    ] is not None
                    and row[
                        "eight_bit_bf16_median_direction_cosine"
                    ] >= precision_threshold
                    for row in rows
                )),
            })

    supported = [
        {
            "model": row["model"],
            "operation": row["operation"],
            "event_id": row["event_id"],
            "bf16_both_panel_count": row["bf16_both_panel_count"],
            "bf16_cross_panel_direction_consistency": row[
                "bf16_cross_panel_direction_consistency"
            ],
            "eight_bit_bf16_median_direction_cosine": row[
                "eight_bit_bf16_median_direction_cosine"
            ],
        }
        for row in rows if row["precision_supported"]
    ]
    result = {
        "schema_version": "phase1014_bf16_precision_summary.v1",
        "phase": 1014,
        "precision_protocol_digest": protocol[
            "precision_protocol_digest"
        ],
        "selection_used_confirmation": False,
        "selected_event_count": len(rows),
        "singleton_forward_count": sum(
            value["singleton_forward_count"]
            for value in model_summaries.values()
        ),
        "precision_supported_event_count": len(supported),
        "models": model_summaries,
        "by_operation": by_operation,
        "precision_supported_events": supported,
        "threshold_sensitivity": sensitivity,
        "interpretation": {
            "within_panel_precision_alignment": (
                "8-bit and BF16 directions are usually highly aligned "
                "for Q candidates"
            ),
            "cross_pattern_reuse": (
                "the same physical heads recur, but a single shared "
                "direction across language families is much rarer"
            ),
            "strongest_current_unit": (
                "a model-local conditional direction family, not a "
                "universal fixed vector or causal edge"
            ),
        },
        "claim_limits": [
            "BF16 replication does not prove semantic decoding",
            "same-head recurrence does not prove head necessity",
            "cross-pattern direction mismatch can reflect conditional "
            "subspaces, lexical routing, or both",
            "operational thresholds are sensitivity rulers, not theory",
        ],
    }
    path = OUT_ROOT / "summary.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
