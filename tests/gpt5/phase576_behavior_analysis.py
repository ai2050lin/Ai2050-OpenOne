#!/usr/bin/env python3
"""Audit Phase576 behavior failures without reading causal or sealed rows."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase576_natural_fruit"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUTPUT = OUT_DIR / "phase576_behavior_failure_analysis.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(model: str) -> Iterator[dict[str, Any]]:
    path = OUT_DIR / f"phase576_{model}_behavior_rows.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(1, len(rows))


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "semantic_accuracy": rate(rows, "semantic_correct"),
        "strict_sequence_accuracy": rate(rows, "strict_sequence_correct"),
        "unrecoverable_rate": sum(
            row["semantic_event"] == "unrecoverable" for row in rows
        ) / max(1, len(rows)),
        "other_relation_confusion_rate": sum(
            row["semantic_event"] == "same_object_other_relation" for row in rows
        ) / max(1, len(rows)),
        "event_counts": dict(Counter(row["semantic_event"] for row in rows)),
    }


def analyze_model(model: str) -> dict[str, Any]:
    rows = list(iter_rows(model))
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "split_relation": defaultdict(list),
        "split_order": defaultdict(list),
        "split_surface": defaultdict(list),
        "split_object": defaultdict(list),
    }
    for row in rows:
        groups["split_relation"][f"{row['split']}:{row['relation']}"] .append(row)
        groups["split_order"][f"{row['split']}:{row['surface_order']}"] .append(row)
        groups["split_surface"][f"{row['split']}:surface{row['surface_id']:02d}"] .append(row)
        groups["split_object"][f"{row['split']}:{row['object_id']}:{row['relation']}"] .append(row)
    unrecoverable = [row for row in rows if row["semantic_event"] == "unrecoverable"]
    common_unrecoverable = Counter(
        row["normalized_generated"] for row in unrecoverable
    ).most_common(30)
    selected_wrong = Counter(
        row["selected_candidate"]
        for row in rows
        if row["semantic_event"] == "registered_other"
    ).most_common(20)
    return {
        "row_count": len(rows),
        "unique_case_count": len({row["case_id"] for row in rows}),
        "overall": summarize_group(rows),
        "by_split_relation": {
            key: summarize_group(value)
            for key, value in sorted(groups["split_relation"].items())
        },
        "by_split_order": {
            key: summarize_group(value)
            for key, value in sorted(groups["split_order"].items())
        },
        "by_split_surface": {
            key: summarize_group(value)
            for key, value in sorted(groups["split_surface"].items())
        },
        "by_split_object_relation": {
            key: summarize_group(value)
            for key, value in sorted(groups["split_object"].items())
        },
        "most_common_unrecoverable_outputs": [
            {"text": text, "count": count} for text, count in common_unrecoverable
        ],
        "most_common_registered_wrong_candidates": [
            {"candidate": candidate, "count": count}
            for candidate, count in selected_wrong
        ],
        "causal_splits_read": False,
        "sealed_split_read": False,
    }


def main() -> None:
    payload = {
        "schema_version": "phase576_behavior_failure_analysis.v1",
        "phase_id": "Phase576",
        "created_at": now(),
        "status": "complete_behavior_only_no_internal_trace",
        "models": {model: analyze_model(model) for model in MODELS},
        "internal_trace_authorized_models": [],
        "causal_intervention_authorized": False,
        "sealed_split_read": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
