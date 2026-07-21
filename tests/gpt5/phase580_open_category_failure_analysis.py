#!/usr/bin/env python3
"""Describe Phase580 contract failures without changing the frozen gate."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import phase580_open_category_protocol as protocol


OUTPUT = protocol.OUT_DIR / "phase580_open_category_failure_analysis.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_rows(model: str) -> Iterator[dict[str, Any]]:
    path = protocol.OUT_DIR / f"phase580_{model}_open_category_behavior_rows.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyze_model(model: str) -> dict[str, Any]:
    rows = [row for row in iter_rows(model) if row["execution_repeat"] == "noop1"]
    by_surface: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_surface[(row["split"], int(row["surface_id"]))].append(row)
    surface_metrics = {}
    for (split, surface_id), values in sorted(by_surface.items()):
        surface_metrics[f"{split}:surface{surface_id:02d}"] = {
            "case_count": len(values),
            "semantic_accuracy": sum(row["semantic_correct"] for row in values)
            / len(values),
            "event_counts": dict(Counter(row["semantic_event"] for row in values)),
        }
    wrong_registered = [
        row
        for row in rows
        if row["selected_category"] is not None and not row["semantic_correct"]
    ]
    unrecoverable = [row for row in rows if row["selected_category"] is None]
    accuracies = [value["semantic_accuracy"] for value in surface_metrics.values()]
    return {
        "case_count": len(rows),
        "semantic_accuracy": sum(row["semantic_correct"] for row in rows) / len(rows),
        "registered_wrong_category_count": len(wrong_registered),
        "unrecoverable_count": len(unrecoverable),
        "surface_accuracy_minimum": min(accuracies),
        "surface_accuracy_maximum": max(accuracies),
        "surface_accuracy_range": max(accuracies) - min(accuracies),
        "surface_metrics": surface_metrics,
        "most_common_unrecoverable_outputs": [
            {"text": text, "count": count}
            for text, count in Counter(
                row["normalized_generated"] for row in unrecoverable
            ).most_common(40)
        ],
        "sealed_split_read": False,
    }


def main() -> None:
    models = {model: analyze_model(model) for model in protocol.MODELS}
    payload = {
        "schema_version": "phase580_open_category_failure_analysis.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "complete_behavior_contract_failure",
        "models": models,
        "cross_model_registered_wrong_category_count": sum(
            item["registered_wrong_category_count"] for item in models.values()
        ),
        "observed_facts": {
            "all_models_have_surface_accuracy_range_at_least": min(
                item["surface_accuracy_range"] for item in models.values()
            ),
            "all_errors_are_outside_registered_four_category_vocabulary": all(
                item["registered_wrong_category_count"] == 0
                for item in models.values()
            ),
            "behavior_gate_authorizes_internal_trace": False,
        },
        "causal_intervention_authorized": False,
        "sealed_split_read": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cross_model_registered_wrong_category_count": payload[
                    "cross_model_registered_wrong_category_count"
                ],
                "observed_facts": payload["observed_facts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
