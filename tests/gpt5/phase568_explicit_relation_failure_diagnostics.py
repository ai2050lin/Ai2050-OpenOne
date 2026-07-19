#!/usr/bin/env python3
"""Describe Phase568 failures without changing the frozen qualification result."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase568_explicit_relation_binding"
SUMMARY_PATH = OUT_DIR / "phase568_behavior_summary.json"
OUTPUT_PATH = OUT_DIR / "phase568_behavior_failure_diagnostics.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
GATE_SPLITS = ("gate_discovery", "gate_confirmation")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def generated_candidate(row: dict[str, Any]) -> str | None:
    text = row["normalized_generated"]
    matches = []
    for value in row["all_candidates"]:
        match = re.search(
            rf"(?<!\w){re.escape(value)}(?!\w)", text, flags=re.IGNORECASE
        )
        if match is not None:
            matches.append((match.start(), value))
    return min(matches)[1] if matches else None


def relation_value(row: dict[str, Any], relation: str, object_index: int) -> str:
    value_index = row["relation_maps"][relation][object_index]
    return row["values"][value_index]


def ordered_object_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record for record in row["fact_records"]
        if record["object_index"] == row["query_object_index"]
    ]


def error_signatures(row: dict[str, Any]) -> dict[str, bool]:
    candidate = generated_candidate(row)
    if candidate is None:
        return {
            "registered_candidate_recovered": False,
            "same_object_other_relation": False,
            "nearest_query_object_fact": False,
            "farthest_query_object_fact": False,
            "last_fact_global": False,
        }
    other_relation_value = relation_value(
        row, row["other_relation"], row["query_object_index"]
    )
    object_records = ordered_object_records(row)
    return {
        "registered_candidate_recovered": True,
        "same_object_other_relation": candidate == other_relation_value,
        "nearest_query_object_fact": bool(object_records and candidate == object_records[-1]["value"]),
        "farthest_query_object_fact": bool(object_records and candidate == object_records[0]["value"]),
        "last_fact_global": candidate == row["fact_records"][-1]["value"],
    }


def cell_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["factorial_cell"]].append(row)
    report = []
    for cell, group in groups.items():
        correct = sum(row["semantic_correct"] for row in group)
        wrong = len(group) - correct
        same_other = sum(
            error_signatures(row)["same_object_other_relation"]
            for row in group if not row["semantic_correct"]
        )
        nearest = sum(
            error_signatures(row)["nearest_query_object_fact"]
            for row in group if not row["semantic_correct"]
        )
        report.append({
            "factorial_cell": cell,
            "n": len(group),
            "correct": correct,
            "wrong": wrong,
            "accuracy": correct / len(group),
            "same_object_other_relation_error_count": same_other,
            "nearest_query_object_fact_error_count": nearest,
        })
    return sorted(report, key=lambda item: (item["accuracy"], item["factorial_cell"]))


def model_report(model: str) -> dict[str, Any]:
    rows = read_jsonl(OUT_DIR / f"phase568_{model}_behavior_rows.jsonl")
    errors = [row for row in rows if not row["semantic_correct"]]
    signature_counts = Counter()
    for row in errors:
        for name, matched in error_signatures(row).items():
            signature_counts[name] += int(matched)
    split_cells = {
        split: cell_report([row for row in rows if row["split"] == split])
        for split in GATE_SPLITS
    }
    top_sets = {
        split: {entry["factorial_cell"] for entry in report[:10]}
        for split, report in split_cells.items()
    }
    example_rows = []
    for row in errors:
        signatures = error_signatures(row)
        if signatures["same_object_other_relation"] and len(example_rows) < 12:
            example_rows.append({
                "case_id": row["case_id"],
                "split": row["split"],
                "factorial_cell": row["factorial_cell"],
                "query_object": row["query_object"],
                "query_relation": row["query_relation"],
                "target": row["target"],
                "generated": row["normalized_generated"],
                "same_object_other_relation_value": relation_value(
                    row, row["other_relation"], row["query_object_index"]
                ),
                "surface_id": row["surface_id"],
                "fact_order": row["fact_order"],
            })
    return {
        "model": model,
        "row_count": len(rows),
        "error_count": len(errors),
        "error_rate": len(errors) / len(rows),
        "signature_counts": dict(sorted(signature_counts.items())),
        "signature_rates_over_errors": {
            key: count / len(errors) if errors else 0.0
            for key, count in sorted(signature_counts.items())
        },
        "gate_split_worst_cells": {
            split: report[:20] for split, report in split_cells.items()
        },
        "top_10_worst_cell_overlap_count": len(
            top_sets["gate_discovery"] & top_sets["gate_confirmation"]
        ),
        "same_object_other_relation_examples": example_rows,
    }


def main() -> None:
    summary = read_json(SUMMARY_PATH)
    payload = {
        "schema_version": "phase568_behavior_failure_diagnostics.v1",
        "phase_id": "Phase568",
        "created_at": now(),
        "frozen_authorized_models": summary["authorized_models"],
        "phase568_result_reclassified": False,
        "models": [model_report(model) for model in MODELS],
        "interpretation_limits": {
            "same_object_other_relation_is_behavioral_error_signature_not_internal_mechanism": True,
            "nearest_fact_match_does_not_prove_attention_copy_edge": True,
            "diagnostic_does_not_change_frozen_gate": True,
            "sealed_split_read": False,
        },
    }
    write_json(OUTPUT_PATH, payload)
    print(json.dumps({
        "authorized_models": payload["frozen_authorized_models"],
        "models": [{
            "model": report["model"],
            "errors": report["error_count"],
            "same_object_other_relation_rate_over_errors": report[
                "signature_rates_over_errors"
            ]["same_object_other_relation"],
            "nearest_query_object_fact_rate_over_errors": report[
                "signature_rates_over_errors"
            ]["nearest_query_object_fact"],
            "top_10_worst_cell_overlap_count": report["top_10_worst_cell_overlap_count"],
        } for report in payload["models"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
