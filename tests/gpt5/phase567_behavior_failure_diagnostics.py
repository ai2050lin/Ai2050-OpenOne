#!/usr/bin/env python3
"""Diagnose Phase567 behavior failures without changing the frozen gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase567_multi_relation_binding"
OUTPUT_PATH = OUT_DIR / "phase567_behavior_failure_diagnostics.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rate(group: list[dict[str, Any]]) -> float:
    return sum(row["semantic_correct"] for row in group) / len(group) if group else 0.0


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        value: {
            "n": len(group),
            "correct": sum(row["semantic_correct"] for row in group),
            "accuracy": rate(group),
        }
        for value, group in sorted(groups.items())
    }


def diagnose_model(model: str) -> dict[str, Any]:
    rows = read_jsonl(OUT_DIR / f"phase567_{model}_behavior_rows.jsonl")
    errors = [row for row in rows if not row["semantic_correct"]]
    confusion = Counter(
        (row["target"], row["semantic_event"], row["normalized_generated"].casefold())
        for row in errors
    )
    split_reports = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        triplets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            cells[row["factorial_cell"]].append(row)
            triplets[row["triplet_id"]].append(row)
            worlds[row["anchor_id"]].append(row)
        worst_cells = sorted(
            (
                {
                    "cell": cell,
                    "n": len(group),
                    "correct": sum(row["semantic_correct"] for row in group),
                    "accuracy": rate(group),
                    "target_counts": dict(sorted(Counter(row["target"] for row in group).items())),
                    "error_target_counts": dict(sorted(Counter(
                        row["target"] for row in group if not row["semantic_correct"]
                    ).items())),
                }
                for cell, group in cells.items()
            ),
            key=lambda item: (item["accuracy"], item["cell"]),
        )[:12]
        triplet_rate = sum(
            len(group) == 3 and all(row["semantic_correct"] for row in group)
            for group in triplets.values()
        ) / len(triplets)
        world_rate = sum(
            len(group) == 108 and all(row["semantic_correct"] for row in group)
            for group in worlds.values()
        ) / len(worlds)
        split_reports[split] = {
            "row_count": len(selected),
            "accuracy": rate(selected),
            "error_count": sum(not row["semantic_correct"] for row in selected),
            "triplet_all_correct_rate": triplet_rate,
            "world_all_108_correct_rate": world_rate,
            "world_rate_expected_if_errors_independent": rate(selected) ** 108,
            "worst_cells": worst_cells,
            "target_metrics": grouped(selected, "target"),
        }
    return {
        "model": model,
        "row_count": len(rows),
        "accuracy": rate(rows),
        "error_count": len(errors),
        "error_by_binding": dict(sorted(Counter(str(row["binding"]) for row in errors).items())),
        "error_by_query_object": dict(sorted(Counter(
            str(row["query_object_index"]) for row in errors
        ).items())),
        "error_by_query_relation": dict(sorted(Counter(row["query_relation"] for row in errors).items())),
        "error_by_surface": dict(sorted(Counter(str(row["surface_id"]) for row in errors).items())),
        "error_by_fact_order": dict(sorted(Counter(str(row["fact_order"]) for row in errors).items())),
        "error_by_target": dict(sorted(Counter(row["target"] for row in errors).items())),
        "top_confusions": [
            {"target": key[0], "event": key[1], "generated": key[2], "count": count}
            for key, count in confusion.most_common(30)
        ],
        "split_reports": split_reports,
    }


def diagnose() -> dict[str, Any]:
    payload = {
        "schema_version": "phase567_behavior_failure_diagnostics.v1",
        "phase_id": "Phase567",
        "created_at": now(),
        "frozen_gate_changed": False,
        "model_reports": [diagnose_model(model) for model in MODELS],
    }
    write_json(OUTPUT_PATH, payload)
    print(json.dumps({
        "models": [{
            "model": report["model"],
            "accuracy": report["accuracy"],
            "errors": report["error_count"],
            "behavior_discovery_worst_cells": report["split_reports"]["behavior_discovery"]["worst_cells"][:5],
        } for report in payload["model_reports"]],
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    diagnose()
