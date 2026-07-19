#!/usr/bin/env python3
"""Separate Phase556 natural fruit behavior from ambiguous control taxonomy."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUTPUT = OUT_DIR / "phase556_natural_behavior_diagnostics.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["semantic_correct"]) for row in rows)
    return {"correct": correct, "n": len(rows), "rate": correct / len(rows) if rows else 0.0}


def analyze() -> dict[str, Any]:
    model_reports: dict[str, Any] = {}
    for model in MODELS:
        rows = [
            row for row in read_jsonl(OUT_DIR / f"phase556_{model}_behavior_rows.jsonl")
            if row["case_type"] == "natural_knowledge"
        ]
        split_reports: dict[str, Any] = {}
        for split in ("discovery", "independent_confirmation"):
            split_rows = [row for row in rows if row["split"] == split]
            relation_reports: dict[str, Any] = {}
            for relation in sorted({row["natural_relation"] for row in split_rows}):
                relation_rows = [row for row in split_rows if row["natural_relation"] == relation]
                relation_reports[relation] = {
                    "all_objects": rate(relation_rows),
                    "fruits_only": rate([row for row in relation_rows if row["is_fruit"]]),
                    "controls_only": rate([row for row in relation_rows if not row["is_fruit"]]),
                    "by_surface": {
                        str(surface): rate([row for row in relation_rows if int(row["surface_id"]) == surface])
                        for surface in range(4)
                    },
                }
            object_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in split_rows:
                object_groups[row["object_id"]].append(row)
            split_reports[split] = {
                "all_natural": rate(split_rows),
                "relations": relation_reports,
                "objects": {key: rate(value) for key, value in sorted(object_groups.items())},
            }
        model_reports[model] = {"splits": split_reports}
    payload = {
        "schema_version": "phase556_natural_behavior_diagnostics.v1",
        "phase_id": "Phase556",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_reports": model_reports,
        "diagnostic_findings": {
            "fruit_and_control_taxonomy_are_separately_reported": True,
            "original_preregistered_authorization_is_unchanged": True,
            "failed_relation_is_not_retroactively_authorized": True,
            "natural_contract_requires_future_ambiguity_repair": True,
            "sealed_split_read": False,
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return payload


if __name__ == "__main__":
    analyze()
