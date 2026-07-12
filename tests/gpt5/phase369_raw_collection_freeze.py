#!/usr/bin/env python3
"""Create the label-blind private execution list for Phase369 raw collection."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase369"
SCHEMA_VERSION = "46.0.0"
BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
QUALIFIED = BASE / "behavior_qualification_final_v2"
OUT = BASE / "raw_collection_freeze"
FORBIDDEN = {
    "family_id", "mechanism_id", "semantic_group_id", "contrast_condition",
    "target", "target_aliases", "distractors", "operation_demanded",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    qualification = read_json(QUALIFIED / "phase369_behavior_qualification_final_v2_summary.json")
    if not qualification["authorization"]["fresh_discovery_raw_collection"]:
        raise RuntimeError("Behavior gate does not authorize Phase369 raw collection")
    if qualification["authorization"]["physical_holdout_execution"]:
        raise RuntimeError("Physical holdout authorization must remain false")
    blind_rows = read_jsonl(QUALIFIED / "phase369_qualified_discovery_blind_cases.jsonl")
    private_rows = read_jsonl(QUALIFIED / "private/phase369_qualified_behavior_rows.jsonl")
    model_by_case = {row["blind_case_id"]: row["model"] for row in private_rows}
    execution_rows = [
        {
            **row,
            "private_execution_model": model_by_case[row["blind_case_id"]],
            "semantic_labels_available_to_collector": False,
            "target_specific_competition_available_to_collector": False,
        }
        for row in blind_rows
    ]
    if any(set(row) & FORBIDDEN for row in execution_rows):
        raise RuntimeError("Semantic field leaked into the Phase369 collection execution list")
    counts = Counter(row["private_execution_model"] for row in execution_rows)
    if len(execution_rows) != 336 or set(counts.values()) != {112}:
        raise RuntimeError(f"Invalid collection denominator: {len(execution_rows)} {counts}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "freeze_label_blind_discovery_only_raw_collection_execution_list",
        "case_count": len(execution_rows),
        "case_count_by_model": dict(sorted(counts.items())),
        "semantic_field_count": 0,
        "calibration_case_count": 0,
        "physical_holdout_case_count": 0,
        "generation_time_count": 3,
        "role_scope": ["source", "query", "answer_start", "current_generation"],
        "authorization": {
            "sequential_qwen3_glm4_deepseek7b_collection": True,
            "calibration_collection": False,
            "physical_holdout_collection": False,
        },
    }
    write_jsonl(OUT / "private/phase369_collection_execution_cases.jsonl", execution_rows)
    write_json(OUT / "phase369_raw_collection_freeze_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
