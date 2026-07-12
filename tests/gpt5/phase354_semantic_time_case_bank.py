#!/usr/bin/env python3
"""Freeze Phase353-qualified discovery/calibration cases for semantic-time tracing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase353_family_contracts/family_specific_contract_compiler"
OUT = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace"
ROUND_DEFAULT = "qualified_contract_semantic_time"
PHASE = "Phase354"
SCHEMA_VERSION = "30.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
ALLOWED_SPLITS = {"physical_discovery", "physical_calibration"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def build(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    qualified = {
        (row["family_id"], row["mechanism_id"])
        for row in read_jsonl(SOURCE / "phase353_cross_model_contract_summary.jsonl")
        if row["cross_model_trace_entry"]
    }
    source_cases = read_jsonl(SOURCE / "phase353_registered_cases.jsonl")
    rows = []
    for case in source_cases:
        if (case["family_id"], case["mechanism_id"]) not in qualified or case["split"] not in ALLOWED_SPLITS:
            continue
        rows.append({
            **case,
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "source_phase_id": "Phase353",
            "trace_scope": "discovery_calibration_only",
            "teacher_forced_trace_allowed": True,
            "free_rollout_trace_allowed": True,
            "internal_intervention_allowed": False,
            "single_unit_intervention_allowed": False,
        })
    rows.sort(key=lambda row: row["case_id"])
    root = OUT / round_name
    write_jsonl(root / "phase354_registered_cases.jsonl", rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "denominator": {
            "qualified_contract_count": len(qualified),
            "registered_case_count": len(rows),
            "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS},
            "split_case_count": {split: sum(row["split"] == split for row in rows) for split in sorted(ALLOWED_SPLITS)},
        },
        "qualified_contracts": [f"{family}/{mechanism}" for family, mechanism in sorted(qualified)],
        "physical_heldout_case_count": sum(row["split"] == "physical_heldout" for row in rows),
        "causal_sealed_case_count": sum(row["split"] == "causal_sealed" for row in rows),
        "internal_intervention_executed_count": 0,
        "single_unit_causal_count": 0,
        "language_encoding_mechanism_closed": False,
    }
    write_json(root / "phase354_protocol_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
