#!/usr/bin/env python3
"""Freeze a stratified natural-input matrix for execution-invariance checks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase342"
SCHEMA_VERSION = "18.0.0"
ROUND_DEFAULT = "copy_relay_execution_invariance"
OUT = ROOT / "tests/gpt5/result/phase342_copy_relay_execution"
PHASE340 = ROOT / "tests/gpt5/result/phase340_cross_task_protocol/fresh_cross_task_protocol_repair"
ITEMS = (0, 5, 13, 16)
MODES = (
    ("b1_left_cache0", 1, "left", False),
    ("b2_left_cache0", 2, "left", False),
    ("b4_left_cache0", 4, "left", False),
    ("b6_left_cache0", 6, "left", False),
    ("b1_left_cache1", 1, "left", True),
    ("b2_left_cache1", 2, "left", True),
    ("b4_left_cache1", 4, "left", True),
    ("b6_left_cache1", 6, "left", True),
    ("b2_right_cache0", 2, "right", False),
    ("b4_right_cache0", 4, "right", False),
    ("b6_right_cache0", 6, "right", False),
)


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


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    source = read_jsonl(PHASE340 / "phase340_registered_cases.jsonl")
    rows = []
    for row in source:
        if row["item_index"] not in ITEMS:
            continue
        rows.append({
            **row, "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "created_at": now(), "phase340_case_id": row["case_id"],
            "case_id": row["case_id"].replace("phase340_", "phase342_exec_", 1),
            "baseline_only": True, "internal_intervention_allowed": False,
        })
    if len(rows) != 216 or len({row["case_id"] for row in rows}) != 216:
        raise RuntimeError(f"Invalid Phase342 execution denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Qualify natural execution paths before copy-relay causal testing.",
        "registered_case_count": len(rows), "case_count_per_model": 72,
        "task_count": 9, "selected_item_indices": list(ITEMS),
        "reference_mode": MODES[0][0],
        "execution_modes": [
            {"mode_id": mode, "batch_size": batch, "padding_side": side, "use_cache": cache}
            for mode, batch, side, cache in MODES
        ],
        "thresholds": {
            "text_invariance_rate_min": 0.99,
            "correctness_invariance_rate_min": 1.0,
            "top_token_invariance_rate_min": 1.0,
            "source_hidden_cosine_min": 0.999,
            "target_first_logit_abs_delta_max": 0.05,
        },
        "claim_boundaries": [
            "Natural forward and generation only; no activation intervention is allowed.",
            "Batch, cache, and padding modes are execution variables, not language mechanisms.",
            "A failed mode cannot be used for later causal behavior evidence.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "mode_count": len(MODES),
        "expected_result_row_count": len(rows) * len(MODES),
        "model_case_count": {
            model: sum(row["model"] == model for row in rows)
            for model in ("qwen3", "glm4", "deepseek7b")
        },
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase342_registered_cases.jsonl", rows)
    write_json(root / "phase342_registered_protocol.json", protocol)
    write_json(root / "phase342_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
