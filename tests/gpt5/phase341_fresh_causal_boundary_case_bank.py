#!/usr/bin/env python3
"""Freeze Phase340-qualified tasks for a fresh causal boundary audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase341"
SCHEMA_VERSION = "17.0.0"
ROUND_DEFAULT = "qualified_six_task_causal_boundary"
OUT = ROOT / "tests/gpt5/result/phase341_fresh_causal_boundary"
PHASE340 = ROOT / "tests/gpt5/result/phase340_cross_task_protocol/fresh_cross_task_protocol_repair"
SELECTED_TASKS = (
    "material_relation_binding", "part_relation_binding", "location_relation_binding",
    "identity_copy", "direct_entailment", "answer_only_protocol",
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
        if row["mechanism_id"] not in SELECTED_TASKS:
            continue
        rows.append({
            **row,
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "phase340_case_id": row["case_id"],
            "case_id": row["case_id"].replace("phase340_", "phase341_", 1),
            "baseline_only": False,
            "internal_intervention_allowed": True,
            "block_reselection_allowed": False,
            "layer_shrink_allowed": False,
            "single_unit_intervention_allowed": False,
        })
    if len(rows) != 648 or len({row["case_id"] for row in rows}) != 648:
        raise RuntimeError(f"Invalid Phase341 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Re-audit Phase338 frozen blocks on six independently baseline-qualified tasks.",
        "source_protocol": str(PHASE340 / "phase340_registered_protocol.json"),
        "selected_tasks": list(SELECTED_TASKS), "registered_case_count": len(rows),
        "phrase_conditions": [
            "baseline", "correct_zero", "correct_half", "correct_permutation",
            "wrong_depth_zero", "wrong_position_zero",
        ],
        "rollout_conditions": [
            "baseline", "correct_zero", "wrong_depth_zero", "wrong_position_zero",
        ],
        "rollout_batch_size": 1,
        "thresholds": {
            "baseline_capability_rate_min": 0.8,
            "required_phrase_valid_rate_min": 1.0,
            "correct_behavior_loss_rate_min": 0.5,
            "wrong_control_behavior_loss_rate_max": 0.1,
            "phrase_control_superiority_min": 0.05,
        },
        "claim_boundaries": [
            "Phase338 blocks remain frozen and are not reselected.",
            "All causal rollout behavior uses batch size one after the Phase340 invariance failure.",
            "Layer, channel, and neuron shrinking remain closed until this boundary gate passes.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows),
        "model_case_count": {
            model: sum(row["model"] == model for row in rows)
            for model in ("qwen3", "glm4", "deepseek7b")
        },
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase341_registered_cases.jsonl", rows)
    write_json(root / "phase341_registered_protocol.json", protocol)
    write_json(root / "phase341_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
