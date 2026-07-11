#!/usr/bin/env python3
"""Freeze a baseline-adjusted coarse-block screen without revealing causal heldout results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase348"
SCHEMA_VERSION = "24.0.0"
ROUND_DEFAULT = "adjusted_natural_candidate_block_screen"
OUT = ROOT / "tests/gpt5/result/phase348_adjusted_block_screen"
PHASE345 = ROOT / "tests/gpt5/result/phase345_three_core_protocol/three_core_protocol_qualification"
PHASE346 = ROOT / "tests/gpt5/result/phase346_protocol_repair/three_core_protocol_repair"
PHASE347 = ROOT / "tests/gpt5/result/phase347_three_core_natural_trace/three_core_natural_physical_trace"
TARGETS = ("contiguous_multi_token_answer", "no_morphology_control")
CONTROLS = (
    "explicit_copy_control", "simple_no_source_answer",
    "sentence_past_tense", "direct_fact_control",
)
TASK_SOURCE = {
    "contiguous_multi_token_answer": PHASE346 / "phase346_registered_cases.jsonl",
    "simple_no_source_answer": PHASE346 / "phase346_registered_cases.jsonl",
    "no_morphology_control": PHASE345 / "phase345_registered_cases.jsonl",
    "explicit_copy_control": PHASE345 / "phase345_registered_cases.jsonl",
    "sentence_past_tense": PHASE345 / "phase345_registered_cases.jsonl",
    "direct_fact_control": PHASE345 / "phase345_registered_cases.jsonl",
}


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
    dominant = read_jsonl(PHASE347 / "phase347_dominant_natural_regions.jsonl")
    frozen_blocks = []
    for model in ("qwen3", "glm4", "deepseek7b"):
        for task in TARGETS:
            row = next(value for value in dominant if value["model"] == model and value["mechanism_id"] == task)
            if (row["component"], row["depth_bin"], row["position_role"]) != (
                "attention_output", "late", "answer_start",
            ):
                raise RuntimeError(f"Phase347 exact-node candidate changed: {model}/{task}")
            frozen_blocks.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "target_mechanism_id": task,
                "block_id": "attention_output__late__answer_start",
                "component": "attention_output", "depth_bin": "late",
                "position_role": "answer_start", "source_phase": "Phase347",
                "selection_scope": "cross_task_baseline_adjusted_natural_trace",
                "causal_status": "not_tested",
            })

    rows = []
    for task, path in TASK_SOURCE.items():
        for row in read_jsonl(path):
            if row["mechanism_id"] != task:
                continue
            rows.append({
                **row, "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "source_case_id": row["case_id"],
                "case_id": row["case_id"].replace("phase345_", "phase348_").replace("phase346_", "phase348_"),
                "candidate_role": "target" if task in TARGETS else "matched_control",
                "baseline_only": False, "internal_intervention_allowed": True,
                "single_unit_intervention_allowed": False,
            })
    if len(rows) != 1296 or len({row["case_id"] for row in rows}) != 1296:
        raise RuntimeError(f"Invalid Phase348 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Test two adjusted natural-trace block candidates against spatial and matched-task controls.",
        "registered_case_count": len(rows), "target_tasks": list(TARGETS),
        "matched_control_tasks": list(CONTROLS),
        "candidate_control_map": {
            "contiguous_multi_token_answer": ["explicit_copy_control", "simple_no_source_answer"],
            "no_morphology_control": ["sentence_past_tense", "direct_fact_control"],
        },
        "screen_splits": ["discovery", "calibration"],
        "sealed_splits": ["heldout", "private_heldout"],
        "conditions": ["baseline", "correct_zero", "correct_half", "wrong_depth_zero", "wrong_position_zero"],
        "official_execution_mode": "b1_left_cache0",
        "thresholds": {
            "split_baseline_target_win_rate_min": 0.8,
            "split_zero_mean_phrase_margin_loss_min": 0.1,
            "split_zero_positive_rate_min": 0.65,
            "split_spatial_control_superiority_min": 0.05,
            "split_matched_task_superiority_min": 0.05,
            "split_half_mean_phrase_margin_loss_min": 0.0,
        },
        "stop_rules": [
            "Do not reveal causal heldout outcomes unless discovery and calibration both pass.",
            "Do not enter MCUE unless a target beats spatial and matched-task controls.",
            "This screen cannot establish neuron-level causality or sufficiency.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "frozen_block_count": len(frozen_blocks),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in ("qwen3", "glm4", "deepseek7b")},
        "screen_case_count": sum(row["split"] in {"discovery", "calibration"} for row in rows),
        "sealed_case_count": sum(row["split"] in {"heldout", "private_heldout"} for row in rows),
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase348_registered_cases.jsonl", rows)
    write_jsonl(root / "phase348_frozen_blocks.jsonl", frozen_blocks)
    write_json(root / "phase348_registered_protocol.json", protocol)
    write_json(root / "phase348_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
