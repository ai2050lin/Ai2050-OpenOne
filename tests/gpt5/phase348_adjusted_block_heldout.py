#!/usr/bin/env python3
"""Reveal only the Phase348 candidate that passed discovery and calibration."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase339_cross_task_boundary_audit import generate_batch  # noqa: E402
from phase338_block_causal_screen import wrong_block  # noqa: E402
from phase348_adjusted_block_screen import score_case_b1  # noqa: E402
from phase348_adjusted_block_screen_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODEL = "deepseek7b"
TASKS = ("no_morphology_control", "sentence_past_tense", "direct_fact_control")
CONDITIONS = ("baseline", "correct_zero", "correct_half", "wrong_depth_zero", "wrong_position_zero")
ROLLOUT_CONDITIONS = ("baseline", "correct_zero", "wrong_depth_zero", "wrong_position_zero")


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


def run(round_name: str = ROUND_DEFAULT, max_new_tokens: int = 12) -> dict[str, Any]:
    root = OUT / round_name
    cases = [
        row for row in read_jsonl(root / "phase348_registered_cases.jsonl")
        if row["model"] == MODEL and row["mechanism_id"] in TASKS
        and row["split"] in {"heldout", "private_heldout"}
    ]
    block = {
        "block_id": "attention_output__late__answer_start",
        "component": "attention_output", "depth_bin": "late", "position_role": "answer_start",
    }
    wrong_depth = wrong_block(block, "depth")
    wrong_position = wrong_block(block, "position")
    specs = {
        "baseline": (None, None), "correct_zero": (block, "zero"),
        "correct_half": (block, "half"),
        "wrong_depth_zero": (wrong_depth, "zero"),
        "wrong_position_zero": (wrong_position, "zero"),
    }
    phrase_rows = []
    rollout_rows = []
    loaded = None
    try:
        loaded = load_probe_model(MODEL)
        for index, case in enumerate(cases, 1):
            baseline = score_case_b1(loaded, case, None, None)
            for condition in CONDITIONS:
                selected, mode = specs[condition]
                score = baseline if condition == "baseline" else score_case_b1(loaded, case, selected, mode)
                phrase_rows.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "model": MODEL, "case_id": case["case_id"], "mechanism_id": case["mechanism_id"],
                    "split": case["split"], "template_id": case["template_id"], "condition": condition,
                    "frozen_block_id": block["block_id"],
                    "intervened_block_id": selected["block_id"] if selected else None,
                    **{key: round(value, 7) for key, value in score.items()},
                    "score_valid": all(torch.isfinite(torch.tensor(value)).item() for value in score.values()),
                    "target_wins": score["phrase_margin"] > 0,
                    "phrase_margin_loss_vs_baseline": round(baseline["phrase_margin"] - score["phrase_margin"], 7),
                    "actual_model_batch_size": 1, "use_cache": False, "single_unit_causal": False,
                })
            for condition in ROLLOUT_CONDITIONS:
                selected, mode = specs[condition]
                generated = generate_batch(loaded, [case], selected, mode, max_new_tokens)[0]
                rollout_rows.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "model": MODEL, "case_id": case["case_id"], "mechanism_id": case["mechanism_id"],
                    "split": case["split"], "template_id": case["template_id"], "condition": condition,
                    "frozen_block_id": block["block_id"],
                    "intervened_block_id": selected["block_id"] if selected else None,
                    **generated, "actual_model_batch_size": 1, "use_cache": False,
                    "single_unit_causal": False,
                })
            if index % 15 == 0 or index == len(cases):
                print(f"[{MODEL} heldout] {index}/{len(cases)}", flush=True)
        baseline_rollout = {row["case_id"]: row for row in rollout_rows if row["condition"] == "baseline"}
        for row in rollout_rows:
            base = baseline_rollout[row["case_id"]]
            row["behavior_lost_vs_baseline"] = bool(
                base["answer_head_semantic_correct"] and not row["answer_head_semantic_correct"]
            )
        write_jsonl(root / "models" / MODEL / "phase348_heldout_phrase_rows.jsonl", phrase_rows)
        write_jsonl(root / "models" / MODEL / "phase348_heldout_rollout_rows.jsonl", rollout_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": MODEL, "revealed_candidate": "no_morphology_control",
            "revealed_task_count": len(TASKS), "case_count": len(cases),
            "phrase_row_count": len(phrase_rows), "rollout_row_count": len(rollout_rows),
            "actual_model_batch_size": 1,
            "valid": len(cases) == 63 and len(phrase_rows) == 315 and len(rollout_rows) == 252,
        }
        write_json(root / "models" / MODEL / "phase348_heldout_complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args.round, args.max_new_tokens), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
