#!/usr/bin/env python3
"""Apply frozen Phase338 blocks to the Phase341 six-task denominator."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase338_block_causal_screen import score_cases, wrong_block  # noqa: E402
from phase339_cross_task_boundary_audit import generate_batch  # noqa: E402
from phase341_fresh_causal_boundary_case_bank import (  # noqa: E402
    OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
PHRASE_CONDITIONS = (
    "baseline", "correct_zero", "correct_half", "correct_permutation",
    "wrong_depth_zero", "wrong_position_zero",
)
ROLLOUT_CONDITIONS = ("baseline", "correct_zero", "wrong_depth_zero", "wrong_position_zero")
PHASE338 = ROOT / "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen"


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


def base(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "model": model, "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "family_id": case["family_id"], "mechanism_id": case["mechanism_id"],
        "task_class": case["task_class"], "split": case["split"],
        "template_id": case["template_id"], "target": case["target"],
    }


def run_model(model: str, round_name: str = ROUND_DEFAULT, phrase_batch_size: int = 6) -> dict[str, Any]:
    root = OUT / round_name
    cases = [row for row in read_jsonl(root / "phase341_registered_cases.jsonl") if row["model"] == model]
    frozen = read_jsonl(PHASE338 / "models" / model / "phase338_frozen_heldout_block.jsonl")
    if len(frozen) != 1:
        raise RuntimeError(f"Expected one frozen block for {model}")
    block = frozen[0]
    specs = {
        "baseline": (None, None),
        "correct_zero": (block, "zero"),
        "correct_half": (block, "half"),
        "correct_permutation": (block, "permutation"),
        "wrong_depth_zero": (wrong_block(block, "depth"), "zero"),
        "wrong_position_zero": (wrong_block(block, "position"), "zero"),
    }
    phrase_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    loaded = None
    try:
        loaded = load_probe_model(model)
        for start in range(0, len(cases), phrase_batch_size):
            batch = cases[start:start + phrase_batch_size]
            baseline = score_cases(loaded, batch, None, None)
            for condition in PHRASE_CONDITIONS:
                selected, mode = specs[condition]
                scores = baseline if condition == "baseline" else score_cases(loaded, batch, selected, mode)
                for case, score, baseline_score in zip(batch, scores, baseline, strict=True):
                    valid = all(math.isfinite(float(value)) for value in score.values())
                    base_valid = all(math.isfinite(float(value)) for value in baseline_score.values())
                    phrase_rows.append({
                        **base(case, model), "condition": condition,
                        "frozen_block_id": block["block_id"],
                        "intervened_block_id": selected["block_id"] if selected else None,
                        **{
                            key: round(float(value), 7) if math.isfinite(float(value)) else None
                            for key, value in score.items()
                        },
                        "score_valid": valid,
                        "phrase_margin_loss_vs_baseline": (
                            round(baseline_score["phrase_margin"] - score["phrase_margin"], 7)
                            if valid and base_valid else None
                        ),
                        "rollout_batch_size": 1,
                        "single_unit_causal": False,
                    })
            if (start // phrase_batch_size + 1) % 9 == 0 or start + phrase_batch_size >= len(cases):
                print(f"[{model} phrase] {min(start + phrase_batch_size, len(cases))}/{len(cases)}", flush=True)

        audit_cases = [case for case in cases if case["split"] in {"heldout", "private_heldout"}]
        for index, case in enumerate(audit_cases):
            for condition in ROLLOUT_CONDITIONS:
                selected, mode = specs[condition]
                result = generate_batch(loaded, [case], selected, mode, 16)[0]
                rollout_rows.append({
                    **base(case, model), "condition": condition,
                    "frozen_block_id": block["block_id"],
                    "intervened_block_id": selected["block_id"] if selected else None,
                    **result, "rollout_batch_size": 1, "single_unit_causal": False,
                })
            if (index + 1) % 20 == 0 or index + 1 == len(audit_cases):
                print(f"[{model} rollout] {index + 1}/{len(audit_cases)}", flush=True)
        baseline_rollout = {row["case_id"]: row for row in rollout_rows if row["condition"] == "baseline"}
        for row in rollout_rows:
            baseline_row = baseline_rollout[row["case_id"]]
            row["behavior_lost_vs_baseline"] = bool(
                baseline_row["answer_head_semantic_correct"]
                and not row["answer_head_semantic_correct"]
            )
        model_root = root / "models" / model
        write_jsonl(model_root / "phase341_phrase_rows.jsonl", phrase_rows)
        write_jsonl(model_root / "phase341_rollout_rows.jsonl", rollout_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "frozen_block_id": block["block_id"],
            "registered_case_count": len(cases), "phrase_row_count": len(phrase_rows),
            "rollout_case_count": len(audit_cases), "rollout_row_count": len(rollout_rows),
            "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase_rows),
            "valid": len(cases) == 216 and len(phrase_rows) == 1296 and len(rollout_rows) == 240,
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
