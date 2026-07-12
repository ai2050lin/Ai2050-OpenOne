#!/usr/bin/env python3
"""Run Phase392 paired-field behavior qualification sequentially by model."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
CASES = OUT / "protocol/private/phase392_candidate_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
DTYPES = {"qwen3": "float16", "glm4": "float16", "deepseek7b": "bfloat16"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@torch.inference_mode()
def run(model: str, max_new_tokens: int) -> dict[str, Any]:
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    if len(cases) != 80:
        raise RuntimeError(f"Expected 80 Phase392 cases for {model}, got {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if dtype != DTYPES[model]:
            raise RuntimeError(f"Phase392 dtype mismatch for {model}")
        for index, case in enumerate(cases, 1):
            result = generate_batch(loaded, [case], max_new_tokens)[0]
            rows.append(
                {
                    "schema_version": "66.1.0",
                    "phase_id": "Phase392-BehaviorQualification",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "blind_case_id": case["blind_case_id"],
                    "parallel_group_id": case["parallel_group_id"],
                    "group_priority": case["group_priority"],
                    "condition": case["condition"],
                    "target_private": case["target"],
                    **result,
                }
            )
            if index % 20 == 0:
                print(
                    f"[{model}] Phase392 behavior {index}/80 "
                    f"pass={sum(row['strict_behavior_correct'] for row in rows)}",
                    flush=True,
                )
        summary = {
            "schema_version": "66.1.0",
            "phase_id": "Phase392-BehaviorQualification",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "runtime_dtype": dtype,
            "batch_size": 1,
            "max_new_tokens": max_new_tokens,
            "case_count": len(rows),
            "strict_correct_count": sum(row["strict_behavior_correct"] for row in rows),
            "exact_match_count": sum(row["exact_answer_match"] for row in rows),
            "valid": len(rows) == 80,
        }
        write_jsonl(OUT / "behavior/private" / model / "rows.jsonl", rows)
        write_json(OUT / "behavior" / model / "complete.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    run(args.model, args.max_new_tokens)
