#!/usr/bin/env python3
"""Run the frozen Phase394 formal binding behavior denominator by model."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase394_binding_separation"
CASES = OUT / "protocol/private/phase394_candidate_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
EXPECTED_PER_MODEL = 3 * 24 * 4
FROZEN_DTYPE = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


@torch.inference_mode()
def run(model: str, max_new_tokens: int) -> dict[str, Any]:
    cases = [
        row for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
    ]
    if len(cases) != EXPECTED_PER_MODEL:
        raise RuntimeError(
            f"Expected {EXPECTED_PER_MODEL} Phase394 cases for {model}, got {len(cases)}"
        )
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPE[model]:
            raise RuntimeError(
                f"Phase394 dtype mismatch for {model}: {runtime_dtype} != {FROZEN_DTYPE[model]}"
            )
        for index, case in enumerate(cases, 1):
            result = generate_batch(loaded, [case], max_new_tokens)[0]
            rows.append(
                {
                    "schema_version": "68.1.0",
                    "phase_id": "Phase394-BehaviorQualification",
                    "created_at": now(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "execution_batch_size": 1,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case["anonymous_parallel_group_id"],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "group_priority": case["group_priority"],
                    "task_surface_private": case["task_surface_private"],
                    "contrast_condition_private": case["contrast_condition"],
                    "target_private": case["target"],
                    **result,
                }
            )
            if index % 24 == 0 or index == len(cases):
                print(
                    f"[{model}] Phase394 behavior {index}/{len(cases)} "
                    f"strict={sum(row['strict_behavior_correct'] for row in rows)}",
                    flush=True,
                )
            if index % 48 == 0:
                gc.collect()

        counts = Counter(row["task_surface_private"] for row in rows)
        correct = Counter(
            row["task_surface_private"]
            for row in rows
            if row["strict_behavior_correct"]
        )
        exact = Counter(
            row["task_surface_private"]
            for row in rows
            if row["exact_answer_match"]
        )
        summary = {
            "schema_version": "68.1.0",
            "phase_id": "Phase394-BehaviorQualification",
            "created_at": now(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "execution_batch_size": 1,
            "max_new_tokens": max_new_tokens,
            "case_count": len(rows),
            "strict_correct_count": sum(row["strict_behavior_correct"] for row in rows),
            "exact_answer_match_count": sum(row["exact_answer_match"] for row in rows),
            "surfaces": [
                {
                    "task_surface": surface,
                    "case_count": counts[surface],
                    "strict_correct_count": correct[surface],
                    "exact_answer_match_count": exact[surface],
                }
                for surface in sorted(counts)
            ],
            "valid": len(rows) == EXPECTED_PER_MODEL,
            "claim_boundary": {
                "formal_pointer_behavior_is_binding_mechanism": False,
                "formal_pointer_behavior_is_natural_language_transfer": False,
                "failed_cases_selectively_replaced": False,
            },
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
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    run(args.model, args.max_new_tokens)
