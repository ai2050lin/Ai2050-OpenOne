#!/usr/bin/env python3
"""Run the frozen Phase399 four-surface behavior denominator."""

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
from phase399_dynamic_binding_protocol import (  # noqa: E402
    CANDIDATE_GROUPS_PER_SURFACE,
    CONDITIONS,
    MODELS,
    SURFACES,
)


OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
CASES = OUT / "protocol/private/phase399_candidate_cases.jsonl"
EXPECTED_PER_MODEL = len(SURFACES) * CANDIDATE_GROUPS_PER_SURFACE * len(CONDITIONS)
FROZEN_DTYPE = {"qwen3": "float16", "glm4": "float16", "deepseek7b": "bfloat16"}


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
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
    ]
    if len(cases) != EXPECTED_PER_MODEL:
        raise RuntimeError(
            f"Expected {EXPECTED_PER_MODEL} Phase399 cases for {model}, got {len(cases)}"
        )
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPE[model]:
            raise RuntimeError(f"Phase399 dtype mismatch for {model}: {runtime_dtype}")
        for index, case in enumerate(cases, 1):
            result = generate_batch(loaded, [case], max_new_tokens)[0]
            rows.append(
                {
                    "schema_version": "73.1.0",
                    "phase_id": "Phase399-BehaviorQualification",
                    "created_at": now(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "execution_batch_size": 1,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case[
                        "anonymous_parallel_group_id"
                    ],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "group_priority": case["group_priority"],
                    "task_surface_private": case["task_surface_private"],
                    "target_private": case["target"],
                    **result,
                }
            )
            if index % 64 == 0 or index == len(cases):
                strict = sum(row["strict_behavior_correct"] for row in rows)
                print(
                    f"[{model}] Phase399 behavior {index}/{len(cases)} strict={strict}",
                    flush=True,
                )
            if index % 128 == 0:
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
            "schema_version": "73.1.0",
            "phase_id": "Phase399-BehaviorQualification",
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
                for surface in SURFACES
            ],
            "valid": len(rows) == EXPECTED_PER_MODEL,
            "claim_boundary": {
                "factorial_behavior_is_dynamic_binding_mechanism": False,
                "failed_groups_are_deleted": False,
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
