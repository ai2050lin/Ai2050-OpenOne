#!/usr/bin/env python3
"""Run Phase386 behavior qualification on the frozen single-sample path."""

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


PHASE_ROOT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
CASE_FILE = PHASE_ROOT / "protocol/private/phase386_candidate_execution_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
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
def process(model: str, max_new_tokens: int) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(CASE_FILE)
        if row["private_execution_model"] == model
    ]
    expected = 6 * 40 * 4
    if len(cases) != expected:
        raise RuntimeError(f"Expected {expected} Phase386 cases for {model}, got {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPE[model]:
            raise RuntimeError(
                f"Phase386 dtype mismatch for {model}: {runtime_dtype} != {FROZEN_DTYPE[model]}"
            )
        for index, case in enumerate(cases, 1):
            result = generate_batch(loaded, [case], max_new_tokens)[0]
            rows.append(
                {
                    "schema_version": "60.1.0",
                    "phase_id": "Phase386-BehaviorQualification",
                    "created_at": now(),
                    "model": model,
                    "runtime_dtype": runtime_dtype,
                    "execution_batch_size": 1,
                    "output_attentions": False,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case[
                        "anonymous_parallel_group_id"
                    ],
                    "anonymous_group_id": case["anonymous_group_id"],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "family_id_private": case["family_id"],
                    "mechanism_id_private": case["mechanism_id"],
                    "semantic_group_id_private": case["semantic_group_id"],
                    "contrast_condition_private": case["contrast_condition"],
                    "target_private": case["target"],
                    "target_aliases_private": case["target_aliases"],
                    "distractors_private": case["distractors"],
                    **result,
                }
            )
            if index % 20 == 0 or index == len(cases):
                print(
                    f"[{model}] Phase386 behavior {index}/{len(cases)} "
                    f"pass={sum(row['strict_behavior_correct'] for row in rows)}",
                    flush=True,
                )
            gc.collect()

        counts = Counter(row["mechanism_id_private"] for row in rows)
        correct = Counter(
            row["mechanism_id_private"]
            for row in rows
            if row["strict_behavior_correct"]
        )
        exact = Counter(
            row["mechanism_id_private"] for row in rows if row["exact_answer_match"]
        )
        summary = {
            "schema_version": "60.1.0",
            "phase_id": "Phase386-BehaviorQualification",
            "created_at": now(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "execution_batch_size": 1,
            "output_attentions": False,
            "case_count": len(rows),
            "strict_correct_count": sum(row["strict_behavior_correct"] for row in rows),
            "exact_answer_match_count": sum(row["exact_answer_match"] for row in rows),
            "cells": [
                {
                    "mechanism_id": mechanism,
                    "case_count": counts[mechanism],
                    "strict_correct_count": correct[mechanism],
                    "exact_answer_match_count": exact[mechanism],
                }
                for mechanism in sorted(counts)
            ],
            "valid": len(rows) == expected,
            "claim_boundary": {
                "behavior_qualification_is_internal_replay": False,
                "exact_trace_must_requalify_target_decision": True,
            },
        }
        write_jsonl(
            PHASE_ROOT
            / "behavior/private/models"
            / model
            / "phase386_behavior_rows.jsonl",
            rows,
        )
        write_json(PHASE_ROOT / "behavior/models" / model / "complete.json", summary)
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    process(args.model, args.max_new_tokens)
