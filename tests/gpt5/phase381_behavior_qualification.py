#!/usr/bin/env python3
"""Run the Phase381 fresh behavior gate for one local model."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
CASES = OUT / "private/phase381_execution_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


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
def process(model: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
    ]
    if len(cases) != 288:
        raise RuntimeError(f"Expected 288 Phase381 cases for {model}, got {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            buckets[int(case["prompt_token_count"])].append(case)
        completed = 0
        for _length, bucket in sorted(buckets.items()):
            for start in range(0, len(bucket), batch_size):
                selected = bucket[start : start + batch_size]
                generated = generate_batch(loaded, selected, max_new_tokens)
                for case, result in zip(selected, generated, strict=True):
                    rows.append(
                        {
                            "schema_version": "54.1.0",
                            "phase_id": "Phase381-Behavior",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "model": model,
                            "blind_case_id": case["blind_case_id"],
                            "anonymous_parallel_group_id": case[
                                "anonymous_parallel_group_id"
                            ],
                            "anonymous_group_id": case["anonymous_group_id"],
                            "family_id": case["family_id"],
                            "mechanism_id": case["mechanism_id"],
                            "semantic_group_id": case["semantic_group_id"],
                            "contrast_condition": case["contrast_condition"],
                            "target": case["target"],
                            "target_aliases": case["target_aliases"],
                            "distractors": case["distractors"],
                            **result,
                        }
                    )
                completed += len(selected)
                print(f"[{model}] Phase381 behavior {completed}/288", flush=True)
        path = OUT / "behavior/private/models" / model / "phase381_behavior_rows.jsonl"
        write_jsonl(path, rows)
        counts = Counter(row["mechanism_id"] for row in rows)
        correct = Counter(
            row["mechanism_id"] for row in rows if row["strict_behavior_correct"]
        )
        summary = {
            "schema_version": "54.1.0",
            "phase_id": "Phase381-Behavior",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "case_count": len(rows),
            "strict_correct_count": sum(
                row["strict_behavior_correct"] for row in rows
            ),
            "cells": [
                {
                    "mechanism_id": mechanism,
                    "case_count": counts[mechanism],
                    "strict_correct_count": correct[mechanism],
                }
                for mechanism in sorted(counts)
            ],
            "valid": len(rows) == 288,
        }
        write_json(OUT / "behavior/models" / model / "complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    print(
        json.dumps(
            process(args.model, args.batch_size, args.max_new_tokens),
            ensure_ascii=False,
            indent=2,
        )
    )
