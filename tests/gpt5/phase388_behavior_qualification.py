#!/usr/bin/env python3
"""Run fresh Phase388 behavior qualification for one local model."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
CASE_FILE = OUT / "protocol/private/phase388_candidate_execution_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
FROZEN_DTYPE = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@torch.inference_mode()
def run(model: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(CASE_FILE)
        if row["private_execution_model"] == model
    ]
    if len(cases) != 48:
        raise RuntimeError(f"Expected 48 Phase388 cases for {model}, found {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPE[model]:
            raise RuntimeError(
                f"Phase388 dtype mismatch for {model}: {runtime_dtype} != {FROZEN_DTYPE[model]}"
            )
        for start in range(0, len(cases), batch_size):
            selected = cases[start : start + batch_size]
            generated = generate_batch(loaded, selected, max_new_tokens)
            for case, result in zip(selected, generated, strict=True):
                rows.append(
                    {
                        "schema_version": "62.1.0",
                        "phase_id": "Phase388-BehaviorQualification",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "runtime_dtype": runtime_dtype,
                        "blind_case_id": case["blind_case_id"],
                        "parallel_group_id": case["parallel_group_id"],
                        "group_priority": case["group_priority"],
                        "condition": case["condition"],
                        "target": case["target"],
                        "target_aliases": case["target_aliases"],
                        "distractors": case["distractors"],
                        **result,
                    }
                )
            print(
                f"[{model}] Phase388 behavior {len(rows)}/{len(cases)} "
                f"exact={sum(row['exact_answer_match'] for row in rows)}",
                flush=True,
            )
        counts = Counter(row["condition"] for row in rows)
        exact = Counter(row["condition"] for row in rows if row["exact_answer_match"])
        summary = {
            "schema_version": "62.1.0",
            "phase_id": "Phase388-BehaviorQualification",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "runtime_dtype": runtime_dtype,
            "case_count": len(rows),
            "strict_correct_count": sum(row["strict_behavior_correct"] for row in rows),
            "exact_answer_match_count": sum(row["exact_answer_match"] for row in rows),
            "condition_cells": [
                {
                    "condition": condition,
                    "case_count": counts[condition],
                    "exact_answer_match_count": exact[condition],
                }
                for condition in sorted(counts)
            ],
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "valid": len(rows) == 48,
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens)
