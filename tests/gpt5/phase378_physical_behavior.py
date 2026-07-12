#!/usr/bin/env python3
"""Run frozen physical behavior qualification for two Phase378 mechanisms."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
PROTOCOL = OUT / "phase378_physical_protocol.json"
FREEZE = OUT / "phase378_behavior_execution_freeze.json"
CASES = (
    PHASE371
    / "phase371c_case_bank/sealed/private/phase371c_physical_execution_cases.jsonl"
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


@torch.inference_mode()
def process_model(model: str, batch_size: int) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    freeze = read_json(FREEZE)
    if not protocol["authorization"]["open_two_mechanism_physical_behavior_cases"]:
        raise RuntimeError("Physical behavior not authorized")
    if not freeze["valid"]:
        raise RuntimeError("Behavior freeze invalid")
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
        and row["mechanism_id"] in {"relation_binding", "entity_recency"}
    ]
    if len(cases) != 32:
        raise RuntimeError(f"Expected 32 physical cases for {model}, got {len(cases)}")
    loaded = None
    output_rows = []
    try:
        loaded = load_probe_model(model)
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            buckets[int(case["prompt_token_count"])].append(case)
        batches = [
            bucket[start : start + batch_size]
            for _length, bucket in sorted(buckets.items())
            for start in range(0, len(bucket), batch_size)
        ]
        completed = 0
        for selected in batches:
            generated = generate_batch(
                loaded, selected, int(protocol["behavior_gate"]["max_new_tokens"])
            )
            for case, result in zip(selected, generated, strict=True):
                output_rows.append(
                    {
                        **case,
                        **result,
                        "schema_version": "51.1.0",
                        "phase_id": "Phase378-PhysicalBehavior",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "physical_scope": True,
                    }
                )
            completed += len(selected)
            print(f"[{model}] physical behavior {completed}/32", flush=True)
        counts = Counter(row["mechanism_id"] for row in output_rows)
        correct = Counter(
            row["mechanism_id"]
            for row in output_rows
            if row["strict_behavior_correct"]
        )
        private = OUT / "phase378_behavior/models" / model / "private/phase378_behavior_rows.jsonl"
        write_jsonl(private, output_rows)
        summary = {
            "schema_version": "51.1.0",
            "phase_id": "Phase378-PhysicalBehavior",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "physical_case_count": len(output_rows),
            "strict_correct_case_count": sum(
                row["strict_behavior_correct"] for row in output_rows
            ),
            "batch_size": batch_size,
            "prompt_length_bucket_count": len(buckets),
            "cells": [
                {
                    "mechanism_id": mechanism,
                    "case_count": counts[mechanism],
                    "strict_correct_count": correct[mechanism],
                }
                for mechanism in sorted(counts)
            ],
            "other_mechanism_case_count": 0,
            "valid": len(output_rows) == 32,
        }
        write_json(
            OUT / "phase378_behavior/models" / model / "phase378_behavior_summary.json",
            summary,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    process_model(args.model, args.batch_size)


if __name__ == "__main__":
    main()
