#!/usr/bin/env python3
"""Run one model on the Phase381 behavior-only target expansion."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import generate_batch  # noqa: E402
from phase381_joint_state_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation/target_expansion"
CASES = OUT / "private/phase381x_execution_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


@torch.inference_mode()
def process(model: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
    ]
    if len(cases) != 96:
        raise RuntimeError(f"Expected 96 expansion cases for {model}, got {len(cases)}")
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
                            "schema_version": "54.2.2",
                            "phase_id": "Phase381-TargetExpansionBehavior",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "model": model,
                            "blind_case_id": case["blind_case_id"],
                            "anonymous_parallel_group_id": case[
                                "anonymous_parallel_group_id"
                            ],
                            "anonymous_group_id": case["anonymous_group_id"],
                            "mechanism_id": "target_vs_wrong",
                            "semantic_group_id": case["semantic_group_id"],
                            "contrast_condition": case["contrast_condition"],
                            "target": case["target"],
                            "target_aliases": case["target_aliases"],
                            "distractors": case["distractors"],
                            **result,
                        }
                    )
                completed += len(selected)
                print(f"[{model}] Phase381 target expansion {completed}/96", flush=True)
        write_jsonl(
            OUT / "behavior/private/models" / model / "rows.jsonl",
            rows,
        )
        summary = {
            "schema_version": "54.2.2",
            "phase_id": "Phase381-TargetExpansionBehavior",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "case_count": len(rows),
            "strict_correct_count": sum(
                row["strict_behavior_correct"] for row in rows
            ),
            "valid": len(rows) == 96,
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
