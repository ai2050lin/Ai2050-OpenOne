#!/usr/bin/env python3
"""Run Phase369 natural-answer qualification for one model at a time."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402


PHASE = "Phase369"
SCHEMA_VERSION = "46.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
BASE = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
PREREG = BASE / "raw_topology_preregister"
OUT = BASE / "behavior_qualification"
ALLOWED_SPLITS = {"fresh_discovery", "fresh_calibration"}


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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def contains_alias(text: str, aliases: Iterable[str]) -> bool:
    value = normalize(text)
    return any(
        alias_value and re.search(rf"(?<!\w){re.escape(alias_value)}(?!\w)", value)
        for alias_value in (normalize(alias) for alias in aliases)
    )


def answer_head(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


@torch.inference_mode()
def generate_batch(loaded: Any, cases: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    tokenizer = loaded.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(
            [case["prompt"] for case in cases],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    batch = {key: value.to(loaded.input_device) for key, value in batch.items()}
    width = int(batch["input_ids"].shape[1])
    generated = loaded.model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    suffix = generated[:, width:]
    rows = []
    for index, case in enumerate(cases):
        token_ids = [int(item) for item in suffix[index].tolist()]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        head = answer_head(text)
        target_present = contains_alias(head, case["target_aliases"])
        distractor_present = contains_alias(head, case["distractors"])
        exact_values = {normalize(alias).strip(" .,:;!?'\"") for alias in case["target_aliases"]}
        exact_match = normalize(head).strip(" .,:;!?'\"") in exact_values
        rows.append({
            "generated_text": text,
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "answer_head_text": head,
            "target_present": target_present,
            "distractor_present": distractor_present,
            "exact_answer_match": exact_match,
            "strict_behavior_correct": bool(target_present and not distractor_present),
        })
    return rows


def run_model(
    model: str,
    batch_size: int,
    max_new_tokens: int,
    case_file: Path,
    run_tag: str,
    expected_case_count: int,
) -> dict[str, Any]:
    all_cases = read_jsonl(case_file)
    cases = [
        row for row in all_cases
        if row["private_execution_model"] == model and row["phase369_split"] in ALLOWED_SPLITS
    ]
    if len(cases) != expected_case_count:
        raise RuntimeError(
            f"Expected {expected_case_count} nonphysical cases for {model}, got {len(cases)}"
        )
    if any(row["phase369_split"] == "physical_holdout_sealed" for row in cases):
        raise RuntimeError("Physical holdout must remain sealed during behavior qualification")
    loaded = None
    output_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        length_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            length_buckets[int(case["prompt_token_count"])].append(case)
        work_batches = [
            bucket[start:start + batch_size]
            for _length, bucket in sorted(length_buckets.items())
            for start in range(0, len(bucket), batch_size)
        ]
        completed = 0
        for selected in work_batches:
            generated = generate_batch(loaded, selected, max_new_tokens)
            for case, result in zip(selected, generated, strict=True):
                output_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case["anonymous_parallel_group_id"],
                    "anonymous_group_id": case["anonymous_group_id"],
                    "phase369_split": case["phase369_split"],
                    "family_id": case["family_id"],
                    "mechanism_id": case["mechanism_id"],
                    "semantic_group_id": case["semantic_group_id"],
                    "contrast_condition": case["contrast_condition"],
                    "target": case["target"],
                    "target_aliases": case["target_aliases"],
                    "distractors": case["distractors"],
                    **result,
                })
            completed += len(selected)
            print(f"[{model}] qualified {completed}/{len(cases)}", flush=True)
        counts = Counter((row["phase369_split"], row["mechanism_id"]) for row in output_rows)
        correct_counts = Counter(
            (row["phase369_split"], row["mechanism_id"])
            for row in output_rows if row["strict_behavior_correct"]
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "run_tag": run_tag,
            "nonphysical_case_count": len(cases),
            "physical_holdout_case_count_loaded": 0,
            "strict_correct_case_count": sum(row["strict_behavior_correct"] for row in output_rows),
            "exact_match_case_count": sum(row["exact_answer_match"] for row in output_rows),
            "distractor_present_case_count": sum(row["distractor_present"] for row in output_rows),
            "equal_token_length_bucketed_batching": True,
            "prompt_length_bucket_count": len(length_buckets),
            "max_new_tokens": max_new_tokens,
            "cells": [
                {
                    "split": split,
                    "mechanism_id": mechanism,
                    "case_count": counts[(split, mechanism)],
                    "strict_correct_count": correct_counts[(split, mechanism)],
                }
                for split, mechanism in sorted(counts)
            ],
            "valid": len(output_rows) == expected_case_count,
        }
        run_root = OUT if run_tag == "initial" else OUT / run_tag
        model_root = run_root / "private/models" / model
        write_jsonl(model_root / "phase369_behavior_rows.jsonl", output_rows)
        write_json(model_root / "complete.json", summary)
        public = {key: value for key, value in summary.items() if key != "created_at"}
        write_json(run_root / "models" / model / "complete.json", public)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--case-file",
        type=Path,
        default=PREREG / "private/phase369_execution_cases.jsonl",
    )
    parser.add_argument("--run-tag", default="initial")
    parser.add_argument("--expected-case-count", type=int, default=144)
    args = parser.parse_args()
    print(json.dumps(
        run_model(
            args.model,
            args.batch_size,
            args.max_new_tokens,
            args.case_file,
            args.run_tag,
            args.expected_case_count,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
