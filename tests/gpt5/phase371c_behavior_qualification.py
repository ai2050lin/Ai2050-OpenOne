#!/usr/bin/env python3
"""Run frozen nonphysical Phase371C behavior qualification for one model."""

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


CASE_FILE = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/phase371c_case_bank/private/phase371c_nonphysical_execution_cases.jsonl"
OUT = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity/phase371c_behavior_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")
ALLOWED_SPLITS = {"fresh_discovery", "sealed_calibration"}


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
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold().strip())


def contains_alias(text: str, aliases: Iterable[str]) -> bool:
    value = normalize(text)
    return any(
        alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", value)
        for alias in (normalize(item) for item in aliases)
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
            [row["prompt"] for row in cases],
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
    results = []
    for index, case in enumerate(cases):
        token_ids = [int(item) for item in suffix[index].tolist()]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        head = answer_head(text)
        target_present = contains_alias(head, case["target_aliases"])
        distractor_present = contains_alias(head, case["distractors"])
        exact_targets = {normalize(item).strip(" .,:;!?'\"") for item in case["target_aliases"]}
        results.append({
            "generated_text": text,
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "answer_head_text": head,
            "target_present": target_present,
            "distractor_present": distractor_present,
            "exact_answer_match": normalize(head).strip(" .,:;!?'\"") in exact_targets,
            "strict_behavior_correct": bool(target_present and not distractor_present),
        })
    return results


@torch.inference_mode()
def run_model(model: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    all_cases = read_jsonl(CASE_FILE)
    cases = [row for row in all_cases if row["private_execution_model"] == model]
    if len(cases) != 288:
        raise RuntimeError(f"Expected 288 nonphysical Phase371C cases for {model}, got {len(cases)}")
    if any(row["phase371c_split"] not in ALLOWED_SPLITS for row in cases):
        raise RuntimeError("Physical Phase371C case reached behavior execution")
    loaded = None
    output_rows = []
    try:
        loaded = load_probe_model(model)
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            buckets[int(case["prompt_token_count"])].append(case)
        batches = [
            bucket[start:start + batch_size]
            for _length, bucket in sorted(buckets.items())
            for start in range(0, len(bucket), batch_size)
        ]
        completed = 0
        for selected in batches:
            generated = generate_batch(loaded, selected, max_new_tokens)
            for case, result in zip(selected, generated, strict=True):
                output_rows.append({
                    "schema_version": "47.9.0",
                    "phase_id": "Phase371C",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "blind_case_id": case["blind_case_id"],
                    "anonymous_parallel_group_id": case["anonymous_parallel_group_id"],
                    "anonymous_group_id": case["anonymous_group_id"],
                    "anonymous_condition_slot": case["anonymous_condition_slot"],
                    "phase371c_split": case["phase371c_split"],
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
            print(f"[{model}] Phase371C behavior {completed}/{len(cases)}", flush=True)
        counts = Counter((row["phase371c_split"], row["mechanism_id"]) for row in output_rows)
        correct = Counter(
            (row["phase371c_split"], row["mechanism_id"])
            for row in output_rows if row["strict_behavior_correct"]
        )
        summary = {
            "schema_version": "47.9.0",
            "phase_id": "Phase371C",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "nonphysical_case_count": len(cases),
            "physical_case_count_loaded": 0,
            "strict_correct_case_count": sum(row["strict_behavior_correct"] for row in output_rows),
            "exact_match_case_count": sum(row["exact_answer_match"] for row in output_rows),
            "distractor_present_case_count": sum(row["distractor_present"] for row in output_rows),
            "equal_token_length_bucketed_batching": True,
            "prompt_length_bucket_count": len(buckets),
            "max_new_tokens": max_new_tokens,
            "cells": [
                {
                    "split": split,
                    "mechanism_id": mechanism,
                    "case_count": counts[(split, mechanism)],
                    "strict_correct_count": correct[(split, mechanism)],
                }
                for split, mechanism in sorted(counts)
            ],
            "valid": len(output_rows) == 288,
        }
        write_jsonl(OUT / "private/models" / model / "phase371c_behavior_rows.jsonl", output_rows)
        write_json(OUT / "private/models" / model / "complete.json", summary)
        write_json(OUT / "models" / model / "complete.json", {
            key: value for key, value in summary.items() if key != "created_at"
        })
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
    print(json.dumps(run_model(args.model, args.batch_size, args.max_new_tokens), ensure_ascii=False, indent=2))
