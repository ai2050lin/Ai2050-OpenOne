#!/usr/bin/env python3
"""Run Phase401 semantic-span behavior under a frozen execution shape."""

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
from phase401_behavior_protocol import (  # noqa: E402
    CANDIDATE_GROUPS_PER_SURFACE,
    CONDITIONS,
    MODELS,
    OUT,
    SURFACES,
)
from phase401_local_edge_protocol import FROZEN_DTYPES  # noqa: E402


FORMAL_CASES = OUT / "protocol/private/phase401_candidate_cases.jsonl"
PILOT_CASES = OUT / "protocol/private/phase401_batch_pilot_cases.jsonl"
EXPECTED_FORMAL_PER_MODEL = len(SURFACES) * CANDIDATE_GROUPS_PER_SURFACE * len(CONDITIONS)
EXPECTED_PILOT_PER_MODEL = len(SURFACES) * len(CONDITIONS)


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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold().strip())


def alias_pattern(alias: str) -> re.Pattern[str]:
    value = normalize(alias)
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)")


def alias_count(text: str, aliases: Iterable[str]) -> int:
    value = normalize(text)
    spans: set[tuple[int, int]] = set()
    for alias in aliases:
        if not normalize(alias):
            continue
        spans.update(match.span() for match in alias_pattern(alias).finditer(value))
    return len(spans)


def contains_target(text: str, aliases: Iterable[str]) -> bool:
    return alias_count(text, aliases) > 0


def eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def stop_split(token_ids: list[int], tokenizer: Any) -> tuple[list[int], int | None, str]:
    stops = eos_ids(tokenizer)
    stop_step = next((index for index, token in enumerate(token_ids) if token in stops), None)
    if stop_step is None:
        return token_ids, None, "generation_limit"
    return token_ids[:stop_step], stop_step, "eos"


def semantic_span(
    tokenizer: Any,
    pre_stop_ids: list[int],
    aliases: list[str],
) -> tuple[int | None, int | None]:
    completion = next(
        (
            index
            for index in range(len(pre_stop_ids))
            if contains_target(tokenizer.decode(pre_stop_ids[: index + 1]), aliases)
        ),
        None,
    )
    if completion is None:
        return None, None
    starts = [
        start
        for start in range(completion + 1)
        if contains_target(tokenizer.decode(pre_stop_ids[start : completion + 1]), aliases)
    ]
    if not starts:
        raise RuntimeError("Phase401 target completion found without a target span")
    return max(starts), completion


def parse_generated(
    tokenizer: Any,
    case: dict[str, Any],
    raw_token_ids: list[int],
) -> dict[str, Any]:
    pre_stop_ids, stop_step, stop_kind = stop_split(raw_token_ids, tokenizer)
    text = tokenizer.decode(pre_stop_ids, skip_special_tokens=True)
    start, completion = semantic_span(tokenizer, pre_stop_ids, case["target_aliases"])
    target_occurrences = alias_count(text, case["target_aliases"])
    distractor_occurrences = alias_count(text, case["distractors"])
    resolved = start is not None and completion is not None
    semantic_correct = (
        resolved and target_occurrences >= 1 and distractor_occurrences == 0
    )
    prefix_ids = pre_stop_ids[:start] if start is not None else pre_stop_ids
    answer_ids = (
        pre_stop_ids[start : completion + 1]
        if start is not None and completion is not None
        else []
    )
    suffix_ids = pre_stop_ids[completion + 1 :] if completion is not None else []
    exact_targets = {normalize(alias).strip(" .,:;!?'\"") for alias in case["target_aliases"]}
    return {
        "generated_text_before_stop": text,
        "generated_token_ids_raw": raw_token_ids,
        "generated_token_ids_before_stop": pre_stop_ids,
        "generated_token_count_before_stop": len(pre_stop_ids),
        "effective_generated_token_ids": (
            raw_token_ids[: stop_step + 1] if stop_step is not None else raw_token_ids
        ),
        "semantic_start_step": start,
        "semantic_completion_step": completion,
        "semantic_span_resolved": resolved,
        "semantic_correct": semantic_correct,
        "target_occurrence_count": target_occurrences,
        "target_repetition_count": max(target_occurrences - 1, 0),
        "distractor_occurrence_count": distractor_occurrences,
        "format_prefix_token_ids": prefix_ids,
        "semantic_answer_token_ids": answer_ids,
        "format_suffix_token_ids": suffix_ids,
        "format_prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
        "semantic_answer_text": tokenizer.decode(answer_ids, skip_special_tokens=True),
        "format_suffix_text": tokenizer.decode(suffix_ids, skip_special_tokens=True),
        "stop_step": stop_step,
        "stop_kind": stop_kind,
        "stop_observed": stop_step is not None,
        "post_semantic_token_available": (
            completion is not None
            and (completion + 1 < len(pre_stop_ids) or stop_step is not None)
        ),
        "exact_format_match": normalize(text).strip(" .,:;!?'\"") in exact_targets,
    }


@torch.inference_mode()
def generate_batch(
    loaded: Any,
    cases: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = loaded.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(
            [row["prompt"] for row in cases],
            add_special_tokens=False,
            padding=len(cases) > 1,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    batch = {key: value.to(loaded.input_device) for key, value in batch.items()}
    width = int(batch["input_ids"].shape[1])
    unpadded = [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()]
    if len(cases) == 1 and unpadded[0] != width:
        raise RuntimeError("Phase401 batch=1 execution unexpectedly contains padding")
    generated = loaded.model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    suffix = generated[:, width:]
    parsed = [
        parse_generated(
            tokenizer,
            case,
            [int(item) for item in suffix[index].tolist()],
        )
        for index, case in enumerate(cases)
    ]
    return parsed, {
        "batch_size": len(cases),
        "padded_width": width,
        "unpadded_prompt_lengths": unpadded,
        "padding_side": "left",
        "formal_single_case_unpadded": len(cases) != 1 or unpadded[0] == width,
    }


@torch.inference_mode()
def run(model: str, mode: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    if mode == "formal" and batch_size != 1:
        raise ValueError("Phase401 formal behavior is frozen to batch_size=1")
    source = FORMAL_CASES if mode == "formal" else PILOT_CASES
    expected = EXPECTED_FORMAL_PER_MODEL if mode == "formal" else EXPECTED_PILOT_PER_MODEL
    cases = [row for row in read_jsonl(source) if row["private_execution_model"] == model]
    if len(cases) != expected:
        raise RuntimeError(f"Expected {expected} Phase401 {mode} cases for {model}, got {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
    execution_audits: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        runtime_dtype = str(next(loaded.model.parameters()).dtype).replace("torch.", "")
        if runtime_dtype != FROZEN_DTYPES[model]:
            raise RuntimeError(f"Phase401 dtype mismatch for {model}: {runtime_dtype}")
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
            generated, execution = generate_batch(loaded, selected, max_new_tokens)
            execution_audits.append(execution)
            for case, result in zip(selected, generated, strict=True):
                rows.append(
                    {
                        "schema_version": "75.2.0",
                        "phase_id": "Phase401-BehaviorQualification",
                        "created_at": now(),
                        "model": model,
                        "mode": mode,
                        "runtime_dtype": runtime_dtype,
                        "requested_batch_size": batch_size,
                        "actual_batch_size": len(selected),
                        "attention_implementation": "eager",
                        "use_cache": True,
                        "blind_case_id": case["blind_case_id"],
                        "anonymous_parallel_group_id": case[
                            "anonymous_parallel_group_id"
                        ],
                        "anonymous_condition_slot": case["anonymous_condition_slot"],
                        "candidate_split_private": case["candidate_split_private"],
                        "selection_priority_private": case[
                            "selection_priority_private"
                        ],
                        "group_priority": case["group_priority"],
                        "task_surface_private": case["task_surface_private"],
                        "target_private": case["target"],
                        "prompt_token_count": case["prompt_token_count"],
                        **result,
                    }
                )
            completed += len(selected)
            if completed % 128 == 0 or completed == len(cases):
                correct = sum(row["semantic_correct"] for row in rows)
                exact = sum(row["exact_format_match"] for row in rows)
                print(
                    f"[{model}/{mode}/b{batch_size}] {completed}/{len(cases)} "
                    f"semantic={correct} exact={exact}",
                    flush=True,
                )
            if completed % 256 == 0:
                gc.collect()

        counts = Counter(row["task_surface_private"] for row in rows)
        correct = Counter(
            row["task_surface_private"] for row in rows if row["semantic_correct"]
        )
        resolved = Counter(
            row["task_surface_private"]
            for row in rows
            if row["semantic_span_resolved"]
        )
        exact = Counter(
            row["task_surface_private"] for row in rows if row["exact_format_match"]
        )
        payload = {
            "schema_version": "75.2.0",
            "phase_id": "Phase401-BehaviorQualification",
            "created_at": now(),
            "model": model,
            "mode": mode,
            "runtime_dtype": runtime_dtype,
            "requested_batch_size": batch_size,
            "actual_batch_sizes": sorted({row["actual_batch_size"] for row in rows}),
            "prompt_length_bucket_count": len(buckets),
            "max_new_tokens": max_new_tokens,
            "case_count": len(rows),
            "semantic_correct_count": sum(row["semantic_correct"] for row in rows),
            "semantic_span_resolved_count": sum(
                row["semantic_span_resolved"] for row in rows
            ),
            "exact_format_match_count": sum(row["exact_format_match"] for row in rows),
            "stop_observed_count": sum(row["stop_observed"] for row in rows),
            "unpadding_contract_pass": all(
                item["formal_single_case_unpadded"] for item in execution_audits
            ),
            "surfaces": [
                {
                    "task_surface": surface,
                    "case_count": counts[surface],
                    "semantic_correct_count": correct[surface],
                    "semantic_span_resolved_count": resolved[surface],
                    "exact_format_match_count": exact[surface],
                }
                for surface in SURFACES
            ],
            "valid": len(rows) == expected
            and all(item["formal_single_case_unpadded"] for item in execution_audits),
            "claim_boundary": {
                "semantic_behavior_is_a_local_edge": False,
                "format_mismatch_is_semantic_failure": False,
                "failed_groups_are_deleted": False,
            },
        }
        root = (
            OUT / "behavior/formal" / model
            if mode == "formal"
            else OUT / "behavior/batch_pilot" / f"batch_{batch_size}" / model
        )
        private_root = (
            OUT / "behavior/formal/private" / model
            if mode == "formal"
            else OUT
            / "behavior/batch_pilot/private"
            / f"batch_{batch_size}"
            / model
        )
        write_jsonl(private_root / "rows.jsonl", rows)
        write_json(root / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("formal", "pilot"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    run(args.model, args.mode, args.batch_size, args.max_new_tokens)
