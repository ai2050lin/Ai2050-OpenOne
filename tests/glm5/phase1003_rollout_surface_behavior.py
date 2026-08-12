#!/usr/bin/env python3
"""Behavior and EOS qualification for Phase1003 output surfaces."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_crossparadigm_protocol import (
    MODELS,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1003_rollout_surface_protocol import (
    ROLLOUT_ROOT,
    SURFACES,
)


def case_tensors(
    cases: list[dict[str, Any]], device
) -> tuple[torch.Tensor, torch.Tensor]:
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift: {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def batches(cases: list[dict[str, Any]], batch_size: int):
    groups = defaultdict(list)
    for case in cases:
        groups[(
            case["surface"],
            int(case["template"]),
            int(case["input_token_count"]),
        )].append(case)
    for _, values in sorted(groups.items()):
        values.sort(key=lambda case: case["record_id"])
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]


def strip_eos(
    values: list[int], eos_set: set[int]
) -> tuple[list[int], int | None]:
    for index, token_id in enumerate(values):
        if token_id in eos_set:
            return values[:index], index
    return values, None


def semantic_label(
    ids: list[int], case: dict[str, Any]
) -> str | None:
    step = int(case["semantic_step"])
    if len(ids) <= step:
        return None
    lookup = {
        int(token_id): label
        for label, token_id in case["candidate_token_ids"].items()
    }
    return lookup.get(ids[step])


def generate_rows(
    model,
    tokenizer,
    device,
    cases: list[dict[str, Any]],
    effective_eos: list[int],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    eos_set = set(effective_eos)
    all_batches = list(batches(cases, batch_size))
    for batch_number, batch in enumerate(all_batches, 1):
        input_ids, attention = case_tensors(batch, device)
        prompt_width = input_ids.shape[1]
        max_answer = max(
            len(case["answer_token_ids"]) for case in batch
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=max_answer + 5,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        suffixes = generated.sequences[:, prompt_width:].detach().cpu()
        for index, case in enumerate(batch):
            values = [int(value) for value in suffixes[index].tolist()]
            ids, eos_position = strip_eos(values, eos_set)
            label = semantic_label(ids, case)
            expected = [
                int(value) for value in case["answer_token_ids"]
            ]
            rows.append({
                "schema_version": (
                    "phase1003_rollout_surface_behavior_row.v1"
                ),
                "phase": PHASE,
                "model": case["model"],
                "record_id": case["record_id"],
                "split": case["split"],
                "surface": case["surface"],
                "gold": case["gold"],
                "prediction": label,
                "semantic_correct": label == case["gold"],
                "exact_answer": ids == expected,
                "eos_observed": eos_position is not None,
                "eos_position": eos_position,
                "expected_eos_position": len(expected),
                "eos_at_expected_boundary": (
                    eos_position == len(expected)
                ),
                "generated_ids": ids,
                "generated_text": tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            })
        del input_ids, attention, generated
        print(
            f"[behavior/{cases[0]['model']}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    return rows


def summarize(
    model_name: str,
    rows: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    prereg = read_json(
        ROLLOUT_ROOT / "preregistered_protocol.json"
    )
    cells = {}
    for surface in SURFACES:
        for split in ("discovery", "confirmation"):
            values = [
                row
                for row in rows
                if row["surface"] == surface
                and row["split"] == split
            ]
            item = {
                "n": len(values),
                "semantic_accuracy": float(np.mean([
                    row["semantic_correct"] for row in values
                ])),
                "exact_answer_rate": float(np.mean([
                    row["exact_answer"] for row in values
                ])),
                "eos_observed_rate": float(np.mean([
                    row["eos_observed"] for row in values
                ])),
                "eos_at_expected_boundary_rate": float(np.mean([
                    row["eos_at_expected_boundary"]
                    for row in values
                ])),
            }
            item["behavior_gate"] = (
                item["semantic_accuracy"]
                >= prereg["behavior_semantic_gate"]
                and item["exact_answer_rate"]
                >= prereg["behavior_exact_gate"]
                and item["eos_at_expected_boundary_rate"]
                >= prereg["eos_observed_gate"]
            )
            cells[f"{surface}:{split}"] = item
    surface_gates = {
        surface: (
            cells[f"{surface}:discovery"]["behavior_gate"]
            and cells[f"{surface}:confirmation"]["behavior_gate"]
        )
        for surface in SURFACES
    }
    return {
        "schema_version": (
            "phase1003_rollout_surface_behavior_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "case_count": len(rows),
        "cells": cells,
        "surface_gates": surface_gates,
        "passing_surface_count": sum(surface_gates.values()),
        "elapsed_seconds": elapsed,
    }


def run_model(
    model_name: str, batch_size: int
) -> dict[str, Any]:
    cases = read_jsonl(
        ROLLOUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        effective_eos = eos_ids(model, tokenizer)
        rows = generate_rows(
            model,
            tokenizer,
            device,
            cases,
            effective_eos,
            batch_size,
        )
        summary = summarize(
            model_name, rows, time.time() - started
        )
        root = ROLLOUT_ROOT / "behavior" / model_name
        write_jsonl(root / "rows.jsonl", rows)
        write_json(root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            ROLLOUT_ROOT
            / "behavior"
            / model_name
            / "summary.json"
        )
        if path.exists():
            summaries[model_name] = read_json(path)
    prereg = read_json(
        ROLLOUT_ROOT / "preregistered_protocol.json"
    )
    counts = {
        surface: sum(
            summary["surface_gates"][surface]
            for summary in summaries.values()
        )
        for surface in SURFACES
    }
    payload = {
        "schema_version": (
            "phase1003_rollout_surface_behavior_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_model_surface_counts": counts,
        "cross_model_surface_gates": {
            surface: count >= prereg["cross_model_minimum"]
            for surface, count in counts.items()
        },
    }
    write_json(ROLLOUT_ROOT / "behavior" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model, args.batch_size)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
