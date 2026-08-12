#!/usr/bin/env python3
"""Behavior qualification for Phase1003 structural stress tasks."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_crossparadigm_behavior import (
    natural_rows,
    teacher_forced_rows,
)
from phase1003_crossparadigm_protocol import (
    MODELS,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1003_structural_stress_protocol import STRESS_ROOT, TASKS


def summarize(
    model_name: str,
    cases: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    natural: list[dict[str, Any]],
    effective_eos: list[int],
    elapsed: float,
) -> dict[str, Any]:
    prereg = read_json(STRESS_ROOT / "preregistered_protocol.json")
    thresholds = prereg["thresholds"]
    teacher_by_id = {row["record_id"]: row for row in teacher}
    natural_by_id = {row["record_id"]: row for row in natural}
    cells = {}
    for task in TASKS:
        for split in ("discovery", "confirmation"):
            selected = [
                row
                for row in cases
                if row["task"] == task and row["split"] == split
            ]
            teacher_values = [
                teacher_by_id[row["record_id"]] for row in selected
            ]
            natural_values = [
                natural_by_id[row["record_id"]] for row in selected
            ]
            item = {
                "n": len(selected),
                "teacher_candidate_accuracy": float(np.mean([
                    row["semantic_candidate_correct"]
                    for row in teacher_values
                ])),
                "natural_exact_answer_rate": float(np.mean([
                    row["exact_answer"] for row in natural_values
                ])),
                "natural_semantic_accuracy": float(np.mean([
                    row["semantic_correct"] for row in natural_values
                ])),
            }
            item["behavior_gate"] = (
                item["teacher_candidate_accuracy"]
                >= thresholds["behavior_candidate_accuracy"]
                and item["natural_exact_answer_rate"]
                >= thresholds["behavior_exact_answer_rate"]
            )
            cells[f"{task}:{split}"] = item
    task_gates = {
        task: (
            cells[f"{task}:discovery"]["behavior_gate"]
            and cells[f"{task}:confirmation"]["behavior_gate"]
        )
        for task in TASKS
    }
    return {
        "schema_version": (
            "phase1003_structural_stress_behavior_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "case_count": len(cases),
        "effective_eos_ids": effective_eos,
        "cells": cells,
        "task_gates": task_gates,
        "passing_task_count": sum(task_gates.values()),
        "all_tasks_pass": all(task_gates.values()),
        "thresholds": {
            "candidate_accuracy": thresholds[
                "behavior_candidate_accuracy"
            ],
            "exact_answer_rate": thresholds[
                "behavior_exact_answer_rate"
            ],
        },
        "elapsed_seconds": elapsed,
    }


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    cases = read_jsonl(
        STRESS_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    root = STRESS_ROOT / "behavior" / model_name
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        effective_eos = eos_ids(model, tokenizer)
        teacher = teacher_forced_rows(
            model, device, cases, batch_size
        )
        natural = natural_rows(
            model,
            tokenizer,
            device,
            cases,
            batch_size,
            effective_eos,
        )
        summary = summarize(
            model_name,
            cases,
            teacher,
            natural,
            effective_eos,
            time.time() - started,
        )
        write_jsonl(root / "teacher_rows.jsonl", teacher)
        write_jsonl(root / "natural_rows.jsonl", natural)
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
        path = STRESS_ROOT / "behavior" / model_name / "summary.json"
        if path.exists():
            summaries[model_name] = read_json(path)
    prereg = read_json(STRESS_ROOT / "preregistered_protocol.json")
    minimum = prereg["thresholds"]["cross_model_minimum"]
    task_pass_counts = {
        task: sum(
            summary["task_gates"][task]
            for summary in summaries.values()
        )
        for task in TASKS
    }
    payload = {
        "schema_version": (
            "phase1003_structural_stress_behavior_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "task_pass_counts": task_pass_counts,
        "cross_model_task_gates": {
            task: count >= minimum
            for task, count in task_pass_counts.items()
        },
    }
    write_json(STRESS_ROOT / "behavior" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
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
