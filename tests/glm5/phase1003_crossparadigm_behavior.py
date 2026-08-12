#!/usr/bin/env python3
"""Qualify the frozen Phase 1003 behavior denominator."""
from __future__ import annotations

import argparse
import gc
import itertools
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
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def formal_cases(model_name: str) -> list[dict[str, Any]]:
    root = OUT_ROOT / "protocol" / model_name
    case_by_id = {
        row["record_id"]: row for row in read_jsonl(root / "cases.jsonl")
    }
    selected_ids = set()
    for domain, split in itertools.product(
        DOMAINS, ("discovery", "confirmation")
    ):
        for pair in read_jsonl(
            root / f"{domain}_{split}_selected_pairs.jsonl"
        ):
            selected_ids.add(pair["arm0_record_id"])
            selected_ids.add(pair["arm1_record_id"])
    return [
        case_by_id[record_id] for record_id in sorted(selected_ids)
    ]


def batches_by_shape(
    rows: list[dict[str, Any]], batch_size: int
):
    groups: dict[
        tuple[str, str, int, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            row["domain"],
            row["split"],
            int(row["template"]),
            int(row["input_token_count"]),
            len(row["answer_token_ids"]),
        )
        groups[key].append(row)
    for key, values in sorted(groups.items()):
        values.sort(key=lambda row: row["record_id"])
        for start in range(0, len(values), batch_size):
            yield key, values[start : start + batch_size]


def strip_eos(
    values: list[int], eos_set: set[int]
) -> tuple[list[int], int | None]:
    for index, token_id in enumerate(values):
        if token_id in eos_set:
            return values[:index], index
    return values, None


def teacher_forced_rows(
    model,
    device,
    cases: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    result = []
    all_batches = list(batches_by_shape(cases, batch_size))
    for batch_number, (_, batch) in enumerate(all_batches, 1):
        answer_len = len(batch[0]["answer_token_ids"])
        sequences = [
            row["input_ids"] + row["answer_token_ids"] for row in batch
        ]
        input_ids = torch.tensor(
            sequences, dtype=torch.long, device=device
        )
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=answer_len + 1,
                return_dict=True,
            )
        step_logits = output.logits[:, :answer_len, :]
        predictions = step_logits.argmax(dim=-1).detach().cpu().tolist()
        semantic_step = int(batch[0]["semantic_step"])
        candidate_order = list(batch[0]["candidate_token_ids"])
        candidate_tensor = torch.tensor(
            [
                batch[0]["candidate_token_ids"][value]
                for value in candidate_order
            ],
            dtype=torch.long,
            device=device,
        )
        semantic_logits = step_logits[:, semantic_step, :].index_select(
            1, candidate_tensor
        )
        candidate_predictions = (
            semantic_logits.argmax(dim=-1).detach().cpu().tolist()
        )
        for index, row in enumerate(batch):
            expected = [int(value) for value in row["answer_token_ids"]]
            predicted = [int(value) for value in predictions[index]]
            step_correct = [
                predicted[step] == expected[step]
                for step in range(answer_len)
            ]
            candidate_prediction = candidate_order[
                int(candidate_predictions[index])
            ]
            result.append({
                "schema_version": (
                    "phase1003_teacher_forced_behavior_row.v1"
                ),
                "phase": PHASE,
                "model": row["model"],
                "record_id": row["record_id"],
                "domain": row["domain"],
                "split": row["split"],
                "template": row["template"],
                "gold": row["gold"],
                "expected_token_ids": expected,
                "predicted_token_ids": predicted,
                "step_correct": step_correct,
                "all_steps_global_argmax": all(step_correct),
                "semantic_candidate_prediction": candidate_prediction,
                "semantic_candidate_correct": (
                    candidate_prediction == row["gold"]
                ),
            })
        del (
            output,
            input_ids,
            attention,
            step_logits,
            semantic_logits,
        )
        if batch_number % 16 == 0:
            print(
                f"[teacher] {batch_number}/{len(all_batches)}",
                flush=True,
            )
    return result


def natural_rows(
    model,
    tokenizer,
    device,
    cases: list[dict[str, Any]],
    batch_size: int,
    effective_eos: list[int],
) -> list[dict[str, Any]]:
    result = []
    eos_set = set(effective_eos)
    all_batches = list(batches_by_shape(cases, batch_size))
    for batch_number, (_, batch) in enumerate(all_batches, 1):
        prompt_len = int(batch[0]["input_token_count"])
        answer_len = len(batch[0]["answer_token_ids"])
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch],
            dtype=torch.long,
            device=device,
        )
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=answer_len + 3,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        suffixes = (
            generated.sequences[:, prompt_len:].detach().cpu().tolist()
        )
        for row, suffix in zip(batch, suffixes):
            before_eos, eos_position = strip_eos(
                [int(value) for value in suffix], eos_set
            )
            semantic_step = int(row["semantic_step"])
            label_by_id = {
                int(token_id): label
                for label, token_id in row[
                    "candidate_token_ids"
                ].items()
            }
            semantic_prediction = (
                label_by_id.get(before_eos[semantic_step])
                if len(before_eos) > semantic_step
                else None
            )
            expected = [
                int(value) for value in row["answer_token_ids"]
            ]
            result.append({
                "schema_version": "phase1003_natural_behavior_row.v1",
                "phase": PHASE,
                "model": row["model"],
                "record_id": row["record_id"],
                "domain": row["domain"],
                "split": row["split"],
                "template": row["template"],
                "gold": row["gold"],
                "suffix_ids": [int(value) for value in suffix],
                "before_eos_ids": before_eos,
                "eos_position": eos_position,
                "eos_seen": eos_position is not None,
                "exact_answer": before_eos == expected,
                "semantic_prediction": semantic_prediction,
                "semantic_correct": semantic_prediction == row["gold"],
                "text": tokenizer.decode(
                    before_eos,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            })
        del generated, input_ids, attention
        if batch_number % 16 == 0:
            print(
                f"[natural] {batch_number}/{len(all_batches)}",
                flush=True,
            )
    return result


def summarize(
    model_name: str,
    cases: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    natural: list[dict[str, Any]],
    effective_eos: list[int],
    elapsed: float,
) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    teacher_by_id = {row["record_id"]: row for row in teacher}
    natural_by_id = {row["record_id"]: row for row in natural}
    cells = {}
    for domain, split in itertools.product(
        DOMAINS, ("discovery", "confirmation")
    ):
        selected = [
            row
            for row in cases
            if row["domain"] == domain and row["split"] == split
        ]
        teacher_values = [
            teacher_by_id[row["record_id"]] for row in selected
        ]
        natural_values = [
            natural_by_id[row["record_id"]] for row in selected
        ]
        item = {
            "n": len(selected),
            "teacher_semantic_candidate_accuracy": float(np.mean([
                row["semantic_candidate_correct"]
                for row in teacher_values
            ])),
            "teacher_full_global_argmax_rate": float(np.mean([
                row["all_steps_global_argmax"]
                for row in teacher_values
            ])),
            "natural_exact_answer_rate": float(np.mean([
                row["exact_answer"] for row in natural_values
            ])),
            "natural_semantic_accuracy": float(np.mean([
                row["semantic_correct"] for row in natural_values
            ])),
            "natural_eos_rate": float(np.mean([
                row["eos_seen"] for row in natural_values
            ])),
        }
        item["candidate_gate"] = (
            item["teacher_semantic_candidate_accuracy"]
            >= thresholds["behavior_candidate_accuracy"]
        )
        item["exact_gate"] = (
            item["natural_exact_answer_rate"]
            >= thresholds["behavior_exact_answer_rate"]
        )
        item["behavior_gate"] = (
            item["candidate_gate"] and item["exact_gate"]
        )
        cells[f"{domain}:{split}"] = item
    domain_gates = {
        domain: (
            cells[f"{domain}:discovery"]["behavior_gate"]
            and cells[f"{domain}:confirmation"]["behavior_gate"]
        )
        for domain in DOMAINS
    }
    return {
        "schema_version": "phase1003_crossparadigm_behavior_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "formal_case_count": len(cases),
        "effective_eos_ids": effective_eos,
        "cells": cells,
        "domain_gates": domain_gates,
        "passing_domain_count": sum(domain_gates.values()),
        "all_domains_pass": all(domain_gates.values()),
        "minimum_two_domains_pass": (
            sum(domain_gates.values())
            >= thresholds["cross_domain_minimum_pass_count"]
        ),
        "thresholds": {
            "candidate_accuracy": thresholds[
                "behavior_candidate_accuracy"
            ],
            "exact_answer_rate": thresholds[
                "behavior_exact_answer_rate"
            ],
        },
        "quantized_8bit": True,
        "elapsed_seconds": elapsed,
    }


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    cases = formal_cases(model_name)
    output_root = OUT_ROOT / "behavior" / model_name
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
        write_jsonl(
            output_root / "teacher_forced_rows.jsonl", teacher
        )
        write_jsonl(output_root / "natural_rows.jsonl", natural)
        write_json(output_root / "summary.json", summary)
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
        path = OUT_ROOT / "behavior" / model_name / "summary.json"
        if path.exists():
            summaries[model_name] = read_json(path)
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    minimum = prereg["primary_thresholds"][
        "cross_model_minimum_pass_count"
    ]
    domain_pass_counts = {
        domain: sum(
            summary["domain_gates"][domain]
            for summary in summaries.values()
        )
        for domain in DOMAINS
    }
    payload = {
        "schema_version": "phase1003_behavior_aggregate.v1",
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "domain_pass_counts": domain_pass_counts,
        "cross_model_domain_gates": {
            domain: count >= minimum
            for domain, count in domain_pass_counts.items()
        },
    }
    payload["cross_model_behavior_pass"] = (
        payload["all_models_complete"]
        and sum(payload["cross_model_domain_gates"].values())
        >= prereg["primary_thresholds"][
            "cross_domain_minimum_pass_count"
        ]
    )
    write_json(OUT_ROOT / "behavior" / "summary.json", payload)
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
