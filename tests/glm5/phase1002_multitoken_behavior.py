#!/usr/bin/env python3
"""Phase 1002 full-denominator multi-token behavior qualification."""
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
from phase1002_multitoken_protocol import (
    MODELS,
    OUT_ROOT,
    canonical,
    write_json,
    write_jsonl,
)


PHASE = 1002


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def batches_by_shape(
    rows: list[dict[str, Any]], batch_size: int
):
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], int(row["template"]))].append(row)
    for key, values in sorted(groups.items()):
        values = sorted(values, key=lambda item: item["record_id"])
        for start in range(0, len(values), batch_size):
            yield key, values[start:start + batch_size]


def strip_eos(values: list[int], eos_set: set[int]) -> tuple[list[int], int | None]:
    for index, value in enumerate(values):
        if value in eos_set:
            return values[:index], index
    return values, None


def teacher_forced_rows(
    model,
    device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    result = []
    for batch_number, (key, batch) in enumerate(
        batches_by_shape(rows, batch_size), 1
    ):
        prompt_len = int(batch[0]["input_token_count"])
        answer_len = len(batch[0]["answer_token_ids"])
        sequences = [
            item["input_ids"] + item["answer_token_ids"] for item in batch
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
                return_dict=True,
            )
        step_logits = torch.stack([
            output.logits[:, prompt_len - 1 + step, :]
            for step in range(answer_len)
        ], dim=1)
        global_predictions = step_logits.argmax(dim=-1).detach().cpu()
        semantic_step = int(batch[0]["semantic_step"])
        candidate_ids = batch[0]["candidate_token_ids"]
        candidate_order = list(candidate_ids)
        candidate_tensor = torch.tensor(
            [candidate_ids[color] for color in candidate_order],
            dtype=torch.long,
            device=step_logits.device,
        )
        semantic_logits = step_logits[:, semantic_step, :].index_select(
            1, candidate_tensor
        )
        semantic_predictions = semantic_logits.argmax(dim=-1).detach().cpu()
        for index, item in enumerate(batch):
            expected = [int(value) for value in item["answer_token_ids"]]
            predicted = [
                int(value) for value in global_predictions[index].tolist()
            ]
            step_correct = [
                predicted[step] == expected[step]
                for step in range(answer_len)
            ]
            semantic_prediction = candidate_order[
                int(semantic_predictions[index])
            ]
            result.append({
                "schema_version": "phase1002_teacher_forced_behavior.v1",
                "phase": PHASE,
                "model": item["model"],
                "record_id": item["record_id"],
                "split": item["split"],
                "template": item["template"],
                "gold": item["gold"],
                "semantic_step": semantic_step,
                "expected_token_ids": expected,
                "predicted_token_ids": predicted,
                "step_correct": step_correct,
                "all_steps_global_argmax": all(step_correct),
                "semantic_candidate_prediction": semantic_prediction,
                "semantic_candidate_correct": (
                    semantic_prediction == item["gold"]
                ),
            })
        del output, input_ids, attention, step_logits, semantic_logits
        if batch_number % 32 == 0:
            print(f"[teacher] batch={batch_number} key={key}", flush=True)
    return result


def natural_rows(
    model,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    batch_size: int,
    effective_eos: list[int],
) -> list[dict[str, Any]]:
    result = []
    eos_set = set(effective_eos)
    for batch_number, (key, batch) in enumerate(
        batches_by_shape(rows, batch_size), 1
    ):
        prompt_len = int(batch[0]["input_token_count"])
        answer_len = len(batch[0]["answer_token_ids"])
        input_ids = torch.tensor(
            [item["input_ids"] for item in batch],
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
                max_new_tokens=answer_len + 2,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        suffixes = generated.sequences[:, prompt_len:].detach().cpu().tolist()
        for index, item in enumerate(batch):
            suffix = [int(value) for value in suffixes[index]]
            before_eos, eos_position = strip_eos(suffix, eos_set)
            expected = [int(value) for value in item["answer_token_ids"]]
            color_id_to_label = {
                int(token_id): color
                for color, token_id in item["candidate_token_ids"].items()
            }
            semantic_step = int(item["semantic_step"])
            semantic_prediction = (
                color_id_to_label.get(before_eos[semantic_step])
                if len(before_eos) > semantic_step
                else None
            )
            result.append({
                "schema_version": "phase1002_natural_behavior.v1",
                "phase": PHASE,
                "model": item["model"],
                "record_id": item["record_id"],
                "split": item["split"],
                "template": item["template"],
                "gold": item["gold"],
                "suffix_ids": suffix,
                "before_eos_ids": before_eos,
                "eos_position": eos_position,
                "eos_seen": eos_position is not None,
                "exact_sentence": before_eos == expected,
                "semantic_prediction": semantic_prediction,
                "semantic_correct": semantic_prediction == item["gold"],
                "text": tokenizer.decode(
                    before_eos,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            })
        del generated, input_ids, attention
        if batch_number % 32 == 0:
            print(f"[natural] batch={batch_number} key={key}", flush=True)
    return result


def summarize(
    model_name: str,
    cases: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    natural: list[dict[str, Any]],
    effective_eos: list[int],
    elapsed: float,
) -> dict[str, Any]:
    teacher_by_id = {row["record_id"]: row for row in teacher}
    natural_by_id = {row["record_id"]: row for row in natural}
    split_summary = {}
    for split in ("discovery", "confirmation"):
        split_cases = [row for row in cases if row["split"] == split]
        teacher_values = [teacher_by_id[row["record_id"]] for row in split_cases]
        natural_values = [natural_by_id[row["record_id"]] for row in split_cases]
        answer_len = len(split_cases[0]["answer_token_ids"])
        split_summary[split] = {
            "n": len(split_cases),
            "teacher_semantic_candidate_accuracy": float(np.mean([
                row["semantic_candidate_correct"] for row in teacher_values
            ])),
            "teacher_full_global_argmax_rate": float(np.mean([
                row["all_steps_global_argmax"] for row in teacher_values
            ])),
            "teacher_step_accuracy": {
                str(step): float(np.mean([
                    row["step_correct"][step] for row in teacher_values
                ]))
                for step in range(answer_len)
            },
            "natural_exact_sentence_rate": float(np.mean([
                row["exact_sentence"] for row in natural_values
            ])),
            "natural_semantic_accuracy": float(np.mean([
                row["semantic_correct"] for row in natural_values
            ])),
            "natural_eos_rate": float(np.mean([
                row["eos_seen"] for row in natural_values
            ])),
            "template_exact_sentence_rate": {
                str(template): float(np.mean([
                    row["exact_sentence"] for row in natural_values
                    if int(row["template"]) == template
                ]))
                for template in range(4)
            },
        }
    gate_checks = {
        f"{split}_candidate": (
            split_summary[split]["teacher_semantic_candidate_accuracy"] >= 0.95
        )
        for split in ("discovery", "confirmation")
    }
    gate_checks.update({
        f"{split}_exact_sentence": (
            split_summary[split]["natural_exact_sentence_rate"] >= 0.95
        )
        for split in ("discovery", "confirmation")
    })
    return {
        "schema_version": "phase1002_multitoken_behavior_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "case_count": len(cases),
        "effective_eos_ids": effective_eos,
        "split_summary": split_summary,
        "thresholds": {
            "candidate_accuracy": 0.95,
            "exact_sentence_rate": 0.95,
        },
        "gate_checks": gate_checks,
        "behavior_gate_pass": all(gate_checks.values()),
        "quantized_8bit": True,
        "elapsed_seconds": elapsed,
    }


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    protocol_root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(protocol_root / "cases.jsonl")
    output_root = OUT_ROOT / "behavior" / model_name
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        effective_eos = eos_ids(model, tokenizer)
        teacher = teacher_forced_rows(model, device, cases, batch_size)
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
        write_jsonl(output_root / "teacher_forced_rows.jsonl", teacher)
        write_jsonl(output_root / "natural_rows.jsonl", natural)
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> dict[str, Any]:
    rows = []
    for model_name in MODELS:
        path = OUT_ROOT / "behavior" / model_name / "summary.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "model": model_name,
                "gate_pass": bool(value["behavior_gate_pass"]),
                "discovery_exact_sentence": value["split_summary"][
                    "discovery"
                ]["natural_exact_sentence_rate"],
                "confirmation_exact_sentence": value["split_summary"][
                    "confirmation"
                ]["natural_exact_sentence_rate"],
                "discovery_candidate": value["split_summary"][
                    "discovery"
                ]["teacher_semantic_candidate_accuracy"],
                "confirmation_candidate": value["split_summary"][
                    "confirmation"
                ]["teacher_semantic_candidate_accuracy"],
            })
    payload = {
        "schema_version": "phase1002_behavior_aggregate.v1",
        "phase": PHASE,
        "rows": rows,
        "all_models_complete": len(rows) == len(MODELS),
        "all_models_pass": (
            len(rows) == len(MODELS)
            and all(row["gate_pass"] for row in rows)
        ),
    }
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
        parser.error("--model is required unless --aggregate is used")


if __name__ == "__main__":
    main()
