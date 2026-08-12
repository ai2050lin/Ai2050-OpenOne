#!/usr/bin/env python3
"""Calibrate natural prompt mode and behavior for Phase1016."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_model_info, release_model
from phase1014_bf16_precision_confirmation import load_bf16
from phase1016_query_factorial_protocol import (
    FACTORIAL_STATES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROMPT_MODES,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 16
GENERATION_TOKENS = 4


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def homogeneous_batches(
    rows: list[dict[str, Any]],
    size: int,
) -> Iterable[list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[len(row["input_ids"])].append(row)
    for width in sorted(grouped):
        yield from chunks(grouped[width], size)


def first_word(text: str) -> str | None:
    match = re.search(r"[A-Za-z]+", text)
    return None if match is None else match.group(0)


def calibration_cases(
    model_name: str,
    prompt_mode: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    return [
        row for row in rows
        if row["split"] == "discovery"
        and int(row["world_index"]) in {0, 1}
        and row["state"] in FACTORIAL_STATES
    ]


def evaluate_mode(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    prompt_mode: str,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    started = time.time()
    for batch in homogeneous_batches(cases, BATCH_SIZE):
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch],
            dtype=torch.long,
            device=device,
        )
        attention = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[:, -1, :].float()
        full_predictions = logits.argmax(dim=-1)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=GENERATION_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        suffixes = generated[:, input_ids.shape[1]:].detach().cpu()
        for index, case in enumerate(batch):
            gold_id = int(case["candidate_token_ids"][case["gold"]])
            foil_id = int(case["candidate_token_ids"][case["foil"]])
            pair = torch.tensor(
                [gold_id, foil_id],
                dtype=torch.long,
                device=logits.device,
            )
            pair_prediction = int(
                pair[logits[index].index_select(0, pair).argmax()].item()
            )
            generated_ids = [
                int(value) for value in suffixes[index].tolist()
            ]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            word = first_word(generated_text)
            rows.append({
                "schema_version": "phase1016_calibration_row.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "prompt_mode": prompt_mode,
                "family": case["family"],
                "template": int(case["template"]),
                "name_pool": int(case["name_pool"]),
                "world_index": int(case["world_index"]),
                "unit_id": case["unit_id"],
                "record_id": case["record_id"],
                "state": case["state"],
                "gold": case["gold"],
                "foil": case["foil"],
                "gold_id": gold_id,
                "foil_id": foil_id,
                "full_vocabulary_prediction_id": int(
                    full_predictions[index].item()
                ),
                "candidate_prediction_id": pair_prediction,
                "full_vocabulary_hit": bool(
                    int(full_predictions[index].item()) == gold_id
                ),
                "candidate_hit": bool(pair_prediction == gold_id),
                "candidate_margin": float(
                    logits[index, gold_id].item()
                    - logits[index, foil_id].item()
                ),
                "generated_ids": generated_ids,
                "generated_text": generated_text,
                "generated_first_word": word,
                "generation_first_word_hit": bool(
                    word is not None
                    and word.casefold() == str(case["gold"]).casefold()
                ),
            })
        del output, logits, generated, suffixes, input_ids, attention

    summary = {
        "schema_version": "phase1016_calibration_mode_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "n": len(rows),
        "full_vocabulary_accuracy": float(np.mean([
            row["full_vocabulary_hit"] for row in rows
        ])),
        "candidate_accuracy": float(np.mean([
            row["candidate_hit"] for row in rows
        ])),
        "mean_candidate_margin": float(np.mean([
            row["candidate_margin"] for row in rows
        ])),
        "generation_first_word_accuracy": float(np.mean([
            row["generation_first_word_hit"] for row in rows
        ])),
        "by_family": [],
        "elapsed_seconds": time.time() - started,
        "precision": "bf16",
        "selection_data": "discovery_only",
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    for family, group in sorted(grouped.items()):
        summary["by_family"].append({
            "family": family,
            "n": len(group),
            "full_vocabulary_accuracy": float(np.mean([
                row["full_vocabulary_hit"] for row in group
            ])),
            "candidate_accuracy": float(np.mean([
                row["candidate_hit"] for row in group
            ])),
            "mean_candidate_margin": float(np.mean([
                row["candidate_margin"] for row in group
            ])),
            "generation_first_word_accuracy": float(np.mean([
                row["generation_first_word_hit"] for row in group
            ])),
        })
    return rows, summary


def selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["generation_first_word_accuracy"]),
        float(summary["candidate_accuracy"]),
        float(summary["mean_candidate_margin"]),
        float(summary["full_vocabulary_accuracy"]),
        float(summary["prompt_mode"] == "native_chat"),
    )


def run_model(model_name: str) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    if int(prereg["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1016 protocol revision drift")
    model = tokenizer = device = None
    started = time.time()
    summaries = []
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        info = get_model_info(model, model_name)
        output_root = OUT_ROOT / "behavior_calibration" / model_name
        output_root.mkdir(parents=True, exist_ok=True)
        for prompt_mode in PROMPT_MODES:
            cases = calibration_cases(model_name, prompt_mode)
            rows, summary = evaluate_mode(
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                prompt_mode=prompt_mode,
                cases=cases,
            )
            write_jsonl(
                output_root / f"rows.{prompt_mode}.jsonl",
                rows,
            )
            write_json(
                output_root / f"summary.{prompt_mode}.json",
                summary,
            )
            summaries.append(summary)
            print(
                f"[calibration] {model_name}/{prompt_mode} "
                f"vocab={summary['full_vocabulary_accuracy']:.3f} "
                f"pair={summary['candidate_accuracy']:.3f} "
                f"rollout={summary['generation_first_word_accuracy']:.3f}",
                flush=True,
            )
        selected = max(summaries, key=selection_key)
        selection = {
            "schema_version": "phase1016_prompt_mode_selection.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "selected_prompt_mode": selected["prompt_mode"],
            "selection_key": list(selection_key(selected)),
            "selection_rule": (
                "max(generation_first_word_accuracy, candidate_accuracy, "
                "mean_candidate_margin, full_vocab_accuracy, "
                "native_chat_tie)"
            ),
            "confirmation_data_used": False,
            "mode_summaries": summaries,
            "placement": placement,
            "model_info": {
                "layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "intermediate_size": int(info.intermediate_size),
                "model_class": info.model_class,
            },
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "selection.json", selection)
        print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
        return selection
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
