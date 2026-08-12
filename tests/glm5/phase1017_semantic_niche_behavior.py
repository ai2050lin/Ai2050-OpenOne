#!/usr/bin/env python3
"""Calibrate prompt mode and record Phase1017 natural behavior."""

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

from model_utils import release_model
from phase1014_bf16_precision_confirmation import load_bf16
from phase1017_semantic_niche_protocol import (
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
AMBIGUOUS_STATES = tuple(
    state for state in FACTORIAL_STATES if state.startswith("a")
)


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


def selected_cases(
    model_name: str,
    prompt_mode: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    rows = [row for row in rows if row["state"] in AMBIGUOUS_STATES]
    if scope == "calibration":
        rows = [
            row for row in rows
            if row["split"] == "discovery"
            and int(row["world"]) == 0
        ]
    return rows


def evaluate_cases(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    prompt_mode: str,
    scope: str,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    started = time.time()
    for batch_index, batch in enumerate(
        homogeneous_batches(cases, BATCH_SIZE),
        1,
    ):
        input_ids = torch.tensor(
            [row["input_ids"] for row in batch],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[:, -1, :].float()
        full_predictions = logits.argmax(dim=-1)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
                "schema_version": "phase1017_behavior_row.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "prompt_mode": prompt_mode,
                "scope": scope,
                "record_id": case["record_id"],
                "unit_id": case["unit_id"],
                "word": case["word"],
                "split": case["split"],
                "template": int(case["template"]),
                "output_mode": case["output_mode"],
                "world": int(case["world"]),
                "state": case["state"],
                "branch": int(case["branch"]),
                "lexical": int(case["lexical"]),
                "gold": case["gold"],
                "foil": case["foil"],
                "candidate_margin": float(
                    logits[index, gold_id].item()
                    - logits[index, foil_id].item()
                ),
                "candidate_hit": bool(pair_prediction == gold_id),
                "full_vocabulary_prediction_id": int(
                    full_predictions[index].item()
                ),
                "full_vocabulary_hit": bool(
                    int(full_predictions[index].item()) == gold_id
                ),
                "generated_ids": generated_ids,
                "generated_text": generated_text,
                "generated_first_word": word,
                "generation_first_word_hit": bool(
                    word is not None
                    and word.casefold() == str(case["gold"]).casefold()
                ),
            })
        del (
            output,
            logits,
            full_predictions,
            generated,
            suffixes,
            input_ids,
            attention_mask,
        )
        if batch_index % 16 == 0:
            print(
                f"[behavior] {model_name}/{prompt_mode}/{scope} "
                f"batch={batch_index}",
                flush=True,
            )

    summary = {
        "schema_version": "phase1017_behavior_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "scope": scope,
        "case_count": len(rows),
        "candidate_accuracy": float(np.mean([
            row["candidate_hit"] for row in rows
        ])),
        "full_vocabulary_accuracy": float(np.mean([
            row["full_vocabulary_hit"] for row in rows
        ])),
        "generation_first_word_accuracy": float(np.mean([
            row["generation_first_word_hit"] for row in rows
        ])),
        "by_output_mode": {
            output_mode: {
                "count": len(subset),
                "candidate_accuracy": float(np.mean([
                    row["candidate_hit"] for row in subset
                ])),
                "generation_first_word_accuracy": float(np.mean([
                    row["generation_first_word_hit"] for row in subset
                ])),
            }
            for output_mode in ("semantic",)
            for subset in [[
                row for row in rows
                if row["output_mode"] == output_mode
            ]]
        },
        "by_split": {
            split: {
                "count": len(subset),
                "candidate_accuracy": float(np.mean([
                    row["candidate_hit"] for row in subset
                ])),
                "generation_first_word_accuracy": float(np.mean([
                    row["generation_first_word_hit"] for row in subset
                ])),
            }
            for split in ("discovery", "confirmation")
            for subset in [[row for row in rows if row["split"] == split]]
            if subset
        },
        "elapsed_seconds": time.time() - started,
    }
    return rows, summary


def run_model(model_name: str) -> None:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    output_root = OUT_ROOT / "behavior" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        calibration_summaries = []
        for prompt_mode in PROMPT_MODES:
            cases = selected_cases(
                model_name,
                prompt_mode,
                "calibration",
            )
            rows, summary = evaluate_cases(
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                prompt_mode=prompt_mode,
                scope="calibration",
                cases=cases,
            )
            summary["protocol_digest"] = prereg["protocol_digest"]
            summary["placement"] = placement
            write_jsonl(
                output_root / f"calibration.{prompt_mode}.jsonl",
                rows,
            )
            write_json(
                output_root / f"calibration.{prompt_mode}.summary.json",
                summary,
            )
            calibration_summaries.append(summary)

        ranked = sorted(
            calibration_summaries,
            key=lambda row: (
                row["generation_first_word_accuracy"],
                row["candidate_accuracy"],
                row["full_vocabulary_accuracy"],
                row["prompt_mode"] == "native_chat",
            ),
            reverse=True,
        )
        selected_mode = ranked[0]["prompt_mode"]
        selection = {
            "schema_version": "phase1017_prompt_mode_selection.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "selected_prompt_mode": selected_mode,
            "selection_primary": "generation_first_word_accuracy",
            "selection_secondary": "candidate_accuracy",
            "calibration": {
                row["prompt_mode"]: row for row in calibration_summaries
            },
        }
        write_json(output_root / "selection.json", selection)

        formal_cases = selected_cases(
            model_name,
            selected_mode,
            "formal",
        )
        rows, summary = evaluate_cases(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_name,
            prompt_mode=selected_mode,
            scope="formal",
            cases=formal_cases,
        )
        summary["protocol_digest"] = prereg["protocol_digest"]
        summary["placement"] = placement
        write_jsonl(output_root / "formal.jsonl", rows)
        write_json(output_root / "formal.summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "selected_prompt_mode": selected_mode,
            "calibration_generation_accuracy": (
                ranked[0]["generation_first_word_accuracy"]
            ),
            "formal_generation_accuracy": (
                summary["generation_first_word_accuracy"]
            ),
            "formal_candidate_accuracy": summary["candidate_accuracy"],
            "formal_case_count": summary["case_count"],
        }, indent=2))
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer, device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
