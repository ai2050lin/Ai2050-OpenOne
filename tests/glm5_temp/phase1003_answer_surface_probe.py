#!/usr/bin/env python3
"""Compare preregistered answer surfaces on calibration-only examples."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_crossparadigm_protocol import (
    ANSWER_PREFIX,
    CALIBRATION_NAMES,
    DOMAINS,
    MODELS,
    OUT_ROOT,
    render_user_prompt,
    write_json,
    write_jsonl,
)
from phase548_shared_attention_compute_protocol import render_chat


SURFACES = {
    "one_word": {
        "instruction": (
            "Answer with only the lowercase value. Do not add punctuation "
            "or any other words."
        ),
        "answer": lambda value: value,
    },
    "answer_label": {
        "instruction": (
            "Answer exactly in this form: Answer: [value] Replace [value] "
            "with the lowercase answer."
        ),
        "answer": lambda value: f"Answer: {value}",
    },
    "value_sentence": {
        "instruction": (
            "Answer exactly in this form: The value is [value]. Replace "
            "[value] with the lowercase answer."
        ),
        "answer": lambda value: f"The value is {value}.",
    },
    "lower_sentence": {
        "instruction": (
            "Answer exactly in this lowercase form: the answer is [value]. "
            "Replace [value] with the lowercase answer."
        ),
        "answer": lambda value: f"the answer is {value}.",
    },
}


def base_prompt(
    template: int,
    domain: str,
    entity0: str,
    value0: str,
    entity1: str,
    value1: str,
    query: str,
) -> str:
    current = render_user_prompt(
        template,
        domain,
        entity0,
        value0,
        entity1,
        value1,
        query,
    )
    return current.rsplit("\n", 1)[0]


def strip_eos(values: list[int], eos_set: set[int]) -> list[int]:
    for index, token_id in enumerate(values):
        if token_id in eos_set:
            return values[:index]
    return values


def run_model(model_name: str) -> dict:
    model = tokenizer = None
    rows = []
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        effective_eos = eos_ids(model, tokenizer)
        eos_set = set(effective_eos)
        cases = []
        for surface_name, surface in SURFACES.items():
            for domain, values in DOMAINS.items():
                for template, query_role in itertools.product(
                    range(4), (0, 1)
                ):
                    entity0, entity1 = CALIBRATION_NAMES[:2]
                    value0, value1 = values[:2]
                    query = (entity0, entity1)[query_role]
                    gold = (value0, value1)[query_role]
                    raw = (
                        base_prompt(
                            template,
                            domain,
                            entity0,
                            value0,
                            entity1,
                            value1,
                            query,
                        )
                        + "\n"
                        + surface["instruction"]
                    )
                    rendered = render_chat(tokenizer, model_name, raw)
                    expected_text = (
                        ANSWER_PREFIX[model_name]
                        + surface["answer"](gold)
                    )
                    cases.append({
                        "surface": surface_name,
                        "domain": domain,
                        "template": template,
                        "query_role": query_role,
                        "gold": gold,
                        "input_ids": tokenizer.encode(
                            rendered, add_special_tokens=False
                        ),
                        "expected_ids": tokenizer.encode(
                            expected_text, add_special_tokens=False
                        ),
                    })

        groups = {}
        for case in cases:
            key = (len(case["input_ids"]), len(case["expected_ids"]))
            groups.setdefault(key, []).append(case)
        for _, values in sorted(groups.items()):
            for start in range(0, len(values), 16):
                batch = values[start : start + 16]
                prompt_len = len(batch[0]["input_ids"])
                answer_len = len(batch[0]["expected_ids"])
                input_ids = torch.tensor(
                    [case["input_ids"] for case in batch],
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
                        max_new_tokens=answer_len + 4,
                        eos_token_id=effective_eos,
                        pad_token_id=int(tokenizer.pad_token_id),
                        return_dict_in_generate=True,
                    )
                suffixes = (
                    generated.sequences[:, prompt_len:]
                    .detach()
                    .cpu()
                    .tolist()
                )
                for case, suffix in zip(batch, suffixes):
                    before_eos = strip_eos(
                        [int(value) for value in suffix], eos_set
                    )
                    text = tokenizer.decode(
                        before_eos,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    rows.append({
                        "schema_version": (
                            "phase1003_answer_surface_probe_row.v1"
                        ),
                        "model": model_name,
                        "surface": case["surface"],
                        "domain": case["domain"],
                        "template": case["template"],
                        "query_role": case["query_role"],
                        "gold": case["gold"],
                        "expected_ids": case["expected_ids"],
                        "generated_ids": before_eos,
                        "generated_text": text,
                        "exact": before_eos == case["expected_ids"],
                        "contains_gold": (
                            case["gold"].lower() in text.lower()
                        ),
                    })
                del generated, input_ids, attention

        surfaces = {}
        for surface_name in SURFACES:
            values = [
                row for row in rows if row["surface"] == surface_name
            ]
            cells = {}
            for domain in DOMAINS:
                cell = [row for row in values if row["domain"] == domain]
                cells[domain] = {
                    "n": len(cell),
                    "exact_rate": (
                        sum(row["exact"] for row in cell) / len(cell)
                    ),
                    "contains_gold_rate": (
                        sum(row["contains_gold"] for row in cell) / len(cell)
                    ),
                }
            surfaces[surface_name] = {
                "n": len(values),
                "exact_rate": (
                    sum(row["exact"] for row in values) / len(values)
                ),
                "contains_gold_rate": (
                    sum(row["contains_gold"] for row in values) / len(values)
                ),
                "cells": cells,
            }
        summary = {
            "schema_version": "phase1003_answer_surface_probe.v1",
            "model": model_name,
            "surface_order": list(SURFACES),
            "formal_names_used": False,
            "internal_states_observed": False,
            "surfaces": surfaces,
        }
        root = OUT_ROOT / "answer_surface_probe" / model_name
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
