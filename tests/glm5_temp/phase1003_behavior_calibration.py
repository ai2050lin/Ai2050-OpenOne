#!/usr/bin/env python3
"""Calibration-only behavior probe for the Phase 1003 wording."""
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
    CALIBRATION_NAMES,
    DOMAINS,
    MODELS,
    OUT_ROOT,
    answer_text,
    render_user_prompt,
    write_json,
    write_jsonl,
)
from phase548_shared_attention_compute_protocol import render_chat


def strip_eos(values: list[int], eos_set: set[int]) -> list[int]:
    for index, value in enumerate(values):
        if value in eos_set:
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
        for domain, values in DOMAINS.items():
            for template, pair_index, query_role, value_swap in itertools.product(
                range(4), range(2), (0, 1), (0, 1)
            ):
                entity0 = CALIBRATION_NAMES[2 * pair_index]
                entity1 = CALIBRATION_NAMES[2 * pair_index + 1]
                value0, value1 = values[:2]
                if value_swap:
                    value0, value1 = value1, value0
                query = (entity0, entity1)[query_role]
                gold = (value0, value1)[query_role]
                raw = render_user_prompt(
                    template,
                    domain,
                    entity0,
                    value0,
                    entity1,
                    value1,
                    query,
                )
                rendered = render_chat(tokenizer, model_name, raw)
                input_ids = tokenizer.encode(
                    rendered, add_special_tokens=False
                )
                expected = tokenizer.encode(
                    answer_text(model_name, gold),
                    add_special_tokens=False,
                )
                cases.append({
                    "domain": domain,
                    "template": template,
                    "pair_index": pair_index,
                    "query_role": query_role,
                    "value_swap": value_swap,
                    "gold": gold,
                    "input_ids": input_ids,
                    "expected": expected,
                })

        groups = {}
        for case in cases:
            key = (
                case["domain"],
                case["template"],
                len(case["input_ids"]),
                len(case["expected"]),
            )
            groups.setdefault(key, []).append(case)
        for key, values in sorted(groups.items()):
            for start in range(0, len(values), 16):
                batch = values[start : start + 16]
                prompt_len = len(batch[0]["input_ids"])
                answer_len = len(batch[0]["expected"])
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
                        max_new_tokens=answer_len + 2,
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
                    rows.append({
                        "schema_version": (
                            "phase1003_behavior_calibration_row.v1"
                        ),
                        "model": model_name,
                        "domain": case["domain"],
                        "template": case["template"],
                        "pair_index": case["pair_index"],
                        "query_role": case["query_role"],
                        "value_swap": case["value_swap"],
                        "gold": case["gold"],
                        "expected_ids": case["expected"],
                        "generated_ids": before_eos,
                        "exact": before_eos == case["expected"],
                        "generated_text": tokenizer.decode(
                            before_eos,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    })
                del generated, input_ids, attention

        cells = {}
        for domain, template in itertools.product(DOMAINS, range(4)):
            values = [
                row
                for row in rows
                if row["domain"] == domain
                and row["template"] == template
            ]
            cells[f"{domain}:t{template}"] = {
                "n": len(values),
                "exact_rate": sum(row["exact"] for row in values) / len(values),
            }
        summary = {
            "schema_version": "phase1003_behavior_calibration.v1",
            "model": model_name,
            "formal_names_used": False,
            "internal_states_observed": False,
            "n": len(rows),
            "exact_rate": sum(row["exact"] for row in rows) / len(rows),
            "cells": cells,
            "all_cells_at_least_0_95": all(
                value["exact_rate"] >= 0.95 for value in cells.values()
            ),
        }
        root = OUT_ROOT / "calibration" / model_name
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
