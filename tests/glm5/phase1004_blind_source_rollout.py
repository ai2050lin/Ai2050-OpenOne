#!/usr/bin/env python3
"""Confirm blind prompt-source sets through natural rollout and EOS."""
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

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1004_blind_causal_basis_protocol import (
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    selected_directional_rows,
    write_json,
    write_jsonl,
)
from phase1004_blind_source_fingerprints import (
    SOURCE_DEPTH,
    case_tensors,
    choose_donors,
)


BATCH_SIZE = 8


def source_summary(
    model_name: str,
    domain: str,
    template: int,
    precision_root: str,
) -> dict[str, Any]:
    path = (
        OUT_ROOT
        / precision_root
        / model_name
        / domain
        / "confirmation"
        / f"template_{template}"
        / "summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def capture_prompt_depth(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
) -> torch.Tensor:
    input_ids, attention = case_tensors(cases, device)
    captured = []

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach())

    handle = layers[SOURCE_DEPTH - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if len(captured) != 1:
            raise RuntimeError(
                f"prompt capture count drift: {len(captured)}"
            )
        return captured[0]
    finally:
        handle.remove()
        del input_ids, attention


def generate(
    model,
    tokenizer,
    layers,
    device,
    target_cases: list[dict[str, Any]],
    target_hidden: torch.Tensor,
    source_hidden: torch.Tensor | None,
    source_positions: list[int],
    effective_eos: list[int],
) -> list[list[int]]:
    input_ids, attention = case_tensors(target_cases, device)
    prompt_width = input_ids.shape[1]
    count = [0]
    handle = None
    if source_hidden is not None:
        patch = target_hidden.clone()
        if source_positions:
            patch[:, source_positions, :] = source_hidden[
                :, source_positions, :
            ]

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.shape[1] != prompt_width:
                return output
            count[0] += 1
            replacement = patch.to(
                device=value.device, dtype=value.dtype
            )
            return (
                (replacement,) + output[1:]
                if isinstance(output, tuple)
                else replacement
            )

        handle = layers[SOURCE_DEPTH - 1].register_forward_hook(hook)
    try:
        answer_width = max(
            len(case["answer_token_ids"]) for case in target_cases
        )
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=answer_width + 3,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        if source_hidden is not None and count[0] != 1:
            raise RuntimeError(
                f"generation source count drift: {count[0]}"
            )
        return [
            [int(value) for value in row]
            for row in output.sequences[:, prompt_width:]
            .detach()
            .cpu()
            .tolist()
        ]
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention


def strip_eos(
    values: list[int],
    eos_set: set[int],
) -> tuple[list[int], int | None]:
    for index, token_id in enumerate(values):
        if token_id in eos_set:
            return values[:index], index
    return values, None


def semantic_prediction(
    values: list[int],
    case: dict[str, Any],
) -> str | None:
    step = int(case["semantic_step"])
    if len(values) <= step:
        return None
    lookup = {
        int(token_id): label
        for label, token_id in case["candidate_token_ids"].items()
    }
    return lookup.get(int(values[step]))


def batches(
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    same_donors: list[dict[str, Any]],
    batch_size: int,
):
    for start in range(0, len(rows), batch_size):
        stop = start + batch_size
        yield (
            rows[start:stop],
            donors[start:stop],
            same_donors[start:stop],
        )


def run_template(
    model,
    tokenizer,
    layers,
    device,
    model_name: str,
    domain: str,
    template: int,
    rows: list[dict[str, Any]],
    source_positions: list[int],
    effective_eos: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donors, donor_audit = choose_donors(
        rows, same_answer=False
    )
    same_donors, same_audit = choose_donors(
        rows, same_answer=True
    )
    eos_set = set(effective_eos)
    result = []
    all_batches = list(
        batches(rows, donors, same_donors, batch_size)
    )
    for batch_number, (batch, donor_batch, same_batch) in enumerate(
        all_batches, 1
    ):
        target_cases = [row["target"] for row in batch]
        target_hidden = capture_prompt_depth(
            model, layers, device, target_cases
        )
        donor_hidden = capture_prompt_depth(
            model, layers, device, donor_batch
        )
        same_hidden = capture_prompt_depth(
            model, layers, device, same_batch
        )
        generated_by_condition = {
            "clean_target": generate(
                model,
                tokenizer,
                layers,
                device,
                target_cases,
                target_hidden,
                None,
                [],
                effective_eos,
            ),
            "frozen_target_noop": generate(
                model,
                tokenizer,
                layers,
                device,
                target_cases,
                target_hidden,
                target_hidden,
                source_positions,
                effective_eos,
            ),
            "frozen_source": generate(
                model,
                tokenizer,
                layers,
                device,
                target_cases,
                target_hidden,
                donor_hidden,
                source_positions,
                effective_eos,
            ),
            "frozen_same_answer_control": generate(
                model,
                tokenizer,
                layers,
                device,
                target_cases,
                target_hidden,
                same_hidden,
                source_positions,
                effective_eos,
            ),
        }
        for condition, generated_batch in generated_by_condition.items():
            for index, item in enumerate(batch):
                generated, eos_position = strip_eos(
                    generated_batch[index], eos_set
                )
                target_expected = [
                    int(value)
                    for value in target_cases[index][
                        "answer_token_ids"
                    ]
                ]
                donor_expected = [
                    int(value)
                    for value in donor_batch[index][
                        "answer_token_ids"
                    ]
                ]
                prediction = semantic_prediction(
                    generated, target_cases[index]
                )
                result.append({
                    "schema_version": (
                        "phase1004_blind_source_rollout_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "split": "confirmation",
                    "template": template,
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "condition": condition,
                    "source_positions": source_positions,
                    "target_gold": target_cases[index]["gold"],
                    "donor_gold": donor_batch[index]["gold"],
                    "semantic_prediction": prediction,
                    "predicted_target": (
                        prediction == target_cases[index]["gold"]
                    ),
                    "predicted_donor": (
                        prediction == donor_batch[index]["gold"]
                    ),
                    "generated_ids": generated,
                    "generated_text": tokenizer.decode(
                        generated,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "target_exact": generated == target_expected,
                    "donor_exact": generated == donor_expected,
                    "eos_position": eos_position,
                    "target_eos_boundary": (
                        generated == target_expected
                        and eos_position == len(target_expected)
                    ),
                    "donor_eos_boundary": (
                        generated == donor_expected
                        and eos_position == len(donor_expected)
                    ),
                })
        print(
            f"[rollout] {model_name}/{domain}/t{template} "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
        del target_hidden, donor_hidden, same_hidden
    clean_lookup = {
        (row["pair_id"], row["direction"]): row
        for row in result
        if row["condition"] == "clean_target"
    }
    condition_summary = {}
    for condition in sorted({
        row["condition"] for row in result
    }):
        values = [
            row for row in result
            if row["condition"] == condition
        ]
        item = {
            "n": len(values),
            "target_semantic_rate": float(np.mean([
                row["predicted_target"] for row in values
            ])),
            "donor_semantic_rate": float(np.mean([
                row["predicted_donor"] for row in values
            ])),
            "target_exact_rate": float(np.mean([
                row["target_exact"] for row in values
            ])),
            "donor_exact_rate": float(np.mean([
                row["donor_exact"] for row in values
            ])),
            "target_eos_boundary_rate": float(np.mean([
                row["target_eos_boundary"] for row in values
            ])),
            "donor_eos_boundary_rate": float(np.mean([
                row["donor_eos_boundary"] for row in values
            ])),
        }
        if condition == "frozen_target_noop":
            item["noop_sequence_agreement"] = float(np.mean([
                row["generated_ids"]
                == clean_lookup[
                    (row["pair_id"], row["direction"])
                ]["generated_ids"]
                for row in values
            ]))
        condition_summary[condition] = item
    return result, {
        "model": model_name,
        "domain": domain,
        "template": template,
        "n": len(rows),
        "source_positions": source_positions,
        "donor_audit": donor_audit,
        "same_answer_donor_audit": same_audit,
        "conditions": condition_summary,
        "rollout_gate_pass": (
            condition_summary["clean_target"][
                "target_semantic_rate"
            ] >= 0.95
            and condition_summary["frozen_source"][
                "donor_semantic_rate"
            ] >= 0.70
            and condition_summary["frozen_source"][
                "donor_eos_boundary_rate"
            ] >= 0.95
            and condition_summary["frozen_target_noop"][
                "noop_sequence_agreement"
            ] >= 0.99
            and condition_summary["frozen_same_answer_control"][
                "target_semantic_rate"
            ] >= 0.95
        ),
    }


def run_model(
    model_name: str,
    batch_size: int,
    *,
    use_8bit: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase1004 requires CUDA")
    source_root = (
        "blind_source" if use_8bit else "blind_source_bf16"
    )
    rollout_root = (
        "blind_rollout" if use_8bit else "blind_rollout_bf16"
    )
    output_root = OUT_ROOT / rollout_root / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name,
            dtype=torch.bfloat16,
            use_8bit=use_8bit,
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        cell_summaries = []
        for domain in DOMAINS:
            all_rows = selected_directional_rows(
                model_name, domain, "confirmation"
            )
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in all_rows:
                grouped[int(row["target"]["template"])].append(row)
            for template, rows in sorted(grouped.items()):
                rows.sort(
                    key=lambda row: (
                        row["pair_id"], row["direction"]
                    )
                )
                source = source_summary(
                    model_name,
                    domain,
                    template,
                    source_root,
                )
                if not source["final_source_gate_pass"]:
                    cell_summaries.append({
                        "model": model_name,
                        "domain": domain,
                        "template": template,
                        "status": "source_parent_gate_failed",
                        "rollout_gate_pass": False,
                    })
                    continue
                result_rows, summary = run_template(
                    model,
                    tokenizer,
                    layers,
                    device,
                    model_name,
                    domain,
                    template,
                    rows,
                    source["frozen_physical_positions"],
                    effective_eos,
                    batch_size,
                )
                cell_root = output_root / domain / f"template_{template}"
                write_jsonl(cell_root / "rows.jsonl", result_rows)
                write_json(cell_root / "summary.json", summary)
                cell_summaries.append(summary)
        summary = {
            "schema_version": "phase1004_rollout_model_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "precision": "8bit" if use_8bit else "bf16",
            "cell_count": len(cell_summaries),
            "rollout_gate_pass_count": sum(
                item.get("rollout_gate_pass", False)
                for item in cell_summaries
            ),
            "cells": cell_summaries,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    summary = run_model(
        args.model,
        args.batch_size,
        use_8bit=not args.bf16,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
