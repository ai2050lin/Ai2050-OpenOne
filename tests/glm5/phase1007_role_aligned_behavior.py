#!/usr/bin/env python3
"""Run Phase1007 formal behavior qualification."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1006_blind_source_and_behavior import (
    eos_token_ids,
    natural_generate,
    sequence_metrics,
)
from phase1007_role_aligned_causal_source_protocol import (
    CONTRASTS,
    MODELS,
    OUT_ROOT,
    PHASE,
    SPLITS,
    TEMPLATES_BY_SPLIT,
    decision_case,
    read_json,
    selected_directional_rows,
    semantic_answer_ids,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 16


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def case_tensors(cases: list[dict[str, Any]], device):
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift: {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def forward_step(
    model,
    device,
    cases: list[dict[str, Any]],
    logical_step: int,
    semantic_prefixes: list[list[int]],
) -> dict[str, Any]:
    if len(cases) != len(semantic_prefixes):
        raise RuntimeError("semantic prefix batch drift")
    step_cases = [
        decision_case(
            case,
            semantic_prefix=[
                int(value) for value in semantic_prefixes[index]
            ],
            logical_step=logical_step,
        )
        for index, case in enumerate(cases)
    ]
    input_ids, attention = case_tensors(step_cases, device)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[:, -1, :]
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        del output, logits
        return {"prediction_ids": [int(value) for value in predictions]}
    finally:
        del input_ids, attention


def forward_two_step(
    model,
    device,
    cases: list[dict[str, Any]],
) -> list[list[int]]:
    step0 = forward_step(
        model,
        device,
        cases,
        0,
        [[] for _ in cases],
    )
    step1 = forward_step(
        model,
        device,
        cases,
        1,
        [[value] for value in step0["prediction_ids"]],
    )
    return [
        [step0["prediction_ids"][index], step1["prediction_ids"][index]]
        for index in range(len(cases))
    ]


def teacher_forced_step1(
    model,
    device,
    cases: list[dict[str, Any]],
) -> list[int]:
    output = forward_step(
        model,
        device,
        cases,
        1,
        [[semantic_answer_ids(case)[0]] for case in cases],
    )
    return output["prediction_ids"]


def behavior_cell(
    model,
    layers,
    tokenizer,
    device,
    model_name: str,
    split: str,
    template: int,
    contrast: str,
    protocol_digest: str,
) -> dict[str, Any]:
    directional = selected_directional_rows(
        model_name, split, template, contrast, "formal"
    )
    cases = [item["target"] for item in directional]
    effective_eos = eos_token_ids(model, tokenizer, model_name)
    detail = []
    generated_all = []
    for batch in chunks(cases, BATCH_SIZE):
        predictions = forward_two_step(model, device, batch)
        teacher = teacher_forced_step1(model, device, batch)
        generated_all.extend(natural_generate(
            model,
            layers,
            tokenizer,
            device,
            batch,
            effective_eos_ids=effective_eos,
        ))
        for index, case in enumerate(batch):
            expected = semantic_answer_ids(case)
            detail.append({
                "schema_version": "phase1007_behavior_row.v1",
                "phase": PHASE,
                "model": model_name,
                "split": split,
                "template": template,
                "contrast": contrast,
                "record_id": case["record_id"],
                "prediction_ids": predictions[index],
                "expected_ids": expected,
                "step_hits": [
                    predictions[index][step] == expected[step]
                    for step in (0, 1)
                ],
                "teacher_step1_hit": (
                    int(teacher[index]) == expected[1]
                ),
            })
    rollout_rows, rollout = sequence_metrics(
        generated_all, cases, effective_eos
    )
    rollout_by_id = {row["record_id"]: row for row in rollout_rows}
    for row in detail:
        row.update({
            key: value
            for key, value in rollout_by_id[row["record_id"]].items()
            if key != "record_id"
        })

    summary = {
        "schema_version": "phase1007_behavior_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "template": template,
        "contrast": contrast,
        "partition": "formal",
        "protocol_digest": protocol_digest,
        "n": len(detail),
        "effective_termination_ids": sorted(effective_eos),
        "step0_accuracy": float(np.mean([
            row["step_hits"][0] for row in detail
        ])),
        "step1_accuracy": float(np.mean([
            row["step_hits"][1] for row in detail
        ])),
        "teacher_step1_accuracy": float(np.mean([
            row["teacher_step1_hit"] for row in detail
        ])),
        **rollout,
    }
    summary["behavior_gate_pass"] = (
        summary["step0_accuracy"] >= 0.95
        and summary["step1_accuracy"] >= 0.95
        and summary["teacher_step1_accuracy"] >= 0.95
        and summary["natural_exact_rate"] >= 0.90
        and summary["natural_protocol_prefix_rate"] >= 0.90
        and summary["immediate_eos_rate"] >= 0.90
    )
    cell_root = (
        OUT_ROOT
        / "behavior"
        / model_name
        / split
        / f"template{template}"
        / contrast
    )
    write_jsonl(cell_root / "rows.jsonl", detail)
    write_json(cell_root / "summary.json", summary)
    print(
        f"[behavior] {model_name}/{split}/t{template}/{contrast} "
        f"s0={summary['step0_accuracy']:.3f} "
        f"s1={summary['step1_accuracy']:.3f} "
        f"exact={summary['natural_exact_rate']:.3f} "
        f"eos={summary['immediate_eos_rate']:.3f} "
        f"pass={summary['behavior_gate_pass']}",
        flush=True,
    )
    return summary


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != 1:
        raise RuntimeError("Phase1007 protocol revision drift")
    protocol_digest = protocol["preregistration_digest"]
    started = time.time()
    model = tokenizer = device = None
    cells = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        for split in SPLITS:
            for template in TEMPLATES_BY_SPLIT[split]:
                for contrast in CONTRASTS:
                    cells.append(behavior_cell(
                        model,
                        layers,
                        tokenizer,
                        device,
                        model_name,
                        split,
                        int(template),
                        contrast,
                        protocol_digest,
                    ))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        summary = {
            "schema_version": "phase1007_behavior_model.v1",
            "phase": PHASE,
            "model": model_name,
            "precision": "8bit",
            "protocol_digest": protocol_digest,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
            },
            "cells": cells,
            "gate_pass_count": sum(
                item["behavior_gate_pass"] for item in cells
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_json(
            OUT_ROOT / "behavior" / model_name / "summary.json",
            summary,
        )
        return summary
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
    summary = run_model(args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
