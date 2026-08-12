#!/usr/bin/env python3
"""Run non-blocking behavior qualification for the Phase1008 atlas."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
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
from phase1008_global_response_atlas_protocol import (
    MODELS,
    OUT_ROOT,
    PAIR_OPERATIONS,
    PHASE,
    SPLITS,
    canonical,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 16


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def semantic_answer_ids(case: dict[str, Any]) -> list[int]:
    return [
        int(case["answer_token_ids"][int(index)])
        for index in case["semantic_steps"]
    ]


def step_cases(
    cases: list[dict[str, Any]],
    prefixes: list[list[int]],
) -> list[dict[str, Any]]:
    if len(cases) != len(prefixes):
        raise RuntimeError("prefix batch drift")
    result = []
    for case, semantic_prefix in zip(cases, prefixes):
        row = dict(case)
        row["input_ids"] = (
            [int(value) for value in case["input_ids"]]
            + [int(value) for value in case["protocol_prefix_ids"]]
            + [int(value) for value in semantic_prefix]
        )
        result.append(row)
    return result


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


def forward_predictions(
    model,
    device,
    cases: list[dict[str, Any]],
) -> list[int]:
    input_ids, attention = case_tensors(cases, device)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        values = output.logits[:, -1, :].argmax(dim=-1)
        result = [int(value) for value in values.detach().cpu().tolist()]
        del output, values
        return result
    finally:
        del input_ids, attention


def behavior_batch(
    model,
    layers,
    tokenizer,
    device,
    model_name: str,
    cases: list[dict[str, Any]],
    effective_eos: set[int],
) -> list[dict[str, Any]]:
    stage0_cases = step_cases(cases, [[] for _ in cases])
    prediction0 = forward_predictions(model, device, stage0_cases)
    stage1_auto = step_cases(cases, [[value] for value in prediction0])
    prediction1 = forward_predictions(model, device, stage1_auto)
    gold0 = [semantic_answer_ids(case)[0] for case in cases]
    stage1_teacher = step_cases(cases, [[value] for value in gold0])
    teacher1 = forward_predictions(model, device, stage1_teacher)
    generated = natural_generate(
        model,
        layers,
        tokenizer,
        device,
        cases,
        effective_eos_ids=effective_eos,
    )
    rollout_rows, _ = sequence_metrics(generated, cases, effective_eos)
    rollout_by_id = {row["record_id"]: row for row in rollout_rows}
    rows = []
    for index, case in enumerate(cases):
        expected = semantic_answer_ids(case)
        rollout = rollout_by_id[case["record_id"]]
        semantic_hits = [
            int(prediction0[index]) == expected[0],
            int(prediction1[index]) == expected[1],
            int(teacher1[index]) == expected[1],
        ]
        rows.append({
            "schema_version": "phase1008_behavior_row.v1",
            "phase": PHASE,
            "model": model_name,
            "split": case["split"],
            "template": int(case["template"]),
            "unit_id": case["unit_id"],
            "record_id": case["record_id"],
            "state": case["state"],
            "operation": case["operation"],
            "expected_semantic_ids": expected,
            "autonomous_semantic_ids": [
                int(prediction0[index]),
                int(prediction1[index]),
            ],
            "teacher_semantic1_id": int(teacher1[index]),
            "semantic_step0_hit": semantic_hits[0],
            "semantic_step1_hit": semantic_hits[1],
            "teacher_semantic1_hit": semantic_hits[2],
            "semantic_gate": all(semantic_hits),
            "generated_ids": rollout["generated_ids"],
            "natural_exact": bool(rollout["exact"]),
            "natural_protocol_prefix_match": bool(
                rollout["protocol_prefix_match"]
            ),
            "immediate_eos": bool(rollout["immediate_eos"]),
            "eos_position": rollout["eos_position"],
            "rollout_gate": bool(
                rollout["exact"] and rollout["immediate_eos"]
            ),
        })
    return rows


def summarize_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["split"], int(row["template"]), row["state"])
        ].append(row)
    result = []
    for (split, template, state), group in sorted(grouped.items()):
        result.append({
            "schema_version": "phase1008_behavior_cell.v1",
            "phase": PHASE,
            "model": group[0]["model"],
            "split": split,
            "template": template,
            "state": state,
            "n": len(group),
            "semantic_step0_accuracy": float(np.mean([
                row["semantic_step0_hit"] for row in group
            ])),
            "semantic_step1_accuracy": float(np.mean([
                row["semantic_step1_hit"] for row in group
            ])),
            "teacher_semantic1_accuracy": float(np.mean([
                row["teacher_semantic1_hit"] for row in group
            ])),
            "semantic_gate_rate": float(np.mean([
                row["semantic_gate"] for row in group
            ])),
            "natural_exact_rate": float(np.mean([
                row["natural_exact"] for row in group
            ])),
            "immediate_eos_rate": float(np.mean([
                row["immediate_eos"] for row in group
            ])),
            "rollout_gate_rate": float(np.mean([
                row["rollout_gate"] for row in group
            ])),
        })
    return result


def pair_rows(
    units: list[dict[str, Any]],
    behavior_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for unit in units:
        for operation in PAIR_OPERATIONS:
            pair = unit["operation_pairs"][operation]
            base = behavior_by_id[pair["base"]]
            variant = behavior_by_id[pair["variant"]]
            result.append({
                "schema_version": "phase1008_pair_qualification.v1",
                "phase": PHASE,
                "model": unit["model"],
                "split": unit["split"],
                "template": int(unit["template"]),
                "name_pool": int(unit["name_pool"]),
                "world_index": int(unit["world_index"]),
                "unit_id": unit["unit_id"],
                "operation": operation,
                "base_record_id": base["record_id"],
                "variant_record_id": variant["record_id"],
                "base_semantic_gate": bool(base["semantic_gate"]),
                "variant_semantic_gate": bool(variant["semantic_gate"]),
                "semantic_pair_qualified": bool(
                    base["semantic_gate"] and variant["semantic_gate"]
                ),
                "base_rollout_gate": bool(base["rollout_gate"]),
                "variant_rollout_gate": bool(variant["rollout_gate"]),
                "rollout_pair_qualified": bool(
                    base["rollout_gate"] and variant["rollout_gate"]
                ),
            })
    return result


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != 1:
        raise RuntimeError("Phase1008 protocol revision drift")
    model_root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(model_root / "cases.jsonl")
    units = read_jsonl(model_root / "units.jsonl")
    started = time.time()
    model = tokenizer = device = None
    rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            grouped[(case["split"], int(case["template"]))].append(case)
        for (split, template), group in sorted(grouped.items()):
            for batch in chunks(group, BATCH_SIZE):
                rows.extend(behavior_batch(
                    model,
                    layers,
                    tokenizer,
                    device,
                    model_name,
                    batch,
                    effective_eos,
                ))
            cell_rows = [
                row
                for row in rows
                if row["split"] == split and int(row["template"]) == template
            ]
            print(
                f"[behavior] {model_name}/{split}/t{template} "
                f"n={len(cell_rows)} "
                f"semantic={np.mean([r['semantic_gate'] for r in cell_rows]):.3f} "
                f"rollout={np.mean([r['rollout_gate'] for r in cell_rows]):.3f}",
                flush=True,
            )
        behavior_by_id = {row["record_id"]: row for row in rows}
        if len(behavior_by_id) != len(cases):
            raise RuntimeError("behavior record coverage drift")
        pairs = pair_rows(units, behavior_by_id)
        cells = summarize_cells(rows)
        operation_summary = {}
        for operation in PAIR_OPERATIONS:
            selected = [row for row in pairs if row["operation"] == operation]
            operation_summary[operation] = {
                "n": len(selected),
                "semantic_pair_qualified": sum(
                    row["semantic_pair_qualified"] for row in selected
                ),
                "semantic_pair_rate": float(np.mean([
                    row["semantic_pair_qualified"] for row in selected
                ])),
                "rollout_pair_qualified": sum(
                    row["rollout_pair_qualified"] for row in selected
                ),
                "rollout_pair_rate": float(np.mean([
                    row["rollout_pair_qualified"] for row in selected
                ])),
            }
        summary = {
            "schema_version": "phase1008_behavior_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "protocol_digest": protocol["preregistration_digest"],
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "model_class": info.model_class,
                "loaded_8bit": True,
            },
            "case_count": len(rows),
            "pair_count": len(pairs),
            "effective_eos_ids": sorted(effective_eos),
            "state_cells": cells,
            "operation_summary": operation_summary,
            "overall_semantic_case_rate": float(np.mean([
                row["semantic_gate"] for row in rows
            ])),
            "overall_rollout_case_rate": float(np.mean([
                row["rollout_gate"] for row in rows
            ])),
            "elapsed_seconds": time.time() - started,
            "policy": (
                "qualification annotates atlas denominators and never "
                "blocks unrelated cells"
            ),
        }
        output_root = OUT_ROOT / "behavior" / model_name
        write_jsonl(output_root / "rows.jsonl", rows)
        write_jsonl(output_root / "pair_qualification.jsonl", pairs)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "cases": len(rows),
            "pairs": len(pairs),
            "semantic_case_rate": summary["overall_semantic_case_rate"],
            "rollout_case_rate": summary["overall_rollout_case_rate"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
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
