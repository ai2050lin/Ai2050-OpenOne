#!/usr/bin/env python3
"""Run Phase1009 behavior qualification without blocking atlas collection."""
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
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PAIR_OPERATIONS,
    PHASE,
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


def with_suffix(
    case: dict[str, Any],
    suffix: list[int],
) -> dict[str, Any]:
    row = dict(case)
    row["input_ids"] = (
        [int(value) for value in case["input_ids"]]
        + [int(value) for value in suffix]
    )
    return row


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


def prediction_rows(
    model,
    device,
    cases: list[dict[str, Any]],
    candidate_panel: bool,
) -> tuple[list[int], list[int] | None]:
    input_ids, attention = case_tensors(cases, device)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        logits = output.logits[:, -1, :]
        full = [
            int(value)
            for value in logits.argmax(dim=-1).detach().cpu().tolist()
        ]
        panel = None
        if candidate_panel:
            panel = []
            for index, case in enumerate(cases):
                ids = [
                    int(value)
                    for value in case["candidate_name_ids"].values()
                ]
                selected = int(
                    torch.tensor(ids, device=logits.device)[
                        logits[index, ids].argmax()
                    ].item()
                )
                panel.append(selected)
        del output, logits
        return full, panel
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
    semantic_cases = [
        with_suffix(case, list(case["protocol_prefix_ids"]))
        for case in cases
    ]
    semantic_full, semantic_panel = prediction_rows(
        model,
        device,
        semantic_cases,
        candidate_panel=True,
    )
    if semantic_panel is None:
        raise RuntimeError("candidate panel missing")
    gold_ids = [
        int(case["answer_token_ids"][int(case["semantic_step"])])
        for case in cases
    ]
    function_cases = [
        with_suffix(
            case,
            list(case["protocol_prefix_ids"]) + [gold_id],
        )
        for case, gold_id in zip(cases, gold_ids)
    ]
    function_full, _ = prediction_rows(
        model,
        device,
        function_cases,
        candidate_panel=False,
    )
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
        rollout = rollout_by_id[case["record_id"]]
        panel_hit = int(semantic_panel[index]) == gold_ids[index]
        full_hit = int(semantic_full[index]) == gold_ids[index]
        function_hit = (
            int(function_full[index]) == int(case["function_token_id"])
        )
        rows.append({
            "schema_version": "phase1009_behavior_row.v1",
            "phase": PHASE,
            "model": model_name,
            "family": case["family"],
            "split": case["split"],
            "template": int(case["template"]),
            "name_pool": int(case["name_pool"]),
            "world_index": int(case["world_index"]),
            "unit_id": case["unit_id"],
            "record_id": case["record_id"],
            "state": case["state"],
            "operation": case["operation"],
            "expected_semantic_id": gold_ids[index],
            "candidate_panel_prediction_id": int(semantic_panel[index]),
            "full_vocabulary_prediction_id": int(semantic_full[index]),
            "function_prediction_id": int(function_full[index]),
            "semantic_panel_hit": bool(panel_hit),
            "semantic_full_vocab_hit": bool(full_hit),
            "function_token_hit": bool(function_hit),
            "semantic_gate": bool(panel_hit),
            "strict_teacher_gate": bool(full_hit and function_hit),
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


def pair_rows(
    units: list[dict[str, Any]],
    behavior_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        for operation in PAIR_OPERATIONS:
            pair = unit["operation_pairs"][operation]
            base = behavior_by_id[pair["base"]]
            variant = behavior_by_id[pair["variant"]]
            rows.append({
                "schema_version": "phase1009_pair_qualification.v1",
                "phase": PHASE,
                "model": unit["model"],
                "family": unit["family"],
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
                "strict_teacher_pair_qualified": bool(
                    base["strict_teacher_gate"]
                    and variant["strict_teacher_gate"]
                ),
                "rollout_pair_qualified": bool(
                    base["rollout_gate"] and variant["rollout_gate"]
                ),
            })
    return rows


def aggregate_rates(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, values)}
        item.update({
            "n": len(group),
            "semantic_panel_rate": float(np.mean([
                row["semantic_panel_hit"] for row in group
            ])),
            "semantic_full_vocab_rate": float(np.mean([
                row["semantic_full_vocab_hit"] for row in group
            ])),
            "function_token_rate": float(np.mean([
                row["function_token_hit"] for row in group
            ])),
            "strict_teacher_rate": float(np.mean([
                row["strict_teacher_gate"] for row in group
            ])),
            "natural_exact_rate": float(np.mean([
                row["natural_exact"] for row in group
            ])),
            "rollout_gate_rate": float(np.mean([
                row["rollout_gate"] for row in group
            ])),
        })
        result.append(item)
    return result


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != 1:
        raise RuntimeError("Phase1009 protocol revision drift")
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
        grouped: dict[
            tuple[str, str, int, int],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for case in cases:
            grouped[
                (
                    case["family"],
                    case["split"],
                    int(case["template"]),
                    len(case["input_ids"]),
                )
            ].append(case)
        for (family, split, template, _), group in sorted(grouped.items()):
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
            selected = [
                row
                for row in rows
                if row["family"] == family
                and row["split"] == split
                and int(row["template"]) == template
            ]
            print(
                f"[behavior] {model_name}/{family}/{split}/t{template} "
                f"n={len(selected)} "
                f"panel={np.mean([r['semantic_gate'] for r in selected]):.3f} "
                f"strict={np.mean([r['strict_teacher_gate'] for r in selected]):.3f} "
                f"rollout={np.mean([r['rollout_gate'] for r in selected]):.3f}",
                flush=True,
            )
        behavior_by_id = {row["record_id"]: row for row in rows}
        if len(behavior_by_id) != len(cases):
            raise RuntimeError(
                f"behavior coverage {len(behavior_by_id)} != {len(cases)}"
            )
        pairs = pair_rows(units, behavior_by_id)
        family_rates = aggregate_rates(rows, ("family",))
        family_state_rates = aggregate_rates(rows, ("family", "state"))
        operation_summary = {}
        for family in FAMILIES:
            operation_summary[family] = {}
            for operation in PAIR_OPERATIONS:
                selected = [
                    row for row in pairs
                    if row["family"] == family
                    and row["operation"] == operation
                ]
                operation_summary[family][operation] = {
                    "n": len(selected),
                    "semantic_pair_qualified": int(sum(
                        row["semantic_pair_qualified"] for row in selected
                    )),
                    "semantic_pair_rate": float(np.mean([
                        row["semantic_pair_qualified"] for row in selected
                    ])),
                    "strict_teacher_pair_rate": float(np.mean([
                        row["strict_teacher_pair_qualified"]
                        for row in selected
                    ])),
                    "rollout_pair_rate": float(np.mean([
                        row["rollout_pair_qualified"] for row in selected
                    ])),
                }
        summary = {
            "schema_version": "phase1009_behavior_summary.v1",
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
            "family_rates": family_rates,
            "family_state_rates": family_state_rates,
            "operation_summary": operation_summary,
            "overall_semantic_panel_rate": float(np.mean([
                row["semantic_gate"] for row in rows
            ])),
            "overall_semantic_full_vocab_rate": float(np.mean([
                row["semantic_full_vocab_hit"] for row in rows
            ])),
            "overall_strict_teacher_rate": float(np.mean([
                row["strict_teacher_gate"] for row in rows
            ])),
            "overall_rollout_case_rate": float(np.mean([
                row["rollout_gate"] for row in rows
            ])),
            "elapsed_seconds": time.time() - started,
            "policy": (
                "behavior qualification annotates the atlas and does not "
                "block unrelated families or cells"
            ),
        }
        output_root = OUT_ROOT / "behavior" / model_name
        write_jsonl(output_root / "rows.jsonl", rows)
        write_jsonl(output_root / "pair_qualification.jsonl", pairs)
        write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "cases": len(rows),
            "semantic_panel_rate": summary["overall_semantic_panel_rate"],
            "strict_teacher_rate": summary["overall_strict_teacher_rate"],
            "rollout_rate": summary["overall_rollout_case_rate"],
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
