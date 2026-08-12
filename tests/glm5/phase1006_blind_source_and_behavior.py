#!/usr/bin/env python3
"""Run Phase1006 behavior qualification and blind prompt-source discovery."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1006_autoregressive_temporal_aggregation_protocol import (
    DOMAIN,
    MODELS,
    OUT_ROOT,
    PHASE,
    SOURCE_DEPTH,
    SPLITS,
    TEMPLATES_BY_SPLIT,
    canonical,
    decision_case,
    read_json,
    read_jsonl,
    selected_directional_rows,
    stable_order,
    write_json,
    write_jsonl,
)


BATCH_SIZE = 16
SCREEN_N = 16
EPSILON = 1e-6


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


def candidate_panel(
    logits: torch.Tensor,
    cases: list[dict[str, Any]],
    step: int,
) -> tuple[list[str], torch.Tensor]:
    mapping = cases[0]["candidate_ids_by_step"][step]
    if any(
        case["candidate_ids_by_step"][step] != mapping for case in cases
    ):
        raise RuntimeError("candidate panel drift")
    labels = list(mapping)
    ids = torch.tensor(
        [int(mapping[label]) for label in labels],
        dtype=torch.long,
        device=logits.device,
    )
    return labels, logits.index_select(-1, ids).float().detach().cpu()


def label_for_token(
    token_id: int,
    case: dict[str, Any],
    step: int,
) -> str | None:
    reverse = {
        int(value): label
        for label, value in case["candidate_ids_by_step"][step].items()
    }
    return reverse.get(int(token_id))


def semantic_answer_ids(case: dict[str, Any]) -> list[int]:
    return [
        int(case["answer_token_ids"][int(absolute_step)])
        for absolute_step in case["semantic_steps"]
    ]


def capture_depth1(model, device, cases: list[dict[str, Any]]) -> torch.Tensor:
    input_ids, attention = case_tensors(cases, device)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = output.hidden_states[SOURCE_DEPTH].detach()
        del output
        return hidden
    finally:
        del input_ids, attention


def forward_step(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    step: int,
    semantic_prefixes: list[list[int]],
    *,
    source_hidden: torch.Tensor | None = None,
    positions: list[int] | None = None,
) -> dict[str, Any]:
    if len(semantic_prefixes) != len(cases):
        raise RuntimeError("semantic-prefix batch drift")
    step_cases = [
        decision_case(
            case,
            prefix_ids=(
                [int(value) for value in case["protocol_prefix_ids"]]
                + [int(value) for value in semantic_prefixes[index]]
            ),
            logical_step=step,
        )
        for index, case in enumerate(cases)
    ]
    input_ids, attention = case_tensors(step_cases, device)
    handle = None
    count = [0]
    if source_hidden is not None:
        frozen_positions = sorted({int(value) for value in positions or []})

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            raw_width = source_hidden.shape[1]
            if value.shape[1] < raw_width:
                return output
            patched = value.clone()
            if frozen_positions:
                patched[:, frozen_positions, :] = source_hidden[
                    :, frozen_positions, :
                ].to(device=value.device, dtype=value.dtype)
            count[0] += 1
            return (
                (patched,) + output[1:]
                if isinstance(output, tuple)
                else patched
            )

        handle = layers[SOURCE_DEPTH - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if source_hidden is not None and count[0] != 1:
            raise RuntimeError(f"source hook count drift: {count[0]}")
        logits = output.logits[:, -1, :]
        prediction_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        labels, panel = candidate_panel(logits, cases, step)
        del output, logits
        return {
            "labels": labels,
            "panel": panel,
            "prediction_ids": [int(value) for value in prediction_ids],
        }
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention


def forward_two_step(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    *,
    source_hidden: torch.Tensor | None = None,
    positions: list[int] | None = None,
) -> dict[str, Any]:
    step0 = forward_step(
        model,
        layers,
        device,
        cases,
        0,
        [[] for _ in cases],
        source_hidden=source_hidden,
        positions=positions,
    )
    step1 = forward_step(
        model,
        layers,
        device,
        cases,
        1,
        [[int(value)] for value in step0["prediction_ids"]],
        source_hidden=source_hidden,
        positions=positions,
    )
    return {"steps": [step0, step1]}


def forward_teacher_forced_step1(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return forward_step(
        model,
        layers,
        device,
        cases,
        1,
        [[semantic_answer_ids(case)[0]] for case in cases],
    )


def eos_token_ids(model, tokenizer, model_name: str) -> set[int]:
    value = getattr(model.generation_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    values = [] if value is None else ([value] if isinstance(value, int) else value)
    result = {int(item) for item in values if item is not None}
    token_names = {
        "qwen3": ("<|im_end|>",),
        "glm4": ("<|user|>",),
        "deepseek7b": ("<｜end▁of▁sentence｜>",),
    }[model_name]
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    for token in token_names:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id is not None
            and int(token_id) >= 0
            and (unknown_id is None or int(token_id) != int(unknown_id))
        ):
            result.add(int(token_id))
    if not result:
        raise RuntimeError(f"{model_name}: no effective termination IDs")
    return result


def natural_generate(
    model,
    layers,
    tokenizer,
    device,
    cases: list[dict[str, Any]],
    *,
    effective_eos_ids: set[int],
    source_hidden: torch.Tensor | None = None,
    positions: list[int] | None = None,
) -> list[list[int]]:
    input_ids, attention = case_tensors(cases, device)
    initial_width = input_ids.shape[1]
    handle = None
    count = [0]
    if source_hidden is not None:
        frozen_positions = sorted({int(value) for value in positions or []})

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.shape[1] != initial_width:
                return output
            patched = value.clone()
            if frozen_positions:
                patched[:, frozen_positions, :] = source_hidden[
                    :, frozen_positions, :
                ].to(device=value.device, dtype=value.dtype)
            count[0] += 1
            return (
                (patched,) + output[1:]
                if isinstance(output, tuple)
                else patched
            )

        handle = layers[SOURCE_DEPTH - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max(
                    len(case["answer_token_ids"]) for case in cases
                ) + 2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(effective_eos_ids),
                return_dict_in_generate=True,
            )
        if source_hidden is not None and count[0] != 1:
            raise RuntimeError(f"rollout source hook drift: {count[0]}")
        return [
            [int(value) for value in row[initial_width:].tolist()]
            for row in output.sequences
        ]
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention


def sequence_metrics(
    generated: list[list[int]],
    cases: list[dict[str, Any]],
    eos_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    for tokens, case in zip(generated, cases):
        eos_position = next(
            (index for index, value in enumerate(tokens) if value in eos_ids),
            None,
        )
        content = (
            tokens if eos_position is None else tokens[:eos_position]
        )
        expected = [int(value) for value in case["answer_token_ids"]]
        prefix = [int(value) for value in case["protocol_prefix_ids"]]
        rows.append({
            "record_id": case["record_id"],
            "generated_ids": tokens,
            "content_ids": content,
            "expected_ids": expected,
            "protocol_prefix_ids": prefix,
            "protocol_prefix_match": content[:len(prefix)] == prefix,
            "exact": content == expected,
            "immediate_eos": eos_position == len(expected),
            "eos_position": eos_position,
        })
    return rows, {
        "natural_exact_rate": float(np.mean([
            row["exact"] for row in rows
        ])),
        "immediate_eos_rate": float(np.mean([
            row["immediate_eos"] for row in rows
        ])),
        "natural_protocol_prefix_rate": float(np.mean([
            row["protocol_prefix_match"] for row in rows
        ])),
    }


def behavior_cell(
    model,
    layers,
    tokenizer,
    device,
    model_name: str,
    split: str,
    template: int,
    output_root: Path,
) -> dict[str, Any]:
    directional = [
        row
        for row in selected_directional_rows(model_name, split)
        if int(row["template"]) == template
    ]
    cases = [row["target"] for row in directional]
    if len(cases) != 32:
        raise RuntimeError(
            f"{model_name}/{split}/t{template}: behavior n={len(cases)}"
        )
    detail_rows = []
    step_hits = [[], []]
    teacher_step1_hits = []
    generated_all = []
    effective_eos = eos_token_ids(model, tokenizer, model_name)
    for batch in chunks(cases, BATCH_SIZE):
        output = forward_two_step(model, layers, device, batch)
        teacher = forward_teacher_forced_step1(
            model,
            layers,
            device,
            batch,
        )
        generated = natural_generate(
            model,
            layers,
            tokenizer,
            device,
            batch,
            effective_eos_ids=effective_eos,
        )
        generated_all.extend(generated)
        for index, case in enumerate(batch):
            predictions = [
                int(output["steps"][step]["prediction_ids"][index])
                for step in (0, 1)
            ]
            expected = [
                int(value) for value in semantic_answer_ids(case)
            ]
            hits = [
                predictions[step] == expected[step]
                for step in (0, 1)
            ]
            teacher_hit = (
                int(teacher["prediction_ids"][index]) == expected[1]
            )
            for step in (0, 1):
                step_hits[step].append(hits[step])
            teacher_step1_hits.append(teacher_hit)
            detail_rows.append({
                "schema_version": "phase1006_behavior_row.v1",
                "phase": PHASE,
                "model": model_name,
                "split": split,
                "template": template,
                "record_id": case["record_id"],
                "prediction_ids": predictions,
                "expected_ids": expected,
                "step_hits": hits,
                "teacher_forced_step1_hit": teacher_hit,
            })
    rollout_rows, rollout = sequence_metrics(
        generated_all,
        cases,
        effective_eos,
    )
    rollout_by_id = {row["record_id"]: row for row in rollout_rows}
    for row in detail_rows:
        row.update({
            key: value
            for key, value in rollout_by_id[row["record_id"]].items()
            if key != "record_id"
        })
    summary = {
        "schema_version": "phase1006_behavior_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "template": template,
        "n": len(cases),
        "effective_termination_ids": sorted(effective_eos),
        "step0_autoregressive_accuracy": float(np.mean(step_hits[0])),
        "step1_autoregressive_accuracy": float(np.mean(step_hits[1])),
        "step1_teacher_forced_accuracy": float(
            np.mean(teacher_step1_hits)
        ),
        **rollout,
    }
    summary["behavior_gate_pass"] = (
        summary["step0_autoregressive_accuracy"] >= 0.95
        and summary["step1_autoregressive_accuracy"] >= 0.95
        and summary["step1_teacher_forced_accuracy"] >= 0.95
        and summary["natural_protocol_prefix_rate"] >= 0.90
        and summary["natural_exact_rate"] >= 0.90
        and summary["immediate_eos_rate"] >= 0.90
    )
    cell_root = output_root / split / f"template{template}"
    write_jsonl(cell_root / "rows.jsonl", detail_rows)
    write_json(cell_root / "summary.json", summary)
    print(
        f"[behavior] {model_name}/{split}/t{template} "
        f"s0={summary['step0_autoregressive_accuracy']:.3f} "
        f"s1={summary['step1_autoregressive_accuracy']:.3f} "
        f"exact={summary['natural_exact_rate']:.3f} "
        f"eos={summary['immediate_eos_rate']:.3f} "
        f"pass={summary['behavior_gate_pass']}",
        flush=True,
    )
    return summary


def choose_donors(
    model_name: str,
    split: str,
    template: int,
    directional: list[dict[str, Any]],
    *,
    same_answer: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = [
        row
        for row in read_jsonl(
            OUT_ROOT / "protocol" / model_name / "cases.jsonl"
        )
        if row["split"] == split and int(row["template"]) == template
    ]
    selected = []
    disjoint_checks = []
    for item in directional:
        target = item["target"]
        target_codes = set(target["base_codes"])
        candidates = []
        for candidate in pool:
            if candidate["world_id"] == target["world_id"]:
                continue
            if (candidate["gold"] == target["gold"]) != same_answer:
                continue
            if not same_answer:
                disjoint = target_codes.isdisjoint(
                    set(candidate["base_codes"])
                )
                if not disjoint:
                    continue
                disjoint_checks.append(disjoint)
            candidates.append(candidate)
        if not candidates:
            raise RuntimeError(
                f"no {'same' if same_answer else 'different'} donor for "
                f"{target['record_id']}"
            )
        donor = sorted(
            candidates,
            key=lambda row: stable_order(
                row["record_id"],
                (
                    f"donor:{model_name}:{split}:t{template}:"
                    f"{same_answer}:{target['record_id']}"
                ),
            ),
        )[0]
        selected.append(donor)
    audit = {
        "same_answer": same_answer,
        "recipient_count": len(directional),
        "donor_count": len(selected),
        "all_cross_world": all(
            donor["world_id"] != item["target"]["world_id"]
            for donor, item in zip(selected, directional)
        ),
        "all_answer_contracts_hold": all(
            (donor["gold"] == item["target"]["gold"]) == same_answer
            for donor, item in zip(selected, directional)
        ),
        "different_answer_code_sets_disjoint": (
            None if same_answer else all(disjoint_checks)
        ),
        "assignments_digest": stable_order(
            canonical([
                {
                    "recipient": item["target"]["record_id"],
                    "donor": donor["record_id"],
                }
                for item, donor in zip(directional, selected)
            ]),
            f"donor-audit:{same_answer}",
        ),
    }
    return selected, audit


def prepare_batches(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
    donors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for start in range(0, len(directional), BATCH_SIZE):
        item_batch = directional[start:start + BATCH_SIZE]
        donor_batch = donors[start:start + BATCH_SIZE]
        target_cases = [item["target"] for item in item_batch]
        target_hidden = capture_depth1(model, device, target_cases)
        donor_hidden = capture_depth1(model, device, donor_batch)
        target_clean = forward_two_step(
            model,
            layers,
            device,
            target_cases,
        )
        donor_clean = forward_two_step(
            model,
            layers,
            device,
            donor_batch,
        )
        prepared.append({
            "items": item_batch,
            "target_cases": target_cases,
            "donor_cases": donor_batch,
            "target_hidden": target_hidden,
            "donor_hidden": donor_hidden,
            "target_clean": target_clean,
            "donor_clean": donor_clean,
        })
    return prepared


def contrast_margin(
    panel: torch.Tensor,
    labels: list[str],
    donor_cases: list[dict[str, Any]],
    target_cases: list[dict[str, Any]],
    step: int,
) -> torch.Tensor:
    index = {label: position for position, label in enumerate(labels)}
    batch = torch.arange(panel.shape[0])
    donor_index = torch.tensor([
        index[case["gold_parts"][step]] for case in donor_cases
    ])
    target_index = torch.tensor([
        index[case["gold_parts"][step]] for case in target_cases
    ])
    return panel[batch, donor_index] - panel[batch, target_index]


def evaluate_positions(
    model,
    layers,
    device,
    prepared: list[dict[str, Any]],
    positions: list[int],
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for batch in prepared:
        output = forward_two_step(
            model,
            layers,
            device,
            batch["target_cases"],
            source_hidden=batch["donor_hidden"],
            positions=positions,
        )
        margins = []
        target_margins = []
        donor_margins = []
        for step in (0, 1):
            labels = output["steps"][step]["labels"]
            margins.append(contrast_margin(
                output["steps"][step]["panel"],
                labels,
                batch["donor_cases"],
                batch["target_cases"],
                step,
            ))
            target_margins.append(contrast_margin(
                batch["target_clean"]["steps"][step]["panel"],
                batch["target_clean"]["steps"][step]["labels"],
                batch["donor_cases"],
                batch["target_cases"],
                step,
            ))
            donor_margins.append(contrast_margin(
                batch["donor_clean"]["steps"][step]["panel"],
                batch["donor_clean"]["steps"][step]["labels"],
                batch["donor_cases"],
                batch["target_cases"],
                step,
            ))
        for index, (target, donor) in enumerate(zip(
            batch["target_cases"],
            batch["donor_cases"],
        )):
            prediction_ids = [
                int(output["steps"][step]["prediction_ids"][index])
                for step in (0, 1)
            ]
            prediction_parts = [
                label_for_token(prediction_ids[step], target, step)
                for step in (0, 1)
            ]
            donor_semantic_ids = semantic_answer_ids(donor)
            target_semantic_ids = semantic_answer_ids(target)
            donor_hit_steps = [
                prediction_ids[step]
                == donor_semantic_ids[step]
                for step in (0, 1)
            ]
            target_hit_steps = [
                prediction_ids[step]
                == target_semantic_ids[step]
                for step in (0, 1)
            ]
            step_rows = []
            for step in (0, 1):
                margin = float(margins[step][index].item())
                target_margin = float(
                    target_margins[step][index].item()
                )
                donor_margin = float(
                    donor_margins[step][index].item()
                )
                transfer = (
                    (margin - target_margin)
                    / max(abs(donor_margin - target_margin), EPSILON)
                )
                step_rows.append({
                    "step": step,
                    "margin": margin,
                    "target_margin": target_margin,
                    "donor_margin": donor_margin,
                    "normalized_transfer": float(transfer),
                    "donor_hit": donor_hit_steps[step],
                    "target_hit": target_hit_steps[step],
                })
            rows.append({
                "schema_version": "phase1006_source_condition_row.v1",
                "phase": PHASE,
                "condition": condition,
                "record_id": target["record_id"],
                "donor_record_id": donor["record_id"],
                "prediction_ids": prediction_ids,
                "prediction_parts": prediction_parts,
                "target_parts": list(target["gold_parts"]),
                "donor_parts": list(donor["gold_parts"]),
                "donor_sequence_hit": all(donor_hit_steps),
                "target_sequence_hit": all(target_hit_steps),
                "steps": step_rows,
                "positions": list(positions),
            })
    transfers = [
        step["normalized_transfer"]
        for row in rows
        for step in row["steps"]
    ]
    summary = {
        "condition": condition,
        "n": len(rows),
        "positions": list(positions),
        "position_count": len(positions),
        "donor_sequence_rate": float(np.mean([
            row["donor_sequence_hit"] for row in rows
        ])),
        "target_sequence_rate": float(np.mean([
            row["target_sequence_hit"] for row in rows
        ])),
        "step_donor_rates": [
            float(np.mean([
                row["steps"][step]["donor_hit"] for row in rows
            ]))
            for step in (0, 1)
        ],
        "step_target_rates": [
            float(np.mean([
                row["steps"][step]["target_hit"] for row in rows
            ]))
            for step in (0, 1)
        ],
        "median_normalized_transfer": float(np.median(transfers)),
        "mean_normalized_transfer": float(np.mean(transfers)),
    }
    summary["basic_source_gate"] = (
        summary["donor_sequence_rate"] >= 0.80
        and summary["median_normalized_transfer"] >= 0.50
    )
    return summary, rows


def mediation_between(
    full_rows: list[dict[str, Any]],
    reduced_rows: list[dict[str, Any]],
) -> float:
    reduced = {row["record_id"]: row for row in reduced_rows}
    values = []
    for full in full_rows:
        other = reduced[full["record_id"]]
        for step in (0, 1):
            full_margin = float(full["steps"][step]["margin"])
            reduced_margin = float(other["steps"][step]["margin"])
            target_margin = float(full["steps"][step]["target_margin"])
            values.append(
                (full_margin - reduced_margin)
                / max(abs(full_margin - target_margin), EPSILON)
            )
    return float(np.median(values))


def release_prepared(prepared: list[dict[str, Any]]) -> None:
    for batch in prepared:
        batch.clear()
    prepared.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def source_cell(
    model,
    layers,
    device,
    model_name: str,
    split: str,
    template: int,
    output_root: Path,
    behavior: dict[str, Any],
    protocol_digest: str,
) -> dict[str, Any]:
    cell_root = output_root / split / f"template{template}"
    if not behavior["behavior_gate_pass"]:
        summary = {
            "schema_version": "phase1006_source_cell.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "template": template,
            "protocol_digest": protocol_digest,
            "source_run": False,
            "skip_reason": "behavior_gate_failed",
            "source_gate_pass": False,
        }
        write_json(cell_root / "summary.json", summary)
        return summary

    directional = [
        row
        for row in selected_directional_rows(model_name, split)
        if int(row["template"]) == template
    ]
    directional = sorted(
        directional,
        key=lambda row: stable_order(
            row["target"]["record_id"],
            f"screen:{model_name}:{split}:t{template}",
        ),
    )
    if len(directional) != 32:
        raise RuntimeError(f"source cell n={len(directional)}")
    donors, donor_audit = choose_donors(
        model_name,
        split,
        template,
        directional,
        same_answer=False,
    )
    screen_directional = directional[:SCREEN_N]
    screen_donors = donors[:SCREEN_N]
    screen = prepare_batches(
        model,
        layers,
        device,
        screen_directional,
        screen_donors,
    )
    raw_widths = {
        int(item["target"]["raw_prompt_token_count"])
        for item in screen_directional
    }
    if len(raw_widths) != 1:
        raise RuntimeError(f"raw width drift: {raw_widths}")
    raw_width = next(iter(raw_widths))
    span_starts = {
        int(item["target"]["user_content_start"])
        for item in screen_directional
    }
    span_ends = {
        int(item["target"]["user_content_end"])
        for item in screen_directional
    }
    if len(span_starts) != 1 or len(span_ends) != 1:
        raise RuntimeError(
            f"user-content span drift: {span_starts}/{span_ends}"
        )
    user_start = next(iter(span_starts))
    user_end = next(iter(span_ends))
    universe = list(range(user_start, user_end))
    event_rows = []

    full_summary, full_rows = evaluate_positions(
        model,
        layers,
        device,
        screen,
        universe,
        "screen_full_prompt",
    )
    event_rows.extend(full_rows)
    ranking_rows = []
    for position in universe:
        single_summary, single_rows = evaluate_positions(
            model,
            layers,
            device,
            screen,
            [position],
            f"screen_single_p{position:03d}",
        )
        leave_positions = [
            value for value in universe if value != position
        ]
        loo_summary, loo_rows = evaluate_positions(
            model,
            layers,
            device,
            screen,
            leave_positions,
            f"screen_loo_p{position:03d}",
        )
        event_rows.extend(single_rows)
        event_rows.extend(loo_rows)
        ranking_rows.append({
            "position": position,
            "loo_target_sequence_rate": loo_summary[
                "target_sequence_rate"
            ],
            "loo_median_mediation": mediation_between(
                full_rows,
                loo_rows,
            ),
            "single_donor_sequence_rate": single_summary[
                "donor_sequence_rate"
            ],
            "single_median_transfer": single_summary[
                "median_normalized_transfer"
            ],
        })
    ranking_rows.sort(
        key=lambda row: (
            -row["loo_target_sequence_rate"],
            -row["loo_median_mediation"],
            -row["single_donor_sequence_rate"],
            int(row["position"]),
        )
    )

    selected: list[int] = []
    build_trace = []
    for rank, ranked in enumerate(ranking_rows, start=1):
        selected.append(int(ranked["position"]))
        build_summary, build_rows = evaluate_positions(
            model,
            layers,
            device,
            screen,
            sorted(selected),
            f"screen_build_k{len(selected):03d}",
        )
        event_rows.extend(build_rows)
        build_trace.append({
            "rank": rank,
            "added_position": int(ranked["position"]),
            **build_summary,
        })
        if build_summary["basic_source_gate"]:
            break

    reverse_trace = []
    if build_trace and build_trace[-1]["basic_source_gate"]:
        for position in list(reversed(selected)):
            trial = [value for value in selected if value != position]
            trial_summary, trial_rows = evaluate_positions(
                model,
                layers,
                device,
                screen,
                sorted(trial),
                f"screen_reverse_drop_p{position:03d}",
            )
            event_rows.extend(trial_rows)
            keep_drop = bool(trial_summary["basic_source_gate"])
            reverse_trace.append({
                "position": position,
                "drop_retains_gate": keep_drop,
                **trial_summary,
            })
            if keep_drop:
                selected = trial

    frozen_positions = sorted(selected)
    release_prepared(screen)

    full_prepared = prepare_batches(
        model,
        layers,
        device,
        directional,
        donors,
    )
    frozen_summary, frozen_rows = evaluate_positions(
        model,
        layers,
        device,
        full_prepared,
        frozen_positions,
        "frozen_different_answer",
    )
    full_eval_summary, full_eval_rows = evaluate_positions(
        model,
        layers,
        device,
        full_prepared,
        universe,
        "full_prompt_different_answer",
    )
    event_rows.extend(frozen_rows)
    event_rows.extend(full_eval_rows)
    release_prepared(full_prepared)

    same_donors, same_audit = choose_donors(
        model_name,
        split,
        template,
        directional,
        same_answer=True,
    )
    same_prepared = prepare_batches(
        model,
        layers,
        device,
        directional,
        same_donors,
    )
    same_summary, same_rows = evaluate_positions(
        model,
        layers,
        device,
        same_prepared,
        frozen_positions,
        "frozen_same_answer_control",
    )
    event_rows.extend(same_rows)
    release_prepared(same_prepared)

    target_donors = [item["target"] for item in directional]
    noop_prepared = prepare_batches(
        model,
        layers,
        device,
        directional,
        target_donors,
    )
    noop_summary, noop_rows = evaluate_positions(
        model,
        layers,
        device,
        noop_prepared,
        frozen_positions,
        "frozen_target_noop",
    )
    event_rows.extend(noop_rows)
    release_prepared(noop_prepared)

    role_counts: dict[int, Counter[str]] = {
        position: Counter() for position in frozen_positions
    }
    for item in directional:
        roles = item["target"]["sealed_semantic_role_positions"]
        for position in frozen_positions:
            matched = [
                role
                for role, role_position in roles.items()
                if int(role_position) == position
            ]
            role_counts[position].update(matched or ["other"])
    role_audit = {
        str(position): dict(counter)
        for position, counter in role_counts.items()
    }

    source_gate = (
        frozen_summary["basic_source_gate"]
        and same_summary["target_sequence_rate"] >= 0.95
        and noop_summary["target_sequence_rate"] >= 0.99
    )
    summary = {
        "schema_version": "phase1006_source_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "template": template,
        "protocol_digest": protocol_digest,
        "source_run": True,
        "screen_n": SCREEN_N,
        "frozen_n": len(directional),
        "raw_prompt_width": raw_width,
        "user_content_start": user_start,
        "user_content_end": user_end,
        "event_universe_size": len(universe),
        "donor_audit": donor_audit,
        "same_answer_donor_audit": same_audit,
        "screen_full_prompt": full_summary,
        "ranked_positions": ranking_rows,
        "build_trace": build_trace,
        "reverse_delete_trace": reverse_trace,
        "frozen_positions": frozen_positions,
        "frozen_position_count": len(frozen_positions),
        "frozen_different_answer": frozen_summary,
        "full_prompt_different_answer": full_eval_summary,
        "frozen_same_answer_control": same_summary,
        "frozen_target_noop": noop_summary,
        "semantic_reconstruction_audit": role_audit,
        "semantic_labels_used_for_selection": False,
        "source_gate_pass": source_gate,
    }
    write_jsonl(cell_root / "condition_rows.jsonl", event_rows)
    write_json(cell_root / "summary.json", summary)
    print(
        f"[source] {model_name}/{split}/t{template} "
        f"k={len(frozen_positions)} "
        f"donor={frozen_summary['donor_sequence_rate']:.3f} "
        f"tau={frozen_summary['median_normalized_transfer']:.3f} "
        f"same={same_summary['target_sequence_rate']:.3f} "
        f"noop={noop_summary['target_sequence_rate']:.3f} "
        f"pass={source_gate}",
        flush=True,
    )
    return summary


def run_model(
    model_name: str,
    *,
    use_8bit: bool,
    behavior_only: bool = False,
    source_only: bool = False,
) -> dict[str, Any]:
    if behavior_only and source_only:
        raise ValueError("behavior_only and source_only are mutually exclusive")
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != 4:
        raise RuntimeError("Phase1006 protocol revision drift")
    protocol_digest = protocol["preregistration_digest"]
    precision = "8bit" if use_8bit else "bf16"
    behavior_name = "behavior" if use_8bit else "behavior_bf16"
    source_name = "blind_source" if use_8bit else "blind_source_bf16"
    behavior_root = OUT_ROOT / behavior_name / model_name
    source_root = OUT_ROOT / source_name / model_name
    frozen_behavior: dict[tuple[str, int], dict[str, Any]] = {}
    if source_only:
        behavior_model = read_json(behavior_root / "summary.json")
        if behavior_model["protocol_digest"] != protocol_digest:
            raise RuntimeError("behavior/protocol digest drift")
        frozen_behavior = {
            (str(item["split"]), int(item["template"])): item
            for item in behavior_model["cells"]
        }

    started = time.time()
    model = tokenizer = device = None
    behavior_summaries = []
    source_summaries = []
    try:
        model, tokenizer, device = load_model(
            model_name,
            use_8bit=use_8bit,
        )
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        for split in SPLITS:
            for template in TEMPLATES_BY_SPLIT[split]:
                if source_only:
                    behavior = frozen_behavior[(split, int(template))]
                else:
                    behavior = behavior_cell(
                        model,
                        layers,
                        tokenizer,
                        device,
                        model_name,
                        split,
                        int(template),
                        behavior_root,
                    )
                behavior_summaries.append(behavior)
                if not behavior_only:
                    source_summaries.append(source_cell(
                        model,
                        layers,
                        device,
                        model_name,
                        split,
                        int(template),
                        source_root,
                        behavior,
                        protocol_digest,
                    ))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        summary = {
            "schema_version": "phase1006_source_model.v1",
            "phase": PHASE,
            "model": model_name,
            "precision": precision,
            "protocol_digest": protocol_digest,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
            },
            "behavior_cells": behavior_summaries,
            "source_cells": source_summaries,
            "behavior_gate_pass_count": sum(
                item["behavior_gate_pass"]
                for item in behavior_summaries
            ),
            "source_gate_pass_count": sum(
                item["source_gate_pass"]
                for item in source_summaries
            ),
            "behavior_only": behavior_only,
            "source_only": source_only,
            "elapsed_seconds": time.time() - started,
        }
        if not behavior_only:
            write_json(source_root / "summary.json", summary)
        write_json(behavior_root / "summary.json", {
            "schema_version": "phase1006_behavior_model.v1",
            "phase": PHASE,
            "model": model_name,
            "precision": precision,
            "protocol_digest": protocol_digest,
            "cells": behavior_summaries,
            "gate_pass_count": summary["behavior_gate_pass_count"],
        })
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--behavior-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.bf16 and args.model != "qwen3":
        raise SystemExit("Formal BF16 audit is restricted to qwen3")
    summary = run_model(
        args.model,
        use_8bit=not args.bf16,
        behavior_only=args.behavior_only,
        source_only=args.source_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
