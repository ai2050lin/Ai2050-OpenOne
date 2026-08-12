#!/usr/bin/env python3
"""Discover and test Phase1007 role-aligned causal source sets."""
from __future__ import annotations

import argparse
import gc
import json
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
from phase1007_role_aligned_causal_source_protocol import (
    CONTRASTS,
    MODELS,
    OUT_ROOT,
    PHASE,
    SOURCE_DEPTH,
    SPLITS,
    TEMPLATES_BY_SPLIT,
    decision_case,
    read_json,
    selected_directional_rows,
    semantic_answer_ids,
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
    logical_step: int,
    semantic_prefixes: list[list[int]],
    *,
    positions: list[int] | None = None,
    operation: str | None = None,
    source_a: torch.Tensor | None = None,
    source_b: torch.Tensor | None = None,
) -> dict[str, Any]:
    step_cases = [
        decision_case(
            case,
            semantic_prefix=semantic_prefixes[index],
            logical_step=logical_step,
        )
        for index, case in enumerate(cases)
    ]
    input_ids, attention = case_tensors(step_cases, device)
    handle = None
    hook_count = [0]
    frozen_positions = sorted({int(value) for value in positions or []})
    if operation is not None:
        if source_a is None:
            raise ValueError("source_a is required")
        if operation == "delta" and source_b is None:
            raise ValueError("source_b is required for delta")

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            raw_width = source_a.shape[1]
            if value.shape[1] < raw_width:
                return output
            patched = value.clone()
            if frozen_positions:
                if operation == "replace":
                    patched[:, frozen_positions, :] = source_a[
                        :, frozen_positions, :
                    ].to(device=value.device, dtype=value.dtype)
                elif operation == "delta":
                    delta = (
                        source_a[:, frozen_positions, :]
                        - source_b[:, frozen_positions, :]
                    ).to(device=value.device, dtype=value.dtype)
                    patched[:, frozen_positions, :] = (
                        value[:, frozen_positions, :] + delta
                    )
                else:
                    raise KeyError(operation)
            hook_count[0] += 1
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
        if operation is not None and hook_count[0] != 1:
            raise RuntimeError(f"source hook drift: {hook_count[0]}")
        logits = output.logits[:, -1, :]
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        labels, panel = candidate_panel(logits, cases, logical_step)
        del output, logits
        return {
            "prediction_ids": [int(value) for value in predictions],
            "labels": labels,
            "panel": panel,
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
    positions: list[int] | None = None,
    operation: str | None = None,
    source_a: torch.Tensor | None = None,
    source_b: torch.Tensor | None = None,
) -> dict[str, Any]:
    step0 = forward_step(
        model,
        layers,
        device,
        cases,
        0,
        [[] for _ in cases],
        positions=positions,
        operation=operation,
        source_a=source_a,
        source_b=source_b,
    )
    step1 = forward_step(
        model,
        layers,
        device,
        cases,
        1,
        [[value] for value in step0["prediction_ids"]],
        positions=positions,
        operation=operation,
        source_a=source_a,
        source_b=source_b,
    )
    return {"steps": [step0, step1]}


def prepare_batches(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for batch_rows in chunks(directional, BATCH_SIZE):
        case_keys = (
            "target",
            "within_donor",
            "cross_same",
            "cross_different",
            "nuisance2_same",
        )
        cases = {
            key: [item[key] for item in batch_rows]
            for key in case_keys
        }
        hidden = {
            key: capture_depth1(model, device, value)
            for key, value in cases.items()
        }
        prepared.append({
            "items": batch_rows,
            "cases": cases,
            "hidden": hidden,
            "target_clean": forward_two_step(
                model, layers, device, cases["target"]
            ),
            "donor_clean": forward_two_step(
                model, layers, device, cases["within_donor"]
            ),
        })
    return prepared


def release_prepared(prepared: list[dict[str, Any]]) -> None:
    for batch in prepared:
        batch.clear()
    prepared.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def evaluate_condition(
    model,
    layers,
    device,
    prepared: list[dict[str, Any]],
    positions: list[int],
    condition: str,
    *,
    operation: str,
    source_a_key: str,
    source_b_key: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for batch in prepared:
        target_cases = batch["cases"]["target"]
        donor_cases = batch["cases"]["within_donor"]
        output = forward_two_step(
            model,
            layers,
            device,
            target_cases,
            positions=positions,
            operation=operation,
            source_a=batch["hidden"][source_a_key],
            source_b=(
                None
                if source_b_key is None
                else batch["hidden"][source_b_key]
            ),
        )
        margins = []
        target_margins = []
        donor_margins = []
        for step in (0, 1):
            margins.append(contrast_margin(
                output["steps"][step]["panel"],
                output["steps"][step]["labels"],
                donor_cases,
                target_cases,
                step,
            ))
            target_margins.append(contrast_margin(
                batch["target_clean"]["steps"][step]["panel"],
                batch["target_clean"]["steps"][step]["labels"],
                donor_cases,
                target_cases,
                step,
            ))
            donor_margins.append(contrast_margin(
                batch["donor_clean"]["steps"][step]["panel"],
                batch["donor_clean"]["steps"][step]["labels"],
                donor_cases,
                target_cases,
                step,
            ))
        for index, (target, donor) in enumerate(
            zip(target_cases, donor_cases)
        ):
            prediction_ids = [
                int(output["steps"][step]["prediction_ids"][index])
                for step in (0, 1)
            ]
            target_ids = semantic_answer_ids(target)
            donor_ids = semantic_answer_ids(donor)
            donor_hits = [
                prediction_ids[step] == donor_ids[step]
                for step in (0, 1)
            ]
            target_hits = [
                prediction_ids[step] == target_ids[step]
                for step in (0, 1)
            ]
            steps = []
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
                    / max(
                        abs(donor_margin - target_margin),
                        EPSILON,
                    )
                )
                steps.append({
                    "step": step,
                    "margin": margin,
                    "target_margin": target_margin,
                    "donor_margin": donor_margin,
                    "normalized_transfer": float(transfer),
                    "donor_hit": donor_hits[step],
                    "target_hit": target_hits[step],
                })
            rows.append({
                "schema_version": (
                    "phase1007_source_condition_row.v1"
                ),
                "phase": PHASE,
                "condition": condition,
                "record_id": target["record_id"],
                "donor_record_id": donor["record_id"],
                "prediction_ids": prediction_ids,
                "donor_sequence_hit": all(donor_hits),
                "target_sequence_hit": all(target_hits),
                "steps": steps,
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
    summary["basic_transfer_gate"] = (
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


def role_audit(
    directional: list[dict[str, Any]],
    positions: list[int],
) -> dict[str, Any]:
    counts = {position: Counter() for position in positions}
    for item in directional:
        roles = item["target"]["sealed_semantic_role_positions"]
        for position in positions:
            matched = [
                role
                for role, role_position in roles.items()
                if int(role_position) == position
            ]
            counts[position].update(matched or ["other"])
    return {
        str(position): dict(counter)
        for position, counter in counts.items()
    }


def source_cell(
    model,
    layers,
    device,
    model_name: str,
    split: str,
    template: int,
    contrast: str,
    behavior: dict[str, Any],
    protocol_digest: str,
) -> dict[str, Any]:
    cell_root = (
        OUT_ROOT
        / "source"
        / model_name
        / split
        / f"template{template}"
        / contrast
    )
    if not behavior["behavior_gate_pass"]:
        summary = {
            "schema_version": "phase1007_source_cell.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "template": template,
            "contrast": contrast,
            "protocol_digest": protocol_digest,
            "source_run": False,
            "skip_reason": "behavior_gate_failed",
            "whole_source_gate": False,
            "delta_source_gate": False,
        }
        write_json(cell_root / "summary.json", summary)
        return summary

    directional = selected_directional_rows(
        model_name, split, template, contrast, "formal"
    )
    directional = sorted(
        directional,
        key=lambda row: stable_order(
            row["target"]["record_id"],
            f"screen:{model_name}:{split}:t{template}:{contrast}",
        ),
    )
    screen_directional = directional[:SCREEN_N]
    screen = prepare_batches(
        model, layers, device, screen_directional
    )
    starts = {
        int(item["target"]["user_content_start"])
        for item in screen_directional
    }
    ends = {
        int(item["target"]["user_content_end"])
        for item in screen_directional
    }
    if len(starts) != 1 or len(ends) != 1:
        raise RuntimeError("user span drift")
    user_start = next(iter(starts))
    user_end = next(iter(ends))
    universe = list(range(user_start, user_end))
    condition_rows = []

    full_summary, full_rows = evaluate_condition(
        model,
        layers,
        device,
        screen,
        universe,
        "screen_full_within_minimal",
        operation="replace",
        source_a_key="within_donor",
    )
    condition_rows.extend(full_rows)
    ranking = []
    for position in universe:
        single_summary, single_rows = evaluate_condition(
            model,
            layers,
            device,
            screen,
            [position],
            f"screen_single_p{position:03d}",
            operation="replace",
            source_a_key="within_donor",
        )
        loo_positions = [
            value for value in universe if value != position
        ]
        loo_summary, loo_rows = evaluate_condition(
            model,
            layers,
            device,
            screen,
            loo_positions,
            f"screen_loo_p{position:03d}",
            operation="replace",
            source_a_key="within_donor",
        )
        condition_rows.extend(single_rows)
        condition_rows.extend(loo_rows)
        ranking.append({
            "position": position,
            "loo_target_sequence_rate": loo_summary[
                "target_sequence_rate"
            ],
            "loo_median_mediation": mediation_between(
                full_rows, loo_rows
            ),
            "single_donor_sequence_rate": single_summary[
                "donor_sequence_rate"
            ],
            "single_median_transfer": single_summary[
                "median_normalized_transfer"
            ],
        })
    ranking.sort(
        key=lambda row: (
            -row["loo_target_sequence_rate"],
            -row["loo_median_mediation"],
            -row["single_donor_sequence_rate"],
            int(row["position"]),
        )
    )

    selected: list[int] = []
    build_trace = []
    for rank_index, item in enumerate(ranking, start=1):
        selected.append(int(item["position"]))
        build_summary, build_rows = evaluate_condition(
            model,
            layers,
            device,
            screen,
            sorted(selected),
            f"screen_build_k{len(selected):03d}",
            operation="replace",
            source_a_key="within_donor",
        )
        condition_rows.extend(build_rows)
        build_trace.append({
            "rank": rank_index,
            "added_position": int(item["position"]),
            **build_summary,
        })
        if build_summary["basic_transfer_gate"]:
            break

    reverse_trace = []
    if build_trace and build_trace[-1]["basic_transfer_gate"]:
        for position in list(reversed(selected)):
            trial = [value for value in selected if value != position]
            trial_summary, trial_rows = evaluate_condition(
                model,
                layers,
                device,
                screen,
                sorted(trial),
                f"screen_reverse_drop_p{position:03d}",
                operation="replace",
                source_a_key="within_donor",
            )
            condition_rows.extend(trial_rows)
            keep_drop = bool(trial_summary["basic_transfer_gate"])
            reverse_trace.append({
                "position": position,
                "drop_retains_gate": keep_drop,
                **trial_summary,
            })
            if keep_drop:
                selected = trial
    frozen_positions = sorted(selected)
    release_prepared(screen)

    full = prepare_batches(model, layers, device, directional)
    condition_specs = (
        (
            "within_minimal_replace",
            "replace",
            "within_donor",
            None,
        ),
        (
            "cross_world_whole",
            "replace",
            "cross_different",
            None,
        ),
        (
            "cross_world_same_answer_whole",
            "replace",
            "cross_same",
            None,
        ),
        (
            "causal_delta",
            "delta",
            "cross_different",
            "cross_same",
        ),
        (
            "nuisance_delta",
            "delta",
            "nuisance2_same",
            "cross_same",
        ),
        (
            "target_noop",
            "replace",
            "target",
            None,
        ),
    )
    final_conditions = {}
    for name, operation, source_a, source_b in condition_specs:
        condition_summary, rows = evaluate_condition(
            model,
            layers,
            device,
            full,
            frozen_positions,
            name,
            operation=operation,
            source_a_key=source_a,
            source_b_key=source_b,
        )
        final_conditions[name] = condition_summary
        condition_rows.extend(rows)
    release_prepared(full)

    within = final_conditions["within_minimal_replace"]
    cross_whole = final_conditions["cross_world_whole"]
    same_whole = final_conditions[
        "cross_world_same_answer_whole"
    ]
    causal_delta = final_conditions["causal_delta"]
    nuisance_delta = final_conditions["nuisance_delta"]
    noop = final_conditions["target_noop"]
    whole_gate = (
        within["basic_transfer_gate"]
        and cross_whole["basic_transfer_gate"]
        and same_whole["target_sequence_rate"] >= 0.95
        and noop["target_sequence_rate"] >= 0.99
    )
    delta_gate = (
        within["basic_transfer_gate"]
        and causal_delta["basic_transfer_gate"]
        and nuisance_delta["target_sequence_rate"] >= 0.95
        and noop["target_sequence_rate"] >= 0.99
    )
    summary = {
        "schema_version": "phase1007_source_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "split": split,
        "template": template,
        "contrast": contrast,
        "protocol_digest": protocol_digest,
        "source_run": True,
        "screen_n": SCREEN_N,
        "frozen_n": len(directional),
        "user_content_start": user_start,
        "user_content_end": user_end,
        "event_universe_size": len(universe),
        "screen_full_within_minimal": full_summary,
        "ranking": ranking,
        "build_trace": build_trace,
        "reverse_delete_trace": reverse_trace,
        "frozen_positions": frozen_positions,
        "frozen_position_count": len(frozen_positions),
        "final_conditions": final_conditions,
        "semantic_role_audit": role_audit(
            directional, frozen_positions
        ),
        "semantic_labels_used_for_selection": False,
        "whole_source_gate": whole_gate,
        "delta_source_gate": delta_gate,
    }
    write_jsonl(cell_root / "condition_rows.jsonl", condition_rows)
    write_json(cell_root / "summary.json", summary)
    print(
        f"[source] {model_name}/{split}/t{template}/{contrast} "
        f"k={len(frozen_positions)} "
        f"within={within['donor_sequence_rate']:.3f} "
        f"whole={cross_whole['donor_sequence_rate']:.3f} "
        f"delta={causal_delta['donor_sequence_rate']:.3f} "
        f"nuisance={nuisance_delta['target_sequence_rate']:.3f} "
        f"whole_gate={whole_gate} delta_gate={delta_gate}",
        flush=True,
    )
    return summary


def run_model(model_name: str, split: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    protocol_digest = protocol["preregistration_digest"]
    behavior_model = read_json(
        OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    if behavior_model["protocol_digest"] != protocol_digest:
        raise RuntimeError("behavior/protocol digest drift")
    behavior_lookup = {
        (
            item["split"],
            int(item["template"]),
            item["contrast"],
        ): item
        for item in behavior_model["cells"]
    }
    started = time.time()
    model = tokenizer = device = None
    cells = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        for template in TEMPLATES_BY_SPLIT[split]:
            for contrast in CONTRASTS:
                behavior = behavior_lookup[
                    (split, int(template), contrast)
                ]
                cells.append(source_cell(
                    model,
                    layers,
                    device,
                    model_name,
                    split,
                    int(template),
                    contrast,
                    behavior,
                    protocol_digest,
                ))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = {
            "schema_version": "phase1007_source_model_split.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "precision": "8bit",
            "protocol_digest": protocol_digest,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
            },
            "cells": cells,
            "source_run_count": sum(
                item["source_run"] for item in cells
            ),
            "whole_gate_pass_count": sum(
                item["whole_source_gate"] for item in cells
            ),
            "delta_gate_pass_count": sum(
                item["delta_source_gate"] for item in cells
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_json(
            OUT_ROOT
            / "source"
            / model_name
            / split
            / "summary.json",
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
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    summary = run_model(args.model, args.split)
    compact = {
        "phase": PHASE,
        "model": args.model,
        "split": args.split,
        "source_run_count": summary["source_run_count"],
        "whole_gate_pass_count": summary["whole_gate_pass_count"],
        "delta_gate_pass_count": summary["delta_gate_pass_count"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
