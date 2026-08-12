#!/usr/bin/env python3
"""Phase 1001 greedy minimum joint head cut with disjoint confirmation.

Validation starts from the frozen 12-head set and removes one head at a time.
Each deletion is evaluated on all validation directions for both necessity
and sufficiency. The smallest greedy prefix passing candidate and natural
gates is frozen. Confirmation never changes that set and tests leave-one-out
deletions plus no-op, wrong-role, and cross-pair controls.
"""
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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import COLORS, MODEL
from phase1000_scpg_discovery import (
    batches_by_template,
    candidate_tensor,
    capture_residuals,
    case_tensors,
    prediction_colors,
    register_source_patch,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1000_source_control_audit import valid_derangement_shifts
from phase1001_attention_head_discovery import (
    HEAD_COUNT,
    HEAD_DIM,
    RESULT_ROOT,
    SOURCE_DEPTH,
    capture_attention_states,
    forward_with_patches,
    generate_with_patches,
    read_json,
    write_json,
)
from phase1001_attention_source_path_decomposition import selected_inputs


CUT_ROOT = RESULT_ROOT / "minimum_head_cut"
HEAD_DISCOVERY_ROOT = RESULT_ROOT / "head_discovery"
CUT_THRESHOLDS = {
    "median_mediation": 0.50,
    "mean_sufficiency": 0.50,
    "natural_target_rate": 0.50,
    "max_wrong_role_mediation": 0.10,
    "max_cross_pair_mediation": 0.10,
    "source_do_flip_rate": 0.90,
}


def subset_id(events: list[dict[str, Any]]) -> str:
    return "+".join(event["event_id"] for event in events)


def capture_role_heads(
    model,
    layers,
    device,
    rows,
    candidate_ids,
    role,
    source_patch=None,
):
    input_ids, attention_mask = case_tensors(rows, device)
    heads = {}
    handles = []
    source_handle = None
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        for layer_number in (25, 30, 31):
            positions = torch.tensor(
                [row["role_positions"][role] for row in rows],
                dtype=torch.long,
                device=device,
            )
            counter = [0]

            def make_hook(number, pos, count):
                def hook(module, args):
                    value = args[0]
                    batch_index = torch.arange(
                        value.shape[0], device=value.device
                    )
                    heads[number] = (
                        value[batch_index, pos.to(value.device), :]
                        .reshape(value.shape[0], HEAD_COUNT, HEAD_DIM)
                        .detach()
                    )
                    count[0] += 1

                return hook

            handles.append(
                layers[layer_number - 1]
                .self_attn.o_proj.register_forward_pre_hook(
                    make_hook(layer_number, positions, counter)
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        candidates = candidate_tensor(
            output.logits[:, -1, :], candidate_ids
        ).detach()
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError("source role capture count drift")
        del output, input_ids, attention_mask
        return candidates, heads
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def forward_role_restore(
    model,
    layers,
    device,
    rows,
    candidate_ids,
    events,
    vectors_by_layer,
    source_patch,
    role,
):
    input_ids, attention_mask = case_tensors(rows, device)
    grouped = defaultdict(list)
    for event in events:
        grouped[int(event["layer_number"])].append(event)
    source_handle = None
    handles = []
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        for layer_number, layer_events in grouped.items():
            positions = torch.tensor(
                [row["role_positions"][role] for row in rows],
                dtype=torch.long,
                device=device,
            )
            counter = [0]

            def make_hook(items, pos, count, number):
                def hook(module, args):
                    value = args[0]
                    patched = value.clone()
                    batch_index = torch.arange(
                        value.shape[0], device=value.device
                    )
                    for event in items:
                        head = int(event["head_index"])
                        start = head * HEAD_DIM
                        stop = start + HEAD_DIM
                        patched[
                            batch_index,
                            pos.to(value.device),
                            start:stop,
                        ] = vectors_by_layer[number][
                            :, head, :
                        ].to(device=value.device, dtype=value.dtype)
                    count[0] += 1
                    return (patched,) + tuple(args[1:])

                return hook

            handles.append(
                layers[layer_number - 1]
                .self_attn.o_proj.register_forward_pre_hook(
                    make_hook(
                        layer_events,
                        positions,
                        counter,
                        layer_number,
                    )
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        candidates = candidate_tensor(
            output.logits[:, -1, :], candidate_ids
        ).detach()
        if source_count[0] != 1:
            raise RuntimeError("source wrong-role count drift")
        del output, input_ids, attention_mask
        return candidates
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def evaluate_sets_batch(
    model,
    layers,
    device,
    batch,
    sets,
    candidate_ids,
    source_logits,
    target_logits,
    do_logits,
    source_patch,
    target_heads,
    do_heads,
):
    target_cases = [item["target"] for item in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    rows = []
    for events in sets:
        restore_patches = [
            {
                "event": event,
                "vectors": target_heads[int(event["layer_number"])][
                    :, int(event["head_index"]), :
                ],
            }
            for event in events
        ]
        sufficiency_patches = [
            {
                "event": event,
                "vectors": do_heads[int(event["layer_number"])][
                    :, int(event["head_index"]), :
                ],
            }
            for event in events
        ]
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=restore_patches,
        )
        sufficiency_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=sufficiency_patches,
        )
        restored_margin = semantic_margin(restored_logits, batch)
        sufficiency_margin = semantic_margin(sufficiency_logits, batch)
        restored_prediction = prediction_colors(restored_logits)
        sufficiency_prediction = prediction_colors(sufficiency_logits)
        identifier = subset_id(events)
        event_ids = [event["event_id"] for event in events]
        for index, item in enumerate(batch):
            do_effect = float(do_margin[index] - target_margin[index])
            natural_effect = float(
                source_margin[index] - target_margin[index]
            )
            rows.append(
                {
                    "schema_version": "phase1001_head_cut_candidate.v1",
                    "phase": 1001,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["target"]["template"],
                    "subset_id": identifier,
                    "event_ids": event_ids,
                    "size": len(events),
                    "mediation_fraction": float(
                        (do_margin[index] - restored_margin[index])
                        / max(abs(do_effect), 1e-8)
                    ),
                    "sufficiency_transfer": float(
                        (
                            sufficiency_margin[index]
                            - target_margin[index]
                        )
                        / max(abs(natural_effect), 1e-8)
                    ),
                    "restored_to_target": (
                        restored_prediction[index]
                        == item["target"]["gold"]
                    ),
                    "sufficiency_flipped": (
                        sufficiency_prediction[index]
                        == item["source"]["gold"]
                    ),
                }
            )
        del restored_logits, sufficiency_logits
    return rows


def summarize_sets(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subset_id"]].append(row)
    return {
        identifier: {
            "subset_id": identifier,
            "event_ids": values[0]["event_ids"],
            "size": values[0]["size"],
            "n": len(values),
            "median_mediation_fraction": float(
                np.median(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "mean_mediation_fraction": float(
                np.mean(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "mean_sufficiency_transfer": float(
                np.mean(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "median_sufficiency_transfer": float(
                np.median(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "template_median_mediation": {
                str(template): float(
                    np.median(
                        [
                            row["mediation_fraction"]
                            for row in values
                            if int(row["template"]) == template
                        ]
                    )
                )
                for template in range(4)
            },
        }
        for identifier, values in groups.items()
    }


def passes_candidate(metric):
    return bool(
        metric["median_mediation_fraction"]
        >= CUT_THRESHOLDS["median_mediation"]
        and metric["mean_sufficiency_transfer"]
        >= CUT_THRESHOLDS["mean_sufficiency"]
    )


def capture_batch_states(
    model,
    layers,
    device,
    batch,
    candidate_ids,
):
    source_cases = [item["source"] for item in batch]
    target_cases = [item["target"] for item in batch]
    source_logits, source_residuals = capture_residuals(
        model,
        device,
        source_cases,
        (SOURCE_DEPTH,),
        candidate_ids,
    )
    target_logits, target_heads, _ = capture_attention_states(
        model, layers, device, target_cases, candidate_ids
    )
    source_patch = source_patch_spec(
        SOURCE_DEPTH,
        target_cases,
        source_residuals[SOURCE_DEPTH],
        "joint",
    )
    do_logits, do_heads, _ = capture_attention_states(
        model,
        layers,
        device,
        target_cases,
        candidate_ids,
        source_patch=source_patch,
    )
    return (
        source_logits,
        target_logits,
        do_logits,
        source_patch,
        target_heads,
        do_heads,
        source_residuals,
    )


def evaluate_sets(
    model,
    layers,
    device,
    directional,
    sets,
    candidate_ids,
    batch_size,
    label,
):
    rows = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        (
            source_logits,
            target_logits,
            do_logits,
            source_patch,
            target_heads,
            do_heads,
            source_residuals,
        ) = capture_batch_states(
            model, layers, device, batch, candidate_ids
        )
        rows.extend(
            evaluate_sets_batch(
                model,
                layers,
                device,
                batch,
                sets,
                candidate_ids,
                source_logits,
                target_logits,
                do_logits,
                source_patch,
                target_heads,
                do_heads,
            )
        )
        del (
            source_logits,
            target_logits,
            do_logits,
            source_patch,
            target_heads,
            do_heads,
            source_residuals,
        )
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(
                f"[head-cut-{label}] {batch_number}/{len(batches)}",
                flush=True,
            )
    return rows


def natural_rows(
    model,
    layers,
    tokenizer,
    device,
    directional,
    path_sets,
    candidate_ids,
    effective_eos,
    batch_size,
    budget,
):
    rows = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [item["source"] for item in batch]
        target_cases = [item["target"] for item in batch]
        _, source_residuals = capture_residuals(
            model,
            device,
            source_cases,
            (SOURCE_DEPTH,),
            candidate_ids,
        )
        target_logits, target_heads, _ = capture_attention_states(
            model, layers, device, target_cases, candidate_ids
        )
        del target_logits
        source_patch = source_patch_spec(
            SOURCE_DEPTH,
            target_cases,
            source_residuals[SOURCE_DEPTH],
            "joint",
        )
        conditions = [("source_do", None, [])]
        for events in path_sets:
            patches = [
                {
                    "event": event,
                    "vectors": target_heads[
                        int(event["layer_number"])
                    ][:, int(event["head_index"]), :],
                }
                for event in events
            ]
            conditions.append(
                ("source_plus_head_cut_restore", events, patches)
            )
        for condition, events, patches in conditions:
            generated = generate_with_patches(
                model,
                layers,
                tokenizer,
                device,
                target_cases,
                source_patch,
                patches,
                {},
                effective_eos,
                budget,
            )
            for index, item in enumerate(batch):
                result = generated[index]
                rows.append(
                    {
                        "schema_version": (
                            "phase1001_head_cut_natural.v1"
                        ),
                        "phase": 1001,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "subset_id": (
                            subset_id(events) if events else None
                        ),
                        "size": len(events) if events else None,
                        "prediction": result["prediction"],
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "flipped_to_source": (
                            result["prediction"]
                            == item["source"]["gold"]
                        ),
                        "restored_to_target": (
                            result["prediction"]
                            == item["target"]["gold"]
                        ),
                        "eos_seen": result["eos_seen"],
                        "exact_short": result["exact_short"],
                    }
                )
        del source_residuals, target_heads
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(
                f"[head-cut-natural] {batch_number}/{len(batches)}",
                flush=True,
            )
    return rows


def summarize_natural(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["condition"]
            if row["subset_id"] is None
            else row["subset_id"]
        )
        groups[key].append(row)
    return {
        key: {
            "condition": values[0]["condition"],
            "subset_id": values[0]["subset_id"],
            "size": values[0]["size"],
            "n": len(values),
            "flip_rate": float(
                np.mean([row["flipped_to_source"] for row in values])
            ),
            "target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "eos_rate": float(
                np.mean([row["eos_seen"] for row in values])
            ),
            "exact_short_rate": float(
                np.mean([row["exact_short"] for row in values])
            ),
        }
        for key, values in groups.items()
    }


def control_rows(
    model,
    layers,
    device,
    directional,
    frozen_events,
    candidate_ids,
    batch_size,
):
    rows = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        target_cases = [item["target"] for item in batch]
        (
            source_logits,
            target_logits,
            do_logits,
            source_patch,
            target_heads,
            do_heads,
            source_residuals,
        ) = capture_batch_states(
            model, layers, device, batch, candidate_ids
        )
        _, target_query_heads = capture_role_heads(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            "query_name",
        )
        wrong_role_logits = forward_role_restore(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            frozen_events,
            target_query_heads,
            source_patch,
            "query_name",
        )
        shift = valid_derangement_shifts(batch, 1)[0]
        cross_patches = [
            {
                "event": event,
                "vectors": torch.roll(
                    target_heads[int(event["layer_number"])][
                        :, int(event["head_index"]), :
                    ],
                    shifts=shift,
                    dims=0,
                ),
            }
            for event in frozen_events
        ]
        cross_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=cross_patches,
        )
        noop_patches = [
            {
                "event": event,
                "vectors": do_heads[int(event["layer_number"])][
                    :, int(event["head_index"]), :
                ],
            }
            for event in frozen_events
        ]
        noop_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=noop_patches,
        )
        source_margin = semantic_margin(source_logits, batch)
        target_margin = semantic_margin(target_logits, batch)
        do_margin = semantic_margin(do_logits, batch)
        wrong_margin = semantic_margin(wrong_role_logits, batch)
        cross_margin = semantic_margin(cross_logits, batch)
        noop_margin = semantic_margin(noop_logits, batch)
        for index, item in enumerate(batch):
            denominator = float(
                do_margin[index] - target_margin[index]
            )
            rows.append(
                {
                    "schema_version": "phase1001_head_cut_control.v1",
                    "phase": 1001,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "wrong_role_mediation": float(
                        (do_margin[index] - wrong_margin[index])
                        / max(abs(denominator), 1e-8)
                    ),
                    "cross_pair_mediation": float(
                        (do_margin[index] - cross_margin[index])
                        / max(abs(denominator), 1e-8)
                    ),
                    "noop_mediation": float(
                        (do_margin[index] - noop_margin[index])
                        / max(abs(denominator), 1e-8)
                    ),
                }
            )
        del (
            source_logits,
            target_logits,
            do_logits,
            source_patch,
            target_heads,
            do_heads,
            source_residuals,
            target_query_heads,
            wrong_role_logits,
            cross_logits,
            noop_logits,
        )
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(
                f"[head-cut-controls] {batch_number}/{len(batches)}",
                flush=True,
            )
    return rows


def summarize_controls(rows):
    return {
        "n": len(rows),
        "median_wrong_role_mediation": float(
            np.median([row["wrong_role_mediation"] for row in rows])
        ),
        "mean_wrong_role_mediation": float(
            np.mean([row["wrong_role_mediation"] for row in rows])
        ),
        "median_cross_pair_mediation": float(
            np.median([row["cross_pair_mediation"] for row in rows])
        ),
        "mean_cross_pair_mediation": float(
            np.mean([row["cross_pair_mediation"] for row in rows])
        ),
        "max_abs_noop_mediation": float(
            np.max(np.abs([row["noop_mediation"] for row in rows]))
        ),
    }


def run(stage, batch_size, natural_budget):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 head cut requires CUDA")
    protocol, selected_pairs, directional, _ = selected_inputs(stage)
    output_root = CUT_ROOT / stage
    output_root.mkdir(parents=True, exist_ok=True)
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    frozen_head_spec = read_json(HEAD_DISCOVERY_ROOT / "frozen_spec.json")
    event_lookup = {
        event_id: {
            "event_id": event_id,
            "layer_number": int(event_id.split(".")[0][1:]),
            "head_index": int(event_id.split(".")[1][1:]),
            "role": "answer_boundary",
        }
        for event_id in frozen_head_spec["frozen_joint_event_ids"]
    }
    initial_events = [
        event_lookup[event_id]
        for event_id in frozen_head_spec["frozen_joint_event_ids"]
    ]
    if stage == "confirmation":
        discovery_spec = read_json(
            CUT_ROOT / "discovery" / "frozen_spec.json"
        )
        frozen_events = [
            event_lookup[event_id]
            for event_id in discovery_spec["frozen_event_ids"]
        ]
    else:
        discovery_spec = None
        frozen_events = []

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        effective_eos = eos_ids(model, tokenizer)

        all_candidate_rows = []
        greedy_path = []
        if stage == "discovery":
            current = initial_events
            initial_rows = evaluate_sets(
                model,
                layers,
                device,
                directional,
                [current],
                candidate_ids,
                batch_size,
                "initial",
            )
            all_candidate_rows.extend(initial_rows)
            initial_metric = summarize_sets(initial_rows)[subset_id(current)]
            greedy_path.append(initial_metric)
            while len(current) > 1:
                candidates = [
                    [
                        event
                        for event in current
                        if event["event_id"] != removed["event_id"]
                    ]
                    for removed in current
                ]
                round_rows = evaluate_sets(
                    model,
                    layers,
                    device,
                    directional,
                    candidates,
                    candidate_ids,
                    batch_size,
                    f"remove-from-{len(current)}",
                )
                all_candidate_rows.extend(round_rows)
                round_summary = summarize_sets(round_rows)
                passing = [
                    metric
                    for metric in round_summary.values()
                    if passes_candidate(metric)
                ]
                if not passing:
                    break
                passing.sort(
                    key=lambda metric: (
                        -min(
                            metric["median_mediation_fraction"],
                            metric["mean_sufficiency_transfer"],
                        ),
                        -metric["median_mediation_fraction"],
                        metric["subset_id"],
                    )
                )
                chosen = passing[0]
                current = [
                    event_lookup[event_id]
                    for event_id in chosen["event_ids"]
                ]
                greedy_path.append(chosen)
            path_sets = [
                [event_lookup[event_id] for event_id in item["event_ids"]]
                for item in greedy_path
            ]
        else:
            leave_one_out = [
                [
                    event
                    for event in frozen_events
                    if event["event_id"] != removed["event_id"]
                ]
                for removed in frozen_events
            ]
            path_sets = [frozen_events]
            all_candidate_rows = evaluate_sets(
                model,
                layers,
                device,
                directional,
                [frozen_events, *leave_one_out],
                candidate_ids,
                batch_size,
                "confirmation",
            )
            candidate_summary_now = summarize_sets(all_candidate_rows)
            greedy_path = [
                candidate_summary_now[subset_id(frozen_events)]
            ]

        natural = natural_rows(
            model,
            layers,
            tokenizer,
            device,
            directional,
            path_sets,
            candidate_ids,
            effective_eos,
            batch_size,
            natural_budget,
        )
        natural_summary = summarize_natural(natural)
        candidate_summary = summarize_sets(all_candidate_rows)

        if stage == "discovery":
            eligible = [
                item
                for item in greedy_path
                if passes_candidate(item)
                and natural_summary[item["subset_id"]]["target_rate"]
                >= CUT_THRESHOLDS["natural_target_rate"]
            ]
            eligible.sort(key=lambda item: item["size"])
            if not eligible:
                frozen_metric = greedy_path[0]
                natural_gate = False
            else:
                frozen_metric = eligible[0]
                natural_gate = True
            frozen_events = [
                event_lookup[event_id]
                for event_id in frozen_metric["event_ids"]
            ]
        else:
            frozen_metric = candidate_summary[subset_id(frozen_events)]
            natural_gate = (
                natural_summary[subset_id(frozen_events)]["target_rate"]
                >= CUT_THRESHOLDS["natural_target_rate"]
            )

        controls = control_rows(
            model,
            layers,
            device,
            directional,
            frozen_events,
            candidate_ids,
            batch_size,
        )
        control_summary = summarize_controls(controls)
        write_rows(
            output_root / "candidate_rows.jsonl", all_candidate_rows
        )
        write_rows(output_root / "natural_rows.jsonl", natural)
        write_rows(output_root / "control_rows.jsonl", controls)
        write_json(
            output_root / "candidate_summary.json", candidate_summary
        )
        write_json(
            output_root / "natural_summary.json", natural_summary
        )
        write_json(
            output_root / "control_summary.json", control_summary
        )

        frozen_id = subset_id(frozen_events)
        if stage == "confirmation":
            deletion_metrics = [
                metric
                for identifier, metric in candidate_summary.items()
                if identifier != frozen_id
            ]
            no_single_deletion_passes = not any(
                passes_candidate(metric) for metric in deletion_metrics
            )
        else:
            no_single_deletion_passes = True
            deletion_metrics = []
        gate_checks = {
            "candidate_mediation": (
                frozen_metric["median_mediation_fraction"]
                >= CUT_THRESHOLDS["median_mediation"]
            ),
            "candidate_sufficiency": (
                frozen_metric["mean_sufficiency_transfer"]
                >= CUT_THRESHOLDS["mean_sufficiency"]
            ),
            "natural_restoration": natural_gate,
            "source_do_natural": (
                natural_summary["source_do"]["flip_rate"]
                >= CUT_THRESHOLDS["source_do_flip_rate"]
            ),
            "wrong_role_control": (
                abs(control_summary["median_wrong_role_mediation"])
                <= CUT_THRESHOLDS["max_wrong_role_mediation"]
            ),
            "cross_pair_control": (
                abs(control_summary["median_cross_pair_mediation"])
                <= CUT_THRESHOLDS["max_cross_pair_mediation"]
            ),
            "noop_control": (
                control_summary["max_abs_noop_mediation"] <= 1e-5
            ),
            "greedy_minimality": no_single_deletion_passes,
        }
        frozen_spec = {
            "schema_version": "phase1001_minimum_head_cut_spec.v1",
            "phase": 1001,
            "model": MODEL,
            "initial_event_ids": [
                event["event_id"] for event in initial_events
            ],
            "frozen_event_ids": [
                event["event_id"] for event in frozen_events
            ],
            "frozen_size": len(frozen_events),
            "selection_partition": "validation",
            "selection_uses_holdout": False,
            "frozen_before_holdout": stage == "discovery",
        }
        summary = {
            "schema_version": f"phase1001_head_cut_{stage}_summary.v1",
            "phase": 1001,
            "model": MODEL,
            "stage": stage,
            "partition": directional[0]["partition"],
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "initial_size": len(initial_events),
            "initial_event_ids": frozen_spec["initial_event_ids"],
            "greedy_path": greedy_path,
            "frozen_size": len(frozen_events),
            "frozen_event_ids": frozen_spec["frozen_event_ids"],
            "frozen_candidate_metrics": frozen_metric,
            "natural_summary": natural_summary,
            "control_summary": control_summary,
            "confirmation_deletion_metrics": deletion_metrics,
            "thresholds": CUT_THRESHOLDS,
            "gate_checks": gate_checks,
            "minimum_head_cut_gate_pass": all(gate_checks.values()),
            "minimality_scope": (
                "greedy backward elimination; not exhaustive powerset"
            ),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(output_root / "frozen_spec.json", frozen_spec)
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("discovery", "confirmation"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(
        args.stage, args.batch_size, args.natural_max_new_tokens
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "passed": summary["minimum_head_cut_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "frozen_size": summary["frozen_size"],
                "frozen_event_ids": summary["frozen_event_ids"],
                "frozen_candidate_metrics": summary[
                    "frozen_candidate_metrics"
                ],
                "natural": summary["natural_summary"],
                "controls": summary["control_summary"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
