#!/usr/bin/env python3
"""Phase 1001 physical attention-channel sparsification.

The test operates on the 128 real coordinates of each head output immediately
before o_proj. Paired source-driven changes and exact O/unembedding geometry
generate several candidate rankings; only downstream interventions determine
whether a channel set is useful. The result is a physical channel set, not yet
an MLP-neuron mechanism.
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
    capture_residuals,
    prediction_colors,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1001_attention_head_discovery import (
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
from phase1001_minimum_head_cut import CUT_ROOT
from phase1001_minimum_head_cut_control_audit import build_donor_maps


CHANNEL_ROOT = RESULT_ROOT / "channel_sparsification"
OBSERVATION_PATH = CHANNEL_ROOT / "observation_probe.json"
CHANNEL_SIZES = (16, 32, 64, 96, 128, 192, 256, 384, 512, 768)
FAMILIES = (
    "cross_stratum_robust",
    "mean_direct",
    "sign_stable",
    "paired_change",
)
CHANNEL_THRESHOLDS = {
    "median_mediation": 0.50,
    "mean_sufficiency": 0.50,
    "natural_target_rate": 0.50,
    "semantic_null_max_mediation": 0.10,
    "semantic_null_source_rate": 0.90,
}


def parse_channel(channel_id):
    event_id, channel_text = channel_id.rsplit(".c", 1)
    return event_id, int(channel_text)


def channel_rankings(observation):
    channels = observation["channels"]
    return {
        "cross_stratum_robust": sorted(
            channels,
            key=lambda item: (
                -min(
                    item["min_template_mean_direct"],
                    item["min_color_mean_direct"],
                ),
                -item["mean_direct_effect"],
                item["channel_id"],
            ),
        ),
        "mean_direct": sorted(
            channels,
            key=lambda item: (
                -item["mean_direct_effect"],
                -item["positive_direct_rate"],
                item["channel_id"],
            ),
        ),
        "sign_stable": sorted(
            channels,
            key=lambda item: (
                -item["positive_direct_rate"],
                -item["mean_direct_effect"],
                item["channel_id"],
            ),
        ),
        "paired_change": sorted(
            channels,
            key=lambda item: (
                -item["mean_abs_paired_delta"],
                -item["mean_direct_effect"],
                item["channel_id"],
            ),
        ),
    }


def subset_spec(family, size, ranked):
    channels = [item["channel_id"] for item in ranked[:size]]
    return {
        "subset_id": f"{family}/k{size}",
        "family": family,
        "size": size,
        "channel_ids": channels,
    }


def channel_patches(
    subset,
    event_lookup,
    target_heads,
    do_heads,
    mode,
    donor_heads=None,
):
    grouped = defaultdict(list)
    for channel_id in subset["channel_ids"]:
        event_id, channel = parse_channel(channel_id)
        grouped[event_id].append(channel)
    patches = []
    for event_id, channels in grouped.items():
        event = event_lookup[event_id]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        if mode == "restore":
            vectors = do_heads[layer_number][:, head_index, :].clone()
            vectors[:, channels] = target_heads[layer_number][
                :, head_index, channels
            ]
        elif mode == "sufficiency":
            vectors = target_heads[layer_number][:, head_index, :].clone()
            vectors[:, channels] = do_heads[layer_number][
                :, head_index, channels
            ]
        elif mode == "semantic_null":
            if donor_heads is None:
                raise ValueError("donor_heads required")
            vectors = do_heads[layer_number][:, head_index, :].clone()
            vectors[:, channels] = donor_heads[layer_number][
                :, head_index, channels
            ]
        else:
            raise ValueError(mode)
        patches.append({"event": event, "vectors": vectors})
    return patches


def candidate_rows_batch(
    model,
    layers,
    device,
    batch,
    subsets,
    event_lookup,
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
    for subset in subsets:
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=channel_patches(
                subset,
                event_lookup,
                target_heads,
                do_heads,
                "restore",
            ),
        )
        sufficiency_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=channel_patches(
                subset,
                event_lookup,
                target_heads,
                do_heads,
                "sufficiency",
            ),
        )
        restored_margin = semantic_margin(restored_logits, batch)
        sufficiency_margin = semantic_margin(sufficiency_logits, batch)
        restored_prediction = prediction_colors(restored_logits)
        sufficiency_prediction = prediction_colors(sufficiency_logits)
        for index, item in enumerate(batch):
            do_effect = float(do_margin[index] - target_margin[index])
            natural_effect = float(
                source_margin[index] - target_margin[index]
            )
            rows.append(
                {
                    "schema_version": (
                        "phase1001_channel_candidate_causal.v1"
                    ),
                    "phase": 1001,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["target"]["template"],
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "subset_id": subset["subset_id"],
                    "family": subset["family"],
                    "size": subset["size"],
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


def summarize_candidate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subset_id"]].append(row)
    return {
        subset_id: {
            "subset_id": subset_id,
            "family": values[0]["family"],
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
            "color_median_mediation": {
                color: float(
                    np.median(
                        [
                            row["mediation_fraction"]
                            for row in values
                            if row["source_gold"] == color
                        ]
                    )
                )
                for color in COLORS
            },
        }
        for subset_id, values in groups.items()
    }


def candidate_pass(metric):
    return bool(
        metric["median_mediation_fraction"]
        >= CHANNEL_THRESHOLDS["median_mediation"]
        and metric["mean_sufficiency_transfer"]
        >= CHANNEL_THRESHOLDS["mean_sufficiency"]
    )


def capture_states(
    model, layers, device, batch, candidate_ids
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


def natural_rows_for_subsets(
    model,
    layers,
    tokenizer,
    device,
    directional,
    subsets,
    event_lookup,
    candidate_ids,
    effective_eos,
    batch_size,
    budget,
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
        ) = capture_states(
            model, layers, device, batch, candidate_ids
        )
        del source_logits, target_logits, do_logits
        conditions = [("source_do", None, [])]
        for subset in subsets:
            conditions.append(
                (
                    "source_plus_channel_restore",
                    subset,
                    channel_patches(
                        subset,
                        event_lookup,
                        target_heads,
                        do_heads,
                        "restore",
                    ),
                )
            )
        for condition, subset, patches in conditions:
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
                            "phase1001_channel_natural.v1"
                        ),
                        "phase": 1001,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "subset_id": (
                            subset["subset_id"] if subset else None
                        ),
                        "family": (
                            subset["family"] if subset else None
                        ),
                        "size": subset["size"] if subset else None,
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
        del (
            source_patch,
            target_heads,
            do_heads,
            source_residuals,
        )
        if batch_number % 2 == 0 or batch_number == len(batches):
            print(
                f"[channel-natural] {batch_number}/{len(batches)}",
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
            "family": values[0]["family"],
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


def semantic_control_rows(
    model,
    layers,
    tokenizer,
    device,
    directional,
    subset,
    event_lookup,
    donor_maps,
    candidate_ids,
    effective_eos,
    batch_size,
    budget,
):
    candidate_rows = []
    natural_rows = []
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
        ) = capture_states(
            model, layers, device, batch, candidate_ids
        )
        conditions = []
        correct_patches = channel_patches(
            subset,
            event_lookup,
            target_heads,
            do_heads,
            "restore",
        )
        correct_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=correct_patches,
        )
        conditions.append(("correct", correct_logits, correct_patches))
        for null_index, donor_map in enumerate(donor_maps):
            donor_items = [
                donor_map[(item["pair_id"], item["direction"])]
                for item in batch
            ]
            donor_cases = [item["target"] for item in donor_items]
            _, donor_heads, _ = capture_attention_states(
                model, layers, device, donor_cases, candidate_ids
            )
            patches = channel_patches(
                subset,
                event_lookup,
                target_heads,
                do_heads,
                "semantic_null",
                donor_heads=donor_heads,
            )
            logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                head_patches=patches,
            )
            conditions.append(
                (f"semantic_null_{null_index}", logits, patches)
            )
            del donor_heads
        target_margin = semantic_margin(target_logits, batch)
        do_margin = semantic_margin(do_logits, batch)
        for condition, logits, _ in conditions:
            margin = semantic_margin(logits, batch)
            prediction = prediction_colors(logits)
            for index, item in enumerate(batch):
                denominator = float(
                    do_margin[index] - target_margin[index]
                )
                candidate_rows.append(
                    {
                        "schema_version": (
                            "phase1001_channel_semantic_control.v1"
                        ),
                        "phase": 1001,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "mediation_fraction": float(
                            (do_margin[index] - margin[index])
                            / max(abs(denominator), 1e-8)
                        ),
                        "restored_to_target": (
                            prediction[index]
                            == item["target"]["gold"]
                        ),
                        "remained_source": (
                            prediction[index]
                            == item["source"]["gold"]
                        ),
                    }
                )
        for condition, _, patches in conditions[:2]:
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
                natural_rows.append(
                    {
                        "schema_version": (
                            "phase1001_channel_semantic_natural.v1"
                        ),
                        "phase": 1001,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "restored_to_target": (
                            result["prediction"]
                            == item["target"]["gold"]
                        ),
                        "remained_source": (
                            result["prediction"]
                            == item["source"]["gold"]
                        ),
                        "eos_seen": result["eos_seen"],
                        "exact_short": result["exact_short"],
                    }
                )
        for _, logits, _ in conditions:
            del logits
        del (
            source_logits,
            target_logits,
            do_logits,
            source_patch,
            target_heads,
            do_heads,
            source_residuals,
        )
        if batch_number % 2 == 0 or batch_number == len(batches):
            print(
                f"[channel-controls] {batch_number}/{len(batches)}",
                flush=True,
            )
    return candidate_rows, natural_rows


def summarize_control(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
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
            "target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "source_rate": float(
                np.mean([row["remained_source"] for row in values])
            ),
        }
        for condition, values in groups.items()
    }


def summarize_control_natural(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "source_rate": float(
                np.mean([row["remained_source"] for row in values])
            ),
            "eos_rate": float(
                np.mean([row["eos_seen"] for row in values])
            ),
        }
        for condition, values in groups.items()
    }


def run(stage, batch_size, natural_budget):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 channel test requires CUDA")
    observation = json.loads(
        OBSERVATION_PATH.read_text(encoding="utf-8")
    )
    if observation["channel_count"] != 768:
        raise RuntimeError("channel observation count drift")
    rankings = channel_rankings(observation)
    protocol, selected_pairs, directional, _ = selected_inputs(stage)
    output_root = CHANNEL_ROOT / stage
    output_root.mkdir(parents=True, exist_ok=True)
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    head_spec = read_json(CUT_ROOT / "discovery" / "frozen_spec.json")
    event_lookup = {
        event_id: {
            "event_id": event_id,
            "layer_number": int(event_id.split(".")[0][1:]),
            "head_index": int(event_id.split(".")[1][1:]),
            "role": "answer_boundary",
        }
        for event_id in head_spec["frozen_event_ids"]
    }

    if stage == "discovery":
        subsets = [
            subset_spec(family, size, rankings[family])
            for family in FAMILIES
            for size in CHANNEL_SIZES
        ]
    else:
        frozen = read_json(CHANNEL_ROOT / "discovery/frozen_spec.json")
        family = frozen["family"]
        size = int(frozen["size"])
        subset = subset_spec(family, size, rankings[family])
        if subset["channel_ids"] != frozen["channel_ids"]:
            raise RuntimeError("frozen channel ordering drift")
        subsets = [subset]

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

        candidate_rows = []
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
            ) = capture_states(
                model, layers, device, batch, candidate_ids
            )
            candidate_rows.extend(
                candidate_rows_batch(
                    model,
                    layers,
                    device,
                    batch,
                    subsets,
                    event_lookup,
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
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[channel-{stage}-candidate] "
                    f"{batch_number}/{len(batches)}",
                    flush=True,
                )
        candidate_summary = summarize_candidate(candidate_rows)
        write_rows(output_root / "candidate_rows.jsonl", candidate_rows)
        write_json(
            output_root / "candidate_summary.json",
            candidate_summary,
        )

        if stage == "discovery":
            natural_subsets = [
                subset
                for subset in subsets
                if candidate_pass(candidate_summary[subset["subset_id"]])
            ]
        else:
            natural_subsets = subsets
        if not natural_subsets:
            natural_subsets = [
                subset
                for subset in subsets
                if subset["size"] == max(CHANNEL_SIZES)
            ]
        natural_rows = natural_rows_for_subsets(
            model,
            layers,
            tokenizer,
            device,
            directional,
            natural_subsets,
            event_lookup,
            candidate_ids,
            effective_eos,
            batch_size,
            natural_budget,
        )
        natural_summary = summarize_natural(natural_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
        write_json(
            output_root / "natural_summary.json", natural_summary
        )

        if stage == "discovery":
            eligible = [
                subset
                for subset in natural_subsets
                if natural_summary[subset["subset_id"]]["target_rate"]
                >= CHANNEL_THRESHOLDS["natural_target_rate"]
            ]
            eligible.sort(
                key=lambda subset: (
                    subset["size"],
                    -min(
                        candidate_summary[subset["subset_id"]][
                            "median_mediation_fraction"
                        ],
                        candidate_summary[subset["subset_id"]][
                            "mean_sufficiency_transfer"
                        ],
                    ),
                    FAMILIES.index(subset["family"]),
                )
            )
            if not eligible:
                frozen_subset = natural_subsets[-1]
                natural_gate = False
            else:
                frozen_subset = eligible[0]
                natural_gate = True
        else:
            frozen_subset = subsets[0]
            natural_gate = (
                natural_summary[frozen_subset["subset_id"]][
                    "target_rate"
                ]
                >= CHANNEL_THRESHOLDS["natural_target_rate"]
            )

        donor_maps, donor_manifest = build_donor_maps(directional, 4)
        control_rows, control_natural_rows = semantic_control_rows(
            model,
            layers,
            tokenizer,
            device,
            directional,
            frozen_subset,
            event_lookup,
            donor_maps,
            candidate_ids,
            effective_eos,
            batch_size,
            natural_budget,
        )
        control_summary = summarize_control(control_rows)
        control_natural_summary = summarize_control_natural(
            control_natural_rows
        )
        write_rows(output_root / "donor_manifest.jsonl", donor_manifest)
        write_rows(output_root / "control_rows.jsonl", control_rows)
        write_rows(
            output_root / "control_natural_rows.jsonl",
            control_natural_rows,
        )
        write_json(
            output_root / "control_summary.json", control_summary
        )
        write_json(
            output_root / "control_natural_summary.json",
            control_natural_summary,
        )

        metric = candidate_summary[frozen_subset["subset_id"]]
        null_metrics = [
            control_summary[f"semantic_null_{index}"]
            for index in range(4)
        ]
        gate_checks = {
            "nontrivial_sparsification": (
                frozen_subset["size"] < observation["channel_count"]
            ),
            "candidate_mediation": metric[
                "median_mediation_fraction"
            ]
            >= CHANNEL_THRESHOLDS["median_mediation"],
            "candidate_sufficiency": metric[
                "mean_sufficiency_transfer"
            ]
            >= CHANNEL_THRESHOLDS["mean_sufficiency"],
            "natural_restoration": natural_gate,
            "semantic_null_candidate": all(
                abs(item["median_mediation_fraction"])
                <= CHANNEL_THRESHOLDS[
                    "semantic_null_max_mediation"
                ]
                for item in null_metrics
            ),
            "semantic_null_natural": control_natural_summary[
                "semantic_null_0"
            ]["source_rate"]
            >= CHANNEL_THRESHOLDS["semantic_null_source_rate"],
        }
        frozen_spec = {
            "schema_version": "phase1001_frozen_channel_spec.v1",
            "phase": 1001,
            "model": MODEL,
            "family": frozen_subset["family"],
            "size": frozen_subset["size"],
            "channel_ids": frozen_subset["channel_ids"],
            "parent_head_event_ids": head_spec["frozen_event_ids"],
            "selection_partition": "validation",
            "selection_uses_holdout": False,
            "frozen_before_holdout": stage == "discovery",
        }
        summary = {
            "schema_version": (
                f"phase1001_channel_{stage}_summary.v1"
            ),
            "phase": 1001,
            "model": MODEL,
            "stage": stage,
            "partition": directional[0]["partition"],
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "physical_channel_count": 768,
            "ranking_families": list(FAMILIES),
            "candidate_sizes": list(CHANNEL_SIZES),
            "candidate_summary": candidate_summary,
            "natural_summary": natural_summary,
            "frozen_family": frozen_subset["family"],
            "frozen_size": frozen_subset["size"],
            "frozen_channel_ids": frozen_subset["channel_ids"],
            "frozen_candidate_metrics": metric,
            "frozen_natural_metrics": natural_summary[
                frozen_subset["subset_id"]
            ],
            "control_summary": control_summary,
            "control_natural_summary": control_natural_summary,
            "thresholds": CHANNEL_THRESHOLDS,
            "gate_checks": gate_checks,
            "channel_sparsification_gate_pass": all(
                gate_checks.values()
            ),
            "adjudicated_status": (
                "GO_NONTRIVIAL_CHANNEL_SET"
                if all(gate_checks.values())
                else "NO_GO_CHANNEL_SPARSIFICATION"
            ),
            "interpretation_limit": (
                "physical attention output channels; not MLP neurons"
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
                "passed": summary["channel_sparsification_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "frozen_family": summary["frozen_family"],
                "frozen_size": summary["frozen_size"],
                "candidate": summary["frozen_candidate_metrics"],
                "natural": summary["frozen_natural_metrics"],
                "controls": summary["control_summary"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
