#!/usr/bin/env python3
"""Localize token-position groups inside frozen cache-value carrier layers."""
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
from transformers import DynamicCache


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1002_kv_value_layer_localization import (
    base_cache_logits,
    continue_from_cache,
    prepare_batch,
)
from phase1002_multitoken_frozen_topology import read_json
from phase1002_multitoken_kv_cache_decomposition import (
    margin,
    predictions,
)
from phase1002_multitoken_protocol import (
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)
from phase1002_multitoken_temporal_rollout import (
    batches,
    selected_directional_rows,
)


PHASE = 1002
SCREEN_PAIRS_PER_STRATUM = 1
FORMAL_PAIRS_PER_STRATUM = 4
POSITION_GROUPS = (
    "slot0_entity",
    "slot1_entity",
    "slot0_color",
    "slot1_color",
    "query_name",
    "prompt_boundary",
    "generated_prefix",
    "other_prompt",
)


def position_groups(
    rows: list[dict[str, Any]],
    prefix_length: int,
) -> dict[str, list[list[int]]]:
    result = {name: [] for name in POSITION_GROUPS}
    for row in rows:
        target = row["target"]
        roles = target["role_positions"]
        prompt_length = int(target["input_token_count"])
        atomic = {
            "slot0_entity": [int(roles["slot0_entity"])],
            "slot1_entity": [int(roles["slot1_entity"])],
            "slot0_color": [int(roles["slot0_color"])],
            "slot1_color": [int(roles["slot1_color"])],
            "query_name": [int(roles["query_name"])],
            "prompt_boundary": [prompt_length - 1],
            "generated_prefix": list(
                range(prompt_length, prefix_length)
            ),
        }
        used = {
            position
            for values in atomic.values()
            for position in values
        }
        atomic["other_prompt"] = [
            position
            for position in range(prompt_length)
            if position not in used
        ]
        flattened = [
            position
            for group in POSITION_GROUPS
            for position in atomic[group]
        ]
        if (
            len(flattened) != prefix_length
            or len(set(flattened)) != prefix_length
        ):
            raise RuntimeError(
                f"position partition drift: "
                f"{len(flattened)}/{len(set(flattened))}/"
                f"{prefix_length}"
            )
        for group in POSITION_GROUPS:
            result[group].append(atomic[group])
    return result


def clone_position_mix(
    target_cache,
    source_cache,
    selected_layers: set[int],
    source_positions: list[list[int]],
    model_config,
) -> DynamicCache:
    data = []
    for layer_index, (target_layer, source_layer) in enumerate(
        zip(target_cache.layers, source_cache.layers)
    ):
        values = target_layer.values.detach().clone()
        if layer_index in selected_layers:
            for batch_index, positions in enumerate(source_positions):
                if positions:
                    values[
                        batch_index,
                        :,
                        positions,
                        :,
                    ] = source_layer.values[
                        batch_index,
                        :,
                        positions,
                        :,
                    ]
        data.append((
            target_layer.keys.detach().clone(),
            values,
        ))
    return DynamicCache(data, config=model_config)


def screen_groups(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    selected_layers: set[int],
    batch_size: int,
    group_names: tuple[str, ...] = POSITION_GROUPS,
    group_builder=position_groups,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = selected_directional_rows(
        model_name, "discovery", SCREEN_PAIRS_PER_STRATUM
    )
    result_rows = []
    row_batches = list(batches(rows, batch_size))
    for batch_number, batch in enumerate(row_batches, 1):
        prepared = prepare_batch(
            model, layers, device, batch, source_depth
        )
        target_logits, _ = base_cache_logits(
            model, device, prepared, len(layers)
        )
        groups = group_builder(batch, prepared["prefix_length"])
        all_positions = [
            list(range(prepared["prefix_length"]))
            for _ in batch
        ]
        selected_source_logits = continue_from_cache(
            model,
            device,
            prepared["current_ids"],
            prepared["prefix_length"],
            clone_position_mix(
                prepared["target_cache"],
                prepared["source_cache"],
                selected_layers,
                all_positions,
                model.config,
            ),
            prepared["candidate_ids"],
        )
        source_margin = margin(prepared["source_full"], batch)
        target_margin = margin(target_logits, batch)
        selected_margin = margin(selected_source_logits, batch)
        selected_predictions = predictions(selected_source_logits)
        for group in group_names:
            only_group_logits = continue_from_cache(
                model,
                device,
                prepared["current_ids"],
                prepared["prefix_length"],
                clone_position_mix(
                    prepared["target_cache"],
                    prepared["source_cache"],
                    selected_layers,
                    groups[group],
                    model.config,
                ),
                prepared["candidate_ids"],
            )
            complement = [
                [
                    position
                    for position in range(prepared["prefix_length"])
                    if position not in set(group_positions)
                ]
                for group_positions in groups[group]
            ]
            restore_group_logits = continue_from_cache(
                model,
                device,
                prepared["current_ids"],
                prepared["prefix_length"],
                clone_position_mix(
                    prepared["target_cache"],
                    prepared["source_cache"],
                    selected_layers,
                    complement,
                    model.config,
                ),
                prepared["candidate_ids"],
            )
            only_margin = margin(only_group_logits, batch)
            restore_margin = margin(restore_group_logits, batch)
            only_predictions = predictions(only_group_logits)
            restore_predictions = predictions(restore_group_logits)
            for index, item in enumerate(batch):
                clean_span = float(
                    source_margin[index] - target_margin[index]
                )
                selected_effect = float(
                    selected_margin[index] - target_margin[index]
                )
                result_rows.append({
                    "schema_version": (
                        "phase1002_kv_value_position_screen_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "split": "discovery",
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["template"],
                    "position_group": group,
                    "position_count": len(groups[group][index]),
                    "selected_layer_numbers": sorted(
                        layer_index + 1
                        for layer_index in selected_layers
                    ),
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "all_selected_prediction": (
                        selected_predictions[index]
                    ),
                    "only_group_prediction": only_predictions[index],
                    "restore_group_prediction": (
                        restore_predictions[index]
                    ),
                    "only_group_transfer": float(
                        (only_margin[index] - target_margin[index])
                        / max(abs(clean_span), 1e-8)
                    ),
                    "restore_group_mediation": float(
                        (selected_margin[index] - restore_margin[index])
                        / max(abs(selected_effect), 1e-8)
                    ),
                })
        del prepared, target_logits, selected_source_logits
        print(
            f"[screen/{model_name}] "
            f"{batch_number}/{len(row_batches)}",
            flush=True,
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[row["position_group"]].append(row)
    ranking = []
    for group, values in grouped.items():
        median_mediation = float(np.median([
            row["restore_group_mediation"] for row in values
        ]))
        mean_sufficiency = float(np.mean([
            row["only_group_transfer"] for row in values
        ]))
        ranking.append({
            "position_group": group,
            "n": len(values),
            "median_position_count": float(np.median([
                row["position_count"] for row in values
            ])),
            "median_restore_group_mediation": median_mediation,
            "mean_restore_group_mediation": float(np.mean([
                row["restore_group_mediation"] for row in values
            ])),
            "restore_group_target_rate": float(np.mean([
                row["restore_group_prediction"] == row["target_gold"]
                for row in values
            ])),
            "mean_only_group_transfer": mean_sufficiency,
            "median_only_group_transfer": float(np.median([
                row["only_group_transfer"] for row in values
            ])),
            "only_group_source_rate": float(np.mean([
                row["only_group_prediction"] == row["source_gold"]
                for row in values
            ])),
            "causal_score": (
                max(0.0, median_mediation)
                + max(0.0, mean_sufficiency)
            ),
        })
    ranking.sort(key=lambda item: (
        -item["causal_score"],
        -item["median_restore_group_mediation"],
        -item["mean_only_group_transfer"],
        item["position_group"],
    ))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
        item["selection_split"] = "discovery_screen"
        item["selection_uses_confirmation"] = False
    return result_rows, ranking


def evaluate_group_sizes(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    selected_layers: set[int],
    ranking: list[dict[str, Any]],
    batch_size: int,
    group_names: tuple[str, ...] = POSITION_GROUPS,
    group_builder=position_groups,
) -> list[dict[str, Any]]:
    result_rows = []
    for split in ("discovery", "confirmation"):
        rows = selected_directional_rows(
            model_name, split, FORMAL_PAIRS_PER_STRATUM
        )
        row_batches = list(batches(rows, batch_size))
        for batch_number, batch in enumerate(row_batches, 1):
            prepared = prepare_batch(
                model, layers, device, batch, source_depth
            )
            target_logits, _ = base_cache_logits(
                model, device, prepared, len(layers)
            )
            groups = group_builder(batch, prepared["prefix_length"])
            all_positions = [
                list(range(prepared["prefix_length"]))
                for _ in batch
            ]
            selected_source_logits = continue_from_cache(
                model,
                device,
                prepared["current_ids"],
                prepared["prefix_length"],
                clone_position_mix(
                    prepared["target_cache"],
                    prepared["source_cache"],
                    selected_layers,
                    all_positions,
                    model.config,
                ),
                prepared["candidate_ids"],
            )
            source_margin = margin(prepared["source_full"], batch)
            target_margin = margin(target_logits, batch)
            selected_margin = margin(selected_source_logits, batch)
            selected_predictions = predictions(selected_source_logits)
            for size in range(1, len(group_names) + 1):
                selected_groups = [
                    item["position_group"] for item in ranking[:size]
                ]
                selected_positions = [
                    sorted({
                        position
                        for group in selected_groups
                        for position in groups[group][index]
                    })
                    for index in range(len(batch))
                ]
                complement = [
                    [
                        position
                        for position in range(prepared["prefix_length"])
                        if position not in set(selected_positions[index])
                    ]
                    for index in range(len(batch))
                ]
                sufficiency_logits = continue_from_cache(
                    model,
                    device,
                    prepared["current_ids"],
                    prepared["prefix_length"],
                    clone_position_mix(
                        prepared["target_cache"],
                        prepared["source_cache"],
                        selected_layers,
                        selected_positions,
                        model.config,
                    ),
                    prepared["candidate_ids"],
                )
                restore_logits = continue_from_cache(
                    model,
                    device,
                    prepared["current_ids"],
                    prepared["prefix_length"],
                    clone_position_mix(
                        prepared["target_cache"],
                        prepared["source_cache"],
                        selected_layers,
                        complement,
                        model.config,
                    ),
                    prepared["candidate_ids"],
                )
                sufficiency_margin = margin(sufficiency_logits, batch)
                restore_margin = margin(restore_logits, batch)
                sufficiency_predictions = predictions(sufficiency_logits)
                restore_predictions = predictions(restore_logits)
                for index, item in enumerate(batch):
                    clean_span = float(
                        source_margin[index] - target_margin[index]
                    )
                    selected_effect = float(
                        selected_margin[index] - target_margin[index]
                    )
                    result_rows.append({
                        "schema_version": (
                            "phase1002_kv_value_position_joint_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "template": item["template"],
                        "group_count": size,
                        "selected_groups": selected_groups,
                        "selected_position_count": len(
                            selected_positions[index]
                        ),
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "all_selected_prediction": (
                            selected_predictions[index]
                        ),
                        "joint_sufficiency_prediction": (
                            sufficiency_predictions[index]
                        ),
                        "joint_restore_prediction": (
                            restore_predictions[index]
                        ),
                        "joint_sufficiency_transfer": float(
                            (
                                sufficiency_margin[index]
                                - target_margin[index]
                            )
                            / max(abs(clean_span), 1e-8)
                        ),
                        "joint_mediation": float(
                            (
                                selected_margin[index]
                                - restore_margin[index]
                            )
                            / max(abs(selected_effect), 1e-8)
                        ),
                    })
            del prepared, target_logits, selected_source_logits
            print(
                f"[joint/{model_name}/{split}] "
                f"{batch_number}/{len(row_batches)}",
                flush=True,
            )
    return result_rows


def summarize_joint(
    rows: list[dict[str, Any]],
    group_count: int = len(POSITION_GROUPS),
) -> dict[str, Any]:
    result = {}
    for split in ("discovery", "confirmation"):
        result[split] = {}
        for size in range(1, group_count + 1):
            values = [
                row for row in rows
                if row["split"] == split
                and int(row["group_count"]) == size
            ]
            result[split][str(size)] = {
                "n": len(values),
                "all_selected_source_rate": float(np.mean([
                    row["all_selected_prediction"] == row["source_gold"]
                    for row in values
                ])),
                "sufficiency_source_rate": float(np.mean([
                    row["joint_sufficiency_prediction"]
                    == row["source_gold"]
                    for row in values
                ])),
                "restore_target_rate": float(np.mean([
                    row["joint_restore_prediction"] == row["target_gold"]
                    for row in values
                ])),
                "median_sufficiency_transfer": float(np.median([
                    row["joint_sufficiency_transfer"] for row in values
                ])),
                "median_mediation": float(np.median([
                    row["joint_mediation"] for row in values
                ])),
                "selected_groups": values[0]["selected_groups"],
                "median_selected_position_count": float(np.median([
                    row["selected_position_count"] for row in values
                ])),
            }
    return result


def choose_size(
    discovery: dict[str, Any],
    source_threshold: float,
    restore_threshold: float,
) -> tuple[int, bool]:
    eligible = [
        int(size)
        for size, metrics in discovery.items()
        if metrics["sufficiency_source_rate"] >= source_threshold
        and metrics["restore_target_rate"] >= restore_threshold
        and metrics["median_mediation"] >= 0.30
    ]
    if eligible:
        return min(eligible), True
    best = max(
        discovery,
        key=lambda size: (
            discovery[size]["sufficiency_source_rate"]
            + discovery[size]["restore_target_rate"],
            discovery[size]["median_mediation"],
            -int(size),
        ),
    )
    return int(best), False


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    layer_summary = read_json(
        OUT_ROOT
        / "kv_value_layer_localization"
        / model_name
        / "summary.json"
    )
    if not layer_summary["value_layer_localization_pass"]:
        raise RuntimeError(f"{model_name}: value layer gate failed")
    selected_layer_numbers = layer_summary["selection"][
        "selected_layer_numbers"
    ]
    selected_layers = {
        int(layer_number) - 1
        for layer_number in selected_layer_numbers
    }
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(
        prereg["frozen_phase1001_topology"][model_name]["source_depth"]
    )
    source_threshold = prereg["primary_thresholds"][
        "source_do_semantic_flip_rate"
    ]
    restore_threshold = prereg["primary_thresholds"][
        "frozen_topology_semantic_restore_rate"
    ]
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        screen_rows, ranking = screen_groups(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            batch_size,
        )
        joint_rows = evaluate_group_sizes(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            ranking,
            batch_size,
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    model_root = OUT_ROOT / "kv_value_position_localization" / model_name
    write_jsonl(model_root / "discovery_screen_rows.jsonl", screen_rows)
    write_json(model_root / "frozen_position_ranking.json", ranking)
    write_jsonl(model_root / "joint_rows.jsonl", joint_rows)
    joint_summary = summarize_joint(joint_rows)
    selected_size, discovery_gate = choose_size(
        joint_summary["discovery"],
        source_threshold,
        restore_threshold,
    )
    confirmation = joint_summary["confirmation"][str(selected_size)]
    confirmation_gate = (
        confirmation["sufficiency_source_rate"] >= source_threshold
        and confirmation["restore_target_rate"] >= restore_threshold
        and confirmation["median_mediation"] >= 0.30
    )
    summary = {
        "schema_version": (
            "phase1002_kv_value_position_localization_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "source_depth": source_depth,
        "selected_layer_numbers": selected_layer_numbers,
        "screen_direction_count": 64,
        "formal_direction_count_per_split": 256,
        "ranking_selection_uses_confirmation": False,
        "position_ranking": ranking,
        "joint_summary": joint_summary,
        "selection": {
            "selected_group_count": selected_size,
            "selected_groups": [
                item["position_group"]
                for item in ranking[:selected_size]
            ],
            "discovery_gate": discovery_gate,
            "confirmation_gate": confirmation_gate,
        },
        "value_position_localization_pass": (
            discovery_gate and confirmation_gate
        ),
        "thresholds": {
            "sufficiency_source_rate": source_threshold,
            "restore_target_rate": restore_threshold,
            "median_mediation": 0.30,
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "Position groups are semantic regions of the fixed prompt. "
            "A broad other_prompt result must be refined before any "
            "source-token or neuron-level claim."
        ),
    }
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT
            / "kv_value_position_localization"
            / model_name
            / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "kv_value_position_localization"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": (
            "phase1002_kv_value_position_localization_cross_model.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["value_position_localization_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["value_position_localization_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(
        OUT_ROOT / "kv_value_position_localization" / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model, args.batch_size)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
