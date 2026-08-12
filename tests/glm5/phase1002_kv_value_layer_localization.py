#!/usr/bin/env python3
"""Causally localize the cache-value layers carrying the Phase 1002 source."""
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

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, load_model, release_model
from phase1002_multitoken_frozen_topology import read_json
from phase1002_multitoken_kv_cache_decomposition import (
    build_cache,
    candidate_logits,
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
    step_case,
)


PHASE = 1002
SCREEN_PAIRS_PER_STRATUM = 1
FORMAL_PAIRS_PER_STRATUM = 4
JOINT_SIZES = (1, 2, 4, 8, 12, 16, 24, 28, 32, 36, 40)


def clone_value_layer_mix(
    target_cache,
    source_cache,
    source_value_layers: set[int],
    model_config,
) -> DynamicCache:
    if len(target_cache.layers) != len(source_cache.layers):
        raise RuntimeError("cache layer count drift")
    data = []
    for layer_index, (target_layer, source_layer) in enumerate(
        zip(target_cache.layers, source_cache.layers)
    ):
        values = (
            source_layer.values
            if layer_index in source_value_layers
            else target_layer.values
        )
        data.append((
            target_layer.keys.detach().clone(),
            values.detach().clone(),
        ))
    return DynamicCache(data, config=model_config)


def continue_from_cache(
    model,
    device,
    current_ids: list[int],
    prefix_length: int,
    cache,
    candidate_ids: dict[str, int],
) -> torch.Tensor:
    input_ids = torch.tensor(
        [[token_id] for token_id in current_ids],
        dtype=torch.long,
        device=device,
    )
    attention = torch.ones(
        (len(current_ids), prefix_length + 1),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    return candidate_logits(output.logits[:, -1, :], candidate_ids)


def prepare_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    source_depth: int,
):
    source_cases = [row["source"] for row in batch]
    target_cases = [row["target"] for row in batch]
    candidate_ids = target_cases[0]["candidate_token_ids"]
    semantic_step = int(target_cases[0]["semantic_step"])
    cache_prefix_step = semantic_step - 1
    target_prefix_cases = [
        step_case(row["target"], cache_prefix_step)
        for row in batch
    ]
    source_semantic_cases = [
        step_case(row["source"], semantic_step)
        for row in batch
    ]
    current_ids = [
        int(row["target"]["answer_token_ids"][cache_prefix_step])
        for row in batch
    ]
    _, source_residuals = scpg.capture_residuals(
        model,
        device,
        source_cases,
        (source_depth,),
        candidate_ids,
    )
    source_vectors = source_residuals[source_depth]
    prefix_patch = scpg.source_patch_spec(
        source_depth,
        target_prefix_cases,
        source_vectors,
        "joint",
    )
    target_cache = build_cache(
        model, layers, device, target_prefix_cases, None
    )
    source_cache = build_cache(
        model, layers, device, target_prefix_cases, prefix_patch
    )
    source_full = scpg.forward_candidate(
        model,
        layers,
        device,
        source_semantic_cases,
        candidate_ids,
    )
    return {
        "candidate_ids": candidate_ids,
        "current_ids": current_ids,
        "prefix_length": int(
            target_prefix_cases[0]["input_token_count"]
        ),
        "target_cache": target_cache,
        "source_cache": source_cache,
        "source_full": source_full,
        "source_residuals": source_residuals,
    }


def base_cache_logits(
    model,
    device,
    prepared: dict[str, Any],
    layer_count: int,
):
    target = continue_from_cache(
        model,
        device,
        prepared["current_ids"],
        prepared["prefix_length"],
        clone_value_layer_mix(
            prepared["target_cache"],
            prepared["source_cache"],
            set(),
            model.config,
        ),
        prepared["candidate_ids"],
    )
    source_values = continue_from_cache(
        model,
        device,
        prepared["current_ids"],
        prepared["prefix_length"],
        clone_value_layer_mix(
            prepared["target_cache"],
            prepared["source_cache"],
            set(range(layer_count)),
            model.config,
        ),
        prepared["candidate_ids"],
    )
    return target, source_values


def screen_layers(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = selected_directional_rows(
        model_name, "discovery", SCREEN_PAIRS_PER_STRATUM
    )
    result_rows = []
    layer_count = len(layers)
    all_layers = set(range(layer_count))
    row_batches = list(batches(rows, batch_size))
    for batch_number, batch in enumerate(row_batches, 1):
        prepared = prepare_batch(
            model, layers, device, batch, source_depth
        )
        target_logits, do_logits = base_cache_logits(
            model, device, prepared, layer_count
        )
        source_margin = margin(prepared["source_full"], batch)
        target_margin = margin(target_logits, batch)
        do_margin = margin(do_logits, batch)
        do_predictions = predictions(do_logits)
        for layer_index in range(layer_count):
            sufficiency_logits = continue_from_cache(
                model,
                device,
                prepared["current_ids"],
                prepared["prefix_length"],
                clone_value_layer_mix(
                    prepared["target_cache"],
                    prepared["source_cache"],
                    {layer_index},
                    model.config,
                ),
                prepared["candidate_ids"],
            )
            restore_logits = continue_from_cache(
                model,
                device,
                prepared["current_ids"],
                prepared["prefix_length"],
                clone_value_layer_mix(
                    prepared["target_cache"],
                    prepared["source_cache"],
                    all_layers - {layer_index},
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
                source_effect = float(
                    do_margin[index] - target_margin[index]
                )
                result_rows.append({
                    "schema_version": (
                        "phase1002_kv_value_layer_screen_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "split": "discovery",
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["template"],
                    "layer_index": layer_index,
                    "layer_number": layer_index + 1,
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "source_values_prediction": do_predictions[index],
                    "single_layer_prediction": (
                        sufficiency_predictions[index]
                    ),
                    "restore_layer_prediction": (
                        restore_predictions[index]
                    ),
                    "single_layer_transfer": float(
                        (
                            sufficiency_margin[index]
                            - target_margin[index]
                        )
                        / max(abs(clean_span), 1e-8)
                    ),
                    "restore_layer_mediation": float(
                        (do_margin[index] - restore_margin[index])
                        / max(abs(source_effect), 1e-8)
                    ),
                    "single_layer_source": (
                        sufficiency_predictions[index]
                        == item["source"]["gold"]
                    ),
                    "restore_layer_target": (
                        restore_predictions[index]
                        == item["target"]["gold"]
                    ),
                })
        del (
            prepared,
            target_logits,
            do_logits,
        )
        print(
            f"[screen/{model_name}] "
            f"{batch_number}/{len(row_batches)}",
            flush=True,
        )

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        groups[int(row["layer_index"])].append(row)
    ranking = []
    for layer_index, values in groups.items():
        median_mediation = float(np.median([
            row["restore_layer_mediation"] for row in values
        ]))
        mean_sufficiency = float(np.mean([
            row["single_layer_transfer"] for row in values
        ]))
        ranking.append({
            "layer_index": layer_index,
            "layer_number": layer_index + 1,
            "n": len(values),
            "median_restore_layer_mediation": median_mediation,
            "mean_restore_layer_mediation": float(np.mean([
                row["restore_layer_mediation"] for row in values
            ])),
            "restore_layer_target_rate": float(np.mean([
                row["restore_layer_target"] for row in values
            ])),
            "mean_single_layer_transfer": mean_sufficiency,
            "median_single_layer_transfer": float(np.median([
                row["single_layer_transfer"] for row in values
            ])),
            "single_layer_source_rate": float(np.mean([
                row["single_layer_source"] for row in values
            ])),
            "causal_score": (
                max(0.0, median_mediation)
                + max(0.0, mean_sufficiency)
            ),
        })
    ranking.sort(key=lambda item: (
        -item["causal_score"],
        -item["median_restore_layer_mediation"],
        -item["mean_single_layer_transfer"],
        item["layer_number"],
    ))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
        item["selection_split"] = "discovery_screen"
        item["selection_uses_confirmation"] = False
    return result_rows, ranking


def evaluate_joint_sizes(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    ranking: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    layer_count = len(layers)
    sizes = tuple(
        size for size in JOINT_SIZES if size <= layer_count
    )
    all_layers = set(range(layer_count))
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
            target_logits, do_logits = base_cache_logits(
                model, device, prepared, layer_count
            )
            source_margin = margin(prepared["source_full"], batch)
            target_margin = margin(target_logits, batch)
            do_margin = margin(do_logits, batch)
            do_predictions = predictions(do_logits)
            for size in sizes:
                selected = {
                    int(item["layer_index"])
                    for item in ranking[:size]
                }
                sufficiency_logits = continue_from_cache(
                    model,
                    device,
                    prepared["current_ids"],
                    prepared["prefix_length"],
                    clone_value_layer_mix(
                        prepared["target_cache"],
                        prepared["source_cache"],
                        selected,
                        model.config,
                    ),
                    prepared["candidate_ids"],
                )
                restore_logits = continue_from_cache(
                    model,
                    device,
                    prepared["current_ids"],
                    prepared["prefix_length"],
                    clone_value_layer_mix(
                        prepared["target_cache"],
                        prepared["source_cache"],
                        all_layers - selected,
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
                    source_effect = float(
                        do_margin[index] - target_margin[index]
                    )
                    result_rows.append({
                        "schema_version": (
                            "phase1002_kv_value_layer_joint_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "template": item["template"],
                        "joint_size": size,
                        "selected_layer_numbers": [
                            int(item["layer_number"])
                            for item in ranking[:size]
                        ],
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "source_values_prediction": do_predictions[index],
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
                            (do_margin[index] - restore_margin[index])
                            / max(abs(source_effect), 1e-8)
                        ),
                    })
            del prepared, target_logits, do_logits
            print(
                f"[joint/{model_name}/{split}] "
                f"{batch_number}/{len(row_batches)}",
                flush=True,
            )
    return result_rows


def summarize_joint(
    rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    summary = {}
    for split in ("discovery", "confirmation"):
        summary[split] = {}
        for size in JOINT_SIZES:
            values = [
                row for row in rows
                if row["split"] == split
                and int(row["joint_size"]) == size
            ]
            if not values:
                continue
            summary[split][str(size)] = {
                "n": len(values),
                "source_values_source_rate": float(np.mean([
                    row["source_values_prediction"]
                    == row["source_gold"]
                    for row in values
                ])),
                "sufficiency_source_rate": float(np.mean([
                    row["joint_sufficiency_prediction"]
                    == row["source_gold"]
                    for row in values
                ])),
                "restore_target_rate": float(np.mean([
                    row["joint_restore_prediction"]
                    == row["target_gold"]
                    for row in values
                ])),
                "median_sufficiency_transfer": float(np.median([
                    row["joint_sufficiency_transfer"] for row in values
                ])),
                "mean_sufficiency_transfer": float(np.mean([
                    row["joint_sufficiency_transfer"] for row in values
                ])),
                "median_mediation": float(np.median([
                    row["joint_mediation"] for row in values
                ])),
                "mean_mediation": float(np.mean([
                    row["joint_mediation"] for row in values
                ])),
                "selected_layer_numbers": values[0][
                    "selected_layer_numbers"
                ],
            }
    return summary


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
    cache_summary = read_json(
        OUT_ROOT
        / "kv_cache_decomposition"
        / model_name
        / "summary.json"
    )
    if not cache_summary["cache_transport_pass"]:
        raise RuntimeError(f"{model_name}: cache transport gate failed")
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
        screen_rows, ranking = screen_layers(
            model,
            layers,
            device,
            model_name,
            source_depth,
            batch_size,
        )
        joint_rows = evaluate_joint_sizes(
            model,
            layers,
            device,
            model_name,
            source_depth,
            ranking,
            batch_size,
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    model_root = OUT_ROOT / "kv_value_layer_localization" / model_name
    write_jsonl(model_root / "discovery_screen_rows.jsonl", screen_rows)
    write_json(model_root / "frozen_layer_ranking.json", ranking)
    write_jsonl(model_root / "joint_rows.jsonl", joint_rows)
    joint_summary = summarize_joint(joint_rows)
    selected_size, discovery_gate = choose_size(
        joint_summary["discovery"],
        source_threshold,
        restore_threshold,
    )
    selected_confirmation = joint_summary["confirmation"][
        str(selected_size)
    ]
    confirmation_gate = (
        selected_confirmation["sufficiency_source_rate"]
        >= source_threshold
        and selected_confirmation["restore_target_rate"]
        >= restore_threshold
        and selected_confirmation["median_mediation"] >= 0.30
    )
    summary = {
        "schema_version": (
            "phase1002_kv_value_layer_localization_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "layer_count": len(ranking),
        "source_depth": source_depth,
        "screen_direction_count": len(screen_rows) // len(ranking),
        "formal_direction_count_per_split": 256,
        "ranking_selection_uses_confirmation": False,
        "top_layers": ranking[:16],
        "joint_summary": joint_summary,
        "selection": {
            "selected_size": selected_size,
            "selected_layer_numbers": [
                int(item["layer_number"])
                for item in ranking[:selected_size]
            ],
            "discovery_gate": discovery_gate,
            "confirmation_gate": confirmation_gate,
        },
        "value_layer_localization_pass": (
            discovery_gate and confirmation_gate
        ),
        "thresholds": {
            "sufficiency_source_rate": source_threshold,
            "restore_target_rate": restore_threshold,
            "median_mediation": 0.30,
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "This localizes cache-value layer arms for the fixed binding "
            "task. It does not yet localize token positions, KV heads, or "
            "individual value channels."
        ),
    }
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT
            / "kv_value_layer_localization"
            / model_name
            / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "kv_value_layer_localization"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": (
            "phase1002_kv_value_layer_localization_cross_model.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["value_layer_localization_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["value_layer_localization_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(
        OUT_ROOT / "kv_value_layer_localization" / "summary.json",
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
