#!/usr/bin/env python3
"""Localize fixed value-channel blocks inside Phase1003-frozen KV heads."""
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
from phase1003_anchor_subset_exhaustive import choose_donors
from phase1003_crossparadigm_protocol import (
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    selected_directional_rows,
    write_json,
    write_jsonl,
)
from phase1003_value_head_localization import parse_event_id
from phase1003_value_layer_relocalization import (
    batches,
    contrast_margin,
    prepare_batch,
    predictions,
)


BLOCK_WIDTH = 16
JOINT_SIZES = (
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160,
    192, 224, 256, 320, 384,
)


def block_event_id(
    layer_index: int,
    head_index: int,
    start: int,
    end: int,
) -> str:
    return (
        f"l{layer_index + 1}.kvh{head_index}."
        f"v{start:03d}_{end:03d}"
    )


def parse_block_event_id(
    event_id: str,
) -> tuple[int, int, int, int]:
    layer_text, head_text, block_text = event_id.split(".")
    start_text, end_text = block_text[1:].split("_")
    return (
        int(layer_text[1:]) - 1,
        int(head_text[3:]),
        int(start_text),
        int(end_text),
    )


def clone_block_mix(
    target_cache,
    source_cache,
    source_blocks: set[tuple[int, int, int, int]],
    model_config,
) -> DynamicCache:
    if len(target_cache.layers) != len(source_cache.layers):
        raise RuntimeError("cache layer count drift")
    by_layer = defaultdict(list)
    for event in source_blocks:
        by_layer[int(event[0])].append(event)
    data = []
    for layer_index, (target_layer, source_layer) in enumerate(
        zip(target_cache.layers, source_cache.layers)
    ):
        values = target_layer.values.detach().clone()
        for _, head_index, start, end in by_layer.get(
            layer_index, []
        ):
            values[:, head_index, :, start:end] = source_layer.values[
                :, head_index, :, start:end
            ]
        data.append((
            target_layer.keys.detach().clone(),
            values,
        ))
    return DynamicCache(data, config=model_config)


def continue_blocks(
    model,
    device,
    prepared: dict[str, Any],
    source_blocks: set[tuple[int, int, int, int]],
) -> torch.Tensor:
    from phase1003_crossparadigm_kv_replication import continue_cache

    return continue_cache(
        model,
        device,
        prepared["current_ids"],
        prepared["prefix_length"],
        clone_block_mix(
            prepared["target_cache"],
            prepared["source_cache"],
            source_blocks,
            model.config,
        ),
        prepared["candidate_ids"],
    )


def block_universe(
    selected_head_ids: list[str],
    head_dim: int,
) -> list[tuple[int, int, int, int]]:
    if head_dim % BLOCK_WIDTH:
        raise RuntimeError(
            f"head_dim {head_dim} not divisible by {BLOCK_WIDTH}"
        )
    return [
        (layer_index, head_index, start, start + BLOCK_WIDTH)
        for layer_index, head_index in (
            parse_event_id(event_id)
            for event_id in selected_head_ids
        )
        for start in range(0, head_dim, BLOCK_WIDTH)
    ]


def evaluate_block_set(
    model,
    device,
    prepared: dict[str, Any],
    batch: list[dict[str, Any]],
    donor_batch: list[dict[str, Any]],
    selected_blocks: set[tuple[int, int, int, int]],
    parent_blocks: set[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    target_logits = continue_blocks(
        model, device, prepared, set()
    )
    parent_logits = continue_blocks(
        model, device, prepared, parent_blocks
    )
    sufficiency_logits = continue_blocks(
        model, device, prepared, selected_blocks
    )
    restore_logits = continue_blocks(
        model,
        device,
        prepared,
        parent_blocks - selected_blocks,
    )
    labels = list(prepared["candidate_ids"])
    margins = {
        "target": contrast_margin(
            target_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        ),
        "parent": contrast_margin(
            parent_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        ),
        "sufficiency": contrast_margin(
            sufficiency_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        ),
        "restore": contrast_margin(
            restore_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        ),
    }
    parent_predictions = predictions(
        parent_logits, prepared["candidate_ids"]
    )
    sufficiency_predictions = predictions(
        sufficiency_logits, prepared["candidate_ids"]
    )
    restore_predictions = predictions(
        restore_logits, prepared["candidate_ids"]
    )
    result = []
    for index, row in enumerate(batch):
        denominator = max(
            abs(float(
                margins["parent"][index]
                - margins["target"][index]
            )),
            1e-8,
        )
        result.append({
            "pair_id": row["pair_id"],
            "direction": row["direction"],
            "template": row["template"],
            "target_gold": row["target"]["gold"],
            "donor_gold": donor_batch[index]["gold"],
            "parent_donor": (
                parent_predictions[index] == donor_batch[index]["gold"]
            ),
            "sufficiency_donor": (
                sufficiency_predictions[index]
                == donor_batch[index]["gold"]
            ),
            "restore_target": (
                restore_predictions[index] == row["target"]["gold"]
            ),
            "normalized_sufficiency": float(
                (
                    margins["sufficiency"][index]
                    - margins["target"][index]
                )
                / denominator
            ),
            "normalized_restoration_mediation": float(
                (
                    margins["parent"][index]
                    - margins["restore"][index]
                )
                / denominator
            ),
        })
    return result


def screen_blocks(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    universe: list[tuple[int, int, int, int]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = selected_directional_rows(
        model_name, "color", "discovery"
    )
    donors, _ = choose_donors(
        rows, model_name, "color", "discovery"
    )
    parent_blocks = set(universe)
    result_rows = []
    all_batches = list(batches(rows, donors, batch_size))
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        prepared = prepare_batch(
            model,
            layers,
            device,
            batch,
            donor_batch,
            source_depth,
        )
        for event in universe:
            values = evaluate_block_set(
                model,
                device,
                prepared,
                batch,
                donor_batch,
                {event},
                parent_blocks,
            )
            for value in values:
                result_rows.append({
                    "schema_version": (
                        "phase1003_value_channel_block_screen_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": "color",
                    "split": "discovery",
                    "event_id": block_event_id(*event),
                    "layer_index": event[0],
                    "layer_number": event[0] + 1,
                    "kv_head_index": event[1],
                    "channel_start": event[2],
                    "channel_end": event[3],
                    **value,
                })
        del prepared
        print(
            f"[block-screen/{model_name}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    groups = defaultdict(list)
    for row in result_rows:
        groups[row["event_id"]].append(row)
    ranking = []
    event_by_id = {
        block_event_id(*event): event for event in universe
    }
    for event_id, values in groups.items():
        event = event_by_id[event_id]
        ranking.append({
            "event_id": event_id,
            "layer_index": event[0],
            "layer_number": event[0] + 1,
            "kv_head_index": event[1],
            "channel_start": event[2],
            "channel_end": event[3],
            "n": len(values),
            "median_restoration_mediation": float(np.median([
                row["normalized_restoration_mediation"]
                for row in values
            ])),
            "mean_restoration_mediation": float(np.mean([
                row["normalized_restoration_mediation"]
                for row in values
            ])),
            "restore_target_rate": float(np.mean([
                row["restore_target"] for row in values
            ])),
            "median_sufficiency_transfer": float(np.median([
                row["normalized_sufficiency"] for row in values
            ])),
            "mean_sufficiency_transfer": float(np.mean([
                row["normalized_sufficiency"] for row in values
            ])),
            "single_block_donor_rate": float(np.mean([
                row["sufficiency_donor"] for row in values
            ])),
        })
    ranking.sort(key=lambda item: (
        -item["median_restoration_mediation"],
        -item["median_sufficiency_transfer"],
        item["layer_number"],
        item["kv_head_index"],
        item["channel_start"],
    ))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
        item["weighted_score_used"] = False
        item["selection_uses_confirmation"] = False
    return result_rows, ranking


def joint_discovery(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    ranking: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = selected_directional_rows(
        model_name, "color", "discovery"
    )
    donors, _ = choose_donors(
        rows, model_name, "color", "discovery"
    )
    event_order = [
        parse_block_event_id(item["event_id"]) for item in ranking
    ]
    parent_blocks = set(event_order)
    sizes = [size for size in JOINT_SIZES if size <= len(event_order)]
    if len(event_order) not in sizes:
        sizes.append(len(event_order))
    result_rows = []
    all_batches = list(batches(rows, donors, batch_size))
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        prepared = prepare_batch(
            model,
            layers,
            device,
            batch,
            donor_batch,
            source_depth,
        )
        for size in sizes:
            selected = set(event_order[:size])
            values = evaluate_block_set(
                model,
                device,
                prepared,
                batch,
                donor_batch,
                selected,
                parent_blocks,
            )
            event_ids = [
                block_event_id(*event)
                for event in event_order[:size]
            ]
            for value in values:
                result_rows.append({
                    "schema_version": (
                        "phase1003_value_channel_block_joint_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": "color",
                    "split": "discovery",
                    "joint_size": size,
                    "selected_event_ids": event_ids,
                    **value,
                })
        del prepared
        print(
            f"[block-joint/{model_name}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    by_size = {}
    for size in sizes:
        values = [
            row for row in result_rows
            if row["joint_size"] == size
        ]
        item = {
            "n": len(values),
            "selected_event_ids": values[0]["selected_event_ids"],
            "parent_donor_rate": float(np.mean([
                row["parent_donor"] for row in values
            ])),
            "sufficiency_donor_rate": float(np.mean([
                row["sufficiency_donor"] for row in values
            ])),
            "restore_target_rate": float(np.mean([
                row["restore_target"] for row in values
            ])),
            "median_normalized_sufficiency": float(np.median([
                row["normalized_sufficiency"] for row in values
            ])),
            "median_restoration_mediation": float(np.median([
                row["normalized_restoration_mediation"]
                for row in values
            ])),
        }
        item["discovery_gate"] = (
            item["sufficiency_donor_rate"]
            >= thresholds["head_joint_sufficiency_rate"]
            and item["restore_target_rate"]
            >= thresholds["head_joint_restore_rate"]
        )
        by_size[str(size)] = item
    passing = [
        size for size in sizes if by_size[str(size)]["discovery_gate"]
    ]
    if passing:
        selected_size = min(passing)
        selection = {
            "status": "FROZEN_FROM_COLOR_DISCOVERY",
            "selected_size": selected_size,
            "selected_event_ids": by_size[str(selected_size)][
                "selected_event_ids"
            ],
            "selection_uses_confirmation": False,
        }
    else:
        selection = {
            "status": "NO_BLOCK_SET_PASSES",
            "selected_size": None,
            "selected_event_ids": [],
            "selection_uses_confirmation": False,
        }
    return result_rows, {
        "schema_version": (
            "phase1003_value_channel_block_joint_discovery.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "domain": "color",
        "split": "discovery",
        "sizes": by_size,
        "selection": selection,
    }


def frozen_crossparadigm(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    parent_blocks: set[tuple[int, int, int, int]],
    selected_event_ids: list[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_blocks = {
        parse_block_event_id(event_id)
        for event_id in selected_event_ids
    }
    head_summary = read_json(
        OUT_ROOT
        / "value_head_localization"
        / model_name
        / "summary.json"
    )
    domains = [
        domain
        for domain, passed in head_summary[
            "frozen_crossparadigm"
        ]["domain_gates"].items()
        if passed
    ]
    result_rows = []
    donor_audits = {}
    for domain in domains:
        for split in ("discovery", "confirmation"):
            rows = selected_directional_rows(
                model_name, domain, split
            )
            donors, donor_audit = choose_donors(
                rows, model_name, domain, split
            )
            donor_audits[f"{domain}:{split}"] = donor_audit
            all_batches = list(batches(rows, donors, batch_size))
            for batch_number, (batch, donor_batch) in enumerate(
                all_batches, 1
            ):
                prepared = prepare_batch(
                    model,
                    layers,
                    device,
                    batch,
                    donor_batch,
                    source_depth,
                )
                values = evaluate_block_set(
                    model,
                    device,
                    prepared,
                    batch,
                    donor_batch,
                    selected_blocks,
                    parent_blocks,
                )
                for value in values:
                    result_rows.append({
                        "schema_version": (
                            "phase1003_frozen_value_channel_block_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "domain": domain,
                        "split": split,
                        "selected_event_ids": selected_event_ids,
                        **value,
                    })
                del prepared
                print(
                    f"[block-frozen/{model_name}/{domain}/{split}] "
                    f"{batch_number}/{len(all_batches)}",
                    flush=True,
                )
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    cells = {}
    for domain in domains:
        for split in ("discovery", "confirmation"):
            values = [
                row
                for row in result_rows
                if row["domain"] == domain
                and row["split"] == split
            ]
            item = {
                "n": len(values),
                "parent_donor_rate": float(np.mean([
                    row["parent_donor"] for row in values
                ])),
                "sufficiency_donor_rate": float(np.mean([
                    row["sufficiency_donor"] for row in values
                ])),
                "restore_target_rate": float(np.mean([
                    row["restore_target"] for row in values
                ])),
                "median_normalized_sufficiency": float(np.median([
                    row["normalized_sufficiency"] for row in values
                ])),
                "median_restoration_mediation": float(np.median([
                    row["normalized_restoration_mediation"]
                    for row in values
                ])),
            }
            item["frozen_gate"] = (
                item["sufficiency_donor_rate"]
                >= thresholds["head_joint_sufficiency_rate"]
                and item["restore_target_rate"]
                >= thresholds["head_joint_restore_rate"]
            )
            cells[f"{domain}:{split}"] = item
    domain_gates = {
        domain: (
            cells[f"{domain}:discovery"]["frozen_gate"]
            and cells[f"{domain}:confirmation"]["frozen_gate"]
        )
        for domain in domains
    }
    return result_rows, {
        "selected_event_ids": selected_event_ids,
        "donor_audits": donor_audits,
        "cells": cells,
        "domain_gates": domain_gates,
        "passing_domain_count": sum(domain_gates.values()),
    }


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    head_summary = read_json(
        OUT_ROOT
        / "value_head_localization"
        / model_name
        / "summary.json"
    )
    selected_head_ids = head_summary["selection"][
        "selected_event_ids"
    ]
    head_dim = int(head_summary["geometry"]["head_dim"])
    universe = block_universe(selected_head_ids, head_dim)
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][model_name])
    root = (
        OUT_ROOT / "value_channel_block_localization" / model_name
    )
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        screen_rows, ranking = screen_blocks(
            model,
            layers,
            device,
            model_name,
            source_depth,
            universe,
            batch_size,
        )
        write_jsonl(root / "screen_rows.jsonl", screen_rows)
        write_json(root / "block_ranking.json", {
            "schema_version": (
                "phase1003_value_channel_block_ranking.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "selection_domain": "color",
            "selection_split": "discovery",
            "weighted_score_used": False,
            "block_width": BLOCK_WIDTH,
            "event_count": len(universe),
            "ranking": ranking,
        })
        joint_rows, joint = joint_discovery(
            model,
            layers,
            device,
            model_name,
            source_depth,
            ranking,
            batch_size,
        )
        write_jsonl(root / "joint_discovery_rows.jsonl", joint_rows)
        write_json(root / "joint_discovery.json", joint)
        selection = joint["selection"]
        if not selection["selected_event_ids"]:
            raise RuntimeError(
                f"{model_name}: no channel-block set passes"
            )
        frozen_rows, frozen = frozen_crossparadigm(
            model,
            layers,
            device,
            model_name,
            source_depth,
            set(universe),
            selection["selected_event_ids"],
            batch_size,
        )
        write_jsonl(root / "frozen_rows.jsonl", frozen_rows)
        selected_channel_count = (
            selection["selected_size"] * BLOCK_WIDTH
        )
        parent_channel_count = len(universe) * BLOCK_WIDTH
        individual_descent_allowed = (
            selection["selected_size"] <= 16
            and frozen["passing_domain_count"] >= 2
        )
        summary = {
            "schema_version": (
                "phase1003_value_channel_block_summary.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "source_depth": source_depth,
            "selection_domain": "color",
            "selection_split": "discovery",
            "ranking_uses_weighted_score": False,
            "parent_head_event_ids": selected_head_ids,
            "block_width": BLOCK_WIDTH,
            "event_count": len(universe),
            "parent_channel_count": parent_channel_count,
            "ranking_top": ranking[:30],
            "joint_discovery": joint,
            "selection": selection,
            "selected_channel_count": selected_channel_count,
            "compression_fraction": (
                selected_channel_count / parent_channel_count
            ),
            "frozen_crossparadigm": frozen,
            "individual_channel_descent_allowed": (
                individual_descent_allowed
            ),
            "individual_channel_descent_status": (
                "ELIGIBLE"
                if individual_descent_allowed
                else "NO_GO_BROAD_BLOCK_SET_OR_WEAK_CROSS_DOMAIN"
            ),
            "elapsed_seconds": time.time() - started,
            "claim_boundary": (
                "Events are fixed 16-dimensional KV-value blocks, not "
                "individual channels or MLP neurons. Scalar descent is "
                "allowed only after compact cross-domain block closure."
            ),
        }
        write_json(root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            OUT_ROOT
            / "value_channel_block_localization"
            / model_name
            / "summary.json"
        )
        if path.exists():
            summaries[model_name] = read_json(path)
    cross_domain = {}
    for domain in DOMAINS:
        values = [
            summary["frozen_crossparadigm"]["domain_gates"][domain]
            for summary in summaries.values()
            if domain in summary["frozen_crossparadigm"][
                "domain_gates"
            ]
        ]
        cross_domain[domain] = {
            "tested_model_count": len(values),
            "pass_count": sum(values),
        }
    payload = {
        "schema_version": (
            "phase1003_value_channel_block_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "individual_channel_eligible_models": [
            model_name
            for model_name, summary in summaries.items()
            if summary["individual_channel_descent_allowed"]
        ],
        "cross_domain": cross_domain,
    }
    write_json(
        OUT_ROOT
        / "value_channel_block_localization"
        / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
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
