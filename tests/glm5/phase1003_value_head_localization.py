#!/usr/bin/env python3
"""Causally localize KV-value heads inside Phase1003-frozen layers."""
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
from phase1003_value_layer_relocalization import (
    batches,
    contrast_margin,
    prepare_batch,
    predictions,
)


JOINT_SIZES = (
    1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128,
)


def head_event_id(layer_index: int, head_index: int) -> str:
    return f"l{layer_index + 1}.kvh{head_index}"


def clone_head_mix(
    target_cache,
    source_cache,
    source_heads: set[tuple[int, int]],
    model_config,
) -> DynamicCache:
    if len(target_cache.layers) != len(source_cache.layers):
        raise RuntimeError("cache layer count drift")
    by_layer = defaultdict(list)
    for layer_index, head_index in source_heads:
        by_layer[int(layer_index)].append(int(head_index))
    data = []
    for layer_index, (target_layer, source_layer) in enumerate(
        zip(target_cache.layers, source_cache.layers)
    ):
        values = target_layer.values.detach().clone()
        for head_index in by_layer.get(layer_index, []):
            values[:, head_index, :, :] = source_layer.values[
                :, head_index, :, :
            ]
        data.append((
            target_layer.keys.detach().clone(),
            values,
        ))
    return DynamicCache(data, config=model_config)


def continue_head_cache(
    model,
    device,
    prepared: dict[str, Any],
    source_heads: set[tuple[int, int]],
) -> torch.Tensor:
    from phase1003_crossparadigm_kv_replication import continue_cache

    return continue_cache(
        model,
        device,
        prepared["current_ids"],
        prepared["prefix_length"],
        clone_head_mix(
            prepared["target_cache"],
            prepared["source_cache"],
            source_heads,
            model.config,
        ),
        prepared["candidate_ids"],
    )


def head_universe(
    prepared: dict[str, Any],
    selected_layer_numbers: list[int],
) -> tuple[list[tuple[int, int]], int, int]:
    selected_layers = [
        int(number) - 1 for number in selected_layer_numbers
    ]
    head_counts = {
        int(
            prepared["target_cache"].layers[layer_index]
            .values.shape[1]
        )
        for layer_index in selected_layers
    }
    head_dims = {
        int(
            prepared["target_cache"].layers[layer_index]
            .values.shape[-1]
        )
        for layer_index in selected_layers
    }
    if len(head_counts) != 1 or len(head_dims) != 1:
        raise RuntimeError(
            f"KV geometry drift heads={head_counts}, dims={head_dims}"
        )
    head_count = next(iter(head_counts))
    head_dim = next(iter(head_dims))
    universe = [
        (layer_index, head_index)
        for layer_index in selected_layers
        for head_index in range(head_count)
    ]
    return universe, head_count, head_dim


def evaluate_head_set(
    model,
    device,
    prepared: dict[str, Any],
    batch: list[dict[str, Any]],
    donor_batch: list[dict[str, Any]],
    selected_heads: set[tuple[int, int]],
    parent_heads: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    target_logits = continue_head_cache(
        model, device, prepared, set()
    )
    parent_logits = continue_head_cache(
        model, device, prepared, parent_heads
    )
    sufficiency_logits = continue_head_cache(
        model, device, prepared, selected_heads
    )
    restore_logits = continue_head_cache(
        model,
        device,
        prepared,
        parent_heads - selected_heads,
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


def screen_heads(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    selected_layer_numbers: list[int],
    batch_size: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    domain = "color"
    split = "discovery"
    rows = selected_directional_rows(model_name, domain, split)
    donors, _ = choose_donors(rows, model_name, domain, split)
    result_rows = []
    universe = None
    kv_head_count = head_dim = None
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
        current_universe, current_head_count, current_head_dim = (
            head_universe(prepared, selected_layer_numbers)
        )
        if universe is None:
            universe = current_universe
            kv_head_count = current_head_count
            head_dim = current_head_dim
        elif universe != current_universe:
            raise RuntimeError("head universe drift")
        parent_heads = set(universe)
        for event in universe:
            values = evaluate_head_set(
                model,
                device,
                prepared,
                batch,
                donor_batch,
                {event},
                parent_heads,
            )
            for value in values:
                result_rows.append({
                    "schema_version": (
                        "phase1003_value_head_screen_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "split": split,
                    "event_id": head_event_id(*event),
                    "layer_index": event[0],
                    "layer_number": event[0] + 1,
                    "kv_head_index": event[1],
                    **value,
                })
        del prepared
        print(
            f"[head-screen/{model_name}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    if universe is None:
        raise RuntimeError("empty head universe")
    groups = defaultdict(list)
    for row in result_rows:
        groups[row["event_id"]].append(row)
    ranking = []
    event_by_id = {
        head_event_id(*event): event for event in universe
    }
    for event_id, values in groups.items():
        layer_index, head_index = event_by_id[event_id]
        ranking.append({
            "event_id": event_id,
            "layer_index": layer_index,
            "layer_number": layer_index + 1,
            "kv_head_index": head_index,
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
            "single_head_donor_rate": float(np.mean([
                row["sufficiency_donor"] for row in values
            ])),
        })
    ranking.sort(key=lambda item: (
        -item["median_restoration_mediation"],
        -item["median_sufficiency_transfer"],
        item["layer_number"],
        item["kv_head_index"],
    ))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
        item["weighted_score_used"] = False
        item["selection_uses_confirmation"] = False
    geometry = {
        "selected_layer_numbers": selected_layer_numbers,
        "kv_head_count_per_layer": kv_head_count,
        "head_dim": head_dim,
        "event_count": len(universe),
    }
    return result_rows, ranking, geometry


def joint_discovery(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    selected_layer_numbers: list[int],
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
        (int(item["layer_index"]), int(item["kv_head_index"]))
        for item in ranking
    ]
    parent_heads = set(event_order)
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
            values = evaluate_head_set(
                model,
                device,
                prepared,
                batch,
                donor_batch,
                selected,
                parent_heads,
            )
            event_ids = [
                head_event_id(*event) for event in event_order[:size]
            ]
            for value in values:
                result_rows.append({
                    "schema_version": (
                        "phase1003_value_head_joint_row.v1"
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
            f"[head-joint/{model_name}] "
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
            "status": "NO_HEAD_SET_PASSES",
            "selected_size": None,
            "selected_event_ids": [],
            "selection_uses_confirmation": False,
        }
    return result_rows, {
        "schema_version": "phase1003_value_head_joint_discovery.v1",
        "phase": PHASE,
        "model": model_name,
        "domain": "color",
        "split": "discovery",
        "sizes": by_size,
        "selection": selection,
    }


def parse_event_id(event_id: str) -> tuple[int, int]:
    layer_text, head_text = event_id.split(".")
    return int(layer_text[1:]) - 1, int(head_text[3:])


def frozen_crossparadigm(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    selected_layer_numbers: list[int],
    selected_event_ids: list[str],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_heads = {
        parse_event_id(event_id) for event_id in selected_event_ids
    }
    layer_summary = read_json(
        OUT_ROOT
        / "value_layer_relocalization"
        / model_name
        / "summary.json"
    )
    domains = [
        domain
        for domain, passed in layer_summary[
            "frozen_crossparadigm"
        ]["domain_gates"].items()
        if passed
    ]
    result_rows = []
    donor_audits = {}
    parent_heads = None
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
                universe, _, _ = head_universe(
                    prepared, selected_layer_numbers
                )
                if parent_heads is None:
                    parent_heads = set(universe)
                values = evaluate_head_set(
                    model,
                    device,
                    prepared,
                    batch,
                    donor_batch,
                    selected_heads,
                    parent_heads,
                )
                for value in values:
                    result_rows.append({
                        "schema_version": (
                            "phase1003_frozen_value_head_row.v1"
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
                    f"[head-frozen/{model_name}/{domain}/{split}] "
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
    layer_summary = read_json(
        OUT_ROOT
        / "value_layer_relocalization"
        / model_name
        / "summary.json"
    )
    selected_layers = layer_summary["selection"][
        "selected_layer_numbers"
    ]
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][model_name])
    root = OUT_ROOT / "value_head_localization" / model_name
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        screen_rows, ranking, geometry = screen_heads(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            batch_size,
        )
        write_jsonl(root / "screen_rows.jsonl", screen_rows)
        write_json(root / "head_ranking.json", {
            "schema_version": "phase1003_value_head_ranking.v1",
            "phase": PHASE,
            "model": model_name,
            "selection_domain": "color",
            "selection_split": "discovery",
            "weighted_score_used": False,
            "geometry": geometry,
            "ranking": ranking,
        })
        joint_rows, joint = joint_discovery(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            ranking,
            batch_size,
        )
        write_jsonl(root / "joint_discovery_rows.jsonl", joint_rows)
        write_json(root / "joint_discovery.json", joint)
        selection = joint["selection"]
        if not selection["selected_event_ids"]:
            raise RuntimeError(
                f"{model_name}: no head set passes discovery"
            )
        frozen_rows, frozen = frozen_crossparadigm(
            model,
            layers,
            device,
            model_name,
            source_depth,
            selected_layers,
            selection["selected_event_ids"],
            batch_size,
        )
        write_jsonl(root / "frozen_rows.jsonl", frozen_rows)
        summary = {
            "schema_version": (
                "phase1003_value_head_localization_summary.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "source_depth": source_depth,
            "selection_domain": "color",
            "selection_split": "discovery",
            "ranking_uses_weighted_score": False,
            "parent_layer_numbers": selected_layers,
            "geometry": geometry,
            "ranking_top": ranking[:20],
            "joint_discovery": joint,
            "selection": selection,
            "frozen_crossparadigm": frozen,
            "compression_fraction": (
                selection["selected_size"]
                / geometry["event_count"]
            ),
            "elapsed_seconds": time.time() - started,
            "claim_boundary": (
                "Events are KV-cache value heads, not attention query "
                "heads or individual neurons. Localization is conditional "
                "on the Phase1003 parent layer set."
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
            / "value_head_localization"
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
            "phase1003_value_head_localization_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_domain": cross_domain,
    }
    write_json(
        OUT_ROOT / "value_head_localization" / "summary.json",
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
