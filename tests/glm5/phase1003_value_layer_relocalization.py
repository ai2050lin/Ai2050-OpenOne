#!/usr/bin/env python3
"""Relocalize cache-value layers after frozen Phase1002 layers fail.

Selection uses only Phase1003 color discovery data.  Layers are ranked
lexicographically by direct restoration mediation and then direct sufficiency;
no weighted score is formed.
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1003_anchor_natural_confirmation import capture_prompt_depth
from phase1003_anchor_subset_exhaustive import choose_donors
from phase1003_crossparadigm_kv_replication import (
    build_cache,
    clone_cache_mix,
    continue_cache,
    contrast_margin,
    patch_spec,
    predictions,
    step_case,
)
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


JOINT_SIZES = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40)


def batches(
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
):
    groups = defaultdict(list)
    for row, donor in zip(rows, donors):
        groups[int(row["template"])].append((row, donor))
    for _, values in sorted(groups.items()):
        values.sort(key=lambda item: (
            item[0]["pair_id"], item[0]["direction"]
        ))
        for start in range(0, len(values), batch_size):
            chunk = values[start : start + batch_size]
            yield (
                [item[0] for item in chunk],
                [item[1] for item in chunk],
            )


def prepare_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    donor_batch: list[dict[str, Any]],
    source_depth: int,
) -> dict[str, Any]:
    target_cases = [row["target"] for row in batch]
    donor_hidden = capture_prompt_depth(
        model, layers, device, donor_batch, source_depth
    )
    semantic_step = int(target_cases[0]["semantic_step"])
    prefix_step = semantic_step - 1
    if prefix_step < 0:
        raise RuntimeError("semantic step lacks generated prefix")
    prefix_cases = [
        step_case(case, prefix_step) for case in target_cases
    ]
    prefix_patch = patch_spec(
        source_depth, prefix_cases, donor_batch, donor_hidden
    )
    target_cache = build_cache(
        model, layers, device, prefix_cases, None
    )
    source_cache = build_cache(
        model, layers, device, prefix_cases, prefix_patch
    )
    return {
        "target_cases": target_cases,
        "donor_batch": donor_batch,
        "candidate_ids": target_cases[0]["candidate_token_ids"],
        "current_ids": [
            int(case["answer_token_ids"][prefix_step])
            for case in target_cases
        ],
        "prefix_length": len(prefix_cases[0]["input_ids"]),
        "target_cache": target_cache,
        "source_cache": source_cache,
        "donor_hidden": donor_hidden,
    }


def cache_logits(
    model,
    device,
    prepared: dict[str, Any],
    source_value_layers: set[int],
) -> torch.Tensor:
    return continue_cache(
        model,
        device,
        prepared["current_ids"],
        prepared["prefix_length"],
        clone_cache_mix(
            prepared["target_cache"],
            prepared["source_cache"],
            set(),
            source_value_layers,
            model.config,
        ),
        prepared["candidate_ids"],
    )


def screen_layers(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    domain = "color"
    split = "discovery"
    rows = selected_directional_rows(model_name, domain, split)
    donors, _ = choose_donors(rows, model_name, domain, split)
    layer_count = len(layers)
    all_layers = set(range(layer_count))
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
        target_logits = cache_logits(model, device, prepared, set())
        value_logits = cache_logits(
            model, device, prepared, all_layers
        )
        labels = list(prepared["candidate_ids"])
        target_margin = contrast_margin(
            target_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        )
        value_margin = contrast_margin(
            value_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        )
        value_span = value_margin - target_margin
        for layer_index in range(layer_count):
            sufficiency_logits = cache_logits(
                model, device, prepared, {layer_index}
            )
            restore_logits = cache_logits(
                model,
                device,
                prepared,
                all_layers - {layer_index},
            )
            sufficiency_margin = contrast_margin(
                sufficiency_logits,
                labels,
                donor_batch,
                prepared["target_cases"],
            )
            restore_margin = contrast_margin(
                restore_logits,
                labels,
                donor_batch,
                prepared["target_cases"],
            )
            sufficiency_predictions = predictions(
                sufficiency_logits, prepared["candidate_ids"]
            )
            restore_predictions = predictions(
                restore_logits, prepared["candidate_ids"]
            )
            for index, row in enumerate(batch):
                denominator = max(
                    abs(float(value_span[index])), 1e-8
                )
                result_rows.append({
                    "schema_version": (
                        "phase1003_value_layer_screen_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "split": split,
                    "pair_id": row["pair_id"],
                    "direction": row["direction"],
                    "template": row["template"],
                    "layer_index": layer_index,
                    "layer_number": layer_index + 1,
                    "target_gold": row["target"]["gold"],
                    "donor_gold": donor_batch[index]["gold"],
                    "single_layer_donor": (
                        sufficiency_predictions[index]
                        == donor_batch[index]["gold"]
                    ),
                    "restore_layer_target": (
                        restore_predictions[index]
                        == row["target"]["gold"]
                    ),
                    "normalized_single_layer_sufficiency": float(
                        (
                            sufficiency_margin[index]
                            - target_margin[index]
                        )
                        / denominator
                    ),
                    "normalized_restore_layer_mediation": float(
                        (
                            value_margin[index]
                            - restore_margin[index]
                        )
                        / denominator
                    ),
                })
        del prepared, target_logits, value_logits
        print(
            f"[screen/{model_name}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )

    groups = defaultdict(list)
    for row in result_rows:
        groups[int(row["layer_index"])].append(row)
    ranking = []
    for layer_index, values in groups.items():
        ranking.append({
            "layer_index": layer_index,
            "layer_number": layer_index + 1,
            "n": len(values),
            "median_restoration_mediation": float(np.median([
                row["normalized_restore_layer_mediation"]
                for row in values
            ])),
            "mean_restoration_mediation": float(np.mean([
                row["normalized_restore_layer_mediation"]
                for row in values
            ])),
            "restore_target_rate": float(np.mean([
                row["restore_layer_target"] for row in values
            ])),
            "median_sufficiency_transfer": float(np.median([
                row["normalized_single_layer_sufficiency"]
                for row in values
            ])),
            "mean_sufficiency_transfer": float(np.mean([
                row["normalized_single_layer_sufficiency"]
                for row in values
            ])),
            "single_layer_donor_rate": float(np.mean([
                row["single_layer_donor"] for row in values
            ])),
        })
    ranking.sort(key=lambda item: (
        -item["median_restoration_mediation"],
        -item["median_sufficiency_transfer"],
        item["layer_number"],
    ))
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
        item["ranking_rule"] = (
            "lexicographic(restoration_mediation, "
            "sufficiency_transfer, layer_number)"
        )
        item["weighted_score_used"] = False
        item["selection_uses_confirmation"] = False
    return result_rows, ranking


def evaluate_layer_set(
    model,
    device,
    prepared: dict[str, Any],
    batch: list[dict[str, Any]],
    donor_batch: list[dict[str, Any]],
    selected_layers: set[int],
    layer_count: int,
) -> list[dict[str, Any]]:
    all_layers = set(range(layer_count))
    target_logits = cache_logits(model, device, prepared, set())
    value_logits = cache_logits(
        model, device, prepared, all_layers
    )
    sufficiency_logits = cache_logits(
        model, device, prepared, selected_layers
    )
    restore_logits = cache_logits(
        model,
        device,
        prepared,
        all_layers - selected_layers,
    )
    labels = list(prepared["candidate_ids"])
    margins = {
        "target": contrast_margin(
            target_logits,
            labels,
            donor_batch,
            prepared["target_cases"],
        ),
        "value": contrast_margin(
            value_logits,
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
    sufficiency_predictions = predictions(
        sufficiency_logits, prepared["candidate_ids"]
    )
    restore_predictions = predictions(
        restore_logits, prepared["candidate_ids"]
    )
    value_predictions = predictions(
        value_logits, prepared["candidate_ids"]
    )
    result = []
    for index, row in enumerate(batch):
        denominator = max(
            abs(float(
                margins["value"][index]
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
            "value_parent_donor": (
                value_predictions[index] == donor_batch[index]["gold"]
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
                    margins["value"][index]
                    - margins["restore"][index]
                )
                / denominator
            ),
        })
    return result


def joint_discovery(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    batch_size: int,
    ranking: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    domain = "color"
    split = "discovery"
    rows = selected_directional_rows(model_name, domain, split)
    donors, _ = choose_donors(rows, model_name, domain, split)
    layer_count = len(layers)
    sizes = [size for size in JOINT_SIZES if size <= layer_count]
    if layer_count not in sizes:
        sizes.append(layer_count)
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
            selected = {
                int(item["layer_index"])
                for item in ranking[:size]
            }
            values = evaluate_layer_set(
                model,
                device,
                prepared,
                batch,
                donor_batch,
                selected,
                layer_count,
            )
            for value in values:
                result_rows.append({
                    "schema_version": (
                        "phase1003_value_layer_joint_discovery_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "split": split,
                    "joint_size": size,
                    "selected_layer_numbers": [
                        item["layer_number"]
                        for item in ranking[:size]
                    ],
                    **value,
                })
        del prepared
        print(
            f"[joint/{model_name}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )

    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    summary_by_size = {}
    for size in sizes:
        values = [
            row for row in result_rows
            if row["joint_size"] == size
        ]
        item = {
            "n": len(values),
            "selected_layer_numbers": values[0][
                "selected_layer_numbers"
            ],
            "value_parent_donor_rate": float(np.mean([
                row["value_parent_donor"] for row in values
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
            >= thresholds["frozen_value_layer_sufficiency_rate"]
            and item["restore_target_rate"]
            >= thresholds["frozen_value_layer_restore_rate"]
        )
        summary_by_size[str(size)] = item
    passing = [
        size
        for size in sizes
        if summary_by_size[str(size)]["discovery_gate"]
    ]
    if passing:
        selected_size = min(passing)
        selection = {
            "status": "FROZEN_FROM_COLOR_DISCOVERY",
            "selected_size": selected_size,
            "selected_layer_numbers": summary_by_size[
                str(selected_size)
            ]["selected_layer_numbers"],
            "selection_uses_confirmation": False,
        }
    else:
        selection = {
            "status": "NO_LAYER_SET_PASSES",
            "selected_size": None,
            "selected_layer_numbers": [],
            "selection_uses_confirmation": False,
        }
    return result_rows, {
        "schema_version": "phase1003_value_layer_joint_discovery.v1",
        "phase": PHASE,
        "model": model_name,
        "domain": "color",
        "split": "discovery",
        "sizes": summary_by_size,
        "selection": selection,
    }


def frozen_crossparadigm(
    model,
    layers,
    device,
    model_name: str,
    source_depth: int,
    batch_size: int,
    selected_layer_numbers: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_layers = {
        int(number) - 1 for number in selected_layer_numbers
    }
    layer_count = len(layers)
    natural = read_json(
        OUT_ROOT
        / "anchor_natural"
        / model_name
        / "summary.json"
    )
    domains = list(natural["domains"])
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
                values = evaluate_layer_set(
                    model,
                    device,
                    prepared,
                    batch,
                    donor_batch,
                    selected_layers,
                    layer_count,
                )
                for value in values:
                    result_rows.append({
                        "schema_version": (
                            "phase1003_frozen_value_layer_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "domain": domain,
                        "split": split,
                        "selected_layer_numbers": (
                            selected_layer_numbers
                        ),
                        **value,
                    })
                del prepared
                print(
                    f"[frozen/{model_name}/{domain}/{split}] "
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
                "value_parent_donor_rate": float(np.mean([
                    row["value_parent_donor"] for row in values
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
                item["value_parent_donor_rate"]
                >= thresholds["cache_value_donor_rate"]
                and item["sufficiency_donor_rate"]
                >= thresholds[
                    "frozen_value_layer_sufficiency_rate"
                ]
                and item["restore_target_rate"]
                >= thresholds["frozen_value_layer_restore_rate"]
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
        "selected_layer_numbers": selected_layer_numbers,
        "donor_audits": donor_audits,
        "cells": cells,
        "domain_gates": domain_gates,
        "passing_domain_count": sum(domain_gates.values()),
    }


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][model_name])
    model = tokenizer = None
    started = time.time()
    root = OUT_ROOT / "value_layer_relocalization" / model_name
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
        write_jsonl(root / "screen_rows.jsonl", screen_rows)
        write_json(root / "layer_ranking.json", {
            "schema_version": "phase1003_value_layer_ranking.v1",
            "phase": PHASE,
            "model": model_name,
            "selection_domain": "color",
            "selection_split": "discovery",
            "weighted_score_used": False,
            "ranking": ranking,
        })
        joint_rows, joint = joint_discovery(
            model,
            layers,
            device,
            model_name,
            source_depth,
            batch_size,
            ranking,
        )
        write_jsonl(
            root / "joint_discovery_rows.jsonl", joint_rows
        )
        write_json(root / "joint_discovery.json", joint)
        selection = joint["selection"]
        if not selection["selected_layer_numbers"]:
            raise RuntimeError(
                f"{model_name}: no value layer set passes discovery"
            )
        frozen_rows, frozen = frozen_crossparadigm(
            model,
            layers,
            device,
            model_name,
            source_depth,
            batch_size,
            selection["selected_layer_numbers"],
        )
        write_jsonl(root / "frozen_rows.jsonl", frozen_rows)
        summary = {
            "schema_version": (
                "phase1003_value_layer_relocalization_summary.v1"
            ),
            "phase": PHASE,
            "model": model_name,
            "status": "complete",
            "source_depth": source_depth,
            "layer_count": len(layers),
            "selection_domain": "color",
            "selection_split": "discovery",
            "ranking_uses_weighted_score": False,
            "ranking_top": ranking[:10],
            "joint_discovery": joint,
            "selection": selection,
            "frozen_crossparadigm": frozen,
            "elapsed_seconds": time.time() - started,
            "claim_boundary": (
                "Layers are newly localized for the Phase1003 protocol. "
                "Cross-domain repetition is functional; it is not evidence "
                "that layer numbers are universal across prompts or models."
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
            / "value_layer_relocalization"
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
            "phase1003_value_layer_relocalization_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_domain": cross_domain,
    }
    write_json(
        OUT_ROOT / "value_layer_relocalization" / "summary.json",
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
