#!/usr/bin/env python3
"""Phase 1001 causal block-pruning alternative for attention channels.

The direct-effect prefix method failed to produce a nontrivial sparse set.
This independent method partitions each frozen head into fixed contiguous
8-channel blocks, measures the causal damage of deleting each block from the
full six-head state, then validates cumulative deletions in the real model.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import COLORS, MODEL
from phase1000_scpg_discovery import (
    batches_by_template,
    write_rows,
)
from phase1001_attention_head_discovery import (
    HEAD_DIM,
    RESULT_ROOT,
    read_json,
    write_json,
)
from phase1001_attention_channel_sparsification import (
    CHANNEL_THRESHOLDS,
    candidate_pass,
    candidate_rows_batch,
    capture_states,
    natural_rows_for_subsets,
    semantic_control_rows,
    summarize_candidate,
    summarize_control,
    summarize_control_natural,
    summarize_natural,
)
from phase1001_attention_source_path_decomposition import selected_inputs
from phase1001_minimum_head_cut import CUT_ROOT
from phase1001_minimum_head_cut_control_audit import build_donor_maps


BLOCK_ROOT = RESULT_ROOT / "channel_block_pruning"
BLOCK_WIDTH = 8
RETAINED_BLOCK_COUNTS = (96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8)


def channel_id(event_id, channel):
    return f"{event_id}.c{channel:03d}"


def make_blocks(event_ids):
    blocks = []
    for event_id in event_ids:
        for start in range(0, HEAD_DIM, BLOCK_WIDTH):
            blocks.append(
                {
                    "block_id": f"{event_id}.b{start // BLOCK_WIDTH:02d}",
                    "event_id": event_id,
                    "start": start,
                    "stop": start + BLOCK_WIDTH,
                    "channel_ids": [
                        channel_id(event_id, channel)
                        for channel in range(start, start + BLOCK_WIDTH)
                    ],
                }
            )
    return blocks


def subset_from_blocks(name, blocks):
    channels = [
        channel
        for block in blocks
        for channel in block["channel_ids"]
    ]
    return {
        "subset_id": name,
        "family": "causal_block_pruning",
        "size": len(channels),
        "channel_ids": channels,
        "block_ids": [block["block_id"] for block in blocks],
        "block_count": len(blocks),
    }


def evaluate_subsets(
    model,
    layers,
    device,
    directional,
    subsets,
    event_lookup,
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
        ) = capture_states(
            model, layers, device, batch, candidate_ids
        )
        rows.extend(
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
                f"[block-{label}] {batch_number}/{len(batches)}",
                flush=True,
            )
    return rows


def rank_block_deletions(full_metric, blocks, leaveout_summary):
    full_balance = min(
        full_metric["median_mediation_fraction"],
        full_metric["mean_sufficiency_transfer"],
    )
    rows = []
    for block in blocks:
        metric = leaveout_summary[f"leaveout/{block['block_id']}"]
        balance = min(
            metric["median_mediation_fraction"],
            metric["mean_sufficiency_transfer"],
        )
        rows.append(
            {
                **block,
                "leaveout_median_mediation": metric[
                    "median_mediation_fraction"
                ],
                "leaveout_mean_sufficiency": metric[
                    "mean_sufficiency_transfer"
                ],
                "balanced_deletion_damage": full_balance - balance,
            }
        )
    rows.sort(
        key=lambda item: (
            item["balanced_deletion_damage"],
            item["block_id"],
        )
    )
    for index, item in enumerate(rows, 1):
        item["removal_rank"] = index
    return rows


def run(stage, batch_size, natural_budget):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 block pruning requires CUDA")
    protocol, selected_pairs, directional, _ = selected_inputs(stage)
    output_root = BLOCK_ROOT / stage
    output_root.mkdir(parents=True, exist_ok=True)
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    head_spec = read_json(CUT_ROOT / "discovery/frozen_spec.json")
    event_ids = list(head_spec["frozen_event_ids"])
    event_lookup = {
        event_id: {
            "event_id": event_id,
            "layer_number": int(event_id.split(".")[0][1:]),
            "head_index": int(event_id.split(".")[1][1:]),
            "role": "answer_boundary",
        }
        for event_id in event_ids
    }
    blocks = make_blocks(event_ids)
    if len(blocks) != 96:
        raise RuntimeError("block count drift")

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

        if stage == "discovery":
            full_subset = subset_from_blocks("full/k96", blocks)
            full_rows = evaluate_subsets(
                model,
                layers,
                device,
                directional,
                [full_subset],
                event_lookup,
                candidate_ids,
                batch_size,
                "full",
            )
            full_metric = summarize_candidate(full_rows)[
                full_subset["subset_id"]
            ]
            leaveout_subsets = [
                subset_from_blocks(
                    f"leaveout/{removed['block_id']}",
                    [
                        block
                        for block in blocks
                        if block["block_id"] != removed["block_id"]
                    ],
                )
                for removed in blocks
            ]
            leaveout_rows = evaluate_subsets(
                model,
                layers,
                device,
                directional,
                leaveout_subsets,
                event_lookup,
                candidate_ids,
                batch_size,
                "leaveout",
            )
            leaveout_summary = summarize_candidate(leaveout_rows)
            removal_ranking = rank_block_deletions(
                full_metric, blocks, leaveout_summary
            )
            removed_order = [
                item["block_id"] for item in removal_ranking
            ]
            block_lookup = {
                block["block_id"]: block for block in blocks
            }
            cumulative_subsets = []
            for retained_count in RETAINED_BLOCK_COUNTS:
                removed_count = len(blocks) - retained_count
                removed = set(removed_order[:removed_count])
                retained = [
                    block
                    for block in blocks
                    if block["block_id"] not in removed
                ]
                cumulative_subsets.append(
                    subset_from_blocks(
                        f"causal_retain/k{retained_count}",
                        retained,
                    )
                )
            cumulative_rows = evaluate_subsets(
                model,
                layers,
                device,
                directional,
                cumulative_subsets,
                event_lookup,
                candidate_ids,
                batch_size,
                "cumulative",
            )
            cumulative_summary = summarize_candidate(cumulative_rows)
            candidate_rows = full_rows + leaveout_rows + cumulative_rows
            candidate_summary = {
                **{full_subset["subset_id"]: full_metric},
                **leaveout_summary,
                **cumulative_summary,
            }
            natural_subsets = [
                subset
                for subset in cumulative_subsets
                if candidate_pass(
                    cumulative_summary[subset["subset_id"]]
                )
            ]
            if not natural_subsets:
                natural_subsets = [full_subset]
        else:
            frozen = read_json(BLOCK_ROOT / "discovery/frozen_spec.json")
            block_lookup = {
                block["block_id"]: block for block in blocks
            }
            retained = [
                block_lookup[block_id]
                for block_id in frozen["retained_block_ids"]
            ]
            frozen_subset = subset_from_blocks(
                frozen["subset_id"], retained
            )
            candidate_rows = evaluate_subsets(
                model,
                layers,
                device,
                directional,
                [frozen_subset],
                event_lookup,
                candidate_ids,
                batch_size,
                "confirmation",
            )
            candidate_summary = summarize_candidate(candidate_rows)
            cumulative_summary = candidate_summary
            full_metric = None
            leaveout_summary = {}
            removal_ranking = []
            cumulative_subsets = [frozen_subset]
            natural_subsets = [frozen_subset]

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

        if stage == "discovery":
            eligible = [
                subset
                for subset in natural_subsets
                if natural_summary[subset["subset_id"]]["target_rate"]
                >= CHANNEL_THRESHOLDS["natural_target_rate"]
            ]
            eligible.sort(key=lambda subset: subset["size"])
            if eligible:
                frozen_subset = eligible[0]
                natural_gate = True
            else:
                frozen_subset = natural_subsets[-1]
                natural_gate = False
        else:
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
        write_rows(output_root / "candidate_rows.jsonl", candidate_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
        write_rows(output_root / "control_rows.jsonl", control_rows)
        write_rows(
            output_root / "control_natural_rows.jsonl",
            control_natural_rows,
        )
        write_rows(output_root / "donor_manifest.jsonl", donor_manifest)
        write_json(
            output_root / "candidate_summary.json", candidate_summary
        )
        write_json(
            output_root / "natural_summary.json", natural_summary
        )
        write_json(
            output_root / "control_summary.json", control_summary
        )
        write_json(
            output_root / "control_natural_summary.json",
            control_natural_summary,
        )
        if stage == "discovery":
            write_json(
                output_root / "block_removal_ranking.json",
                {
                    "schema_version": (
                        "phase1001_block_removal_ranking.v1"
                    ),
                    "blocks": removal_ranking,
                },
            )

        metric = candidate_summary[frozen_subset["subset_id"]]
        null_metrics = [
            control_summary[f"semantic_null_{index}"]
            for index in range(4)
        ]
        gate_checks = {
            "nontrivial_pruning": frozen_subset["size"] < 768,
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
            "schema_version": "phase1001_frozen_channel_blocks.v1",
            "phase": 1001,
            "model": MODEL,
            "subset_id": frozen_subset["subset_id"],
            "retained_block_ids": frozen_subset["block_ids"],
            "retained_block_count": frozen_subset["block_count"],
            "channel_ids": frozen_subset["channel_ids"],
            "channel_count": frozen_subset["size"],
            "parent_head_event_ids": event_ids,
            "selection_partition": "validation",
            "selection_uses_holdout": False,
            "frozen_before_holdout": stage == "discovery",
        }
        summary = {
            "schema_version": (
                f"phase1001_channel_block_{stage}_summary.v1"
            ),
            "phase": 1001,
            "model": MODEL,
            "stage": stage,
            "partition": directional[0]["partition"],
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "block_width": BLOCK_WIDTH,
            "initial_block_count": len(blocks),
            "full_metric": full_metric,
            "leaveout_summary": leaveout_summary,
            "block_removal_ranking": removal_ranking,
            "cumulative_summary": cumulative_summary,
            "frozen_subset_id": frozen_subset["subset_id"],
            "frozen_block_count": frozen_subset["block_count"],
            "frozen_channel_count": frozen_subset["size"],
            "frozen_block_ids": frozen_subset["block_ids"],
            "frozen_candidate_metrics": metric,
            "frozen_natural_metrics": natural_summary[
                frozen_subset["subset_id"]
            ],
            "control_summary": control_summary,
            "control_natural_summary": control_natural_summary,
            "thresholds": CHANNEL_THRESHOLDS,
            "gate_checks": gate_checks,
            "channel_block_pruning_gate_pass": all(
                gate_checks.values()
            ),
            "minimality_scope": (
                "fixed 8-channel blocks ranked by single-block "
                "deletion damage; not exhaustive channel subset"
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
                "passed": summary["channel_block_pruning_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "frozen_block_count": summary["frozen_block_count"],
                "frozen_channel_count": summary[
                    "frozen_channel_count"
                ],
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
