#!/usr/bin/env python3
"""Semantic-mismatch audit for the Phase 1001 minimum head cut.

The original pair-safe roll only guaranteed a different pair. It frequently
selected donors with the recipient's target answer, so it was not a semantic
null. This audit preserves that failure and uses fixed donors from another
world whose target answer equals the recipient's source (wrong) answer.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    SOURCE_DEPTH,
    capture_attention_states,
    forward_with_patches,
    generate_with_patches,
    read_json,
    write_json,
)
from phase1001_attention_source_path_decomposition import selected_inputs
from phase1001_minimum_head_cut import CUT_ROOT, CUT_THRESHOLDS


AUDIT_ROOT = CUT_ROOT / "control_audit"
NULL_COUNT = 4


def stable_hash(*parts):
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_donor_maps(directional, count):
    by_template = defaultdict(list)
    for item in directional:
        by_template[int(item["target"]["template"])].append(item)
    maps = [dict() for _ in range(count)]
    manifest = []
    for recipient in directional:
        candidates = [
            donor
            for donor in by_template[int(recipient["target"]["template"])]
            if donor["pair_id"] != recipient["pair_id"]
            and donor["target"]["world_id"]
            != recipient["target"]["world_id"]
            and donor["target"]["gold"]
            == recipient["source"]["gold"]
        ]
        if len(candidates) < count:
            raise RuntimeError(
                f"underfilled semantic donors for {recipient['pair_id']}: "
                f"{len(candidates)}"
            )
        ordered = sorted(
            candidates,
            key=lambda donor: stable_hash(
                recipient["pair_id"],
                recipient["direction"],
                donor["pair_id"],
                donor["direction"],
            ),
        )
        for null_index in range(count):
            donor = ordered[null_index]
            key = (recipient["pair_id"], recipient["direction"])
            maps[null_index][key] = donor
            manifest.append(
                {
                    "recipient_pair_id": recipient["pair_id"],
                    "recipient_direction": recipient["direction"],
                    "recipient_world_id": recipient["target"][
                        "world_id"
                    ],
                    "recipient_source_gold": recipient["source"]["gold"],
                    "recipient_target_gold": recipient["target"]["gold"],
                    "null_index": null_index,
                    "donor_pair_id": donor["pair_id"],
                    "donor_direction": donor["direction"],
                    "donor_world_id": donor["target"]["world_id"],
                    "donor_target_gold": donor["target"]["gold"],
                    "different_pair": (
                        donor["pair_id"] != recipient["pair_id"]
                    ),
                    "different_world": (
                        donor["target"]["world_id"]
                        != recipient["target"]["world_id"]
                    ),
                    "donor_supports_recipient_source": (
                        donor["target"]["gold"]
                        == recipient["source"]["gold"]
                    ),
                    "donor_does_not_support_recipient_target": (
                        donor["target"]["gold"]
                        != recipient["target"]["gold"]
                    ),
                }
            )
    return maps, manifest


def summarize_candidate(rows):
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


def summarize_natural(rows):
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
            "exact_short_rate": float(
                np.mean([row["exact_short"] for row in values])
            ),
        }
        for condition, values in groups.items()
    }


def run(stage, batch_size, natural_budget):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 cut control audit requires CUDA")
    protocol, selected_pairs, directional, _ = selected_inputs(stage)
    output_root = AUDIT_ROOT / stage
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    frozen = read_json(CUT_ROOT / "discovery" / "frozen_spec.json")
    events = [
        {
            "event_id": event_id,
            "layer_number": int(event_id.split(".")[0][1:]),
            "head_index": int(event_id.split(".")[1][1:]),
            "role": "answer_boundary",
        }
        for event_id in frozen["frozen_event_ids"]
    ]
    donor_maps, manifest = build_donor_maps(directional, NULL_COUNT)
    write_rows(output_root / "donor_manifest.jsonl", manifest)
    if not all(
        row["different_pair"]
        and row["different_world"]
        and row["donor_supports_recipient_source"]
        and row["donor_does_not_support_recipient_target"]
        for row in manifest
    ):
        raise RuntimeError("semantic donor manifest failed")

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
        natural_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
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
            do_logits, _, _ = capture_attention_states(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
            )
            correct_patches = [
                {
                    "event": event,
                    "vectors": target_heads[
                        int(event["layer_number"])
                    ][:, int(event["head_index"]), :],
                }
                for event in events
            ]
            correct_logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                head_patches=correct_patches,
            )
            conditions = [("correct_restore", correct_logits)]
            null_patch_sets = {}
            for null_index, donor_map in enumerate(donor_maps):
                donor_items = [
                    donor_map[(item["pair_id"], item["direction"])]
                    for item in batch
                ]
                donor_cases = [item["target"] for item in donor_items]
                _, donor_heads, _ = capture_attention_states(
                    model,
                    layers,
                    device,
                    donor_cases,
                    candidate_ids,
                )
                null_patches = [
                    {
                        "event": event,
                        "vectors": donor_heads[
                            int(event["layer_number"])
                        ][:, int(event["head_index"]), :],
                    }
                    for event in events
                ]
                null_patch_sets[null_index] = null_patches
                null_logits = forward_with_patches(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                    head_patches=null_patches,
                )
                conditions.append(
                    (f"semantic_null_{null_index}", null_logits)
                )
                del donor_heads

            target_margin = semantic_margin(target_logits, batch)
            do_margin = semantic_margin(do_logits, batch)
            for condition, logits in conditions:
                margin = semantic_margin(logits, batch)
                prediction = prediction_colors(logits)
                for index, item in enumerate(batch):
                    denominator = float(
                        do_margin[index] - target_margin[index]
                    )
                    candidate_rows.append(
                        {
                            "schema_version": (
                                "phase1001_head_cut_semantic_control.v1"
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
                            "prediction": prediction[index],
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

            natural_conditions = {
                "correct_restore": correct_patches,
                "semantic_null_0": null_patch_sets[0],
            }
            for condition, patches in natural_conditions.items():
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
                    natural_budget,
                )
                for index, item in enumerate(batch):
                    result = generated[index]
                    natural_rows.append(
                        {
                            "schema_version": (
                                "phase1001_head_cut_semantic_natural.v1"
                            ),
                            "phase": 1001,
                            "model": MODEL,
                            "partition": item["partition"],
                            "pair_id": item["pair_id"],
                            "direction": item["direction"],
                            "condition": condition,
                            "prediction": result["prediction"],
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
            for _, logits in conditions:
                del logits
            del (
                source_logits,
                source_residuals,
                target_logits,
                target_heads,
                do_logits,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[cut-semantic-audit-{stage}] "
                    f"{batch_number}/{len(batches)}",
                    flush=True,
                )

        candidate_summary = summarize_candidate(candidate_rows)
        natural_summary = summarize_natural(natural_rows)
        write_rows(output_root / "candidate_rows.jsonl", candidate_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
        write_json(
            output_root / "candidate_summary.json", candidate_summary
        )
        write_json(
            output_root / "natural_summary.json", natural_summary
        )
        null_metrics = [
            candidate_summary[f"semantic_null_{index}"]
            for index in range(NULL_COUNT)
        ]
        gate_checks = {
            "manifest_semantic_mismatch": all(
                row["donor_supports_recipient_source"]
                and row["donor_does_not_support_recipient_target"]
                for row in manifest
            ),
            "correct_candidate_mediation": candidate_summary[
                "correct_restore"
            ]["median_mediation_fraction"]
            >= CUT_THRESHOLDS["median_mediation"],
            "correct_natural_restoration": natural_summary[
                "correct_restore"
            ]["target_rate"]
            >= CUT_THRESHOLDS["natural_target_rate"],
            "all_semantic_null_candidate": all(
                abs(metric["median_mediation_fraction"])
                <= CUT_THRESHOLDS["max_cross_pair_mediation"]
                for metric in null_metrics
            ),
            "semantic_null_natural": natural_summary[
                "semantic_null_0"
            ]["source_rate"]
            >= CUT_THRESHOLDS["source_do_flip_rate"],
        }
        original_summary = read_json(
            CUT_ROOT / "discovery" / "summary.json"
        )
        summary = {
            "schema_version": (
                f"phase1001_head_cut_control_audit_{stage}.v1"
            ),
            "phase": 1001,
            "model": MODEL,
            "stage": stage,
            "partition": directional[0]["partition"],
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "frozen_event_ids": frozen["frozen_event_ids"],
            "null_count": NULL_COUNT,
            "manifest_row_count": len(manifest),
            "candidate_summary": candidate_summary,
            "natural_summary": natural_summary,
            "original_cross_pair_control_failed": (
                not original_summary["gate_checks"][
                    "cross_pair_control"
                ]
            ),
            "original_cross_pair_median_mediation": original_summary[
                "control_summary"
            ]["median_cross_pair_mediation"],
            "original_control_only_pair_safe_not_semantic_safe": True,
            "gate_checks": gate_checks,
            "semantic_control_audit_pass": all(gate_checks.values()),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
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
                "passed": summary["semantic_control_audit_pass"],
                "gate_checks": summary["gate_checks"],
                "original_cross_pair_median": summary[
                    "original_cross_pair_median_mediation"
                ],
                "candidate": summary["candidate_summary"],
                "natural": summary["natural_summary"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
