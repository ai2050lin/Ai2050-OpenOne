#!/usr/bin/env python3
"""Phase 1001 disjoint holdout confirmation of frozen Qwen3 attention heads.

The validation partition selected and ranked the heads. This script never
re-ranks them: it applies the frozen ordering and frozen joint size to the
Phase 1000 confirmation worlds, then repeats physical instrumentation,
necessity, sufficiency, wrong-output-location, cross-pair, and natural
generation checks.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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
from phase1000_scpg_confirmation import (
    one_direction_per_pair,
    select_confirmation_pairs,
)
from phase1000_scpg_discovery import (
    batches_by_template,
    capture_residuals,
    prediction_colors,
    read_jsonl,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1001_attention_head_discovery import (
    HEAD_COUNT,
    HEAD_DIM,
    HEAD_THRESHOLDS,
    PHASE1000_ROOT,
    RESULT_ROOT,
    SOURCE_DEPTH,
    TARGET_LAYERS,
    capture_attention_states,
    forward_with_patches,
    head_events,
    joint_rows_for_batch,
    mediation_row,
    natural_joint_rows,
    read_json,
    selected_head_controls,
    summarize_controls,
    summarize_joint,
    summarize_mediation,
    summarize_natural,
    write_json,
)


OUTPUT_ROOT = RESULT_ROOT / "head_confirmation"
EXPECTED_CONFIRMATION_PAIRS = 512


def load_frozen_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[str],
    set[str],
]:
    protocol_root = PHASE1000_ROOT / "protocol"
    phase1000_discovery_root = PHASE1000_ROOT / "discovery"
    phase1001_discovery_root = RESULT_ROOT / "head_discovery"

    protocol = read_json(protocol_root / "protocol.json")
    cases = read_jsonl(protocol_root / "cases.jsonl")
    factor_pairs = read_jsonl(protocol_root / "factor_pairs.jsonl")
    phase1000_validation_pairs = read_jsonl(
        phase1000_discovery_root / "selected_pairs.jsonl"
    )
    phase1001_summary = read_json(phase1001_discovery_root / "summary.json")
    frozen = read_json(phase1001_discovery_root / "frozen_spec.json")

    if not phase1001_summary.get("head_discovery_gate_pass"):
        raise RuntimeError("Phase 1001 discovery gate is not open")
    if (
        not frozen.get("frozen_before_holdout")
        or frozen.get("selection_uses_holdout")
        or frozen.get("selection_partition") != "validation"
    ):
        raise RuntimeError("invalid frozen head contract")

    case_by_id = {row["record_id"]: row for row in cases}
    validation_pair_ids = {
        row["pair_id"] for row in phase1000_validation_pairs
    }
    confirmation_pairs = select_confirmation_pairs(
        factor_pairs,
        case_by_id,
        validation_pair_ids,
    )
    directional = one_direction_per_pair(confirmation_pairs, case_by_id)
    validation_worlds = {
        case_by_id[row["arm0_record_id"]]["world_id"]
        for row in phase1000_validation_pairs
    }
    confirmation_worlds = {
        case_by_id[row["arm0_record_id"]]["world_id"]
        for row in confirmation_pairs
    }
    return (
        protocol,
        phase1001_summary,
        frozen,
        confirmation_pairs,
        directional,
        validation_worlds,
        confirmation_worlds,
    )


def instrument_rows_for_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    target_attention: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [row["target"] for row in batch]
    all_head_restore = [
        {
            "event": event,
            "vectors": target_heads[event["layer_number"]][
                :, event["head_index"], :
            ],
        }
        for event in head_events()
    ]
    all_head_logits = forward_with_patches(
        model,
        layers,
        device,
        target_cases,
        candidate_ids,
        source_patch=source_patch,
        head_patches=all_head_restore,
    )
    full_attention_logits = forward_with_patches(
        model,
        layers,
        device,
        target_cases,
        candidate_ids,
        source_patch=source_patch,
        attention_patches=target_attention,
    )
    all_margin = semantic_margin(all_head_logits, batch)
    full_margin = semantic_margin(full_attention_logits, batch)
    all_prediction = prediction_colors(all_head_logits)
    full_prediction = prediction_colors(full_attention_logits)
    rows = []
    for index, item in enumerate(batch):
        rows.append(
            {
                "schema_version": "phase1001_head_confirmation_instrument.v1",
                "phase": 1001,
                "model": MODEL,
                "partition": item["partition"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "all_head_restore_margin": float(all_margin[index]),
                "full_attention_restore_margin": float(full_margin[index]),
                "absolute_margin_error": float(
                    abs(all_margin[index] - full_margin[index])
                ),
                "all_head_prediction": all_prediction[index],
                "full_attention_prediction": full_prediction[index],
                "prediction_agreement": (
                    all_prediction[index] == full_prediction[index]
                ),
            }
        )
    return rows


def mediation_rows_for_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    do_logits: torch.Tensor,
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [row["target"] for row in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    rows = []
    for event in ranked:
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=[
                {
                    "event": event,
                    "vectors": target_heads[layer_number][
                        :, head_index, :
                    ],
                }
            ],
        )
        restored_margin = semantic_margin(restored_logits, batch)
        restored_predictions = prediction_colors(restored_logits)
        for index, item in enumerate(batch):
            rows.append(
                mediation_row(
                    item,
                    event,
                    float(source_margin[index]),
                    float(target_margin[index]),
                    float(do_margin[index]),
                    float(restored_margin[index]),
                    restored_predictions[index],
                )
            )
        del restored_logits
    return rows


def event_passes(
    event_id: str,
    mediation: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> bool:
    med = mediation[event_id]
    control = controls[event_id]
    return bool(
        med["median_mediation_fraction"]
        >= HEAD_THRESHOLDS["single_median_mediation"]
        and control["mean_sufficiency_transfer"]
        >= HEAD_THRESHOLDS["single_mean_sufficiency_transfer"]
        and (
            control["mean_sufficiency_transfer"]
            - control["mean_wrong_o_transfer"]
        )
        >= HEAD_THRESHOLDS["single_wrong_o_excess"]
    )


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 head confirmation requires CUDA")

    (
        protocol,
        discovery_summary,
        frozen,
        selected_pairs,
        directional,
        validation_worlds,
        confirmation_worlds,
    ) = load_frozen_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT_ROOT / "selected_pairs.jsonl", selected_pairs)

    if len(selected_pairs) != EXPECTED_CONFIRMATION_PAIRS:
        raise RuntimeError(
            f"confirmation pair drift: {len(selected_pairs)}"
        )
    if validation_worlds & confirmation_worlds:
        raise RuntimeError("validation/confirmation world leakage")

    ranked_lookup = {
        item["event_id"]: item
        for item in discovery_summary["ranked_heads"]
    }
    ranked_ids = list(frozen["ranked_head_event_ids"])
    ranked = [ranked_lookup[event_id] for event_id in ranked_ids]
    frozen_joint_size = int(frozen["frozen_joint_size"])
    if (
        list(frozen["frozen_joint_event_ids"])
        != ranked_ids[:frozen_joint_size]
    ):
        raise RuntimeError("frozen joint event drift")

    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
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
        if (
            model.config.num_attention_heads != HEAD_COUNT
            or model.config.head_dim != HEAD_DIM
        ):
            raise RuntimeError("Qwen3 head geometry drift")

        instrument_rows = []
        mediation_rows = []
        control_rows = []
        joint_rows = []
        natural_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [row["source"] for row in batch]
            target_cases = [row["target"] for row in batch]
            _, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_logits, _, _ = capture_attention_states(
                model,
                layers,
                device,
                source_cases,
                candidate_ids,
            )
            target_logits, target_heads, target_attention = (
                capture_attention_states(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                )
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

            instrument_rows.extend(
                instrument_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    candidate_ids,
                    source_patch,
                    target_heads,
                    target_attention,
                )
            )
            mediation_rows.extend(
                mediation_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    ranked,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                )
            )
            control_rows.extend(
                selected_head_controls(
                    model,
                    layers,
                    device,
                    batch,
                    ranked,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            joint_rows.extend(
                joint_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    ranked,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            natural_rows.extend(
                natural_joint_rows(
                    model,
                    layers,
                    tokenizer,
                    device,
                    batch,
                    ranked,
                    frozen_joint_size,
                    candidate_ids,
                    source_patch,
                    target_heads,
                    target_attention,
                    effective_eos,
                    natural_budget,
                )
            )
            del (
                source_residuals,
                source_logits,
                target_logits,
                target_heads,
                target_attention,
                do_logits,
                do_heads,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[head-confirmation] {batch_number}/{len(batches)} batches",
                    flush=True,
                )

        write_rows(OUTPUT_ROOT / "instrument_rows.jsonl", instrument_rows)
        write_rows(OUTPUT_ROOT / "mediation_rows.jsonl", mediation_rows)
        write_rows(OUTPUT_ROOT / "control_rows.jsonl", control_rows)
        write_rows(OUTPUT_ROOT / "joint_rows.jsonl", joint_rows)
        write_rows(OUTPUT_ROOT / "natural_rows.jsonl", natural_rows)

        instrument_metrics = {
            "n": len(instrument_rows),
            "mean_absolute_margin_error": float(
                np.mean(
                    [row["absolute_margin_error"] for row in instrument_rows]
                )
            ),
            "max_absolute_margin_error": float(
                np.max(
                    [row["absolute_margin_error"] for row in instrument_rows]
                )
            ),
            "prediction_agreement": float(
                np.mean(
                    [row["prediction_agreement"] for row in instrument_rows]
                )
            ),
        }
        mediation_summary = summarize_mediation(mediation_rows)
        control_summary = summarize_controls(control_rows)
        joint_summary = summarize_joint(joint_rows)
        natural_summary = summarize_natural(natural_rows)
        discovery_single_pass = list(
            discovery_summary["single_head_pass_events"]
        )
        confirmation_single_pass = [
            event_id
            for event_id in ranked_ids
            if event_passes(event_id, mediation_summary, control_summary)
        ]
        replicated_discovery_single = [
            event_id
            for event_id in discovery_single_pass
            if event_id in confirmation_single_pass
        ]
        frozen_joint = joint_summary[str(frozen_joint_size)]
        source_do = natural_summary["source_do"]
        frozen_natural = natural_summary[
            "source_plus_frozen_head_restore"
        ]

        gate_checks = {
            "holdout_disjoint": not bool(
                validation_worlds & confirmation_worlds
            ),
            "frozen_contract": (
                frozen["selection_partition"] == "validation"
                and not frozen["selection_uses_holdout"]
            ),
            "instrument_margin": (
                instrument_metrics["mean_absolute_margin_error"]
                <= HEAD_THRESHOLDS[
                    "instrument_mean_abs_margin_error"
                ]
            ),
            "instrument_prediction": (
                instrument_metrics["prediction_agreement"]
                >= HEAD_THRESHOLDS["instrument_prediction_agreement"]
            ),
            "source_do_natural": source_do["flip_rate"] >= 0.90,
            "single_head_replication": bool(
                replicated_discovery_single
            ),
            "frozen_joint_mediation": (
                frozen_joint["median_mediation_fraction"]
                >= HEAD_THRESHOLDS["joint_median_mediation"]
            ),
            "frozen_joint_sufficiency": (
                frozen_joint["mean_sufficiency_transfer"]
                >= HEAD_THRESHOLDS["joint_median_mediation"]
            ),
            "frozen_joint_natural": (
                frozen_natural["target_rate"]
                >= HEAD_THRESHOLDS[
                    "joint_natural_restoration_rate"
                ]
            ),
        }
        secondary_checks = {
            "all_discovery_single_heads_replicated": (
                set(discovery_single_pass)
                <= set(confirmation_single_pass)
            ),
            "frozen_joint_exact_target_restoration": (
                frozen_natural["target_rate"] >= 0.95
            ),
            "all_attention_exact_target_restoration": (
                natural_summary[
                    "source_plus_all_attention_restore"
                ]["target_rate"]
                >= 0.95
            ),
        }
        summary = {
            "schema_version": "phase1001_head_confirmation_summary.v1",
            "phase": 1001,
            "model": MODEL,
            "partition": "confirmation",
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "validation_world_count": len(validation_worlds),
            "confirmation_world_count": len(confirmation_worlds),
            "world_overlap_count": len(
                validation_worlds & confirmation_worlds
            ),
            "source_depth": SOURCE_DEPTH,
            "target_layers": list(TARGET_LAYERS),
            "ranked_head_event_ids": ranked_ids,
            "frozen_joint_size": frozen_joint_size,
            "frozen_joint_event_ids": ranked_ids[:frozen_joint_size],
            "instrument_metrics": instrument_metrics,
            "mediation_summary": mediation_summary,
            "control_summary": control_summary,
            "joint_summary": joint_summary,
            "natural_summary": natural_summary,
            "discovery_single_head_pass_events": discovery_single_pass,
            "confirmation_single_head_pass_events": (
                confirmation_single_pass
            ),
            "replicated_discovery_single_head_events": (
                replicated_discovery_single
            ),
            "thresholds": {
                **HEAD_THRESHOLDS,
                "source_do_natural_flip_rate": 0.90,
            },
            "gate_checks": gate_checks,
            "secondary_checks": secondary_checks,
            "head_confirmation_gate_pass": all(
                gate_checks.values()
            ),
            "source_token_decomposition_open": all(
                gate_checks.values()
            ),
            "selection_uses_confirmation": False,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(OUTPUT_ROOT / "mediation_summary.json", mediation_summary)
        write_json(OUTPUT_ROOT / "control_summary.json", control_summary)
        write_json(OUTPUT_ROOT / "joint_summary.json", joint_summary)
        write_json(OUTPUT_ROOT / "natural_summary.json", natural_summary)
        write_json(OUTPUT_ROOT / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["head_confirmation_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "secondary_checks": summary["secondary_checks"],
                "replicated_single_heads": summary[
                    "replicated_discovery_single_head_events"
                ],
                "frozen_joint": summary["joint_summary"][
                    str(summary["frozen_joint_size"])
                ],
                "natural": summary["natural_summary"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
