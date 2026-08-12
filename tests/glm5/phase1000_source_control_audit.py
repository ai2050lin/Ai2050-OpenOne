#!/usr/bin/env python3
"""Phase 1000 audit of paired-source specificity against valid derangements.

The original roll(1) scramble was not comparable across validation and
confirmation because validation contained adjacent bidirectional records.
This audit preserves the original failed result and adds fixed derangements
that always use a different counterfactual pair.
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

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import (
    COLORS,
    MODEL,
    OUT_ROOT,
    PHASE,
    write_json,
)
from phase1000_scpg_confirmation import one_direction_per_pair
from phase1000_scpg_discovery import (
    batches_by_template,
    capture_residuals,
    directional_pairs,
    forward_candidate,
    generate_with_interventions,
    intervention_rows,
    read_jsonl,
    source_patch_spec,
    summarize_interventions,
    write_rows,
)


NULL_COUNT = 8
NATURAL_NULL_COUNT = 4
THRESHOLDS = {
    "correct_mean_transfer": 0.90,
    "correct_flip_rate": 0.90,
    "transfer_excess_over_null": 0.30,
    "flip_excess_over_null": 0.30,
    "per_case_correct_above_null_mean_rate": 0.80,
    "joint_over_single_transfer_excess": 0.30,
    "joint_over_single_flip_excess": 0.30,
    "natural_flip_excess_over_null": 0.30,
    "max_reverse_or_noop_flip_rate": 0.10,
}


def valid_derangement_shifts(batch: list[dict[str, Any]], count: int) -> list[int]:
    valid = []
    size = len(batch)
    for shift in range(1, size):
        if all(
            batch[index]["pair_id"] != batch[(index - shift) % size]["pair_id"]
            for index in range(size)
        ):
            valid.append(shift)
    if len(valid) < count:
        raise RuntimeError(
            f"not enough pair-safe derangements: {len(valid)} < {count}"
        )
    if len(valid) == count:
        return valid
    positions = np.linspace(0, len(valid) - 1, count)
    chosen = []
    for position in positions:
        shift = valid[int(round(float(position)))]
        if shift not in chosen:
            chosen.append(shift)
    for shift in valid:
        if len(chosen) >= count:
            break
        if shift not in chosen:
            chosen.append(shift)
    return chosen[:count]


def natural_rows_for(
    batch: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    condition: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(batch):
        output = generated[index]
        rows.append(
            {
                "schema_version": "phase1000_control_audit_natural_row.v1",
                "phase": PHASE,
                "model": MODEL,
                "partition": item["partition"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "condition": condition,
                "source_gold": item["source"]["gold"],
                "target_gold": item["target"]["gold"],
                "prediction": output["prediction"],
                "flipped_to_source": output["prediction"]
                == item["source"]["gold"],
                "remained_target": output["prediction"]
                == item["target"]["gold"],
                "eos_seen": output["eos_position"] is not None,
                "exact_short": output["exact_short"],
                "generated_text": output["text"],
            }
        )
    return rows


def run_partition(
    model,
    layers,
    tokenizer,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    effective_eos: list[int],
    batch_size: int,
    natural_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    natural_rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        target_logits, target_residuals = capture_residuals(
            model, device, target_cases, (source_depth,), candidate_ids
        )
        source_vectors = source_residuals[source_depth]
        target_vectors = target_residuals[source_depth]
        conditions = {
            "correct_joint": source_patch_spec(
                source_depth, target_cases, source_vectors, "joint"
            ),
            "single_slot0": source_patch_spec(
                source_depth, target_cases, source_vectors, "single_slot0"
            ),
            "single_slot1": source_patch_spec(
                source_depth, target_cases, source_vectors, "single_slot1"
            ),
            "reverse_entity": source_patch_spec(
                source_depth, target_cases, source_vectors, "reverse"
            ),
            "noop_target": source_patch_spec(
                source_depth, target_cases, target_vectors, "joint"
            ),
        }
        shifts = valid_derangement_shifts(batch, NULL_COUNT)
        for null_index, shift in enumerate(shifts):
            donor_vectors = {
                role: torch.roll(vector, shifts=shift, dims=0)
                for role, vector in source_vectors.items()
            }
            conditions[f"null_derangement_{null_index}"] = source_patch_spec(
                source_depth, target_cases, donor_vectors, "joint"
            )

        for condition, patch in conditions.items():
            patched = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=patch,
            )
            candidate_rows.extend(
                intervention_rows(
                    batch,
                    source_logits,
                    target_logits,
                    patched,
                    condition,
                    "phase1000_control_audit_candidate_row.v1",
                )
            )
            if (
                condition == "correct_joint"
                or condition in ("reverse_entity", "noop_target")
                or condition.startswith("null_derangement_")
                and int(condition.rsplit("_", 1)[1]) < NATURAL_NULL_COUNT
            ):
                generated = generate_with_interventions(
                    model,
                    layers,
                    tokenizer,
                    device,
                    target_cases,
                    patch,
                    None,
                    effective_eos,
                    natural_budget,
                )
                natural_rows.extend(natural_rows_for(batch, generated, condition))
        del source_logits, target_logits, source_residuals, target_residuals
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(
                f"[control-audit/{directional[0]['partition']}] "
                f"{batch_number}/{len(batches)} batches",
                flush=True,
            )
    return candidate_rows, natural_rows


def summarize_natural(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "flip_rate": float(np.mean([row["flipped_to_source"] for row in values])),
            "target_rate": float(np.mean([row["remained_target"] for row in values])),
            "eos_rate": float(np.mean([row["eos_seen"] for row in values])),
            "exact_short_rate": float(np.mean([row["exact_short"] for row in values])),
        }
        for condition, values in sorted(groups.items())
    }


def specificity_summary(
    candidate_rows: list[dict[str, Any]],
    natural_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate = summarize_interventions(candidate_rows)
    natural = summarize_natural(natural_rows)
    null_conditions = [
        condition for condition in candidate if condition.startswith("null_derangement_")
    ]
    natural_null_conditions = [
        condition for condition in natural if condition.startswith("null_derangement_")
    ]
    null_rows = [
        row
        for row in candidate_rows
        if row["condition"] in null_conditions
    ]
    natural_null_rows = [
        row
        for row in natural_rows
        if row["condition"] in natural_null_conditions
    ]
    correct = candidate["correct_joint"]
    max_single_transfer = max(
        candidate["single_slot0"]["mean_transfer"],
        candidate["single_slot1"]["mean_transfer"],
    )
    max_single_flip = max(
        candidate["single_slot0"]["flip_rate"],
        candidate["single_slot1"]["flip_rate"],
    )
    pooled_null_transfer = float(
        np.mean([row["normalized_transfer"] for row in null_rows])
    )
    pooled_null_flip = float(
        np.mean([row["flipped_to_source"] for row in null_rows])
    )
    pooled_natural_null_flip = float(
        np.mean([row["flipped_to_source"] for row in natural_null_rows])
    )
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for row in candidate_rows:
        grouped[(row["pair_id"], row["direction"])][row["condition"]] = row
    above_null_mean = []
    above_all_null = []
    for values in grouped.values():
        null_values = [
            values[condition]["normalized_transfer"]
            for condition in null_conditions
        ]
        correct_value = values["correct_joint"]["normalized_transfer"]
        above_null_mean.append(correct_value > float(np.mean(null_values)))
        above_all_null.append(correct_value > max(null_values))
    metrics = {
        "candidate_condition_summary": candidate,
        "natural_condition_summary": natural,
        "null_count": len(null_conditions),
        "natural_null_count": len(natural_null_conditions),
        "pooled_null_mean_transfer": pooled_null_transfer,
        "pooled_null_flip_rate": pooled_null_flip,
        "correct_transfer_excess_over_null": (
            correct["mean_transfer"] - pooled_null_transfer
        ),
        "correct_flip_excess_over_null": (
            correct["flip_rate"] - pooled_null_flip
        ),
        "correct_above_per_case_null_mean_rate": float(np.mean(above_null_mean)),
        "correct_above_all_per_case_nulls_rate": float(np.mean(above_all_null)),
        "joint_over_max_single_transfer_excess": (
            correct["mean_transfer"] - max_single_transfer
        ),
        "joint_over_max_single_flip_excess": (
            correct["flip_rate"] - max_single_flip
        ),
        "correct_natural_flip_rate": natural["correct_joint"]["flip_rate"],
        "pooled_natural_null_flip_rate": pooled_natural_null_flip,
        "correct_natural_flip_excess_over_null": (
            natural["correct_joint"]["flip_rate"] - pooled_natural_null_flip
        ),
        "max_reverse_or_noop_candidate_flip_rate": max(
            candidate["reverse_entity"]["flip_rate"],
            candidate["noop_target"]["flip_rate"],
        ),
        "max_reverse_or_noop_natural_flip_rate": max(
            natural["reverse_entity"]["flip_rate"],
            natural["noop_target"]["flip_rate"],
        ),
    }
    checks = {
        "correct_mean_transfer": correct["mean_transfer"]
        >= THRESHOLDS["correct_mean_transfer"],
        "correct_flip_rate": correct["flip_rate"]
        >= THRESHOLDS["correct_flip_rate"],
        "transfer_excess_over_null": metrics[
            "correct_transfer_excess_over_null"
        ]
        >= THRESHOLDS["transfer_excess_over_null"],
        "flip_excess_over_null": metrics["correct_flip_excess_over_null"]
        >= THRESHOLDS["flip_excess_over_null"],
        "per_case_correct_above_null_mean_rate": metrics[
            "correct_above_per_case_null_mean_rate"
        ]
        >= THRESHOLDS["per_case_correct_above_null_mean_rate"],
        "joint_over_single_transfer_excess": metrics[
            "joint_over_max_single_transfer_excess"
        ]
        >= THRESHOLDS["joint_over_single_transfer_excess"],
        "joint_over_single_flip_excess": metrics[
            "joint_over_max_single_flip_excess"
        ]
        >= THRESHOLDS["joint_over_single_flip_excess"],
        "natural_flip_excess_over_null": metrics[
            "correct_natural_flip_excess_over_null"
        ]
        >= THRESHOLDS["natural_flip_excess_over_null"],
        "reverse_noop_candidate": metrics[
            "max_reverse_or_noop_candidate_flip_rate"
        ]
        <= THRESHOLDS["max_reverse_or_noop_flip_rate"],
        "reverse_noop_natural": metrics[
            "max_reverse_or_noop_natural_flip_rate"
        ]
        <= THRESHOLDS["max_reverse_or_noop_flip_rate"],
    }
    return metrics, checks


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1000 control audit requires CUDA")
    protocol_root = OUT_ROOT / "protocol"
    discovery_root = OUT_ROOT / "discovery"
    confirmation_root = OUT_ROOT / "confirmation"
    output_root = OUT_ROOT / "control_audit"
    output_root.mkdir(parents=True, exist_ok=True)

    cases = read_jsonl(protocol_root / "cases.jsonl")
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    discovery_pairs = read_jsonl(discovery_root / "selected_pairs.jsonl")
    confirmation_pairs = read_jsonl(confirmation_root / "selected_pairs.jsonl")
    frozen = json.loads(
        (discovery_root / "frozen_spec.json").read_text(encoding="utf-8")
    )
    case_by_id = {row["record_id"]: row for row in cases}
    validation_directional = directional_pairs(
        discovery_pairs,
        case_by_id,
        "validation",
        bidirectional=True,
    )
    confirmation_directional = one_direction_per_pair(
        confirmation_pairs, case_by_id
    )
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color]) for color in COLORS
    }
    source_depth = int(frozen["source_depth"])

    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        partition_results = {}
        all_checks = {}
        for partition, directional in (
            ("validation", validation_directional),
            ("confirmation", confirmation_directional),
        ):
            candidate_rows, natural_rows = run_partition(
                model,
                layers,
                tokenizer,
                device,
                directional,
                candidate_ids,
                source_depth,
                effective_eos,
                batch_size,
                natural_budget,
            )
            metrics, checks = specificity_summary(candidate_rows, natural_rows)
            write_rows(
                output_root / f"{partition}_candidate_rows.jsonl",
                candidate_rows,
            )
            write_rows(
                output_root / f"{partition}_natural_rows.jsonl",
                natural_rows,
            )
            partition_results[partition] = {
                "direction_count": len(directional),
                "metrics": metrics,
                "checks": checks,
                "gate_pass": all(checks.values()),
            }
            all_checks.update(
                {
                    f"{partition}/{key}": value
                    for key, value in checks.items()
                }
            )
        summary = {
            "schema_version": "phase1000_source_control_audit.v1",
            "phase": PHASE,
            "model": MODEL,
            "reason": (
                "Original roll(1) scramble mixed same-pair reverse donors in "
                "validation but cross-pair donors in confirmation."
            ),
            "original_confirmation_gate_preserved_as_failed": True,
            "null_definition": (
                "Eight fixed batch-local derangements; every donor has a "
                "different counterfactual pair_id."
            ),
            "null_count": NULL_COUNT,
            "natural_null_count": NATURAL_NULL_COUNT,
            "thresholds_predeclared_in_script": THRESHOLDS,
            "source_depth_frozen": source_depth,
            "partitions": partition_results,
            "gate_checks": all_checks,
            "corrected_specificity_gate_pass": all(all_checks.values()),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["corrected_specificity_gate_pass"],
                "validation": summary["partitions"]["validation"],
                "confirmation": summary["partitions"]["confirmation"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
