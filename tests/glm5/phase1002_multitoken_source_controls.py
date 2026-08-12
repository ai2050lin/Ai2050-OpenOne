#!/usr/bin/env python3
"""Semantic and role controls for the Phase 1002 joint entity source."""
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

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, load_model, release_model
from phase1002_multitoken_frozen_topology import (
    directional_rows,
    read_json,
)
from phase1002_multitoken_protocol import (
    COLORS,
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)


PHASE = 1002
BATCH_SIZE = 8


def mismatch_batches(
    rows: list[dict[str, Any]],
    exclude_target: bool = False,
):
    templates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        templates[int(row["template"])].append(row)
    for template in sorted(templates):
        buckets = [
            sorted(
                [
                    row for row in templates[template]
                    if row["source"]["gold"] == color
                ],
                key=lambda row: (row["pair_id"], row["direction"]),
            )
            for color in COLORS
        ]
        ordered = [row for bucket in buckets for row in bucket]
        donor_ordered = None
        if exclude_target:
            donor_for_recipient: list[int | None] = [None] * len(ordered)
            recipient_for_donor: list[int | None] = [None] * len(ordered)

            def allowed(recipient_index: int, donor_index: int) -> bool:
                recipient = ordered[recipient_index]
                donor_gold = ordered[donor_index]["source"]["gold"]
                return donor_gold not in (
                    recipient["source"]["gold"],
                    recipient["target"]["gold"],
                )

            def assign(recipient_index: int, seen: set[int]) -> bool:
                donor_order = list(range(
                    recipient_index + 1, len(ordered)
                )) + list(range(recipient_index + 1))
                for donor_index in donor_order:
                    if (
                        donor_index in seen
                        or not allowed(recipient_index, donor_index)
                    ):
                        continue
                    seen.add(donor_index)
                    previous = recipient_for_donor[donor_index]
                    if previous is None or assign(previous, seen):
                        recipient_for_donor[donor_index] = recipient_index
                        donor_for_recipient[recipient_index] = donor_index
                        return True
                return False

            if all(
                assign(recipient_index, set())
                for recipient_index in range(len(ordered))
            ):
                donor_ordered = [
                    ordered[int(donor_index)]
                    for donor_index in donor_for_recipient
                ]
            else:
                usage = [0] * len(ordered)
                donor_ordered = []
                for recipient_index in range(len(ordered)):
                    candidates = [
                        donor_index
                        for donor_index in range(len(ordered))
                        if allowed(recipient_index, donor_index)
                    ]
                    if not candidates:
                        raise RuntimeError(
                            "no strict third-answer donor candidate"
                        )
                    donor_index = min(
                        candidates,
                        key=lambda value: (
                            usage[value],
                            (
                                value - recipient_index - 1
                            ) % len(ordered),
                        ),
                    )
                    usage[donor_index] += 1
                    donor_ordered.append(ordered[donor_index])
        else:
            for shift in range(1, len(ordered)):
                candidate = ordered[shift:] + ordered[:shift]
                if all(
                    recipient["source"]["gold"]
                    != donor["source"]["gold"]
                    for recipient, donor in zip(ordered, candidate)
                ):
                    donor_ordered = candidate
                    break
        if donor_ordered is None:
            raise RuntimeError(
                f"no different-answer donor permutation for t{template}"
            )
        if (
            not exclude_target
            and len({
                (row["pair_id"], row["direction"])
                for row in donor_ordered
            }) != len(donor_ordered)
        ):
            raise RuntimeError("mismatch donor permutation is not one-to-one")
        for start in range(0, len(ordered), BATCH_SIZE):
            yield (
                ordered[start:start + BATCH_SIZE],
                donor_ordered[start:start + BATCH_SIZE],
            )


def prediction_colors(logits: torch.Tensor) -> list[str]:
    indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [COLORS[int(index)] for index in indices]


def run_model(model_name: str) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(
        prereg["frozen_phase1001_topology"][model_name]["source_depth"]
    )
    model = tokenizer = None
    started = time.time()
    result_rows = []
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        for split in ("discovery", "confirmation"):
            rows = directional_rows(model_name, split)
            split_batches = list(mismatch_batches(rows))
            for batch_number, (batch, donor_rows) in enumerate(
                split_batches, 1
            ):
                source_cases = [row["source"] for row in batch]
                target_cases = [row["target"] for row in batch]
                donor_cases = [row["source"] for row in donor_rows]
                candidate_ids = target_cases[0]["candidate_token_ids"]
                source_logits, source_residuals = scpg.capture_residuals(
                    model,
                    device,
                    source_cases,
                    (source_depth,),
                    candidate_ids,
                )
                target_logits, target_residuals = scpg.capture_residuals(
                    model,
                    device,
                    target_cases,
                    (source_depth,),
                    candidate_ids,
                )
                _, donor_residuals = scpg.capture_residuals(
                    model,
                    device,
                    donor_cases,
                    (source_depth,),
                    candidate_ids,
                )
                source_vectors = source_residuals[source_depth]
                target_vectors = target_residuals[source_depth]
                mismatch_vectors = donor_residuals[source_depth]
                conditions = {
                    "joint_correct": scpg.source_patch_spec(
                        source_depth,
                        target_cases,
                        source_vectors,
                        "joint",
                    ),
                    "single_slot0": scpg.source_patch_spec(
                        source_depth,
                        target_cases,
                        source_vectors,
                        "single_slot0",
                    ),
                    "single_slot1": scpg.source_patch_spec(
                        source_depth,
                        target_cases,
                        source_vectors,
                        "single_slot1",
                    ),
                    "reverse_roles": scpg.source_patch_spec(
                        source_depth,
                        target_cases,
                        source_vectors,
                        "reverse",
                    ),
                    "semantic_mismatch": scpg.source_patch_spec(
                        source_depth,
                        target_cases,
                        mismatch_vectors,
                        "joint",
                    ),
                    "target_noop": scpg.source_patch_spec(
                        source_depth,
                        target_cases,
                        target_vectors,
                        "joint",
                    ),
                }
                source_margin = scpg.semantic_margin(source_logits, batch)
                target_margin = scpg.semantic_margin(target_logits, batch)
                target_predictions = prediction_colors(target_logits)
                for condition, patch in conditions.items():
                    patched_logits = scpg.forward_candidate(
                        model,
                        layers,
                        device,
                        target_cases,
                        candidate_ids,
                        source_patch=patch,
                    )
                    patched_margin = scpg.semantic_margin(
                        patched_logits, batch
                    )
                    patched_predictions = prediction_colors(patched_logits)
                    for index, item in enumerate(batch):
                        clean_span = float(
                            source_margin[index] - target_margin[index]
                        )
                        result_rows.append({
                            "schema_version": (
                                "phase1002_source_control_row.v1"
                            ),
                            "phase": PHASE,
                            "model": model_name,
                            "split": split,
                            "pair_id": item["pair_id"],
                            "direction": item["direction"],
                            "template": item["template"],
                            "condition": condition,
                            "source_gold": item["source"]["gold"],
                            "target_gold": item["target"]["gold"],
                            "mismatch_donor_gold": (
                                donor_rows[index]["source"]["gold"]
                                if condition == "semantic_mismatch"
                                else None
                            ),
                            "prediction": patched_predictions[index],
                            "source_prediction": (
                                patched_predictions[index]
                                == item["source"]["gold"]
                            ),
                            "target_prediction": (
                                patched_predictions[index]
                                == item["target"]["gold"]
                            ),
                            "mismatch_donor_prediction": (
                                patched_predictions[index]
                                == donor_rows[index]["source"]["gold"]
                                if condition == "semantic_mismatch"
                                else None
                            ),
                            "normalized_transfer": float(
                                (
                                    patched_margin[index]
                                    - target_margin[index]
                                )
                                / max(abs(clean_span), 1e-8)
                            ),
                            "target_noop_prediction_agreement": (
                                patched_predictions[index]
                                == target_predictions[index]
                                if condition == "target_noop"
                                else None
                            ),
                            "target_noop_max_abs_difference": (
                                float(torch.max(torch.abs(
                                    patched_logits[index]
                                    - target_logits[index]
                                )))
                                if condition == "target_noop"
                                else None
                            ),
                        })
                    del patched_logits
                del (
                    source_logits,
                    target_logits,
                    source_residuals,
                    target_residuals,
                    donor_residuals,
                )
                print(
                    f"[{model_name}/{split}] "
                    f"{batch_number}/{len(split_batches)}",
                    flush=True,
                )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    split_summary = {}
    for split in ("discovery", "confirmation"):
        split_summary[split] = {}
        for condition in (
            "joint_correct",
            "single_slot0",
            "single_slot1",
            "reverse_roles",
            "semantic_mismatch",
            "target_noop",
        ):
            values = [
                row for row in result_rows
                if row["split"] == split
                and row["condition"] == condition
            ]
            item = {
                "n": len(values),
                "source_rate": float(np.mean([
                    row["source_prediction"] for row in values
                ])),
                "target_rate": float(np.mean([
                    row["target_prediction"] for row in values
                ])),
                "median_normalized_transfer": float(np.median([
                    row["normalized_transfer"] for row in values
                ])),
                "mean_normalized_transfer": float(np.mean([
                    row["normalized_transfer"] for row in values
                ])),
            }
            if condition == "semantic_mismatch":
                item["mismatch_donor_rate"] = float(np.mean([
                    row["mismatch_donor_prediction"] for row in values
                ]))
            if condition == "target_noop":
                item["prediction_agreement"] = float(np.mean([
                    row["target_noop_prediction_agreement"]
                    for row in values
                ]))
                item["max_abs_difference"] = float(max(
                    row["target_noop_max_abs_difference"]
                    for row in values
                ))
            split_summary[split][condition] = item

    checks = {}
    for split in ("discovery", "confirmation"):
        values = split_summary[split]
        checks[split] = {
            "correct_source": (
                values["joint_correct"]["source_rate"] >= 0.80
            ),
            "semantic_specificity": (
                values["semantic_mismatch"]["source_rate"] <= 0.25
                and (
                    values["joint_correct"]["source_rate"]
                    - values["semantic_mismatch"]["source_rate"]
                ) >= 0.50
            ),
            "role_specificity": (
                values["reverse_roles"]["source_rate"] <= 0.25
            ),
            "noop": (
                values["target_noop"]["prediction_agreement"] >= 0.99
            ),
        }
    summary = {
        "schema_version": "phase1002_source_control_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "source_depth": source_depth,
        "direction_count_per_split": 256,
        "semantic_mismatch_has_no_same_gold_donors": True,
        "split_summary": split_summary,
        "checks": checks,
        "source_control_pass": all(
            all(values.values()) for values in checks.values()
        ),
        "thresholds": {
            "correct_source_min": 0.80,
            "mismatch_source_max": 0.25,
            "correct_minus_mismatch_min": 0.50,
            "reverse_source_max": 0.25,
            "noop_agreement_min": 0.99,
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "The mismatch donor is an in-distribution residual from another "
            "world with a different source answer. This reduces, but does "
            "not eliminate, off-manifold intervention risk."
        ),
    }
    model_root = OUT_ROOT / "source_controls" / model_name
    write_jsonl(model_root / "rows.jsonl", result_rows)
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT / "source_controls" / model_name / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT / "source_controls" / model_name / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": "phase1002_source_control_cross_model.v1",
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["source_control_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["source_control_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(OUT_ROOT / "source_controls" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
