#!/usr/bin/env python3
"""Select and confirm a natural full-vocabulary output protocol."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1051_natural_behavior_protocol as protocol


BATCH_SIZE = {"qwen3": 24, "glm4": 8, "deepseek7b": 8}


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def evaluate_cases(
    model,
    device: torch.device,
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for batch_rows in chunks(rows, batch_size):
        lengths = [len(row["input_ids"]) for row in batch_rows]
        width = max(lengths)
        input_ids = torch.full(
            (len(batch_rows), width),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for slot, row in enumerate(batch_rows):
            values = torch.tensor(
                row["input_ids"], dtype=torch.long, device=device
            )
            input_ids[slot, :len(values)] = values
            attention_mask[slot, :len(values)] = 1
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = result.logits.float()
        batch = torch.arange(len(batch_rows), device=logits.device)
        positions = torch.tensor(
            [value - 1 for value in lengths],
            dtype=torch.long,
            device=logits.device,
        )
        boundary = logits[batch, positions, :]
        finite = torch.isfinite(boundary).all(dim=-1)
        safe = torch.where(
            torch.isfinite(boundary),
            boundary,
            torch.full_like(boundary, -torch.inf),
        )
        top2_values, top2_ids = torch.topk(safe, k=2, dim=-1)
        for slot, row in enumerate(batch_rows):
            expected = int(row["expected_first_token_id"])
            expected_value = float(safe[slot, expected].item())
            top1 = int(top2_ids[slot, 0].item())
            competitor = (
                float(top2_values[slot, 1].item())
                if top1 == expected
                else float(top2_values[slot, 0].item())
            )
            output[int(row["semantic_case_index"])] = {
                "finite": bool(finite[slot].item()),
                "top1": top1,
                "expected": expected,
                "exact": bool(finite[slot].item()) and top1 == expected,
                "expected_margin": (
                    expected_value - competitor
                    if math.isfinite(expected_value)
                    and math.isfinite(competitor)
                    else None
                ),
            }
        del result, logits, boundary, safe, top2_values, top2_ids
    return output


def metrics(
    targets: list[dict[str, Any]],
    values: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    arms = []
    pairs = []
    by_family: dict[str, list[bool]] = defaultdict(list)
    margins = []
    for target in targets:
        left = values[int(target["target_case_index"])]
        right = values[int(target["cross_case_index"])]
        arms.extend((left, right))
        pairs.append(bool(left["exact"] and right["exact"]))
        by_family[str(target["target_expected_label"])].append(
            bool(left["exact"])
        )
        by_family[str(target["cross_expected_label"])].append(
            bool(right["exact"])
        )
        for arm in (left, right):
            if arm["expected_margin"] is not None:
                margins.append(float(arm["expected_margin"]))
    finite_count = sum(bool(row["finite"]) for row in arms)
    exact_count = sum(bool(row["exact"]) for row in arms)
    family_accuracy = {
        family: (
            sum(values_) / len(values_) if values_ else 0.0
        )
        for family, values_ in sorted(by_family.items())
    }
    return {
        "pair_count": len(pairs),
        "arm_count": len(arms),
        "finite_count": finite_count,
        "finite_rate": finite_count / len(arms) if arms else 0.0,
        "exact_arm_count": exact_count,
        "arm_accuracy": exact_count / len(arms) if arms else 0.0,
        "correct_pair_count": sum(pairs),
        "pair_accuracy": sum(pairs) / len(pairs) if pairs else 0.0,
        "family_accuracy": family_accuracy,
        "minimum_family_accuracy": (
            min(family_accuracy.values()) if family_accuracy else 0.0
        ),
        "expected_margin_median": (
            float(np.median(margins)) if margins else None
        ),
        "expected_margin_mean": (
            float(np.mean(margins)) if margins else None
        ),
        "correct_pair_mask": pairs,
    }


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics_ = row["metrics"]
    margin = metrics_["expected_margin_mean"]
    return (
        float(metrics_["pair_accuracy"]),
        float(metrics_["minimum_family_accuracy"]),
        float(metrics_["arm_accuracy"]),
        float(margin if margin is not None else -1e30),
        -float(protocol.SELECTION_ORDER.index(row["variant"])),
    )


def rollout_cases(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[len(row["input_ids"])].append(row)
    records = []
    for length in sorted(by_length):
        for batch_rows in chunks(by_length[length], batch_size):
            input_ids = torch.tensor(
                [row["input_ids"] for row in batch_rows],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            generated = [[] for _ in batch_rows]
            for _ in range(steps):
                with torch.inference_mode():
                    result = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                next_token = torch.argmax(
                    result.logits[:, -1, :].float(), dim=-1
                )
                for slot, token in enumerate(
                    next_token.detach().cpu().tolist()
                ):
                    generated[slot].append(int(token))
                input_ids = torch.cat(
                    (input_ids, next_token[:, None]), dim=1
                )
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (len(batch_rows), 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                del result
            for row, tokens in zip(batch_rows, generated):
                expected = [
                    int(value) for value in row["expected_token_ids"]
                ]
                text = tokenizer.decode(
                    tokens, skip_special_tokens=False
                )
                match = re.search(r"[A-Za-z]+", text)
                first_word = (
                    match.group(0).casefold() if match else ""
                )
                records.append({
                    "semantic_case_index": int(
                        row["semantic_case_index"]
                    ),
                    "expected_label": str(row["expected_label"]),
                    "generated_token_ids": tokens,
                    "generated_text": text,
                    "expected_token_ids": expected,
                    "exact_label_token_prefix": (
                        tokens[:len(expected)] == expected
                    ),
                    "normalized_first_word_exact": (
                        first_word == str(row["expected_label"]).casefold()
                    ),
                })
    return records


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1051 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    by_partition = {
        partition: [
            row for row in targets if row["partition"] == partition
        ]
        for partition in protocol.PARTITION_UNIT_COUNTS_PER_FAMILY
    }
    all_cases = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        (row["variant"], int(row["semantic_case_index"])): row
        for row in all_cases
    }
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        discovery_rows = []
        discovery_targets = by_partition["discovery"]
        for variant in protocol.SELECTION_ORDER:
            needed = {
                int(target[key])
                for target in discovery_targets
                for key in ("target_case_index", "cross_case_index")
            }
            variant_cases = [
                cases[(variant, index)] for index in sorted(needed)
            ]
            values = evaluate_cases(
                model,
                device,
                variant_cases,
                pad_token_id=int(pad_token_id),
                batch_size=BATCH_SIZE[model_name],
            )
            result = metrics(discovery_targets, values)
            discovery_rows.append({
                "variant": variant,
                "metrics": result,
                "eligible": (
                    result["finite_rate"]
                    >= prereg["gates"]["discovery_finite_rate_min"]
                ),
            })
        eligible = [row for row in discovery_rows if row["eligible"]]
        if not eligible:
            raise RuntimeError("No finite discovery protocol variant")
        frozen = max(eligible, key=selection_key)["variant"]

        confirmation_targets = by_partition["confirmation"]
        confirmation_needed = {
            int(target[key])
            for target in confirmation_targets
            for key in ("target_case_index", "cross_case_index")
        }
        confirmation_cases = [
            cases[(frozen, index)]
            for index in sorted(confirmation_needed)
        ]
        confirmation_values = evaluate_cases(
            model,
            device,
            confirmation_cases,
            pad_token_id=int(pad_token_id),
            batch_size=BATCH_SIZE[model_name],
        )
        confirmation = metrics(
            confirmation_targets, confirmation_values
        )
        gates = prereg["gates"]
        behavior_passed = (
            confirmation["finite_rate"]
            >= gates["confirmation_finite_rate_min"]
            and confirmation["arm_accuracy"]
            >= gates["confirmation_arm_accuracy_min"]
            and confirmation["pair_accuracy"]
            >= gates["confirmation_pair_accuracy_min"]
            and confirmation["correct_pair_count"]
            >= gates["confirmation_correct_pair_count_min"]
            and confirmation["minimum_family_accuracy"]
            >= gates["confirmation_min_family_accuracy_min"]
        )

        correct_pair_targets = [
            target
            for target, passed in zip(
                confirmation_targets,
                confirmation["correct_pair_mask"],
            )
            if passed
        ][:int(prereg["rollout_pair_limit"])]
        rollout_rows = []
        for target in correct_pair_targets:
            rollout_rows.extend((
                cases[(frozen, int(target["target_case_index"]))],
                cases[(frozen, int(target["cross_case_index"]))],
            ))
        rollouts = rollout_cases(
            model,
            tokenizer,
            device,
            rollout_rows,
            steps=int(prereg["rollout_steps"]),
            batch_size=BATCH_SIZE[model_name],
        )
        rollout_summary = {
            "pair_count": len(correct_pair_targets),
            "arm_count": len(rollouts),
            "exact_label_token_prefix_rate": (
                sum(
                    row["exact_label_token_prefix"]
                    for row in rollouts
                ) / len(rollouts)
                if rollouts else 0.0
            ),
            "normalized_first_word_exact_rate": (
                sum(
                    row["normalized_first_word_exact"]
                    for row in rollouts
                ) / len(rollouts)
                if rollouts else 0.0
            ),
        }
        summary = {
            "schema_version": "phase1051_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "discovery": discovery_rows,
            "frozen_variant": frozen,
            "confirmation": confirmation,
            "behavior_gate_passed": behavior_passed,
            "clean_rollout_summary": rollout_summary,
            "clean_rollouts": rollouts,
            "elapsed_seconds": float(time.time() - started),
        }
        out = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_json(out / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "frozen_variant": frozen,
            "confirmation_arm_accuracy": confirmation["arm_accuracy"],
            "confirmation_pair_accuracy": confirmation["pair_accuracy"],
            "correct_pairs": confirmation["correct_pair_count"],
            "minimum_family_accuracy": (
                confirmation["minimum_family_accuracy"]
            ),
            "behavior_passed": behavior_passed,
            "rollout": rollout_summary,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
