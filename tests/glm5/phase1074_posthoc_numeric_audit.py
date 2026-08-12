#!/usr/bin/env python3
"""Diagnose GLM4 FP16 nonfinite Phase1074 behavior without changing gates."""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1074_polarity_dynamics_protocol as protocol


MODEL = "glm4"
REPEATS = 3


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.full(
        (len(rows), width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(
        len(rows), dtype=torch.long, device=device
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
    return input_ids, attention_mask, lengths


def run() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    automatic_before = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{MODEL}.jsonl"
    )
    behavior_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "behavior"
        / MODEL
        / "candidate_behavior.jsonl"
    )
    nonfinite_indices = sorted(
        int(row["semantic_case_index"])
        for row in behavior_rows
        if row["nonfinite_candidate"]
    )
    if not nonfinite_indices:
        raise RuntimeError("no GLM4 nonfinite cases to audit")
    by_index = {
        int(row["semantic_case_index"]): row for row in cases
    }
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_pair[str(row["pair_id"])].append(row)

    original_batch_size = bridge.PAIR_BATCH_SIZE[MODEL]
    original_batches = {
        index: cases[
            (index // original_batch_size) * original_batch_size:
            (index // original_batch_size + 1)
            * original_batch_size
        ]
        for index in nonfinite_indices
    }
    started = time.time()
    model = tokenizer = None
    records = []
    try:
        model, tokenizer, device, placement = load_fp16(MODEL)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos token")

        mode_batches = {}
        for index in nonfinite_indices:
            row = by_index[index]
            pair_rows = sorted(
                by_pair[str(row["pair_id"])],
                key=lambda value: protocol.TASKS.index(
                    value["task"]
                ),
            )
            pair_width = max(
                len(value["input_ids"]) for value in pair_rows
            )
            mode_batches[(index, "original_batch")] = (
                original_batches[index],
                max(
                    len(value["input_ids"])
                    for value in original_batches[index]
                ),
            )
            mode_batches[(index, "pair_batch")] = (
                pair_rows,
                pair_width,
            )
            mode_batches[(index, "singleton_own_width")] = (
                [row],
                len(row["input_ids"]),
            )
            mode_batches[(index, "singleton_pair_width")] = (
                [row],
                pair_width,
            )

        with torch.inference_mode():
            for (target_index, mode), (
                batch,
                width,
            ) in sorted(mode_batches.items()):
                target_slot = next(
                    index
                    for index, row in enumerate(batch)
                    if int(row["semantic_case_index"])
                    == target_index
                )
                target = by_index[target_index]
                for repeat in range(REPEATS):
                    input_ids, attention_mask, lengths = pad_rows(
                        batch, int(pad_id), device, width
                    )
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    position = int(lengths[target_slot].item()) - 1
                    logits = output.logits[
                        target_slot, position, :
                    ].float()
                    scores = {}
                    for class_name in ("b0", "b1"):
                        token_ids = torch.tensor(
                            target[
                                "candidate_first_token_ids"
                            ][class_name],
                            dtype=torch.long,
                            device=logits.device,
                        )
                        scores[class_name] = float(
                            logits[token_ids].max().item()
                        )
                    expected = str(target["expected_class"])
                    other = "b1" if expected == "b0" else "b0"
                    margin = scores[expected] - scores[other]
                    finite = all(
                        math.isfinite(value)
                        for value in scores.values()
                    ) and math.isfinite(margin)
                    records.append({
                        "schema_version": (
                            "phase1074_glm4_nonfinite_posthoc.v1"
                        ),
                        "phase": protocol.PHASE,
                        "model": MODEL,
                        "protocol_digest": prereg[
                            "protocol_digest"
                        ],
                        "semantic_case_index": target_index,
                        "record_id": target["record_id"],
                        "pair_id": target["pair_id"],
                        "relation": target["relation"],
                        "task": target["task"],
                        "path": target["path"],
                        "split": target["split"],
                        "mode": mode,
                        "repeat": repeat,
                        "batch_size": len(batch),
                        "tensor_width": width,
                        "finite": finite,
                        "candidate_margin": (
                            margin if math.isfinite(margin) else None
                        ),
                        "candidate_hit": bool(
                            finite and margin > 0.0
                        ),
                    })
                    del (
                        output,
                        logits,
                        input_ids,
                        attention_mask,
                        lengths,
                    )
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer

    totals = Counter()
    finite_counts = Counter()
    hit_counts = Counter()
    for row in records:
        mode = str(row["mode"])
        totals[mode] += 1
        finite_counts[mode] += int(row["finite"])
        hit_counts[mode] += int(row["candidate_hit"])
    automatic_after = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    if automatic_before != automatic_after:
        raise RuntimeError("posthoc audit changed automatic decision")
    summary = {
        "schema_version": (
            "phase1074_glm4_nonfinite_posthoc_summary.v1"
        ),
        "phase": protocol.PHASE,
        "model": MODEL,
        "protocol_digest": prereg["protocol_digest"],
        "nonfinite_case_count_in_formal_run": len(
            nonfinite_indices
        ),
        "repeats_per_mode": REPEATS,
        "modes": {
            mode: {
                "attempt_count": totals[mode],
                "finite_rate": (
                    finite_counts[mode] / totals[mode]
                ),
                "candidate_hit_rate": (
                    hit_counts[mode] / totals[mode]
                ),
            }
            for mode in sorted(totals)
        },
        "automatic_decision_unchanged": True,
        "precision": precision,
        "placement": placement,
        "elapsed_seconds": float(time.time() - started),
    }
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "glm4_nonfinite_posthoc.jsonl",
        records,
    )
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "glm4_nonfinite_posthoc_summary.json",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
