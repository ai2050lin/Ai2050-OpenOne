#!/usr/bin/env python3
"""Run the frozen Phase1083 behavior gate without hidden-state access."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as generation
from phase1065_multimode_response_atlas_scan import strict_generated_answer
import phase1083_same_carrier_attribute_protocol as protocol


CANDIDATE_BATCH_SIZE = {"qwen3": 32, "glm4": 32, "deepseek7b": 32}


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"\s+", " ", value)


def label_position(text: str, label: str) -> int | None:
    normalized = normalized_text(text)
    target = normalized_text(label)
    if not target:
        return None
    match = re.search(rf"(?<![\w]){re.escape(target)}(?![\w])", normalized)
    return match.start() if match else None


def classify_generation(generated: str, target: str, distractor: str) -> dict[str, Any]:
    text = normalized_text(generated)
    target_norm = normalized_text(target)
    distractor_norm = normalized_text(distractor)
    target_at = label_position(text, target_norm)
    distractor_at = label_position(text, distractor_norm)
    semantic_first = target_at == 0
    return {
        "normalized_text": text,
        "semantic_first": semantic_first,
        "strict_label_only": text == target_norm,
        "target_before_distractor": target_at is not None and (
            distractor_at is None or target_at < distractor_at
        ),
        "target_position": target_at,
        "distractor_position": distractor_at,
    }


def generation_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    states = (
        "t0_cactive_m0_q0_w0",
        "t0_cactive_m0_q1_w1",
        "t1_cactive_m1_q0_w1",
        "t1_cactive_m1_q1_w0",
    )
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            unit_ids = sorted({
                row["unit_id"] for row in rows
                if row["family"] == family and row["split"] == split
            })[:protocol.GENERATION_UNITS_PER_FAMILY_SPLIT]
            for unit_id, state in zip(unit_ids, states):
                row = next(
                    row for row in rows
                    if row["unit_id"] == unit_id and row["state"] == state
                )
                selected.append({
                    **row,
                    "semantic_case_index": int(row["case_index"]),
                })
    return selected


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["panel"] == "active"]


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1083 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    started = time.time()
    model = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
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
            raise RuntimeError("tokenizer has no pad/eos id")

        candidate_detail: list[dict[str, Any]] = []
        totals: Counter = Counter()
        finite_counts: Counter = Counter()
        hits: Counter = Counter()
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows(rows):
            by_length[len(row["input_ids"])].append(row)

        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                batch_size = CANDIDATE_BATCH_SIZE[model_name]
                for start in range(0, len(panel), batch_size):
                    batch = panel[start:start + batch_size]
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    logits = output.logits[:, -1, :].float()
                    for slot, row in enumerate(batch):
                        values = logits[slot]
                        scores = {}
                        for answer_class in ("a0", "a1"):
                            ids = torch.tensor(
                                row["candidate_first_token_ids"][answer_class],
                                dtype=torch.long,
                                device=values.device,
                            )
                            scores[answer_class] = float(values[ids].max().item())
                        expected = str(row["expected_class"])
                        other = "a1" if expected == "a0" else "a0"
                        margin = scores[expected] - scores[other]
                        finite = all(math.isfinite(v) for v in scores.values()) \
                            and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        key = (row["operation"], row["world"], row["split"])
                        totals[key] += 1
                        finite_counts[key] += int(finite)
                        hits[key] += int(hit)
                        candidate_detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "operation": row["operation"],
                            "world": row["world"],
                            "split": row["split"],
                            "state": row["state"],
                            "expected_class": expected,
                            "target_answer": row["target_answer"],
                            "scores": {
                                key_: value if math.isfinite(value) else None
                                for key_, value in scores.items()
                            },
                            "margin": margin if math.isfinite(margin) else None,
                            "finite": finite,
                            "hit": hit,
                        })
                    del output, logits, input_ids, attention_mask
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "candidate_length_complete": length,
                }), flush=True)

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        generation_rows = generation_selection(rows)
        generated = generation.generate_case_outputs(
            model,
            device,
            generation_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["generation_steps"]),
        )
        generation_detail: list[dict[str, Any]] = []
        gen_totals: Counter = Counter()
        gen_before: Counter = Counter()
        gen_first: Counter = Counter()
        for row in generation_rows:
            output_ids = generated[int(row["case_index"])]
            generated_text = strict_generated_answer(tokenizer, output_ids, eos_ids)
            distractor = row["answer_labels"][1 - int(row["answer_index"])]
            classification = classify_generation(
                generated_text, row["target_answer"], distractor
            )
            key = (row["operation"], row["world"], row["split"])
            gen_totals[key] += 1
            gen_before[key] += int(classification["target_before_distractor"])
            gen_first[key] += int(classification["semantic_first"])
            generation_detail.append({
                "case_index": int(row["case_index"]),
                "unit_id": row["unit_id"],
                "operation": row["operation"],
                "world": row["world"],
                "split": row["split"],
                "state": row["state"],
                "target_answer": row["target_answer"],
                "distractor_answer": distractor,
                "generated_token_ids": [int(value) for value in output_ids],
                "generated_text": generated_text,
                **classification,
            })

        threshold_candidate = prereg["evidence_thresholds"][
            "candidate_accuracy_for_operation_behavior"
        ]
        threshold_generation = prereg["evidence_thresholds"][
            "generation_target_before_distractor_accuracy"
        ]
        per_cell = {}
        passing_worlds: dict[str, list[str]] = defaultdict(list)
        for operation in protocol.OPERATIONS:
            for world in protocol.WORLDS:
                candidate_total = sum(
                    totals[(operation, world, split)] for split in protocol.SPLITS
                )
                candidate_hit = sum(
                    hits[(operation, world, split)] for split in protocol.SPLITS
                )
                candidate_finite = sum(
                    finite_counts[(operation, world, split)]
                    for split in protocol.SPLITS
                )
                generation_total = sum(
                    gen_totals[(operation, world, split)] for split in protocol.SPLITS
                )
                generation_hit = sum(
                    gen_before[(operation, world, split)] for split in protocol.SPLITS
                )
                candidate_accuracy = candidate_hit / candidate_total
                generation_accuracy = generation_hit / generation_total
                passes = (
                    candidate_accuracy >= threshold_candidate
                    and generation_accuracy >= threshold_generation
                )
                if passes:
                    passing_worlds[operation].append(world)
                per_cell[f"{operation}__{world}"] = {
                    "candidate_count": candidate_total,
                    "candidate_finite_count": candidate_finite,
                    "candidate_accuracy": candidate_accuracy,
                    "generation_count": generation_total,
                    "generation_target_before_distractor_accuracy": generation_accuracy,
                    "passes": passes,
                }
        minimum_worlds = int(prereg["evidence_thresholds"][
            "minimum_behavior_worlds_per_operation"
        ])
        passing_operations = [
            operation for operation in protocol.OPERATIONS
            if len(passing_worlds[operation]) >= minimum_worlds
        ]
        result = {
            "schema_version": "phase1083_behavior_gate.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_case_count": len(candidate_detail),
            "generation_case_count": len(generation_detail),
            "candidate_finite_fraction": (
                sum(finite_counts.values()) / sum(totals.values())
            ),
            "passing_worlds_by_operation": dict(passing_worlds),
            "passing_operations": passing_operations,
            "passing_operation_count": len(passing_operations),
            "model_behavior_gate_passed": len(passing_operations) >= int(
                prereg["evidence_thresholds"]["minimum_behavior_operations"]
            ),
            "per_cell": per_cell,
            "elapsed_seconds": time.time() - started,
        }
        result["result_digest"] = protocol.digest(result)
        pilot_root = protocol.OUT_ROOT / "pilot"
        protocol.write_jsonl(
            pilot_root / f"candidate.{model_name}.jsonl", candidate_detail
        )
        protocol.write_jsonl(
            pilot_root / f"generation.{model_name}.jsonl", generation_detail
        )
        protocol.write_json(pilot_root / f"{model_name}.json", result)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "status": "behavior_complete",
            "passing_operations": passing_operations,
            "candidate_finite_fraction": result["candidate_finite_fraction"],
            "elapsed_seconds": result["elapsed_seconds"],
            "result_digest": result["result_digest"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
