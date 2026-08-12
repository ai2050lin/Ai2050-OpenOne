#!/usr/bin/env python3
"""Run Phase1113 forced-choice behavior for one local FP16 model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1113_wordnet_semantic_quadrant_protocol as protocol


BATCH_SIZE = {"qwen3": 16, "glm4": 64, "deepseek7b": 32}


def run(model_name: str) -> None:
    preregistration = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1113 protocol audit failed")
    rows = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    ))
    if protocol.digest(rows) != preregistration["case_digests"][model_name]:
        raise RuntimeError("Phase1113 case digest mismatch")

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
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        totals: Counter[tuple[str, ...]] = Counter()
        finite_counts: Counter[tuple[str, ...]] = Counter()
        hits: Counter[tuple[str, ...]] = Counter()
        direct_candidates: Counter[tuple[str, ...]] = Counter()
        direct_hits: Counter[tuple[str, ...]] = Counter()
        margins: dict[tuple[str, ...], list[float]] = defaultdict(list)
        detail: list[dict[str, Any]] = []
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), BATCH_SIZE[model_name]):
                    batch = panel[start:start + BATCH_SIZE[model_name]]
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
                    top_ids = torch.argmax(logits, dim=-1)
                    for slot, row in enumerate(batch):
                        candidate_ids = {
                            key: int(values[0])
                            for key, values in row["candidate_first_token_ids"].items()
                        }
                        scores = {
                            key: float(logits[slot, token_id].item())
                            for key, token_id in candidate_ids.items()
                        }
                        expected = str(row["expected_class"])
                        other = "b" if expected == "a" else "a"
                        margin = scores[expected] - scores[other]
                        finite = (
                            all(math.isfinite(value) for value in scores.values())
                            and math.isfinite(margin)
                        )
                        hit = finite and margin > 0.0
                        top_token_id = int(top_ids[slot].item())
                        top_class = next(
                            (
                                key for key, token_id in candidate_ids.items()
                                if token_id == top_token_id
                            ),
                            None,
                        )
                        direct_candidate = top_class is not None
                        direct_hit = direct_candidate and top_class == expected
                        key = (
                            str(row["split"]),
                            str(row["template"]),
                            str(row["quadrant"]),
                            str(row["answer_order"]),
                        )
                        totals[key] += 1
                        finite_counts[key] += int(finite)
                        hits[key] += int(hit)
                        direct_candidates[key] += int(direct_candidate)
                        direct_hits[key] += int(direct_hit)
                        if finite:
                            margins[key].append(margin)
                        detail.append({
                            "case_index": int(row["case_index"]),
                            "record_id": row["record_id"],
                            "concept_id": row["concept_id"],
                            "split": row["split"],
                            "template": int(row["template"]),
                            "quadrant": row["quadrant"],
                            "surface_same": bool(row["surface_same"]),
                            "semantic_same": bool(row["semantic_same"]),
                            "answer_order": int(row["answer_order"]),
                            "base": row["base"],
                            "right_term": row["right_term"],
                            "expected_class": expected,
                            "scores": scores if finite else None,
                            "margin": margin if finite else None,
                            "finite": finite,
                            "hit": hit,
                            "top_token_id": top_token_id,
                            "top_token_text": tokenizer.decode([top_token_id]),
                            "top_class": top_class,
                            "direct_candidate": direct_candidate,
                            "direct_hit": direct_hit,
                        })
                    del output, logits, top_ids, input_ids, attention_mask
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "candidate_length_complete": length,
                }), flush=True)
        per_cell: dict[str, Any] = {}
        for key in sorted(totals):
            values = sorted(margins[key])
            per_cell["|".join(key)] = {
                "candidate_count": totals[key],
                "candidate_finite_fraction": finite_counts[key] / totals[key],
                "candidate_accuracy": hits[key] / max(finite_counts[key], 1),
                "direct_candidate_output_rate": direct_candidates[key] / totals[key],
                "direct_exact_accuracy": direct_hits[key] / totals[key],
                "median_expected_margin": values[len(values) // 2] if values else None,
            }
        total = sum(totals.values())
        finite_total = sum(finite_counts.values())
        summary = {
            "schema_version": "phase1113_wordnet_semantic_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": preregistration["protocol_digest"],
            "case_digest": preregistration["case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_count": total,
            "candidate_finite_fraction": finite_total / max(total, 1),
            "candidate_accuracy": sum(hits.values()) / max(finite_total, 1),
            "direct_candidate_output_rate": sum(direct_candidates.values()) / max(total, 1),
            "direct_exact_accuracy": sum(direct_hits.values()) / max(total, 1),
            "per_cell": per_cell,
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = protocol.digest(summary)
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", detail)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "candidate_count": total,
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "candidate_accuracy": summary["candidate_accuracy"],
            "direct_candidate_output_rate": summary["direct_candidate_output_rate"],
            "direct_exact_accuracy": summary["direct_exact_accuracy"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False, indent=2))
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
