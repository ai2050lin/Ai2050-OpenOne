#!/usr/bin/env python3
"""Run Phase1112 candidate-logit behavior for one local FP16 model."""

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
import phase1112_one_shot_body_reader_protocol as protocol


BATCH_SIZE = {"qwen3": 16, "glm4": 64, "deepseek7b": 32}


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1112 protocol audit failed")
    rows = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    ))
    if protocol.digest(rows) != prereg["case_digests"][model_name]:
        raise RuntimeError("Phase1112 case digest mismatch")
    started = time.time()
    model = None
    try:
        model, _tokenizer, device, placement = load_fp16(model_name)
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
        totals: Counter = Counter()
        finite_counts: Counter = Counter()
        hits: Counter = Counter()
        margins: dict[tuple[str, ...], list[float]] = defaultdict(list)
        detail = []
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
                    for slot, row in enumerate(batch):
                        scores = {
                            answer_class: float(logits[slot, ids[0]].item())
                            for answer_class, ids in row["candidate_first_token_ids"].items()
                        }
                        expected = str(row["expected_class"])
                        other = "e1" if expected == "e0" else "e0"
                        margin = scores[expected] - scores[other]
                        finite = (
                            all(math.isfinite(value) for value in scores.values())
                            and math.isfinite(margin)
                        )
                        hit = finite and margin > 0.0
                        key = (
                            str(row["relation_pair"]),
                            str(row["surface"]),
                            str(row["split"]),
                            str(row["label_regime"]),
                            str(row["route_type"]),
                            str(row["congruence"]),
                        )
                        totals[key] += 1
                        finite_counts[key] += int(finite)
                        hits[key] += int(hit)
                        if finite:
                            margins[key].append(margin)
                        detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "relation_pair": row["relation_pair"],
                            "surface": row["surface"],
                            "split": row["split"],
                            "label_regime": row["label_regime"],
                            "route_type": row["route_type"],
                            "congruence": row["congruence"],
                            "state": row["state"],
                            "expected_class": expected,
                            "margin": margin if finite else None,
                            "finite": finite,
                            "hit": hit,
                        })
                    del output, logits, input_ids, attention_mask
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "candidate_length_complete": length,
                }), flush=True)
        per_cell = {}
        for key in sorted(totals):
            values = sorted(margins[key])
            per_cell["|".join(key)] = {
                "candidate_count": totals[key],
                "candidate_finite_fraction": finite_counts[key] / totals[key],
                "candidate_accuracy": hits[key] / totals[key],
                "median_expected_margin": values[len(values) // 2] if values else None,
            }
        total = sum(totals.values())
        finite_total = sum(finite_counts.values())
        summary = {
            "schema_version": "phase1112_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_count": total,
            "candidate_finite_fraction": finite_total / max(total, 1),
            "candidate_accuracy": sum(hits.values()) / max(finite_total, 1),
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
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "candidate_accuracy": summary["candidate_accuracy"],
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
