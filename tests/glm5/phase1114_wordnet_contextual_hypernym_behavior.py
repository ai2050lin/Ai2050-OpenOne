#!/usr/bin/env python3
"""Run Phase1114 behavior for one local FP16 model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1114_wordnet_contextual_hypernym_protocol as protocol


BATCH_SIZE = {"qwen3": 16, "glm4": 64, "deepseek7b": 32}


def run(model_name: str) -> None:
    preregistration = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1114 protocol audit failed")
    rows = list(
        protocol.read_jsonl(
            protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        )
    )
    if protocol.digest(rows) != preregistration["case_digests"][model_name]:
        raise RuntimeError("Phase1114 case digest mismatch")

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
        detail: list[dict[str, Any]] = []
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), BATCH_SIZE[model_name]):
                    batch = panel[start : start + BATCH_SIZE[model_name]]
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
                        z = scores["sense0"] - scores["sense1"]
                        expected = str(row["expected_class"])
                        other = "sense1" if expected == "sense0" else "sense0"
                        expected_margin = scores[expected] - scores[other]
                        finite = (
                            all(math.isfinite(value) for value in scores.values())
                            and math.isfinite(z)
                            and math.isfinite(expected_margin)
                        )
                        hit = finite and expected_margin > 0.0
                        top_token_id = int(top_ids[slot].item())
                        top_class = next(
                            (
                                key
                                for key, token_id in candidate_ids.items()
                                if token_id == top_token_id
                            ),
                            None,
                        )
                        direct_candidate = top_class is not None
                        direct_hit = direct_candidate and top_class == expected
                        detail.append(
                            {
                                "case_index": int(row["case_index"]),
                                "record_id": row["record_id"],
                                "pair_id": row["pair_id"],
                                "concept_id": row["concept_id"],
                                "split": row["split"],
                                "template": int(row["template"]),
                                "sense": int(row["sense"]),
                                "base": row["base"],
                                "sense_offset": row["sense_offset"],
                                "native_example": row["native_example"],
                                "candidate_labels": row["candidate_labels"],
                                "expected_class": expected,
                                "scores": scores if finite else None,
                                "sense0_minus_sense1": z if finite else None,
                                "expected_margin": expected_margin if finite else None,
                                "finite": finite,
                                "hit": hit,
                                "top_token_id": top_token_id,
                                "top_token_text": tokenizer.decode([top_token_id]),
                                "top_class": top_class,
                                "direct_candidate": direct_candidate,
                                "direct_hit": direct_hit,
                            }
                        )
                    del output, logits, top_ids, input_ids, attention_mask
                print(
                    json.dumps(
                        {
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "candidate_length_complete": length,
                        }
                    ),
                    flush=True,
                )

        finite_rows = [row for row in detail if row["finite"]]
        summary = {
            "schema_version": "phase1114_contextual_hypernym_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": preregistration["protocol_digest"],
            "case_digest": preregistration["case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_count": len(detail),
            "candidate_finite_fraction": len(finite_rows) / max(len(detail), 1),
            "candidate_accuracy": sum(bool(row["hit"]) for row in finite_rows)
            / max(len(finite_rows), 1),
            "direct_candidate_output_rate": sum(
                bool(row["direct_candidate"]) for row in detail
            )
            / max(len(detail), 1),
            "direct_exact_accuracy": sum(bool(row["direct_hit"]) for row in detail)
            / max(len(detail), 1),
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = protocol.digest(summary)
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", detail)
        protocol.write_json(output_root / "summary.json", summary)
        print(
            json.dumps(
                {
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "candidate_count": summary["candidate_count"],
                    "candidate_finite_fraction": summary[
                        "candidate_finite_fraction"
                    ],
                    "candidate_accuracy": summary["candidate_accuracy"],
                    "direct_candidate_output_rate": summary[
                        "direct_candidate_output_rate"
                    ],
                    "direct_exact_accuracy": summary["direct_exact_accuracy"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "summary_digest": summary["summary_digest"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
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
