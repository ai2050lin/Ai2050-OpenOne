#!/usr/bin/env python3
"""Score one Phase1126 model in FP16 and release it before the next model."""

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
import phase1126_semeval_lexsub_natural_cloze_protocol as protocol


BATCH_SIZES = {"qwen3": 10, "glm4": 4, "deepseek7b": 4}


def score_batch(model: Any, input_ids: torch.Tensor, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits
    selected_rows: list[int] = []
    selected_prediction_positions: list[int] = []
    selected_target_ids: list[int] = []
    selected_kinds: list[str] = []
    selected_case_offsets: list[int] = []
    for batch_index, row in enumerate(rows):
        candidate_positions = [int(value) for value in row["candidate_positions"]]
        positions = [*candidate_positions, *[int(value) for value in row["suffix_positions"]]]
        for local_index, position in enumerate(positions):
            selected_rows.append(batch_index)
            selected_prediction_positions.append(position - 1)
            selected_target_ids.append(int(row["input_ids"][position]))
            selected_kinds.append("candidate" if local_index < len(candidate_positions) else "suffix")
            selected_case_offsets.append(batch_index)

    row_index = torch.tensor(selected_rows, device=logits.device, dtype=torch.long)
    prediction_index = torch.tensor(selected_prediction_positions, device=logits.device, dtype=torch.long)
    target_ids = torch.tensor(selected_target_ids, device=logits.device, dtype=torch.long)
    selected_logits = logits[row_index, prediction_index, :].float()
    selected_log_probs = (
        selected_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        - torch.logsumexp(selected_logits, dim=-1)
    ).detach().cpu().tolist()

    per_case: list[dict[str, list[float]]] = [
        {"candidate": [], "suffix": []} for _ in rows
    ]
    for case_offset, kind, value in zip(selected_case_offsets, selected_kinds, selected_log_probs):
        per_case[case_offset][kind].append(float(value))
    results = []
    for values in per_case:
        candidate_logp = sum(values["candidate"])
        suffix_mean = sum(values["suffix"]) / len(values["suffix"]) if values["suffix"] else 0.0
        results.append({
            "candidate_logp": candidate_logp,
            "suffix_mean_logp": suffix_mean,
            "total_score": candidate_logp + suffix_mean,
        })
    del output, logits, selected_logits
    return results


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1126 protocol audit failed")
    if audit["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1126 protocol link mismatch")
    rows = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    if protocol.digest(rows) != prereg["case_digests"][model_name]:
        raise RuntimeError("Phase1126 case digest mismatch")

    started = time.time()
    model = None
    try:
        model, _tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        details: list[dict[str, Any]] = []
        batch_size = BATCH_SIZES[model_name]
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), batch_size):
                    batch = panel[start:start + batch_size]
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    scores = score_batch(model, input_ids, batch)
                    for row, score in zip(batch, scores):
                        finite = all(math.isfinite(value) for value in score.values())
                        details.append({
                            "case_index": row["case_index"],
                            "panel_index": row["panel_index"],
                            "item": row["item"],
                            "pos": row["pos"],
                            "partition": row["partition"],
                            "replica": row["replica"],
                            "route": row["route"],
                            "route_item": row["route_item"],
                            "context_sense": row["context_sense"],
                            "candidate_side": row["candidate_side"],
                            "candidate": row["candidate"],
                            "source_instance_id": row["source_instance_id"],
                            "lexical_overlap": row["lexical_overlap"],
                            "suffix_token_count": len(row["suffix_positions"]),
                            **score,
                            "finite": finite,
                        })
                    del input_ids

        details.sort(key=lambda row: row["case_index"])
        if [row["case_index"] for row in details] != list(range(len(rows))):
            raise RuntimeError("Phase1126 result case order mismatch")
        finite_count = sum(bool(row["finite"]) for row in details)
        summary = {
            "schema_version": "phase1126_semeval_lexsub_natural_cloze_behavior.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digests"][model_name],
            "case_count": len(details),
            "finite_count": finite_count,
            "finite_rate": finite_count / len(details),
            "precision": precision,
            "placement": placement,
            "batch_size": batch_size,
            "elapsed_seconds": time.time() - started,
            "detail_digest": protocol.digest(details),
        }
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "scores.jsonl", details)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
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
