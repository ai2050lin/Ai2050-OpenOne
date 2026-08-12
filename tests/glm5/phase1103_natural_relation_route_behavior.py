#!/usr/bin/env python3
"""Run Phase1103 behavior gates for one local FP16 model."""

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
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1103_natural_relation_route_protocol as protocol


CANDIDATE_BATCH_SIZE = {"qwen3": 16, "glm4": 16, "deepseek7b": 16}
GENERATION_BATCH_SIZE = {"qwen3": 8, "glm4": 8, "deepseek7b": 8}


def normalized(value: str) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", value).strip().casefold()
    )


def label_position(text: str, label: str) -> int | None:
    match = re.search(
        rf"(?<![\w]){re.escape(normalized(label))}(?![\w])",
        normalized(text),
    )
    return match.start() if match else None


def generation_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (
            row["route_type"] not in ("exact", "paraphrase")
            or row["congruence"] != "conflict"
        ):
            continue
        key = (
            str(row["relation_pair"]), str(row["surface"]),
            str(row["split"]), str(row["route_type"]),
        )
        grouped[key][str(row["unit_id"])].append(row)
    selected = []
    for key in sorted(grouped):
        unit_ids = sorted(grouped[key])[:protocol.GENERATION_ITEMS_PER_CELL]
        for unit_id in unit_ids:
            panel = grouped[key][unit_id]
            for target in protocol.TARGET_RELATIONS:
                for orientation in protocol.ORIENTATIONS:
                    selected.append(next(
                        row for row in panel
                        if int(row["target_relation"]) == target
                        and int(row["orientation"]) == orientation
                        and int(row["relation_order"]) == 0
                    ))
    return selected


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1103 protocol audit failed")
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

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        detail = []
        totals: Counter = Counter()
        finite_counts: Counter = Counter()
        hits: Counter = Counter()
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
                        scores = {
                            answer_class: float(logits[slot, ids[0]].item())
                            for answer_class, ids in row[
                                "candidate_first_token_ids"
                            ].items()
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
                            str(row["relation_pair"]), str(row["surface"]),
                            str(row["split"]), str(row["route_type"]),
                            str(row["congruence"]),
                        )
                        totals[key] += 1
                        finite_counts[key] += int(finite)
                        hits[key] += int(hit)
                        detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "relation_pair": row["relation_pair"],
                            "family": row["family"],
                            "surface": row["surface"],
                            "split": row["split"],
                            "route_type": row["route_type"],
                            "congruence": row["congruence"],
                            "state": row["state"],
                            "target_relation": row["target_relation"],
                            "relation_order": row["relation_order"],
                            "orientation": row["orientation"],
                            "expected_class": expected,
                            "expected_entity": row["expected_entity"],
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

        generation_rows = generation_selection(rows)
        generation_detail = []
        generation_totals: Counter = Counter()
        generation_hits: Counter = Counter()
        generation_first: Counter = Counter()
        by_generation_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in generation_rows:
            by_generation_length[len(row["input_ids"])].append(row)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        with torch.inference_mode():
            for length in sorted(by_generation_length):
                panel = by_generation_length[length]
                batch_size = GENERATION_BATCH_SIZE[model_name]
                for start in range(0, len(panel), batch_size):
                    batch = panel[start:start + batch_size]
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.ones_like(input_ids)
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=int(prereg["generation_steps"]),
                        pad_token_id=pad_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    suffixes = output_ids[:, input_ids.shape[1]:]
                    for slot, row in enumerate(batch):
                        text = tokenizer.decode(
                            suffixes[slot].detach().cpu().tolist(),
                            skip_special_tokens=True,
                        )
                        expected = str(row["expected_entity"])
                        other_class = (
                            "e1" if row["expected_class"] == "e0" else "e0"
                        )
                        other = str(row["candidate_labels"][other_class])
                        expected_at = label_position(text, expected)
                        other_at = label_position(text, other)
                        target_before = expected_at is not None and (
                            other_at is None or expected_at < other_at
                        )
                        semantic_first = expected_at == 0
                        key = (
                            str(row["relation_pair"]), str(row["surface"]),
                            str(row["split"]), str(row["route_type"]),
                        )
                        generation_totals[key] += 1
                        generation_hits[key] += int(target_before)
                        generation_first[key] += int(semantic_first)
                        generation_detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "relation_pair": row["relation_pair"],
                            "surface": row["surface"],
                            "split": row["split"],
                            "route_type": row["route_type"],
                            "state": row["state"],
                            "expected_entity": expected,
                            "other_entity": other,
                            "generated_text": text,
                            "target_before_distractor": target_before,
                            "semantic_first": semantic_first,
                        })
                    del output_ids, suffixes, input_ids, attention_mask
                print(json.dumps({
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "generation_length_complete": length,
                }), flush=True)

        per_cell = {}
        for key in sorted(totals):
            total = totals[key]
            per_cell["|".join(key)] = {
                "candidate_count": total,
                "candidate_finite_fraction": finite_counts[key] / total,
                "candidate_accuracy": hits[key] / total,
            }
        per_generation_cell = {}
        for key in sorted(generation_totals):
            total = generation_totals[key]
            per_generation_cell["|".join(key)] = {
                "generation_count": total,
                "target_before_distractor_accuracy": (
                    generation_hits[key] / total
                ),
                "semantic_first_rate": generation_first[key] / total,
            }
        elapsed = time.time() - started
        total_candidates = sum(totals.values())
        total_finite = sum(finite_counts.values())
        summary = {
            "schema_version": "phase1103_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_count": total_candidates,
            "candidate_finite_fraction": total_finite / max(total_candidates, 1),
            "candidate_accuracy": sum(hits.values()) / max(total_finite, 1),
            "generation_count": sum(generation_totals.values()),
            "generation_target_before_distractor_accuracy": (
                sum(generation_hits.values())
                / max(sum(generation_totals.values()), 1)
            ),
            "generation_semantic_first_rate": (
                sum(generation_first.values())
                / max(sum(generation_totals.values()), 1)
            ),
            "per_cell": per_cell,
            "per_generation_cell": per_generation_cell,
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", detail)
        protocol.write_jsonl(
            output_root / "generation_detail.jsonl", generation_detail
        )
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "candidate_finite_fraction": summary[
                "candidate_finite_fraction"
            ],
            "candidate_accuracy": summary["candidate_accuracy"],
            "generation_accuracy": summary[
                "generation_target_before_distractor_accuracy"
            ],
            "elapsed_seconds": elapsed,
            "summary_digest": summary["summary_digest"],
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
