#!/usr/bin/env python3
"""Run Phase1096 behavior gates, one local FP16 model at a time."""

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
import phase1096_comparison_dynamics_protocol as protocol


CANDIDATE_BATCH_SIZE = {"qwen3": 64, "glm4": 32, "deepseek7b": 32}
GENERATION_BATCH_SIZE = {"qwen3": 16, "glm4": 8, "deepseek7b": 8}


def normalized(value: str) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", value).strip().casefold()
    )


def label_position(text: str, label: str) -> int | None:
    haystack = normalized(text)
    needle = normalized(label)
    match = re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack)
    return match.start() if match else None


def generation_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    selected_states = (
        "prelational_tmax_o0_c0",
        "prelational_tmin_o0_c0",
        "prelational_tmax_o1_c0",
        "prelational_tmin_o1_c0",
    )
    grouped: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for row in rows:
        if row["panel"] != "relational":
            continue
        key = (str(row["relation"]), str(row["surface"]), int(row["template"]))
        grouped[key].append(str(row["unit_id"]))
    for key in sorted(grouped):
        unit_ids = sorted(set(grouped[key]))[:protocol.GENERATION_ITEMS_PER_CELL]
        for unit_id in unit_ids:
            unit_rows = {str(row["state"]): row for row in rows if row["unit_id"] == unit_id}
            result.extend(unit_rows[state] for state in selected_states)
    return result


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1096 protocol audit failed")
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
        candidate_detail = []
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
                            for answer_class, ids in row["candidate_first_token_ids"].items()
                        }
                        expected = str(row["expected_class"])
                        other = "e1" if expected == "e0" else "e0"
                        margin = scores[expected] - scores[other]
                        finite = all(math.isfinite(value) for value in scores.values()) and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        key = (
                            str(row["relation"]), str(row["surface"]),
                            str(row["split"]), str(row["panel"]),
                        )
                        totals[key] += 1
                        finite_counts[key] += int(finite)
                        hits[key] += int(hit)
                        candidate_detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "relation": row["relation"],
                            "surface": row["surface"],
                            "split": row["split"],
                            "panel": row["panel"],
                            "state": row["state"],
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
        generated_detail = []
        generation_totals: Counter = Counter()
        generation_hits: Counter = Counter()
        generation_first: Counter = Counter()
        generation_by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in generation_rows:
            generation_by_length[len(row["input_ids"])].append(row)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        with torch.inference_mode():
            for length in sorted(generation_by_length):
                panel = generation_by_length[length]
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
                        other_class = "e1" if row["expected_class"] == "e0" else "e0"
                        other = str(row["candidate_labels"][other_class])
                        expected_at = label_position(text, expected)
                        other_at = label_position(text, other)
                        target_before = expected_at is not None and (
                            other_at is None or expected_at < other_at
                        )
                        semantic_first = expected_at == 0
                        key = (str(row["relation"]), str(row["surface"]), str(row["split"]))
                        generation_totals[key] += 1
                        generation_hits[key] += int(target_before)
                        generation_first[key] += int(semantic_first)
                        generated_detail.append({
                            "case_index": int(row["case_index"]),
                            "unit_id": row["unit_id"],
                            "relation": row["relation"],
                            "surface": row["surface"],
                            "split": row["split"],
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
        for relation in protocol.RELATIONS:
            for surface in protocol.SURFACES:
                for split in protocol.SPLITS:
                    for panel_name in protocol.PANELS:
                        key = (relation, surface, split, panel_name)
                        total = totals[key]
                        per_cell["|".join(key)] = {
                            "candidate_count": total,
                            "candidate_finite_fraction": finite_counts[key] / total if total else 0.0,
                            "candidate_accuracy": hits[key] / total if total else 0.0,
                        }
        per_generation_cell = {}
        for relation in protocol.RELATIONS:
            for surface in protocol.SURFACES:
                for split in protocol.SPLITS:
                    key = (relation, surface, split)
                    total = generation_totals[key]
                    per_generation_cell["|".join(key)] = {
                        "generation_count": total,
                        "target_before_distractor_accuracy": generation_hits[key] / total if total else 0.0,
                        "semantic_first_rate": generation_first[key] / total if total else 0.0,
                    }
        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1096_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "candidate_count": sum(totals.values()),
            "candidate_finite_fraction": sum(finite_counts.values()) / sum(totals.values()),
            "candidate_accuracy": sum(hits.values()) / sum(totals.values()),
            "generation_count": sum(generation_totals.values()),
            "generation_target_before_distractor_accuracy": (
                sum(generation_hits.values()) / sum(generation_totals.values())
            ),
            "per_cell": per_cell,
            "per_generation_cell": per_generation_cell,
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", candidate_detail)
        protocol.write_jsonl(output_root / "generation_detail.jsonl", generated_detail)
        protocol.write_json(output_root / "summary.json", summary)
        print({
            "phase": protocol.PHASE,
            "model": model_name,
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "candidate_accuracy": summary["candidate_accuracy"],
            "generation_accuracy": summary["generation_target_before_distractor_accuracy"],
            "elapsed_seconds": elapsed,
        })
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
