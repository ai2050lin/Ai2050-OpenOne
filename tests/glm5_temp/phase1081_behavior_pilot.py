#!/usr/bin/env python3
"""Behavior-only pilot for the frozen Phase1081 natural cloze protocol."""

from __future__ import annotations

import argparse
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
import phase1081_latin_route_atlas_protocol as protocol


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"\s+", " ", value)


def semantic_first(generated: str, target: str) -> bool:
    text = normalized_text(generated)
    label = normalized_text(target)
    hit = text.startswith(label)
    if hit and label and label[-1].isalnum():
        tail = text[len(label):]
        hit = not tail or not tail[0].isalnum()
    return hit


def label_position(text: str, label: str) -> int | None:
    normalized = normalized_text(text)
    target = normalized_text(label)
    if not target:
        return None
    if target[0].isalnum() or target[-1].isalnum():
        match = re.search(
            rf"(?<![\w]){re.escape(target)}(?![\w])",
            normalized,
        )
        return match.start() if match else None
    position = normalized.find(target)
    return position if position >= 0 else None


def target_before_distractor(
    generated: str,
    target: str,
    distractor: str,
) -> tuple[bool, int | None, int | None]:
    target_position = label_position(generated, target)
    distractor_position = label_position(generated, distractor)
    hit = target_position is not None and (
        distractor_position is None or target_position < distractor_position
    )
    return hit, target_position, distractor_position


def pad(rows: list[dict[str, Any]], pad_id: int, device):
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
    return input_ids, attention_mask, lengths


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["template"] == 0
        and row["panel"] == "active"
        and row["label_swap"] == 0
    ]


def generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            units = sorted({
                row["unit_id"] for row in rows
                if row["family"] == family and row["split"] == split
            })[:protocol.GENERATION_UNITS_PER_FAMILY_SPLIT]
            for local, unit_id in enumerate(units):
                query = local % 2
                state = f"t0_cactive_m0_q{query}_w0"
                row = next(
                    row for row in rows
                    if row["unit_id"] == unit_id and row["state"] == state
                )
                selected.append({
                    **row,
                    "semantic_case_index": int(row["case_index"]),
                })
    return selected


def run(model_name: str) -> None:
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    pilot_rows = selected_rows(rows)
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

        totals: Counter = Counter()
        hits: Counter = Counter()
        finite_counts: Counter = Counter()
        examples = []
        batch_size = {"qwen3": 12, "glm4": 6, "deepseek7b": 8}[model_name]
        with torch.inference_mode():
            for start in range(0, len(pilot_rows), batch_size):
                batch = pilot_rows[start:start + batch_size]
                input_ids, attention_mask, lengths = pad(batch, int(pad_id), device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = output.logits
                positions = (lengths - 1).to(logits.device)
                axis = torch.arange(len(batch), device=logits.device)
                final = logits[axis, positions].float()
                for local, row in enumerate(batch):
                    scores = {}
                    for answer_class in ("a0", "a1"):
                        ids = torch.tensor(
                            row["candidate_first_token_ids"][answer_class],
                            dtype=torch.long,
                            device=final.device,
                        )
                        scores[answer_class] = float(final[local, ids].max().item())
                    expected = row["expected_class"]
                    other = "a1" if expected == "a0" else "a0"
                    finite = all(math.isfinite(value) for value in scores.values())
                    hit = finite and scores[expected] > scores[other]
                    key = (row["family"], row["split"])
                    totals[key] += 1
                    finite_counts[key] += int(finite)
                    hits[key] += int(hit)
                    if len(examples) < 16:
                        examples.append({
                            "family": row["family"],
                            "split": row["split"],
                            "state": row["state"],
                            "target": row["target_answer"],
                            "margin": (
                                scores[expected] - scores[other]
                                if finite else None
                            ),
                        })
                del output, logits, final, input_ids, attention_mask, lengths
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        selected_generation = generation_rows(rows)
        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        generated = generation.generate_case_outputs(
            model,
            device,
            selected_generation,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=protocol.GENERATION_STEPS,
        )
        generation_totals: Counter = Counter()
        generation_hits: Counter = Counter()
        generation_first_hits: Counter = Counter()
        generation_examples = []
        for row in selected_generation:
            token_ids = generated[int(row["case_index"])]
            text = strict_generated_answer(tokenizer, token_ids, eos_ids)
            key = (row["family"], row["split"])
            distractor = row["answer_labels"][1 - int(row["answer_index"])]
            hit, target_position, distractor_position = target_before_distractor(
                text, row["target_answer"], distractor
            )
            first_hit = semantic_first(text, row["target_answer"])
            generation_totals[key] += 1
            generation_hits[key] += int(hit)
            generation_first_hits[key] += int(first_hit)
            generation_examples.append({
                "family": row["family"],
                "split": row["split"],
                "state": row["state"],
                "target": row["target_answer"],
                "distractor": distractor,
                "generated": text,
                "semantic_first": first_hit,
                "target_before_distractor": hit,
                "target_position": target_position,
                "distractor_position": distractor_position,
            })

        by_family = {}
        for family in protocol.FAMILIES:
            family_candidate_total = sum(
                totals[(family, split)] for split in protocol.SPLITS
            )
            family_candidate_hits = sum(
                hits[(family, split)] for split in protocol.SPLITS
            )
            family_generation_total = sum(
                generation_totals[(family, split)] for split in protocol.SPLITS
            )
            family_generation_hits = sum(
                generation_hits[(family, split)] for split in protocol.SPLITS
            )
            family_generation_first_hits = sum(
                generation_first_hits[(family, split)]
                for split in protocol.SPLITS
            )
            by_family[family] = {
                "candidate_count": family_candidate_total,
                "candidate_accuracy": (
                    family_candidate_hits / family_candidate_total
                    if family_candidate_total else None
                ),
                "generation_count": family_generation_total,
                "generation_accuracy": (
                    family_generation_hits / family_generation_total
                    if family_generation_total else None
                ),
                "generation_first_accuracy": (
                    family_generation_first_hits / family_generation_total
                    if family_generation_total else None
                ),
            }
        result = {
            "schema_version": "phase1081_behavior_pilot.v1",
            "phase": protocol.PHASE,
            "status": "behavior_only_no_hidden_states",
            "model": model_name,
            "protocol_digest": protocol.read_json(
                protocol.OUT_ROOT / "protocol" / "preregistration.json"
            )["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "candidate_case_count": len(pilot_rows),
            "generation_case_count": len(selected_generation),
            "by_family": by_family,
            "candidate_examples": examples,
            "generation_examples": generation_examples,
            "elapsed_seconds": time.time() - started,
        }
        result["pilot_digest"] = protocol.digest(result)
        protocol.write_json(
            protocol.OUT_ROOT / "pilot" / f"{model_name}.json", result
        )
        print({
            "phase": protocol.PHASE,
            "model": model_name,
            "status": result["status"],
            "by_family": by_family,
            "pilot_digest": result["pilot_digest"],
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
