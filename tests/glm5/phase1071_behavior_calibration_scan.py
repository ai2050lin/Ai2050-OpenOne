#!/usr/bin/env python3
"""Run the Phase1071 behavior-only prompt calibration in FP16."""

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
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as generation
import phase1062_text_equivalence_scan as text_tools
import phase1069_local_coordinate_scan as previous
import phase1071_behavior_calibration_protocol as protocol


FORWARD_BATCH_SIZE = {
    "qwen3": 12,
    "glm4": 4,
    "deepseek7b": 4,
}


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
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


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1071 calibration protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
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

        candidate_records = []
        candidate_total = Counter()
        candidate_hit = Counter()
        greedy_hit = Counter()
        nonfinite_candidate = 0
        with torch.inference_mode():
            for start in range(
                0, len(rows), FORWARD_BATCH_SIZE[model_name]
            ):
                batch = rows[
                    start:start + FORWARD_BATCH_SIZE[model_name]
                ]
                input_ids, attention_mask, lengths = pad_rows(
                    batch, int(pad_id), device
                )
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                positions = (lengths - 1).to(output.logits.device)
                axes = torch.arange(
                    len(batch), device=output.logits.device
                )
                logits = output.logits[
                    axes, positions, :
                ].float()
                for index, row in enumerate(batch):
                    scores = {}
                    for class_name in ("b0", "b1"):
                        token_ids = row[
                            "candidate_first_token_ids"
                        ][class_name]
                        class_values = logits[index, token_ids]
                        scores[class_name] = float(
                            class_values.max().item()
                        )
                    expected = str(row["expected_class"])
                    other = "b1" if expected == "b0" else "b0"
                    margin = scores[expected] - scores[other]
                    finite = all(
                        math.isfinite(value)
                        for value in scores.values()
                    ) and math.isfinite(margin)
                    hit = bool(finite and margin > 0.0)
                    greedy_id = int(logits[index].argmax().item())
                    greedy = bool(
                        finite
                        and greedy_id
                        in row["candidate_first_token_ids"][expected]
                    )
                    nonfinite_candidate += int(not finite)
                    keys = (
                        ("style", int(row["prompt_style"])),
                        (
                            "style_relation",
                            int(row["prompt_style"]),
                            str(row["relation"]),
                        ),
                        (
                            "style_path",
                            int(row["prompt_style"]),
                            str(row["path_name"]),
                        ),
                    )
                    for key in keys:
                        candidate_total[key] += 1
                        candidate_hit[key] += int(hit)
                        greedy_hit[key] += int(greedy)
                    candidate_records.append({
                        "schema_version": (
                            "phase1071_calibration_candidate.v1"
                        ),
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "record_id": row["record_id"],
                        "prompt_style": row["prompt_style"],
                        "relation": row["relation"],
                        "query_type": row["query_type"],
                        "path_name": row["path_name"],
                        "expected_class": expected,
                        "candidate_scores": (
                            scores if finite else None
                        ),
                        "candidate_margin": (
                            margin if finite else None
                        ),
                        "candidate_hit": hit,
                        "greedy_first_token_id": greedy_id,
                        "greedy_hit": greedy,
                        "nonfinite_candidate": not finite,
                    })
                del output, logits, input_ids, attention_mask, lengths
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = min(
                    start + len(batch), len(rows)
                )
                if completed % 160 == 0 or completed == len(rows):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "stage": "calibration_candidate",
                        "model": model_name,
                        "complete": completed,
                        "total": len(rows),
                    }), flush=True)

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        generated = generation.generate_case_outputs(
            model,
            device,
            rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["natural_generation_steps"]),
        )
        natural_records = []
        natural_total = Counter()
        semantic_first = Counter()
        strict_name = Counter()
        terminated_count = Counter()
        tail_counts: dict[tuple[Any, ...], Counter] = defaultdict(
            Counter
        )
        for row in rows:
            output_ids = generated[int(row["semantic_case_index"])]
            answer = text_tools.decode_content(
                tokenizer, output_ids, eos_ids
            )
            terminated = generation.terminated(output_ids, eos_ids)
            classification = previous.natural_classification(
                answer,
                list(row["acceptable_labels"]),
                terminated,
            )
            keys = (
                ("style", int(row["prompt_style"])),
                (
                    "style_relation",
                    int(row["prompt_style"]),
                    str(row["relation"]),
                ),
                (
                    "style_path",
                    int(row["prompt_style"]),
                    str(row["path_name"]),
                ),
            )
            for key in keys:
                natural_total[key] += 1
                semantic_first[key] += int(
                    classification["semantic_first"]
                )
                strict_name[key] += int(
                    classification["strict_name_only"]
                )
                terminated_count[key] += int(terminated)
                tail_counts[key][classification["tail_class"]] += 1
            natural_records.append({
                "schema_version": (
                    "phase1071_calibration_natural.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "record_id": row["record_id"],
                "prompt_style": row["prompt_style"],
                "relation": row["relation"],
                "query_type": row["query_type"],
                "path_name": row["path_name"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                "terminated": terminated,
                **classification,
            })

        style_summaries = {}
        for style in protocol.PROMPT_STYLES:
            style_key = ("style", style)
            relation_rows = {}
            for relation in protocol.RELATION_NAMES:
                key = ("style_relation", style, relation)
                relation_rows[relation] = {
                    "candidate_count": candidate_total[key],
                    "candidate_accuracy": rate(
                        candidate_hit[key], candidate_total[key]
                    ),
                    "semantic_first_rate": rate(
                        semantic_first[key], natural_total[key]
                    ),
                }
            path_rows = {}
            for path_name in protocol.PATH_NAMES.values():
                key = ("style_path", style, path_name)
                path_rows[path_name] = {
                    "candidate_count": candidate_total[key],
                    "candidate_accuracy": rate(
                        candidate_hit[key], candidate_total[key]
                    ),
                    "semantic_first_rate": rate(
                        semantic_first[key], natural_total[key]
                    ),
                }
            style_summaries[str(style)] = {
                "prompt_style": style,
                "prompt_style_label": protocol.STYLE_LABELS[style],
                "candidate_count": candidate_total[style_key],
                "candidate_accuracy": rate(
                    candidate_hit[style_key],
                    candidate_total[style_key],
                ),
                "greedy_first_token_accuracy": rate(
                    greedy_hit[style_key],
                    candidate_total[style_key],
                ),
                "natural_count": natural_total[style_key],
                "semantic_first_rate": rate(
                    semantic_first[style_key],
                    natural_total[style_key],
                ),
                "strict_name_only_rate": rate(
                    strict_name[style_key],
                    natural_total[style_key],
                ),
                "terminated_rate": rate(
                    terminated_count[style_key],
                    natural_total[style_key],
                ),
                "tail_class_counts": dict(
                    tail_counts[style_key]
                ),
                "relations": relation_rows,
                "paths": path_rows,
            }

        finite_rate = (
            1.0 - nonfinite_candidate / len(rows)
            if rows else 0.0
        )
        summary = {
            "schema_version": (
                "phase1071_calibration_model_summary.v1"
            ),
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "case_count": len(rows),
            "candidate_finite_rate": finite_rate,
            "nonfinite_candidate_count": nonfinite_candidate,
            "styles": style_summaries,
            "elapsed_seconds": time.time() - started,
        }
        atlas_root = (
            protocol.CALIBRATION_ROOT / "atlas" / model_name
        )
        protocol.write_jsonl(
            atlas_root / "candidate_behavior.jsonl",
            candidate_records,
        )
        protocol.write_jsonl(
            atlas_root / "natural_generation_audit.jsonl",
            natural_records,
        )
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "stage": "calibration_complete",
            "model": model_name,
            "candidate_finite_rate": finite_rate,
            "styles": {
                key: {
                    "candidate": value["candidate_accuracy"],
                    "semantic": value["semantic_first_rate"],
                    "strict": value["strict_name_only_rate"],
                }
                for key, value in style_summaries.items()
            },
            "elapsed_seconds": summary["elapsed_seconds"],
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
