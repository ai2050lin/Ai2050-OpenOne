#!/usr/bin/env python3
"""Run the frozen Phase1076 polarity and surface behavior gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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
import phase1069_local_coordinate_scan as natural_tools
import phase1076_polarity_head_causal_protocol as protocol


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device: torch.device,
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


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["contrast"]),
            str(row["task"]),
            str(row["path"]),
            str(row["layout"]),
            str(row["template_index"]),
        )].append(row)
    selected = []
    for key, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda row: hashlib.sha256(
                (
                    "phase1076-natural|"
                    + str(row["record_id"])
                ).encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < 2:
            raise RuntimeError(
                f"insufficient natural cases for {key}"
            )
        selected.extend(ranked[:2])
    expected = (
        len(protocol.CONTRASTS)
        * 2
        * len(protocol.PATHS)
        * len(protocol.LAYOUTS)
        * len(protocol.TEMPLATES)
        * 2
    )
    if len(selected) != expected:
        raise RuntimeError(
            f"natural selection drift: {len(selected)} != {expected}"
        )
    return selected


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def class_scores(
    logits: torch.Tensor,
    row: dict[str, Any],
) -> dict[str, float]:
    scores = {}
    for class_name in ("b0", "b1"):
        token_ids = torch.tensor(
            row["candidate_first_token_ids"][class_name],
            dtype=torch.long,
            device=logits.device,
        )
        scores[class_name] = float(logits[token_ids].max().item())
    return scores


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1076 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    if len(rows) != prereg["case_count_per_model"]:
        raise RuntimeError("Phase1076 case count drift")

    started = time.time()
    model = tokenizer = None
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
            raise RuntimeError("tokenizer has no pad/eos token")

        totals: Counter = Counter()
        finite_hits: Counter = Counter()
        hits: Counter = Counter()
        greedy_hits: Counter = Counter()
        behavior_rows = []
        for batch in chunks(rows, bridge.PAIR_BATCH_SIZE[model_name]):
            input_ids, attention_mask, lengths = pad_rows(
                batch, int(pad_id), device
            )
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            axes = torch.arange(
                len(batch), device=output.logits.device
            )
            logits = output.logits[
                axes, (lengths - 1).to(output.logits.device), :
            ].float()
            for index, row in enumerate(batch):
                values = logits[index]
                scores = class_scores(values, row)
                expected = str(row["expected_class"])
                other = "b1" if expected == "b0" else "b0"
                margin = scores[expected] - scores[other]
                finite = all(
                    math.isfinite(value) for value in scores.values()
                ) and math.isfinite(margin)
                hit = bool(finite and margin > 0.0)
                greedy_token = int(torch.argmax(values).item())
                greedy_hit = bool(
                    finite
                    and greedy_token
                    in {
                        int(value)
                        for value in row[
                            "candidate_first_token_ids"
                        ][expected]
                    }
                )
                keys = (
                    ("overall",),
                    ("contrast", row["contrast"]),
                    (
                        "contrast_task",
                        row["contrast"],
                        row["task"],
                    ),
                    (
                        "contrast_path",
                        row["contrast"],
                        row["path"],
                    ),
                    (
                        "contrast_layout",
                        row["contrast"],
                        row["layout"],
                    ),
                )
                for key in keys:
                    totals[key] += 1
                    finite_hits[key] += int(finite)
                    hits[key] += int(hit)
                    greedy_hits[key] += int(greedy_hit)
                behavior_rows.append({
                    "schema_version": (
                        "phase1076_candidate_behavior.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "semantic_case_index": int(
                        row["semantic_case_index"]
                    ),
                    "record_id": row["record_id"],
                    "pair_id": row["pair_id"],
                    "factor_id": row["factor_id"],
                    "contrast": row["contrast"],
                    "task": row["task"],
                    "path": row["path"],
                    "layout": row["layout"],
                    "template_index": row["template_index"],
                    "replicate": row["replicate"],
                    "orientation": row["orientation"],
                    "lexical_branch": row["lexical_branch"],
                    "expected_answer": row["expected_answer"],
                    "expected_class": expected,
                    "candidate_class_scores": {
                        key: (
                            value if math.isfinite(value) else None
                        )
                        for key, value in scores.items()
                    },
                    "candidate_margin": (
                        margin if math.isfinite(margin) else None
                    ),
                    "nonfinite_candidate": not finite,
                    "candidate_hit": hit,
                    "greedy_first_token_id": greedy_token,
                    "greedy_first_token_text": tokenizer.decode(
                        [greedy_token]
                    ),
                    "greedy_first_token_hit": greedy_hit,
                })
            del (
                output,
                logits,
                input_ids,
                attention_mask,
                lengths,
            )

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        natural_rows = natural_selection(rows)
        natural_outputs = generation.generate_case_outputs(
            model,
            device,
            natural_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["natural_generation_steps"]),
        )
        natural_total: Counter = Counter()
        natural_semantic: Counter = Counter()
        natural_strict: Counter = Counter()
        natural_terminated: Counter = Counter()
        natural_records = []
        for row in natural_rows:
            index = int(row["semantic_case_index"])
            output_ids = natural_outputs[index]
            answer = text_tools.decode_content(
                tokenizer, output_ids, eos_ids
            )
            terminated = generation.terminated(output_ids, eos_ids)
            classification = natural_tools.natural_classification(
                answer,
                list(row["acceptable_labels"]),
                terminated,
            )
            keys = (
                ("overall",),
                ("contrast", row["contrast"]),
                (
                    "contrast_task",
                    row["contrast"],
                    row["task"],
                ),
            )
            for key in keys:
                natural_total[key] += 1
                natural_semantic[key] += int(
                    classification["semantic_first"]
                )
                natural_strict[key] += int(
                    classification["strict_name_only"]
                )
                natural_terminated[key] += int(terminated)
            natural_records.append({
                "schema_version": (
                    "phase1076_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "record_id": row["record_id"],
                "contrast": row["contrast"],
                "task": row["task"],
                "path": row["path"],
                "layout": row["layout"],
                "template_index": row["template_index"],
                "expected_answer": row["expected_answer"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                **classification,
            })

        def value(counter: Counter, *key: str) -> float:
            return rate(counter[tuple(key)], totals[tuple(key)])

        def natural_value(*key: str) -> float:
            return rate(
                natural_semantic[tuple(key)],
                natural_total[tuple(key)],
            )

        gates = prereg["gates"]
        contrast_summaries = {}
        for contrast in protocol.CONTRASTS:
            finite_rate = value(
                finite_hits, "contrast", contrast
            )
            accuracy = value(hits, "contrast", contrast)
            tasks = protocol.TASKS_BY_CONTRAST[contrast]
            by_task = {
                task: value(
                    hits, "contrast_task", contrast, task
                )
                for task in tasks
            }
            by_path = {
                path: value(
                    hits, "contrast_path", contrast, path
                )
                for path in protocol.PATHS
            }
            by_layout = {
                layout: value(
                    hits, "contrast_layout", contrast, layout
                )
                for layout in protocol.LAYOUTS
            }
            natural_rate = natural_value("contrast", contrast)
            checks = {
                "finite_rate": (
                    finite_rate
                    >= gates["behavior_finite_rate_min"]
                ),
                "candidate_accuracy": (
                    accuracy
                    >= gates["behavior_contrast_accuracy_min"]
                ),
                "per_task_accuracy": (
                    min(by_task.values())
                    >= gates["behavior_task_accuracy_min"]
                ),
                "per_path_accuracy": (
                    min(by_path.values())
                    >= gates["behavior_path_accuracy_min"]
                ),
                "natural_semantic_first": (
                    natural_rate
                    >= gates[
                        "behavior_natural_semantic_first_min"
                    ]
                ),
            }
            contrast_summaries[contrast] = {
                "case_count": totals[("contrast", contrast)],
                "candidate_finite_rate": finite_rate,
                "candidate_accuracy": accuracy,
                "greedy_first_token_accuracy": value(
                    greedy_hits, "contrast", contrast
                ),
                "by_task": by_task,
                "by_path": by_path,
                "by_layout": by_layout,
                "natural_semantic_first_rate": natural_rate,
                "gate_checks": checks,
                "contrast_behavior_gate_passed": all(
                    checks.values()
                ),
            }
        model_passed = all(
            summary["contrast_behavior_gate_passed"]
            for summary in contrast_summaries.values()
        )
        summary = {
            "schema_version": "phase1076_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "case_count": len(rows),
            "natural_case_count": len(natural_rows),
            "overall_candidate_finite_rate": value(
                finite_hits, "overall"
            ),
            "overall_candidate_accuracy": value(hits, "overall"),
            "overall_greedy_first_token_accuracy": value(
                greedy_hits, "overall"
            ),
            "overall_natural_semantic_first_rate": rate(
                natural_semantic[("overall",)],
                natural_total[("overall",)],
            ),
            "contrasts": contrast_summaries,
            "model_behavior_gate_passed": model_passed,
            "elapsed_seconds": float(time.time() - started),
        }
        out_dir = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(
            out_dir / "candidate_behavior.jsonl", behavior_rows
        )
        protocol.write_jsonl(
            out_dir / "natural_generation.jsonl", natural_records
        )
        protocol.write_json(out_dir / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "overall_candidate_accuracy": summary[
                "overall_candidate_accuracy"
            ],
            "overall_finite_rate": summary[
                "overall_candidate_finite_rate"
            ],
            "contrast_accuracy": {
                key: value["candidate_accuracy"]
                for key, value in contrast_summaries.items()
            },
            "model_behavior_gate_passed": model_passed,
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=protocol.MODELS, required=True
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
