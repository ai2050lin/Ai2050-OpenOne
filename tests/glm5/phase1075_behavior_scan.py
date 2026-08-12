#!/usr/bin/env python3
"""Run the Phase1075 held-out relation-level behavior benchmark."""

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
import phase1075_relation_polarity_protocol as protocol


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
            str(row["relation"]),
            str(row["task"]),
            str(row["path"]),
            str(row["layout"]),
            str(row["split"]),
        )].append(row)
    selected = []
    for key, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda row: hashlib.sha256(
                (
                    "phase1075-natural|"
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
        len(protocol.RELATIONS)
        * len(protocol.TASKS)
        * len(protocol.PATHS)
        * len(protocol.LAYOUTS)
        * len(protocol.SPLITS)
        * 2
    )
    if len(selected) != expected:
        raise RuntimeError(
            f"natural selection drift: {len(selected)} != {expected}"
        )
    return selected


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1075 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    if len(rows) != prereg["case_count_per_model"]:
        raise RuntimeError("Phase1075 case count drift")

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
                )
            axes = torch.arange(
                len(batch), device=output.logits.device
            )
            logits = output.logits[
                axes, (lengths - 1).to(output.logits.device), :
            ].float()
            for index, row in enumerate(batch):
                values = logits[index]
                scores = {}
                for class_name in ("b0", "b1"):
                    token_ids = torch.tensor(
                        row["candidate_first_token_ids"][class_name],
                        dtype=torch.long,
                        device=values.device,
                    )
                    scores[class_name] = float(
                        values[token_ids].max().item()
                    )
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
                    in set(
                        int(value)
                        for value in row[
                            "candidate_first_token_ids"
                        ][expected]
                    )
                )
                keys = (
                    ("overall",),
                    ("relation", row["relation"]),
                    ("relation_task", row["relation"], row["task"]),
                    ("relation_path", row["relation"], row["path"]),
                    ("relation_layout", row["relation"], row["layout"]),
                    ("relation_split", row["relation"], row["split"]),
                    (
                        "relation_lexical",
                        row["relation"],
                        str(row["lexical_branch"]),
                    ),
                    (
                        "relation_orientation",
                        row["relation"],
                        str(row["orientation"]),
                    ),
                )
                for key in keys:
                    totals[key] += 1
                    finite_hits[key] += int(finite)
                    hits[key] += int(hit)
                    greedy_hits[key] += int(greedy_hit)
                behavior_rows.append({
                    "schema_version": (
                        "phase1075_candidate_behavior.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "semantic_case_index": int(
                        row["semantic_case_index"]
                    ),
                    "record_id": row["record_id"],
                    "pair_id": row["pair_id"],
                    "unit_id": row["unit_id"],
                    "relation": row["relation"],
                    "task": row["task"],
                    "path": row["path"],
                    "layout": row["layout"],
                    "template_index": row["template_index"],
                    "replicate": row["replicate"],
                    "split": row["split"],
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
                ("relation", row["relation"]),
                ("relation_split", row["relation"], row["split"]),
                ("relation_task", row["relation"], row["task"]),
                ("relation_path", row["relation"], row["path"]),
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
                    "phase1075_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "record_id": row["record_id"],
                "relation": row["relation"],
                "task": row["task"],
                "path": row["path"],
                "layout": row["layout"],
                "split": row["split"],
                "orientation": row["orientation"],
                "lexical_branch": row["lexical_branch"],
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
        relation_summaries = {}
        confirmed_relations = []
        for relation in protocol.RELATIONS:
            relation_accuracy = value(
                hits, "relation", relation
            )
            finite_rate = value(
                finite_hits, "relation", relation
            )
            by_task = {
                task: value(
                    hits, "relation_task", relation, task
                )
                for task in protocol.TASKS
            }
            by_path = {
                path: value(
                    hits, "relation_path", relation, path
                )
                for path in protocol.PATHS
            }
            by_layout = {
                layout: value(
                    hits, "relation_layout", relation, layout
                )
                for layout in protocol.LAYOUTS
            }
            by_lexical = {
                str(branch): value(
                    hits,
                    "relation_lexical",
                    relation,
                    str(branch),
                )
                for branch in protocol.LEXICAL_BRANCHES
            }
            by_orientation = {
                str(orientation): value(
                    hits,
                    "relation_orientation",
                    relation,
                    str(orientation),
                )
                for orientation in protocol.ORIENTATIONS
            }
            confirmation = value(
                hits, "relation_split", relation, "confirmation"
            )
            natural_rate = natural_value("relation", relation)
            confirmation_natural = natural_value(
                "relation_split", relation, "confirmation"
            )
            checks = {
                "finite_rate": (
                    finite_rate
                    >= gates["relation_finite_rate_min"]
                ),
                "candidate_accuracy": (
                    relation_accuracy
                    >= gates["relation_candidate_accuracy_min"]
                ),
                "per_task_accuracy": (
                    min(by_task.values())
                    >= gates["relation_task_accuracy_min"]
                ),
                "per_path_accuracy": (
                    min(by_path.values())
                    >= gates["relation_path_accuracy_min"]
                ),
                "per_layout_accuracy": (
                    min(by_layout.values())
                    >= gates["relation_layout_accuracy_min"]
                ),
                "per_lexical_accuracy": (
                    min(by_lexical.values())
                    >= gates["relation_lexical_accuracy_min"]
                ),
                "per_orientation_accuracy": (
                    min(by_orientation.values())
                    >= gates["relation_orientation_accuracy_min"]
                ),
                "confirmation_accuracy": (
                    confirmation
                    >= gates["relation_confirmation_accuracy_min"]
                ),
                "natural_semantic_first": (
                    natural_rate
                    >= gates["relation_natural_semantic_first_min"]
                ),
                "confirmation_natural": (
                    confirmation_natural
                    >= gates[
                        "relation_confirmation_natural_min"
                    ]
                ),
            }
            passed = all(checks.values())
            if passed:
                confirmed_relations.append(relation)
            relation_summaries[relation] = {
                "case_count": totals[("relation", relation)],
                "candidate_finite_rate": finite_rate,
                "candidate_accuracy": relation_accuracy,
                "greedy_first_token_accuracy": value(
                    greedy_hits, "relation", relation
                ),
                "by_task": by_task,
                "by_path": by_path,
                "by_layout": by_layout,
                "by_lexical_branch": by_lexical,
                "by_orientation": by_orientation,
                "confirmation_candidate_accuracy": confirmation,
                "natural_semantic_first_rate": natural_rate,
                "confirmation_natural_semantic_first_rate": (
                    confirmation_natural
                ),
                "gate_checks": checks,
                "relation_behavior_gate_passed": passed,
            }

        summary = {
            "schema_version": "phase1075_behavior_summary.v1",
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
            "overall_natural_strict_name_only_rate": rate(
                natural_strict[("overall",)],
                natural_total[("overall",)],
            ),
            "overall_natural_terminated_rate": rate(
                natural_terminated[("overall",)],
                natural_total[("overall",)],
            ),
            "relations": relation_summaries,
            "confirmed_relations": confirmed_relations,
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
            "confirmed_relations": confirmed_relations,
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
