#!/usr/bin/env python3
"""Run the Phase1074 behavior-first late-polarity benchmark in FP16."""

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
import phase1069_local_coordinate_scan as previous
import phase1074_polarity_dynamics_protocol as protocol


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
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = (
        defaultdict(list)
    )
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
                    "phase1074-natural|"
                    + str(row["record_id"])
                ).encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < 2:
            raise RuntimeError(
                f"insufficient natural audit candidates for {key}"
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
        raise RuntimeError("Phase1074 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
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

        behavior_rows = []
        totals = Counter()
        hits = Counter()
        greedy_hits = Counter()
        nonfinite = 0
        for batch in chunks(
            rows, bridge.PAIR_BATCH_SIZE[model_name]
        ):
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
                nonfinite += int(not finite)
                keys = (
                    ("overall",),
                    ("task", row["task"]),
                    ("split", row["split"]),
                    ("path", row["path"]),
                    ("layout", row["layout"]),
                    ("relation", row["relation"]),
                    (
                        "relation_task",
                        row["relation"],
                        row["task"],
                    ),
                    (
                        "relation_path",
                        row["relation"],
                        row["path"],
                    ),
                    (
                        "relation_split",
                        row["relation"],
                        row["split"],
                    ),
                )
                for key in keys:
                    totals[key] += 1
                    hits[key] += int(hit)
                    greedy_hits[key] += int(greedy_hit)
                behavior_rows.append({
                    "schema_version": (
                        "phase1074_candidate_behavior.v1"
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
        natural_records = []
        natural_total = Counter()
        natural_semantic = Counter()
        natural_strict = Counter()
        natural_terminated = Counter()
        for row in natural_rows:
            index = int(row["semantic_case_index"])
            output_ids = natural_outputs[index]
            answer = text_tools.decode_content(
                tokenizer, output_ids, eos_ids
            )
            terminated = generation.terminated(
                output_ids, eos_ids
            )
            classification = previous.natural_classification(
                answer,
                list(row["acceptable_labels"]),
                terminated,
            )
            keys = (
                ("overall",),
                ("split", row["split"]),
                ("task", row["task"]),
                ("path", row["path"]),
                ("relation", row["relation"]),
                (
                    "relation_task",
                    row["relation"],
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
                    "phase1074_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "record_id": row["record_id"],
                "pair_id": row["pair_id"],
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

        gates = prereg["gates"]

        def candidate_value(*key: str) -> float:
            return rate(hits[tuple(key)], totals[tuple(key)])

        def natural_value(*key: str) -> float:
            return rate(
                natural_semantic[tuple(key)],
                natural_total[tuple(key)],
            )

        relation_summaries = {}
        strong_relations = []
        for relation in protocol.RELATIONS:
            by_task = {
                task: candidate_value(
                    "relation_task", relation, task
                )
                for task in protocol.TASKS
            }
            by_path = {
                path: rate(
                    hits[("relation_path", relation, path)],
                    totals[("relation_path", relation, path)],
                )
                for path in protocol.PATHS
            }
            confirmation = rate(
                hits[("relation_split", relation, "confirmation")],
                totals[
                    ("relation_split", relation, "confirmation")
                ],
            )
            relation_accuracy = candidate_value(
                "relation", relation
            )
            relation_gate = bool(
                relation_accuracy
                >= gates["per_path_candidate_accuracy_min"]
                and min(by_task.values())
                >= gates["relation_task_candidate_accuracy_min"]
                and min(by_path.values())
                >= gates["relation_task_candidate_accuracy_min"]
                and confirmation
                >= gates["relation_task_candidate_accuracy_min"]
            )
            if relation_gate:
                strong_relations.append(relation)
            relation_summaries[relation] = {
                "candidate_accuracy": relation_accuracy,
                "by_task": by_task,
                "by_path": by_path,
                "confirmation_candidate_accuracy": confirmation,
                "natural_semantic_first_rate": natural_value(
                    "relation", relation
                ),
                "strong_relation_gate_passed": relation_gate,
            }

        finite_rate = rate(
            len(rows) - nonfinite, len(rows)
        )
        overall_accuracy = candidate_value("overall")
        task_accuracy = {
            task: candidate_value("task", task)
            for task in protocol.TASKS
        }
        split_accuracy = {
            split: candidate_value("split", split)
            for split in protocol.SPLITS
        }
        path_accuracy = {
            path: candidate_value("path", path)
            for path in protocol.PATHS
        }
        natural_rate = natural_value("overall")
        confirmation_natural = natural_value(
            "split", "confirmation"
        )
        model_gate_checks = {
            "candidate_finite_rate": (
                finite_rate
                >= gates["candidate_finite_rate_min"]
            ),
            "overall_candidate_accuracy": (
                overall_accuracy
                >= gates["overall_candidate_accuracy_min"]
            ),
            "per_task_candidate_accuracy": (
                min(task_accuracy.values())
                >= gates["per_task_candidate_accuracy_min"]
            ),
            "confirmation_candidate_accuracy": (
                split_accuracy["confirmation"]
                >= gates["confirmation_candidate_accuracy_min"]
            ),
            "per_path_candidate_accuracy": (
                min(path_accuracy.values())
                >= gates["per_path_candidate_accuracy_min"]
            ),
            "semantic_first_natural_rate": (
                natural_rate
                >= gates["semantic_first_natural_rate_min"]
            ),
            "confirmation_semantic_first_rate": (
                confirmation_natural
                >= gates["confirmation_semantic_first_rate_min"]
            ),
            "minimum_strong_relations": (
                len(strong_relations)
                >= gates["minimum_strong_relations_per_model"]
            ),
        }
        summary = {
            "schema_version": "phase1074_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "case_count": len(rows),
            "natural_case_count": len(natural_rows),
            "candidate_finite_rate": finite_rate,
            "candidate_accuracy": overall_accuracy,
            "greedy_first_token_accuracy": rate(
                greedy_hits[("overall",)],
                totals[("overall",)],
            ),
            "by_task": task_accuracy,
            "by_split": split_accuracy,
            "by_path": path_accuracy,
            "by_layout": {
                layout: candidate_value("layout", layout)
                for layout in protocol.LAYOUTS
            },
            "natural_semantic_first_rate": natural_rate,
            "natural_strict_name_only_rate": rate(
                natural_strict[("overall",)],
                natural_total[("overall",)],
            ),
            "natural_terminated_rate": rate(
                natural_terminated[("overall",)],
                natural_total[("overall",)],
            ),
            "confirmation_natural_semantic_first_rate": (
                confirmation_natural
            ),
            "relations": relation_summaries,
            "strong_relations": strong_relations,
            "model_gate_checks": model_gate_checks,
            "model_behavior_gate_passed": all(
                model_gate_checks.values()
            ),
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
            "candidate_accuracy": overall_accuracy,
            "task_accuracy": task_accuracy,
            "confirmation_accuracy": (
                split_accuracy["confirmation"]
            ),
            "natural_semantic_first_rate": natural_rate,
            "strong_relations": strong_relations,
            "behavior_gate_passed": summary[
                "model_behavior_gate_passed"
            ],
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
