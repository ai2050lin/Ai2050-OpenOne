#!/usr/bin/env python3
"""Run the Phase1073 late-query operation-selection atlas in FP16."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
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
from phase1070_process_answer_scan import safe_cosine, state_name
import phase1073_late_query_protocol as protocol


EPSILON = 1e-12
CONDITIONING = ("all", "behavior_conditioned")
METRICS = (
    "operation_contrast_relative_magnitude",
    "transitive_did_relative_magnitude",
    "key_copy_did_relative_magnitude",
    "task_did_cosine",
    "operation_lexical_reuse_cosine",
    "operation_answer_invariance_cosine",
)


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device,
    fixed_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required_width = max(len(row["input_ids"]) for row in rows)
    width = required_width if fixed_width is None else int(fixed_width)
    if width < required_width:
        raise ValueError(
            f"fixed width {width} is below required width {required_width}"
        )
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
    positions = torch.zeros(
        (len(rows), len(protocol.CAPTURE_ROLES)),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor(
            [
                int(row["role_positions"][role])
                for role in protocol.CAPTURE_ROLES
            ],
            dtype=torch.long,
            device=device,
        )
    return input_ids, attention_mask, lengths, positions


def relative_from_denominator(
    delta: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(delta, dim=-1)
    result = torch.full_like(
        numerator, float("nan"), dtype=torch.float32
    )
    finite = (
        torch.isfinite(delta).all(dim=-1)
        & torch.isfinite(numerator)
        & torch.isfinite(denominator)
        & (denominator > EPSILON)
    )
    result[finite] = numerator[finite] / denominator[finite]
    return result


def did_vectors(
    value: torch.Tensor,
    state_positions: dict[str, int],
) -> torch.Tensor:
    result = []
    for answer in protocol.ANSWER_BRANCHES:
        for lexical in protocol.LEXICAL_BRANCHES:
            names = {
                (anchor, switch): state_name(
                    anchor, switch, answer, lexical
                )
                for anchor in protocol.ANCHOR_BRANCHES
                for switch in protocol.SWITCH_BRANCHES
            }
            result.append(
                (
                    value[state_positions[names[(0, 1)]]]
                    - value[state_positions[names[(0, 0)]]]
                )
                - (
                    value[state_positions[names[(1, 1)]]]
                    - value[state_positions[names[(1, 0)]]]
                )
            )
    return torch.stack(result)


def contrast_index(answer: int, lexical: int) -> int:
    return answer * len(protocol.LEXICAL_BRANCHES) + lexical


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["relation"],
            row["query_type"],
            row["path_name"],
            row["split"],
        )].append(row)
    selected = []
    for condition in protocol.RELATION_NAMES:
        for query_type in protocol.QUERY_TYPES:
            for path_name in protocol.PATH_NAMES.values():
                for split in protocol.SPLITS:
                    values = grouped[(
                        condition, query_type, path_name, split
                    )]
                    ranked = sorted(
                        values,
                        key=lambda row: hashlib.sha256(
                            (
                                "phase1073-natural|"
                                + str(row["record_id"])
                            ).encode("utf-8")
                        ).hexdigest(),
                    )
                    if not ranked:
                        raise RuntimeError(
                            "missing Phase1073 natural audit cell"
                        )
                    selected.append(ranked[0])
    expected = (
        len(protocol.RELATION_NAMES)
        * protocol.NATURAL_AUDIT_PER_CONDITION
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
        raise RuntimeError("Phase1073 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["operation_unit_id"])].append(row)
    units = []
    for operation_unit_id, values in sorted(grouped.items()):
        by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in values:
            by_task[str(row["task_family"])][str(row["state"])] = row
        if set(by_task) != set(protocol.TASK_FAMILIES):
            raise RuntimeError(
                f"incomplete task pair: {operation_unit_id}"
            )
        if any(
            set(by_task[task]) != set(protocol.STATES)
            for task in protocol.TASK_FAMILIES
        ):
            raise RuntimeError(
                f"incomplete factorial task: {operation_unit_id}"
            )
        reference = values[0]
        units.append({
            "operation_unit_id": operation_unit_id,
            "operation_condition": reference["operation_condition"],
            "base_relation": reference["base_relation"],
            "prompt_branch": reference["prompt_branch"],
            "key_alignment": reference["key_alignment"],
            "evidence_order": reference["evidence_order"],
            "query_type": reference["query_type"],
            "split": reference["split"],
            "template_index": int(reference["template_index"]),
            "replicate": int(reference["replicate"]),
            "tasks": by_task,
        })

    started = time.time()
    model = tokenizer = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")

        layers = list(get_layers(model))
        events = previous.event_definitions(len(layers))
        operation_index = {
            value: index
            for index, value in enumerate(protocol.OPERATION_CONDITIONS)
        }
        split_index = {
            value: index for index, value in enumerate(protocol.SPLITS)
        }
        query_index = {
            value: index
            for index, value in enumerate(protocol.QUERY_TYPES)
        }
        role_index = {
            value: index
            for index, value in enumerate(protocol.CAPTURE_ROLES)
        }
        conditioning_index = {
            value: index for index, value in enumerate(CONDITIONING)
        }
        response_shape = (
            len(protocol.OPERATION_CONDITIONS),
            len(protocol.SPLITS),
            len(protocol.QUERY_TYPES),
            len(events),
            len(protocol.CAPTURE_ROLES),
            len(CONDITIONING),
        )
        metric_sums = {
            metric: np.zeros(response_shape, dtype=np.float64)
            for metric in METRICS
        }
        metric_counts = {
            metric: np.zeros(response_shape, dtype=np.int32)
            for metric in METRICS
        }
        residual_metric_attempt_count = 0
        nonfinite_residual_metric_count = 0

        def accumulate(
            metric: str,
            prefix: tuple[int, int, int, int],
            values: torch.Tensor,
            conditionings: list[int],
        ) -> None:
            nonlocal residual_metric_attempt_count
            nonlocal nonfinite_residual_metric_count
            array = values.detach().float().cpu().numpy()
            finite = np.isfinite(array)
            residual_metric_attempt_count += int(array.size)
            nonfinite_residual_metric_count += int((~finite).sum())
            clean = np.where(finite, array, 0.0)
            valid = finite.astype(np.int32)
            for conditioning in conditionings:
                key = (*prefix, slice(None), conditioning)
                metric_sums[metric][key] += clean
                metric_counts[metric][key] += valid

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        behavior_records = []
        case_hit: dict[int, bool] = {}
        behavior_total = Counter()
        behavior_hit = Counter()
        behavior_greedy = Counter()
        nonfinite_candidate_count = 0
        state_order = list(protocol.STATES)
        state_positions = {
            state: index for index, state in enumerate(state_order)
        }

        capture = previous.ResidualRoleCapture(model, layers)
        capture.register()
        with torch.inference_mode():
            for unit_number, unit in enumerate(units, 1):
                task_stats: dict[
                    str, dict[int, tuple[torch.Tensor, torch.Tensor]]
                ] = {}
                local_hits: dict[str, dict[str, bool]] = {}
                shared_width = max(
                    len(unit["tasks"][task][state]["input_ids"])
                    for task in protocol.TASK_FAMILIES
                    for state in state_order
                )
                for task in protocol.TASK_FAMILIES:
                    task_rows = [
                        unit["tasks"][task][state]
                        for state in state_order
                    ]
                    (
                        input_ids,
                        attention_mask,
                        lengths,
                        positions,
                    ) = pad_rows(
                        task_rows,
                        int(pad_id),
                        device,
                        fixed_width=shared_width,
                    )
                    capture.begin(positions)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    capture.validate()
                    last_positions = (lengths - 1).to(
                        output.logits.device
                    )
                    axes = torch.arange(
                        len(task_rows), device=output.logits.device
                    )
                    last_logits = output.logits[
                        axes, last_positions, :
                    ].float()

                    task_hits = {}
                    for index, row in enumerate(task_rows):
                        logits = last_logits[index]
                        class_scores = {}
                        for class_name in ("b0", "b1"):
                            token_ids = torch.tensor(
                                row["candidate_first_token_ids"][
                                    class_name
                                ],
                                dtype=torch.long,
                                device=logits.device,
                            )
                            class_scores[class_name] = float(
                                logits[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = "b1" if expected == "b0" else "b0"
                        margin = (
                            class_scores[expected]
                            - class_scores[other]
                        )
                        finite = all(
                            math.isfinite(value)
                            for value in class_scores.values()
                        ) and math.isfinite(margin)
                        hit = bool(finite and margin > 0.0)
                        greedy_token = int(torch.argmax(logits).item())
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
                        nonfinite_candidate_count += int(not finite)
                        case_index = int(row["semantic_case_index"])
                        case_hit[case_index] = hit
                        task_hits[str(row["state"])] = hit
                        keys = (
                            ("condition", row["relation"]),
                            (
                                "condition_query",
                                row["relation"],
                                row["query_type"],
                            ),
                            (
                                "condition_path",
                                row["relation"],
                                row["path_name"],
                            ),
                            (
                                "base_task",
                                row["base_relation"],
                                row["task_family"],
                            ),
                            (
                                "base_alignment",
                                row["base_relation"],
                                row["key_alignment"],
                            ),
                        )
                        for key in keys:
                            behavior_total[key] += 1
                            behavior_hit[key] += int(hit)
                            behavior_greedy[key] += int(greedy_hit)
                        behavior_records.append({
                            "schema_version": (
                                "phase1073_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "semantic_case_index": case_index,
                            "record_id": row["record_id"],
                            "unit_id": row["unit_id"],
                            "operation_unit_id": (
                                row["operation_unit_id"]
                            ),
                            "relation": row["relation"],
                            "base_relation": row["base_relation"],
                            "task_family": row["task_family"],
                            "prompt_branch": row["prompt_branch"],
                            "key_alignment": row["key_alignment"],
                            "evidence_order": row["evidence_order"],
                            "query_type": row["query_type"],
                            "split": row["split"],
                            "state": row["state"],
                            "path_name": row["path_name"],
                            "anchor_branch": row["anchor_branch"],
                            "switch_branch": row["switch_branch"],
                            "answer_branch": row["answer_branch"],
                            "lexical_branch": row["lexical_branch"],
                            "expected_class": expected,
                            "candidate_class_scores": {
                                key: (
                                    value
                                    if math.isfinite(value)
                                    else None
                                )
                                for key, value in class_scores.items()
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
                    local_hits[task] = task_hits

                    stats = {}
                    for event_index, event in enumerate(events):
                        value = capture.values[
                            int(event["depth"])
                        ].float()
                        stats[event_index] = (
                            did_vectors(value, state_positions).detach(),
                            torch.linalg.vector_norm(
                                value, dim=-1
                            ).mean(dim=0).detach(),
                        )
                    task_stats[task] = stats
                    del (
                        output,
                        last_logits,
                        input_ids,
                        attention_mask,
                        lengths,
                        positions,
                    )
                    capture.values = {}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                operation = operation_index[
                    unit["operation_condition"]
                ]
                split = split_index[unit["split"]]
                query = query_index[unit["query_type"]]
                for event_index, _event in enumerate(events):
                    transitive_dids, transitive_reference = task_stats[
                        "transitive"
                    ][event_index]
                    key_dids, key_reference = task_stats[
                        "key_copy"
                    ][event_index]
                    operation_dids = transitive_dids - key_dids
                    shared_reference = (
                        transitive_reference + key_reference
                    ) / 2.0
                    prefix = (operation, split, query, event_index)

                    for answer in protocol.ANSWER_BRANCHES:
                        for lexical in protocol.LEXICAL_BRANCHES:
                            index = contrast_index(answer, lexical)
                            state_names = [
                                state_name(
                                    anchor,
                                    switch,
                                    answer,
                                    lexical,
                                )
                                for anchor in protocol.ANCHOR_BRANCHES
                                for switch in protocol.SWITCH_BRANCHES
                            ]
                            conditionings = [
                                conditioning_index["all"]
                            ]
                            if all(
                                local_hits[task][name]
                                for task in protocol.TASK_FAMILIES
                                for name in state_names
                            ):
                                conditionings.append(
                                    conditioning_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            accumulate(
                                "operation_contrast_relative_magnitude",
                                prefix,
                                relative_from_denominator(
                                    operation_dids[index],
                                    shared_reference,
                                ),
                                conditionings,
                            )
                            accumulate(
                                "transitive_did_relative_magnitude",
                                prefix,
                                relative_from_denominator(
                                    transitive_dids[index],
                                    transitive_reference,
                                ),
                                conditionings,
                            )
                            accumulate(
                                "key_copy_did_relative_magnitude",
                                prefix,
                                relative_from_denominator(
                                    key_dids[index],
                                    key_reference,
                                ),
                                conditionings,
                            )
                            accumulate(
                                "task_did_cosine",
                                prefix,
                                safe_cosine(
                                    transitive_dids[index],
                                    key_dids[index],
                                ),
                                conditionings,
                            )

                    for answer in protocol.ANSWER_BRANCHES:
                        conditionings = [conditioning_index["all"]]
                        state_names = [
                            state_name(anchor, switch, answer, lexical)
                            for task in protocol.TASK_FAMILIES
                            for anchor in protocol.ANCHOR_BRANCHES
                            for switch in protocol.SWITCH_BRANCHES
                            for lexical in protocol.LEXICAL_BRANCHES
                        ]
                        if all(
                            local_hits[task][name]
                            for task in protocol.TASK_FAMILIES
                            for name in set(state_names)
                        ):
                            conditionings.append(
                                conditioning_index[
                                    "behavior_conditioned"
                                ]
                            )
                        accumulate(
                            "operation_lexical_reuse_cosine",
                            prefix,
                            safe_cosine(
                                operation_dids[
                                    contrast_index(answer, 0)
                                ],
                                operation_dids[
                                    contrast_index(answer, 1)
                                ],
                            ),
                            conditionings,
                        )

                    for lexical in protocol.LEXICAL_BRANCHES:
                        conditionings = [conditioning_index["all"]]
                        state_names = [
                            state_name(anchor, switch, answer, lexical)
                            for anchor in protocol.ANCHOR_BRANCHES
                            for switch in protocol.SWITCH_BRANCHES
                            for answer in protocol.ANSWER_BRANCHES
                        ]
                        if all(
                            local_hits[task][name]
                            for task in protocol.TASK_FAMILIES
                            for name in state_names
                        ):
                            conditionings.append(
                                conditioning_index[
                                    "behavior_conditioned"
                                ]
                            )
                        accumulate(
                            "operation_answer_invariance_cosine",
                            prefix,
                            safe_cosine(
                                operation_dids[
                                    contrast_index(0, lexical)
                                ],
                                operation_dids[
                                    contrast_index(1, lexical)
                                ],
                            ),
                            conditionings,
                        )

                del task_stats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if unit_number % 10 == 0 or unit_number == len(units):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "operation_units_complete": unit_number,
                        "operation_units_total": len(units),
                    }), flush=True)

        capture.close()
        capture = None

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
        natural_semantic_first = Counter()
        natural_strict = Counter()
        natural_terminated = Counter()
        natural_tail = Counter()
        for row in natural_rows:
            index = int(row["semantic_case_index"])
            output_ids = natural_outputs[index]
            answer = text_tools.decode_content(
                tokenizer, output_ids, eos_ids
            )
            terminated = generation.terminated(output_ids, eos_ids)
            classification = previous.natural_classification(
                answer,
                list(row["acceptable_labels"]),
                terminated,
            )
            condition = str(row["relation"])
            natural_total[condition] += 1
            natural_total[(condition, row["path_name"])] += 1
            natural_semantic_first[condition] += int(
                classification["semantic_first"]
            )
            natural_semantic_first[
                (condition, row["path_name"])
            ] += int(classification["semantic_first"])
            natural_strict[condition] += int(
                classification["strict_name_only"]
            )
            natural_terminated[condition] += int(terminated)
            natural_tail[
                (condition, classification["tail_class"])
            ] += 1
            natural_records.append({
                "schema_version": (
                    "phase1073_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "record_id": row["record_id"],
                "unit_id": row["unit_id"],
                "relation": condition,
                "base_relation": row["base_relation"],
                "task_family": row["task_family"],
                "prompt_branch": row["prompt_branch"],
                "key_alignment": row["key_alignment"],
                "evidence_order": row["evidence_order"],
                "query_type": row["query_type"],
                "path_name": row["path_name"],
                "split": row["split"],
                "state": row["state"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                **classification,
            })

        response_rows = []
        for operation_name, operation in operation_index.items():
            relation, prompt_branch, alignment, order = (
                operation_name.split("::")
            )
            for split_name, split in split_index.items():
                for query_name, query in query_index.items():
                    for event_index, event in enumerate(events):
                        for role_name, role in role_index.items():
                            for conditioning_name, conditioning in (
                                conditioning_index.items()
                            ):
                                key = (
                                    operation,
                                    split,
                                    query,
                                    event_index,
                                    role,
                                    conditioning,
                                )
                                record = {
                                    "schema_version": (
                                        "phase1073_response_metric.v1"
                                    ),
                                    "phase": protocol.PHASE,
                                    "model": model_name,
                                    "operation_condition": operation_name,
                                    "base_relation": relation,
                                    "prompt_branch": prompt_branch,
                                    "key_alignment": alignment,
                                    "evidence_order": order,
                                    "split": split_name,
                                    "query_type": query_name,
                                    "event_id": event["event_id"],
                                    "depth": event["depth"],
                                    "relative_depth": event[
                                        "relative_depth"
                                    ],
                                    "role": role_name,
                                    "conditioning": conditioning_name,
                                }
                                for metric in METRICS:
                                    count = int(
                                        metric_counts[metric][key]
                                    )
                                    record[f"{metric}_count"] = count
                                    record[f"mean_{metric}"] = (
                                        float(
                                            metric_sums[metric][key]
                                            / count
                                        )
                                        if count
                                        else None
                                    )
                                response_rows.append(record)

        condition_summaries = {}
        for condition in protocol.RELATION_NAMES:
            condition_key = ("condition", condition)
            total = behavior_total[condition_key]
            natural_count = natural_total[condition]
            by_query = {}
            for query_type in protocol.QUERY_TYPES:
                key = ("condition_query", condition, query_type)
                by_query[query_type] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": rate(
                        behavior_hit[key], behavior_total[key]
                    ),
                }
            by_path = {}
            for path_name in protocol.PATH_NAMES.values():
                key = ("condition_path", condition, path_name)
                path_natural = natural_total[(condition, path_name)]
                by_path[path_name] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": rate(
                        behavior_hit[key], behavior_total[key]
                    ),
                    "natural_case_count": path_natural,
                    "semantic_first_natural_rate": rate(
                        natural_semantic_first[
                            (condition, path_name)
                        ],
                        path_natural,
                    ),
                }
            candidate_accuracy = rate(
                behavior_hit[condition_key], total
            )
            semantic_rate = rate(
                natural_semantic_first[condition], natural_count
            )
            condition_gate = bool(
                candidate_accuracy
                >= protocol.GATES["formal_candidate_accuracy_min"]
                and semantic_rate
                >= protocol.GATES["formal_semantic_first_rate_min"]
                and all(
                    value["candidate_accuracy"]
                    >= protocol.GATES["per_path_candidate_accuracy_min"]
                    for value in by_path.values()
                )
            )
            condition_summaries[condition] = {
                **protocol.parse_condition(condition),
                "case_count": total,
                "candidate_hit_count": behavior_hit[condition_key],
                "candidate_first_token_accuracy": candidate_accuracy,
                "greedy_first_token_accuracy": rate(
                    behavior_greedy[condition_key], total
                ),
                "natural_audit_case_count": natural_count,
                "semantic_first_natural_rate": semantic_rate,
                "strict_name_only_rate": rate(
                    natural_strict[condition], natural_count
                ),
                "terminated_rate": rate(
                    natural_terminated[condition], natural_count
                ),
                "tail_class_counts": {
                    tail: natural_tail[(condition, tail)]
                    for tail in (
                        "strict_name_only",
                        "name_plus_punctuation",
                        "name_plus_extra_content",
                        "wrong_first_content",
                        "empty",
                    )
                },
                "by_query": by_query,
                "by_path": by_path,
                "formal_condition_behavior_gate_passed": condition_gate,
            }

        candidate_finite_rate = (
            1.0 - nonfinite_candidate_count / len(rows)
            if rows
            else 0.0
        )
        residual_finite_rate = (
            1.0
            - nonfinite_residual_metric_count
            / residual_metric_attempt_count
            if residual_metric_attempt_count
            else 0.0
        )
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(
            atlas_root / "candidate_behavior.jsonl", behavior_records
        )
        protocol.write_jsonl(
            atlas_root / "natural_generation_audit.jsonl",
            natural_records,
        )
        protocol.write_jsonl(
            atlas_root / "response_metrics.jsonl", response_rows
        )
        summary = {
            "schema_version": "phase1073_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": len(layers),
                "d_model": int(
                    model.get_input_embeddings().weight.shape[1]
                ),
            },
            "case_count": len(rows),
            "operation_unit_count": len(units),
            "event_count": len(events),
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "candidate_finite_rate": candidate_finite_rate,
            "residual_metric_attempt_count": (
                residual_metric_attempt_count
            ),
            "nonfinite_residual_metric_count": (
                nonfinite_residual_metric_count
            ),
            "residual_metric_finite_rate": residual_finite_rate,
            "zero_norm_cosine_convention": (
                "Finite zero-norm contrasts are recorded as cosine 0; "
                "NaN is reserved for nonfinite tensors."
            ),
            "conditions": condition_summaries,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg["interpretation_limits"],
        }
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "candidate_finite_rate": candidate_finite_rate,
            "residual_metric_finite_rate": residual_finite_rate,
            "formal_behavior_conditions_passed": sum(
                row["formal_condition_behavior_gate_passed"]
                for row in condition_summaries.values()
            ),
            "formal_behavior_condition_count": len(condition_summaries),
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
