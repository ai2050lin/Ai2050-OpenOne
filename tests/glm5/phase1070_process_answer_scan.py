#!/usr/bin/env python3
"""Run the Phase1070 FP16 process/answer/surface orthogonal atlas."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
import phase1070_process_answer_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 1,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12
CONDITIONS = ("all", "behavior_conditioned")
METRICS = (
    "main_switch_relative_magnitude",
    "anchor_switch_control_relative_magnitude",
    "process_did_relative_magnitude",
    "process_lexical_reuse_cosine",
    "process_answer_invariance_cosine",
    "answer_relative_magnitude",
    "answer_lexical_reuse_cosine",
    "answer_path_invariance_cosine",
    "surface_relative_magnitude",
    "process_answer_absolute_cosine",
)


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def gather_candidate_rows(
    model,
    units: list[dict[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise RuntimeError("model has no output embedding weight")
    weight = output.weight
    if getattr(weight, "is_meta", False):
        hook = getattr(output, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        try:
            weight = (
                weights_map["weight"]
                if weights_map is not None
                else None
            )
        except (KeyError, TypeError):
            weight = None
        if weight is None or getattr(weight, "is_meta", False):
            raise RuntimeError(
                "unable to resolve offloaded output embedding weight"
            )
    all_ids = sorted({
        int(token_id)
        for unit in units
        for class_name in ("b0", "b1")
        for token_id in unit["states"][
            protocol.STATES[0]
        ]["candidate_first_token_ids"][class_name]
    })
    index = torch.tensor(
        all_ids, dtype=torch.long, device=weight.device
    )
    selected = (
        weight.index_select(0, index)
        .detach()
        .float()
        .cpu()
    )
    by_id = {
        token_id: selected[position]
        for position, token_id in enumerate(all_ids)
    }
    result = {}
    for unit in units:
        row = unit["states"][protocol.STATES[0]]
        result[unit["unit_id"]] = {
            class_name: torch.stack([
                by_id[int(token_id)]
                for token_id in row[
                    "candidate_first_token_ids"
                ][class_name]
            ]).mean(dim=0)
            for class_name in ("b0", "b1")
        }
    return result


def state_name(
    anchor: int,
    switch: int,
    answer: int,
    lexical: int,
) -> str:
    return f"a{anchor}_b{switch}_y{answer}_l{lexical}"


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    denominator = left_norm * right_norm
    result = torch.full_like(
        denominator, float("nan"), dtype=torch.float32
    )
    finite_inputs = (
        torch.isfinite(left).all(dim=-1)
        & torch.isfinite(right).all(dim=-1)
        & torch.isfinite(denominator)
    )
    # A finite zero contrast is an expected negative-control outcome (most
    # notably the exact embedding-level process DiD), not numerical failure.
    # We record its cosine as zero and reserve NaN for genuinely nonfinite
    # tensors.
    result[finite_inputs] = 0.0
    valid = finite_inputs & (denominator > EPSILON)
    result[valid] = (
        (left[valid] * right[valid]).sum(dim=-1)
        / denominator[valid]
    )
    return result


def relative_magnitude(
    delta: torch.Tensor,
    references: list[torch.Tensor],
) -> torch.Tensor:
    denominator = sum(
        torch.linalg.vector_norm(value, dim=-1)
        for value in references
    ) / len(references)
    numerator = torch.linalg.vector_norm(delta, dim=-1)
    result = torch.full_like(
        numerator, float("nan"), dtype=torch.float32
    )
    finite = (
        torch.isfinite(delta).all(dim=-1)
        & torch.stack([
            torch.isfinite(value).all(dim=-1)
            for value in references
        ]).all(dim=0)
        & torch.isfinite(denominator)
        & torch.isfinite(numerator)
        & (denominator > EPSILON)
    )
    result[finite] = numerator[finite] / denominator[finite]
    return result


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["relation"],
            row["path_name"],
            int(row["answer_branch"]),
            int(row["lexical_branch"]),
            row["split"],
        )].append(row)
    for relation in protocol.RELATION_NAMES:
        for path_name in protocol.PATH_NAMES.values():
            for answer in protocol.ANSWER_BRANCHES:
                for lexical in protocol.LEXICAL_BRANCHES:
                    for split in protocol.SPLITS:
                        values = grouped[(
                            relation,
                            path_name,
                            answer,
                            lexical,
                            split,
                        )]
                        ranked = sorted(
                            values,
                            key=lambda row: hashlib.sha256(
                                (
                                    "phase1070-natural|"
                                    + str(row["record_id"])
                                ).encode("utf-8")
                            ).hexdigest(),
                        )
                        selected.extend(ranked[:3])
    expected = (
        len(protocol.RELATION_NAMES)
        * len(protocol.PATH_NAMES)
        * protocol.NATURAL_AUDIT_PER_PATH
    )
    if len(selected) != expected:
        raise RuntimeError(
            f"natural selection drift: {len(selected)} != {expected}"
        )
    return selected


def safe_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1070 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["unit_id"])].append(row)
    units = []
    for unit_id, values in sorted(by_unit.items()):
        by_state = {str(row["state"]): row for row in values}
        if set(by_state) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit: {unit_id}")
        reference = values[0]
        units.append({
            "unit_id": unit_id,
            "relation": reference["relation"],
            "query_type": reference["query_type"],
            "layout": reference["layout"],
            "split": reference["split"],
            "template_index": int(reference["template_index"]),
            "replicate": int(reference["replicate"]),
            "mismatch_unit_id": reference["mismatch_unit_id"],
            "states": by_state,
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
        relation_index = {
            value: index
            for index, value in enumerate(protocol.RELATION_NAMES)
        }
        split_index = {
            value: index
            for index, value in enumerate(protocol.SPLITS)
        }
        query_index = {
            value: index
            for index, value in enumerate(protocol.QUERY_TYPES)
        }
        role_index = {
            value: index
            for index, value in enumerate(protocol.CAPTURE_ROLES)
        }
        condition_index = {
            value: index
            for index, value in enumerate(CONDITIONS)
        }
        response_shape = (
            len(protocol.RELATION_NAMES),
            len(protocol.SPLITS),
            len(protocol.QUERY_TYPES),
            len(events),
            len(protocol.CAPTURE_ROLES),
            len(CONDITIONS),
        )
        metric_sums = {
            metric: np.zeros(response_shape, dtype=np.float64)
            for metric in METRICS
        }
        metric_counts = {
            metric: np.zeros(response_shape, dtype=np.int32)
            for metric in METRICS
        }
        readout_shape = (
            len(protocol.RELATION_NAMES),
            len(protocol.SPLITS),
            len(protocol.QUERY_TYPES),
            len(events),
            len(CONDITIONS),
        )
        answer_shift_sum = np.zeros(readout_shape, dtype=np.float64)
        mismatch_shift_sum = np.zeros(readout_shape, dtype=np.float64)
        answer_positive = np.zeros(readout_shape, dtype=np.int32)
        mismatch_positive = np.zeros(readout_shape, dtype=np.int32)
        answer_axis_cosine_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        mismatch_axis_cosine_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        answer_readout_count = np.zeros(
            readout_shape, dtype=np.int32
        )
        process_did_abs_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        process_axis_abs_cosine_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        process_readout_count = np.zeros(
            readout_shape, dtype=np.int32
        )

        residual_metric_attempt_count = 0
        nonfinite_residual_metric_count = 0
        internal_readout_attempt_count = 0
        nonfinite_internal_readout_count = 0

        def accumulate(
            metric: str,
            prefix: tuple[int, int, int, int],
            values: torch.Tensor,
            conditions: list[int],
        ) -> None:
            nonlocal residual_metric_attempt_count
            nonlocal nonfinite_residual_metric_count
            array = values.detach().float().cpu().numpy()
            finite = np.isfinite(array)
            residual_metric_attempt_count += int(array.size)
            nonfinite_residual_metric_count += int((~finite).sum())
            clean = np.where(finite, array, 0.0)
            valid = finite.astype(np.int32)
            for condition in conditions:
                key = (*prefix, slice(None), condition)
                metric_sums[metric][key] += clean
                metric_counts[metric][key] += valid

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        candidates_cpu = gather_candidate_rows(model, units)
        coordinate_cache = previous.CandidateCoordinateCache(
            candidates_cpu
        )
        final_norm = previous.final_norm_module(model)
        norm_device, norm_dtype = previous.module_device_dtype(
            final_norm
        )

        behavior_records = []
        case_hit: dict[int, bool] = {}
        behavior_total = Counter()
        behavior_hit = Counter()
        behavior_greedy = Counter()
        valid_process_quads = Counter()
        valid_answer_pairs = Counter()
        complete_units = Counter()
        nonfinite_candidate_count = 0

        capture = previous.ResidualRoleCapture(model, layers)
        capture.register()
        state_order = list(protocol.STATES)
        with torch.inference_mode():
            for batch_start in range(
                0,
                len(units),
                UNIT_BATCH_SIZE[model_name],
            ):
                batch_units = units[
                    batch_start:
                    batch_start + UNIT_BATCH_SIZE[model_name]
                ]
                forward_rows = [
                    unit["states"][state]
                    for unit in batch_units
                    for state in state_order
                ]
                (
                    input_ids,
                    attention_mask,
                    lengths,
                    positions,
                ) = pad_rows(forward_rows, int(pad_id), device)
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                capture.validate()
                last_positions = (
                    lengths - 1
                ).to(output.logits.device)
                batch_axis = torch.arange(
                    output.logits.shape[0],
                    device=output.logits.device,
                )
                last_logits = output.logits[
                    batch_axis, last_positions, :
                ].float()
                del output

                for unit_offset, unit in enumerate(batch_units):
                    offset = unit_offset * len(state_order)
                    local_hits = {}
                    for local_index, state in enumerate(state_order):
                        row = unit["states"][state]
                        logits = last_logits[offset + local_index]
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
                        nonfinite_candidate_count += int(not finite)
                        hit = bool(finite and margin > 0.0)
                        greedy_token = int(torch.argmax(logits).item())
                        greedy_hit = greedy_token in set(
                            int(value)
                            for value in row[
                                "candidate_first_token_ids"
                            ][expected]
                        )
                        case_index = int(row["semantic_case_index"])
                        case_hit[case_index] = hit
                        local_hits[state] = hit
                        keys = (
                            ("relation", row["relation"]),
                            (
                                "relation_query",
                                row["relation"],
                                row["query_type"],
                            ),
                            (
                                "relation_path",
                                row["relation"],
                                row["path_name"],
                            ),
                        )
                        for key in keys:
                            behavior_total[key] += 1
                            behavior_hit[key] += int(hit)
                            behavior_greedy[key] += int(greedy_hit)
                        behavior_records.append({
                            "schema_version": (
                                "phase1070_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "semantic_case_index": case_index,
                            "unit_id": row["unit_id"],
                            "relation": row["relation"],
                            "query_type": row["query_type"],
                            "layout": row["layout"],
                            "split": row["split"],
                            "state": state,
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
                                margin
                                if math.isfinite(margin)
                                else None
                            ),
                            "nonfinite_candidate": not finite,
                            "candidate_hit": hit,
                            "greedy_first_token_id": greedy_token,
                            "greedy_first_token_text": (
                                tokenizer.decode([greedy_token])
                            ),
                            "greedy_first_token_hit": greedy_hit,
                        })
                    for answer in protocol.ANSWER_BRANCHES:
                        for lexical in protocol.LEXICAL_BRANCHES:
                            names = [
                                state_name(
                                    anchor,
                                    switch,
                                    answer,
                                    lexical,
                                )
                                for anchor in protocol.ANCHOR_BRANCHES
                                for switch in protocol.SWITCH_BRANCHES
                            ]
                            if all(local_hits[name] for name in names):
                                valid_process_quads[
                                    ("relation", unit["relation"])
                                ] += 1
                    for anchor in protocol.ANCHOR_BRANCHES:
                        for switch in protocol.SWITCH_BRANCHES:
                            for lexical in protocol.LEXICAL_BRANCHES:
                                names = [
                                    state_name(
                                        anchor,
                                        switch,
                                        answer,
                                        lexical,
                                    )
                                    for answer in protocol.ANSWER_BRANCHES
                                ]
                                if all(local_hits[name] for name in names):
                                    valid_answer_pairs[
                                        ("relation", unit["relation"])
                                    ] += 1
                    if all(local_hits.values()):
                        complete_units[
                            ("relation", unit["relation"])
                        ] += 1

                for event_index, event in enumerate(events):
                    value = capture.values[
                        int(event["depth"])
                    ].float()
                    answer_role = role_index["answer_boundary"]
                    answer_values = value[:, answer_role, :]
                    normed_answer = final_norm(
                        answer_values.to(
                            device=norm_device,
                            dtype=norm_dtype,
                        )
                    ).float()
                    for unit_offset, unit in enumerate(batch_units):
                        offset = unit_offset * len(state_order)
                        relation = relation_index[unit["relation"]]
                        split = split_index[unit["split"]]
                        query = query_index[unit["query_type"]]
                        prefix = (
                            relation,
                            split,
                            query,
                            event_index,
                        )
                        states = {
                            state: value[offset + local_index]
                            for local_index, state in enumerate(
                                state_order
                            )
                        }

                        process_dids: dict[
                            tuple[int, int], torch.Tensor
                        ] = {}
                        answer_deltas: dict[
                            tuple[int, int, int], torch.Tensor
                        ] = {}
                        for answer in protocol.ANSWER_BRANCHES:
                            for lexical in protocol.LEXICAL_BRANCHES:
                                names = {
                                    (anchor, switch): state_name(
                                        anchor,
                                        switch,
                                        answer,
                                        lexical,
                                    )
                                    for anchor in protocol.ANCHOR_BRANCHES
                                    for switch in protocol.SWITCH_BRANCHES
                                }
                                main = (
                                    states[names[(0, 1)]]
                                    - states[names[(0, 0)]]
                                )
                                control = (
                                    states[names[(1, 1)]]
                                    - states[names[(1, 0)]]
                                )
                                did = main - control
                                process_dids[(answer, lexical)] = did
                                hit_names = list(names.values())
                                conditions = [condition_index["all"]]
                                if all(
                                    case_hit[int(unit["states"][name][
                                        "semantic_case_index"
                                    ])]
                                    for name in hit_names
                                ):
                                    conditions.append(
                                        condition_index[
                                            "behavior_conditioned"
                                        ]
                                    )
                                accumulate(
                                    "main_switch_relative_magnitude",
                                    prefix,
                                    relative_magnitude(
                                        main,
                                        [
                                            states[names[(0, 1)]],
                                            states[names[(0, 0)]],
                                        ],
                                    ),
                                    conditions,
                                )
                                accumulate(
                                    "anchor_switch_control_relative_magnitude",
                                    prefix,
                                    relative_magnitude(
                                        control,
                                        [
                                            states[names[(1, 1)]],
                                            states[names[(1, 0)]],
                                        ],
                                    ),
                                    conditions,
                                )
                                accumulate(
                                    "process_did_relative_magnitude",
                                    prefix,
                                    relative_magnitude(
                                        did,
                                        [
                                            states[name]
                                            for name in hit_names
                                        ],
                                    ),
                                    conditions,
                                )

                        for answer in protocol.ANSWER_BRANCHES:
                            conditions = [condition_index["all"]]
                            names = [
                                state_name(a, b, answer, lexical)
                                for a in protocol.ANCHOR_BRANCHES
                                for b in protocol.SWITCH_BRANCHES
                                for lexical in protocol.LEXICAL_BRANCHES
                            ]
                            if all(
                                case_hit[int(unit["states"][name][
                                    "semantic_case_index"
                                ])]
                                for name in names
                            ):
                                conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            accumulate(
                                "process_lexical_reuse_cosine",
                                prefix,
                                safe_cosine(
                                    process_dids[(answer, 0)],
                                    process_dids[(answer, 1)],
                                ),
                                conditions,
                            )
                        for lexical in protocol.LEXICAL_BRANCHES:
                            conditions = [condition_index["all"]]
                            names = [
                                state_name(a, b, answer, lexical)
                                for a in protocol.ANCHOR_BRANCHES
                                for b in protocol.SWITCH_BRANCHES
                                for answer in protocol.ANSWER_BRANCHES
                            ]
                            if all(
                                case_hit[int(unit["states"][name][
                                    "semantic_case_index"
                                ])]
                                for name in names
                            ):
                                conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            accumulate(
                                "process_answer_invariance_cosine",
                                prefix,
                                safe_cosine(
                                    process_dids[(0, lexical)],
                                    process_dids[(1, lexical)],
                                ),
                                conditions,
                            )

                        for anchor in protocol.ANCHOR_BRANCHES:
                            for switch in protocol.SWITCH_BRANCHES:
                                for lexical in protocol.LEXICAL_BRANCHES:
                                    left_name = state_name(
                                        anchor, switch, 0, lexical
                                    )
                                    right_name = state_name(
                                        anchor, switch, 1, lexical
                                    )
                                    delta = (
                                        states[right_name]
                                        - states[left_name]
                                    )
                                    answer_deltas[
                                        (anchor, switch, lexical)
                                    ] = delta
                                    conditions = [
                                        condition_index["all"]
                                    ]
                                    if all(
                                        case_hit[int(unit["states"][name][
                                            "semantic_case_index"
                                        ])]
                                        for name in (
                                            left_name,
                                            right_name,
                                        )
                                    ):
                                        conditions.append(
                                            condition_index[
                                                "behavior_conditioned"
                                            ]
                                        )
                                    accumulate(
                                        "answer_relative_magnitude",
                                        prefix,
                                        relative_magnitude(
                                            delta,
                                            [
                                                states[left_name],
                                                states[right_name],
                                            ],
                                        ),
                                        conditions,
                                    )

                        for anchor in protocol.ANCHOR_BRANCHES:
                            for switch in protocol.SWITCH_BRANCHES:
                                names = [
                                    state_name(
                                        anchor, switch, answer, lexical
                                    )
                                    for answer in protocol.ANSWER_BRANCHES
                                    for lexical in protocol.LEXICAL_BRANCHES
                                ]
                                conditions = [condition_index["all"]]
                                if all(
                                    case_hit[int(unit["states"][name][
                                        "semantic_case_index"
                                    ])]
                                    for name in names
                                ):
                                    conditions.append(
                                        condition_index[
                                            "behavior_conditioned"
                                        ]
                                    )
                                accumulate(
                                    "answer_lexical_reuse_cosine",
                                    prefix,
                                    safe_cosine(
                                        answer_deltas[
                                            (anchor, switch, 0)
                                        ],
                                        answer_deltas[
                                            (anchor, switch, 1)
                                        ],
                                    ),
                                    conditions,
                                )

                        paths = list(protocol.PATH_NAMES)
                        for lexical in protocol.LEXICAL_BRANCHES:
                            names = [
                                state_name(a, b, answer, lexical)
                                for a, b in paths
                                for answer in protocol.ANSWER_BRANCHES
                            ]
                            conditions = [condition_index["all"]]
                            if all(
                                case_hit[int(unit["states"][name][
                                    "semantic_case_index"
                                ])]
                                for name in names
                            ):
                                conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            for left_path, right_path in (
                                itertools.combinations(paths, 2)
                            ):
                                accumulate(
                                    "answer_path_invariance_cosine",
                                    prefix,
                                    safe_cosine(
                                        answer_deltas[
                                            (*left_path, lexical)
                                        ],
                                        answer_deltas[
                                            (*right_path, lexical)
                                        ],
                                    ),
                                    conditions,
                                )

                        for anchor in protocol.ANCHOR_BRANCHES:
                            for switch in protocol.SWITCH_BRANCHES:
                                for answer in protocol.ANSWER_BRANCHES:
                                    left_name = state_name(
                                        anchor, switch, answer, 0
                                    )
                                    right_name = state_name(
                                        anchor, switch, answer, 1
                                    )
                                    delta = (
                                        states[right_name]
                                        - states[left_name]
                                    )
                                    conditions = [
                                        condition_index["all"]
                                    ]
                                    if all(
                                        case_hit[int(unit["states"][name][
                                            "semantic_case_index"
                                        ])]
                                        for name in (
                                            left_name,
                                            right_name,
                                        )
                                    ):
                                        conditions.append(
                                            condition_index[
                                                "behavior_conditioned"
                                            ]
                                        )
                                    accumulate(
                                        "surface_relative_magnitude",
                                        prefix,
                                        relative_magnitude(
                                            delta,
                                            [
                                                states[left_name],
                                                states[right_name],
                                            ],
                                        ),
                                        conditions,
                                    )

                        for answer in protocol.ANSWER_BRANCHES:
                            for lexical in protocol.LEXICAL_BRANCHES:
                                did = process_dids[(answer, lexical)]
                                process_names = [
                                    state_name(
                                        a, b, answer, lexical
                                    )
                                    for a, b in paths
                                ]
                                for anchor, switch in paths:
                                    answer_names = [
                                        state_name(
                                            anchor,
                                            switch,
                                            value,
                                            lexical,
                                        )
                                        for value in protocol.ANSWER_BRANCHES
                                    ]
                                    conditions = [
                                        condition_index["all"]
                                    ]
                                    if all(
                                        case_hit[int(unit["states"][name][
                                            "semantic_case_index"
                                        ])]
                                        for name in (
                                            process_names
                                            + answer_names
                                        )
                                    ):
                                        conditions.append(
                                            condition_index[
                                                "behavior_conditioned"
                                            ]
                                        )
                                    accumulate(
                                        "process_answer_absolute_cosine",
                                        prefix,
                                        safe_cosine(
                                            did,
                                            answer_deltas[
                                                (
                                                    anchor,
                                                    switch,
                                                    lexical,
                                                )
                                            ],
                                        ).abs(),
                                        conditions,
                                    )

                        matched_candidates = coordinate_cache.get(
                            unit["unit_id"], norm_device
                        )
                        mismatch_candidates = coordinate_cache.get(
                            str(unit["mismatch_unit_id"]),
                            norm_device,
                        )
                        normed_states = {
                            state: normed_answer[
                                offset + local_index
                            ]
                            for local_index, state in enumerate(
                                state_order
                            )
                        }
                        matched_margins = {
                            state: previous.class_margin(
                                state_value, matched_candidates
                            )
                            for state, state_value in normed_states.items()
                        }
                        mismatch_margins = {
                            state: previous.class_margin(
                                state_value, mismatch_candidates
                            )
                            for state, state_value in normed_states.items()
                        }
                        matched_axis = (
                            matched_candidates["b1"]
                            - matched_candidates["b0"]
                        )
                        mismatch_axis = (
                            mismatch_candidates["b1"]
                            - mismatch_candidates["b0"]
                        )
                        raw_device = states[protocol.STATES[0]].device
                        matched_axis_raw = matched_axis.to(raw_device)
                        mismatch_axis_raw = mismatch_axis.to(raw_device)

                        for anchor in protocol.ANCHOR_BRANCHES:
                            for switch in protocol.SWITCH_BRANCHES:
                                for lexical in protocol.LEXICAL_BRANCHES:
                                    left_name = state_name(
                                        anchor, switch, 0, lexical
                                    )
                                    right_name = state_name(
                                        anchor, switch, 1, lexical
                                    )
                                    matched_shift = safe_scalar(
                                        matched_margins[right_name]
                                        - matched_margins[left_name]
                                    )
                                    mismatch_shift = safe_scalar(
                                        mismatch_margins[right_name]
                                        - mismatch_margins[left_name]
                                    )
                                    answer_delta = (
                                        states[right_name]
                                        - states[left_name]
                                    )[answer_role]
                                    matched_cos = safe_scalar(
                                        safe_cosine(
                                            answer_delta.unsqueeze(0),
                                            matched_axis_raw.unsqueeze(0),
                                        )[0]
                                    )
                                    mismatch_cos = safe_scalar(
                                        safe_cosine(
                                            answer_delta.unsqueeze(0),
                                            mismatch_axis_raw.unsqueeze(0),
                                        )[0]
                                    )
                                    values = (
                                        matched_shift,
                                        mismatch_shift,
                                        matched_cos,
                                        mismatch_cos,
                                    )
                                    internal_readout_attempt_count += len(
                                        values
                                    )
                                    if not all(
                                        math.isfinite(value)
                                        for value in values
                                    ):
                                        nonfinite_internal_readout_count += sum(
                                            not math.isfinite(value)
                                            for value in values
                                        )
                                        continue
                                    conditions = [
                                        condition_index["all"]
                                    ]
                                    if all(
                                        case_hit[int(unit["states"][name][
                                            "semantic_case_index"
                                        ])]
                                        for name in (
                                            left_name,
                                            right_name,
                                        )
                                    ):
                                        conditions.append(
                                            condition_index[
                                                "behavior_conditioned"
                                            ]
                                        )
                                    for condition in conditions:
                                        key = (
                                            relation,
                                            split,
                                            query,
                                            event_index,
                                            condition,
                                        )
                                        answer_shift_sum[key] += (
                                            matched_shift
                                        )
                                        mismatch_shift_sum[key] += (
                                            mismatch_shift
                                        )
                                        answer_positive[key] += int(
                                            matched_shift > 0.0
                                        )
                                        mismatch_positive[key] += int(
                                            mismatch_shift > 0.0
                                        )
                                        answer_axis_cosine_sum[key] += (
                                            matched_cos
                                        )
                                        mismatch_axis_cosine_sum[key] += (
                                            mismatch_cos
                                        )
                                        answer_readout_count[key] += 1

                        for answer in protocol.ANSWER_BRANCHES:
                            for lexical in protocol.LEXICAL_BRANCHES:
                                names = {
                                    (anchor, switch): state_name(
                                        anchor,
                                        switch,
                                        answer,
                                        lexical,
                                    )
                                    for anchor in protocol.ANCHOR_BRANCHES
                                    for switch in protocol.SWITCH_BRANCHES
                                }
                                did_margin = safe_scalar(
                                    (
                                        matched_margins[names[(0, 1)]]
                                        - matched_margins[names[(0, 0)]]
                                    )
                                    - (
                                        matched_margins[names[(1, 1)]]
                                        - matched_margins[names[(1, 0)]]
                                    )
                                )
                                did_vector = (
                                    (
                                        states[names[(0, 1)]]
                                        - states[names[(0, 0)]]
                                    )
                                    - (
                                        states[names[(1, 1)]]
                                        - states[names[(1, 0)]]
                                    )
                                )[answer_role]
                                axis_cos = safe_scalar(
                                    safe_cosine(
                                        did_vector.unsqueeze(0),
                                        matched_axis_raw.unsqueeze(0),
                                    )[0]
                                )
                                values = (did_margin, axis_cos)
                                internal_readout_attempt_count += len(values)
                                if not all(
                                    math.isfinite(value)
                                    for value in values
                                ):
                                    nonfinite_internal_readout_count += sum(
                                        not math.isfinite(value)
                                        for value in values
                                    )
                                    continue
                                state_names = list(names.values())
                                conditions = [condition_index["all"]]
                                if all(
                                    case_hit[int(unit["states"][name][
                                        "semantic_case_index"
                                    ])]
                                    for name in state_names
                                ):
                                    conditions.append(
                                        condition_index[
                                            "behavior_conditioned"
                                        ]
                                    )
                                for condition in conditions:
                                    key = (
                                        relation,
                                        split,
                                        query,
                                        event_index,
                                        condition,
                                    )
                                    process_did_abs_sum[key] += abs(
                                        did_margin
                                    )
                                    process_axis_abs_cosine_sum[key] += abs(
                                        axis_cos
                                    )
                                    process_readout_count[key] += 1
                    del normed_answer, answer_values, value

                del (
                    last_logits,
                    input_ids,
                    attention_mask,
                    lengths,
                    positions,
                )
                capture.values = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = min(
                    batch_start + len(batch_units), len(units)
                )
                if completed % 20 == 0 or completed == len(units):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "units_complete": completed,
                        "units_total": len(units),
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
            relation = str(row["relation"])
            natural_total[relation] += 1
            natural_total[(relation, row["path_name"])] += 1
            natural_semantic_first[relation] += int(
                classification["semantic_first"]
            )
            natural_semantic_first[
                (relation, row["path_name"])
            ] += int(classification["semantic_first"])
            natural_strict[relation] += int(
                classification["strict_name_only"]
            )
            natural_terminated[relation] += int(terminated)
            natural_tail[
                (relation, classification["tail_class"])
            ] += 1
            natural_records.append({
                "schema_version": (
                    "phase1070_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "unit_id": row["unit_id"],
                "relation": relation,
                "query_type": row["query_type"],
                "layout": row["layout"],
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
        for relation_name, relation in relation_index.items():
            for split_name, split in split_index.items():
                for query_name, query in query_index.items():
                    for event_index, event in enumerate(events):
                        for role_name, role in role_index.items():
                            for condition_name, condition in (
                                condition_index.items()
                            ):
                                record = {
                                    "schema_version": (
                                        "phase1070_response_metric.v1"
                                    ),
                                    "phase": protocol.PHASE,
                                    "model": model_name,
                                    "relation": relation_name,
                                    "split": split_name,
                                    "query_type": query_name,
                                    "event_id": event["event_id"],
                                    "depth": event["depth"],
                                    "relative_depth": event[
                                        "relative_depth"
                                    ],
                                    "role": role_name,
                                    "conditioning": condition_name,
                                }
                                key = (
                                    relation,
                                    split,
                                    query,
                                    event_index,
                                    role,
                                    condition,
                                )
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
                                        if count else None
                                    )
                                response_rows.append(record)

        readout_rows = []
        for relation_name, relation in relation_index.items():
            for split_name, split in split_index.items():
                for query_name, query in query_index.items():
                    for event_index, event in enumerate(events):
                        for condition_name, condition in (
                            condition_index.items()
                        ):
                            key = (
                                relation,
                                split,
                                query,
                                event_index,
                                condition,
                            )
                            answer_count = int(
                                answer_readout_count[key]
                            )
                            process_count = int(
                                process_readout_count[key]
                            )
                            mean_answer_abs = (
                                abs(answer_shift_sum[key] / answer_count)
                                if answer_count else None
                            )
                            mean_process_abs = (
                                process_did_abs_sum[key] / process_count
                                if process_count else None
                            )
                            readout_rows.append({
                                "schema_version": (
                                    "phase1070_readout_metric.v1"
                                ),
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "relation": relation_name,
                                "split": split_name,
                                "query_type": query_name,
                                "event_id": event["event_id"],
                                "depth": event["depth"],
                                "relative_depth": event[
                                    "relative_depth"
                                ],
                                "role": "answer_boundary",
                                "conditioning": condition_name,
                                "answer_observation_count": answer_count,
                                "mean_matched_answer_shift": (
                                    answer_shift_sum[key] / answer_count
                                    if answer_count else None
                                ),
                                "mean_mismatched_answer_shift": (
                                    mismatch_shift_sum[key] / answer_count
                                    if answer_count else None
                                ),
                                "matched_answer_positive_rate": (
                                    answer_positive[key] / answer_count
                                    if answer_count else None
                                ),
                                "mismatched_answer_positive_rate": (
                                    mismatch_positive[key] / answer_count
                                    if answer_count else None
                                ),
                                "positive_rate_gap": (
                                    (
                                        answer_positive[key]
                                        - mismatch_positive[key]
                                    ) / answer_count
                                    if answer_count else None
                                ),
                                "mean_matched_answer_axis_cosine": (
                                    answer_axis_cosine_sum[key]
                                    / answer_count
                                    if answer_count else None
                                ),
                                "mean_mismatched_answer_axis_cosine": (
                                    mismatch_axis_cosine_sum[key]
                                    / answer_count
                                    if answer_count else None
                                ),
                                "process_observation_count": process_count,
                                "mean_absolute_process_did_readout": (
                                    mean_process_abs
                                ),
                                "mean_absolute_process_axis_cosine": (
                                    process_axis_abs_cosine_sum[key]
                                    / process_count
                                    if process_count else None
                                ),
                                "absolute_process_to_answer_readout_ratio": (
                                    mean_process_abs / mean_answer_abs
                                    if (
                                        mean_process_abs is not None
                                        and mean_answer_abs is not None
                                        and mean_answer_abs > EPSILON
                                    )
                                    else None
                                ),
                            })

        candidate_finite_rate = (
            1.0 - nonfinite_candidate_count / len(rows)
            if rows else 0.0
        )
        residual_finite_rate = (
            1.0
            - nonfinite_residual_metric_count
            / residual_metric_attempt_count
            if residual_metric_attempt_count else 0.0
        )
        internal_finite_rate = (
            1.0
            - nonfinite_internal_readout_count
            / internal_readout_attempt_count
            if internal_readout_attempt_count else 0.0
        )
        relation_summaries = {}
        for relation in protocol.RELATION_NAMES:
            relation_key = ("relation", relation)
            total = behavior_total[relation_key]
            natural_count = natural_total[relation]
            by_query = {}
            for query_type in protocol.QUERY_TYPES:
                key = ("relation_query", relation, query_type)
                by_query[query_type] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key] else 0.0
                    ),
                }
            by_path = {}
            for path_name in protocol.PATH_NAMES.values():
                key = ("relation_path", relation, path_name)
                path_natural_count = natural_total[
                    (relation, path_name)
                ]
                by_path[path_name] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key] else 0.0
                    ),
                    "natural_case_count": path_natural_count,
                    "semantic_first_natural_rate": (
                        natural_semantic_first[
                            (relation, path_name)
                        ] / path_natural_count
                        if path_natural_count else 0.0
                    ),
                }
            semantic_first_rate = (
                natural_semantic_first[relation] / natural_count
                if natural_count else 0.0
            )
            strong_behavior = bool(
                total
                and behavior_hit[relation_key] / total
                >= prereg["gates"][
                    "candidate_first_token_accuracy_min"
                ]
                and semantic_first_rate
                >= prereg["gates"][
                    "semantic_first_natural_rate_min"
                ]
                and all(
                    row["candidate_accuracy"]
                    >= prereg["gates"][
                        "per_query_candidate_accuracy_min"
                    ]
                    for row in by_query.values()
                )
                and all(
                    row["candidate_accuracy"]
                    >= prereg["gates"][
                        "per_path_candidate_accuracy_min"
                    ]
                    for row in by_path.values()
                )
                and valid_process_quads[relation_key]
                >= prereg["gates"][
                    "valid_process_quad_per_relation_min"
                ]
                and valid_answer_pairs[relation_key]
                >= prereg["gates"][
                    "valid_answer_pair_per_relation_min"
                ]
                and complete_units[relation_key]
                >= prereg["gates"][
                    "complete_factorial_unit_per_relation_min"
                ]
            )
            relation_summaries[relation] = {
                "case_count": total,
                "candidate_hit_count": behavior_hit[relation_key],
                "candidate_first_token_accuracy": (
                    behavior_hit[relation_key] / total
                    if total else 0.0
                ),
                "greedy_first_token_accuracy": (
                    behavior_greedy[relation_key] / total
                    if total else 0.0
                ),
                "valid_process_quad_count": (
                    valid_process_quads[relation_key]
                ),
                "valid_answer_pair_count": (
                    valid_answer_pairs[relation_key]
                ),
                "complete_factorial_unit_count": (
                    complete_units[relation_key]
                ),
                "natural_audit_case_count": natural_count,
                "semantic_first_natural_rate": semantic_first_rate,
                "strict_name_only_rate": (
                    natural_strict[relation] / natural_count
                    if natural_count else 0.0
                ),
                "terminated_rate": (
                    natural_terminated[relation] / natural_count
                    if natural_count else 0.0
                ),
                "tail_class_counts": {
                    tail: natural_tail[(relation, tail)]
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
                "strong_behavior_gate_passed": strong_behavior,
            }

        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(
            atlas_root / "candidate_behavior.jsonl",
            behavior_records,
        )
        protocol.write_jsonl(
            atlas_root / "natural_generation_audit.jsonl",
            natural_records,
        )
        protocol.write_jsonl(
            atlas_root / "response_metrics.jsonl",
            response_rows,
        )
        protocol.write_jsonl(
            atlas_root / "local_readout_metrics.jsonl",
            readout_rows,
        )
        summary = {
            "schema_version": "phase1070_model_summary.v1",
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
                "final_norm_device": str(norm_device),
                "final_norm_dtype": str(norm_dtype),
            },
            "case_count": len(rows),
            "unit_count": len(units),
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
            "internal_readout_attempt_count": (
                internal_readout_attempt_count
            ),
            "nonfinite_internal_readout_count": (
                nonfinite_internal_readout_count
            ),
            "internal_readout_finite_rate": internal_finite_rate,
            "zero_norm_cosine_convention": (
                "Finite zero-norm contrasts are recorded as cosine 0; "
                "NaN is reserved for nonfinite model tensors."
            ),
            "relations": relation_summaries,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg[
                "interpretation_limits"
            ],
        }
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "relations": {
                relation: {
                    "candidate_accuracy": row[
                        "candidate_first_token_accuracy"
                    ],
                    "semantic_first": row[
                        "semantic_first_natural_rate"
                    ],
                    "strict": row["strict_name_only_rate"],
                    "strong_behavior": row[
                        "strong_behavior_gate_passed"
                    ],
                }
                for relation, row in relation_summaries.items()
            },
            "candidate_finite_rate": candidate_finite_rate,
            "residual_metric_finite_rate": residual_finite_rate,
            "internal_readout_finite_rate": internal_finite_rate,
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
