#!/usr/bin/env python3
"""Run one Phase1079 model in FP16 without quantization."""

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
from phase1065_multimode_response_atlas_scan import (
    RoleCapture,
    event_definitions,
    pairwise_direction_consistency,
    strict_generated_answer,
)
import phase1079_output_orthogonal_pattern_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 1,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12


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
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    positions = torch.zeros(
        (len(rows), len(protocol.CAPTURE_ROLES)),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        values = torch.tensor(
            row["input_ids"],
            dtype=torch.long,
            device=device,
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        lengths[index] = len(values)
        positions[index] = torch.tensor([
            int(row["role_positions"][role])
            for role in protocol.CAPTURE_ROLES
        ], dtype=torch.long, device=device)
    return input_ids, attention_mask, lengths, positions


def delta_stats(
    delta: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    norms = torch.linalg.vector_norm(delta, dim=-1)
    base = 0.5 * (
        torch.linalg.vector_norm(left, dim=-1)
        + torch.linalg.vector_norm(right, dim=-1)
    )
    finite = (
        torch.isfinite(delta).all(dim=-1)
        & torch.isfinite(norms)
        & torch.isfinite(base)
    )
    magnitude_valid = finite & (base > EPSILON)
    relative = torch.zeros_like(norms, dtype=torch.float32)
    relative[magnitude_valid] = (
        norms[magnitude_valid] / base[magnitude_valid]
    )
    direction_valid = magnitude_valid & (norms > EPSILON)
    direction = torch.zeros_like(delta, dtype=torch.float32)
    direction[direction_valid] = (
        delta[direction_valid] / norms[direction_valid, None]
    )
    return relative, magnitude_valid, direction, direction_valid


def interaction_relative(
    left_delta: torch.Tensor,
    right_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    interaction = right_delta - left_delta
    numerator = torch.linalg.vector_norm(interaction, dim=-1)
    denominator = (
        torch.linalg.vector_norm(left_delta, dim=-1)
        + torch.linalg.vector_norm(right_delta, dim=-1)
    )
    valid = (
        torch.isfinite(interaction).all(dim=-1)
        & torch.isfinite(numerator)
        & torch.isfinite(denominator)
        & (denominator > EPSILON)
    )
    result = torch.zeros_like(numerator, dtype=torch.float32)
    result[valid] = numerator[valid] / denominator[valid]
    return result, valid


def vector_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    denominator = left_norm * right_norm
    valid = (
        torch.isfinite(left).all(dim=-1)
        & torch.isfinite(right).all(dim=-1)
        & torch.isfinite(denominator)
        & (denominator > EPSILON)
    )
    result = torch.zeros_like(denominator, dtype=torch.float32)
    result[valid] = (
        (left[valid] * right[valid]).sum(dim=-1)
        / denominator[valid]
    )
    return result, valid


def add_direction_field(
    *,
    delta: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    conditionings: list[int],
    base_index: tuple[int, int, int],
    direction_sum: np.ndarray,
    direction_count: np.ndarray,
    relative_sum: np.ndarray,
    relative_count: np.ndarray,
) -> int:
    relative, magnitude_valid, direction, direction_valid = delta_stats(
        delta, left, right
    )
    relative_np = relative.cpu().numpy()
    magnitude_valid_np = (
        magnitude_valid.cpu().numpy().astype(np.int32)
    )
    direction_np = direction.cpu().numpy()
    direction_valid_np = (
        direction_valid.cpu().numpy().astype(np.int32)
    )
    for conditioning in conditionings:
        index = (conditioning, *base_index)
        relative_sum[index] += relative_np
        relative_count[index] += magnitude_valid_np
        direction_sum[index] += direction_np
        direction_count[index] += direction_valid_np
    return int((~magnitude_valid).sum().item())


def add_scalar_field(
    *,
    values: torch.Tensor,
    valid: torch.Tensor,
    conditionings: list[int],
    base_index: tuple[int, int, int],
    value_sum: np.ndarray,
    value_count: np.ndarray,
) -> None:
    values_np = values.cpu().numpy()
    valid_np = valid.cpu().numpy().astype(np.int32)
    for conditioning in conditionings:
        index = (conditioning, *base_index)
        value_sum[index] += values_np
        value_count[index] += valid_np


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"\s+", " ", value)


def classify_generation(
    generated: str,
    labels: list[str],
) -> dict[str, Any]:
    text = normalized_text(generated)
    normalized_labels = [normalized_text(label) for label in labels]
    matched = None
    for label in normalized_labels:
        if not text.startswith(label):
            continue
        if label and label[-1].isalnum():
            tail = text[len(label):]
            if tail and tail[0].isalnum():
                continue
        matched = label
        break
    strict = text in normalized_labels
    return {
        "normalized_text": text,
        "acceptable_normalized_labels": normalized_labels,
        "matched_label": matched,
        "semantic_first": matched is not None,
        "strict_label_only": strict,
    }


def natural_generation_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    count = protocol.NATURAL_GENERATION_CASES_PER_FAMILY_SPLIT
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            eligible = [
                row
                for row in rows
                if row["family"] == family
                and row["split"] == split
                and row["panel"] == "natural"
                and row["state"] in {
                    "t0_a0_l0",
                    "t1_a1_l1",
                }
            ]
            selected.extend(generation.evenly_spaced(eligible, count))
    return selected


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1079 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    unit_meta = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["panel"])][
            str(row["state"])
        ] = row
        unit_meta[str(row["unit_id"])] = {
            "family": str(row["family"]),
            "split": str(row["split"]),
        }
    units = []
    for unit_id in sorted(grouped):
        panels = grouped[unit_id]
        for panel in protocol.PANELS:
            if set(panels[panel]) != set(
                protocol.STATES_BY_PANEL[panel]
            ):
                raise RuntimeError(f"incomplete {unit_id}/{panel}")
        units.append({
            "unit_id": unit_id,
            **unit_meta[unit_id],
            "panels": panels,
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
        events = event_definitions(len(layers))
        event_keys = [
            (str(row["component"]), int(row["depth"]))
            for row in events
        ]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        conditioning_index = {
            value: index
            for index, value in enumerate(protocol.CONDITIONINGS)
        }
        family_index = {
            value: index
            for index, value in enumerate(protocol.FAMILIES)
        }
        split_index = {
            value: index
            for index, value in enumerate(protocol.SPLITS)
        }
        role_index = {
            value: index
            for index, value in enumerate(protocol.CAPTURE_ROLES)
        }
        shape = (
            len(protocol.CONDITIONINGS),
            len(protocol.FAMILIES),
            len(protocol.SPLITS),
            len(events),
            len(protocol.CAPTURE_ROLES),
        )

        def vector_arrays():
            return (
                np.zeros((*shape, d_model), dtype=np.float32),
                np.zeros(shape, dtype=np.int32),
                np.zeros(shape, dtype=np.float64),
                np.zeros(shape, dtype=np.int32),
            )

        (
            operation_direction_sum,
            operation_direction_count,
            operation_relative_sum,
            operation_relative_count,
        ) = vector_arrays()
        (
            controlled_answer_direction_sum,
            controlled_answer_direction_count,
            controlled_answer_relative_sum,
            controlled_answer_relative_count,
        ) = vector_arrays()
        (
            natural_answer_direction_sum,
            natural_answer_direction_count,
            natural_answer_relative_sum,
            natural_answer_relative_count,
        ) = vector_arrays()

        def scalar_arrays():
            return (
                np.zeros(shape, dtype=np.float64),
                np.zeros(shape, dtype=np.int32),
            )

        surface_sum, surface_count = scalar_arrays()
        shell_sum, shell_count = scalar_arrays()
        operation_answer_interaction_sum, (
            operation_answer_interaction_count
        ) = scalar_arrays()
        operation_cross_answer_sum, (
            operation_cross_answer_count
        ) = scalar_arrays()
        operation_cross_surface_sum, (
            operation_cross_surface_count
        ) = scalar_arrays()
        operation_cross_shell_sum, (
            operation_cross_shell_count
        ) = scalar_arrays()

        behavior_records = []
        candidate_totals = Counter()
        candidate_hits = Counter()
        candidate_finite = Counter()
        supported_units = Counter()
        nonfinite_candidate_count = 0
        nonfinite_hidden_count = 0
        pre_mode_operation_max_abs = 0.0
        identity_maximum = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        capture = RoleCapture(model, layers)
        capture.register()
        batch_size = UNIT_BATCH_SIZE[model_name]
        controlled_order = list(protocol.CONTROLLED_STATES)
        natural_order = list(protocol.NATURAL_STATES)
        with torch.inference_mode():
            for batch_start in range(0, len(units), batch_size):
                batch_units = units[batch_start:batch_start + batch_size]
                forward_rows = []
                offsets = []
                for unit in batch_units:
                    offset = len(forward_rows)
                    controlled_rows = [
                        unit["panels"]["controlled"][state]
                        for state in controlled_order
                    ]
                    natural_rows = [
                        unit["panels"]["natural"][state]
                        for state in natural_order
                    ]
                    forward_rows.extend(controlled_rows)
                    forward_rows.extend(natural_rows)
                    if batch_start == 0:
                        forward_rows.append(controlled_rows[0])
                        identity_offset = len(forward_rows) - 1
                    else:
                        identity_offset = None
                    offsets.append({
                        "controlled": offset,
                        "natural": offset + len(controlled_order),
                        "identity": identity_offset,
                    })
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
                logits = output.logits
                final_positions = (lengths - 1).to(logits.device)
                batch_axis = torch.arange(
                    logits.shape[0], device=logits.device
                )
                final_logits = logits[
                    batch_axis, final_positions, :
                ].float()
                del output, logits

                unit_behavior_support = {}
                for unit, offset in zip(batch_units, offsets):
                    relevant_hits = []
                    all_rows = []
                    for local, state in enumerate(controlled_order):
                        all_rows.append((
                            unit["panels"]["controlled"][state],
                            offset["controlled"] + local,
                        ))
                    for local, state in enumerate(natural_order):
                        all_rows.append((
                            unit["panels"]["natural"][state],
                            offset["natural"] + local,
                        ))
                    for row, row_index in all_rows:
                        values = final_logits[row_index]
                        scores = {}
                        for answer_class in ("a0", "a1"):
                            token_ids = torch.tensor(
                                row["candidate_first_token_ids"][
                                    answer_class
                                ],
                                dtype=torch.long,
                                device=values.device,
                            )
                            scores[answer_class] = float(
                                values[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = "a1" if expected == "a0" else "a0"
                        margin = scores[expected] - scores[other]
                        finite = all(
                            math.isfinite(value)
                            for value in scores.values()
                        ) and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        greedy = int(torch.argmax(values).item())
                        key = (
                            unit["family"],
                            unit["split"],
                            str(row["panel"]),
                            (
                                str(row["mode"])
                                if row["panel"] == "controlled"
                                else "natural"
                            ),
                        )
                        candidate_totals[key] += 1
                        candidate_finite[key] += int(finite)
                        candidate_hits[key] += int(hit)
                        nonfinite_candidate_count += int(not finite)
                        if (
                            row["panel"] == "natural"
                            or row["operation_branch"] == 1
                        ):
                            relevant_hits.append(int(hit))
                        behavior_records.append({
                            "schema_version": (
                                "phase1079_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "case_index": int(row["case_index"]),
                            "unit_id": unit["unit_id"],
                            "family": unit["family"],
                            "split": unit["split"],
                            "panel": row["panel"],
                            "state": row["state"],
                            "mode": row["mode"],
                            "expected_class": expected,
                            "target_answer": row["target_answer"],
                            "candidate_scores": {
                                key_name: (
                                    value
                                    if math.isfinite(value)
                                    else None
                                )
                                for key_name, value in scores.items()
                            },
                            "candidate_margin": (
                                margin if math.isfinite(margin) else None
                            ),
                            "finite_candidate": finite,
                            "candidate_hit": hit,
                            "greedy_first_token_id": greedy,
                            "greedy_first_token_text": tokenizer.decode(
                                [greedy]
                            ),
                        })
                    fraction = (
                        sum(relevant_hits) / len(relevant_hits)
                        if relevant_hits else 0.0
                    )
                    supported = fraction >= float(
                        prereg["evidence_thresholds"][
                            "unit_behavior_support_fraction"
                        ]
                    )
                    unit_behavior_support[unit["unit_id"]] = supported
                    supported_units[
                        (unit["family"], unit["split"])
                    ] += int(supported)

                for event_index, event_key in enumerate(event_keys):
                    values = capture.values[event_key].float()
                    for unit, offset in zip(batch_units, offsets):
                        controlled = {
                            state: values[
                                offset["controlled"] + local
                            ]
                            for local, state in enumerate(
                                controlled_order
                            )
                        }
                        natural = {
                            state: values[offset["natural"] + local]
                            for local, state in enumerate(natural_order)
                        }
                        if offset["identity"] is not None:
                            identity_delta = (
                                values[offset["identity"]]
                                - controlled[controlled_order[0]]
                            )
                            if torch.isfinite(identity_delta).all():
                                identity_maximum = max(
                                    identity_maximum,
                                    float(torch.max(torch.abs(
                                        identity_delta
                                    )).item()),
                                )

                        family = family_index[unit["family"]]
                        split = split_index[unit["split"]]
                        base_index = (family, split, event_index)
                        conditionings = [
                            conditioning_index["all_finite"]
                        ]
                        if unit_behavior_support[unit["unit_id"]]:
                            conditionings.append(
                                conditioning_index[
                                    "behavior_supported"
                                ]
                            )

                        operation_deltas = {}
                        for template in (0, 1):
                            for answer in (0, 1):
                                for surface in (0, 1):
                                    left = controlled[
                                        f"t{template}_o0_a{answer}_l{surface}"
                                    ]
                                    right = controlled[
                                        f"t{template}_o1_a{answer}_l{surface}"
                                    ]
                                    delta = right - left
                                    operation_deltas[
                                        (template, answer, surface)
                                    ] = delta
                                    nonfinite_hidden_count += (
                                        add_direction_field(
                                            delta=delta,
                                            left=left,
                                            right=right,
                                            conditionings=conditionings,
                                            base_index=base_index,
                                            direction_sum=(
                                                operation_direction_sum
                                            ),
                                            direction_count=(
                                                operation_direction_count
                                            ),
                                            relative_sum=(
                                                operation_relative_sum
                                            ),
                                            relative_count=(
                                                operation_relative_count
                                            ),
                                        )
                                    )
                                    for role_name in (
                                        protocol.PRE_MODE_ROLES
                                    ):
                                        role = role_index[role_name]
                                        role_delta = delta[role]
                                        if torch.isfinite(role_delta).all():
                                            pre_mode_operation_max_abs = max(
                                                pre_mode_operation_max_abs,
                                                float(torch.max(torch.abs(
                                                    role_delta
                                                )).item()),
                                            )

                        for template in (0, 1):
                            for surface in (0, 1):
                                left = controlled[
                                    f"t{template}_o1_a0_l{surface}"
                                ]
                                right = controlled[
                                    f"t{template}_o1_a1_l{surface}"
                                ]
                                nonfinite_hidden_count += (
                                    add_direction_field(
                                        delta=right - left,
                                        left=left,
                                        right=right,
                                        conditionings=conditionings,
                                        base_index=base_index,
                                        direction_sum=(
                                            controlled_answer_direction_sum
                                        ),
                                        direction_count=(
                                            controlled_answer_direction_count
                                        ),
                                        relative_sum=(
                                            controlled_answer_relative_sum
                                        ),
                                        relative_count=(
                                            controlled_answer_relative_count
                                        ),
                                    )
                                )

                        for template in (0, 1):
                            for surface in (0, 1):
                                left = natural[
                                    f"t{template}_a0_l{surface}"
                                ]
                                right = natural[
                                    f"t{template}_a1_l{surface}"
                                ]
                                nonfinite_hidden_count += (
                                    add_direction_field(
                                        delta=right - left,
                                        left=left,
                                        right=right,
                                        conditionings=conditionings,
                                        base_index=base_index,
                                        direction_sum=(
                                            natural_answer_direction_sum
                                        ),
                                        direction_count=(
                                            natural_answer_direction_count
                                        ),
                                        relative_sum=(
                                            natural_answer_relative_sum
                                        ),
                                        relative_count=(
                                            natural_answer_relative_count
                                        ),
                                    )
                                )

                        for panel_name, panel_states in (
                            ("controlled", controlled),
                            ("natural", natural),
                        ):
                            if panel_name == "controlled":
                                for template in (0, 1):
                                    for operation in (0, 1):
                                        for answer in (0, 1):
                                            left = panel_states[
                                                f"t{template}_o{operation}_a{answer}_l0"
                                            ]
                                            right = panel_states[
                                                f"t{template}_o{operation}_a{answer}_l1"
                                            ]
                                            relative, valid, _, _ = (
                                                delta_stats(
                                                    right - left,
                                                    left,
                                                    right,
                                                )
                                            )
                                            add_scalar_field(
                                                values=relative,
                                                valid=valid,
                                                conditionings=conditionings,
                                                base_index=base_index,
                                                value_sum=surface_sum,
                                                value_count=surface_count,
                                            )
                                for operation in (0, 1):
                                    for answer in (0, 1):
                                        for surface in (0, 1):
                                            left = panel_states[
                                                f"t0_o{operation}_a{answer}_l{surface}"
                                            ]
                                            right = panel_states[
                                                f"t1_o{operation}_a{answer}_l{surface}"
                                            ]
                                            relative, valid, _, _ = (
                                                delta_stats(
                                                    right - left,
                                                    left,
                                                    right,
                                                )
                                            )
                                            add_scalar_field(
                                                values=relative,
                                                valid=valid,
                                                conditionings=conditionings,
                                                base_index=base_index,
                                                value_sum=shell_sum,
                                                value_count=shell_count,
                                            )
                            else:
                                for template in (0, 1):
                                    for answer in (0, 1):
                                        left = panel_states[
                                            f"t{template}_a{answer}_l0"
                                        ]
                                        right = panel_states[
                                            f"t{template}_a{answer}_l1"
                                        ]
                                        relative, valid, _, _ = (
                                            delta_stats(
                                                right - left,
                                                left,
                                                right,
                                            )
                                        )
                                        add_scalar_field(
                                            values=relative,
                                            valid=valid,
                                            conditionings=conditionings,
                                            base_index=base_index,
                                            value_sum=surface_sum,
                                            value_count=surface_count,
                                        )
                                for answer in (0, 1):
                                    for surface in (0, 1):
                                        left = panel_states[
                                            f"t0_a{answer}_l{surface}"
                                        ]
                                        right = panel_states[
                                            f"t1_a{answer}_l{surface}"
                                        ]
                                        relative, valid, _, _ = (
                                            delta_stats(
                                                right - left,
                                                left,
                                                right,
                                            )
                                        )
                                        add_scalar_field(
                                            values=relative,
                                            valid=valid,
                                            conditionings=conditionings,
                                            base_index=base_index,
                                            value_sum=shell_sum,
                                            value_count=shell_count,
                                        )

                        for template in (0, 1):
                            for surface in (0, 1):
                                left_delta = operation_deltas[
                                    (template, 0, surface)
                                ]
                                right_delta = operation_deltas[
                                    (template, 1, surface)
                                ]
                                interaction, valid = interaction_relative(
                                    left_delta, right_delta
                                )
                                add_scalar_field(
                                    values=interaction,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=(
                                        operation_answer_interaction_sum
                                    ),
                                    value_count=(
                                        operation_answer_interaction_count
                                    ),
                                )
                                cosine, valid = vector_cosine(
                                    left_delta, right_delta
                                )
                                add_scalar_field(
                                    values=cosine,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=operation_cross_answer_sum,
                                    value_count=operation_cross_answer_count,
                                )

                        for template in (0, 1):
                            for answer in (0, 1):
                                cosine, valid = vector_cosine(
                                    operation_deltas[
                                        (template, answer, 0)
                                    ],
                                    operation_deltas[
                                        (template, answer, 1)
                                    ],
                                )
                                add_scalar_field(
                                    values=cosine,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=operation_cross_surface_sum,
                                    value_count=operation_cross_surface_count,
                                )
                        for answer in (0, 1):
                            for surface in (0, 1):
                                cosine, valid = vector_cosine(
                                    operation_deltas[(0, answer, surface)],
                                    operation_deltas[(1, answer, surface)],
                                )
                                add_scalar_field(
                                    values=cosine,
                                    valid=valid,
                                    conditionings=conditionings,
                                    base_index=base_index,
                                    value_sum=operation_cross_shell_sum,
                                    value_count=operation_cross_shell_count,
                                )
                    del values

                del (
                    final_logits,
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
                if completed % 12 == 0 or completed == len(units):
                    print(json.dumps({
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "units_complete": completed,
                        "units_total": len(units),
                    }), flush=True)

        capture.close()
        capture = None

        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        generation_rows = natural_generation_selection(rows)
        generated = generation.generate_case_outputs(
            model,
            device,
            generation_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["natural_generation_steps"]),
        )
        generation_records = []
        generation_totals = Counter()
        generation_hits = Counter()
        generation_strict = Counter()
        for row in generation_rows:
            case_index = int(row["case_index"])
            output_ids = generated[case_index]
            text = strict_generated_answer(
                tokenizer, output_ids, eos_ids
            )
            classification = classify_generation(
                text, row["acceptable_labels"]
            )
            key = (str(row["family"]), str(row["split"]))
            generation_totals[key] += 1
            generation_hits[key] += int(
                classification["semantic_first"]
            )
            generation_strict[key] += int(
                classification["strict_label_only"]
            )
            generation_records.append({
                "schema_version": "phase1079_natural_generation.v1",
                "phase": protocol.PHASE,
                "model": model_name,
                "case_index": case_index,
                "unit_id": row["unit_id"],
                "family": row["family"],
                "split": row["split"],
                "state": row["state"],
                "target_answer": row["target_answer"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": text,
                "terminated": generation.terminated(
                    output_ids, eos_ids
                ),
                **classification,
            })

        def metric_value(
            value_sum: np.ndarray,
            value_count: np.ndarray,
            index: tuple[int, ...],
) -> float | None:
            count = int(value_count[index])
            return float(value_sum[index] / count) if count else None

        metric_rows = []
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for split_name, split in split_index.items():
                    for event_index, event in enumerate(events):
                        for role_name, role in role_index.items():
                            index = (
                                conditioning,
                                family,
                                split,
                                event_index,
                                role,
                            )
                            operation_n = int(
                                operation_direction_count[index]
                            )
                            controlled_answer_n = int(
                                controlled_answer_direction_count[index]
                            )
                            natural_answer_n = int(
                                natural_answer_direction_count[index]
                            )
                            metric_rows.append({
                                "schema_version": (
                                    "phase1079_response_metric.v1"
                                ),
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "conditioning": conditioning_name,
                                "family": family_name,
                                "split": split_name,
                                "role": role_name,
                                **event,
                                "operation_magnitude_count": int(
                                    operation_relative_count[index]
                                ),
                                "mean_operation_relative_magnitude": (
                                    metric_value(
                                        operation_relative_sum,
                                        operation_relative_count,
                                        index,
                                    )
                                ),
                                "operation_direction_count": operation_n,
                                "operation_direction_consistency": (
                                    pairwise_direction_consistency(
                                        operation_direction_sum[index],
                                        operation_n,
                                    )
                                ),
                                "controlled_answer_magnitude_count": int(
                                    controlled_answer_relative_count[index]
                                ),
                                "mean_controlled_answer_relative_magnitude": (
                                    metric_value(
                                        controlled_answer_relative_sum,
                                        controlled_answer_relative_count,
                                        index,
                                    )
                                ),
                                "controlled_answer_direction_count": (
                                    controlled_answer_n
                                ),
                                "controlled_answer_direction_consistency": (
                                    pairwise_direction_consistency(
                                        controlled_answer_direction_sum[
                                            index
                                        ],
                                        controlled_answer_n,
                                    )
                                ),
                                "natural_answer_magnitude_count": int(
                                    natural_answer_relative_count[index]
                                ),
                                "mean_natural_answer_relative_magnitude": (
                                    metric_value(
                                        natural_answer_relative_sum,
                                        natural_answer_relative_count,
                                        index,
                                    )
                                ),
                                "natural_answer_direction_count": (
                                    natural_answer_n
                                ),
                                "natural_answer_direction_consistency": (
                                    pairwise_direction_consistency(
                                        natural_answer_direction_sum[index],
                                        natural_answer_n,
                                    )
                                ),
                                "surface_count": int(surface_count[index]),
                                "mean_surface_relative_magnitude": (
                                    metric_value(
                                        surface_sum,
                                        surface_count,
                                        index,
                                    )
                                ),
                                "shell_count": int(shell_count[index]),
                                "mean_shell_relative_magnitude": (
                                    metric_value(
                                        shell_sum,
                                        shell_count,
                                        index,
                                    )
                                ),
                                "operation_answer_interaction_count": int(
                                    operation_answer_interaction_count[
                                        index
                                    ]
                                ),
                                "mean_operation_answer_interaction": (
                                    metric_value(
                                        operation_answer_interaction_sum,
                                        operation_answer_interaction_count,
                                        index,
                                    )
                                ),
                                "operation_cross_answer_count": int(
                                    operation_cross_answer_count[index]
                                ),
                                "mean_operation_cross_answer_cosine": (
                                    metric_value(
                                        operation_cross_answer_sum,
                                        operation_cross_answer_count,
                                        index,
                                    )
                                ),
                                "operation_cross_surface_count": int(
                                    operation_cross_surface_count[index]
                                ),
                                "mean_operation_cross_surface_cosine": (
                                    metric_value(
                                        operation_cross_surface_sum,
                                        operation_cross_surface_count,
                                        index,
                                    )
                                ),
                                "operation_cross_shell_count": int(
                                    operation_cross_shell_count[index]
                                ),
                                "mean_operation_cross_shell_cosine": (
                                    metric_value(
                                        operation_cross_shell_sum,
                                        operation_cross_shell_count,
                                        index,
                                    )
                                ),
                            })

        split_direction_rows = []
        discovery = split_index["discovery"]
        confirmation = split_index["confirmation"]
        fields = {
            "operation": (
                operation_direction_sum,
                operation_direction_count,
            ),
            "controlled_answer": (
                controlled_answer_direction_sum,
                controlled_answer_direction_count,
            ),
            "natural_answer": (
                natural_answer_direction_sum,
                natural_answer_direction_count,
            ),
        }
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for event_index, event in enumerate(events):
                    for role_name, role in role_index.items():
                        record = {
                            "schema_version": (
                                "phase1079_split_direction_repeat.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "conditioning": conditioning_name,
                            "family": family_name,
                            "role": role_name,
                            **event,
                        }
                        for field_name, (sums, counts) in fields.items():
                            left_index = (
                                conditioning,
                                family,
                                discovery,
                                event_index,
                                role,
                            )
                            right_index = (
                                conditioning,
                                family,
                                confirmation,
                                event_index,
                                role,
                            )
                            left = sums[left_index].astype(
                                np.float64, copy=False
                            )
                            right = sums[right_index].astype(
                                np.float64, copy=False
                            )
                            denominator = float(
                                np.linalg.norm(left)
                                * np.linalg.norm(right)
                            )
                            record[
                                f"{field_name}_discovery_confirmation_cosine"
                            ] = (
                                float(np.dot(left, right) / denominator)
                                if denominator > EPSILON
                                else None
                            )
                            record[
                                f"{field_name}_discovery_count"
                            ] = int(counts[left_index])
                            record[
                                f"{field_name}_confirmation_count"
                            ] = int(counts[right_index])
                        split_direction_rows.append(record)

        behavior_summary = {}
        for family in protocol.FAMILIES:
            behavior_summary[family] = {}
            for split in protocol.SPLITS:
                panel_rows = {}
                for panel, mode in (
                    ("controlled", "semantic"),
                    ("controlled", "index"),
                    ("natural", "natural"),
                ):
                    key = (family, split, panel, mode)
                    total = int(candidate_totals[key])
                    panel_rows[f"{panel}_{mode}"] = {
                        "candidate_count": total,
                        "candidate_finite_count": int(
                            candidate_finite[key]
                        ),
                        "candidate_hit_count": int(candidate_hits[key]),
                        "candidate_accuracy": (
                            candidate_hits[key] / total
                            if total else None
                        ),
                    }
                generation_key = (family, split)
                total_generation = int(
                    generation_totals[generation_key]
                )
                panel_rows["natural_generation"] = {
                    "case_count": total_generation,
                    "semantic_first_count": int(
                        generation_hits[generation_key]
                    ),
                    "semantic_first_accuracy": (
                        generation_hits[generation_key]
                        / total_generation
                        if total_generation else None
                    ),
                    "strict_count": int(
                        generation_strict[generation_key]
                    ),
                }
                panel_rows["behavior_supported_unit_count"] = int(
                    supported_units[(family, split)]
                )
                behavior_summary[family][split] = panel_rows

        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(
            atlas_root / "candidate_behavior.jsonl",
            behavior_records,
        )
        protocol.write_jsonl(
            atlas_root / "natural_generation.jsonl",
            generation_records,
        )
        protocol.write_jsonl(
            atlas_root / "response_metrics.jsonl",
            metric_rows,
        )
        protocol.write_jsonl(
            atlas_root / "split_direction_repeat.jsonl",
            split_direction_rows,
        )

        residual_event_indices = [
            index
            for index, event in enumerate(events)
            if event["component"] == "residual"
        ]
        selected_role_indices = [
            role_index["active_mode"],
            role_index["answer_boundary"],
        ]

        def mean_selected(
            sums: np.ndarray,
            counts: np.ndarray,
) -> np.ndarray:
            selected_sums = sums[
                :,
                :,
                :,
                residual_event_indices,
                :,
                :,
            ][:, :, :, :, selected_role_indices, :]
            selected_counts = counts[
                :,
                :,
                :,
                residual_event_indices,
                :,
            ][:, :, :, :, selected_role_indices]
            result = np.zeros_like(selected_sums, dtype=np.float32)
            np.divide(
                selected_sums,
                selected_counts[..., None],
                out=result,
                where=selected_counts[..., None] > 0,
            )
            return result.astype(np.float16)

        np.savez_compressed(
            atlas_root / "selected_mean_directions.fp16.npz",
            operation=mean_selected(
                operation_direction_sum,
                operation_direction_count,
            ),
            controlled_answer=mean_selected(
                controlled_answer_direction_sum,
                controlled_answer_direction_count,
            ),
            natural_answer=mean_selected(
                natural_answer_direction_sum,
                natural_answer_direction_count,
            ),
            conditionings=np.array(protocol.CONDITIONINGS),
            families=np.array(protocol.FAMILIES),
            splits=np.array(protocol.SPLITS),
            residual_depths=np.array([
                int(events[index]["depth"])
                for index in residual_event_indices
            ]),
            roles=np.array(("active_mode", "answer_boundary")),
        )

        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1079_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["model_case_digests"][model_name],
            "case_count": len(rows),
            "unit_count": len(units),
            "event_count": len(events),
            "layer_count": len(layers),
            "d_model": d_model,
            "precision": precision,
            "placement": placement,
            "behavior_summary": behavior_summary,
            "pre_mode_operation_max_abs": (
                pre_mode_operation_max_abs
            ),
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "nonfinite_hidden_magnitude_role_count": (
                nonfinite_hidden_count
            ),
            "elapsed_seconds": elapsed,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "status": "complete",
            "case_count": len(rows),
            "unit_count": len(units),
            "elapsed_seconds": elapsed,
            "pre_mode_operation_max_abs": (
                pre_mode_operation_max_abs
            ),
            "summary_digest": summary["summary_digest"],
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
