#!/usr/bin/env python3
"""Run the Phase1078 shared-shell FP16 three-factor response atlas."""

from __future__ import annotations

import argparse
import json
import math
import re
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
from phase1065_multimode_response_atlas_scan import (
    RoleCapture,
    event_definitions,
    pairwise_direction_consistency,
    strict_generated_answer,
)
import phase1078_shared_shell_pattern_atlas_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 2,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12


def pad_rows(
    rows: list[dict[str, Any]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch while using the Phase1078 role vocabulary."""
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


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    per_group = protocol.NATURAL_AUDIT_CASES_PER_FAMILY_SPLIT
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            eligible = [
                row
                for row in rows
                if row["family"] == family
                and row["split"] == split
                and row["state"] in {
                    "t0_b0_l0",
                    "t1_b1_l1",
                }
            ]
            selected.extend(
                generation.evenly_spaced(eligible, per_group)
            )
    return selected


def natural_classification(
    generated_text: str,
    acceptable_labels: list[str],
    terminated: bool,
) -> dict[str, Any]:
    normalized = text_tools.normalize_text(generated_text)
    acceptable = [
        text_tools.normalize_text(str(value))
        for value in acceptable_labels
    ]
    strict = normalized in acceptable
    matched_label = None
    semantic_first = False
    punctuation_only = False
    for label in acceptable:
        match = re.match(
            rf"^{re.escape(label)}(?=$|[\s.,;:!?])",
            normalized,
        )
        if not match:
            continue
        matched_label = label
        semantic_first = True
        tail = normalized[match.end():].strip()
        punctuation_only = bool(tail) and bool(
            re.fullmatch(r"[\s.,;:!?]+", tail)
        )
        break
    if not normalized:
        tail_class = "empty"
    elif strict:
        tail_class = "strict_label_only"
    elif semantic_first and punctuation_only:
        tail_class = "label_plus_punctuation"
    elif semantic_first:
        tail_class = "label_plus_extra_content"
    else:
        tail_class = "wrong_first_content"
    return {
        "normalized_text": normalized,
        "acceptable_normalized_labels": acceptable,
        "matched_label": matched_label,
        "semantic_first": semantic_first,
        "strict_label_only": strict,
        "terminated": terminated,
        "tail_class": tail_class,
    }


def safe_relative(
    delta: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    valid = finite & (norms > EPSILON) & (base > EPSILON)
    relative = torch.zeros_like(norms, dtype=torch.float32)
    direction = torch.zeros_like(delta, dtype=torch.float32)
    relative[valid] = norms[valid] / base[valid]
    direction[valid] = delta[valid] / norms[valid, None]
    return direction, relative, valid


def safe_vector_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    valid = (
        torch.isfinite(left).all(dim=-1)
        & torch.isfinite(right).all(dim=-1)
        & torch.isfinite(left_norm)
        & torch.isfinite(right_norm)
        & (left_norm > EPSILON)
        & (right_norm > EPSILON)
    )
    result = torch.zeros_like(left_norm, dtype=torch.float32)
    denominator = left_norm * right_norm
    result[valid] = (
        (left[valid] * right[valid]).sum(dim=-1)
        / denominator[valid]
    )
    return result, valid


def interaction_relative(
    left_delta: torch.Tensor,
    right_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm = torch.linalg.vector_norm(left_delta, dim=-1)
    right_norm = torch.linalg.vector_norm(right_delta, dim=-1)
    interaction = right_delta - left_delta
    interaction_norm = torch.linalg.vector_norm(interaction, dim=-1)
    denominator = left_norm + right_norm
    valid = (
        torch.isfinite(left_delta).all(dim=-1)
        & torch.isfinite(right_delta).all(dim=-1)
        & torch.isfinite(interaction_norm)
        & torch.isfinite(denominator)
        & (denominator > EPSILON)
    )
    result = torch.zeros_like(interaction_norm, dtype=torch.float32)
    result[valid] = interaction_norm[valid] / denominator[valid]
    return result, valid


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1078 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[str(row["unit_id"])].append(row)
    units = []
    for unit_id, values in sorted(by_unit.items()):
        by_state = {str(row["state"]): row for row in values}
        if set(by_state) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit {unit_id}")
        units.append({
            "unit_id": unit_id,
            "family": values[0]["family"],
            "split": values[0]["split"],
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
        direction_sum = np.zeros((*shape, d_model), dtype=np.float32)
        truth_count = np.zeros(shape, dtype=np.int32)
        truth_relative_sum = np.zeros(shape, dtype=np.float64)
        surface_count = np.zeros(shape, dtype=np.int32)
        surface_relative_sum = np.zeros(shape, dtype=np.float64)
        shell_count = np.zeros(shape, dtype=np.int32)
        shell_relative_sum = np.zeros(shape, dtype=np.float64)
        truth_surface_count = np.zeros(shape, dtype=np.int32)
        truth_surface_relative_sum = np.zeros(shape, dtype=np.float64)
        truth_shell_count = np.zeros(shape, dtype=np.int32)
        truth_shell_relative_sum = np.zeros(shape, dtype=np.float64)
        cross_surface_count = np.zeros(shape, dtype=np.int32)
        cross_surface_cosine_sum = np.zeros(shape, dtype=np.float64)
        cross_shell_count = np.zeros(shape, dtype=np.int32)
        cross_shell_cosine_sum = np.zeros(shape, dtype=np.float64)

        behavior_records = []
        total_cases = Counter()
        hit_cases = Counter()
        greedy_hits = Counter()
        finite_cases = Counter()
        complete_units = Counter()
        identity_maximum = 0.0
        nonfinite_hidden_truth_role_count = 0
        nonfinite_candidate_count = 0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        capture = RoleCapture(model, layers)
        capture.register()
        batch_size = UNIT_BATCH_SIZE[model_name]
        state_order = list(protocol.STATES)
        with torch.inference_mode():
            for batch_start in range(0, len(units), batch_size):
                batch_units = units[batch_start:batch_start + batch_size]
                forward_rows = []
                unit_offsets = []
                for unit in batch_units:
                    offset = len(forward_rows)
                    forward_rows.extend([
                        unit["states"][state] for state in state_order
                    ])
                    forward_rows.append(
                        unit["states"]["t0_b0_l0"]
                    )
                    unit_offsets.append(offset)
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
                last_positions = (lengths - 1).to(logits.device)
                batch_axis = torch.arange(
                    logits.shape[0],
                    device=logits.device,
                )
                last_logits = logits[
                    batch_axis,
                    last_positions,
                    :,
                ].float()
                del output, logits

                batch_complete: dict[str, bool] = {}
                for unit, offset in zip(batch_units, unit_offsets):
                    local_hits = {}
                    for local_index, state in enumerate(state_order):
                        row = unit["states"][state]
                        values = last_logits[offset + local_index]
                        class_scores = {}
                        for class_name in ("b0", "b1"):
                            token_ids = torch.tensor(
                                row["candidate_first_token_ids"][
                                    class_name
                                ],
                                dtype=torch.long,
                                device=values.device,
                            )
                            class_scores[class_name] = float(
                                values[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = "b1" if expected == "b0" else "b0"
                        margin = (
                            class_scores[expected] - class_scores[other]
                        )
                        finite = all(
                            math.isfinite(value)
                            for value in class_scores.values()
                        ) and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        greedy_token = int(torch.argmax(values).item())
                        greedy_hit = greedy_token in set(
                            int(value)
                            for value in row[
                                "candidate_first_token_ids"
                            ][expected]
                        )
                        key = (unit["family"], unit["split"])
                        total_cases[key] += 1
                        finite_cases[key] += int(finite)
                        hit_cases[key] += int(hit)
                        greedy_hits[key] += int(greedy_hit)
                        nonfinite_candidate_count += int(not finite)
                        local_hits[state] = hit
                        behavior_records.append({
                            "schema_version": (
                                "phase1078_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "case_index": int(row["case_index"]),
                            "unit_id": unit["unit_id"],
                            "family": unit["family"],
                            "split": unit["split"],
                            "state": state,
                            "expected_class": expected,
                            "candidate_class_scores": {
                                key_name: (
                                    value
                                    if math.isfinite(value)
                                    else None
                                )
                                for key_name, value
                                in class_scores.items()
                            },
                            "candidate_margin": (
                                margin
                                if math.isfinite(margin)
                                else None
                            ),
                            "finite_candidate": finite,
                            "candidate_hit": hit,
                            "greedy_first_token_id": greedy_token,
                            "greedy_first_token_text": tokenizer.decode(
                                [greedy_token]
                            ),
                            "greedy_first_token_hit": greedy_hit,
                        })
                    complete = all(local_hits.values())
                    batch_complete[unit["unit_id"]] = complete
                    complete_units[
                        (unit["family"], unit["split"])
                    ] += int(complete)

                for event_index, key in enumerate(event_keys):
                    value = capture.values[key].float()
                    for unit, offset in zip(batch_units, unit_offsets):
                        family = family_index[str(unit["family"])]
                        split = split_index[str(unit["split"])]
                        states = {
                            state: value[offset + local_index]
                            for local_index, state in enumerate(state_order)
                        }
                        identity = value[offset + len(state_order)]
                        identity_delta = (
                            identity - states["t0_b0_l0"]
                        )
                        if torch.isfinite(identity_delta).all():
                            identity_maximum = max(
                                identity_maximum,
                                float(torch.max(torch.abs(
                                    identity_delta
                                )).item()),
                            )
                        conditionings = [
                            conditioning_index["all_finite"]
                        ]
                        if batch_complete[unit["unit_id"]]:
                            conditionings.append(
                                conditioning_index[
                                    "behavior_complete"
                                ]
                            )
                        base_index = (
                            family,
                            split,
                            event_index,
                        )

                        truth_deltas: dict[
                            tuple[int, int],
                            torch.Tensor,
                        ] = {}
                        for template in (0, 1):
                            for surface in (0, 1):
                                left = states[
                                    f"t{template}_b0_l{surface}"
                                ]
                                right = states[
                                    f"t{template}_b1_l{surface}"
                                ]
                                delta = right - left
                                truth_deltas[(template, surface)] = delta
                                direction, relative, valid = safe_relative(
                                    delta,
                                    left,
                                    right,
                                )
                                nonfinite_hidden_truth_role_count += int(
                                    (
                                        ~torch.isfinite(delta).all(
                                            dim=-1
                                        )
                                    ).sum().item()
                                )
                                direction_np = direction.cpu().numpy()
                                relative_np = relative.cpu().numpy()
                                valid_np = (
                                    valid.cpu().numpy().astype(np.int32)
                                )
                                for conditioning in conditionings:
                                    index = (
                                        conditioning,
                                        *base_index,
                                    )
                                    direction_sum[index] += direction_np
                                    truth_count[index] += valid_np
                                    truth_relative_sum[
                                        index
                                    ] += relative_np

                        for template in (0, 1):
                            for truth in (0, 1):
                                left = states[
                                    f"t{template}_b{truth}_l0"
                                ]
                                right = states[
                                    f"t{template}_b{truth}_l1"
                                ]
                                _, relative, valid = safe_relative(
                                    right - left,
                                    left,
                                    right,
                                )
                                relative_np = relative.cpu().numpy()
                                valid_np = (
                                    valid.cpu().numpy().astype(np.int32)
                                )
                                for conditioning in conditionings:
                                    index = (
                                        conditioning,
                                        *base_index,
                                    )
                                    surface_count[index] += valid_np
                                    surface_relative_sum[
                                        index
                                    ] += relative_np

                        for truth in (0, 1):
                            for surface in (0, 1):
                                left = states[
                                    f"t0_b{truth}_l{surface}"
                                ]
                                right = states[
                                    f"t1_b{truth}_l{surface}"
                                ]
                                _, relative, valid = safe_relative(
                                    right - left,
                                    left,
                                    right,
                                )
                                relative_np = relative.cpu().numpy()
                                valid_np = (
                                    valid.cpu().numpy().astype(np.int32)
                                )
                                for conditioning in conditionings:
                                    index = (
                                        conditioning,
                                        *base_index,
                                    )
                                    shell_count[index] += valid_np
                                    shell_relative_sum[
                                        index
                                    ] += relative_np

                        for template in (0, 1):
                            result, valid = interaction_relative(
                                truth_deltas[(template, 0)],
                                truth_deltas[(template, 1)],
                            )
                            result_np = result.cpu().numpy()
                            valid_np = (
                                valid.cpu().numpy().astype(np.int32)
                            )
                            cosine_value, cosine_valid = (
                                safe_vector_cosine(
                                    truth_deltas[(template, 0)],
                                    truth_deltas[(template, 1)],
                                )
                            )
                            cosine_np = cosine_value.cpu().numpy()
                            cosine_valid_np = (
                                cosine_valid.cpu().numpy().astype(
                                    np.int32
                                )
                            )
                            for conditioning in conditionings:
                                index = (
                                    conditioning,
                                    *base_index,
                                )
                                truth_surface_count[index] += valid_np
                                truth_surface_relative_sum[
                                    index
                                ] += result_np
                                cross_surface_count[
                                    index
                                ] += cosine_valid_np
                                cross_surface_cosine_sum[
                                    index
                                ] += cosine_np

                        for surface in (0, 1):
                            result, valid = interaction_relative(
                                truth_deltas[(0, surface)],
                                truth_deltas[(1, surface)],
                            )
                            result_np = result.cpu().numpy()
                            valid_np = (
                                valid.cpu().numpy().astype(np.int32)
                            )
                            cosine_value, cosine_valid = (
                                safe_vector_cosine(
                                    truth_deltas[(0, surface)],
                                    truth_deltas[(1, surface)],
                                )
                            )
                            cosine_np = cosine_value.cpu().numpy()
                            cosine_valid_np = (
                                cosine_valid.cpu().numpy().astype(
                                    np.int32
                                )
                            )
                            for conditioning in conditionings:
                                index = (
                                    conditioning,
                                    *base_index,
                                )
                                truth_shell_count[index] += valid_np
                                truth_shell_relative_sum[
                                    index
                                ] += result_np
                                cross_shell_count[
                                    index
                                ] += cosine_valid_np
                                cross_shell_cosine_sum[
                                    index
                                ] += cosine_np
                    del value

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
                    batch_start + len(batch_units),
                    len(units),
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
        natural_counts = Counter()
        natural_semantic_first = Counter()
        natural_strict = Counter()
        natural_terminated = Counter()
        for row in natural_rows:
            index = int(row["case_index"])
            output_ids = natural_outputs[index]
            answer = strict_generated_answer(
                tokenizer,
                output_ids,
                eos_ids,
            )
            terminated = generation.terminated(output_ids, eos_ids)
            classification = natural_classification(
                answer,
                row["acceptable_labels"],
                terminated,
            )
            key = (str(row["family"]), str(row["split"]))
            natural_counts[key] += 1
            natural_semantic_first[key] += int(
                classification["semantic_first"]
            )
            natural_strict[key] += int(
                classification["strict_label_only"]
            )
            natural_terminated[key] += int(terminated)
            natural_records.append({
                "schema_version": "phase1078_natural_audit.v1",
                "phase": protocol.PHASE,
                "model": model_name,
                "case_index": index,
                "unit_id": row["unit_id"],
                "family": row["family"],
                "split": row["split"],
                "state": row["state"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                "acceptable_labels": row["acceptable_labels"],
                **classification,
            })

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
                            truth_n = int(truth_count[index])
                            surface_n = int(surface_count[index])
                            shell_n = int(shell_count[index])
                            ts_n = int(truth_surface_count[index])
                            tt_n = int(truth_shell_count[index])
                            cs_n = int(cross_surface_count[index])
                            ct_n = int(cross_shell_count[index])
                            vector = direction_sum[index]
                            metric_rows.append({
                                "schema_version": (
                                    "phase1078_response_metric.v1"
                                ),
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "conditioning": conditioning_name,
                                "family": family_name,
                                "split": split_name,
                                "role": role_name,
                                **event,
                                "truth_observation_count": truth_n,
                                "truth_direction_consistency": (
                                    pairwise_direction_consistency(
                                        vector,
                                        truth_n,
                                    )
                                ),
                                "mean_truth_relative_magnitude": (
                                    float(
                                        truth_relative_sum[index]
                                        / truth_n
                                    )
                                    if truth_n else None
                                ),
                                "surface_observation_count": surface_n,
                                "mean_surface_relative_magnitude": (
                                    float(
                                        surface_relative_sum[index]
                                        / surface_n
                                    )
                                    if surface_n else None
                                ),
                                "shell_observation_count": shell_n,
                                "mean_shell_relative_magnitude": (
                                    float(
                                        shell_relative_sum[index]
                                        / shell_n
                                    )
                                    if shell_n else None
                                ),
                                "truth_surface_interaction_count": ts_n,
                                "mean_truth_surface_interaction": (
                                    float(
                                        truth_surface_relative_sum[
                                            index
                                        ] / ts_n
                                    )
                                    if ts_n else None
                                ),
                                "truth_shell_interaction_count": tt_n,
                                "mean_truth_shell_interaction": (
                                    float(
                                        truth_shell_relative_sum[
                                            index
                                        ] / tt_n
                                    )
                                    if tt_n else None
                                ),
                                "truth_cross_surface_count": cs_n,
                                "mean_truth_cross_surface_cosine": (
                                    float(
                                        cross_surface_cosine_sum[
                                            index
                                        ] / cs_n
                                    )
                                    if cs_n else None
                                ),
                                "truth_cross_shell_count": ct_n,
                                "mean_truth_cross_shell_cosine": (
                                    float(
                                        cross_shell_cosine_sum[
                                            index
                                        ] / ct_n
                                    )
                                    if ct_n else None
                                ),
                            })

        split_direction_rows = []
        discovery = split_index["discovery"]
        confirmation = split_index["confirmation"]
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for event_index, event in enumerate(events):
                    for role_name, role in role_index.items():
                        left = direction_sum[
                            conditioning,
                            family,
                            discovery,
                            event_index,
                            role,
                        ]
                        right = direction_sum[
                            conditioning,
                            family,
                            confirmation,
                            event_index,
                            role,
                        ]
                        denominator = float(
                            np.linalg.norm(left.astype(np.float64))
                            * np.linalg.norm(right.astype(np.float64))
                        )
                        split_cosine = (
                            float(np.dot(
                                left.astype(np.float64),
                                right.astype(np.float64),
                            ) / denominator)
                            if denominator > EPSILON
                            else None
                        )
                        split_direction_rows.append({
                            "schema_version": (
                                "phase1078_split_direction_repeat.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "conditioning": conditioning_name,
                            "family": family_name,
                            "role": role_name,
                            **event,
                            "discovery_confirmation_truth_direction_cosine": (
                                split_cosine
                            ),
                        })

        family_summaries = {}
        thresholds = prereg["evidence_thresholds"]
        for family in protocol.FAMILIES:
            by_split = {}
            behavior_split_passes = []
            for split in protocol.SPLITS:
                key = (family, split)
                total = int(total_cases[key])
                natural_total = int(natural_counts[key])
                candidate_rate = (
                    hit_cases[key] / total if total else 0.0
                )
                natural_rate = (
                    natural_semantic_first[key] / natural_total
                    if natural_total
                    else 0.0
                )
                split_pass = (
                    finite_cases[key] == total
                    and candidate_rate
                    >= thresholds[
                        "candidate_accuracy_for_behavior_annotation"
                    ]
                    and natural_rate
                    >= thresholds[
                        "natural_semantic_first_for_behavior_annotation"
                    ]
                )
                behavior_split_passes.append(split_pass)
                by_split[split] = {
                    "case_count": total,
                    "finite_candidate_count": int(finite_cases[key]),
                    "candidate_hit_count": int(hit_cases[key]),
                    "candidate_first_token_accuracy": candidate_rate,
                    "greedy_first_token_hit_count": int(
                        greedy_hits[key]
                    ),
                    "greedy_first_token_accuracy": (
                        greedy_hits[key] / total if total else 0.0
                    ),
                    "complete_three_factor_unit_count": int(
                        complete_units[key]
                    ),
                    "natural_audit_case_count": natural_total,
                    "natural_semantic_first_count": int(
                        natural_semantic_first[key]
                    ),
                    "natural_semantic_first_rate": natural_rate,
                    "natural_strict_count": int(natural_strict[key]),
                    "natural_terminated_count": int(
                        natural_terminated[key]
                    ),
                    "behavior_annotation_passed": split_pass,
                }
            family_summaries[family] = {
                "by_split": by_split,
                "behavior_annotation_passed": all(
                    behavior_split_passes
                ),
                "atlas_retained_regardless_of_behavior": True,
            }

        residual_indices = [
            index
            for index, event in enumerate(events)
            if event["component"] == "residual"
        ]
        residual_means = np.zeros(
            (
                len(protocol.CONDITIONINGS),
                len(protocol.FAMILIES),
                len(protocol.SPLITS),
                len(residual_indices),
                len(protocol.CAPTURE_ROLES),
                d_model,
            ),
            dtype=np.float16,
        )
        residual_counts = truth_count[
            :,
            :,
            :,
            residual_indices,
            :,
        ]
        for conditioning in range(len(protocol.CONDITIONINGS)):
            for family in range(len(protocol.FAMILIES)):
                for split in range(len(protocol.SPLITS)):
                    for local_index, event_index in enumerate(
                        residual_indices
                    ):
                        for role in range(
                            len(protocol.CAPTURE_ROLES)
                        ):
                            count = int(residual_counts[
                                conditioning,
                                family,
                                split,
                                local_index,
                                role,
                            ])
                            if count:
                                residual_means[
                                    conditioning,
                                    family,
                                    split,
                                    local_index,
                                    role,
                                ] = (
                                    direction_sum[
                                        conditioning,
                                        family,
                                        split,
                                        event_index,
                                        role,
                                    ] / count
                                ).astype(np.float16)

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
            metric_rows,
        )
        protocol.write_jsonl(
            atlas_root / "split_direction_repeat.jsonl",
            split_direction_rows,
        )
        atlas_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            atlas_root / "residual_mean_truth_directions.fp16.npz",
            mean_directions=residual_means,
            counts=residual_counts,
            relative_depth=np.asarray([
                events[index]["relative_depth"]
                for index in residual_indices
            ], dtype=np.float32),
        )
        summary = {
            "schema_version": "phase1078_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "precision": precision,
            "placement": placement,
            "case_count": len(rows),
            "unit_count": len(units),
            "event_count": len(events),
            "d_model": d_model,
            "families": family_summaries,
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "nonfinite_hidden_truth_role_count": (
                nonfinite_hidden_truth_role_count
            ),
            "primary_population": prereg["primary_population"],
            "secondary_population": prereg["secondary_population"],
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "status": "complete",
            "case_count": len(rows),
            "unit_count": len(units),
            "elapsed_seconds": summary["elapsed_seconds"],
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
