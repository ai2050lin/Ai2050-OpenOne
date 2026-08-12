#!/usr/bin/env python3
"""Run one frozen Phase1081 Latin-route atlas model in FP16."""

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
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as generation
from phase1065_multimode_response_atlas_scan import (
    RoleCapture,
    event_definitions,
    pairwise_direction_consistency,
    strict_generated_answer,
)
from phase1079_output_orthogonal_pattern_scan import (
    add_direction_field,
    add_scalar_field,
    delta_stats,
    interaction_relative,
    pad_rows,
    vector_cosine,
)
import phase1079_output_orthogonal_pattern_scan as scan_math
import phase1081_latin_route_atlas_protocol as protocol


# pad_rows is reused mechanically, but its role list must come from the
# frozen Phase1081 protocol rather than the source Phase1079 module.
scan_math.protocol = protocol


UNIT_BATCH_SIZE = {"qwen3": 1, "glm4": 1, "deepseek7b": 1}
EPSILON = 1e-12
VECTOR_FIELDS = (
    "active_route",
    "duplicate_route",
    "content_route",
    "content_label0",
    "content_label1",
    "answer",
    "query_active",
    "query_duplicate",
)
SCALAR_FIELDS = (
    "label_swap",
    "shell",
    "content_cross_label",
    "content_cross_shell",
    "active_duplicate_cosine",
    "content_answer_cosine",
)


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"\s+", " ", value)


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


def classify_generation(
    generated: str,
    target_label: str,
    distractor_label: str,
) -> dict[str, Any]:
    text = normalized_text(generated)
    target = normalized_text(target_label)
    distractor = normalized_text(distractor_label)
    semantic_first = text.startswith(target)
    if semantic_first and target and target[-1].isalnum():
        tail = text[len(target):]
        semantic_first = not tail or not tail[0].isalnum()
    target_at = label_position(text, target)
    distractor_at = label_position(text, distractor)
    target_before_distractor = target_at is not None and (
        distractor_at is None or target_at < distractor_at
    )
    return {
        "normalized_text": text,
        "acceptable_normalized_labels": [target],
        "matched_label": target if semantic_first else None,
        "semantic_first": semantic_first,
        "strict_label_only": text == target,
        "target_before_distractor": target_before_distractor,
        "target_position": target_at,
        "distractor_position": distractor_at,
    }


def generation_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            units = sorted({
                row["unit_id"] for row in rows
                if row["family"] == family and row["split"] == split
            })[:protocol.GENERATION_UNITS_PER_FAMILY_SPLIT]
            for local, unit_id in enumerate(units):
                mapping = local % 2
                query = (local // 2) % 2
                label_swap = (local // 4) % 2
                state = (
                    f"t0_cactive_m{mapping}_q{query}_w{label_swap}"
                )
                row = next(
                    row for row in rows
                    if row["unit_id"] == unit_id and row["state"] == state
                )
                selected.append({
                    **row,
                    "semantic_case_index": int(row["case_index"]),
                })
    return selected


def mean_value(
    sums: np.ndarray,
    counts: np.ndarray,
    index: tuple[int, ...],
) -> float | None:
    count = int(counts[index])
    return float(sums[index] / count) if count else None


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1081 protocol audit failed")
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, str]] = {}
    for row in rows:
        grouped[str(row["unit_id"])][str(row["state"])] = row
        unit_meta[str(row["unit_id"])] = {
            "family": str(row["family"]),
            "split": str(row["split"]),
        }
    units = []
    for unit_id in sorted(grouped):
        if set(grouped[unit_id]) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit: {unit_id}")
        units.append({
            "unit_id": unit_id,
            **unit_meta[unit_id],
            "states": grouped[unit_id],
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
            (str(row["component"]), int(row["depth"])) for row in events
        ]
        d_model = int(model.get_input_embeddings().weight.shape[1])
        conditioning_index = {
            value: index for index, value in enumerate(protocol.CONDITIONINGS)
        }
        family_index = {
            value: index for index, value in enumerate(protocol.FAMILIES)
        }
        split_index = {
            value: index for index, value in enumerate(protocol.SPLITS)
        }
        role_index = {
            value: index for index, value in enumerate(protocol.CAPTURE_ROLES)
        }
        shape = (
            len(protocol.CONDITIONINGS),
            len(protocol.FAMILIES),
            len(protocol.SPLITS),
            len(events),
            len(protocol.CAPTURE_ROLES),
        )

        def vector_arrays() -> dict[str, np.ndarray]:
            return {
                "direction_sum": np.zeros((*shape, d_model), np.float32),
                "direction_count": np.zeros(shape, np.int32),
                "relative_sum": np.zeros(shape, np.float64),
                "relative_count": np.zeros(shape, np.int32),
            }

        vector_data = {name: vector_arrays() for name in VECTOR_FIELDS}
        scalar_data = {
            name: {
                "sum": np.zeros(shape, np.float64),
                "count": np.zeros(shape, np.int32),
            }
            for name in SCALAR_FIELDS
        }

        behavior_records: list[dict[str, Any]] = []
        candidate_totals: Counter = Counter()
        candidate_hits: Counter = Counter()
        candidate_finite: Counter = Counter()
        supported_units: Counter = Counter()
        nonfinite_candidate_count = 0
        nonfinite_hidden_count = 0
        pre_query_max_abs = {
            "query_active": 0.0,
            "query_duplicate": 0.0,
        }
        identity_maximum = 0.0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        capture = RoleCapture(model, layers)
        capture.register()
        state_order = list(protocol.STATES)
        batch_size = UNIT_BATCH_SIZE[model_name]
        with torch.inference_mode():
            for batch_start in range(0, len(units), batch_size):
                batch_units = units[batch_start:batch_start + batch_size]
                forward_rows: list[dict[str, Any]] = []
                offsets: list[dict[str, int | None]] = []
                for unit in batch_units:
                    offset = len(forward_rows)
                    state_rows = [unit["states"][state] for state in state_order]
                    forward_rows.extend(state_rows)
                    identity_offset = None
                    if batch_start == 0:
                        forward_rows.append(state_rows[0])
                        identity_offset = len(forward_rows) - 1
                    offsets.append({"states": offset, "identity": identity_offset})

                input_ids, attention_mask, lengths, positions = pad_rows(
                    forward_rows, int(pad_id), device
                )
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                capture.validate()
                logits = output.logits
                final_positions = (lengths - 1).to(logits.device)
                batch_axis = torch.arange(logits.shape[0], device=logits.device)
                final_logits = logits[batch_axis, final_positions, :].float()
                del output, logits

                unit_behavior_support: dict[str, bool] = {}
                for unit, offset in zip(batch_units, offsets):
                    hits: list[int] = []
                    for local, state in enumerate(state_order):
                        row = unit["states"][state]
                        row_index = int(offset["states"]) + local
                        values = final_logits[row_index]
                        scores = {}
                        for answer_class in ("a0", "a1"):
                            token_ids = torch.tensor(
                                row["candidate_first_token_ids"][answer_class],
                                dtype=torch.long,
                                device=values.device,
                            )
                            scores[answer_class] = float(
                                values[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = "a1" if expected == "a0" else "a0"
                        margin = scores[expected] - scores[other]
                        finite = all(math.isfinite(v) for v in scores.values()) \
                            and math.isfinite(margin)
                        hit = finite and margin > 0.0
                        greedy = int(torch.argmax(values).item())
                        key = (unit["family"], unit["split"], row["panel"])
                        candidate_totals[key] += 1
                        candidate_finite[key] += int(finite)
                        candidate_hits[key] += int(hit)
                        nonfinite_candidate_count += int(not finite)
                        if row["panel"] == "active":
                            hits.append(int(hit))
                        behavior_records.append({
                            "schema_version": "phase1081_candidate_behavior.v1",
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "case_index": int(row["case_index"]),
                            "unit_id": unit["unit_id"],
                            "family": unit["family"],
                            "split": unit["split"],
                            "state": row["state"],
                            "panel": row["panel"],
                            "mapping": int(row["mapping"]),
                            "query": int(row["query"]),
                            "label_swap": int(row["label_swap"]),
                            "expected_class": expected,
                            "target_answer": row["target_answer"],
                            "candidate_scores": {
                                k: v if math.isfinite(v) else None
                                for k, v in scores.items()
                            },
                            "candidate_margin": margin if math.isfinite(margin) else None,
                            "finite_candidate": finite,
                            "candidate_hit": hit,
                            "greedy_first_token_id": greedy,
                            "greedy_first_token_text": tokenizer.decode([greedy]),
                        })
                    fraction = sum(hits) / len(hits) if hits else 0.0
                    supported = fraction >= float(
                        prereg["evidence_thresholds"][
                            "unit_behavior_support_fraction"
                        ]
                    )
                    unit_behavior_support[unit["unit_id"]] = supported
                    supported_units[(unit["family"], unit["split"])] += int(supported)

                for event_index, event_key in enumerate(event_keys):
                    values = capture.values[event_key].float()
                    for unit, offset in zip(batch_units, offsets):
                        states = {
                            state: values[int(offset["states"]) + local]
                            for local, state in enumerate(state_order)
                        }
                        if offset["identity"] is not None:
                            identity_delta = (
                                values[int(offset["identity"])] - states[state_order[0]]
                            )
                            if torch.isfinite(identity_delta).all():
                                identity_maximum = max(
                                    identity_maximum,
                                    float(identity_delta.abs().max().item()),
                                )

                        base_index = (
                            family_index[unit["family"]],
                            split_index[unit["split"]],
                            event_index,
                        )
                        conditionings = [conditioning_index["all_finite"]]
                        if unit_behavior_support[unit["unit_id"]]:
                            conditionings.append(
                                conditioning_index["behavior_supported"]
                            )

                        def state(
                            template: int,
                            panel: str,
                            mapping: int,
                            query: int,
                            label_swap: int,
                        ) -> torch.Tensor:
                            return states[
                                f"t{template}_c{panel}_m{mapping}_q{query}_w{label_swap}"
                            ]

                        def add_vector(
                            field: str,
                            left: torch.Tensor,
                            right: torch.Tensor,
                        ) -> None:
                            vector_observations[field].append((left, right))

                        vector_observations: dict[
                            str, list[tuple[torch.Tensor, torch.Tensor]]
                        ] = {field: [] for field in VECTOR_FIELDS}
                        relative_control_pairs: dict[
                            str, list[tuple[torch.Tensor, torch.Tensor]]
                        ] = {"label_swap": [], "shell": []}
                        scalar_observations: dict[
                            str, list[tuple[torch.Tensor, torch.Tensor]]
                        ] = {
                            field: []
                            for field in SCALAR_FIELDS
                            if field not in relative_control_pairs
                        }
                        prequery_observations: dict[
                            str, list[torch.Tensor]
                        ] = {
                            "query_active": [],
                            "query_duplicate": [],
                        }

                        route_states: dict[
                            tuple[int, int], dict[str, torch.Tensor]
                        ] = {}
                        answer_deltas: dict[tuple[int, int], torch.Tensor] = {}
                        for template in (0, 1):
                            for label_swap in (0, 1):
                                active_left = 0.5 * (
                                    state(template, "active", 0, 0, label_swap)
                                    + state(template, "active", 1, 0, label_swap)
                                )
                                active_right = 0.5 * (
                                    state(template, "active", 1, 1, label_swap)
                                    + state(template, "active", 0, 1, label_swap)
                                )
                                duplicate_left = 0.5 * (
                                    state(template, "duplicate", 0, 0, label_swap)
                                    + state(template, "duplicate", 1, 0, label_swap)
                                )
                                duplicate_right = 0.5 * (
                                    state(template, "duplicate", 0, 1, label_swap)
                                    + state(template, "duplicate", 1, 1, label_swap)
                                )
                                # Difference-in-differences, represented as two
                                # convex state averages so its relative magnitude
                                # keeps the same scale as the other fields.
                                content_left = 0.5 * (
                                    active_left + duplicate_right
                                )
                                content_right = 0.5 * (
                                    active_right + duplicate_left
                                )
                                route_states[(template, label_swap)] = {
                                    "active_left": active_left,
                                    "active_right": active_right,
                                    "duplicate_left": duplicate_left,
                                    "duplicate_right": duplicate_right,
                                    "content_left": content_left,
                                    "content_right": content_right,
                                }
                                add_vector(
                                    "active_route", active_left, active_right
                                )
                                add_vector(
                                    "duplicate_route",
                                    duplicate_left,
                                    duplicate_right,
                                )
                                add_vector(
                                    "content_route", content_left, content_right
                                )
                                add_vector(
                                    f"content_label{label_swap}",
                                    content_left,
                                    content_right,
                                )

                                answer0 = 0.5 * (
                                    state(template, "active", 0, 0, label_swap)
                                    + state(template, "active", 1, 1, label_swap)
                                )
                                answer1 = 0.5 * (
                                    state(template, "active", 0, 1, label_swap)
                                    + state(template, "active", 1, 0, label_swap)
                                )
                                answer_deltas[(template, label_swap)] = (
                                    answer1 - answer0
                                )
                                add_vector("answer", answer1, answer0)

                        for panel in protocol.PANELS:
                            field = f"query_{panel}"
                            for template in (0, 1):
                                for mapping in (0, 1):
                                    for label_swap in (0, 1):
                                        left = state(
                                            template, panel, mapping, 0, label_swap
                                        )
                                        right = state(
                                            template, panel, mapping, 1, label_swap
                                        )
                                        add_vector(field, left, right)
                                        prequery_observations[field].append(
                                            torch.stack([
                                                left[role_index[role_name]]
                                                - right[role_index[role_name]]
                                                for role_name in protocol.PRE_QUERY_ROLES
                                            ])
                                        )

                        for template in (0, 1):
                            for panel in protocol.PANELS:
                                for mapping in (0, 1):
                                    for query in (0, 1):
                                        left = state(
                                            template, panel, mapping, query, 0
                                        )
                                        right = state(
                                            template, panel, mapping, query, 1
                                        )
                                        relative_control_pairs[
                                            "label_swap"
                                        ].append(
                                            (left, right)
                                        )

                        for panel in protocol.PANELS:
                            for mapping in (0, 1):
                                for query in (0, 1):
                                    for label_swap in (0, 1):
                                        left = state(
                                            0, panel, mapping, query, label_swap
                                        )
                                        right = state(
                                            1, panel, mapping, query, label_swap
                                        )
                                        relative_control_pairs["shell"].append(
                                            (left, right)
                                        )

                        for template in (0, 1):
                            left_route = route_states[(template, 0)]
                            right_route = route_states[(template, 1)]
                            left_content = (
                                left_route["content_left"]
                                - left_route["content_right"]
                            )
                            right_content = (
                                right_route["content_left"]
                                - right_route["content_right"]
                            )
                            cosine, valid = vector_cosine(
                                left_content, right_content
                            )
                            scalar_observations[
                                "content_cross_label"
                            ].append(
                                (cosine, valid)
                            )
                            for label_swap in (0, 1):
                                routes = route_states[(template, label_swap)]
                                active_delta = (
                                    routes["active_left"]
                                    - routes["active_right"]
                                )
                                duplicate_delta = (
                                    routes["duplicate_left"]
                                    - routes["duplicate_right"]
                                )
                                content_delta = (
                                    routes["content_left"]
                                    - routes["content_right"]
                                )
                                cosine, valid = vector_cosine(
                                    active_delta, duplicate_delta
                                )
                                scalar_observations[
                                    "active_duplicate_cosine"
                                ].append(
                                    (cosine, valid)
                                )
                                cosine, valid = vector_cosine(
                                    content_delta,
                                    answer_deltas[(template, label_swap)],
                                )
                                scalar_observations[
                                    "content_answer_cosine"
                                ].append(
                                    (cosine, valid)
                                )

                        content_by_template = {}
                        for template in (0, 1):
                            content_by_template[template] = 0.5 * sum(
                                (
                                    route_states[(template, label_swap)][
                                        "content_left"
                                    ]
                                    - route_states[(template, label_swap)][
                                        "content_right"
                                    ]
                                    for label_swap in (0, 1)
                                ),
                                torch.zeros_like(
                                    route_states[(template, 0)]["content_left"]
                                ),
                            )
                        cosine, valid = vector_cosine(
                            content_by_template[0], content_by_template[1]
                        )
                        scalar_observations[
                            "content_cross_shell"
                        ].append(
                            (cosine, valid)
                        )

                        vector_summaries = []
                        for field in VECTOR_FIELDS:
                            observations = vector_observations[field]
                            left = torch.stack([
                                pair[0] for pair in observations
                            ])
                            right = torch.stack([
                                pair[1] for pair in observations
                            ])
                            relative, magnitude_valid, direction, (
                                direction_valid
                            ) = delta_stats(left - right, left, right)
                            vector_summaries.append((
                                direction.sum(dim=0),
                                direction_valid.sum(dim=0),
                                relative.sum(dim=0),
                                magnitude_valid.sum(dim=0),
                                (~magnitude_valid).sum(),
                            ))
                        direction_sums = torch.stack([
                            row[0] for row in vector_summaries
                        ]).cpu().numpy()
                        direction_counts = torch.stack([
                            row[1] for row in vector_summaries
                        ]).cpu().numpy().astype(np.int32)
                        relative_sums = torch.stack([
                            row[2] for row in vector_summaries
                        ]).cpu().numpy()
                        relative_counts = torch.stack([
                            row[3] for row in vector_summaries
                        ]).cpu().numpy().astype(np.int32)
                        invalid_counts = torch.stack([
                            row[4] for row in vector_summaries
                        ]).cpu().numpy()
                        for field_index, field in enumerate(VECTOR_FIELDS):
                            data = vector_data[field]
                            for conditioning in conditionings:
                                index = (conditioning, *base_index)
                                data["direction_sum"][index] += (
                                    direction_sums[field_index]
                                )
                                data["direction_count"][index] += (
                                    direction_counts[field_index]
                                )
                                data["relative_sum"][index] += (
                                    relative_sums[field_index]
                                )
                                data["relative_count"][index] += (
                                    relative_counts[field_index]
                                )
                            nonfinite_hidden_count += int(
                                invalid_counts[field_index]
                            )

                        scalar_summaries = []
                        scalar_names = []
                        for field, pairs in relative_control_pairs.items():
                            left = torch.stack([pair[0] for pair in pairs])
                            right = torch.stack([pair[1] for pair in pairs])
                            relative, valid, _, _ = delta_stats(
                                right - left, left, right
                            )
                            scalar_names.append(field)
                            scalar_summaries.append((
                                relative.sum(dim=0), valid.sum(dim=0)
                            ))
                        for field, observations in scalar_observations.items():
                            values_batch = torch.stack([
                                pair[0] for pair in observations
                            ])
                            valid_batch = torch.stack([
                                pair[1] for pair in observations
                            ])
                            scalar_names.append(field)
                            scalar_summaries.append((
                                values_batch.sum(dim=0),
                                valid_batch.sum(dim=0),
                            ))
                        scalar_sums = torch.stack([
                            row[0] for row in scalar_summaries
                        ]).cpu().numpy()
                        scalar_counts = torch.stack([
                            row[1] for row in scalar_summaries
                        ]).cpu().numpy().astype(np.int32)
                        for scalar_index, field in enumerate(scalar_names):
                            data = scalar_data[field]
                            for conditioning in conditionings:
                                index = (conditioning, *base_index)
                                data["sum"][index] += scalar_sums[scalar_index]
                                data["count"][index] += scalar_counts[
                                    scalar_index
                                ]

                        prequery_maxima = []
                        for field in ("query_active", "query_duplicate"):
                            values_batch = torch.stack(
                                prequery_observations[field]
                            )
                            finite = torch.isfinite(values_batch)
                            maximum = torch.where(
                                finite,
                                values_batch.abs(),
                                torch.zeros_like(values_batch),
                            ).max()
                            prequery_maxima.append(maximum)
                        prequery_maxima_np = torch.stack(
                            prequery_maxima
                        ).cpu().numpy()
                        for field_index, field in enumerate(
                            ("query_active", "query_duplicate")
                        ):
                            pre_query_max_abs[field] = max(
                                pre_query_max_abs[field],
                                float(prequery_maxima_np[field_index]),
                            )
                    del values

                del final_logits, input_ids, attention_mask, lengths, positions
                capture.values = {}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed = min(batch_start + len(batch_units), len(units))
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
        generation_rows = generation_selection(rows)
        generated = generation.generate_case_outputs(
            model,
            device,
            generation_rows,
            eos_ids=eos_ids,
            batch_size=bridge.PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["generation_steps"]),
        )
        generation_records: list[dict[str, Any]] = []
        generation_totals: Counter = Counter()
        generation_target_first: Counter = Counter()
        generation_target_before: Counter = Counter()
        generation_strict: Counter = Counter()
        for row in generation_rows:
            case_index = int(row["case_index"])
            output_ids = generated[case_index]
            text = strict_generated_answer(tokenizer, output_ids, eos_ids)
            distractor = str(
                row["answer_labels"][1 - int(row["answer_index"])]
            )
            classification = classify_generation(
                text, str(row["target_answer"]), distractor
            )
            key = (str(row["family"]), str(row["split"]))
            generation_totals[key] += 1
            generation_target_first[key] += int(
                classification["semantic_first"]
            )
            generation_target_before[key] += int(
                classification["target_before_distractor"]
            )
            generation_strict[key] += int(classification["strict_label_only"])
            generation_records.append({
                "schema_version": "phase1081_natural_generation.v1",
                "phase": protocol.PHASE,
                "model": model_name,
                "case_index": case_index,
                "unit_id": row["unit_id"],
                "family": row["family"],
                "split": row["split"],
                "state": row["state"],
                "target_answer": row["target_answer"],
                "distractor_answer": distractor,
                "generated_token_ids": [int(value) for value in output_ids],
                "generated_text": text,
                "terminated": generation.terminated(output_ids, eos_ids),
                **classification,
            })

        metric_rows: list[dict[str, Any]] = []
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for split_name, split in split_index.items():
                    for event_index, event in enumerate(events):
                        for role_name, role in role_index.items():
                            index = (conditioning, family, split, event_index, role)
                            record: dict[str, Any] = {
                                "schema_version": "phase1081_response_metric.v1",
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "conditioning": conditioning_name,
                                "family": family_name,
                                "split": split_name,
                                "role": role_name,
                                **event,
                            }
                            for field, data in vector_data.items():
                                direction_n = int(data["direction_count"][index])
                                record[f"{field}_magnitude_count"] = int(
                                    data["relative_count"][index]
                                )
                                record[f"mean_{field}_relative_magnitude"] = mean_value(
                                    data["relative_sum"], data["relative_count"], index
                                )
                                record[f"{field}_direction_count"] = direction_n
                                record[f"{field}_direction_consistency"] = (
                                    pairwise_direction_consistency(
                                        data["direction_sum"][index], direction_n
                                    )
                                )
                            for field, data in scalar_data.items():
                                record[f"{field}_count"] = int(data["count"][index])
                                record[f"mean_{field}"] = mean_value(
                                    data["sum"], data["count"], index
                                )
                            metric_rows.append(record)

        split_direction_rows: list[dict[str, Any]] = []
        discovery = split_index["discovery"]
        confirmation = split_index["confirmation"]
        for conditioning_name, conditioning in conditioning_index.items():
            for family_name, family in family_index.items():
                for event_index, event in enumerate(events):
                    for role_name, role in role_index.items():
                        record = {
                            "schema_version": "phase1081_split_direction_repeat.v1",
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "conditioning": conditioning_name,
                            "family": family_name,
                            "role": role_name,
                            **event,
                        }
                        for field, data in vector_data.items():
                            left_index = (
                                conditioning, family, discovery, event_index, role
                            )
                            right_index = (
                                conditioning, family, confirmation, event_index, role
                            )
                            left = data["direction_sum"][left_index].astype(
                                np.float64, copy=False
                            )
                            right = data["direction_sum"][right_index].astype(
                                np.float64, copy=False
                            )
                            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                            record[f"{field}_discovery_confirmation_cosine"] = (
                                float(np.dot(left, right) / denominator)
                                if denominator > EPSILON else None
                            )
                            record[f"{field}_discovery_count"] = int(
                                data["direction_count"][left_index]
                            )
                            record[f"{field}_confirmation_count"] = int(
                                data["direction_count"][right_index]
                            )
                        split_direction_rows.append(record)

        behavior_summary: dict[str, Any] = {}
        for family in protocol.FAMILIES:
            behavior_summary[family] = {}
            for split in protocol.SPLITS:
                split_row: dict[str, Any] = {}
                for panel in protocol.PANELS:
                    key = (family, split, panel)
                    total = int(candidate_totals[key])
                    split_row[panel] = {
                        "candidate_count": total,
                        "candidate_finite_count": int(candidate_finite[key]),
                        "candidate_hit_count": int(candidate_hits[key]),
                        "candidate_accuracy": (
                            candidate_hits[key] / total if total else None
                        ),
                    }
                generation_key = (family, split)
                generated_total = int(generation_totals[generation_key])
                split_row["natural_generation"] = {
                    "generation_case_count": generated_total,
                    "generation_semantic_first_count": int(
                        generation_target_first[generation_key]
                    ),
                    "generation_semantic_first_accuracy": (
                        generation_target_first[generation_key] / generated_total
                        if generated_total else None
                    ),
                    "generation_target_before_distractor_count": int(
                        generation_target_before[generation_key]
                    ),
                    "generation_target_before_distractor_accuracy": (
                        generation_target_before[generation_key] / generated_total
                        if generated_total else None
                    ),
                    "generation_strict_count": int(
                        generation_strict[generation_key]
                    ),
                }
                split_row["behavior_supported_unit_count"] = int(
                    supported_units[(family, split)]
                )
                behavior_summary[family][split] = split_row

        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_jsonl(atlas_root / "candidate_behavior.jsonl", behavior_records)
        protocol.write_jsonl(atlas_root / "natural_generation.jsonl", generation_records)
        protocol.write_jsonl(atlas_root / "response_metrics.jsonl", metric_rows)
        protocol.write_jsonl(
            atlas_root / "split_direction_repeat.jsonl", split_direction_rows
        )

        elapsed = time.time() - started
        summary = {
            "schema_version": "phase1081_model_summary.v1",
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
            "pre_query_max_abs": pre_query_max_abs,
            "pre_query_global_max_abs": max(pre_query_max_abs.values()),
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "nonfinite_hidden_magnitude_role_count": nonfinite_hidden_count,
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
            "pre_query_global_max_abs": summary["pre_query_global_max_abs"],
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
