#!/usr/bin/env python3
"""Run the Phase1077 nonblocking FP16 multi-family response atlas."""

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
    cosine,
    event_definitions,
    pad_rows,
    pairwise_direction_consistency,
    strict_generated_answer,
)
import phase1077_nonblocking_pattern_atlas_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 2,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    per_group = protocol.NATURAL_AUDIT_CASES_PER_FAMILY_SPLIT
    for family in protocol.FAMILIES:
        for split in protocol.SPLITS:
            eligible = [
                row for row in rows
                if row["family"] == family
                and row["split"] == split
                and row["state"] in {"b0_l0", "b1_l1"}
            ]
            selected.extend(generation.evenly_spaced(
                eligible,
                per_group,
            ))
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


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1077 protocol audit failed")
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
            value: index for index, value in enumerate(protocol.FAMILIES)
        }
        split_index = {
            value: index for index, value in enumerate(protocol.SPLITS)
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
        semantic_count = np.zeros(shape, dtype=np.int32)
        semantic_relative_sum = np.zeros(shape, dtype=np.float64)
        lexical_count = np.zeros(shape, dtype=np.int32)
        lexical_relative_sum = np.zeros(shape, dtype=np.float64)
        interaction_count = np.zeros(shape, dtype=np.int32)
        cross_surface_cosine_sum = np.zeros(shape, dtype=np.float64)
        interaction_relative_sum = np.zeros(shape, dtype=np.float64)

        behavior_records = []
        total_cases = Counter()
        hit_cases = Counter()
        greedy_hits = Counter()
        finite_cases = Counter()
        complete_units = Counter()
        identity_maximum = 0.0
        nonfinite_hidden_role_count = 0
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
                    forward_rows.append(unit["states"]["b0_l0"])
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
                                "phase1077_candidate_behavior.v1"
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
                                margin if math.isfinite(margin) else None
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
                        identity_delta = identity - states["b0_l0"]
                        if torch.isfinite(identity_delta).all():
                            identity_maximum = max(
                                identity_maximum,
                                float(torch.max(torch.abs(
                                    identity_delta
                                )).item()),
                            )
                        conditioning_rows = [
                            conditioning_index["all_finite"]
                        ]
                        if batch_complete[unit["unit_id"]]:
                            conditioning_rows.append(
                                conditioning_index["behavior_complete"]
                            )

                        semantic_deltas = {
                            0: states["b1_l0"] - states["b0_l0"],
                            1: states["b1_l1"] - states["b0_l1"],
                        }
                        semantic_observations = {}
                        for lexical, delta in semantic_deltas.items():
                            direction, relative, valid = safe_relative(
                                delta,
                                states[f"b0_l{lexical}"],
                                states[f"b1_l{lexical}"],
                            )
                            nonfinite_hidden_role_count += int(
                                (~torch.isfinite(delta).all(dim=-1)).sum().item()
                            )
                            semantic_observations[lexical] = (
                                direction,
                                relative,
                                valid,
                            )
                            direction_np = direction.cpu().numpy()
                            relative_np = relative.cpu().numpy()
                            valid_np = valid.cpu().numpy().astype(np.int32)
                            for conditioning in conditioning_rows:
                                direction_sum[
                                    conditioning,
                                    family,
                                    split,
                                    event_index,
                                ] += direction_np
                                semantic_count[
                                    conditioning,
                                    family,
                                    split,
                                    event_index,
                                ] += valid_np
                                semantic_relative_sum[
                                    conditioning,
                                    family,
                                    split,
                                    event_index,
                                ] += relative_np

                        for branch in (0, 1):
                            left = states[f"b{branch}_l0"]
                            right = states[f"b{branch}_l1"]
                            lexical_delta = right - left
                            _, relative, valid = safe_relative(
                                lexical_delta,
                                left,
                                right,
                            )
                            relative_np = relative.cpu().numpy()
                            valid_np = valid.cpu().numpy().astype(np.int32)
                            for conditioning in conditioning_rows:
                                lexical_count[
                                    conditioning,
                                    family,
                                    split,
                                    event_index,
                                ] += valid_np
                                lexical_relative_sum[
                                    conditioning,
                                    family,
                                    split,
                                    event_index,
                                ] += relative_np

                        delta0 = semantic_deltas[0]
                        delta1 = semantic_deltas[1]
                        norm0 = torch.linalg.vector_norm(delta0, dim=-1)
                        norm1 = torch.linalg.vector_norm(delta1, dim=-1)
                        finite_interaction = (
                            torch.isfinite(delta0).all(dim=-1)
                            & torch.isfinite(delta1).all(dim=-1)
                            & torch.isfinite(norm0)
                            & torch.isfinite(norm1)
                        )
                        valid_interaction = (
                            finite_interaction
                            & (norm0 > EPSILON)
                            & (norm1 > EPSILON)
                        )
                        cross_cosine = cosine(delta0, delta1)
                        interaction_norm = torch.linalg.vector_norm(
                            delta1 - delta0,
                            dim=-1,
                        )
                        interaction_relative = torch.zeros_like(
                            interaction_norm,
                            dtype=torch.float32,
                        )
                        denominator = norm0 + norm1
                        interaction_relative[valid_interaction] = (
                            interaction_norm[valid_interaction]
                            / denominator[valid_interaction]
                        )
                        valid_np = valid_interaction.cpu().numpy().astype(
                            np.int32
                        )
                        cosine_np = cross_cosine.cpu().numpy()
                        interaction_np = (
                            interaction_relative.cpu().numpy()
                        )
                        for conditioning in conditioning_rows:
                            interaction_count[
                                conditioning,
                                family,
                                split,
                                event_index,
                            ] += valid_np
                            cross_surface_cosine_sum[
                                conditioning,
                                family,
                                split,
                                event_index,
                            ] += cosine_np
                            interaction_relative_sum[
                                conditioning,
                                family,
                                split,
                                event_index,
                            ] += interaction_np
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
                if completed % 30 == 0 or completed == len(units):
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
                "schema_version": "phase1077_natural_audit.v1",
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
                            semantic_n = int(semantic_count[
                                conditioning,
                                family,
                                split,
                                event_index,
                                role,
                            ])
                            lexical_n = int(lexical_count[
                                conditioning,
                                family,
                                split,
                                event_index,
                                role,
                            ])
                            interaction_n = int(interaction_count[
                                conditioning,
                                family,
                                split,
                                event_index,
                                role,
                            ])
                            vector = direction_sum[
                                conditioning,
                                family,
                                split,
                                event_index,
                                role,
                            ]
                            metric_rows.append({
                                "schema_version": (
                                    "phase1077_response_metric.v1"
                                ),
                                "phase": protocol.PHASE,
                                "model": model_name,
                                "conditioning": conditioning_name,
                                "family": family_name,
                                "split": split_name,
                                "role": role_name,
                                **event,
                                "semantic_pair_count": semantic_n,
                                "semantic_direction_consistency": (
                                    pairwise_direction_consistency(
                                        vector,
                                        semantic_n,
                                    )
                                ),
                                "mean_semantic_relative_magnitude": (
                                    float(
                                        semantic_relative_sum[
                                            conditioning,
                                            family,
                                            split,
                                            event_index,
                                            role,
                                        ] / semantic_n
                                    )
                                    if semantic_n else None
                                ),
                                "lexical_observation_count": lexical_n,
                                "mean_lexical_relative_magnitude": (
                                    float(
                                        lexical_relative_sum[
                                            conditioning,
                                            family,
                                            split,
                                            event_index,
                                            role,
                                        ] / lexical_n
                                    )
                                    if lexical_n else None
                                ),
                                "interaction_observation_count": (
                                    interaction_n
                                ),
                                "mean_semantic_cross_surface_cosine": (
                                    float(
                                        cross_surface_cosine_sum[
                                            conditioning,
                                            family,
                                            split,
                                            event_index,
                                            role,
                                        ] / interaction_n
                                    )
                                    if interaction_n else None
                                ),
                                "mean_interaction_relative_magnitude": (
                                    float(
                                        interaction_relative_sum[
                                            conditioning,
                                            family,
                                            split,
                                            event_index,
                                            role,
                                        ] / interaction_n
                                    )
                                    if interaction_n else None
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
                            if denominator > EPSILON else None
                        )
                        split_direction_rows.append({
                            "schema_version": (
                                "phase1077_split_direction_repeat.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "conditioning": conditioning_name,
                            "family": family_name,
                            "role": role_name,
                            **event,
                            "discovery_confirmation_direction_cosine": (
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
                    if natural_total else 0.0
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
                    "complete_factorial_unit_count": int(
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
        residual_counts = semantic_count[:, :, :, residual_indices, :]
        for conditioning in range(len(protocol.CONDITIONINGS)):
            for family in range(len(protocol.FAMILIES)):
                for split in range(len(protocol.SPLITS)):
                    for output_depth, event_index in enumerate(
                        residual_indices
                    ):
                        for role in range(len(protocol.CAPTURE_ROLES)):
                            vector = direction_sum[
                                conditioning,
                                family,
                                split,
                                event_index,
                                role,
                            ]
                            norm = float(np.linalg.norm(
                                vector.astype(np.float64)
                            ))
                            if norm > EPSILON:
                                residual_means[
                                    conditioning,
                                    family,
                                    split,
                                    output_depth,
                                    role,
                                ] = (vector / norm).astype(np.float16)

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
        np.savez_compressed(
            atlas_root / "residual_mean_directions.fp16.npz",
            mean_directions=residual_means,
            semantic_counts=residual_counts,
            conditionings=np.array(protocol.CONDITIONINGS),
            family_names=np.array(protocol.FAMILIES),
            split_names=np.array(protocol.SPLITS),
            role_names=np.array(protocol.CAPTURE_ROLES),
            relative_depths=np.array([
                events[index]["relative_depth"]
                for index in residual_indices
            ], dtype=np.float32),
        )
        summary = {
            "schema_version": "phase1077_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": len(layers),
                "d_model": d_model,
            },
            "case_count": len(rows),
            "unit_count": len(units),
            "event_count": len(events),
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": nonfinite_candidate_count,
            "nonfinite_hidden_role_count": (
                nonfinite_hidden_role_count
            ),
            "families": family_summaries,
            "primary_population": prereg["primary_population"],
            "secondary_population": prereg["secondary_population"],
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg["interpretation_limits"],
        }
        summary["summary_digest"] = protocol.digest(summary)
        protocol.write_json(atlas_root / "summary.json", summary)
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "behavior_annotations": {
                family: row["behavior_annotation_passed"]
                for family, row in family_summaries.items()
            },
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)
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
