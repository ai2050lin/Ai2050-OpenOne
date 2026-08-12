#!/usr/bin/env python3
"""Run the Phase1068 FP16 reasoning generalization response atlas."""

from __future__ import annotations

import argparse
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
import phase1068_reasoning_generalization_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 2,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12


def event_definitions(n_layers: int) -> list[dict[str, Any]]:
    return [
        {
            "event_index": depth,
            "event_id": f"residual.d{depth:02d}",
            "component": "residual",
            "depth": depth,
            "relative_depth": depth / n_layers,
        }
        for depth in range(n_layers + 1)
    ]


class ResidualRoleCapture:
    def __init__(self, model, layers: list[Any]) -> None:
        self.model = model
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: Counter = Counter()
        self.handles = []

    def _hook(self, depth: int):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(
                value, torch.Tensor
            ):
                raise RuntimeError("capture was not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(
                value.shape[0], device=value.device
            )[:, None]
            self.values[depth] = value[
                batch, positions, :
            ].detach()
            self.counts[depth] += 1
            return output

        return hook

    def register(self) -> None:
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(
                self._hook(0)
            )
        )
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(self._hook(depth))
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = Counter()

    def validate(self) -> None:
        expected = set(range(len(self.layers) + 1))
        if set(self.values) != expected or any(
            count != 1 for count in self.counts.values()
        ):
            raise RuntimeError(
                f"residual capture drift: {self.counts}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


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


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denominator = (
        torch.linalg.vector_norm(a, dim=-1)
        * torch.linalg.vector_norm(b, dim=-1)
    )
    result = torch.zeros_like(
        denominator, dtype=torch.float32
    )
    valid = denominator > EPSILON
    result[valid] = (
        (a[valid] * b[valid]).sum(dim=-1)
        / denominator[valid]
    )
    return result


def pairwise_direction_consistency(
    direction_sum: np.ndarray,
    count: int,
) -> float | None:
    if count < 2:
        return None
    vector = direction_sum.astype(np.float64, copy=False)
    squared = float(np.dot(vector, vector))
    return (squared - count) / (count * (count - 1))


def natural_selection(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    for relation in protocol.RELATION_NAMES:
        eligible = [
            row
            for row in rows
            if row["relation"] == relation
            and row["state"] in {"b0_l0", "b1_l1"}
        ]
        selected.extend(generation.evenly_spaced(
            eligible, protocol.NATURAL_AUDIT_PER_RELATION
        ))
    return selected


def decode_answer(
    tokenizer,
    output_ids: list[int],
    eos_ids: set[int],
) -> str:
    return text_tools.decode_content(
        tokenizer, output_ids, eos_ids
    )


def metric_key(row: dict[str, Any], axis: str) -> tuple[Any, ...]:
    if axis == "relation":
        return ("relation", row["relation"])
    if axis == "relation_chain":
        return (
            "relation_chain",
            row["relation"],
            int(row["chain_length"]),
        )
    if axis == "relation_query":
        return (
            "relation_query",
            row["relation"],
            row["query_type"],
        )
    if axis == "relation_layout":
        return (
            "relation_layout",
            row["relation"],
            row["layout"],
        )
    raise ValueError(axis)


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1068 protocol audit failed")
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
        by_state = {
            str(row["state"]): row for row in values
        }
        if set(by_state) != set(protocol.STATES):
            raise RuntimeError(f"incomplete unit: {unit_id}")
        reference = values[0]
        units.append({
            "unit_id": unit_id,
            "relation": reference["relation"],
            "chain_length": int(reference["chain_length"]),
            "query_type": reference["query_type"],
            "layout": reference["layout"],
            "task_kind": reference["task_kind"],
            "split": reference["split"],
            "response_buckets": list(
                reference["response_buckets"]
            ),
            "states": by_state,
        })

    started = time.time()
    model = tokenizer = capture = None
    try:
        model, tokenizer, device, placement = load_fp16(
            model_name
        )
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        events = event_definitions(len(layers))
        d_model = int(
            model.get_input_embeddings().weight.shape[1]
        )
        bucket_index = {
            value: index
            for index, value in enumerate(
                prereg["response_buckets"]
            )
        }
        split_index = {
            value: index
            for index, value in enumerate(protocol.SPLITS)
        }
        role_index = {
            value: index
            for index, value in enumerate(
                protocol.CAPTURE_ROLES
            )
        }
        direction_bucket_ids = [
            value
            for value in prereg["response_buckets"]
            if value.startswith("relation_query:")
        ]
        direction_bucket_index = {
            value: index
            for index, value in enumerate(direction_bucket_ids)
        }
        shape = (
            len(bucket_index),
            len(protocol.SPLITS),
            len(events),
            len(protocol.CAPTURE_ROLES),
        )
        semantic_relative_sum = np.zeros(
            shape, dtype=np.float64
        )
        semantic_count = np.zeros(shape, dtype=np.int32)
        surface_relative_sum = np.zeros(
            shape, dtype=np.float64
        )
        surface_count = np.zeros(shape, dtype=np.int32)
        branch_cosine_sum = np.zeros(
            shape, dtype=np.float64
        )
        branch_cosine_count = np.zeros(
            shape, dtype=np.int32
        )
        interaction_relative_sum = np.zeros(
            shape, dtype=np.float64
        )
        direction_sum = np.zeros(
            (
                len(direction_bucket_ids),
                len(protocol.SPLITS),
                len(events),
                d_model,
            ),
            dtype=np.float32,
        )
        direction_count = np.zeros(
            (
                len(direction_bucket_ids),
                len(protocol.SPLITS),
                len(events),
            ),
            dtype=np.int32,
        )

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
        valid_pairs = Counter()
        complete_units = Counter()
        nonfinite_candidate_count = 0
        identity_maximum = 0.0

        capture = ResidualRoleCapture(model, layers)
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
                forward_rows = []
                unit_offsets = []
                for unit in batch_units:
                    offset = len(forward_rows)
                    forward_rows.extend(
                        unit["states"][state]
                        for state in state_order
                    )
                    forward_rows.append(
                        unit["states"]["b0_l0"]
                    )
                    unit_offsets.append(offset)
                (
                    input_ids,
                    attention_mask,
                    lengths,
                    positions,
                ) = pad_rows(
                    forward_rows, int(pad_id), device
                )
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

                for unit, offset in zip(
                    batch_units, unit_offsets
                ):
                    local_hits = {}
                    for local_index, state in enumerate(
                        state_order
                    ):
                        row = unit["states"][state]
                        values = last_logits[
                            offset + local_index
                        ]
                        class_scores = {}
                        for class_name in ("b0", "b1"):
                            token_ids = torch.tensor(
                                row[
                                    "candidate_first_token_ids"
                                ][class_name],
                                dtype=torch.long,
                                device=values.device,
                            )
                            class_scores[class_name] = float(
                                values[token_ids].max().item()
                            )
                        expected = str(row["expected_class"])
                        other = (
                            "b1" if expected == "b0" else "b0"
                        )
                        margin = (
                            class_scores[expected]
                            - class_scores[other]
                        )
                        finite = all(
                            math.isfinite(value)
                            for value in class_scores.values()
                        ) and math.isfinite(margin)
                        nonfinite_candidate_count += int(not finite)
                        hit = finite and margin > 0.0
                        greedy_token = int(
                            torch.argmax(values).item()
                        )
                        greedy_hit = greedy_token in set(
                            int(value)
                            for value in row[
                                "candidate_first_token_ids"
                            ][expected]
                        )
                        index = int(
                            row["semantic_case_index"]
                        )
                        case_hit[index] = hit
                        local_hits[state] = hit
                        for axis in (
                            "relation",
                            "relation_chain",
                            "relation_query",
                            "relation_layout",
                        ):
                            key = metric_key(row, axis)
                            behavior_total[key] += 1
                            behavior_hit[key] += int(hit)
                            behavior_greedy[key] += int(
                                greedy_hit
                            )
                        behavior_records.append({
                            "schema_version": (
                                "phase1068_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "semantic_case_index": index,
                            "unit_id": row["unit_id"],
                            "relation": row["relation"],
                            "chain_length": row["chain_length"],
                            "query_type": row["query_type"],
                            "layout": row["layout"],
                            "task_kind": row["task_kind"],
                            "split": row["split"],
                            "state": state,
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
                    for lexical in (0, 1):
                        if (
                            local_hits[f"b0_l{lexical}"]
                            and local_hits[f"b1_l{lexical}"]
                        ):
                            valid_pairs[
                                ("relation", unit["relation"])
                            ] += 1
                            valid_pairs[
                                (
                                    "relation_chain",
                                    unit["relation"],
                                    unit["chain_length"],
                                )
                            ] += 1
                            valid_pairs[
                                (
                                    "relation_query",
                                    unit["relation"],
                                    unit["query_type"],
                                )
                            ] += 1
                            valid_pairs[
                                (
                                    "relation_layout",
                                    unit["relation"],
                                    unit["layout"],
                                )
                            ] += 1
                    if all(local_hits.values()):
                        complete_units[
                            ("relation", unit["relation"])
                        ] += 1

                for event_index, event in enumerate(events):
                    value = capture.values[
                        int(event["depth"])
                    ].float()
                    for unit, offset in zip(
                        batch_units, unit_offsets
                    ):
                        split = split_index[unit["split"]]
                        bucket_indices = [
                            bucket_index[value]
                            for value in unit[
                                "response_buckets"
                            ]
                        ]
                        direction_bucket = direction_bucket_index[
                            (
                                f"relation_query:{unit['relation']}:"
                                f"{unit['query_type']}"
                            )
                        ]
                        states = {
                            state: value[offset + local_index]
                            for local_index, state in enumerate(
                                state_order
                            )
                        }
                        identity = value[
                            offset + len(state_order)
                        ]
                        identity_maximum = max(
                            identity_maximum,
                            float(torch.max(torch.abs(
                                identity - states["b0_l0"]
                            )).item()),
                        )
                        deltas = {
                            0: (
                                states["b1_l0"]
                                - states["b0_l0"]
                            ),
                            1: (
                                states["b1_l1"]
                                - states["b0_l1"]
                            ),
                        }
                        for lexical, delta in deltas.items():
                            pair_valid = (
                                case_hit[int(unit["states"][
                                    f"b0_l{lexical}"
                                ]["semantic_case_index"])]
                                and case_hit[int(unit["states"][
                                    f"b1_l{lexical}"
                                ]["semantic_case_index"])]
                            )
                            if not pair_valid:
                                continue
                            base = 0.5 * (
                                torch.linalg.vector_norm(
                                    states[f"b0_l{lexical}"],
                                    dim=-1,
                                )
                                + torch.linalg.vector_norm(
                                    states[f"b1_l{lexical}"],
                                    dim=-1,
                                )
                            )
                            norms = torch.linalg.vector_norm(
                                delta, dim=-1
                            )
                            relative = torch.zeros_like(norms)
                            valid_base = base > EPSILON
                            relative[valid_base] = (
                                norms[valid_base]
                                / base[valid_base]
                            )
                            relative_np = (
                                relative.cpu().numpy()
                            )
                            valid_np = (
                                (norms > EPSILON)
                                .cpu()
                                .numpy()
                                .astype(np.int32)
                            )
                            for bucket in bucket_indices:
                                semantic_relative_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += relative_np
                                semantic_count[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += valid_np
                            answer_role = role_index[
                                "answer_boundary"
                            ]
                            answer_norm = norms[answer_role]
                            if answer_norm > EPSILON:
                                answer_direction = (
                                    delta[answer_role]
                                    / answer_norm
                                )
                                direction_sum[
                                    direction_bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += (
                                    answer_direction
                                    .cpu()
                                    .numpy()
                                )
                                direction_count[
                                    direction_bucket,
                                    split,
                                    event_index,
                                ] += 1

                        for semantic_branch in (0, 1):
                            left = unit["states"][
                                f"b{semantic_branch}_l0"
                            ]
                            right = unit["states"][
                                f"b{semantic_branch}_l1"
                            ]
                            if not (
                                case_hit[int(left[
                                    "semantic_case_index"
                                ])]
                                and case_hit[int(right[
                                    "semantic_case_index"
                                ])]
                            ):
                                continue
                            surface = (
                                states[
                                    f"b{semantic_branch}_l1"
                                ]
                                - states[
                                    f"b{semantic_branch}_l0"
                                ]
                            )
                            base = 0.5 * (
                                torch.linalg.vector_norm(
                                    states[
                                        f"b{semantic_branch}_l0"
                                    ],
                                    dim=-1,
                                )
                                + torch.linalg.vector_norm(
                                    states[
                                        f"b{semantic_branch}_l1"
                                    ],
                                    dim=-1,
                                )
                            )
                            norms = torch.linalg.vector_norm(
                                surface, dim=-1
                            )
                            relative = torch.zeros_like(norms)
                            valid = base > EPSILON
                            relative[valid] = (
                                norms[valid] / base[valid]
                            )
                            relative_np = relative.cpu().numpy()
                            valid_np = (
                                valid.cpu().numpy().astype(np.int32)
                            )
                            for bucket in bucket_indices:
                                surface_relative_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += relative_np
                                surface_count[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += valid_np

                        if all(
                            case_hit[int(unit["states"][state][
                                "semantic_case_index"
                            ])]
                            for state in state_order
                        ):
                            branch_cos = cosine(
                                deltas[0], deltas[1]
                            )
                            denominator = (
                                torch.linalg.vector_norm(
                                    deltas[0], dim=-1
                                )
                                + torch.linalg.vector_norm(
                                    deltas[1], dim=-1
                                )
                            )
                            interaction = (
                                torch.linalg.vector_norm(
                                    deltas[1] - deltas[0],
                                    dim=-1,
                                )
                            )
                            relative_interaction = torch.zeros_like(
                                interaction
                            )
                            valid = denominator > EPSILON
                            relative_interaction[valid] = (
                                interaction[valid]
                                / denominator[valid]
                            )
                            branch_np = branch_cos.cpu().numpy()
                            interaction_np = (
                                relative_interaction
                                .cpu()
                                .numpy()
                            )
                            for bucket in bucket_indices:
                                branch_cosine_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += branch_np
                                branch_cosine_count[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
                                ] += 1
                                interaction_relative_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    :,
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
                    batch_start + len(batch_units), len(units)
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

        eos_ids = set(
            eos_tools.eos_token_ids(model, tokenizer)
        )
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
        natural_exact = Counter()
        natural_terminated = Counter()
        for row in natural_rows:
            index = int(row["semantic_case_index"])
            output_ids = natural_outputs[index]
            answer = decode_answer(
                tokenizer, output_ids, eos_ids
            )
            acceptable = {
                text_tools.normalize_text(str(value))
                for value in row["acceptable_labels"]
            }
            terminated = generation.terminated(
                output_ids, eos_ids
            )
            exact = terminated and answer in acceptable
            relation = str(row["relation"])
            natural_total[relation] += 1
            natural_exact[relation] += int(exact)
            natural_terminated[relation] += int(terminated)
            natural_records.append({
                "schema_version": (
                    "phase1068_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "unit_id": row["unit_id"],
                "relation": relation,
                "chain_length": row["chain_length"],
                "query_type": row["query_type"],
                "layout": row["layout"],
                "state": row["state"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                "acceptable_labels": sorted(acceptable),
                "terminated": terminated,
                "exact": exact,
            })

        metric_rows = []
        for bucket_id, bucket in bucket_index.items():
            for split_name, split in split_index.items():
                for event_index, event in enumerate(events):
                    for role_name, role in role_index.items():
                        semantic_n = int(semantic_count[
                            bucket, split, event_index, role
                        ])
                        surface_n = int(surface_count[
                            bucket, split, event_index, role
                        ])
                        complete_n = int(
                            branch_cosine_count[
                                bucket,
                                split,
                                event_index,
                                role,
                            ]
                        )
                        metric_rows.append({
                            "schema_version": (
                                "phase1068_response_metric.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "bucket_id": bucket_id,
                            "split": split_name,
                            "event_id": event["event_id"],
                            "depth": event["depth"],
                            "relative_depth": event[
                                "relative_depth"
                            ],
                            "role": role_name,
                            "semantic_pair_count": semantic_n,
                            "mean_semantic_relative_magnitude": (
                                float(semantic_relative_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    role,
                                ] / semantic_n)
                                if semantic_n
                                else None
                            ),
                            "surface_observation_count": surface_n,
                            "mean_surface_relative_magnitude": (
                                float(surface_relative_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    role,
                                ] / surface_n)
                                if surface_n
                                else None
                            ),
                            "complete_factorial_count": complete_n,
                            "mean_surface_branch_semantic_cosine": (
                                float(branch_cosine_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    role,
                                ] / complete_n)
                                if complete_n
                                else None
                            ),
                            "mean_interaction_relative_magnitude": (
                                float(interaction_relative_sum[
                                    bucket,
                                    split,
                                    event_index,
                                    role,
                                ] / complete_n)
                                if complete_n
                                else None
                            ),
                        })

        cross_template_rows = []
        mean_directions = np.zeros_like(
            direction_sum, dtype=np.float16
        )
        for bucket_id, bucket in direction_bucket_index.items():
            for split in range(len(protocol.SPLITS)):
                for event_index in range(len(events)):
                    count = int(direction_count[
                        bucket, split, event_index
                    ])
                    if count:
                        mean_directions[
                            bucket, split, event_index, :
                        ] = (
                            direction_sum[
                                bucket, split, event_index, :
                            ] / count
                        ).astype(np.float16)
            for event_index, event in enumerate(events):
                left_count = int(direction_count[
                    bucket,
                    split_index["discovery"],
                    event_index,
                ])
                right_count = int(direction_count[
                    bucket,
                    split_index["confirmation"],
                    event_index,
                ])
                left = direction_sum[
                    bucket,
                    split_index["discovery"],
                    event_index,
                    :,
                ]
                right = direction_sum[
                    bucket,
                    split_index["confirmation"],
                    event_index,
                    :,
                ]
                denominator = float(
                    np.linalg.norm(left) * np.linalg.norm(right)
                )
                cross_cosine = (
                    float(np.dot(left, right) / denominator)
                    if (
                        left_count
                        and right_count
                        and denominator > EPSILON
                    )
                    else None
                )
                cross_template_rows.append({
                    "schema_version": (
                        "phase1068_cross_template_direction.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "bucket_id": bucket_id,
                    "event_id": event["event_id"],
                    "depth": event["depth"],
                    "relative_depth": event["relative_depth"],
                    "role": "answer_boundary",
                    "discovery_pair_count": left_count,
                    "confirmation_pair_count": right_count,
                    "discovery_confirmation_direction_cosine": (
                        cross_cosine
                    ),
                    "discovery_direction_consistency": (
                        pairwise_direction_consistency(
                            left, left_count
                        )
                    ),
                    "confirmation_direction_consistency": (
                        pairwise_direction_consistency(
                            right, right_count
                        )
                    ),
                })

        relation_summaries = {}
        for relation in protocol.RELATION_NAMES:
            relation_key = ("relation", relation)
            total = behavior_total[relation_key]
            hit_count = behavior_hit[relation_key]
            valid_count = valid_pairs[relation_key]
            natural_count = natural_total[relation]
            natural_rate = (
                natural_exact[relation] / natural_count
                if natural_count
                else 0.0
            )
            by_chain = {}
            for chain_length in protocol.CHAIN_LENGTHS:
                key = (
                    "relation_chain",
                    relation,
                    chain_length,
                )
                by_chain[str(chain_length)] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key]
                        else 0.0
                    ),
                    "valid_semantic_pair_count": (
                        valid_pairs[key]
                    ),
                }
            by_query = {}
            for query_type in protocol.QUERY_TYPES:
                key = (
                    "relation_query",
                    relation,
                    query_type,
                )
                by_query[query_type] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key]
                        else 0.0
                    ),
                    "valid_semantic_pair_count": (
                        valid_pairs[key]
                    ),
                }
            by_layout = {}
            for layout in protocol.LAYOUTS:
                key = (
                    "relation_layout",
                    relation,
                    layout,
                )
                by_layout[layout] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key]
                        else 0.0
                    ),
                    "valid_semantic_pair_count": (
                        valid_pairs[key]
                    ),
                }
            strong_gate = bool(
                total
                and hit_count / total
                >= prereg["gates"][
                    "candidate_first_token_accuracy_min"
                ]
                and valid_count
                >= prereg["gates"][
                    "valid_semantic_pair_per_relation_min"
                ]
                and all(
                    row["valid_semantic_pair_count"]
                    >= prereg["gates"][
                        "valid_semantic_pair_per_chain_min"
                    ]
                    for row in by_chain.values()
                )
                and all(
                    row["valid_semantic_pair_count"]
                    >= prereg["gates"][
                        "valid_semantic_pair_per_query_min"
                    ]
                    for row in by_query.values()
                )
                and natural_rate
                >= prereg["gates"]["natural_exact_rate_min"]
            )
            relation_summaries[relation] = {
                "case_count": total,
                "candidate_hit_count": hit_count,
                "candidate_first_token_accuracy": (
                    hit_count / total if total else 0.0
                ),
                "greedy_first_token_accuracy": (
                    behavior_greedy[relation_key] / total
                    if total
                    else 0.0
                ),
                "valid_semantic_pair_count": valid_count,
                "complete_factorial_unit_count": (
                    complete_units[relation_key]
                ),
                "natural_audit_case_count": natural_count,
                "natural_audit_terminated_count": (
                    natural_terminated[relation]
                ),
                "natural_audit_exact_count": (
                    natural_exact[relation]
                ),
                "natural_audit_exact_rate": natural_rate,
                "by_chain": by_chain,
                "by_query": by_query,
                "by_layout": by_layout,
                "strong_behavior_gate_passed": strong_gate,
            }

        atlas_root = (
            protocol.OUT_ROOT / "atlas" / model_name
        )
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
            atlas_root / "cross_template_directions.jsonl",
            cross_template_rows,
        )
        np.savez_compressed(
            atlas_root / "answer_directions.fp16.npz",
            mean_directions=mean_directions,
            direction_counts=direction_count,
            bucket_ids=np.array(direction_bucket_ids),
            split_names=np.array(protocol.SPLITS),
            relative_depths=np.array([
                event["relative_depth"] for event in events
            ]),
        )
        summary = {
            "schema_version": "phase1068_model_summary.v1",
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
            "nonfinite_candidate_count": (
                nonfinite_candidate_count
            ),
            "relations": relation_summaries,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg[
                "interpretation_limits"
            ],
        }
        protocol.write_json(
            atlas_root / "summary.json", summary
        )
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "relations": {
                relation: {
                    "candidate_accuracy": row[
                        "candidate_first_token_accuracy"
                    ],
                    "valid_pairs": row[
                        "valid_semantic_pair_count"
                    ],
                    "natural_exact": row[
                        "natural_audit_exact_rate"
                    ],
                    "strong_gate": row[
                        "strong_behavior_gate_passed"
                    ],
                }
                for relation, row in relation_summaries.items()
            },
            "identity_maximum": identity_maximum,
            "nonfinite_candidate_count": (
                nonfinite_candidate_count
            ),
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
