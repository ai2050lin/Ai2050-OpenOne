#!/usr/bin/env python3
"""Run the Phase1069 FP16 local-coordinate reasoning atlas."""

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
import phase1069_local_coordinate_protocol as protocol


UNIT_BATCH_SIZE = {
    "qwen3": 2,
    "glm4": 1,
    "deepseek7b": 1,
}
EPSILON = 1e-12
CONDITIONS = ("all", "behavior_conditioned")
TASK_KINDS = ("direct", "transitive")


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
    result = torch.zeros_like(denominator, dtype=torch.float32)
    valid = denominator > EPSILON
    result[valid] = (
        (a[valid] * b[valid]).sum(dim=-1)
        / denominator[valid]
    )
    return result


def module_device_dtype(module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    if getattr(parameter, "is_meta", False):
        hook = getattr(module, "_hf_hook", None)
        execution_device = getattr(hook, "execution_device", None)
        weights_map = getattr(hook, "weights_map", None)
        materialized = None
        try:
            materialized = (
                weights_map["weight"]
                if weights_map is not None
                else None
            )
        except (KeyError, TypeError):
            materialized = None
        if execution_device is None or materialized is None:
            raise RuntimeError(
                f"unable to resolve offloaded module {type(module).__name__}"
            )
        return torch.device(execution_device), materialized.dtype
    return parameter.device, parameter.dtype


def final_norm_module(model):
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "norm"):
        return base.norm
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for name in ("final_layernorm", "ln_f", "norm"):
            if hasattr(transformer, name):
                return getattr(transformer, name)
    raise RuntimeError(
        f"unable to locate final norm on {type(model).__name__}"
    )


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
        for token_id in unit["states"]["b0_l0"][
            "candidate_first_token_ids"
        ][class_name]
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
        row = unit["states"]["b0_l0"]
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


class CandidateCoordinateCache:
    def __init__(
        self,
        candidates: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        self.candidates = candidates
        self.cache: dict[
            tuple[str, str], dict[str, torch.Tensor]
        ] = {}

    def get(
        self,
        unit_id: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        key = (unit_id, str(device))
        if key not in self.cache:
            self.cache[key] = {
                class_name: value.to(device=device)
                for class_name, value in self.candidates[
                    unit_id
                ].items()
            }
        return self.cache[key]


def class_margin(
    state: torch.Tensor,
    candidates: dict[str, torch.Tensor],
) -> torch.Tensor:
    b0 = state @ candidates["b0"].float()
    b1 = state @ candidates["b1"].float()
    return b1 - b0


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
        tail_class = "strict_name_only"
    elif semantic_first and punctuation_only:
        tail_class = "name_plus_punctuation"
    elif semantic_first:
        tail_class = "name_plus_extra_content"
    else:
        tail_class = "wrong_first_content"
    return {
        "normalized_text": normalized,
        "acceptable_normalized_labels": acceptable,
        "matched_label": matched_label,
        "semantic_first": semantic_first,
        "strict_name_only": strict,
        "terminated": terminated,
        "tail_class": tail_class,
    }


def safe_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    result = np.zeros_like(total, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1069 protocol audit failed")
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
            "template_index": int(reference["template_index"]),
            "mismatch_unit_id": reference["mismatch_unit_id"],
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
        relation_index = {
            value: index
            for index, value in enumerate(protocol.RELATION_NAMES)
        }
        split_index = {
            value: index
            for index, value in enumerate(protocol.SPLITS)
        }
        task_index = {
            value: index for index, value in enumerate(TASK_KINDS)
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
            len(TASK_KINDS),
            len(events),
            len(protocol.CAPTURE_ROLES),
            len(CONDITIONS),
        )
        semantic_sum = np.zeros(response_shape, dtype=np.float64)
        semantic_count = np.zeros(response_shape, dtype=np.int32)
        surface_sum = np.zeros(response_shape, dtype=np.float64)
        surface_count = np.zeros(response_shape, dtype=np.int32)
        lexical_cosine_sum = np.zeros(
            response_shape, dtype=np.float64
        )
        factorial_count = np.zeros(response_shape, dtype=np.int32)
        interaction_sum = np.zeros(
            response_shape, dtype=np.float64
        )

        readout_shape = (
            len(protocol.RELATION_NAMES),
            len(protocol.SPLITS),
            len(TASK_KINDS),
            len(protocol.QUERY_TYPES),
            len(events),
            len(CONDITIONS),
        )
        matched_shift_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        mismatch_shift_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        matched_positive = np.zeros(
            readout_shape, dtype=np.int32
        )
        mismatch_positive = np.zeros(
            readout_shape, dtype=np.int32
        )
        matched_axis_cosine_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        mismatch_axis_cosine_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        readout_count = np.zeros(
            readout_shape, dtype=np.int32
        )
        surface_shift_abs_sum = np.zeros(
            readout_shape, dtype=np.float64
        )
        surface_readout_count = np.zeros(
            readout_shape, dtype=np.int32
        )

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has no pad/eos id")

        candidates_cpu = gather_candidate_rows(model, units)
        coordinate_cache = CandidateCoordinateCache(
            candidates_cpu
        )
        final_norm = final_norm_module(model)
        norm_device, norm_dtype = module_device_dtype(final_norm)

        behavior_records = []
        case_hit: dict[int, bool] = {}
        behavior_total = Counter()
        behavior_hit = Counter()
        behavior_greedy = Counter()
        valid_pairs = Counter()
        complete_units = Counter()
        nonfinite_candidate_count = 0
        nonfinite_internal_readout_count = 0

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
                        case_index = int(
                            row["semantic_case_index"]
                        )
                        case_hit[case_index] = hit
                        local_hits[state] = hit
                        keys = (
                            ("relation", row["relation"]),
                            (
                                "relation_chain",
                                row["relation"],
                                int(row["chain_length"]),
                            ),
                            (
                                "relation_query",
                                row["relation"],
                                row["query_type"],
                            ),
                            (
                                "relation_layout",
                                row["relation"],
                                row["layout"],
                            ),
                        )
                        for key in keys:
                            behavior_total[key] += 1
                            behavior_hit[key] += int(hit)
                            behavior_greedy[key] += int(greedy_hit)
                        behavior_records.append({
                            "schema_version": (
                                "phase1069_candidate_behavior.v1"
                            ),
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "semantic_case_index": case_index,
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
                        pair_hit = (
                            local_hits[f"b0_l{lexical}"]
                            and local_hits[f"b1_l{lexical}"]
                        )
                        if pair_hit:
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
                    if all(local_hits.values()):
                        complete_units[
                            ("relation", unit["relation"])
                        ] += 1

                for event_index, event in enumerate(events):
                    value = capture.values[
                        int(event["depth"])
                    ].float()
                    answer_role = role_index["answer_boundary"]
                    normed_answer = final_norm(
                        value[:, answer_role, :]
                        .to(device=norm_device, dtype=norm_dtype)
                    ).float()
                    for unit_offset, unit in enumerate(batch_units):
                        offset = unit_offset * len(state_order)
                        relation = relation_index[unit["relation"]]
                        split = split_index[unit["split"]]
                        task = task_index[unit["task_kind"]]
                        query = query_index[unit["query_type"]]
                        states = {
                            state: value[offset + local_index]
                            for local_index, state in enumerate(
                                state_order
                            )
                        }
                        deltas = {
                            lexical: (
                                states[f"b1_l{lexical}"]
                                - states[f"b0_l{lexical}"]
                            )
                            for lexical in (0, 1)
                        }
                        all_hits = all(
                            case_hit[int(unit["states"][state][
                                "semantic_case_index"
                            ])]
                            for state in state_order
                        )
                        condition_values = [condition_index["all"]]
                        if all_hits:
                            condition_values.append(
                                condition_index[
                                    "behavior_conditioned"
                                ]
                            )

                        for lexical, delta in deltas.items():
                            pair_hit = (
                                case_hit[int(unit["states"][
                                    f"b0_l{lexical}"
                                ]["semantic_case_index"])]
                                and case_hit[int(unit["states"][
                                    f"b1_l{lexical}"
                                ]["semantic_case_index"])]
                            )
                            local_conditions = [condition_index["all"]]
                            if pair_hit:
                                local_conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            base = 0.5 * (
                                torch.linalg.vector_norm(
                                    states[f"b0_l{lexical}"], dim=-1
                                )
                                + torch.linalg.vector_norm(
                                    states[f"b1_l{lexical}"], dim=-1
                                )
                            )
                            norms = torch.linalg.vector_norm(
                                delta, dim=-1
                            )
                            relative = torch.zeros_like(norms)
                            valid = base > EPSILON
                            relative[valid] = norms[valid] / base[valid]
                            relative_np = relative.cpu().numpy()
                            valid_np = valid.cpu().numpy().astype(
                                np.int32
                            )
                            for condition in local_conditions:
                                semantic_sum[
                                    relation,
                                    split,
                                    task,
                                    event_index,
                                    :,
                                    condition,
                                ] += relative_np
                                semantic_count[
                                    relation,
                                    split,
                                    task,
                                    event_index,
                                    :,
                                    condition,
                                ] += valid_np

                        for semantic_branch in (0, 1):
                            left_state = f"b{semantic_branch}_l0"
                            right_state = f"b{semantic_branch}_l1"
                            surface = (
                                states[right_state] - states[left_state]
                            )
                            pair_hit = (
                                case_hit[int(unit["states"][
                                    left_state
                                ]["semantic_case_index"])]
                                and case_hit[int(unit["states"][
                                    right_state
                                ]["semantic_case_index"])]
                            )
                            local_conditions = [condition_index["all"]]
                            if pair_hit:
                                local_conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            base = 0.5 * (
                                torch.linalg.vector_norm(
                                    states[left_state], dim=-1
                                )
                                + torch.linalg.vector_norm(
                                    states[right_state], dim=-1
                                )
                            )
                            norms = torch.linalg.vector_norm(
                                surface, dim=-1
                            )
                            relative = torch.zeros_like(norms)
                            valid = base > EPSILON
                            relative[valid] = norms[valid] / base[valid]
                            relative_np = relative.cpu().numpy()
                            valid_np = valid.cpu().numpy().astype(
                                np.int32
                            )
                            for condition in local_conditions:
                                surface_sum[
                                    relation,
                                    split,
                                    task,
                                    event_index,
                                    :,
                                    condition,
                                ] += relative_np
                                surface_count[
                                    relation,
                                    split,
                                    task,
                                    event_index,
                                    :,
                                    condition,
                                ] += valid_np

                        branch_cos = cosine(deltas[0], deltas[1])
                        denominator = (
                            torch.linalg.vector_norm(
                                deltas[0], dim=-1
                            )
                            + torch.linalg.vector_norm(
                                deltas[1], dim=-1
                            )
                        )
                        interaction = torch.linalg.vector_norm(
                            deltas[1] - deltas[0], dim=-1
                        )
                        relative_interaction = torch.zeros_like(
                            interaction
                        )
                        valid = denominator > EPSILON
                        relative_interaction[valid] = (
                            interaction[valid] / denominator[valid]
                        )
                        branch_np = branch_cos.cpu().numpy()
                        interaction_np = (
                            relative_interaction.cpu().numpy()
                        )
                        for condition in condition_values:
                            lexical_cosine_sum[
                                relation,
                                split,
                                task,
                                event_index,
                                :,
                                condition,
                            ] += branch_np
                            interaction_sum[
                                relation,
                                split,
                                task,
                                event_index,
                                :,
                                condition,
                            ] += interaction_np
                            factorial_count[
                                relation,
                                split,
                                task,
                                event_index,
                                :,
                                condition,
                            ] += 1

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
                            state: class_margin(
                                state_value, matched_candidates
                            )
                            for state, state_value in normed_states.items()
                        }
                        mismatch_margins = {
                            state: class_margin(
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
                        answer_device = states["b0_l0"].device
                        matched_axis_raw = matched_axis.to(answer_device)
                        mismatch_axis_raw = mismatch_axis.to(answer_device)
                        for lexical, delta in deltas.items():
                            left_state = f"b0_l{lexical}"
                            right_state = f"b1_l{lexical}"
                            pair_hit = (
                                case_hit[int(unit["states"][
                                    left_state
                                ]["semantic_case_index"])]
                                and case_hit[int(unit["states"][
                                    right_state
                                ]["semantic_case_index"])]
                            )
                            local_conditions = [condition_index["all"]]
                            if pair_hit:
                                local_conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            matched_shift = float(
                                (
                                    matched_margins[right_state]
                                    - matched_margins[left_state]
                                ).item()
                            )
                            mismatch_shift = float(
                                (
                                    mismatch_margins[right_state]
                                    - mismatch_margins[left_state]
                                ).item()
                            )
                            answer_delta = delta[answer_role]
                            matched_cos = float(cosine(
                                answer_delta.unsqueeze(0),
                                matched_axis_raw.unsqueeze(0),
                            )[0].item())
                            mismatch_cos = float(cosine(
                                answer_delta.unsqueeze(0),
                                mismatch_axis_raw.unsqueeze(0),
                            )[0].item())
                            if not all(math.isfinite(item) for item in (
                                matched_shift,
                                mismatch_shift,
                                matched_cos,
                                mismatch_cos,
                            )):
                                nonfinite_internal_readout_count += 1
                                continue
                            for condition in local_conditions:
                                key = (
                                    relation,
                                    split,
                                    task,
                                    query,
                                    event_index,
                                    condition,
                                )
                                matched_shift_sum[key] += matched_shift
                                mismatch_shift_sum[key] += mismatch_shift
                                matched_positive[key] += int(
                                    matched_shift > 0.0
                                )
                                mismatch_positive[key] += int(
                                    mismatch_shift > 0.0
                                )
                                matched_axis_cosine_sum[key] += (
                                    matched_cos
                                )
                                mismatch_axis_cosine_sum[key] += (
                                    mismatch_cos
                                )
                                readout_count[key] += 1

                        for semantic_branch in (0, 1):
                            left_state = f"b{semantic_branch}_l0"
                            right_state = f"b{semantic_branch}_l1"
                            pair_hit = (
                                case_hit[int(unit["states"][
                                    left_state
                                ]["semantic_case_index"])]
                                and case_hit[int(unit["states"][
                                    right_state
                                ]["semantic_case_index"])]
                            )
                            local_conditions = [condition_index["all"]]
                            if pair_hit:
                                local_conditions.append(
                                    condition_index[
                                        "behavior_conditioned"
                                    ]
                                )
                            shift = abs(float(
                                (
                                    matched_margins[right_state]
                                    - matched_margins[left_state]
                                ).item()
                            ))
                            if not math.isfinite(shift):
                                nonfinite_internal_readout_count += 1
                                continue
                            for condition in local_conditions:
                                key = (
                                    relation,
                                    split,
                                    task,
                                    query,
                                    event_index,
                                    condition,
                                )
                                surface_shift_abs_sum[key] += shift
                                surface_readout_count[key] += 1
                    del normed_answer, value

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
            terminated = generation.terminated(
                output_ids, eos_ids
            )
            classification = natural_classification(
                answer,
                list(row["acceptable_labels"]),
                terminated,
            )
            relation = str(row["relation"])
            natural_total[relation] += 1
            natural_semantic_first[relation] += int(
                classification["semantic_first"]
            )
            natural_strict[relation] += int(
                classification["strict_name_only"]
            )
            natural_terminated[relation] += int(terminated)
            natural_tail[
                (relation, classification["tail_class"])
            ] += 1
            natural_records.append({
                "schema_version": (
                    "phase1069_natural_generation_audit.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "semantic_case_index": index,
                "unit_id": row["unit_id"],
                "relation": relation,
                "chain_length": row["chain_length"],
                "query_type": row["query_type"],
                "layout": row["layout"],
                "task_kind": row["task_kind"],
                "split": row["split"],
                "state": row["state"],
                "generated_token_ids": [
                    int(value) for value in output_ids
                ],
                "generated_text": answer,
                **classification,
            })

        semantic_mean = safe_mean(semantic_sum, semantic_count)
        surface_mean = safe_mean(surface_sum, surface_count)
        lexical_mean = safe_mean(
            lexical_cosine_sum, factorial_count
        )
        interaction_mean = safe_mean(
            interaction_sum, factorial_count
        )
        response_rows = []
        for relation_name, relation in relation_index.items():
            for split_name, split in split_index.items():
                for task_name, task in task_index.items():
                    for event_index, event in enumerate(events):
                        for role_name, role in role_index.items():
                            for condition_name, condition in (
                                condition_index.items()
                            ):
                                response_rows.append({
                                    "schema_version": (
                                        "phase1069_response_metric.v1"
                                    ),
                                    "phase": protocol.PHASE,
                                    "model": model_name,
                                    "relation": relation_name,
                                    "split": split_name,
                                    "task_kind": task_name,
                                    "event_id": event["event_id"],
                                    "depth": event["depth"],
                                    "relative_depth": event[
                                        "relative_depth"
                                    ],
                                    "role": role_name,
                                    "conditioning": condition_name,
                                    "semantic_pair_count": int(
                                        semantic_count[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ]
                                    ),
                                    "mean_semantic_relative_magnitude": (
                                        float(semantic_mean[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ])
                                    ),
                                    "surface_observation_count": int(
                                        surface_count[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ]
                                    ),
                                    "mean_surface_relative_magnitude": (
                                        float(surface_mean[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ])
                                    ),
                                    "complete_factorial_count": int(
                                        factorial_count[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ]
                                    ),
                                    "mean_lexical_semantic_cosine": (
                                        float(lexical_mean[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ])
                                    ),
                                    "mean_interaction_relative_magnitude": (
                                        float(interaction_mean[
                                            relation,
                                            split,
                                            task,
                                            event_index,
                                            role,
                                            condition,
                                        ])
                                    ),
                                })

        readout_rows = []
        for relation_name, relation in relation_index.items():
            for split_name, split in split_index.items():
                for task_name, task in task_index.items():
                    for query_name, query in query_index.items():
                        for event_index, event in enumerate(events):
                            for condition_name, condition in (
                                condition_index.items()
                            ):
                                key = (
                                    relation,
                                    split,
                                    task,
                                    query,
                                    event_index,
                                    condition,
                                )
                                count = int(readout_count[key])
                                surface_n = int(
                                    surface_readout_count[key]
                                )
                                readout_rows.append({
                                    "schema_version": (
                                        "phase1069_readout_metric.v1"
                                    ),
                                    "phase": protocol.PHASE,
                                    "model": model_name,
                                    "relation": relation_name,
                                    "split": split_name,
                                    "task_kind": task_name,
                                    "query_type": query_name,
                                    "event_id": event["event_id"],
                                    "depth": event["depth"],
                                    "relative_depth": event[
                                        "relative_depth"
                                    ],
                                    "role": "answer_boundary",
                                    "conditioning": condition_name,
                                    "semantic_pair_count": count,
                                    "mean_matched_readout_shift": (
                                        float(
                                            matched_shift_sum[key]
                                            / count
                                        ) if count else None
                                    ),
                                    "mean_mismatched_readout_shift": (
                                        float(
                                            mismatch_shift_sum[key]
                                            / count
                                        ) if count else None
                                    ),
                                    "matched_readout_positive_rate": (
                                        float(
                                            matched_positive[key]
                                            / count
                                        ) if count else None
                                    ),
                                    "mismatched_readout_positive_rate": (
                                        float(
                                            mismatch_positive[key]
                                            / count
                                        ) if count else None
                                    ),
                                    "positive_rate_gap": (
                                        float(
                                            (
                                                matched_positive[key]
                                                - mismatch_positive[key]
                                            ) / count
                                        ) if count else None
                                    ),
                                    "mean_matched_answer_axis_cosine": (
                                        float(
                                            matched_axis_cosine_sum[key]
                                            / count
                                        ) if count else None
                                    ),
                                    "mean_mismatched_answer_axis_cosine": (
                                        float(
                                            mismatch_axis_cosine_sum[key]
                                            / count
                                        ) if count else None
                                    ),
                                    "surface_observation_count": surface_n,
                                    "mean_absolute_surface_readout_shift": (
                                        float(
                                            surface_shift_abs_sum[key]
                                            / surface_n
                                        ) if surface_n else None
                                    ),
                                })

        relation_summaries = {}
        for relation in protocol.RELATION_NAMES:
            relation_key = ("relation", relation)
            total = behavior_total[relation_key]
            hit_count = behavior_hit[relation_key]
            valid_count = valid_pairs[relation_key]
            natural_count = natural_total[relation]
            by_chain = {}
            for chain_length in protocol.CHAIN_LENGTHS:
                key = (
                    "relation_chain", relation, chain_length
                )
                by_chain[str(chain_length)] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key] else 0.0
                    ),
                    "valid_semantic_pair_count": valid_pairs[key],
                }
            by_query = {}
            for query_type in protocol.QUERY_TYPES:
                key = ("relation_query", relation, query_type)
                by_query[query_type] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key] else 0.0
                    ),
                    "valid_semantic_pair_count": valid_pairs[key],
                }
            by_layout = {}
            for layout in protocol.LAYOUTS:
                key = ("relation_layout", relation, layout)
                by_layout[layout] = {
                    "case_count": behavior_total[key],
                    "candidate_accuracy": (
                        behavior_hit[key] / behavior_total[key]
                        if behavior_total[key] else 0.0
                    ),
                }
            semantic_first_rate = (
                natural_semantic_first[relation] / natural_count
                if natural_count else 0.0
            )
            strict_rate = (
                natural_strict[relation] / natural_count
                if natural_count else 0.0
            )
            terminated_rate = (
                natural_terminated[relation] / natural_count
                if natural_count else 0.0
            )
            strong_gate = bool(
                total
                and hit_count / total
                >= prereg["gates"][
                    "candidate_first_token_accuracy_min"
                ]
                and semantic_first_rate
                >= prereg["gates"][
                    "semantic_first_natural_rate_min"
                ]
                and valid_count
                >= prereg["gates"][
                    "valid_semantic_pair_per_relation_min"
                ]
                and all(
                    row["candidate_accuracy"]
                    >= prereg["gates"][
                        "per_query_candidate_accuracy_min"
                    ]
                    for row in by_query.values()
                )
                and all(
                    row["valid_semantic_pair_count"]
                    >= prereg["gates"][
                        "valid_semantic_pair_per_chain_min"
                    ]
                    for row in by_chain.values()
                )
            )
            relation_summaries[relation] = {
                "case_count": total,
                "candidate_hit_count": hit_count,
                "candidate_first_token_accuracy": (
                    hit_count / total if total else 0.0
                ),
                "greedy_first_token_accuracy": (
                    behavior_greedy[relation_key] / total
                    if total else 0.0
                ),
                "valid_semantic_pair_count": valid_count,
                "complete_factorial_unit_count": (
                    complete_units[relation_key]
                ),
                "natural_audit_case_count": natural_count,
                "semantic_first_natural_rate": (
                    semantic_first_rate
                ),
                "strict_name_only_rate": strict_rate,
                "terminated_rate": terminated_rate,
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
                "by_chain": by_chain,
                "by_query": by_query,
                "by_layout": by_layout,
                "strong_behavior_gate_passed": strong_gate,
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
            "schema_version": "phase1069_model_summary.v1",
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
            "nonfinite_candidate_count": (
                nonfinite_candidate_count
            ),
            "nonfinite_internal_readout_count": (
                nonfinite_internal_readout_count
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
                    "semantic_first": row[
                        "semantic_first_natural_rate"
                    ],
                    "strict": row["strict_name_only_rate"],
                    "strong_gate": row[
                        "strong_behavior_gate_passed"
                    ],
                }
                for relation, row in relation_summaries.items()
            },
            "nonfinite_candidate_count": (
                nonfinite_candidate_count
            ),
            "nonfinite_internal_readout_count": (
                nonfinite_internal_readout_count
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
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
