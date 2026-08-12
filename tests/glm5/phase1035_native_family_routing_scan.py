#!/usr/bin/env python3
"""Scan the Phase1035 native lexical-family-query response atlas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1035_native_family_routing_protocol as protocol


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def safe_cos(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left.float() * right.float(), dim=-1)
    denominator = (
        torch.linalg.vector_norm(left.float(), dim=-1)
        * torch.linalg.vector_norm(right.float(), dim=-1)
    )
    return numerator / torch.clamp(denominator, min=EPS)


def gather_span_means(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    positions = positions.to(hidden.device)
    masks = masks.to(hidden.device)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    batch = batch[:, None, None].expand_as(positions)
    values = hidden[batch, positions, :]
    weights = masks[..., None].to(values.dtype)
    return (values * weights).sum(dim=2) / torch.clamp(
        weights.sum(dim=2), min=1
    )


def grouped(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] % len(protocol.WORLD_FACTORS):
        raise RuntimeError("batch does not preserve eight-world units")
    return values.reshape(
        values.shape[0] // len(protocol.WORLD_FACTORS),
        len(protocol.WORLD_FACTORS),
        *values.shape[1:],
    )


def world_index(binding: int, query: int, lexical: int) -> int:
    return binding + 2 * query + 4 * lexical


def factor_metrics(states: torch.Tensor) -> torch.Tensor:
    """Return observational B, Q, L and interaction measurements.

    ``states`` has shape [unit, world, anchor, d_model].  All formulas below
    are measurement definitions; they do not assume the model internally
    performs explicit subtraction.
    """

    def state(binding: int, query: int, lexical: int) -> torch.Tensor:
        return states[:, world_index(binding, query, lexical)]

    scale = torch.linalg.vector_norm(states.float(), dim=-1).mean(dim=1)

    def rel_norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value.float(), dim=-1) / torch.clamp(
            scale, min=EPS
        )

    binding_q0 = [
        state(1, 0, lexical) - state(0, 0, lexical)
        for lexical in (0, 1)
    ]
    binding_q1 = [
        state(1, 1, lexical) - state(0, 1, lexical)
        for lexical in (0, 1)
    ]
    query_b0 = [
        state(0, 1, lexical) - state(0, 0, lexical)
        for lexical in (0, 1)
    ]
    query_b1 = [
        state(1, 1, lexical) - state(1, 0, lexical)
        for lexical in (0, 1)
    ]
    lexical_effects = [
        state(binding, query, 1) - state(binding, query, 0)
        for binding in (0, 1)
        for query in (0, 1)
    ]
    bq_interactions = [
        (
            state(1, 1, lexical)
            - state(1, 0, lexical)
            - state(0, 1, lexical)
            + state(0, 0, lexical)
        )
        for lexical in (0, 1)
    ]
    bql_interaction = bq_interactions[1] - bq_interactions[0]

    def average(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values, dim=0).mean(dim=0)

    binding_q0_norm = average([rel_norm(value) for value in binding_q0])
    binding_q1_norm = average([rel_norm(value) for value in binding_q1])
    query_b0_norm = average([rel_norm(value) for value in query_b0])
    query_b1_norm = average([rel_norm(value) for value in query_b1])
    lexical_norm = average([rel_norm(value) for value in lexical_effects])
    bq_norm = average([rel_norm(value) for value in bq_interactions])
    bql_norm = rel_norm(bql_interaction)
    binding_query_cosine = average([
        safe_cos(binding_q0[index], binding_q1[index])
        for index in (0, 1)
    ])
    bq_member_invariance = safe_cos(
        bq_interactions[0], bq_interactions[1]
    )
    binding_member_q0 = safe_cos(binding_q0[0], binding_q0[1])
    binding_member_q1 = safe_cos(binding_q1[0], binding_q1[1])
    query_member_b0 = safe_cos(query_b0[0], query_b0[1])
    query_member_b1 = safe_cos(query_b1[0], query_b1[1])
    raw_bq = average([
        torch.linalg.vector_norm(value.float(), dim=-1)
        for value in bq_interactions
    ])
    raw_lexical = average([
        torch.linalg.vector_norm(value.float(), dim=-1)
        for value in lexical_effects
    ])
    bq_to_lexical = raw_bq / torch.clamp(raw_lexical, min=EPS)
    return torch.stack(
        [
            binding_q0_norm,
            binding_q1_norm,
            query_b0_norm,
            query_b1_norm,
            lexical_norm,
            bq_norm,
            bql_norm,
            binding_query_cosine,
            bq_member_invariance,
            binding_member_q0,
            binding_member_q1,
            query_member_b0,
            query_member_b1,
            bq_to_lexical,
        ],
        dim=-1,
    )


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
]:
    expected_worlds = [
        f"{binding}{query}{lexical}"
        for binding, query, lexical in protocol.WORLD_FACTORS
    ]
    world_count = len(expected_worlds)
    if len(rows) % world_count:
        raise RuntimeError("batch size must be a multiple of eight")
    unit_indices = []
    for start in range(0, len(rows), world_count):
        group = rows[start:start + world_count]
        if [row["world"] for row in group] != expected_worlds:
            raise RuntimeError("eight-world order drift")
        if len({int(row["unit_index"]) for row in group}) != 1:
            raise RuntimeError("unit rows are not contiguous")
        unit_indices.append(int(group[0]["unit_index"]))

    width = max(len(row["input_ids"]) for row in rows)
    max_span = max(
        int(end) - int(start) + 1
        for row in rows
        for start, end in row["anchor_spans"].values()
    )
    ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    anchor_positions = torch.zeros(
        (len(rows), len(protocol.ANCHORS), max_span), dtype=torch.long
    )
    anchor_masks = torch.zeros_like(anchor_positions, dtype=torch.bool)
    pre_positions = torch.empty(len(rows), dtype=torch.long)

    for row_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        for anchor_index, anchor in enumerate(protocol.ANCHORS):
            start, end = (
                int(value) for value in row["anchor_spans"][anchor]
            )
            positions = list(range(start, end + 1))
            anchor_positions[
                row_index, anchor_index, :len(positions)
            ] = torch.tensor(positions, dtype=torch.long)
            anchor_masks[
                row_index, anchor_index, :len(positions)
            ] = True
        pre_positions[row_index] = int(
            row["anchor_spans"]["pre_output"][1]
        )

    return (
        ids.to(device),
        attention_mask.to(device),
        anchor_positions,
        anchor_masks,
        pre_positions,
        np.asarray(unit_indices, dtype=np.int64),
        np.asarray(
            [int(row["case_index"]) for row in rows], dtype=np.int64
        ),
        torch.tensor(
            [int(row["expected_index"]) for row in rows],
            dtype=torch.long,
        ),
    )


class NativeFactorCapture:
    def __init__(
        self,
        layers: list[Any],
        residual: np.memmap,
        component: np.memmap,
        closure: np.memmap,
        boundary_states: np.memmap,
        boundary_depths: tuple[int, ...],
    ):
        self.layers = layers
        self.residual = residual
        self.component = component
        self.closure = closure
        self.boundary_states = boundary_states
        self.boundary_depths = boundary_depths
        self.boundary_slots = {
            depth: index for index, depth in enumerate(boundary_depths)
        }
        self.anchor_positions: torch.Tensor | None = None
        self.anchor_masks: torch.Tensor | None = None
        self.unit_indices: np.ndarray | None = None
        self.case_indices: np.ndarray | None = None
        self.current: dict[int, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def begin(
        self,
        anchor_positions: torch.Tensor,
        anchor_masks: torch.Tensor,
        unit_indices: np.ndarray,
        case_indices: np.ndarray,
    ) -> None:
        self.anchor_positions = anchor_positions
        self.anchor_masks = anchor_masks
        self.unit_indices = unit_indices
        self.case_indices = case_indices
        self.current = {}
        self.counts = defaultdict(int)

    def _anchor_states(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.anchor_positions is None or self.anchor_masks is None:
            raise RuntimeError("capture anchors missing")
        return grouped(
            gather_span_means(
                hidden, self.anchor_positions, self.anchor_masks
            )
        ).detach()

    def _pre_hook(self, depth_index: int):
        def hook(module, args):
            if self.unit_indices is None:
                raise RuntimeError("capture unit indices missing")
            states = self._anchor_states(args[0])
            if depth_index == 0:
                self.residual[
                    self.unit_indices, 0, :, :
                ] = factor_metrics(states).float().cpu().numpy()
            self.current[depth_index] = {"input": states}
            self.counts[f"{depth_index}/pre"] += 1
        return hook

    def _component_hook(self, depth_index: int, component_index: int):
        component_name = protocol.COMPONENTS[component_index]

        def hook(module, args, output):
            if self.unit_indices is None:
                raise RuntimeError("capture unit indices missing")
            states = self._anchor_states(output_tensor(output))
            self.component[
                self.unit_indices, depth_index, component_index, :, :
            ] = factor_metrics(states).float().cpu().numpy()
            self.current[depth_index][component_name] = states
            self.counts[f"{depth_index}/{component_name}"] += 1
            return output
        return hook

    def _layer_hook(self, depth_index: int):
        def hook(module, args, output):
            if self.unit_indices is None or self.case_indices is None:
                raise RuntimeError("capture indices missing")
            states = self._anchor_states(output_tensor(output))
            self.residual[
                self.unit_indices, depth_index + 1, :, :
            ] = factor_metrics(states).float().cpu().numpy()

            current = self.current[depth_index]
            input_states = current["input"]
            attention = current["attention"]
            mlp = current["mlp"]
            error = states - input_states - attention - mlp
            transition = states - input_states
            relative_error = (
                torch.linalg.vector_norm(error.float(), dim=-1)
                / torch.clamp(
                    torch.linalg.vector_norm(
                        transition.float(), dim=-1
                    ),
                    min=EPS,
                )
            ).mean(dim=1)
            self.closure[
                self.unit_indices, depth_index, :
            ] = relative_error.float().cpu().numpy()

            physical_depth = depth_index + 1
            if physical_depth in self.boundary_slots:
                slot = self.boundary_slots[physical_depth]
                pre_index = protocol.ANCHORS.index("pre_output")
                values = states[:, :, pre_index, :].reshape(
                    len(self.case_indices), states.shape[-1]
                )
                self.boundary_states[
                    self.case_indices, slot, :
                ] = values.float().cpu().numpy()
            self.counts[f"{depth_index}/layer"] += 1
            return output
        return hook

    def register(self) -> None:
        for depth_index, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_pre_hook(
                    self._pre_hook(depth_index)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._component_hook(depth_index, 0)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._component_hook(depth_index, 1)
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    self._layer_hook(depth_index)
                )
            )

    def end(self) -> None:
        expected = {}
        for depth_index in range(len(self.layers)):
            for stage in ("pre", "attention", "mlp", "layer"):
                expected[f"{depth_index}/{stage}"] = 1
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"component hook count drift: {dict(self.counts)}"
            )
        self.anchor_positions = None
        self.anchor_masks = None
        self.unit_indices = None
        self.case_indices = None
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    if values.ndim >= 2:
        row_finite = np.all(
            finite.reshape(-1, values.shape[-1]), axis=-1
        )
    else:
        row_finite = finite
    return {
        "all_finite": bool(finite.all()),
        "finite_value_rate": float(np.mean(finite)),
        "finite_row_rate": float(np.mean(row_finite)),
        "nonfinite_value_count": int(np.size(values) - finite.sum()),
    }


def group_indices(
    units: list[dict[str, Any]],
) -> dict[str, list[int]]:
    result = {
        "all": list(range(len(units))),
        "discovery": [
            int(row["unit_index"])
            for row in units
            if row["split"] == "discovery"
        ],
        "confirmation": [
            int(row["unit_index"])
            for row in units
            if row["split"] == "confirmation"
        ],
    }
    for template in range(4):
        result[f"template_{template}"] = [
            int(row["unit_index"])
            for row in units
            if int(row["template_index"]) == template
        ]
    return result


def depth_slices(depth_count: int) -> list[tuple[int, int, int]]:
    rows = []
    for bin_index in range(protocol.DEPTH_BIN_COUNT):
        start = int(np.floor(bin_index * depth_count / protocol.DEPTH_BIN_COUNT))
        end = int(
            np.floor((bin_index + 1) * depth_count / protocol.DEPTH_BIN_COUNT)
        )
        end = max(end, start + 1)
        rows.append((bin_index, start, min(end, depth_count)))
    return rows


def summarize_responses(
    residual: np.ndarray,
    component: np.ndarray,
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    groups = group_indices(units)
    sources = (
        ("residual", residual),
        ("attention", component[:, :, 0]),
        ("mlp", component[:, :, 1]),
    )
    cosine_index = protocol.FACTOR_METRICS.index(
        "binding_query_cosine"
    )
    invariant_index = protocol.FACTOR_METRICS.index(
        "bq_member_invariance"
    )
    for component_name, source in sources:
        for group, indices in groups.items():
            for bin_index, start, end in depth_slices(source.shape[1]):
                for anchor_index, anchor in enumerate(protocol.ANCHORS):
                    values = np.asarray(
                        source[indices, start:end, anchor_index, :],
                        dtype=np.float32,
                    ).reshape(-1, len(protocol.FACTOR_METRICS))
                    finite_rows = np.all(np.isfinite(values), axis=-1)
                    clean = values[finite_rows]
                    metric_rows = {}
                    for metric_index, metric in enumerate(
                        protocol.FACTOR_METRICS
                    ):
                        current = clean[:, metric_index]
                        metric_rows[metric] = {
                            "mean": float(np.mean(current))
                            if len(current)
                            else None,
                            "median": float(np.median(current))
                            if len(current)
                            else None,
                        }
                    rows.append({
                        "component": component_name,
                        "group": group,
                        "depth_bin": bin_index,
                        "depth_start_index": start,
                        "depth_end_index_exclusive": end,
                        "anchor": anchor,
                        "finite_row_rate": float(np.mean(finite_rows)),
                        "finite_row_count": int(finite_rows.sum()),
                        "binding_query_negative_rate": (
                            float(np.mean(clean[:, cosine_index] < 0))
                            if len(clean)
                            else None
                        ),
                        "bq_member_positive_rate": (
                            float(np.mean(clean[:, invariant_index] > 0))
                            if len(clean)
                            else None
                        ),
                        "metrics": metric_rows,
                    })
    return rows


def behavior_groups(
    cases: list[dict[str, Any]],
) -> dict[str, list[int]]:
    result = {
        "all": list(range(len(cases))),
        "discovery": [
            int(row["case_index"])
            for row in cases
            if row["split"] == "discovery"
        ],
        "confirmation": [
            int(row["case_index"])
            for row in cases
            if row["split"] == "confirmation"
        ],
    }
    for template in range(4):
        result[f"template_{template}"] = [
            int(row["case_index"])
            for row in cases
            if int(row["template_index"]) == template
        ]
    return result


def summarize_behavior(
    candidate_logits: np.ndarray,
    vocab_finite: np.ndarray,
    top_token_ids: np.ndarray,
    expected_ranks: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = np.asarray(
        [int(row["expected_index"]) for row in cases], dtype=np.int64
    )
    candidate_finite = np.all(np.isfinite(candidate_logits), axis=-1)
    result = {}
    for group, indices in behavior_groups(cases).items():
        selected = np.asarray(indices, dtype=np.int64)
        finite = candidate_finite[selected]
        usable = selected[finite]
        if len(usable):
            scores = candidate_logits[usable]
            labels = expected[usable]
            predictions = np.argmax(scores, axis=-1)
            expected_scores = scores[np.arange(len(usable)), labels]
            masked = scores.copy()
            masked[np.arange(len(usable)), labels] = -np.inf
            margins = expected_scores - np.max(masked, axis=-1)
            candidate_accuracy = float(np.mean(predictions == labels))
            margin_median = float(np.median(margins))
        else:
            candidate_accuracy = None
            margin_median = None
        full_usable = selected[vocab_finite[selected]]
        if len(full_usable):
            candidate_ids = np.asarray(
                cases[0]["candidate_token_ids"], dtype=np.int64
            )
            expected_ids = candidate_ids[expected[full_usable]]
            global_accuracy = float(np.mean(
                top_token_ids[full_usable] == expected_ids
            ))
            rank_median = float(np.median(expected_ranks[full_usable]))
        else:
            global_accuracy = None
            rank_median = None
        result[group] = {
            "row_count": len(indices),
            "candidate_logit_finite_row_rate": float(np.mean(finite)),
            "candidate_set_accuracy": candidate_accuracy,
            "candidate_margin_median": margin_median,
            "full_vocab_finite_row_rate": float(
                np.mean(vocab_finite[selected])
            ),
            "global_expected_top1_rate": global_accuracy,
            "expected_vocab_rank_median": rank_median,
        }

    lexical_agreements = []
    lexical_margin_changes = []
    for unit_index in range(len(cases) // len(protocol.WORLD_FACTORS)):
        start = unit_index * len(protocol.WORLD_FACTORS)
        rows = candidate_logits[start:start + len(protocol.WORLD_FACTORS)]
        if not np.all(np.isfinite(rows)):
            continue
        predictions = np.argmax(rows, axis=-1)
        for binding in (0, 1):
            for query in (0, 1):
                left = world_index(binding, query, 0)
                right = world_index(binding, query, 1)
                lexical_agreements.append(
                    int(predictions[left] == predictions[right])
                )
                label = int(cases[start + left]["expected_index"])
                other_left = np.max(np.delete(rows[left], label))
                other_right = np.max(np.delete(rows[right], label))
                margin_left = rows[left, label] - other_left
                margin_right = rows[right, label] - other_right
                lexical_margin_changes.append(
                    abs(float(margin_right - margin_left))
                )
    result["lexical_member_control"] = {
        "comparison_count": len(lexical_agreements),
        "candidate_prediction_agreement": (
            float(np.mean(lexical_agreements))
            if lexical_agreements
            else None
        ),
        "absolute_margin_change_median": (
            float(np.median(lexical_margin_changes))
            if lexical_margin_changes
            else None
        ),
    }
    return result


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def prototype_readout(
    boundary_states: np.ndarray,
    boundary_depths: tuple[int, ...],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    discovery = np.asarray([
        int(row["case_index"])
        for row in cases
        if row["split"] == "discovery"
    ], dtype=np.int64)
    confirmation = np.asarray([
        int(row["case_index"])
        for row in cases
        if row["split"] == "confirmation"
    ], dtype=np.int64)
    expected = np.asarray(
        [int(row["expected_index"]) for row in cases], dtype=np.int64
    )
    rows = []
    for slot, depth in enumerate(boundary_depths):
        train = np.asarray(
            boundary_states[discovery, slot], dtype=np.float32
        )
        test = np.asarray(
            boundary_states[confirmation, slot], dtype=np.float32
        )
        train_finite = np.all(np.isfinite(train), axis=-1)
        test_finite = np.all(np.isfinite(test), axis=-1)
        prototypes = []
        prototype_ok = True
        for family_index in range(len(protocol.FAMILIES)):
            mask = train_finite & (
                expected[discovery] == family_index
            )
            if not np.any(mask):
                prototype_ok = False
                prototypes.append(np.zeros(train.shape[-1], dtype=np.float32))
            else:
                prototypes.append(np.mean(train[mask], axis=0))
        if prototype_ok and np.any(test_finite):
            prototype_matrix = normalize(np.stack(prototypes))
            scores = normalize(test[test_finite]) @ prototype_matrix.T
            predictions = np.argmax(scores, axis=-1)
            accuracy = float(np.mean(
                predictions == expected[confirmation][test_finite]
            ))
        else:
            accuracy = None
        rows.append({
            "physical_depth": int(depth),
            "discovery_finite_rate": float(np.mean(train_finite)),
            "confirmation_finite_rate": float(np.mean(test_finite)),
            "heldout_family_accuracy": accuracy,
            "training_uses_discovery_only": True,
            "confirmation_surfaces_disjoint": True,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    units = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "units.jsonl"
    )
    atlas_dir = protocol.OUT_ROOT / "atlas" / args.model
    atlas_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None

    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        boundary_depths = tuple(sorted({
            max(1, info.n_layers - 1),
            info.n_layers,
        }))

        residual = np.lib.format.open_memmap(
            atlas_dir / "residual_factor_response.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers + 1,
                len(protocol.ANCHORS),
                len(protocol.FACTOR_METRICS),
            ),
        )
        component = np.lib.format.open_memmap(
            atlas_dir / "component_factor_response.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers,
                len(protocol.COMPONENTS),
                len(protocol.ANCHORS),
                len(protocol.FACTOR_METRICS),
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "residual_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                info.n_layers,
                len(protocol.ANCHORS),
            ),
        )
        boundary_states = np.lib.format.open_memmap(
            atlas_dir / "boundary_states.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases), len(boundary_depths), info.d_model),
        )
        candidate_logits = np.lib.format.open_memmap(
            atlas_dir / "candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases), len(protocol.FAMILIES)),
        )
        vocab_finite = np.lib.format.open_memmap(
            atlas_dir / "vocab_finite.bool.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(len(cases),),
        )
        top_token_ids = np.lib.format.open_memmap(
            atlas_dir / "top_token_ids.int64.npy",
            mode="w+",
            dtype=np.int64,
            shape=(len(cases),),
        )
        expected_ranks = np.lib.format.open_memmap(
            atlas_dir / "expected_ranks.int64.npy",
            mode="w+",
            dtype=np.int64,
            shape=(len(cases),),
        )
        for array in (
            residual,
            component,
            closure,
            boundary_states,
            candidate_logits,
        ):
            array[:] = np.nan
        vocab_finite[:] = False
        top_token_ids[:] = -1
        expected_ranks[:] = -1

        capture = NativeFactorCapture(
            layers,
            residual,
            component,
            closure,
            boundary_states,
            boundary_depths,
        )
        capture.register()
        candidate_ids = torch.tensor(
            cases[0]["candidate_token_ids"], dtype=torch.long
        )
        try:
            for batch_number, row_batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                (
                    input_ids,
                    attention_mask,
                    anchor_positions,
                    anchor_masks,
                    pre_positions,
                    unit_indices,
                    case_indices,
                    expected_indices,
                ) = make_batch(
                    row_batch,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    device=device,
                )
                capture.begin(
                    anchor_positions,
                    anchor_masks,
                    unit_indices,
                    case_indices,
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                capture.end()

                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float()
                ids = candidate_ids.to(selected.device)
                selected_candidates = selected.index_select(-1, ids)
                candidate_logits[case_indices] = (
                    selected_candidates.detach().cpu().numpy()
                )
                finite = torch.isfinite(selected).all(dim=-1)
                vocab_finite[case_indices] = (
                    finite.detach().cpu().numpy()
                )
                safe_selected = torch.where(
                    torch.isfinite(selected),
                    selected,
                    torch.full_like(selected, -torch.inf),
                )
                top_token_ids[case_indices] = (
                    torch.argmax(safe_selected, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                expected_token_ids = ids.index_select(
                    0, expected_indices.to(ids.device)
                )
                expected_scores = safe_selected.gather(
                    1, expected_token_ids[:, None]
                ).squeeze(1)
                ranks = 1 + torch.sum(
                    safe_selected > expected_scores[:, None], dim=-1
                )
                ranks = torch.where(
                    finite,
                    ranks,
                    torch.full_like(ranks, -1),
                )
                expected_ranks[case_indices] = (
                    ranks.detach().cpu().numpy()
                )
                del output, logits, selected, selected_candidates
                if batch_number % 16 == 0:
                    print(
                        f"[phase1035] {args.model} "
                        f"units={int(unit_indices[-1]) + 1}/{len(units)}",
                        flush=True,
                    )
        finally:
            capture.close()

        arrays = (
            residual,
            component,
            closure,
            boundary_states,
            candidate_logits,
            vocab_finite,
            top_token_ids,
            expected_ranks,
        )
        for array in arrays:
            array.flush()

        response_rows = summarize_responses(
            residual, component, units
        )
        behavior = summarize_behavior(
            candidate_logits,
            vocab_finite,
            top_token_ids,
            expected_ranks,
            cases,
        )
        prototypes = prototype_readout(
            boundary_states, boundary_depths, cases
        )
        def closure_stats(values: np.ndarray) -> dict[str, Any]:
            finite = np.asarray(values)[np.isfinite(values)]
            return {
                "finite_value_rate": float(np.mean(np.isfinite(values))),
                "median": (
                    float(np.median(finite)) if len(finite) else None
                ),
                "p95": (
                    float(np.quantile(finite, 0.95))
                    if len(finite)
                    else None
                ),
                "max": float(np.max(finite)) if len(finite) else None,
            }

        # The first token is a causal negative control.  Its layer transition
        # can be nearly zero, so a tiny absolute FP16 addition residual makes
        # the relative denominator unstable.  Preserve that measurement, but
        # use the six active semantic/suffix anchors for the instrument gate.
        closure_summary = {
            "all_anchors": closure_stats(closure),
            "active_anchors_excluding_prefix_control": closure_stats(
                np.asarray(closure)[:, :, 1:]
            ),
        }
        array_audit = {
            "residual_factor_response": finite_summary(residual),
            "component_factor_response": finite_summary(component),
            "residual_closure": finite_summary(closure),
            "boundary_states": finite_summary(boundary_states),
            "candidate_logits": finite_summary(candidate_logits),
        }
        metrics = {
            "schema_version": "phase1035_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "response_depth_bins": response_rows,
            "behavior": behavior,
            "heldout_internal_prototype_readout": prototypes,
            "instrumentation": {
                "residual_addition_relative_error": closure_summary,
            },
        }
        summary = {
            "schema_version": "phase1035_model_summary.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": info.model_class,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
            },
            "sample_counts": {
                "units": len(units),
                "cases": len(cases),
                "discovery_units": 128,
                "confirmation_units": 128,
            },
            "boundary_depths": list(boundary_depths),
            "array_finiteness": array_audit,
            "candidate_logit_gate_passed": (
                behavior["all"]["candidate_logit_finite_row_rate"]
                >= prereg["instrumentation_gate"][
                    "candidate_logit_finite_row_rate_min"
                ]
            ),
            "component_instrumentation_gate_passed": (
                closure_summary[
                    "active_anchors_excluding_prefix_control"
                ]["p95"] is not None
                and closure_summary[
                    "active_anchors_excluding_prefix_control"
                ]["p95"]
                <= prereg["instrumentation_gate"][
                    "residual_addition_relative_error_p95_max"
                ]
            ),
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


if __name__ == "__main__":
    main()
