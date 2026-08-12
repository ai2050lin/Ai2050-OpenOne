#!/usr/bin/env python3
"""Map Phase1024 lexical, contextual, and temporary-binding structure.

The scan is observational.  Discovery data selects residual depths and real
pre-o_proj head / pre-down_proj MLP coordinates.  Confirmation data is only
read after selection and cannot change the frozen candidate coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
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
import phase1024_lexical_semantic_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


ROLES = protocol.ROLES
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def write_json(path: Path, value: Any) -> None:
    protocol.write_json(path, value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    protocol.write_jsonl(path, rows)


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.empty((len(rows), len(ROLES)), dtype=torch.long)
    for batch_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[batch_index, :len(values)] = values
        attention_mask[batch_index, :len(values)] = 1
        for role_index, role in enumerate(ROLES):
            positions[batch_index, role_index] = int(
                row["role_positions"][role]
            )
    return (
        input_ids.to(device),
        attention_mask.to(device),
        positions,
    )


class ResidualCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("residual capture is not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[depth] = value[batch, positions, :].detach()
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
        self.counts = defaultdict(int)

    def stacked_cpu(self) -> torch.Tensor:
        expected = set(range(len(self.layers) + 1))
        if set(self.values) != expected:
            missing = sorted(expected - set(self.values))
            raise RuntimeError(f"missing residual depths: {missing[:8]}")
        repeated = {
            depth: count
            for depth, count in self.counts.items()
            if count != 1
        }
        if repeated:
            raise RuntimeError(f"residual hook repetition: {repeated}")
        return torch.stack([
            self.values[depth].to("cpu")
            for depth in sorted(expected)
        ], dim=2)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


class ComponentCapture:
    def __init__(self, layers, selected_layers: list[int], head_count: int):
        self.layers = layers
        self.selected_layers = selected_layers
        self.head_count = head_count
        self.positions: torch.Tensor | None = None
        self.heads: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def _select(self, value: torch.Tensor) -> torch.Tensor:
        if self.positions is None:
            raise RuntimeError("component capture is not initialized")
        positions = self.positions.to(value.device)
        batch = torch.arange(value.shape[0], device=value.device)[:, None]
        return value[batch, positions, :].detach()

    def _head_hook(self, depth: int):
        def hook(module, args):
            selected = self._select(args[0])
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("pre-o_proj width is not head aligned")
            self.heads[depth] = selected.reshape(
                selected.shape[0],
                selected.shape[1],
                self.head_count,
                selected.shape[-1] // self.head_count,
            )
            self.counts[f"head/{depth}"] += 1
        return hook

    def _mlp_hook(self, depth: int):
        def hook(module, args):
            self.mlp[depth] = self._select(args[0])
            self.counts[f"mlp/{depth}"] += 1
        return hook

    def register(self) -> None:
        for depth in self.selected_layers:
            layer = self.layers[depth - 1]
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._head_hook(depth)
                )
            )
            self.handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    self._mlp_hook(depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.heads = {}
        self.mlp = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        expected = {
            *{f"head/{depth}" for depth in self.selected_layers},
            *{f"mlp/{depth}" for depth in self.selected_layers},
        }
        bad = {
            key: self.counts[key]
            for key in expected
            if self.counts[key] != 1
        }
        if bad:
            raise RuntimeError(f"component hook count drift: {bad}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.heads = {}
        self.mlp = {}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation over axis 0, independently for trailing columns."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.sum(left * right, axis=0)
    denominator = np.sqrt(
        np.sum(left * left, axis=0) * np.sum(right * right, axis=0)
    )
    result = numerator / np.maximum(denominator, EPS)
    result[~np.isfinite(result)] = 0.0
    return result


def finite_audit(values: np.ndarray, block: int = 32) -> dict[str, Any]:
    total = int(np.prod(values.shape))
    nonfinite = 0
    for start in range(0, values.shape[0], block):
        part = np.asarray(values[start:start + block])
        nonfinite += int(np.count_nonzero(~np.isfinite(part)))
    return {
        "shape": list(values.shape),
        "value_count": total,
        "nonfinite_count": nonfinite,
        "all_finite": nonfinite == 0,
    }


def top1_metrics(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float, float, float]:
    left = normalize_rows(left)
    right = normalize_rows(right)
    similarity = left @ right.T
    labels = np.arange(len(left))
    prediction = np.argmax(similarity, axis=1)
    same = similarity[labels, labels]
    wrong = similarity.copy()
    wrong[labels, labels] = -np.inf
    best_wrong = np.max(wrong, axis=1)
    shifted = similarity[labels, np.roll(labels, -1)]
    return (
        float(np.mean(prediction == labels)),
        float(np.mean(same - best_wrong)),
        float(np.mean(same)),
        float(np.mean(shifted)),
    )


def panel_indices(
    cases: list[dict[str, Any]],
    panel: str,
) -> list[int]:
    return [
        index for index, row in enumerate(cases)
        if row["panel"] == panel
    ]


def nonce_grid(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> np.ndarray:
    rows = [
        (index, row)
        for index, row in enumerate(cases)
        if row["panel"] == "nonce_binding" and row["split"] == split
    ]
    shape = (2, 4, 8) + tuple(values.shape[1:])
    grid = np.empty(shape, dtype=np.float32)
    seen = set()
    for index, row in rows:
        key = (
            int(row["template_index"]),
            int(row["surface_index"]),
            int(row["concept_index"]),
        )
        grid[key] = values[index]
        seen.add(key)
    if len(seen) != 64:
        raise RuntimeError(f"nonce grid incomplete for {split}: {len(seen)}")
    return grid


def nonce_vector_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, float]:
    grid = nonce_grid(values, cases, split)
    grid = grid - grid.mean(axis=(1, 2), keepdims=True)
    concept_acc = []
    concept_margin = []
    concept_same = []
    concept_shifted = []
    surface_acc = []
    surface_margin = []
    surface_same = []
    surface_shifted = []
    for left_template, right_template in ((0, 1), (1, 0)):
        for left_surface in range(4):
            for right_surface in range(4):
                if left_surface == right_surface:
                    continue
                metrics = top1_metrics(
                    grid[left_template, left_surface],
                    grid[right_template, right_surface],
                )
                concept_acc.append(metrics[0])
                concept_margin.append(metrics[1])
                concept_same.append(metrics[2])
                concept_shifted.append(metrics[3])
        for left_concept in range(8):
            for right_concept in range(8):
                if left_concept == right_concept:
                    continue
                metrics = top1_metrics(
                    grid[left_template, :, left_concept],
                    grid[right_template, :, right_concept],
                )
                surface_acc.append(metrics[0])
                surface_margin.append(metrics[1])
                surface_same.append(metrics[2])
                surface_shifted.append(metrics[3])
    concept_accuracy = float(np.mean(concept_acc))
    surface_accuracy = float(np.mean(surface_acc))
    concept_excess = concept_accuracy - 1.0 / 8.0
    surface_excess = surface_accuracy - 1.0 / 4.0
    return {
        "concept_cross_surface_top1": concept_accuracy,
        "concept_chance": 1.0 / 8.0,
        "concept_excess": concept_excess,
        "concept_true_vs_wrong_margin": float(np.mean(concept_margin)),
        "concept_same_cosine": float(np.mean(concept_same)),
        "concept_shifted_cosine": float(np.mean(concept_shifted)),
        "surface_cross_concept_top1": surface_accuracy,
        "surface_chance": 1.0 / 4.0,
        "surface_excess": surface_excess,
        "surface_true_vs_wrong_margin": float(np.mean(surface_margin)),
        "surface_same_cosine": float(np.mean(surface_same)),
        "surface_shifted_cosine": float(np.mean(surface_shifted)),
        "semantic_minus_surface_excess": concept_excess - surface_excess,
    }


def nonce_alignment_metrics(
    anchor: np.ndarray,
    focus: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, float]:
    definition = nonce_grid(anchor, cases, split)
    query = nonce_grid(focus, cases, split)
    query = query - query.mean(axis=2, keepdims=True)
    definition = definition.mean(axis=1)
    definition = definition - definition.mean(axis=1, keepdims=True)
    accuracies = []
    margins = []
    same_values = []
    shifted_values = []
    for query_template in range(2):
        for definition_template in range(2):
            target = normalize_rows(definition[definition_template])
            for surface in range(4):
                source = normalize_rows(query[query_template, surface])
                similarity = source @ target.T
                labels = np.arange(8)
                same = similarity[labels, labels]
                wrong = similarity.copy()
                wrong[labels, labels] = -np.inf
                accuracies.append(float(np.mean(
                    np.argmax(similarity, axis=1) == labels
                )))
                margins.append(float(np.mean(
                    same - np.max(wrong, axis=1)
                )))
                same_values.append(float(np.mean(same)))
                shifted_values.append(float(np.mean(
                    similarity[labels, np.roll(labels, -1)]
                )))
    return {
        "definition_query_top1": float(np.mean(accuracies)),
        "chance": 1.0 / 8.0,
        "true_vs_wrong_margin": float(np.mean(margins)),
        "same_concept_cosine": float(np.mean(same_values)),
        "shifted_concept_cosine": float(np.mean(shifted_values)),
        "same_vs_shifted_margin": (
            float(np.mean(same_values)) - float(np.mean(shifted_values))
        ),
    }


def poly_grid(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    partition: str,
) -> np.ndarray:
    rows = [
        (index, row)
        for index, row in enumerate(cases)
        if row["panel"] == "polysemy" and row["partition"] == partition
    ]
    item_ids = sorted({int(row["item_index"]) for _, row in rows})
    item_to_local = {item: index for index, item in enumerate(item_ids)}
    shape = (8, 2, 2, 2) + tuple(values.shape[1:])
    grid = np.empty(shape, dtype=np.float32)
    seen = set()
    split_index = {"discovery": 0, "confirmation": 1}
    for index, row in rows:
        key = (
            item_to_local[int(row["item_index"])],
            split_index[row["split"]],
            int(row["template_index"]),
            int(row["sense_index"]),
        )
        grid[key] = values[index]
        seen.add(key)
    if len(seen) != 64:
        raise RuntimeError(
            f"polysemy grid incomplete for {partition}: {len(seen)}"
        )
    return grid


def binary_classification(
    prototype: np.ndarray,
    query: np.ndarray,
) -> tuple[float, float]:
    prototype = normalize_rows(prototype)
    query = normalize_rows(query)
    similarity = query @ prototype.T
    labels = np.arange(2)
    return (
        float(np.mean(np.argmax(similarity, axis=1) == labels)),
        float(np.mean(
            similarity[labels, labels]
            - similarity[labels, np.roll(labels, -1)]
        )),
    )


def poly_vector_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    partition: str,
    mode: str,
) -> dict[str, float]:
    grid = poly_grid(values, cases, partition)
    grid = grid - grid.mean(axis=(1, 2, 3), keepdims=True)
    accuracies = []
    margins = []
    direction_cosines = []
    if mode == "discovery":
        for item in range(8):
            for left_template, right_template in ((0, 1), (1, 0)):
                accuracy, margin = binary_classification(
                    grid[item, 0, left_template],
                    grid[item, 0, right_template],
                )
                accuracies.append(accuracy)
                margins.append(margin)
            left_delta = (
                grid[item, 0, 0, 0] - grid[item, 0, 0, 1]
            )
            right_delta = (
                grid[item, 0, 1, 0] - grid[item, 0, 1, 1]
            )
            direction_cosines.append(float((
                normalize_rows(left_delta[None, :])
                @ normalize_rows(right_delta[None, :]).T
            ).item()))
    elif mode == "confirmation":
        for item in range(8):
            prototype = grid[item, 0].mean(axis=0)
            for template in range(2):
                accuracy, margin = binary_classification(
                    prototype,
                    grid[item, 1, template],
                )
                accuracies.append(accuracy)
                margins.append(margin)
            discovery_delta = prototype[0] - prototype[1]
            confirmation_mean = grid[item, 1].mean(axis=0)
            confirmation_delta = (
                confirmation_mean[0] - confirmation_mean[1]
            )
            direction_cosines.append(float((
                normalize_rows(discovery_delta[None, :])
                @ normalize_rows(confirmation_delta[None, :]).T
            ).item()))
    else:
        raise ValueError(mode)
    return {
        "sense_top1": float(np.mean(accuracies)),
        "chance": 0.5,
        "true_vs_wrong_margin": float(np.mean(margins)),
        "difference_direction_cosine": float(
            np.mean(direction_cosines)
        ),
        "item_count": 8,
    }


def synonym_grid(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    partition: str,
) -> np.ndarray:
    rows = [
        (index, row)
        for index, row in enumerate(cases)
        if row["panel"] == "synonym" and row["partition"] == partition
    ]
    group_ids = sorted({int(row["group_index"]) for _, row in rows})
    group_to_local = {group: index for index, group in enumerate(group_ids)}
    shape = (8, 2, 2, 3) + tuple(values.shape[1:])
    grid = np.empty(shape, dtype=np.float32)
    seen = set()
    split_index = {"discovery": 0, "confirmation": 1}
    for index, row in rows:
        key = (
            group_to_local[int(row["group_index"])],
            split_index[row["split"]],
            int(row["template_index"]),
            int(row["alias_index"]),
        )
        grid[key] = values[index]
        seen.add(key)
    if len(seen) != 96:
        raise RuntimeError(
            f"synonym grid incomplete for {partition}: {len(seen)}"
        )
    return grid


def synonym_vector_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    partition: str,
    mode: str,
) -> dict[str, float]:
    grid = synonym_grid(values, cases, partition)
    # Remove prompt/alias-index offsets shared by all lexical groups.
    grid = grid - grid.mean(axis=0, keepdims=True)
    accuracies = []
    margins = []
    same_values = []
    shifted_values = []
    if mode == "discovery":
        for template in range(2):
            for left_alias, right_alias in ((0, 1), (1, 0)):
                metrics = top1_metrics(
                    grid[:, 0, template, left_alias],
                    grid[:, 0, 1 - template, right_alias],
                )
                accuracies.append(metrics[0])
                margins.append(metrics[1])
                same_values.append(metrics[2])
                shifted_values.append(metrics[3])
    elif mode == "confirmation":
        prototype = grid[:, 0, :, :2].mean(axis=(1, 2))
        for template in range(2):
            metrics = top1_metrics(
                grid[:, 1, template, 2],
                prototype,
            )
            accuracies.append(metrics[0])
            margins.append(metrics[1])
            same_values.append(metrics[2])
            shifted_values.append(metrics[3])
    else:
        raise ValueError(mode)
    return {
        "group_top1": float(np.mean(accuracies)),
        "chance": 1.0 / 8.0,
        "true_vs_wrong_margin": float(np.mean(margins)),
        "same_group_cosine": float(np.mean(same_values)),
        "shifted_group_cosine": float(np.mean(shifted_values)),
        "same_vs_shifted_margin": (
            float(np.mean(same_values)) - float(np.mean(shifted_values))
        ),
        "group_count": 8,
    }


def vector_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "nonce": {
            split: nonce_vector_metrics(values, cases, split)
            for split in protocol.SPLITS
        },
        "polysemy": {
            "discovery": poly_vector_metrics(
                values, cases, "calibration", "discovery"
            ),
            "confirmation": poly_vector_metrics(
                values, cases, "heldout", "confirmation"
            ),
        },
        "synonym": {
            "discovery": synonym_vector_metrics(
                values, cases, "calibration", "discovery"
            ),
            "confirmation": synonym_vector_metrics(
                values, cases, "heldout", "confirmation"
            ),
        },
    }


def choose_spread(
    ranked_depths: list[int],
    *,
    n_layers: int,
    count: int = 2,
) -> list[int]:
    minimum = max(1, int(math.ceil(n_layers * 0.10)))
    selected = []
    for depth in ranked_depths:
        if depth < 1:
            continue
        if all(abs(depth - prior) >= minimum for prior in selected):
            selected.append(depth)
        if len(selected) == count:
            break
    if len(selected) < count:
        for depth in ranked_depths:
            if depth >= 1 and depth not in selected:
                selected.append(depth)
            if len(selected) == count:
                break
    return selected


def select_layers(
    rows: list[dict[str, Any]],
    *,
    n_layers: int,
) -> dict[str, Any]:
    focus_rows = [row for row in rows if row["role"] == "focus_end"]
    nonce_ranked = sorted(
        focus_rows,
        key=lambda row: (
            row["metrics"]["nonce"]["discovery"][
                "concept_cross_surface_top1"
            ],
            row["metrics"]["nonce"]["discovery"][
                "semantic_minus_surface_excess"
            ],
            row["alignment"]["discovery"]["definition_query_top1"],
            -row["depth"],
        ),
        reverse=True,
    )
    poly_ranked = sorted(
        focus_rows,
        key=lambda row: (
            row["metrics"]["polysemy"]["discovery"]["sense_top1"],
            row["metrics"]["polysemy"]["discovery"][
                "difference_direction_cosine"
            ],
            -row["depth"],
        ),
        reverse=True,
    )
    synonym_ranked = sorted(
        focus_rows,
        key=lambda row: (
            row["metrics"]["synonym"]["discovery"]["group_top1"],
            row["metrics"]["synonym"]["discovery"][
                "same_vs_shifted_margin"
            ],
            -row["depth"],
        ),
        reverse=True,
    )
    selected_by_track = {
        "nonce": choose_spread(
            [int(row["depth"]) for row in nonce_ranked],
            n_layers=n_layers,
        ),
        "polysemy": choose_spread(
            [int(row["depth"]) for row in poly_ranked],
            n_layers=n_layers,
        ),
        "synonym": choose_spread(
            [int(row["depth"]) for row in synonym_ranked],
            n_layers=n_layers,
        ),
    }
    crossing = None
    for row in sorted(focus_rows, key=lambda value: int(value["depth"])):
        metric = row["metrics"]["nonce"]["discovery"]
        if (
            int(row["depth"]) >= 1
            and metric["concept_cross_surface_top1"]
            > metric["concept_chance"]
            and metric["semantic_minus_surface_excess"] > 0
        ):
            crossing = int(row["depth"])
            break
    selected = sorted({
        depth
        for values in selected_by_track.values()
        for depth in values
    } | ({crossing} if crossing is not None else set()))
    return {
        "schema_version": "phase1024_selected_layers.v1",
        "selection_source": "discovery_only",
        "selected_by_track": selected_by_track,
        "first_nonce_semantic_dominance_depth": crossing,
        "selected_layers": selected,
    }


def scan_residual(
    model,
    tokenizer,
    device: torch.device,
    layers,
    cases: list[dict[str, Any]],
    out_dir: Path,
    *,
    batch_size: int,
    d_model: int,
) -> tuple[np.memmap, list[dict[str, Any]], dict[str, Any]]:
    raw_path = out_dir / "residual_states.fp16.npy"
    expected_shape = (
        len(cases),
        len(ROLES),
        len(layers) + 1,
        d_model,
    )
    reuse = False
    if raw_path.exists():
        existing = np.load(raw_path, mmap_mode="r")
        reuse = (
            existing.shape == expected_shape
            and np.any(existing[-1] != 0)
        )
        del existing
    if reuse:
        print("[phase1024-residual] reusing complete capture", flush=True)
        raw = np.load(raw_path, mmap_mode="r+")
    else:
        raw = np.lib.format.open_memmap(
            raw_path,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )
        capture = ResidualCapture(model, layers)
        capture.register()
        try:
            offset = 0
            for batch_index, batch in enumerate(
                chunks(cases, batch_size), 1
            ):
                input_ids, attention_mask, positions = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(positions)
                with torch.inference_mode():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                values = capture.stacked_cpu().numpy().astype(
                    np.float16, copy=False
                )
                raw[offset:offset + len(batch)] = values
                offset += len(batch)
                if batch_index % 5 == 0:
                    print(
                        f"[phase1024-residual] "
                        f"cases={offset}/{len(cases)}",
                        flush=True,
                    )
        finally:
            capture.close()
        raw.flush()

    rows = []
    for role_index, role in enumerate(ROLES):
        for depth in range(len(layers) + 1):
            values = np.asarray(raw[:, role_index, depth, :])
            row = {
                "schema_version": "phase1024_residual_metric.v1",
                "role": role,
                "depth": depth,
                "relative_depth": depth / max(len(layers), 1),
                "metrics": vector_metrics(values, cases),
            }
            if role == "focus_end":
                row["alignment"] = {
                    split: nonce_alignment_metrics(
                        np.asarray(
                            raw[
                                :,
                                ROLE_INDEX["anchor_end"],
                                depth,
                                :,
                            ]
                        ),
                        values,
                        cases,
                        split,
                    )
                    for split in protocol.SPLITS
                }
            rows.append(row)
    selection = select_layers(rows, n_layers=len(layers))
    write_jsonl(out_dir / "residual_metrics.jsonl", rows)
    write_json(out_dir / "selected_layers.json", selection)
    return raw, rows, selection


def head_count_for(model) -> int:
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if config is not None and hasattr(config, "num_attention_heads"):
            return int(config.num_attention_heads)
    raise RuntimeError("num_attention_heads missing")


def capture_components(
    model,
    tokenizer,
    device: torch.device,
    layers,
    cases: list[dict[str, Any]],
    selected_layers: list[int],
    out_dir: Path,
    *,
    batch_size: int,
    head_count: int,
    intermediate_size: int,
) -> tuple[np.memmap, np.memmap]:
    sample_layer = layers[selected_layers[0] - 1]
    head_width = int(sample_layer.self_attn.o_proj.weight.shape[1])
    if head_width % head_count:
        raise RuntimeError("head width is not divisible by head count")
    head_dim = head_width // head_count
    head_path = out_dir / "attention_heads.fp16.npy"
    mlp_path = out_dir / "mlp_intermediate.fp16.npy"
    head_raw = np.lib.format.open_memmap(
        head_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(cases),
            len(ROLES),
            len(selected_layers),
            head_count,
            head_dim,
        ),
    )
    mlp_raw = np.lib.format.open_memmap(
        mlp_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            len(cases),
            len(ROLES),
            len(selected_layers),
            intermediate_size,
        ),
    )
    capture = ComponentCapture(layers, selected_layers, head_count)
    capture.register()
    try:
        offset = 0
        for batch_index, batch in enumerate(chunks(cases, batch_size), 1):
            input_ids, attention_mask, positions = make_batch(
                batch,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            capture.begin(positions)
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            capture.validate()
            for layer_index, depth in enumerate(selected_layers):
                head_raw[
                    offset:offset + len(batch), :, layer_index
                ] = capture.heads[depth].to("cpu").numpy().astype(
                    np.float16, copy=False
                )
                mlp_raw[
                    offset:offset + len(batch), :, layer_index
                ] = capture.mlp[depth].to("cpu").numpy().astype(
                    np.float16, copy=False
                )
            offset += len(batch)
            if batch_index % 5 == 0:
                print(
                    f"[phase1024-components] cases={offset}/{len(cases)}",
                    flush=True,
                )
    finally:
        capture.close()
    head_raw.flush()
    mlp_raw.flush()
    return head_raw, mlp_raw


def analyze_heads(
    raw: np.ndarray,
    cases: list[dict[str, Any]],
    selected_layers: list[int],
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    head_count = raw.shape[3]
    for layer_index, depth in enumerate(selected_layers):
        for role_index, role in enumerate(ROLES):
            for head in range(head_count):
                values = np.asarray(
                    raw[:, role_index, layer_index, head, :]
                )
                row = {
                    "schema_version": "phase1024_attention_head_metric.v1",
                    "depth": depth,
                    "role": role,
                    "head": head,
                    "metrics": vector_metrics(values, cases),
                }
                if role == "focus_end":
                    row["alignment"] = {
                        split: nonce_alignment_metrics(
                            np.asarray(
                                raw[
                                    :,
                                    ROLE_INDEX["anchor_end"],
                                    layer_index,
                                    head,
                                    :,
                                ]
                            ),
                            values,
                            cases,
                            split,
                        )
                        for split in protocol.SPLITS
                    }
                rows.append(row)
    write_jsonl(out_dir / "attention_head_metrics.jsonl", rows)
    return rows


def mlp_profile_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    nonce_discovery = nonce_grid(values, cases, "discovery")
    nonce_confirmation = nonce_grid(values, cases, "confirmation")
    nonce_discovery = (
        nonce_discovery
        - nonce_discovery.mean(axis=(1, 2), keepdims=True)
    )
    nonce_confirmation = (
        nonce_confirmation
        - nonce_confirmation.mean(axis=(1, 2), keepdims=True)
    )
    disc_concept = nonce_discovery.mean(axis=1)
    conf_concept = nonce_confirmation.mean(axis=1)
    disc_surface = nonce_discovery.mean(axis=2)
    conf_surface = nonce_confirmation.mean(axis=2)

    poly_cal = poly_grid(values, cases, "calibration")
    poly_hold = poly_grid(values, cases, "heldout")
    cal_delta = poly_cal[:, 0, :, 0] - poly_cal[:, 0, :, 1]
    hold_disc = poly_hold[:, 0].mean(axis=1)
    hold_conf = poly_hold[:, 1].mean(axis=1)
    hold_disc_delta = hold_disc[:, 0] - hold_disc[:, 1]
    hold_conf_delta = hold_conf[:, 0] - hold_conf[:, 1]

    syn_cal = synonym_grid(values, cases, "calibration")
    syn_hold = synonym_grid(values, cases, "heldout")
    syn_cal = syn_cal - syn_cal.mean(axis=0, keepdims=True)
    syn_hold = syn_hold - syn_hold.mean(axis=0, keepdims=True)
    cal_alias0 = syn_cal[:, 0, :, 0].mean(axis=1)
    cal_alias1 = syn_cal[:, 0, :, 1].mean(axis=1)
    hold_prototype = syn_hold[:, 0, :, :2].mean(axis=(1, 2))
    hold_alias2 = syn_hold[:, 1, :, 2].mean(axis=1)

    return {
        "nonce_discovery_concept_corr": row_corr(
            disc_concept[0], disc_concept[1]
        ),
        "nonce_confirmation_concept_corr": row_corr(
            conf_concept[0], conf_concept[1]
        ),
        "nonce_cross_split_family_corr": row_corr(
            disc_concept.mean(axis=0),
            conf_concept.mean(axis=0),
        ),
        "nonce_discovery_surface_corr": row_corr(
            disc_surface[0], disc_surface[1]
        ),
        "nonce_confirmation_surface_corr": row_corr(
            conf_surface[0], conf_surface[1]
        ),
        "nonce_discovery_concept_std": np.std(
            disc_concept.mean(axis=0), axis=0
        ),
        "polysemy_discovery_corr": row_corr(
            cal_delta[:, 0], cal_delta[:, 1]
        ),
        "polysemy_confirmation_corr": row_corr(
            hold_disc_delta, hold_conf_delta
        ),
        "synonym_discovery_corr": row_corr(cal_alias0, cal_alias1),
        "synonym_confirmation_corr": row_corr(
            hold_prototype, hold_alias2
        ),
    }


def select_coordinates(
    metric: dict[str, np.ndarray],
    *,
    track: str,
    count: int,
) -> np.ndarray:
    key = {
        "nonce": "nonce_discovery_concept_corr",
        "polysemy": "polysemy_discovery_corr",
        "synonym": "synonym_discovery_corr",
    }[track]
    score = metric[key]
    if track == "nonce":
        variance = metric["nonce_discovery_concept_std"]
        eligible = variance >= np.median(variance)
    else:
        eligible = np.ones_like(score, dtype=bool)
    indices = np.flatnonzero(eligible)
    ranked = indices[np.argsort(score[indices])[::-1]]
    return ranked[:count]


def analyze_mlp(
    raw: np.ndarray,
    cases: list[dict[str, Any]],
    selected_layers: list[int],
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(1024)
    for layer_index, depth in enumerate(selected_layers):
        for role_index, role in enumerate(ROLES):
            values = np.asarray(
                raw[:, role_index, layer_index, :],
                dtype=np.float32,
            )
            metric = mlp_profile_metrics(values, cases)
            chosen_by_track = {
                track: select_coordinates(metric, track=track, count=32)
                for track in ("nonce", "polysemy", "synonym")
            }
            selected_union = sorted({
                int(index)
                for indices in chosen_by_track.values()
                for index in indices
            })
            remaining = np.setdiff1d(
                np.arange(values.shape[1]),
                np.asarray(selected_union, dtype=np.int64),
                assume_unique=False,
            )
            random_controls = rng.choice(
                remaining,
                size=min(32, len(remaining)),
                replace=False,
            )
            for candidate_type, indices in (
                ("selected", np.asarray(selected_union, dtype=np.int64)),
                ("random_control", random_controls),
            ):
                for coordinate in indices:
                    coordinate = int(coordinate)
                    rows.append({
                        "schema_version": (
                            "phase1024_mlp_coordinate_metric.v1"
                        ),
                        "depth": depth,
                        "role": role,
                        "coordinate": coordinate,
                        "candidate_type": candidate_type,
                        "selected_tracks": [
                            track
                            for track, track_indices
                            in chosen_by_track.items()
                            if coordinate in set(map(int, track_indices))
                        ],
                        "metrics": {
                            key: float(value[coordinate])
                            for key, value in metric.items()
                        },
                    })
    write_jsonl(out_dir / "mlp_coordinate_metrics.jsonl", rows)
    return rows


def representative_rows(
    residual_rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    lookup = {
        (row["role"], int(row["depth"])): row
        for row in residual_rows
    }
    return {
        track: [
            lookup[("focus_end", int(depth))]
            for depth in depths
        ]
        for track, depths in selection["selected_by_track"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{args.model}.jsonl"
    )
    out_dir = protocol.OUT_ROOT / "atlas" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision_audit = quantization_audit(model)
        if (
            precision_audit["has_quantized_modules"]
            or precision_audit["has_bf16_parameters"]
            or not precision_audit["has_fp16_parameters"]
        ):
            raise RuntimeError(
                "FP16/no-quantization audit failed: "
                + json.dumps(precision_audit)
            )
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        head_count = head_count_for(model)
        residual_raw, residual_rows, selection = scan_residual(
            model,
            tokenizer,
            device,
            layers,
            cases,
            out_dir,
            batch_size=BATCH_SIZE[args.model],
            d_model=info.d_model,
        )
        selected_layers = list(selection["selected_layers"])
        if not selected_layers:
            raise RuntimeError("no component layers selected")
        head_raw, mlp_raw = capture_components(
            model,
            tokenizer,
            device,
            layers,
            cases,
            selected_layers,
            out_dir,
            batch_size=BATCH_SIZE[args.model],
            head_count=head_count,
            intermediate_size=info.intermediate_size,
        )
        head_rows = analyze_heads(
            head_raw,
            cases,
            selected_layers,
            out_dir,
        )
        mlp_rows = analyze_mlp(
            mlp_raw,
            cases,
            selected_layers,
            out_dir,
        )
        tensor_finiteness = {
            "residual": finite_audit(residual_raw),
            "attention_heads": finite_audit(head_raw),
            "mlp_intermediate": finite_audit(mlp_raw),
        }
        summary = {
            "schema_version": "phase1024_model_atlas_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "intermediate_size": info.intermediate_size,
                "head_count": head_count,
                "model_class": info.model_class,
            },
            "case_count": len(cases),
            "selection": selection,
            "representative_residual_metrics": representative_rows(
                residual_rows, selection
            ),
            "attention_head_metric_count": len(head_rows),
            "mlp_coordinate_metric_count": len(mlp_rows),
            "raw_shapes": {
                "residual": list(residual_raw.shape),
                "attention_heads": list(head_raw.shape),
                "mlp_intermediate": list(mlp_raw.shape),
            },
            "tensor_finiteness": tensor_finiteness,
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "repeated internal geometry and component response profiles "
                "only; no causal, optimality, storage-cell, brain-homology, "
                "or closed-mechanism claim"
            ),
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps({
            "model": args.model,
            "selection": selection,
            "representative": summary[
                "representative_residual_metrics"
            ],
            "raw_shapes": summary["raw_shapes"],
        }, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


if __name__ == "__main__":
    main()
