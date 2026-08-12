#!/usr/bin/env python3
"""Measure source-state family contrasts in actual Transformer channels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
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
import phase1035_native_family_routing_protocol as source
import phase1038_source_channel_protocol as protocol


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def grouped(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] % len(source.WORLD_FACTORS):
        raise RuntimeError("batch does not preserve eight-world units")
    return values.reshape(
        values.shape[0] // len(source.WORLD_FACTORS),
        len(source.WORLD_FACTORS),
        *values.shape[1:],
    )


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
]:
    expected_worlds = [
        f"{binding}{query}{lexical}"
        for binding, query, lexical in source.WORLD_FACTORS
    ]
    if len(rows) % len(expected_worlds):
        raise RuntimeError("batch size must be a multiple of eight")
    unit_indices = []
    signs = []
    for start in range(0, len(rows), len(expected_worlds)):
        group = rows[start:start + len(expected_worlds)]
        if [row["world"] for row in group] != expected_worlds:
            raise RuntimeError("eight-world order drift")
        if len({int(row["unit_index"]) for row in group}) != 1:
            raise RuntimeError("unit rows are not contiguous")
        unit_indices.append(int(group[0]["unit_index"]))
        signs.append(
            [1.0, -1.0]
            if str(group[0]["q0_slot"]) == "a"
            else [-1.0, 1.0]
        )

    width = max(len(row["input_ids"]) for row in rows)
    max_span = max(
        int(row["anchor_spans"][role][1])
        - int(row["anchor_spans"][role][0])
        + 1
        for row in rows
        for role in protocol.ROLE_ANCHORS
    )
    ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.zeros(
        (len(rows), len(protocol.ROLE_ANCHORS), max_span),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        for role_index, role in enumerate(protocol.ROLE_ANCHORS):
            span_start, span_end = (
                int(value) for value in row["anchor_spans"][role]
            )
            span = list(range(span_start, span_end + 1))
            positions[row_index, role_index, :len(span)] = torch.tensor(
                span, dtype=torch.long
            )
            masks[row_index, role_index, :len(span)] = True
    return (
        ids.to(device),
        attention_mask.to(device),
        positions,
        masks,
        torch.tensor(signs, dtype=torch.float32),
        np.asarray(unit_indices, dtype=np.int64),
    )


def family_contrasts(
    states: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Return canonical B1-B0 for q=0, by role and lexical member."""
    contrasts = []
    for lexical in protocol.LEXICAL_MEMBERS:
        offset = 4 * lexical
        contrasts.append(states[:, offset + 1] - states[:, offset])
    values = torch.stack(contrasts, dim=2)
    return values * signs.to(values.device)[:, :, None, None]


def lexical_contrast_norms(states: torch.Tensor) -> torch.Tensor:
    """Return ||L1-L0|| for q=0, by role and binding world."""
    contrasts = []
    for binding in (0, 1):
        contrasts.append(
            states[:, 4 + binding] - states[:, binding]
        )
    values = torch.stack(contrasts, dim=2)
    return torch.linalg.vector_norm(values.float(), dim=-1)


class ChannelCapture:
    def __init__(
        self,
        layers: list[Any],
        selected_depths: list[int],
        contrasts: np.memmap,
        lexical_norms: np.memmap,
        closure: np.memmap,
    ):
        self.layers = layers
        self.selected_depths = selected_depths
        self.depth_slots = {
            depth: index for index, depth in enumerate(selected_depths)
        }
        self.contrasts = contrasts
        self.lexical_norms = lexical_norms
        self.closure = closure
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.signs: torch.Tensor | None = None
        self.unit_indices: np.ndarray | None = None
        self.current: dict[int, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        signs: torch.Tensor,
        unit_indices: np.ndarray,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.signs = signs
        self.unit_indices = unit_indices
        self.current = {}
        self.counts = defaultdict(int)

    def _states(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.positions is None or self.masks is None:
            raise RuntimeError("channel capture positions missing")
        return grouped(
            gather_span_means(hidden, self.positions, self.masks)
        ).detach()

    def _pre_hook(self, physical_depth: int):
        def hook(module, args):
            self.current[physical_depth] = {
                "upstream_residual": self._states(args[0])
            }
            self.counts[f"{physical_depth}/pre"] += 1
        return hook

    def _component_hook(self, physical_depth: int, name: str):
        def hook(module, args, output):
            self.current[physical_depth][name] = self._states(
                output_tensor(output)
            )
            self.counts[f"{physical_depth}/{name}"] += 1
            return output
        return hook

    def _layer_hook(self, physical_depth: int):
        def hook(module, args, output):
            if (
                self.signs is None
                or self.unit_indices is None
            ):
                raise RuntimeError("channel capture context missing")
            current = self.current[physical_depth]
            current["layer_output"] = self._states(output_tensor(output))
            depth_slot = self.depth_slots[physical_depth]
            for channel_index, channel in enumerate(protocol.CHANNELS):
                states = current[channel]
                self.contrasts[
                    self.unit_indices,
                    depth_slot,
                    channel_index,
                    :,
                    :,
                    :,
                ] = family_contrasts(
                    states, self.signs
                ).to("cpu", dtype=torch.float16).numpy()
                self.lexical_norms[
                    self.unit_indices,
                    depth_slot,
                    channel_index,
                    :,
                    :,
                ] = lexical_contrast_norms(states).cpu().numpy()

            output_states = current["layer_output"]
            accounted = (
                current["upstream_residual"]
                + current["attention_write"]
                + current["mlp_write"]
            )
            error = torch.linalg.vector_norm(
                (output_states - accounted).float(), dim=-1
            )
            transition = torch.linalg.vector_norm(
                (
                    output_states - current["upstream_residual"]
                ).float(),
                dim=-1,
            )
            relative = error / torch.clamp(transition, min=EPS)
            self.closure[
                self.unit_indices, depth_slot, :
            ] = relative.mean(dim=1).cpu().numpy()
            self.counts[f"{physical_depth}/layer"] += 1
            return output
        return hook

    def register(self) -> None:
        for physical_depth in self.selected_depths:
            layer = self.layers[physical_depth - 1]
            self.handles.append(
                layer.register_forward_pre_hook(
                    self._pre_hook(physical_depth)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._component_hook(
                        physical_depth, "attention_write"
                    )
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._component_hook(
                        physical_depth, "mlp_write"
                    )
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    self._layer_hook(physical_depth)
                )
            )

    def end(self) -> None:
        expected = {}
        for physical_depth in self.selected_depths:
            for stage in ("pre", "attention_write", "mlp_write", "layer"):
                expected[f"{physical_depth}/{stage}"] = 1
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"component hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.signs = None
        self.unit_indices = None
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    finite = np.all(np.isfinite(left), axis=-1) & np.all(
        np.isfinite(right), axis=-1
    )
    result = np.full(left.shape[:-1], np.nan, dtype=np.float32)
    if np.any(finite):
        result[finite] = np.sum(
            normalize(left[finite]) * normalize(right[finite]), axis=-1
        )
    return result


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    clean = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(clean)),
        "finite_rate": float(len(clean) / max(1, len(values))),
        "mean": float(np.mean(clean)) if len(clean) else None,
        "median": float(np.median(clean)) if len(clean) else None,
        "positive_rate": (
            float(np.mean(clean > 0)) if len(clean) else None
        ),
    }


def unit_groups(
    units: list[dict[str, Any]],
) -> dict[tuple[str, int, int], list[int]]:
    result: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in units:
        result[(
            str(row["split"]),
            int(row["target_index"]),
            int(row["donor_index"]),
        )].append(int(row["unit_index"]))
    return result


def channel_metrics(
    contrasts: np.ndarray,
    lexical_norms: np.ndarray,
    depths: list[int],
    depth_slots: list[int],
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = unit_groups(units)
    pair_keys = sorted({
        (int(row["target_index"]), int(row["donor_index"]))
        for row in units
    })
    rows: list[dict[str, Any]] = []
    for local_depth, physical_depth in enumerate(depths):
        for channel_index, channel in enumerate(protocol.CHANNELS):
            for role_index, role in enumerate(protocol.ROLE_ANCHORS):
                values = np.asarray(
                    contrasts[
                        :, local_depth, channel_index, role_index
                    ],
                    dtype=np.float32,
                )
                lexical = np.asarray(
                    lexical_norms[
                        :, local_depth, channel_index, role_index
                    ],
                    dtype=np.float32,
                )
                within_member = row_cos(values[:, 0], values[:, 1])
                family_norm = np.linalg.norm(values, axis=-1)
                mean_family_norm = np.nanmean(family_norm, axis=1)
                mean_lexical_norm = np.nanmean(lexical, axis=1)
                family_to_lexical = (
                    mean_family_norm
                    / np.maximum(mean_lexical_norm, EPS)
                )
                unit_directions = normalize(np.nanmean(values, axis=1))
                split_rows: dict[str, Any] = {}
                centroids: dict[
                    tuple[str, int, int], np.ndarray
                ] = {}
                for split in source.SPLITS:
                    context_cosines = []
                    for target, donor in pair_keys:
                        indices = groups[(split, target, donor)]
                        current = unit_directions[indices]
                        finite = np.all(np.isfinite(current), axis=-1)
                        current = current[finite]
                        if len(current):
                            centroid = np.mean(current, axis=0)
                            centroids[(split, target, donor)] = normalize(
                                centroid[None]
                            )[0]
                        for left, right in combinations(
                            range(len(current)), 2
                        ):
                            context_cosines.append(
                                float(np.dot(current[left], current[right]))
                            )
                    split_indices = [
                        int(row["unit_index"])
                        for row in units
                        if row["split"] == split
                    ]
                    split_rows[split] = {
                        "within_unit_member_invariance": scalar_summary(
                            within_member[split_indices]
                        ),
                        "same_pair_cross_context": scalar_summary(
                            np.asarray(
                                context_cosines, dtype=np.float32
                            )
                        ),
                        "family_contrast_norm": scalar_summary(
                            family_norm[split_indices]
                        ),
                        "same_family_lexical_norm": scalar_summary(
                            lexical[split_indices]
                        ),
                        "family_to_lexical_norm_ratio": scalar_summary(
                            family_to_lexical[split_indices]
                        ),
                    }

                matched = []
                shuffled = []
                advantages = []
                for pair_index, (target, donor) in enumerate(pair_keys):
                    left = centroids.get(("discovery", target, donor))
                    right = centroids.get(("confirmation", target, donor))
                    shuffled_pair = pair_keys[
                        (pair_index + 1) % len(pair_keys)
                    ]
                    wrong = centroids.get((
                        "confirmation",
                        shuffled_pair[0],
                        shuffled_pair[1],
                    ))
                    if left is None or right is None or wrong is None:
                        matched.append(np.nan)
                        shuffled.append(np.nan)
                        advantages.append(np.nan)
                        continue
                    matched_value = float(np.dot(left, right))
                    shuffled_value = float(np.dot(left, wrong))
                    matched.append(matched_value)
                    shuffled.append(shuffled_value)
                    advantages.append(matched_value - shuffled_value)
                rows.append({
                    "physical_depth": int(physical_depth),
                    "normalized_depth_slot": int(
                        depth_slots[local_depth]
                    ),
                    "normalized_depth": float(
                        depth_slots[local_depth] / 8.0
                    ),
                    "channel": channel,
                    "role": role,
                    "splits": split_rows,
                    "same_pair_cross_split": scalar_summary(
                        np.asarray(matched, dtype=np.float32)
                    ),
                    "shuffled_pair_cross_split": scalar_summary(
                        np.asarray(shuffled, dtype=np.float32)
                    ),
                    "matched_minus_shuffled": scalar_summary(
                        np.asarray(advantages, dtype=np.float32)
                    ),
                    "all_family_contrast_norm": scalar_summary(
                        family_norm
                    ),
                    "all_same_family_lexical_norm": scalar_summary(
                        lexical
                    ),
                    "all_family_to_lexical_norm_ratio": scalar_summary(
                        family_to_lexical
                    ),
                })
    return rows


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    return {
        "all_finite": bool(finite.all()),
        "finite_value_rate": float(np.mean(finite)),
        "nonfinite_value_count": int(np.size(values) - finite.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = source.read_jsonl(
        source.OUT_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    units = source.read_jsonl(
        source.OUT_ROOT / "protocol" / "units.jsonl"
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
        selected_depths = [
            int(value)
            for value in prereg["model_physical_depths"][args.model]
        ]
        depth_slots = [
            int(value)
            for value in prereg["normalized_depth_slots"][args.model]
        ]
        if not selected_depths or max(selected_depths) >= info.n_layers:
            raise RuntimeError("selected channel depths are invalid")

        contrasts = np.lib.format.open_memmap(
            atlas_dir / "channel_family_contrasts.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(units),
                len(selected_depths),
                len(protocol.CHANNELS),
                len(protocol.ROLE_ANCHORS),
                len(protocol.LEXICAL_MEMBERS),
                info.d_model,
            ),
        )
        lexical_norms = np.lib.format.open_memmap(
            atlas_dir / "channel_lexical_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                len(selected_depths),
                len(protocol.CHANNELS),
                len(protocol.ROLE_ANCHORS),
                2,
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "channel_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(units),
                len(selected_depths),
                len(protocol.ROLE_ANCHORS),
            ),
        )
        contrasts[:] = np.nan
        lexical_norms[:] = np.nan
        closure[:] = np.nan

        capture = ChannelCapture(
            layers,
            selected_depths,
            contrasts,
            lexical_norms,
            closure,
        )
        capture.register()
        try:
            for batch_number, row_batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                (
                    input_ids,
                    attention_mask,
                    positions,
                    masks,
                    signs,
                    unit_indices,
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
                    positions, masks, signs, unit_indices
                )
                with torch.inference_mode():
                    output = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                capture.end()
                del output
                if batch_number % 16 == 0:
                    print(
                        f"[phase1038] {args.model} "
                        f"units={int(unit_indices[-1]) + 1}/{len(units)}",
                        flush=True,
                    )
        finally:
            capture.close()
        contrasts.flush()
        lexical_norms.flush()
        closure.flush()

        metrics = {
            "schema_version": "phase1038_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "channel_depth_rows": channel_metrics(
                contrasts,
                lexical_norms,
                selected_depths,
                depth_slots,
                units,
            ),
            "instrumentation_closure": scalar_summary(closure),
        }
        summary = {
            "schema_version": "phase1038_model_summary.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "protocol_digest": prereg["protocol_digest"],
            "source_protocol_digest": prereg["source_protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": info.model_class,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
            },
            "selected_depths": selected_depths,
            "normalized_depth_slots": depth_slots,
            "sample_counts": {
                "units": len(units),
                "cases": len(cases),
                "ordered_family_pairs": 16,
                "channels": len(protocol.CHANNELS),
                "roles": len(protocol.ROLE_ANCHORS),
            },
            "array_finiteness": {
                "channel_family_contrasts": finite_summary(contrasts),
                "channel_lexical_norms": finite_summary(lexical_norms),
                "channel_closure": finite_summary(closure),
            },
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
