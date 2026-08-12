#!/usr/bin/env python3
"""Run the Phase1044 paired natural-recomputation trajectory atlas."""

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
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1041_position_write_alliance_scan as batch_tools
import phase1044_natural_recompute_trajectory_protocol as protocol


TARGET_BATCH_SIZE = {"qwen3": 16, "glm4": 4, "deepseek7b": 4}
CLEAN_BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    current = np.asarray(values)
    finite = np.isfinite(current)
    return {
        "all_finite": bool(np.all(finite)),
        "finite_value_rate": float(np.mean(finite)),
        "nonfinite_value_count": int(np.sum(~finite)),
    }


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if not len(current):
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "positive_rate": None,
        }
    return {
        "count": int(len(current)),
        "median": float(np.median(current)),
        "mean": float(np.mean(current)),
        "min": float(np.min(current)),
        "max": float(np.max(current)),
        "positive_rate": float(np.mean(current > 0.0)),
    }


def safe_median(values: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    return float(np.median(current)) if len(current) else float("nan")


def safe_rate(values: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    return float(np.mean(current > 0.0)) if len(current) else float("nan")


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    finite = np.all(np.isfinite(left), axis=-1) & np.all(
        np.isfinite(right), axis=-1
    )
    result = np.full(left.shape[0], np.nan, dtype=np.float32)
    if not np.any(finite):
        return result
    numerator = np.sum(left[finite] * right[finite], axis=-1)
    denominator = (
        np.linalg.norm(left[finite], axis=-1)
        * np.linalg.norm(right[finite], axis=-1)
    )
    valid = denominator > EPS
    selected = np.full(np.sum(finite), np.nan, dtype=np.float32)
    selected[valid] = numerator[valid] / denominator[valid]
    result[finite] = selected
    return result


def role_slot(role: str) -> int:
    return protocol.alliance.ROLE_ORDER.index(role)


class SourceCacheCapture:
    def __init__(
        self,
        layer: Any,
        cache: np.memmap,
        case_to_local: dict[int, int],
    ):
        self.layer = layer
        self.cache = cache
        self.case_to_local = case_to_local
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.local_indices: np.ndarray | None = None
        self.current: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        case_indices: np.ndarray,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.local_indices = np.asarray(
            [self.case_to_local[int(value)] for value in case_indices],
            dtype=np.int64,
        )
        self.current = {}
        self.counts = defaultdict(int)

    def _states(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.positions is None or self.masks is None:
            raise RuntimeError("source-cache positions missing")
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None, None].expand_as(positions)
        values = hidden[batch, positions, :].clone()
        return values.masked_fill(~masks[..., None], 0).detach()

    def _mlp_hook(self, module, args, output):
        self.current["mlp_write"] = self._states(output_tensor(output))
        self.counts["mlp_write"] += 1
        return output

    def _layer_hook(self, module, args, output):
        if self.local_indices is None:
            raise RuntimeError("source-cache local indices missing")
        self.current["full_state"] = self._states(output_tensor(output))
        self.counts["full_state"] += 1
        for mode_slot, mode in enumerate(protocol.SOURCE_MODES):
            self.cache[self.local_indices, mode_slot, :, :, :] = (
                self.current[mode]
                .to("cpu", dtype=torch.float16)
                .numpy()
            )
        return output

    def register(self) -> None:
        self.handles.append(
            self.layer.mlp.register_forward_hook(self._mlp_hook)
        )
        self.handles.append(
            self.layer.register_forward_hook(self._layer_hook)
        )

    def end(self) -> None:
        if dict(self.counts) != {"mlp_write": 1, "full_state": 1}:
            raise RuntimeError(
                f"source-cache hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.local_indices = None
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class SourcePatch:
    def __init__(self, module: Any):
        self.module = module
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.payloads: torch.Tensor | None = None
        self.count = 0
        self.handle = None

    def _hook(self, module, args, output):
        if (
            self.positions is None
            or self.masks is None
            or self.payloads is None
        ):
            raise RuntimeError("source patch context missing")
        hidden = output_tensor(output)
        patched = hidden.clone()
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        payloads = self.payloads.to(hidden.device, dtype=hidden.dtype)
        for span_slot in range(protocol.MAX_SOURCE_SPAN):
            active = torch.where(masks[:, span_slot])[0]
            if len(active) == 0:
                continue
            patched[
                active, positions[active, span_slot], :
            ] += payloads[active, span_slot, :]
        self.count += 1
        return replace_output(output, patched)

    def register(self) -> None:
        self.handle = self.module.register_forward_hook(self._hook)

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        payloads: torch.Tensor,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.payloads = payloads
        self.count = 0

    def end(self) -> None:
        if self.count != 1:
            raise RuntimeError(
                f"source patch hook count drift: {self.count}"
            )
        self.positions = None
        self.masks = None
        self.payloads = None

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class TrajectoryCapture:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        response_norms: np.memmap,
        receiver_vectors: np.memmap,
        closure: np.memmap,
    ):
        self.layers = layers
        self.depths = depths
        self.depth_slots = {
            depth: index for index, depth in enumerate(depths)
        }
        self.response_norms = response_norms
        self.receiver_vectors = receiver_vectors
        self.closure = closure
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None
        self.mode_slot = -1
        self.condition_slot = -1
        self.current: dict[int, dict[str, torch.Tensor]] = {}
        self.counts: dict[str, int] = defaultdict(int)
        self.handles = []

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        target_indices: np.ndarray,
        mode_slot: int,
        condition_slot: int,
    ) -> None:
        if len(positions) != 2 * len(target_indices):
            raise RuntimeError("paired trajectory batch drift")
        self.positions = positions
        self.masks = masks
        self.target_indices = target_indices
        self.mode_slot = mode_slot
        self.condition_slot = condition_slot
        self.current = {}
        self.counts = defaultdict(int)

    def _states(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.positions is None or self.masks is None:
            raise RuntimeError("trajectory positions missing")
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None, None].expand_as(positions)
        values = hidden[batch, positions, :].clone()
        return values.masked_fill(~masks[..., None], 0).detach()

    def _pre_hook(self, depth: int):
        def hook(module, args):
            self.current[depth] = {
                "upstream_residual": self._states(args[0])
            }
            self.counts[f"{depth}/pre"] += 1
        return hook

    def _component_hook(self, depth: int, channel: str):
        def hook(module, args, output):
            self.current[depth][channel] = self._states(
                output_tensor(output)
            )
            self.counts[f"{depth}/{channel}"] += 1
            return output
        return hook

    def _layer_hook(self, depth: int):
        def hook(module, args, output):
            if (
                self.target_indices is None
                or self.masks is None
            ):
                raise RuntimeError("trajectory output context missing")
            current = self.current[depth]
            current["layer_output"] = self._states(
                output_tensor(output)
            )
            depth_slot = self.depth_slots[depth]
            pair_masks = self.masks[0::2].to(
                current["layer_output"].device
            )
            token_counts = torch.clamp(
                pair_masks.sum(dim=-1).float(), min=1.0
            )
            for channel_slot, channel in enumerate(protocol.CHANNELS):
                values = current[channel]
                response = values[0::2] - values[1::2]
                norms = torch.linalg.vector_norm(
                    response.float().flatten(start_dim=2), dim=-1
                ) / torch.sqrt(token_counts)
                self.response_norms[
                    self.target_indices,
                    self.mode_slot,
                    self.condition_slot,
                    depth_slot,
                    channel_slot,
                    :,
                ] = norms.cpu().numpy()
                for receiver_slot, site in enumerate(
                    protocol.RECEIVER_SITES
                ):
                    site_slot = protocol.SEMANTIC_SITES.index(site)
                    count = token_counts[:, site_slot, None]
                    vector = (
                        response[:, site_slot, :, :].float().sum(dim=1)
                        / count
                    )
                    self.receiver_vectors[
                        self.target_indices,
                        self.mode_slot,
                        self.condition_slot,
                        depth_slot,
                        channel_slot,
                        receiver_slot,
                        :,
                    ] = vector.to(
                        "cpu", dtype=torch.float16
                    ).numpy()

            accounted = (
                current["upstream_residual"]
                + current["attention_write"]
                + current["mlp_write"]
            )
            error = torch.linalg.vector_norm(
                (current["layer_output"] - accounted).float(), dim=-1
            )
            transition = torch.linalg.vector_norm(
                (
                    current["layer_output"]
                    - current["upstream_residual"]
                ).float(),
                dim=-1,
            )
            relative = error / torch.clamp(transition, min=EPS)
            relative = relative.masked_fill(
                ~self.masks.to(relative.device), torch.nan
            )
            pair_closure = torch.nanmean(relative, dim=(-1, -2))
            pair_closure = 0.5 * (
                pair_closure[0::2] + pair_closure[1::2]
            )
            self.closure[
                self.target_indices,
                self.mode_slot,
                self.condition_slot,
                depth_slot,
            ] = pair_closure.cpu().numpy()
            self.counts[f"{depth}/layer"] += 1
            return output
        return hook

    def register(self) -> None:
        for depth in self.depths:
            layer = self.layers[depth - 1]
            self.handles.append(
                layer.register_forward_pre_hook(self._pre_hook(depth))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._component_hook(depth, "attention_write")
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._component_hook(depth, "mlp_write")
                )
            )
            self.handles.append(
                layer.register_forward_hook(self._layer_hook(depth))
            )

    def end(self) -> None:
        expected = {
            f"{depth}/{stage}": 1
            for depth in self.depths
            for stage in (
                "pre",
                "attention_write",
                "mlp_write",
                "layer",
            )
        }
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"trajectory hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.target_indices = None
        self.mode_slot = -1
        self.condition_slot = -1
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def source_cache_value(
    cache: np.memmap,
    local_index: int,
    mode_slot: int,
    role: str,
    length: int,
) -> np.ndarray:
    return np.asarray(
        cache[
            local_index,
            mode_slot,
            role_slot(role),
            :length,
            :,
        ],
        dtype=np.float32,
    )


def make_paired_patch_batch(
    target_rows: list[dict[str, Any]],
    condition: str,
    mode_slot: int,
    targets_by_atlas: dict[int, dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    case_to_local: dict[int, int],
    source_cache: np.memmap,
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    model_rows = []
    specs = []
    for target in target_rows:
        spec = protocol.condition_spec(
            target, condition, targets_by_atlas
        )
        row = cases[int(spec["target_case_index"])]
        model_rows.extend((row, row))
        specs.append(spec)
    (
        ids,
        attention_mask,
        trajectory_positions,
        trajectory_masks,
        pre_positions,
        _,
    ) = batch_tools.make_clean_batch(
        model_rows,
        pad_token_id=pad_token_id,
        device=device,
    )
    source_positions = torch.zeros(
        (len(model_rows), protocol.MAX_SOURCE_SPAN), dtype=torch.long
    )
    source_masks = torch.zeros_like(source_positions, dtype=torch.bool)
    payloads = np.zeros(
        (
            len(model_rows),
            protocol.MAX_SOURCE_SPAN,
            source_cache.shape[-1],
        ),
        dtype=np.float32,
    )
    payload_norms = np.zeros(len(target_rows), dtype=np.float32)

    for target_slot, (target, spec) in enumerate(
        zip(target_rows, specs)
    ):
        patched_slot = 2 * target_slot
        donor_target = targets_by_atlas[
            int(spec["donor_atlas_index"])
        ]
        site = str(spec["source_site"])
        target_role = protocol.source.semantic_role(site, target)
        donor_role = protocol.source.semantic_role(site, donor_target)
        target_case = int(spec["target_case_index"])
        donor_case = int(spec["donor_case_index"])
        target_row = cases[target_case]
        donor_row = cases[donor_case]
        target_start, target_end = (
            int(value)
            for value in target_row["anchor_spans"][target_role]
        )
        donor_start, donor_end = (
            int(value)
            for value in donor_row["anchor_spans"][donor_role]
        )
        target_span = list(range(target_start, target_end + 1))
        donor_span = list(range(donor_start, donor_end + 1))
        if (
            len(target_span) != len(donor_span)
            or len(target_span) > protocol.MAX_SOURCE_SPAN
        ):
            raise RuntimeError(
                f"source span mismatch for {condition}: "
                f"{len(target_span)} != {len(donor_span)}"
            )
        target_value = source_cache_value(
            source_cache,
            case_to_local[target_case],
            mode_slot,
            target_role,
            len(target_span),
        )
        donor_value = source_cache_value(
            source_cache,
            case_to_local[donor_case],
            mode_slot,
            donor_role,
            len(donor_span),
        )
        payload = donor_value - target_value
        source_positions[
            patched_slot, :len(target_span)
        ] = torch.tensor(target_span, dtype=torch.long)
        source_masks[patched_slot, :len(target_span)] = True
        payloads[patched_slot, :len(target_span), :] = payload
        payload_norms[target_slot] = float(
            np.linalg.norm(payload.astype(np.float32))
            / math.sqrt(len(target_span))
        )

    return (
        ids,
        attention_mask,
        trajectory_positions,
        trajectory_masks,
        pre_positions,
        source_positions,
        source_masks,
        torch.from_numpy(payloads),
        np.asarray(
            [int(row["atlas_index"]) for row in target_rows],
            dtype=np.int64,
        ),
        payload_norms,
    )


def behavior_summary(
    candidate_logits: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    values = np.asarray(candidate_logits, dtype=np.float32)
    expected = np.asarray(
        [int(row["expected_index"]) for row in cases], dtype=np.int64
    )
    finite = np.all(np.isfinite(values), axis=-1)
    prediction = np.full(len(cases), -1, dtype=np.int64)
    if np.any(finite):
        prediction[finite] = np.argmax(values[finite], axis=-1)
    return {
        "row_count": len(cases),
        "finite_row_rate": float(np.mean(finite)),
        "candidate_accuracy": float(np.mean(prediction == expected)),
    }


def intervention_metrics(
    logits: np.ndarray,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    target_index = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_index = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    rows = np.arange(len(targets), dtype=np.int64)
    patched_margin = (
        logits[rows, 0, cross_index] - logits[rows, 0, target_index]
    )
    zero_margin = (
        logits[rows, 1, cross_index] - logits[rows, 1, target_index]
    )
    return {
        "cross_minus_target_margin_shift": scalar_summary(
            patched_margin - zero_margin
        ),
        "patched_margin": scalar_summary(patched_margin),
        "paired_zero_margin": scalar_summary(zero_margin),
        "finite": finite_summary(logits),
    }


def trajectory_metrics(
    vectors: np.ndarray,
    norms: np.ndarray,
    prereg: dict[str, Any],
) -> list[dict[str, Any]]:
    condition_slot = {
        name: index for index, name in enumerate(protocol.CONDITIONS)
    }
    rows = []
    gate = prereg["descriptive_repetition_gate"]
    for mode_slot, mode in enumerate(protocol.SOURCE_MODES):
        for depth_slot in range(vectors.shape[3]):
            for channel_slot, channel in enumerate(protocol.CHANNELS):
                for receiver_slot, receiver in enumerate(
                    protocol.RECEIVER_SITES
                ):
                    x0 = np.asarray(
                        vectors[
                            :,
                            mode_slot,
                            condition_slot["cross_selected_l0"],
                            depth_slot,
                            channel_slot,
                            receiver_slot,
                            :,
                        ],
                        dtype=np.float32,
                    )
                    x1 = np.asarray(
                        vectors[
                            :,
                            mode_slot,
                            condition_slot["cross_selected_l1"],
                            depth_slot,
                            channel_slot,
                            receiver_slot,
                            :,
                        ],
                        dtype=np.float32,
                    )
                    s0 = np.asarray(
                        vectors[
                            :,
                            mode_slot,
                            condition_slot["cross_shuffled_l0"],
                            depth_slot,
                            channel_slot,
                            receiver_slot,
                            :,
                        ],
                        dtype=np.float32,
                    )
                    s1 = np.asarray(
                        vectors[
                            :,
                            mode_slot,
                            condition_slot["cross_shuffled_l1"],
                            depth_slot,
                            channel_slot,
                            receiver_slot,
                            :,
                        ],
                        dtype=np.float32,
                    )
                    matched = cosine_rows(x0, x1)
                    shuffled = 0.5 * (
                        cosine_rows(x0, s0) + cosine_rows(x1, s1)
                    )
                    advantage = matched - shuffled
                    site_slot = protocol.SEMANTIC_SITES.index(receiver)
                    family_norm = 0.5 * (
                        norms[
                            :,
                            mode_slot,
                            condition_slot["cross_selected_l0"],
                            depth_slot,
                            channel_slot,
                            site_slot,
                        ]
                        + norms[
                            :,
                            mode_slot,
                            condition_slot["cross_selected_l1"],
                            depth_slot,
                            channel_slot,
                            site_slot,
                        ]
                    )
                    lexical_norm = norms[
                        :,
                        mode_slot,
                        condition_slot["same_family_lexical_l0"],
                        depth_slot,
                        channel_slot,
                        site_slot,
                    ]
                    ratio = safe_median(family_norm) / max(
                        safe_median(lexical_norm), EPS
                    )
                    matched_median = safe_median(matched)
                    advantage_median = safe_median(advantage)
                    advantage_rate = safe_rate(advantage)
                    passed = (
                        np.isfinite(matched_median)
                        and matched_median
                        >= gate["cross_lexical_cosine_median_min"]
                        and advantage_median
                        >= gate[
                            "matched_minus_shuffled_cosine_median_min"
                        ]
                        and advantage_rate
                        >= gate["advantage_positive_rate_min"]
                        and ratio
                        >= gate[
                            "family_to_same_lexical_norm_ratio_min"
                        ]
                    )
                    rows.append({
                        "source_mode": mode,
                        "depth_slot": depth_slot + 1,
                        "channel": channel,
                        "receiver_site": receiver,
                        "cross_lexical_cosine": scalar_summary(matched),
                        "shuffled_cosine": scalar_summary(shuffled),
                        "matched_minus_shuffled": scalar_summary(
                            advantage
                        ),
                        "family_response_norm": scalar_summary(
                            family_norm
                        ),
                        "same_lexical_response_norm": scalar_summary(
                            lexical_norm
                        ),
                        "family_to_same_lexical_norm_ratio": float(ratio),
                        "descriptive_gate_passed": bool(passed),
                    })
    return rows


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1044 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    targets_by_atlas = {
        int(row["atlas_index"]): row for row in targets
    }
    cases = {int(row["case_index"]): row for row in cases_list}
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    model_depths = prereg["model_depths"][model_name]
    source_depth = int(model_depths["source_depth"])
    receiver_depths = [
        int(value) for value in model_depths["receiver_depths"]
    ]
    atlas_dir = protocol.OUT_ROOT / "atlas" / model_name
    atlas_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None

    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )

        source_cache = np.lib.format.open_memmap(
            atlas_dir / "source_channels.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(protocol.SOURCE_MODES),
                len(protocol.alliance.ROLE_ORDER),
                protocol.MAX_ROLE_SPAN,
                info.d_model,
            ),
        )
        clean_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(protocol.material.FAMILIES)),
        )
        response_norms = np.lib.format.open_memmap(
            atlas_dir / "response_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_MODES),
                len(protocol.CONDITIONS),
                len(receiver_depths),
                len(protocol.CHANNELS),
                len(protocol.SEMANTIC_SITES),
            ),
        )
        receiver_vectors = np.lib.format.open_memmap(
            atlas_dir / "receiver_response_vectors.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(targets),
                len(protocol.SOURCE_MODES),
                len(protocol.CONDITIONS),
                len(receiver_depths),
                len(protocol.CHANNELS),
                len(protocol.RECEIVER_SITES),
                info.d_model,
            ),
        )
        candidate_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_MODES),
                len(protocol.CONDITIONS),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        source_payload_norms = np.lib.format.open_memmap(
            atlas_dir / "source_payload_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_MODES),
                len(protocol.CONDITIONS),
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "channel_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_MODES),
                len(protocol.CONDITIONS),
                len(receiver_depths),
            ),
        )
        for array in (
            source_cache,
            clean_logits,
            response_norms,
            receiver_vectors,
            candidate_logits,
            source_payload_norms,
            closure,
        ):
            array[:] = np.nan

        source_capture = SourceCacheCapture(
            layers[source_depth - 1], source_cache, case_to_local
        )
        source_capture.register()
        try:
            for row_batch in chunks(
                cases_list, CLEAN_BATCH_SIZE[model_name]
            ):
                (
                    input_ids,
                    attention_mask,
                    positions,
                    masks,
                    pre_positions,
                    case_indices,
                ) = batch_tools.make_clean_batch(
                    row_batch,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                source_capture.begin(positions, masks, case_indices)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                source_capture.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                local = np.asarray(
                    [case_to_local[int(value)] for value in case_indices],
                    dtype=np.int64,
                )
                clean_logits[local] = selected.detach().cpu().numpy()
                del output, logits, selected
        finally:
            source_capture.close()
        source_cache.flush()
        clean_logits.flush()

        trajectory = TrajectoryCapture(
            layers,
            receiver_depths,
            response_norms,
            receiver_vectors,
            closure,
        )
        trajectory.register()
        try:
            for mode_slot, mode in enumerate(protocol.SOURCE_MODES):
                source_layer = layers[source_depth - 1]
                module = (
                    source_layer.mlp
                    if mode == "mlp_write"
                    else source_layer
                )
                patch = SourcePatch(module)
                patch.register()
                try:
                    for condition_slot, condition in enumerate(
                        protocol.CONDITIONS
                    ):
                        for target_batch in chunks(
                            targets, TARGET_BATCH_SIZE[model_name]
                        ):
                            (
                                input_ids,
                                attention_mask,
                                trajectory_positions,
                                trajectory_masks,
                                pre_positions,
                                source_positions,
                                source_masks,
                                payloads,
                                target_indices,
                                payload_norm,
                            ) = make_paired_patch_batch(
                                target_batch,
                                condition,
                                mode_slot,
                                targets_by_atlas,
                                cases,
                                case_to_local,
                                source_cache,
                                pad_token_id=pad_token_id,
                                device=device,
                            )
                            patch.begin(
                                source_positions, source_masks, payloads
                            )
                            trajectory.begin(
                                trajectory_positions,
                                trajectory_masks,
                                target_indices,
                                mode_slot,
                                condition_slot,
                            )
                            with torch.inference_mode():
                                output = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    use_cache=False,
                                    return_dict=True,
                                )
                            trajectory.end()
                            patch.end()
                            logits = output.logits
                            batch = torch.arange(
                                logits.shape[0], device=logits.device
                            )
                            selected = logits[
                                batch,
                                pre_positions.to(logits.device),
                                :,
                            ].float().index_select(
                                -1, candidate_ids.to(logits.device)
                            )
                            pair = selected.reshape(
                                len(target_batch),
                                2,
                                len(protocol.material.FAMILIES),
                            )
                            candidate_logits[
                                target_indices,
                                mode_slot,
                                condition_slot,
                                :,
                                :,
                            ] = pair.detach().cpu().numpy()
                            source_payload_norms[
                                target_indices, mode_slot, condition_slot
                            ] = payload_norm
                            del output, logits, selected, pair
                finally:
                    patch.close()
        finally:
            trajectory.close()

        for array in (
            response_norms,
            receiver_vectors,
            candidate_logits,
            source_payload_norms,
            closure,
        ):
            array.flush()

        behavior = behavior_summary(clean_logits, cases_list)
        intervention = {}
        for mode_slot, mode in enumerate(protocol.SOURCE_MODES):
            intervention[mode] = {}
            for condition_slot, condition in enumerate(
                protocol.CONDITIONS
            ):
                intervention[mode][condition] = intervention_metrics(
                    candidate_logits[
                        :, mode_slot, condition_slot, :, :
                    ],
                    targets,
                )
        trajectory_rows = trajectory_metrics(
            receiver_vectors, response_norms, prereg
        )
        protocol.write_jsonl(
            atlas_dir / "trajectory_cells.jsonl", trajectory_rows
        )
        passed_cells = [
            row for row in trajectory_rows
            if row["descriptive_gate_passed"]
        ]

        clean_zero_max = {}
        for condition_slot, condition in enumerate(protocol.CONDITIONS):
            world = "b0l1" if condition.endswith("_l1") else "b0l0"
            case_indices = np.asarray(
                [
                    case_to_local[int(row["world_case_indices"][world])]
                    for row in targets
                ],
                dtype=np.int64,
            )
            for mode_slot, mode in enumerate(protocol.SOURCE_MODES):
                delta = np.abs(
                    candidate_logits[
                        :, mode_slot, condition_slot, 1, :
                    ]
                    - clean_logits[case_indices]
                )
                finite = delta[np.isfinite(delta)]
                clean_zero_max[f"{mode}/{condition}"] = (
                    float(np.max(finite)) if len(finite) else None
                )

        summary = {
            "schema_version": "phase1044_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "vocab_size": info.vocab_size,
                "model_class": info.model_class,
            },
            "source_depth": source_depth,
            "receiver_depths": receiver_depths,
            "behavior": behavior,
            "source_cache_finite": finite_summary(source_cache),
            "response_norms_finite": finite_summary(response_norms),
            "receiver_vectors_finite": finite_summary(receiver_vectors),
            "candidate_logits_finite": finite_summary(candidate_logits),
            "closure": scalar_summary(closure),
            "paired_zero_vs_clean_max_abs": clean_zero_max,
            "intervention_metrics": intervention,
            "trajectory_cell_count": len(trajectory_rows),
            "descriptive_pass_count": len(passed_cells),
            "descriptive_pass_cells": passed_cells,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "behavior_accuracy": behavior["candidate_accuracy"],
            "descriptive_pass_count": len(passed_cells),
            "source_cache_finite_rate": summary[
                "source_cache_finite"
            ]["finite_value_rate"],
            "response_finite_rate": summary[
                "receiver_vectors_finite"
            ]["finite_value_rate"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
