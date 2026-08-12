#!/usr/bin/env python3
"""Map fresh translation K/V phases and naturally recomputed trajectories."""

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

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1049_qkv_read_path_scan as route_tools
import phase1052_full_vocab_kv_bridge_scan as bridge_scan
import phase1054_joint_kv_rollout_scan as rollout_tools
import phase1057_translation_trajectory_protocol as protocol


PAIR_BATCH_SIZE = bridge_scan.PAIR_BATCH_SIZE
CHANNELS = ("k", "v")


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in (
            "both_counterfactual_mask",
            "valid_target_indices",
        )
    }


def rate(result: dict[str, Any]) -> float:
    return float(result["both_counterfactual_top1_rate"])


def nonfinite_count(value: Any) -> int:
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, dict):
        return sum(nonfinite_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nonfinite_count(item) for item in value)
    return 0


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def valid_targets(
    rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    mask, _ = bridge_scan.clean_mask_and_coverage(rows, cases, clean)
    return [row for row, keep in zip(rows, mask) if keep]


def evenly_spaced(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    indices = [(index * len(rows)) // count for index in range(count)]
    return [rows[index] for index in indices]


class OnlineChannelSwap:
    """Swap selected source-position projection channels within paired rows."""

    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        groups: list[int],
        channels: list[str],
        head_dim: int,
    ) -> None:
        self.layers = layers
        self.depths = depths
        self.groups = groups
        self.channels = channels
        self.head_dim = head_dim
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.depths:
            attention = self.layers[depth - 1].self_attn
            for channel in self.channels:
                projection = getattr(attention, f"{channel}_proj")
                self.handles.append(
                    projection.register_forward_hook(
                        self._hook(channel, depth)
                    )
                )

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.counts = {}

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if self.positions is None or self.masks is None:
                raise RuntimeError("channel swap context missing")
            hidden = route_tools.output_tensor(output)
            if hidden.shape[0] % 2:
                raise RuntimeError("paired batch is not even")
            patched = hidden.clone()
            positions = self.positions.to(hidden.device)
            masks = self.masks.to(hidden.device)
            even = torch.arange(
                0, hidden.shape[0], 2, device=hidden.device
            )
            odd = even + 1
            for span_slot in range(positions.shape[1]):
                valid = masks[even, span_slot] & masks[odd, span_slot]
                pair_slots = torch.where(valid)[0]
                if len(pair_slots) == 0:
                    continue
                even_rows = even[pair_slots]
                odd_rows = odd[pair_slots]
                even_pos = positions[even_rows, span_slot]
                odd_pos = positions[odd_rows, span_slot]
                for group in self.groups:
                    start = group * self.head_dim
                    end = start + self.head_dim
                    even_value = hidden[
                        even_rows, even_pos, start:end
                    ].clone()
                    odd_value = hidden[
                        odd_rows, odd_pos, start:end
                    ].clone()
                    patched[
                        even_rows, even_pos, start:end
                    ] = odd_value
                    patched[
                        odd_rows, odd_pos, start:end
                    ] = even_value
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return route_tools.replace_output(output, patched)

        return hook

    def end(self) -> None:
        expected = {
            (channel, depth)
            for channel in self.channels
            for depth in self.depths
        }
        if set(self.counts) != expected or any(
            value != 1 for value in self.counts.values()
        ):
            raise RuntimeError(
                f"channel swap hook count drift: {self.counts}"
            )
        self.positions = None
        self.masks = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def run_output_condition(
    model,
    device: torch.device,
    layers: list[Any],
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    condition: dict[str, Any] | None,
    *,
    head_dim: int,
    pad_token_id: int,
    pair_batch_size: int,
) -> dict[str, np.ndarray]:
    if condition is None or set(condition["channels"]) == set(CHANNELS):
        bridge_condition = None
        if condition is not None:
            bridge_condition = {
                key: value for key, value in condition.items()
                if key != "channels"
            }
        return bridge_scan.run_condition(
            model,
            device,
            layers,
            target_rows,
            cases,
            bridge_condition,
            head_dim=head_dim,
            pad_token_id=pad_token_id,
            pair_batch_size=pair_batch_size,
        )

    top1 = np.empty((len(target_rows), 2), dtype=np.int32)
    finite = np.empty((len(target_rows), 2), dtype=bool)
    margin = np.empty((len(target_rows), 2), dtype=np.float32)
    swap = OnlineChannelSwap(
        layers,
        [int(value) for value in condition["depths"]],
        [int(value) for value in condition["groups"]],
        [str(value) for value in condition["channels"]],
        head_dim,
    )
    swap.register()
    try:
        for start in range(0, len(target_rows), pair_batch_size):
            target_batch = target_rows[start:start + pair_batch_size]
            (
                input_ids,
                attention_mask,
                lengths,
                positions,
                masks,
                token_ids,
            ) = bridge_scan.pair_batch(
                target_batch,
                cases,
                str(condition["site"]),
                pad_token_id=pad_token_id,
                device=device,
            )
            swap.begin(positions, masks)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            swap.end()
            logits = output.logits.float()
            batch_rows = torch.arange(
                logits.shape[0], device=logits.device
            )
            boundary = logits[
                batch_rows, lengths.to(logits.device) - 1, :
            ]
            is_finite = torch.isfinite(boundary).all(dim=-1)
            safe = torch.where(
                torch.isfinite(boundary),
                boundary,
                torch.full_like(boundary, -torch.inf),
            )
            predicted = torch.argmax(safe, dim=-1)
            token_ids = token_ids.to(safe.device)
            own = safe.gather(1, token_ids[:, :1]).squeeze(1)
            counter = safe.gather(1, token_ids[:, 1:2]).squeeze(1)
            count = len(target_batch)
            top1[start:start + count] = predicted.reshape(
                count, 2
            ).detach().cpu().numpy()
            finite[start:start + count] = is_finite.reshape(
                count, 2
            ).detach().cpu().numpy()
            margin[start:start + count] = (
                counter - own
            ).reshape(count, 2).detach().cpu().numpy()
            del output, logits, boundary, safe, predicted, own, counter
    finally:
        swap.close()
    return {"top1": top1, "finite": finite, "margin": margin}


class TrajectoryController:
    """Capture post-hook source K/V and boundary residual trajectories."""

    def __init__(
        self,
        layers: list[Any],
        groups: list[int],
        head_dim: int,
    ) -> None:
        self.layers = layers
        self.groups = groups
        self.head_dim = head_dim
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.lengths: torch.Tensor | None = None
        self.patch_depths: set[int] = set()
        self.patch_channels: set[str] = set()
        self.capture: dict[str, dict[int, torch.Tensor]] = {}
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth, layer in enumerate(self.layers, start=1):
            attention = layer.self_attn
            for channel in CHANNELS:
                projection = getattr(attention, f"{channel}_proj")
                self.handles.append(
                    projection.register_forward_hook(
                        self._projection_hook(channel, depth)
                    )
                )
            self.handles.append(
                layer.register_forward_hook(
                    self._layer_hook(depth)
                )
            )

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        lengths: torch.Tensor,
        *,
        patch_depths: list[int],
        patch_channels: list[str],
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.lengths = lengths
        self.patch_depths = set(int(value) for value in patch_depths)
        self.patch_channels = set(str(value) for value in patch_channels)
        self.capture = {
            "source_k": {},
            "source_v": {},
            "boundary": {},
        }
        self.counts = {}

    def _source_mean(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.positions is None or self.masks is None:
            raise RuntimeError("trajectory source context missing")
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        rows = []
        for row in range(hidden.shape[0]):
            valid = positions[row][masks[row]]
            if len(valid) == 0:
                raise RuntimeError("empty source span")
            rows.append(hidden[row, valid, :].mean(dim=0))
        return torch.stack(rows).detach().float().cpu()

    def _swap(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.positions is None or self.masks is None:
            raise RuntimeError("trajectory swap context missing")
        patched = hidden.clone()
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        even = torch.arange(
            0, hidden.shape[0], 2, device=hidden.device
        )
        odd = even + 1
        for span_slot in range(positions.shape[1]):
            valid = masks[even, span_slot] & masks[odd, span_slot]
            pair_slots = torch.where(valid)[0]
            if len(pair_slots) == 0:
                continue
            even_rows = even[pair_slots]
            odd_rows = odd[pair_slots]
            even_pos = positions[even_rows, span_slot]
            odd_pos = positions[odd_rows, span_slot]
            for group in self.groups:
                start = group * self.head_dim
                end = start + self.head_dim
                even_value = hidden[
                    even_rows, even_pos, start:end
                ].clone()
                odd_value = hidden[
                    odd_rows, odd_pos, start:end
                ].clone()
                patched[
                    even_rows, even_pos, start:end
                ] = odd_value
                patched[
                    odd_rows, odd_pos, start:end
                ] = even_value
        return patched

    def _projection_hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            hidden = route_tools.output_tensor(output)
            patched = (
                self._swap(hidden)
                if (
                    depth in self.patch_depths
                    and channel in self.patch_channels
                )
                else hidden
            )
            self.capture[f"source_{channel}"][depth] = (
                self._source_mean(patched)
            )
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            if patched is hidden:
                return None
            return route_tools.replace_output(output, patched)

        return hook

    def _layer_hook(self, depth: int):
        def hook(_module, _inputs, output):
            if self.lengths is None:
                raise RuntimeError("trajectory boundary context missing")
            hidden = route_tools.output_tensor(output)
            rows = torch.arange(
                hidden.shape[0], device=hidden.device
            )
            boundary = hidden[
                rows, self.lengths.to(hidden.device) - 1, :
            ]
            self.capture["boundary"][depth] = (
                boundary.detach().float().cpu()
            )
            key = ("boundary", depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return None

        return hook

    def end(self) -> dict[str, dict[int, torch.Tensor]]:
        expected = {
            (kind, depth)
            for depth in range(1, len(self.layers) + 1)
            for kind in ("k", "v", "boundary")
        }
        if set(self.counts) != expected or any(
            value != 1 for value in self.counts.values()
        ):
            raise RuntimeError(
                f"trajectory hook count drift: {self.counts}"
            )
        result = self.capture
        self.positions = None
        self.masks = None
        self.lengths = None
        self.capture = {}
        return result

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def boundary_output(
    output,
    lengths: torch.Tensor,
    token_ids: torch.Tensor,
) -> dict[str, np.ndarray]:
    logits = output.logits.float()
    rows = torch.arange(logits.shape[0], device=logits.device)
    boundary = logits[rows, lengths.to(logits.device) - 1, :]
    finite = torch.isfinite(boundary).all(dim=-1)
    safe = torch.where(
        torch.isfinite(boundary),
        boundary,
        torch.full_like(boundary, -torch.inf),
    )
    top1 = torch.argmax(safe, dim=-1)
    token_ids = token_ids.to(safe.device)
    own = safe.gather(1, token_ids[:, :1]).squeeze(1)
    counter = safe.gather(1, token_ids[:, 1:2]).squeeze(1)
    return {
        "top1": top1.detach().cpu().numpy(),
        "finite": finite.detach().cpu().numpy(),
        "margin": (counter - own).detach().cpu().numpy(),
    }


def relative_metrics(
    clean: torch.Tensor,
    patched: torch.Tensor,
) -> dict[str, np.ndarray]:
    if clean.shape != patched.shape or clean.shape[0] % 2:
        raise RuntimeError("trajectory tensor geometry drift")
    donor_index = torch.arange(clean.shape[0]) ^ 1
    donor = clean[donor_index]
    donor_delta = donor - clean
    move = patched - clean
    donor_norm = torch.linalg.vector_norm(donor_delta, dim=-1)
    move_norm = torch.linalg.vector_norm(move, dim=-1)
    donor_distance = torch.linalg.vector_norm(
        patched - donor, dim=-1
    )
    eps = torch.finfo(torch.float32).eps
    valid = donor_norm > eps
    cosine = torch.zeros_like(donor_norm)
    cosine[valid] = (
        (move[valid] * donor_delta[valid]).sum(dim=-1)
        / (
            move_norm[valid].clamp_min(eps)
            * donor_norm[valid]
        )
    )
    progress = (
        move_norm - donor_distance
    ) / (move_norm + donor_distance).clamp_min(eps)
    move_ratio = move_norm / donor_norm.clamp_min(eps)
    return {
        "valid": valid.numpy(),
        "donor_closer": (donor_distance < move_norm).numpy(),
        "progress": progress.numpy(),
        "cosine": cosine.numpy(),
        "move_ratio": move_ratio.numpy(),
    }


def append_metric(
    store: dict[str, list[float]],
    values: dict[str, np.ndarray],
) -> None:
    valid = values["valid"].astype(bool)
    for key in ("donor_closer", "progress", "cosine", "move_ratio"):
        store[key].extend(
            float(value) for value in values[key][valid]
        )


def summarize_values(values: dict[str, list[float]]) -> dict[str, Any]:
    count = len(values["progress"])
    if not count:
        return {
            "arm_count": 0,
            "donor_closer_rate": None,
            "progress_median": None,
            "cosine_mean": None,
            "move_ratio_median": None,
        }
    return {
        "arm_count": count,
        "donor_closer_rate": float(
            np.mean(values["donor_closer"])
        ),
        "progress_median": float(np.median(values["progress"])),
        "cosine_mean": float(np.mean(values["cosine"])),
        "move_ratio_median": float(
            np.median(values["move_ratio"])
        ),
    }


def run_trajectory(
    model,
    device: torch.device,
    layers: list[Any],
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    plan: dict[str, Any],
    *,
    head_dim: int,
    pad_token_id: int,
    pair_batch_size: int,
) -> dict[str, Any]:
    modes = {
        "early": [int(v) for v in plan["early_depths"]],
        "post": [int(v) for v in plan["postsource_depths"]],
        "all": [int(v) for v in plan["all_layers"]],
    }
    n_layers = len(layers)
    stores: dict[
        str,
        dict[str, dict[int, dict[str, list[float]]]],
    ] = {
        mode: {
            object_name: {
                depth: defaultdict(list)
                for depth in range(1, n_layers + 1)
            }
            for object_name in ("source_kv", "boundary")
        }
        for mode in modes
    }
    output_rows = {
        mode: {
            "top1": np.empty((len(target_rows), 2), dtype=np.int32),
            "finite": np.empty((len(target_rows), 2), dtype=bool),
            "margin": np.empty((len(target_rows), 2), dtype=np.float32),
        }
        for mode in ("clean", *modes)
    }
    late_pair_progress: dict[str, dict[str, list[float]]] = {
        mode: {"flip": [], "nonflip": []} for mode in modes
    }
    controller = TrajectoryController(
        layers,
        [int(value) for value in plan["all_groups"]],
        head_dim,
    )
    controller.register()
    try:
        for start in range(0, len(target_rows), pair_batch_size):
            target_batch = target_rows[start:start + pair_batch_size]
            (
                input_ids,
                attention_mask,
                lengths,
                positions,
                masks,
                token_ids,
            ) = bridge_scan.pair_batch(
                target_batch,
                cases,
                "source_term",
                pad_token_id=pad_token_id,
                device=device,
            )
            captures = {}
            for mode in ("clean", *modes):
                controller.begin(
                    positions,
                    masks,
                    lengths,
                    patch_depths=modes.get(mode, []),
                    patch_channels=list(CHANNELS),
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                captures[mode] = controller.end()
                result = boundary_output(output, lengths, token_ids)
                count = len(target_batch)
                for key in ("top1", "finite", "margin"):
                    output_rows[mode][key][start:start + count] = (
                        result[key].reshape(count, 2)
                    )
                del output

            clean_capture = captures["clean"]
            for mode in modes:
                patched_capture = captures[mode]
                layer_progress = []
                for depth in range(1, n_layers + 1):
                    clean_kv = torch.cat(
                        (
                            clean_capture["source_k"][depth],
                            clean_capture["source_v"][depth],
                        ),
                        dim=-1,
                    )
                    patched_kv = torch.cat(
                        (
                            patched_capture["source_k"][depth],
                            patched_capture["source_v"][depth],
                        ),
                        dim=-1,
                    )
                    source_values = relative_metrics(
                        clean_kv, patched_kv
                    )
                    boundary_values = relative_metrics(
                        clean_capture["boundary"][depth],
                        patched_capture["boundary"][depth],
                    )
                    append_metric(
                        stores[mode]["source_kv"][depth],
                        source_values,
                    )
                    append_metric(
                        stores[mode]["boundary"][depth],
                        boundary_values,
                    )
                    if depth > (2 * n_layers) // 3:
                        layer_progress.append(
                            boundary_values["progress"]
                        )
                late_progress = np.mean(
                    np.stack(layer_progress, axis=0),
                    axis=0,
                ).reshape(len(target_batch), 2).mean(axis=1)
                mode_top1 = output_rows[mode]["top1"][
                    start:start + len(target_batch)
                ]
                clean_top1 = output_rows["clean"]["top1"][
                    start:start + len(target_batch)
                ]
                flipped = (
                    (mode_top1[:, 0] == clean_top1[:, 1])
                    & (mode_top1[:, 1] == clean_top1[:, 0])
                )
                for value, is_flip in zip(late_progress, flipped):
                    key = "flip" if bool(is_flip) else "nonflip"
                    late_pair_progress[mode][key].append(float(value))
            del captures
    finally:
        controller.close()

    summaries = {
        mode: {
            object_name: {
                str(depth): summarize_values(
                    stores[mode][object_name][depth]
                )
                for depth in range(1, n_layers + 1)
            }
            for object_name in ("source_kv", "boundary")
        }
        for mode in modes
    }
    associations = {}
    for mode, groups in late_pair_progress.items():
        associations[mode] = {
            "flip_pair_count": len(groups["flip"]),
            "nonflip_pair_count": len(groups["nonflip"]),
            "flip_late_boundary_progress_median": (
                float(np.median(groups["flip"]))
                if groups["flip"] else None
            ),
            "nonflip_late_boundary_progress_median": (
                float(np.median(groups["nonflip"]))
                if groups["nonflip"] else None
            ),
        }
    return {
        "pair_count": len(target_rows),
        "layer_summaries": summaries,
        "output_rows": output_rows,
        "late_boundary_associations": associations,
    }


def canonical(condition: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(condition["site"]),
        tuple(sorted(str(value) for value in condition["channels"])),
        tuple(sorted(int(value) for value in condition["groups"])),
        tuple(sorted(int(value) for value in condition["depths"])),
    )


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1057 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    discovery_rows = [
        row for row in targets if row["split"] == "discovery"
    ]
    confirmation_rows = [
        row for row in targets if row["split"] == "confirmation"
    ]
    plan = prereg["model_plans"][model_name]
    all_groups = [int(value) for value in plan["all_groups"]]
    early_depths = [int(value) for value in plan["early_depths"]]
    post_depths = [
        int(value) for value in plan["postsource_depths"]
    ]
    all_depths = [int(value) for value in plan["all_layers"]]
    slots = [
        [int(value) for value in slot]
        for slot in plan["postsource_slots"]
    ]
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
        layers = list(get_layers(model))
        width = bridge_scan.projection_width(
            layers[0].self_attn.k_proj
        )
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = width // n_kv_heads
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        batch_size = PAIR_BATCH_SIZE[model_name]

        clean_discovery = run_output_condition(
            model,
            device,
            layers,
            discovery_rows,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=batch_size,
        )
        discovery_valid = valid_targets(
            discovery_rows, cases, clean_discovery
        )
        clean_confirmation = run_output_condition(
            model,
            device,
            layers,
            confirmation_rows,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=batch_size,
        )
        confirmation_valid = valid_targets(
            confirmation_rows, cases, clean_confirmation
        )
        clean_valid = run_output_condition(
            model,
            device,
            layers,
            confirmation_valid,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=batch_size,
        )
        valid_mask = np.ones(len(confirmation_valid), dtype=bool)

        base_specs = {
            "source_early_kv": {
                "site": "source_term",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": early_depths,
            },
            "source_post_kv": {
                "site": "source_term",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": post_depths,
            },
            "source_all_kv": {
                "site": "source_term",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": all_depths,
            },
            "source_post_k_only": {
                "site": "source_term",
                "channels": ["k"],
                "groups": all_groups,
                "depths": post_depths,
            },
            "source_post_v_only": {
                "site": "source_term",
                "channels": ["v"],
                "groups": all_groups,
                "depths": post_depths,
            },
            "source_frozen_rectangle": {
                "site": "source_term",
                "channels": list(CHANNELS),
                "groups": [
                    int(value) for value in plan["frozen_groups"]
                ],
                "depths": [
                    int(value) for value in plan["frozen_depths"]
                ],
            },
            "operator_post_kv": {
                "site": "operator",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": post_depths,
            },
            "target_language_post_kv": {
                "site": "target_language",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": post_depths,
            },
        }
        specs = dict(base_specs)
        prefix: list[int] = []
        for slot_id, slot in enumerate(slots):
            specs[f"source_slot_{slot_id}_kv"] = {
                "site": "source_term",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": list(slot),
            }
            prefix.extend(slot)
            specs[f"source_early_plus_prefix_{slot_id}_kv"] = {
                "site": "source_term",
                "channels": list(CHANNELS),
                "groups": all_groups,
                "depths": sorted(set(early_depths + prefix)),
            }

        raw_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        condition_results = {}
        for name, spec in specs.items():
            key = canonical(spec)
            if key not in raw_cache:
                raw_cache[key] = run_output_condition(
                    model,
                    device,
                    layers,
                    confirmation_valid,
                    cases,
                    spec,
                    head_dim=head_dim,
                    pad_token_id=int(pad_token_id),
                    pair_batch_size=batch_size,
                )
            condition_results[name] = bridge_scan.condition_metrics(
                confirmation_valid,
                cases,
                clean_valid,
                raw_cache[key],
                valid_mask,
            )

        trajectory_targets = evenly_spaced(
            confirmation_valid,
            int(plan["trajectory_pair_limit"]),
        )
        trajectory = run_trajectory(
            model,
            device,
            layers,
            trajectory_targets,
            cases,
            plan,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=batch_size,
        )
        trajectory_output_metrics = {}
        trajectory_mask = np.ones(
            len(trajectory_targets), dtype=bool
        )
        trajectory_clean = trajectory["output_rows"]["clean"]
        for mode in ("early", "post", "all"):
            trajectory_output_metrics[mode] = (
                bridge_scan.condition_metrics(
                    trajectory_targets,
                    cases,
                    trajectory_clean,
                    trajectory["output_rows"][mode],
                    trajectory_mask,
                )
            )
        del trajectory["output_rows"]

        post_result = condition_results["source_post_kv"]
        post_rate = rate(post_result)
        early_rate = rate(condition_results["source_early_kv"])
        all_rate = rate(condition_results["source_all_kv"])
        controls = (
            condition_results["operator_post_kv"],
            condition_results["target_language_post_kv"],
        )
        control_rate = max(rate(value) for value in controls)
        gates = prereg["gates"]
        behavior_gate = (
            len(discovery_valid)
            >= gates["discovery_clean_pair_count_min"]
            and len(confirmation_valid)
            >= gates["confirmation_clean_pair_count_min"]
        )
        bridge_gate = (
            behavior_gate
            and post_result["both_counterfactual_top1_count"]
            >= gates["post_kv_both_counterfactual_count_min"]
            and post_rate
            >= gates["post_kv_both_counterfactual_rate_min"]
            and post_rate - control_rate
            >= gates["source_minus_control_rate_min"]
            and trajectory["pair_count"]
            >= gates["trajectory_pair_count_min"]
        )
        if early_rate >= 0.30 and post_rate >= 0.50 and all_rate <= 0.10:
            phase_class = "early_post_conflict"
        elif (
            early_rate <= 0.10
            and post_rate >= 0.50
            and all_rate >= 0.50
        ):
            phase_class = "late_dominant"
        else:
            phase_class = "mixed_or_unresolved"

        successful_targets = [
            row for row, success in zip(
                confirmation_valid,
                post_result["both_counterfactual_mask"],
            )
            if success
        ]
        raw_rollouts = []
        legacy_rollout = {
            "pair_count": 0,
            "both_match_other_clean_rate": 0.0,
        }
        if successful_targets:
            raw_rollouts, legacy_rollout = bridge_scan.rollout_pairs(
                model,
                tokenizer,
                device,
                layers,
                successful_targets,
                cases,
                {
                    key: value
                    for key, value in specs["source_post_kv"].items()
                    if key != "channels"
                },
                head_dim=head_dim,
                steps=int(prereg["rollout_steps"]),
                pair_limit=int(prereg["rollout_pair_limit"]),
                pair_batch_size=batch_size,
            )
        eos_ids = rollout_tools.eos_token_ids(model, tokenizer)
        audited_rollouts, eos_rollout = rollout_tools.audit_rollouts(
            raw_rollouts, eos_ids
        )
        rollout_gate = (
            bridge_gate
            and eos_rollout["pair_count"]
            >= gates["rollout_pair_count_min"]
            and eos_rollout["both_match_other_clean_rate"]
            >= gates["eos_censored_both_match_rate_min"]
        )

        k_rate = rate(condition_results["source_post_k_only"])
        v_rate = rate(condition_results["source_post_v_only"])
        summary = {
            "schema_version": "phase1057_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "discovery_clean_pair_count": len(discovery_valid),
            "confirmation_clean_pair_count": len(confirmation_valid),
            "behavior_gate_passed": behavior_gate,
            "condition_results": {
                key: compact(value)
                for key, value in condition_results.items()
            },
            "condition_evaluation_count": len(raw_cache),
            "phase_class": phase_class,
            "phase_rates": {
                "early": early_rate,
                "post": post_rate,
                "all": all_rate,
                "post_minus_all": post_rate - all_rate,
                "conflict_summary": early_rate + post_rate - all_rate,
            },
            "channel_rates": {
                "k_only": k_rate,
                "v_only": v_rate,
                "kv": post_rate,
                "kv_minus_best_single": post_rate - max(k_rate, v_rate),
            },
            "maximum_role_control_rate": control_rate,
            "fresh_bridge_gate_passed": bridge_gate,
            "trajectory": trajectory,
            "trajectory_output_metrics": {
                key: compact(value)
                for key, value in trajectory_output_metrics.items()
            },
            "eos_token_ids": eos_ids,
            "legacy_rollout_summary": legacy_rollout,
            "eos_rollout_summary": eos_rollout,
            "rollout_gate_passed": rollout_gate,
            "rollouts": audited_rollouts,
            "elapsed_seconds": float(time.time() - started),
        }
        summary["nonfinite_serialized_value_count"] = nonfinite_count(
            summary
        )
        summary = json_safe(summary)
        out = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_json(out / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "discovery_clean": len(discovery_valid),
            "confirmation_clean": len(confirmation_valid),
            "behavior_gate": behavior_gate,
            "phase_class": phase_class,
            "early": early_rate,
            "post": post_rate,
            "all": all_rate,
            "k_only": k_rate,
            "v_only": v_rate,
            "kv": post_rate,
            "frozen": rate(
                condition_results["source_frozen_rectangle"]
            ),
            "control": control_rate,
            "bridge_gate": bridge_gate,
            "trajectory_pairs": trajectory["pair_count"],
            "rollout": eos_rollout,
            "rollout_gate": rollout_gate,
            "condition_evaluations": len(raw_cache),
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
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
