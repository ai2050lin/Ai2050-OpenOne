#!/usr/bin/env python3
"""Run held-out Q/K/V read-path reset and replay in native FP16."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
import phase1044_natural_recompute_trajectory_scan as trajectory_tools
import phase1045_receiver_mediation_scan as source_tools
import phase1049_qkv_read_path_protocol as protocol


CLEAN_BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
TARGET_BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and torch.is_tensor(output[0])
    ):
        return output[0]
    raise TypeError(f"unsupported projection output {type(output)!r}")


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def projection_width(module: Any) -> int:
    if hasattr(module, "out_features"):
        return int(module.out_features)
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        raise RuntimeError(
            f"cannot infer projection width for {type(module).__name__}"
        )
    return int(weight.shape[0])


def make_paired_batch(
    target_rows: list[dict[str, Any]],
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
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    model_rows = []
    for target in target_rows:
        row = cases[int(target["target_case_index"])]
        model_rows.extend((row, row))
    (
        ids,
        attention_mask,
        _,
        _,
        pre_positions,
        _,
    ) = source_tools.make_clean_batch(
        model_rows,
        pad_token_id=pad_token_id,
        device=device,
    )

    row_count = len(model_rows)
    source_patch_positions = torch.zeros(
        (row_count, protocol.MAX_SOURCE_SPAN), dtype=torch.long
    )
    source_patch_masks = torch.zeros_like(
        source_patch_positions, dtype=torch.bool
    )
    payloads = np.zeros(
        (
            row_count,
            protocol.MAX_SOURCE_SPAN,
            source_cache.shape[-1],
        ),
        dtype=np.float32,
    )
    projection_source_positions = torch.zeros(
        (
            row_count,
            len(protocol.SOURCE_SITES),
            protocol.MAX_SOURCE_SPAN,
        ),
        dtype=torch.long,
    )
    projection_source_masks = torch.zeros_like(
        projection_source_positions, dtype=torch.bool
    )
    q_positions = torch.zeros(
        (row_count, len(protocol.Q_SITES), protocol.MAX_Q_SPAN),
        dtype=torch.long,
    )
    q_masks = torch.zeros_like(q_positions, dtype=torch.bool)
    payload_norms = np.zeros(len(target_rows), dtype=np.float32)

    for target_slot, target in enumerate(target_rows):
        target_case_index = int(target["target_case_index"])
        donor_case_index = int(target["cross_family_case_index"])
        target_row = cases[target_case_index]
        donor_row = cases[donor_case_index]
        even = 2 * target_slot
        odd = even + 1

        selected_role = protocol.semantic_role(
            "selected_concept", target
        )
        target_start, target_end = (
            int(value)
            for value in target_row["anchor_spans"][selected_role]
        )
        donor_start, donor_end = (
            int(value)
            for value in donor_row["anchor_spans"][selected_role]
        )
        target_span = list(range(target_start, target_end + 1))
        donor_span = list(range(donor_start, donor_end + 1))
        if (
            len(target_span) != len(donor_span)
            or len(target_span) > protocol.MAX_SOURCE_SPAN
        ):
            raise RuntimeError("selected source span mismatch")
        role_slot = source_tools.SOURCE_ROLES.index(selected_role)
        target_value = np.asarray(
            source_cache[
                case_to_local[target_case_index],
                role_slot,
                :len(target_span),
                :,
            ],
            dtype=np.float32,
        )
        donor_value = np.asarray(
            source_cache[
                case_to_local[donor_case_index],
                role_slot,
                :len(donor_span),
                :,
            ],
            dtype=np.float32,
        )
        payload = donor_value - target_value
        source_patch_positions[even, :len(target_span)] = torch.tensor(
            target_span, dtype=torch.long
        )
        source_patch_masks[even, :len(target_span)] = True
        payloads[even, :len(target_span), :] = payload
        payload_norms[target_slot] = float(
            np.linalg.norm(payload) / math.sqrt(len(target_span))
        )

        for site_slot, site in enumerate(protocol.SOURCE_SITES):
            role = protocol.semantic_role(site, target)
            start, end = (
                int(value)
                for value in target_row["anchor_spans"][role]
            )
            span = list(range(start, end + 1))
            if len(span) > protocol.MAX_SOURCE_SPAN:
                raise RuntimeError(f"{site} projection span exceeded")
            for row_slot in (even, odd):
                projection_source_positions[
                    row_slot, site_slot, :len(span)
                ] = torch.tensor(span, dtype=torch.long)
                projection_source_masks[
                    row_slot, site_slot, :len(span)
                ] = True

        # Phase1048 measured the endpoint of each destination span.
        for site_slot, site in enumerate(protocol.Q_SITES):
            endpoint = int(target_row["anchor_spans"][site][1])
            for row_slot in (even, odd):
                q_positions[row_slot, site_slot, 0] = endpoint
                q_masks[row_slot, site_slot, 0] = True

    return (
        ids,
        attention_mask,
        pre_positions,
        source_patch_positions,
        source_patch_masks,
        torch.from_numpy(payloads),
        projection_source_positions,
        projection_source_masks,
        q_positions,
        q_masks,
        np.asarray(
            [
                int(row["confirmation_index"])
                for row in target_rows
            ],
            dtype=np.int64,
        ),
        payload_norms,
    )


class ProjectionCache:
    """Cache source-edit and zero trajectories at real Q/K/V outputs."""

    def __init__(
        self,
        layers: list[Any],
        post_depths: list[int],
        q_depths: list[int],
        k_cache: np.memmap,
        v_cache: np.memmap,
        q_cache: np.memmap,
    ) -> None:
        self.layers = layers
        self.post_depths = post_depths
        self.q_depths = q_depths
        self.post_slots = {
            depth: slot for slot, depth in enumerate(post_depths)
        }
        self.q_slots = {
            depth: slot for slot, depth in enumerate(q_depths)
        }
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.q_cache = q_cache
        self.source_positions: torch.Tensor | None = None
        self.source_masks: torch.Tensor | None = None
        self.q_positions: torch.Tensor | None = None
        self.q_masks: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.post_depths:
            attention = self.layers[depth - 1].self_attn
            self.handles.append(
                attention.k_proj.register_forward_hook(
                    self._hook("k", depth)
                )
            )
            self.handles.append(
                attention.v_proj.register_forward_hook(
                    self._hook("v", depth)
                )
            )
        for depth in self.q_depths:
            attention = self.layers[depth - 1].self_attn
            self.handles.append(
                attention.q_proj.register_forward_hook(
                    self._hook("q", depth)
                )
            )

    def begin(
        self,
        source_positions: torch.Tensor,
        source_masks: torch.Tensor,
        q_positions: torch.Tensor,
        q_masks: torch.Tensor,
        target_indices: np.ndarray,
    ) -> None:
        self.source_positions = source_positions
        self.source_masks = source_masks
        self.q_positions = q_positions
        self.q_masks = q_masks
        self.target_indices = target_indices
        self.counts = {}

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if self.target_indices is None:
                raise RuntimeError("projection cache context missing")
            hidden = output_tensor(output)
            if channel == "q":
                positions = self.q_positions
                masks = self.q_masks
                cache = self.q_cache
                depth_slot = self.q_slots[depth]
            else:
                positions = self.source_positions
                masks = self.source_masks
                cache = (
                    self.k_cache if channel == "k" else self.v_cache
                )
                depth_slot = self.post_slots[depth]
            if positions is None or masks is None:
                raise RuntimeError("projection cache positions missing")
            device_positions = positions.to(hidden.device)
            device_masks = masks.to(hidden.device)
            batch = torch.arange(
                hidden.shape[0], device=hidden.device
            )
            batch = batch[:, None, None].expand_as(device_positions)
            values = hidden[batch, device_positions, :].clone()
            values = values.masked_fill(
                ~device_masks[..., None], 0
            )
            target_index = np.repeat(self.target_indices, 2)
            arms = np.tile(
                np.asarray([0, 1], dtype=np.int64),
                len(self.target_indices),
            )
            cache[
                target_index, arms, depth_slot, :, :, :
            ] = values.detach().to(
                "cpu", dtype=torch.float16
            ).numpy()
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return output

        return hook

    def end(self) -> None:
        expected = {
            *(("k", depth) for depth in self.post_depths),
            *(("v", depth) for depth in self.post_depths),
            *(("q", depth) for depth in self.q_depths),
        }
        if set(self.counts) != expected or any(
            count != 1 for count in self.counts.values()
        ):
            raise RuntimeError(
                f"projection cache count drift: {self.counts}"
            )
        self.source_positions = None
        self.source_masks = None
        self.q_positions = None
        self.q_masks = None
        self.target_indices = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class ProjectionSwap:
    """Reset source trajectory and replay zero trajectory projections."""

    def __init__(
        self,
        layers: list[Any],
        spec: dict[str, Any],
        active_depths: list[int],
        post_depths: list[int],
        q_depths: list[int],
        k_cache: np.memmap,
        v_cache: np.memmap,
        q_cache: np.memmap,
    ) -> None:
        self.layers = layers
        self.spec = spec
        self.active_depths = active_depths
        self.post_slots = {
            depth: slot for slot, depth in enumerate(post_depths)
        }
        self.q_slots = {
            depth: slot for slot, depth in enumerate(q_depths)
        }
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.q_cache = q_cache
        self.source_positions: torch.Tensor | None = None
        self.source_masks: torch.Tensor | None = None
        self.q_positions: torch.Tensor | None = None
        self.q_masks: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.active_depths:
            attention = self.layers[depth - 1].self_attn
            for channel in self.spec["channels"]:
                module = getattr(attention, f"{channel}_proj")
                self.handles.append(
                    module.register_forward_hook(
                        self._hook(channel, depth)
                    )
                )

    def begin(
        self,
        source_positions: torch.Tensor,
        source_masks: torch.Tensor,
        q_positions: torch.Tensor,
        q_masks: torch.Tensor,
        target_indices: np.ndarray,
    ) -> None:
        self.source_positions = source_positions
        self.source_masks = source_masks
        self.q_positions = q_positions
        self.q_masks = q_masks
        self.target_indices = target_indices
        self.counts = {}

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if self.target_indices is None:
                raise RuntimeError("projection swap context missing")
            hidden = output_tensor(output)
            if channel == "q":
                positions = self.q_positions
                masks = self.q_masks
                cache = self.q_cache
                depth_slot = self.q_slots[depth]
                site_names = tuple(self.spec["q_sites"])
                all_names = protocol.Q_SITES
            else:
                positions = self.source_positions
                masks = self.source_masks
                cache = (
                    self.k_cache if channel == "k" else self.v_cache
                )
                depth_slot = self.post_slots[depth]
                site_names = (str(self.spec["source_site"]),)
                all_names = protocol.SOURCE_SITES
            if positions is None or masks is None:
                raise RuntimeError("projection swap positions missing")

            patched = hidden.clone()
            target_index = np.repeat(self.target_indices, 2)
            donor_arms = np.tile(
                np.asarray([1, 0], dtype=np.int64),
                len(self.target_indices),
            )
            for site_name in site_names:
                site_slot = all_names.index(site_name)
                values = np.asarray(
                    cache[
                        target_index,
                        donor_arms,
                        depth_slot,
                        site_slot,
                        :,
                        :,
                    ],
                    dtype=np.float16,
                )
                values_tensor = torch.from_numpy(values).to(
                    hidden.device, dtype=hidden.dtype
                )
                site_positions = positions[:, site_slot, :].to(
                    hidden.device
                )
                site_masks = masks[:, site_slot, :].to(hidden.device)
                for span_slot in range(site_positions.shape[1]):
                    active = torch.where(site_masks[:, span_slot])[0]
                    if len(active) == 0:
                        continue
                    patched[
                        active,
                        site_positions[active, span_slot],
                        :,
                    ] = values_tensor[active, span_slot, :]
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return replace_output(output, patched)

        return hook

    def end(self) -> None:
        expected = {
            (channel, depth)
            for depth in self.active_depths
            for channel in self.spec["channels"]
        }
        if set(self.counts) != expected or any(
            count != 1 for count in self.counts.values()
        ):
            raise RuntimeError(
                f"projection swap count drift: {self.counts}"
            )
        self.source_positions = None
        self.source_masks = None
        self.q_positions = None
        self.q_masks = None
        self.target_indices = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def margin_values(
    logits: np.ndarray,
    targets: list[dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    rows = np.arange(len(targets), dtype=np.int64)
    target_index = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_index = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    return (
        values[rows, cross_index] - values[rows, target_index]
    )


def normalized_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    result = np.full(len(numerator), np.nan, dtype=np.float32)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > EPS)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def condition_metrics(
    source_shift: np.ndarray,
    zero_margin: np.ndarray,
    values: np.ndarray,
    targets: list[dict[str, Any]],
    baseline_top1: np.ndarray,
    condition_top1: np.ndarray,
    gate: dict[str, Any],
) -> dict[str, Any]:
    reset_margin = margin_values(values[:, 0, :], targets)
    replay_margin = margin_values(values[:, 1, :], targets)
    reset_shift = reset_margin - zero_margin
    replay_shift = replay_margin - zero_margin
    blocked = source_shift - reset_shift
    mediation = normalized_ratio(blocked, source_shift)
    replay = normalized_ratio(replay_shift, source_shift)
    source_summary = trajectory_tools.scalar_summary(source_shift)
    blocked_summary = trajectory_tools.scalar_summary(blocked)
    mediation_summary = trajectory_tools.scalar_summary(mediation)
    replay_shift_summary = trajectory_tools.scalar_summary(replay_shift)
    replay_summary = trajectory_tools.scalar_summary(replay)
    source_ok = (
        source_summary["median"] is not None
        and source_summary["median"]
        > gate["source_shift_median_min"]
        and source_summary["positive_rate"]
        >= gate["source_positive_rate_min"]
    )
    route_pass = (
        source_ok
        and blocked_summary["positive_rate"]
        >= gate["blocked_positive_rate_min"]
        and mediation_summary["median"]
        >= gate["mediation_fraction_median_min"]
        and replay_shift_summary["positive_rate"]
        >= gate["replay_positive_rate_min"]
        and replay_summary["median"]
        >= gate["replay_recovery_median_min"]
    )
    return {
        "reset_shift": trajectory_tools.scalar_summary(reset_shift),
        "replay_shift": replay_shift_summary,
        "blocked_amount": blocked_summary,
        "mediation_fraction": mediation_summary,
        "replay_recovery": replay_summary,
        "source_gate_passed": bool(source_ok),
        "route_gate_before_specificity": bool(route_pass),
        "full_vocabulary_top1": {
            "reset_changed_from_source_rate": float(np.mean(
                condition_top1[:, 0] != baseline_top1[:, 0]
            )),
            "replay_changed_from_zero_rate": float(np.mean(
                condition_top1[:, 1] != baseline_top1[:, 1]
            )),
            "reset_matches_zero_rate": float(np.mean(
                condition_top1[:, 0] == baseline_top1[:, 1]
            )),
            "replay_matches_source_rate": float(np.mean(
                condition_top1[:, 1] == baseline_top1[:, 0]
            )),
        },
    }


def analyze(
    baseline_logits: np.ndarray,
    condition_logits: np.ndarray,
    baseline_top1: np.ndarray,
    condition_top1: np.ndarray,
    targets: list[dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    source_margin = margin_values(
        baseline_logits[:, 0, :], targets
    )
    zero_margin = margin_values(
        baseline_logits[:, 1, :], targets
    )
    source_shift = source_margin - zero_margin
    gate = prereg["causal_route_gate"]
    rows = {}
    condition_names = list(protocol.CONDITIONS)
    for slot, name in enumerate(condition_names):
        rows[name] = condition_metrics(
            source_shift,
            zero_margin,
            condition_logits[:, slot, :, :],
            targets,
            baseline_top1,
            condition_top1[:, slot, :],
            gate,
        )

    def median(name: str, metric: str) -> float:
        value = rows[name][metric]["median"]
        return float(value) if value is not None else float("nan")

    union_specificity = {
        "selected_minus_unselected_mediation": (
            median("selected_kv_union", "mediation_fraction")
            - median("unselected_kv_union", "mediation_fraction")
        ),
        "selected_minus_unselected_replay": (
            median("selected_kv_union", "replay_recovery")
            - median("unselected_kv_union", "replay_recovery")
        ),
    }
    all_specificity = {
        "selected_minus_unselected_mediation": (
            median(
                "selected_kv_all_postsource", "mediation_fraction"
            )
            - median(
                "unselected_kv_all_postsource",
                "mediation_fraction",
            )
        ),
        "selected_minus_unselected_replay": (
            median(
                "selected_kv_all_postsource", "replay_recovery"
            )
            - median(
                "unselected_kv_all_postsource", "replay_recovery"
            )
        ),
    }
    interactions = {
        "kv_union_minus_k_union_mediation": (
            median("selected_kv_union", "mediation_fraction")
            - median("selected_k_union", "mediation_fraction")
        ),
        "kv_union_minus_v_union_mediation": (
            median("selected_kv_union", "mediation_fraction")
            - median("selected_v_union", "mediation_fraction")
        ),
        "kv_union_minus_k_plus_v_mediation": (
            median("selected_kv_union", "mediation_fraction")
            - median("selected_k_union", "mediation_fraction")
            - median("selected_v_union", "mediation_fraction")
        ),
        "qkv_union_minus_kv_union_mediation": (
            median(
                "selected_kv_query_preoutput_q_union",
                "mediation_fraction",
            )
            - median("selected_kv_union", "mediation_fraction")
        ),
    }
    min_gain = gate["selected_minus_unselected_mediation_min"]
    union_pass = (
        rows["selected_kv_union"][
            "route_gate_before_specificity"
        ]
        and union_specificity[
            "selected_minus_unselected_mediation"
        ]
        >= min_gain
    )
    all_pass = (
        rows["selected_kv_all_postsource"][
            "route_gate_before_specificity"
        ]
        and all_specificity[
            "selected_minus_unselected_mediation"
        ]
        >= min_gain
    )
    return {
        "source_margin": trajectory_tools.scalar_summary(source_margin),
        "zero_margin": trajectory_tools.scalar_summary(zero_margin),
        "source_shift": trajectory_tools.scalar_summary(source_shift),
        "condition_metrics": rows,
        "union_specificity": union_specificity,
        "all_postsource_specificity": all_specificity,
        "channel_interactions": interactions,
        "selected_kv_union_route_passed": bool(union_pass),
        "selected_kv_all_postsource_route_passed": bool(all_pass),
    }


def clean_behavior(
    clean_logits: np.ndarray,
    cases_list: list[dict[str, Any]],
) -> dict[str, Any]:
    values = np.asarray(clean_logits, dtype=np.float32)
    prediction = np.argmax(values, axis=-1)
    expected = np.asarray(
        [int(row["expected_index"]) for row in cases_list],
        dtype=np.int64,
    )
    return {
        "case_count": len(cases_list),
        "candidate_accuracy": float(np.mean(prediction == expected)),
        "finite": trajectory_tools.finite_summary(values),
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1049 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {int(row["case_index"]): row for row in cases_list}
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    model_plan = prereg["model_info"][model_name]
    source_depth = int(model_plan["source_depth"])
    post_depths = [
        int(value) for value in model_plan["all_postsource_depths"]
    ]
    q_depths = [
        int(value) for value in model_plan["frozen_union_depths"]
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
        layers = list(get_layers(model))
        info = get_model_info(model, model_name)
        if len(layers) != int(model_plan["n_layers"]):
            raise RuntimeError("model depth drift")
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )
        q_width = projection_width(layers[0].self_attn.q_proj)
        k_width = projection_width(layers[0].self_attn.k_proj)
        v_width = projection_width(layers[0].self_attn.v_proj)

        source_cache = np.lib.format.open_memmap(
            atlas_dir / "source_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(source_tools.SOURCE_ROLES),
                protocol.MAX_SOURCE_SPAN,
                info.d_model,
            ),
        )
        clean_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(protocol.material.FAMILIES)),
        )
        k_cache = np.lib.format.open_memmap(
            atlas_dir / "baseline_k_source.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(targets),
                2,
                len(post_depths),
                len(protocol.SOURCE_SITES),
                protocol.MAX_SOURCE_SPAN,
                k_width,
            ),
        )
        v_cache = np.lib.format.open_memmap(
            atlas_dir / "baseline_v_source.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(targets),
                2,
                len(post_depths),
                len(protocol.SOURCE_SITES),
                protocol.MAX_SOURCE_SPAN,
                v_width,
            ),
        )
        q_cache = np.lib.format.open_memmap(
            atlas_dir / "baseline_q_destination.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(targets),
                2,
                len(q_depths),
                len(protocol.Q_SITES),
                protocol.MAX_Q_SPAN,
                q_width,
            ),
        )
        baseline_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_baseline_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(targets), 2, len(protocol.material.FAMILIES)),
        )
        baseline_top1 = np.lib.format.open_memmap(
            atlas_dir / "paired_baseline_full_top1.int32.npy",
            mode="w+",
            dtype=np.int32,
            shape=(len(targets), 2),
        )
        condition_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_condition_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.CONDITIONS),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        condition_top1 = np.lib.format.open_memmap(
            atlas_dir / "paired_condition_full_top1.int32.npy",
            mode="w+",
            dtype=np.int32,
            shape=(len(targets), len(protocol.CONDITIONS), 2),
        )
        payload_norms = np.lib.format.open_memmap(
            atlas_dir / "source_payload_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(targets),),
        )
        for array in (
            source_cache,
            clean_logits,
            k_cache,
            v_cache,
            q_cache,
            baseline_logits,
            condition_logits,
            payload_norms,
        ):
            array[:] = np.nan
        baseline_top1[:] = -1
        condition_top1[:] = -1

        source_capture = source_tools.SourceStateCapture(
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
                ) = source_tools.make_clean_batch(
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
                    [
                        case_to_local[int(value)]
                        for value in case_indices
                    ],
                    dtype=np.int64,
                )
                clean_logits[local] = selected.detach().cpu().numpy()
                del output, logits, selected
        finally:
            source_capture.close()
        source_cache.flush()
        clean_logits.flush()

        source_patch = trajectory_tools.SourcePatch(
            layers[source_depth - 1]
        )
        source_patch.register()
        cache_capture = ProjectionCache(
            layers,
            post_depths,
            q_depths,
            k_cache,
            v_cache,
            q_cache,
        )
        cache_capture.register()
        try:
            for target_batch in chunks(
                targets, TARGET_BATCH_SIZE[model_name]
            ):
                (
                    input_ids,
                    attention_mask,
                    pre_positions,
                    source_positions,
                    source_masks,
                    payloads,
                    projection_source_positions,
                    projection_source_masks,
                    q_positions,
                    q_masks,
                    target_indices,
                    batch_payload_norms,
                ) = make_paired_batch(
                    target_batch,
                    cases,
                    case_to_local,
                    source_cache,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                source_patch.begin(
                    source_positions, source_masks, payloads
                )
                cache_capture.begin(
                    projection_source_positions,
                    projection_source_masks,
                    q_positions,
                    q_masks,
                    target_indices,
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                cache_capture.end()
                source_patch.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                boundary = logits[
                    batch, pre_positions.to(logits.device), :
                ].float()
                selected = boundary.index_select(
                    -1, candidate_ids.to(boundary.device)
                )
                baseline_logits[target_indices] = (
                    selected.reshape(
                        len(target_batch),
                        2,
                        len(protocol.material.FAMILIES),
                    ).detach().cpu().numpy()
                )
                baseline_top1[target_indices] = (
                    torch.argmax(boundary, dim=-1)
                    .reshape(len(target_batch), 2)
                    .detach()
                    .cpu()
                    .numpy()
                )
                payload_norms[target_indices] = batch_payload_norms
                del output, logits, boundary, selected
        finally:
            cache_capture.close()
        for array in (
            k_cache,
            v_cache,
            q_cache,
            baseline_logits,
            baseline_top1,
            payload_norms,
        ):
            array.flush()

        try:
            for condition_slot, (condition, spec) in enumerate(
                protocol.CONDITIONS.items()
            ):
                active_depths = protocol.condition_depths(
                    model_name, spec, prereg
                )
                swap = ProjectionSwap(
                    layers,
                    spec,
                    active_depths,
                    post_depths,
                    q_depths,
                    k_cache,
                    v_cache,
                    q_cache,
                )
                swap.register()
                try:
                    for target_batch in chunks(
                        targets, TARGET_BATCH_SIZE[model_name]
                    ):
                        (
                            input_ids,
                            attention_mask,
                            pre_positions,
                            source_positions,
                            source_masks,
                            payloads,
                            projection_source_positions,
                            projection_source_masks,
                            q_positions,
                            q_masks,
                            target_indices,
                            _,
                        ) = make_paired_batch(
                            target_batch,
                            cases,
                            case_to_local,
                            source_cache,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        source_patch.begin(
                            source_positions, source_masks, payloads
                        )
                        swap.begin(
                            projection_source_positions,
                            projection_source_masks,
                            q_positions,
                            q_masks,
                            target_indices,
                        )
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        swap.end()
                        source_patch.end()
                        logits = output.logits
                        batch = torch.arange(
                            logits.shape[0], device=logits.device
                        )
                        boundary = logits[
                            batch,
                            pre_positions.to(logits.device),
                            :,
                        ].float()
                        selected = boundary.index_select(
                            -1, candidate_ids.to(boundary.device)
                        )
                        condition_logits[
                            target_indices, condition_slot, :, :
                        ] = selected.reshape(
                            len(target_batch),
                            2,
                            len(protocol.material.FAMILIES),
                        ).detach().cpu().numpy()
                        condition_top1[
                            target_indices, condition_slot, :
                        ] = torch.argmax(
                            boundary, dim=-1
                        ).reshape(
                            len(target_batch), 2
                        ).detach().cpu().numpy()
                        del output, logits, boundary, selected
                finally:
                    swap.close()
                condition_logits.flush()
                condition_top1.flush()
                print(
                    json.dumps({
                        "model": model_name,
                        "condition": condition,
                        "elapsed_seconds": time.time() - started,
                    }),
                    flush=True,
                )
        finally:
            source_patch.close()

        analysis = analyze(
            baseline_logits,
            condition_logits,
            baseline_top1,
            condition_top1,
            targets,
            prereg,
        )
        summary = {
            "schema_version": "phase1049_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
                "q_projection_width": q_width,
                "k_projection_width": k_width,
                "v_projection_width": v_width,
            },
            "source_depth": source_depth,
            "post_depths": post_depths,
            "frozen_q_depths": q_depths,
            "clean_behavior": clean_behavior(
                clean_logits, cases_list
            ),
            "source_cache_finite": trajectory_tools.finite_summary(
                source_cache
            ),
            "k_cache_finite": trajectory_tools.finite_summary(k_cache),
            "v_cache_finite": trajectory_tools.finite_summary(v_cache),
            "q_cache_finite": trajectory_tools.finite_summary(q_cache),
            "baseline_logits_finite": (
                trajectory_tools.finite_summary(baseline_logits)
            ),
            "condition_logits_finite": (
                trajectory_tools.finite_summary(condition_logits)
            ),
            "payload_norm": trajectory_tools.scalar_summary(
                payload_norms
            ),
            "analysis": analysis,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(
            json.dumps({
                "model": model_name,
                "source_shift": analysis["source_shift"],
                "selected_kv_union": analysis[
                    "condition_metrics"
                ]["selected_kv_union"]["mediation_fraction"],
                "selected_kv_all": analysis[
                    "condition_metrics"
                ]["selected_kv_all_postsource"][
                    "mediation_fraction"
                ],
                "union_specificity": analysis["union_specificity"],
                "union_pass": analysis[
                    "selected_kv_union_route_passed"
                ],
                "all_pass": analysis[
                    "selected_kv_all_postsource_route_passed"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
            }),
            flush=True,
        )
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
