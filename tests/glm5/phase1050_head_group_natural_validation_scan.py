#!/usr/bin/env python3
"""Localize KV groups and validate them on natural lexical holdouts."""

from __future__ import annotations

import argparse
import json
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
import phase1044_natural_recompute_trajectory_scan as common
import phase1045_receiver_mediation_scan as source_tools
import phase1049_qkv_read_path_scan as route_tools
import phase1050_head_group_natural_validation_protocol as protocol


CLEAN_BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
TARGET_BATCH_SIZE = {"qwen3": 8, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return route_tools.output_tensor(output)


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    return route_tools.replace_output(output, hidden)


def projection_width(module: Any) -> int:
    return route_tools.projection_width(module)


class KVCache:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        k_cache: np.memmap,
        v_cache: np.memmap,
    ) -> None:
        self.layers = layers
        self.depths = depths
        self.depth_slots = {
            depth: slot for slot, depth in enumerate(depths)
        }
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.depths:
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

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        target_indices: np.ndarray,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.target_indices = target_indices
        self.counts = {}

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if (
                self.positions is None
                or self.masks is None
                or self.target_indices is None
            ):
                raise RuntimeError("KV cache context missing")
            hidden = output_tensor(output)
            positions = self.positions.to(hidden.device)
            masks = self.masks.to(hidden.device)
            batch = torch.arange(
                hidden.shape[0], device=hidden.device
            )
            batch = batch[:, None, None].expand_as(positions)
            values = hidden[batch, positions, :].clone()
            values = values.masked_fill(~masks[..., None], 0)
            target_index = np.repeat(self.target_indices, 2)
            arms = np.tile(
                np.asarray([0, 1], dtype=np.int64),
                len(self.target_indices),
            )
            cache = self.k_cache if channel == "k" else self.v_cache
            cache[
                target_index,
                arms,
                self.depth_slots[depth],
                :,
                :,
                :,
            ] = values.detach().to(
                "cpu", dtype=torch.float16
            ).numpy()
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return output

        return hook

    def end(self) -> None:
        expected = {
            (channel, depth)
            for depth in self.depths
            for channel in ("k", "v")
        }
        if set(self.counts) != expected or any(
            value != 1 for value in self.counts.values()
        ):
            raise RuntimeError(f"KV cache count drift: {self.counts}")
        self.positions = None
        self.masks = None
        self.target_indices = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class CachedHeadGroupSwap:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        groups: list[int],
        source_site: str,
        head_dim: int,
        k_cache: np.memmap,
        v_cache: np.memmap,
    ) -> None:
        self.layers = layers
        self.depths = depths
        self.depth_slots = {
            depth: slot for slot, depth in enumerate(depths)
        }
        self.groups = groups
        self.source_site = source_site
        self.head_dim = head_dim
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.depths:
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

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        target_indices: np.ndarray,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.target_indices = target_indices
        self.counts = {}

    def _hook(self, channel: str, depth: int):
        def hook(_module, _inputs, output):
            if (
                self.positions is None
                or self.masks is None
                or self.target_indices is None
            ):
                raise RuntimeError("cached head swap context missing")
            hidden = output_tensor(output)
            patched = hidden.clone()
            site_slot = protocol.SOURCE_SITES.index(self.source_site)
            positions = self.positions[:, site_slot, :].to(
                hidden.device
            )
            masks = self.masks[:, site_slot, :].to(hidden.device)
            target_index = np.repeat(self.target_indices, 2)
            donor_arms = np.tile(
                np.asarray([1, 0], dtype=np.int64),
                len(self.target_indices),
            )
            cache = self.k_cache if channel == "k" else self.v_cache
            values = np.asarray(
                cache[
                    target_index,
                    donor_arms,
                    self.depth_slots[depth],
                    site_slot,
                    :,
                    :,
                ],
                dtype=np.float16,
            )
            values_tensor = torch.from_numpy(values).to(
                hidden.device, dtype=hidden.dtype
            )
            for group in self.groups:
                start = group * self.head_dim
                end = start + self.head_dim
                for span_slot in range(positions.shape[1]):
                    active = torch.where(masks[:, span_slot])[0]
                    if len(active) == 0:
                        continue
                    patched[
                        active,
                        positions[active, span_slot],
                        start:end,
                    ] = values_tensor[
                        active, span_slot, start:end
                    ]
            key = (channel, depth)
            self.counts[key] = self.counts.get(key, 0) + 1
            return replace_output(output, patched)

        return hook

    def end(self) -> None:
        expected = {
            (channel, depth)
            for depth in self.depths
            for channel in ("k", "v")
        }
        if set(self.counts) != expected or any(
            value != 1 for value in self.counts.values()
        ):
            raise RuntimeError(
                f"cached head swap count drift: {self.counts}"
            )
        self.positions = None
        self.masks = None
        self.target_indices = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class OnlineHeadGroupSwap:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        groups: list[int],
        source_site: str,
        head_dim: int,
    ) -> None:
        self.layers = layers
        self.depths = depths
        self.groups = groups
        self.source_site = source_site
        self.head_dim = head_dim
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.counts: dict[tuple[str, int], int] = {}
        self.handles = []

    def register(self) -> None:
        for depth in self.depths:
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
            # The same registered hook is intentionally present during the
            # clean rollout. An inactive context is a strict no-op control.
            if self.positions is None or self.masks is None:
                return output
            hidden = output_tensor(output)
            if hidden.shape[0] % 2:
                raise RuntimeError("online pair batch drift")
            patched = hidden.clone()
            site_slot = protocol.SOURCE_SITES.index(self.source_site)
            positions = self.positions[:, site_slot, :].to(
                hidden.device
            )
            masks = self.masks[:, site_slot, :].to(hidden.device)
            even = torch.arange(
                0, hidden.shape[0], 2, device=hidden.device
            )
            odd = even + 1
            for span_slot in range(positions.shape[1]):
                valid = (
                    masks[even, span_slot]
                    & masks[odd, span_slot]
                )
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
            return replace_output(output, patched)

        return hook

    def end(self) -> None:
        expected = {
            (channel, depth)
            for depth in self.depths
            for channel in ("k", "v")
        }
        if set(self.counts) != expected or any(
            value != 1 for value in self.counts.values()
        ):
            raise RuntimeError(
                f"online head swap count drift: {self.counts}"
            )
        self.positions = None
        self.masks = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def make_natural_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    model_rows = []
    for target in target_rows:
        model_rows.extend((
            cases[int(target["target_case_index"])],
            cases[int(target["cross_family_case_index"])],
        ))
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
    positions = torch.zeros(
        (
            len(model_rows),
            len(protocol.SOURCE_SITES),
            protocol.MAX_SOURCE_SPAN,
        ),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    for target_slot, target in enumerate(target_rows):
        for pair_arm, case_key in enumerate(
            ("target_case_index", "cross_family_case_index")
        ):
            row_slot = 2 * target_slot + pair_arm
            row = cases[int(target[case_key])]
            for site_slot, site in enumerate(protocol.SOURCE_SITES):
                role = protocol.semantic_role(site, target)
                start, end = (
                    int(value)
                    for value in row["anchor_spans"][role]
                )
                span = list(range(start, end + 1))
                positions[
                    row_slot, site_slot, :len(span)
                ] = torch.tensor(span, dtype=torch.long)
                masks[row_slot, site_slot, :len(span)] = True
    return ids, attention_mask, pre_positions, positions, masks


def margins(
    values: np.ndarray,
    targets: list[dict[str, Any]],
) -> np.ndarray:
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
        np.asarray(values, dtype=np.float32)[rows, cross_index]
        - np.asarray(values, dtype=np.float32)[rows, target_index]
    )


def ratio(
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


def causal_metrics(
    baseline: np.ndarray,
    intervention: np.ndarray,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    source_margin = margins(baseline[:, 0, :], targets)
    zero_margin = margins(baseline[:, 1, :], targets)
    source_shift = source_margin - zero_margin
    reset_shift = margins(
        intervention[:, 0, :], targets
    ) - zero_margin
    replay_shift = margins(
        intervention[:, 1, :], targets
    ) - zero_margin
    blocked = source_shift - reset_shift
    return {
        "source_shift": common.scalar_summary(source_shift),
        "reset_shift": common.scalar_summary(reset_shift),
        "replay_shift": common.scalar_summary(replay_shift),
        "blocked_amount": common.scalar_summary(blocked),
        "mediation_fraction": common.scalar_summary(
            ratio(blocked, source_shift)
        ),
        "replay_recovery": common.scalar_summary(
            ratio(replay_shift, source_shift)
        ),
    }


def discovery_selection(
    baseline: np.ndarray,
    discovery_values: np.ndarray,
    discovery_targets: list[dict[str, Any]],
    n_kv_heads: int,
    maximum: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows = []
    for group in range(n_kv_heads):
        metrics = causal_metrics(
            baseline, discovery_values[:, group, :, :],
            discovery_targets,
        )
        mediation = metrics["mediation_fraction"]["median"]
        replay = metrics["replay_recovery"]["median"]
        score = min(
            float(mediation) if mediation is not None else -1e9,
            float(replay) if replay is not None else -1e9,
        )
        rows.append({
            "kv_group": group,
            "rank_score": score,
            "metrics": metrics,
        })
    rows.sort(
        key=lambda row: (-float(row["rank_score"]), row["kv_group"])
    )
    frozen = [
        int(row["kv_group"]) for row in rows[:maximum]
    ]
    return rows, frozen


def natural_metrics(
    baseline: np.ndarray,
    patched: np.ndarray,
    baseline_top1: np.ndarray,
    patched_top1: np.ndarray,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    target_family = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_family = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    rows = np.arange(len(targets), dtype=np.int64)
    finite_pair = (
        np.all(np.isfinite(baseline), axis=(1, 2))
        & np.all(np.isfinite(patched), axis=(1, 2))
        & np.all(baseline_top1 >= 0, axis=1)
        & np.all(patched_top1 >= 0, axis=1)
    )
    baseline_target_margin = (
        baseline[rows, 0, cross_family]
        - baseline[rows, 0, target_family]
    )
    patched_target_margin = (
        patched[rows, 0, cross_family]
        - patched[rows, 0, target_family]
    )
    baseline_donor_margin = (
        baseline[rows, 1, target_family]
        - baseline[rows, 1, cross_family]
    )
    patched_donor_margin = (
        patched[rows, 1, target_family]
        - patched[rows, 1, cross_family]
    )
    target_shift = patched_target_margin - baseline_target_margin
    donor_shift = patched_donor_margin - baseline_donor_margin
    directional = np.concatenate((target_shift, donor_shift))
    candidate_prediction = np.full(
        (len(targets), 2), -1, dtype=np.int64
    )
    if np.any(finite_pair):
        candidate_prediction[finite_pair] = np.argmax(
            baseline[finite_pair], axis=-1
        )
    patched_prediction = np.argmax(patched, axis=-1)
    behavior_mask = (
        finite_pair
        & (candidate_prediction[:, 0] == target_family)
        & (candidate_prediction[:, 1] == cross_family)
    )
    gated = np.concatenate((
        target_shift[behavior_mask],
        donor_shift[behavior_mask],
    ))
    return {
        "finite_clean_pair_count": int(np.sum(finite_pair)),
        "invalid_clean_pair_count": int(np.sum(~finite_pair)),
        "clean_correct_pair_count": int(np.sum(behavior_mask)),
        "all_directional_shift": common.scalar_summary(directional),
        "behavior_gated_directional_shift": common.scalar_summary(
            gated
        ),
        "candidate_counterfactual_flip_rate": {
            "target_to_cross": float(np.mean(
                patched_prediction[finite_pair, 0]
                == cross_family[finite_pair]
            )),
            "donor_to_target": float(np.mean(
                patched_prediction[finite_pair, 1]
                == target_family[finite_pair]
            )),
            "both": float(np.mean(
                (
                    patched_prediction[finite_pair, 0]
                    == cross_family[finite_pair]
                )
                & (
                    patched_prediction[finite_pair, 1]
                    == target_family[finite_pair]
                )
            )),
            "both_behavior_gated": (
                float(np.mean(
                    (
                        patched_prediction[
                            behavior_mask, 0
                        ] == cross_family[behavior_mask]
                    )
                    & (
                        patched_prediction[
                            behavior_mask, 1
                        ] == target_family[behavior_mask]
                    )
                ))
                if np.any(behavior_mask)
                else None
            ),
        },
        "full_vocabulary_top1": {
            "target_matches_clean_donor_rate": float(np.mean(
                patched_top1[finite_pair, 0]
                == baseline_top1[finite_pair, 1]
            )),
            "donor_matches_clean_target_rate": float(np.mean(
                patched_top1[finite_pair, 1]
                == baseline_top1[finite_pair, 0]
            )),
            "either_top1_changed_rate": float(np.mean(
                np.any(
                    patched_top1[finite_pair]
                    != baseline_top1[finite_pair],
                    axis=1,
                )
            )),
        },
        "behavior_mask": behavior_mask.tolist(),
    }


def gate_causal(
    selected: dict[str, Any],
    unselected: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[bool, float]:
    specificity = (
        float(selected["mediation_fraction"]["median"])
        - float(unselected["mediation_fraction"]["median"])
    )
    passed = (
        selected["source_shift"]["positive_rate"]
        >= gates["source_positive_rate_min"]
        and selected["blocked_amount"]["positive_rate"]
        >= gates["causal_blocked_positive_rate_min"]
        and selected["mediation_fraction"]["median"]
        >= gates["causal_mediation_median_min"]
        and selected["replay_shift"]["positive_rate"]
        >= gates["causal_replay_positive_rate_min"]
        and selected["replay_recovery"]["median"]
        >= gates["causal_replay_median_min"]
        and specificity
        >= gates["selected_minus_unselected_mediation_min"]
    )
    return bool(passed), float(specificity)


def gate_natural(
    metrics: dict[str, Any],
    gates: dict[str, Any],
) -> bool:
    gated = metrics["behavior_gated_directional_shift"]
    return bool(
        metrics["clean_correct_pair_count"]
        >= gates["natural_behavior_gated_pair_count_min"]
        and gated["median"]
        > gates["natural_directional_shift_median_min"]
        and gated["positive_rate"]
        >= gates[
            "natural_directional_shift_positive_rate_min"
        ]
    )


def rollout_pair(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    positions: torch.Tensor,
    masks: torch.Tensor,
    *,
    swap: OnlineHeadGroupSwap | None,
    steps: int,
) -> dict[str, Any]:
    lengths = {len(row["input_ids"]) for row in rows}
    if len(lengths) != 1:
        raise RuntimeError("rollout pair length mismatch")
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    generated: list[list[int]] = [[], []]
    for _ in range(steps):
        if swap is not None:
            swap.begin(positions, masks)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if swap is not None:
            swap.end()
        next_token = torch.argmax(
            output.logits[:, -1, :].float(), dim=-1
        )
        for row_slot, value in enumerate(
            next_token.detach().cpu().tolist()
        ):
            generated[row_slot].append(int(value))
        input_ids = torch.cat(
            (input_ids, next_token[:, None]), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (2, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
        del output
    return {
        "token_ids": generated,
        "text": [
            tokenizer.decode(values, skip_special_tokens=False)
            for values in generated
        ],
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1050 protocol audit failed")
    all_targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    discovery_targets = [
        row for row in all_targets
        if row["partition"] == "discovery"
    ]
    confirmation_targets = [
        row for row in all_targets
        if row["partition"] == "confirmation"
    ]
    discovery_global = np.asarray(
        [int(row["target_index"]) for row in discovery_targets],
        dtype=np.int64,
    )
    confirmation_global = np.asarray(
        [int(row["target_index"]) for row in confirmation_targets],
        dtype=np.int64,
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
    plan = prereg["model_info"][model_name]
    source_depth = int(plan["source_depth"])
    depths = [int(value) for value in plan["frozen_union_depths"]]
    n_kv_heads = int(plan["n_kv_heads"])
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
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )
        k_width = projection_width(layers[0].self_attn.k_proj)
        v_width = projection_width(layers[0].self_attn.v_proj)
        if k_width != v_width or k_width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = k_width // n_kv_heads

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
        clean_top1 = np.lib.format.open_memmap(
            atlas_dir / "clean_full_top1.int32.npy",
            mode="w+",
            dtype=np.int32,
            shape=(len(cases_list),),
        )
        k_cache = np.lib.format.open_memmap(
            atlas_dir / "baseline_k.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(all_targets),
                2,
                len(depths),
                len(protocol.SOURCE_SITES),
                protocol.MAX_SOURCE_SPAN,
                k_width,
            ),
        )
        v_cache = np.lib.format.open_memmap(
            atlas_dir / "baseline_v.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(all_targets),
                2,
                len(depths),
                len(protocol.SOURCE_SITES),
                protocol.MAX_SOURCE_SPAN,
                v_width,
            ),
        )
        baseline_logits = np.lib.format.open_memmap(
            atlas_dir / "source_baseline_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(all_targets),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        discovery_logits = np.lib.format.open_memmap(
            atlas_dir / "discovery_group_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(discovery_targets),
                n_kv_heads,
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        confirmation_names = list(
            prereg["confirmation_conditions"]
        )
        confirmation_logits = np.lib.format.open_memmap(
            atlas_dir / "confirmation_group_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(confirmation_targets),
                len(confirmation_names),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        natural_names = list(prereg["natural_conditions"])
        natural_logits = np.lib.format.open_memmap(
            atlas_dir / "natural_swap_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(confirmation_targets),
                len(natural_names),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        natural_top1 = np.lib.format.open_memmap(
            atlas_dir / "natural_swap_full_top1.int32.npy",
            mode="w+",
            dtype=np.int32,
            shape=(
                len(confirmation_targets),
                len(natural_names),
                2,
            ),
        )
        for array in (
            source_cache,
            clean_logits,
            k_cache,
            v_cache,
            baseline_logits,
            discovery_logits,
            confirmation_logits,
            natural_logits,
        ):
            array[:] = np.nan
        clean_top1[:] = -1
        natural_top1[:] = -1

        clean_nonfinite_case_indices: list[int] = []
        capture = source_tools.SourceStateCapture(
            layers[source_depth - 1], source_cache, case_to_local
        )
        capture.register()
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
                capture.begin(positions, masks, case_indices)
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
                boundary = logits[
                    batch, pre_positions.to(logits.device), :
                ].float()
                selected = boundary.index_select(
                    -1, candidate_ids.to(boundary.device)
                )
                local = np.asarray(
                    [
                        case_to_local[int(value)]
                        for value in case_indices
                    ],
                    dtype=np.int64,
                )
                clean_logits[local] = selected.detach().cpu().numpy()
                clean_top1[local] = torch.argmax(
                    boundary, dim=-1
                ).detach().cpu().numpy()
                del output, logits, boundary, selected
            bad_clean_rows = np.where(
                ~np.all(np.isfinite(clean_logits), axis=1)
            )[0]
            for local_index in bad_clean_rows:
                row_batch = [cases_list[int(local_index)]]
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
                capture.begin(positions, masks, case_indices)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                capture.end()
                boundary = output.logits[
                    0, int(pre_positions[0]), :
                ].float()
                selected = boundary.index_select(
                    -1, candidate_ids.to(boundary.device)
                )
                if not torch.all(torch.isfinite(selected)):
                    clean_logits[int(local_index)] = np.nan
                    clean_top1[int(local_index)] = -1
                    clean_nonfinite_case_indices.append(
                        int(
                            cases_list[int(local_index)]["case_index"]
                        )
                    )
                    del output, boundary, selected
                    continue
                clean_logits[int(local_index)] = (
                    selected.detach().cpu().numpy()
                )
                clean_top1[int(local_index)] = int(
                    torch.argmax(boundary).item()
                )
                del output, boundary, selected
        finally:
            capture.close()
        source_cache.flush()
        clean_logits.flush()
        clean_top1.flush()

        source_patch = common.SourcePatch(layers[source_depth - 1])
        source_patch.register()
        kv_capture = KVCache(
            layers, depths, k_cache, v_cache
        )
        kv_capture.register()
        try:
            for target_batch in chunks(
                all_targets, TARGET_BATCH_SIZE[model_name]
            ):
                batch_values = route_tools.make_paired_batch(
                    target_batch,
                    cases,
                    case_to_local,
                    source_cache,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                (
                    input_ids,
                    attention_mask,
                    pre_positions,
                    source_positions,
                    source_masks,
                    payloads,
                    projection_positions,
                    projection_masks,
                    _,
                    _,
                    target_indices,
                    _,
                ) = batch_values
                source_patch.begin(
                    source_positions, source_masks, payloads
                )
                kv_capture.begin(
                    projection_positions,
                    projection_masks,
                    target_indices,
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                kv_capture.end()
                source_patch.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                baseline_logits[target_indices] = (
                    selected.reshape(
                        len(target_batch),
                        2,
                        len(protocol.material.FAMILIES),
                    ).detach().cpu().numpy()
                )
                del output, logits, selected
        finally:
            kv_capture.close()
        for array in (k_cache, v_cache, baseline_logits):
            array.flush()

        try:
            for group in range(n_kv_heads):
                swap = CachedHeadGroupSwap(
                    layers,
                    depths,
                    [group],
                    "selected_concept",
                    head_dim,
                    k_cache,
                    v_cache,
                )
                swap.register()
                try:
                    for target_batch in chunks(
                        discovery_targets,
                        TARGET_BATCH_SIZE[model_name],
                    ):
                        batch_values = route_tools.make_paired_batch(
                            target_batch,
                            cases,
                            case_to_local,
                            source_cache,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        (
                            input_ids,
                            attention_mask,
                            pre_positions,
                            source_positions,
                            source_masks,
                            payloads,
                            projection_positions,
                            projection_masks,
                            _,
                            _,
                            target_indices,
                            _,
                        ) = batch_values
                        local = np.asarray(
                            [
                                discovery_global.tolist().index(
                                    int(value)
                                )
                                for value in target_indices
                            ],
                            dtype=np.int64,
                        )
                        source_patch.begin(
                            source_positions,
                            source_masks,
                            payloads,
                        )
                        swap.begin(
                            projection_positions,
                            projection_masks,
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
                        selected = logits[
                            batch,
                            pre_positions.to(logits.device),
                            :,
                        ].float().index_select(
                            -1, candidate_ids.to(logits.device)
                        )
                        discovery_logits[
                            local, group, :, :
                        ] = selected.reshape(
                            len(target_batch),
                            2,
                            len(protocol.material.FAMILIES),
                        ).detach().cpu().numpy()
                        del output, logits, selected
                finally:
                    swap.close()
                discovery_logits.flush()
        finally:
            source_patch.close()

        discovery_rows, frozen_groups = discovery_selection(
            np.asarray(
                baseline_logits[discovery_global], dtype=np.float32
            ),
            discovery_logits,
            discovery_targets,
            n_kv_heads,
            int(plan["maximum_frozen_groups"]),
        )
        top1 = frozen_groups[:1]
        top2 = frozen_groups
        complement = [
            group for group in range(n_kv_heads)
            if group not in top1
        ]
        condition_specs = {
            "selected_top1": (top1, "selected_concept"),
            "unselected_top1": (top1, "unselected_concept"),
            "selected_top2": (top2, "selected_concept"),
            "unselected_top2": (top2, "unselected_concept"),
            "selected_complement_top1": (
                complement,
                "selected_concept",
            ),
        }

        source_patch = common.SourcePatch(layers[source_depth - 1])
        source_patch.register()
        try:
            for condition_slot, condition in enumerate(
                confirmation_names
            ):
                groups, site = condition_specs[condition]
                swap = CachedHeadGroupSwap(
                    layers,
                    depths,
                    groups,
                    site,
                    head_dim,
                    k_cache,
                    v_cache,
                )
                swap.register()
                try:
                    for target_batch in chunks(
                        confirmation_targets,
                        TARGET_BATCH_SIZE[model_name],
                    ):
                        batch_values = route_tools.make_paired_batch(
                            target_batch,
                            cases,
                            case_to_local,
                            source_cache,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        (
                            input_ids,
                            attention_mask,
                            pre_positions,
                            source_positions,
                            source_masks,
                            payloads,
                            projection_positions,
                            projection_masks,
                            _,
                            _,
                            target_indices,
                            _,
                        ) = batch_values
                        local = np.asarray(
                            [
                                confirmation_global.tolist().index(
                                    int(value)
                                )
                                for value in target_indices
                            ],
                            dtype=np.int64,
                        )
                        source_patch.begin(
                            source_positions,
                            source_masks,
                            payloads,
                        )
                        swap.begin(
                            projection_positions,
                            projection_masks,
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
                        selected = logits[
                            batch,
                            pre_positions.to(logits.device),
                            :,
                        ].float().index_select(
                            -1, candidate_ids.to(logits.device)
                        )
                        confirmation_logits[
                            local, condition_slot, :, :
                        ] = selected.reshape(
                            len(target_batch),
                            2,
                            len(protocol.material.FAMILIES),
                        ).detach().cpu().numpy()
                        del output, logits, selected
                finally:
                    swap.close()
                confirmation_logits.flush()
        finally:
            source_patch.close()

        natural_baseline = np.empty(
            (
                len(confirmation_targets),
                2,
                len(protocol.material.FAMILIES),
            ),
            dtype=np.float32,
        )
        natural_baseline_top1 = np.empty(
            (len(confirmation_targets), 2), dtype=np.int32
        )
        for local, target in enumerate(confirmation_targets):
            for arm, key in enumerate(
                ("target_case_index", "cross_family_case_index")
            ):
                case_local = case_to_local[int(target[key])]
                natural_baseline[local, arm] = clean_logits[case_local]
                natural_baseline_top1[local, arm] = clean_top1[
                    case_local
                ]

        natural_specs = {
            "selected_top1": (top1, "selected_concept"),
            "unselected_top1": (top1, "unselected_concept"),
            "selected_top2": (top2, "selected_concept"),
        }
        for condition_slot, condition in enumerate(natural_names):
            groups, site = natural_specs[condition]
            swap = OnlineHeadGroupSwap(
                layers, depths, groups, site, head_dim
            )
            swap.register()
            try:
                for start in range(
                    0,
                    len(confirmation_targets),
                    TARGET_BATCH_SIZE[model_name],
                ):
                    target_batch = confirmation_targets[
                        start:start + TARGET_BATCH_SIZE[model_name]
                    ]
                    (
                        input_ids,
                        attention_mask,
                        pre_positions,
                        positions,
                        masks,
                    ) = make_natural_batch(
                        target_batch,
                        cases,
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
                    natural_logits[
                        start:start + len(target_batch),
                        condition_slot,
                        :,
                        :,
                    ] = selected.reshape(
                        len(target_batch),
                        2,
                        len(protocol.material.FAMILIES),
                    ).detach().cpu().numpy()
                    natural_top1[
                        start:start + len(target_batch),
                        condition_slot,
                        :,
                    ] = torch.argmax(
                        boundary, dim=-1
                    ).reshape(
                        len(target_batch), 2
                    ).detach().cpu().numpy()
                    del output, logits, boundary, selected
            finally:
                swap.close()
            natural_logits.flush()
            natural_top1.flush()

        confirmation_analysis = {}
        confirmation_baseline = np.asarray(
            baseline_logits[confirmation_global], dtype=np.float32
        )
        for slot, condition in enumerate(confirmation_names):
            confirmation_analysis[condition] = causal_metrics(
                confirmation_baseline,
                confirmation_logits[:, slot, :, :],
                confirmation_targets,
            )
        causal_pass, specificity = gate_causal(
            confirmation_analysis["selected_top2"],
            confirmation_analysis["unselected_top2"],
            prereg["gates"],
        )

        natural_analysis = {}
        for slot, condition in enumerate(natural_names):
            natural_analysis[condition] = natural_metrics(
                natural_baseline,
                natural_logits[:, slot, :, :],
                natural_baseline_top1,
                natural_top1[:, slot, :],
                confirmation_targets,
            )
        natural_pass = gate_natural(
            natural_analysis["selected_top2"], prereg["gates"]
        )

        behavior_mask = np.asarray(
            natural_analysis["selected_top2"]["behavior_mask"],
            dtype=bool,
        )
        rollout_indices = np.where(behavior_mask)[0][
            :int(prereg["rollout_pair_limit"])
        ]
        rollouts = []
        rollout_swap = OnlineHeadGroupSwap(
            layers,
            depths,
            top2,
            "selected_concept",
            head_dim,
        )
        rollout_swap.register()
        try:
            for local in rollout_indices:
                target = confirmation_targets[int(local)]
                rows = [
                    cases[int(target["target_case_index"])],
                    cases[int(target["cross_family_case_index"])],
                ]
                (
                    _,
                    _,
                    _,
                    positions,
                    masks,
                ) = make_natural_batch(
                    [target],
                    cases,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                clean_rollout = rollout_pair(
                    model,
                    tokenizer,
                    device,
                    rows,
                    positions,
                    masks,
                    swap=None,
                    steps=int(prereg["rollout_steps"]),
                )
                patched_rollout = rollout_pair(
                    model,
                    tokenizer,
                    device,
                    rows,
                    positions,
                    masks,
                    swap=rollout_swap,
                    steps=int(prereg["rollout_steps"]),
                )
                rollouts.append({
                    "target_index": int(target["target_index"]),
                    "target_family": target["target_family"],
                    "cross_family": target["cross_family"],
                    "clean": clean_rollout,
                    "patched": patched_rollout,
                })
        finally:
            rollout_swap.close()

        summary = {
            "schema_version": "phase1050_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
            },
            "source_depth": source_depth,
            "frozen_depths": depths,
            "clean_finite": common.finite_summary(clean_logits),
            "clean_nonfinite_case_indices": (
                clean_nonfinite_case_indices
            ),
            "source_baseline_finite": common.finite_summary(
                baseline_logits
            ),
            "discovery_finite": common.finite_summary(
                discovery_logits
            ),
            "confirmation_finite": common.finite_summary(
                confirmation_logits
            ),
            "natural_finite": common.finite_summary(natural_logits),
            "discovery_ranking": discovery_rows,
            "frozen_kv_groups": frozen_groups,
            "confirmation_analysis": confirmation_analysis,
            "selected_top2_minus_unselected_top2_mediation": (
                specificity
            ),
            "causal_head_group_gate_passed": causal_pass,
            "natural_analysis": natural_analysis,
            "natural_head_group_gate_passed": natural_pass,
            "rollout_pair_count": len(rollouts),
            "rollouts": rollouts,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(
            json.dumps({
                "model": model_name,
                "frozen_groups": frozen_groups,
                "discovery_top": discovery_rows[0],
                "confirmation_selected_top2": (
                    confirmation_analysis["selected_top2"]
                ),
                "specificity": specificity,
                "causal_pass": causal_pass,
                "natural_selected_top2": natural_analysis[
                    "selected_top2"
                ],
                "natural_pass": natural_pass,
                "rollout_pairs": len(rollouts),
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
