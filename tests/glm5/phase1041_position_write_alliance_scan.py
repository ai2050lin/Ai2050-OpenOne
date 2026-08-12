#!/usr/bin/env python3
"""Run the Phase1041 multi-position current-write alliance discovery."""

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
import phase1037_family_source_causal_scan as metric_tools
import phase1041_position_write_alliance_protocol as protocol


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
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


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    return metric_tools.scalar_summary(np.asarray(values, dtype=np.float32))


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    current = np.asarray(values)
    finite = np.isfinite(current)
    return {
        "all_finite": bool(np.all(finite)),
        "finite_value_rate": float(np.mean(finite)),
        "nonfinite_value_count": int(np.sum(~finite)),
    }


def role_slot(role: str) -> int:
    return protocol.ROLE_ORDER.index(role)


def semantic_role(site: str, target: dict[str, Any]) -> str:
    selected = str(target["selected_slot"])
    unselected = str(target["unselected_slot"])
    mapping = {
        "selected_concept": f"concept_{selected}",
        "selected_nonce": f"definition_nonce_{selected}",
        "unselected_concept": f"concept_{unselected}",
        "unselected_nonce": f"definition_nonce_{unselected}",
        "query_nonce": "query_nonce",
        "pre_output": "pre_output",
    }
    return mapping[site]


def make_clean_batch(
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
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.zeros(
        (
            len(rows),
            len(protocol.ROLE_ORDER),
            protocol.MAX_ROLE_SPAN,
        ),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    pre_positions = torch.empty(len(rows), dtype=torch.long)
    for row_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        for current_role in protocol.ROLE_ORDER:
            start, end = (
                int(value)
                for value in row["anchor_spans"][current_role]
            )
            span = list(range(start, end + 1))
            positions[
                row_index,
                role_slot(current_role),
                :len(span),
            ] = torch.tensor(span, dtype=torch.long)
            masks[
                row_index,
                role_slot(current_role),
                :len(span),
            ] = True
        pre_positions[row_index] = int(
            row["anchor_spans"]["pre_output"][1]
        )
    return (
        ids.to(device),
        attention_mask.to(device),
        positions,
        masks,
        pre_positions,
        np.asarray(
            [int(row["case_index"]) for row in rows], dtype=np.int64
        ),
    )


class ChannelCapture:
    def __init__(
        self,
        layer: Any,
        cache: np.memmap,
        closure: np.memmap,
        case_to_local: dict[int, int],
    ):
        self.layer = layer
        self.cache = cache
        self.closure = closure
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
            raise RuntimeError("capture positions missing")
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None, None].expand_as(positions)
        values = hidden[batch, positions, :].clone()
        return values.masked_fill(~masks[..., None], 0).detach()

    def _pre_hook(self, module, args):
        self.current["upstream_residual"] = self._states(args[0])
        self.counts["pre"] += 1

    def _component_hook(self, name: str):
        def hook(module, args, output):
            self.current[name] = self._states(output_tensor(output))
            self.counts[name] += 1
            return output
        return hook

    def _layer_hook(self, module, args, output):
        if self.local_indices is None or self.masks is None:
            raise RuntimeError("capture local indices missing")
        self.current["layer_output"] = self._states(output_tensor(output))
        for channel_index, channel in enumerate(
            protocol.CACHE_CHANNELS
        ):
            self.cache[
                self.local_indices, channel_index, :, :, :
            ] = self.current[channel].to(
                "cpu", dtype=torch.float16
            ).numpy()
        accounted = (
            self.current["upstream_residual"]
            + self.current["attention_write"]
            + self.current["mlp_write"]
        )
        error = torch.linalg.vector_norm(
            (self.current["layer_output"] - accounted).float(),
            dim=-1,
        )
        transition = torch.linalg.vector_norm(
            (
                self.current["layer_output"]
                - self.current["upstream_residual"]
            ).float(),
            dim=-1,
        )
        relative = error / torch.clamp(transition, min=EPS)
        valid = self.masks.to(relative.device)
        relative = relative.masked_fill(~valid, torch.nan)
        self.closure[self.local_indices, :] = (
            torch.nanmean(relative, dim=-1).cpu().numpy()
        )
        self.counts["layer"] += 1
        return output

    def register(self) -> None:
        self.handles.append(
            self.layer.register_forward_pre_hook(self._pre_hook)
        )
        self.handles.append(
            self.layer.self_attn.register_forward_hook(
                self._component_hook("attention_write")
            )
        )
        self.handles.append(
            self.layer.mlp.register_forward_hook(
                self._component_hook("mlp_write")
            )
        )
        self.handles.append(
            self.layer.register_forward_hook(self._layer_hook)
        )

    def end(self) -> None:
        expected = {
            "pre": 1,
            "attention_write": 1,
            "mlp_write": 1,
            "layer": 1,
        }
        if dict(self.counts) != expected:
            raise RuntimeError(
                f"capture hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.local_indices = None
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def donor_spec(
    target: dict[str, Any],
    condition: str,
    targets: dict[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    if condition == "cross_matched":
        return int(target["cross_family_case_index"]), target
    if condition == "same_lexical":
        return int(target["same_family_case_index"]), target
    if condition == "cross_shuffled":
        donor_target = targets[int(target["shuffled_target_index"])]
        return int(target["shuffled_cross_case_index"]), donor_target
    if condition == "self":
        return int(target["target_case_index"]), target
    raise ValueError(condition)


def cache_value(
    cache: np.memmap,
    local_index: int,
    current_role: str,
    length: int,
    mode: str,
) -> np.ndarray:
    if mode == "attention_write":
        indices = (protocol.CACHE_CHANNELS.index("attention_write"),)
    elif mode == "mlp_write":
        indices = (protocol.CACHE_CHANNELS.index("mlp_write"),)
    elif mode == "current_write":
        indices = (
            protocol.CACHE_CHANNELS.index("attention_write"),
            protocol.CACHE_CHANNELS.index("mlp_write"),
        )
    elif mode == "full_state":
        indices = (protocol.CACHE_CHANNELS.index("layer_output"),)
    else:
        raise ValueError(mode)
    result = np.zeros(
        (length, cache.shape[-1]), dtype=np.float32
    )
    for channel_index in indices:
        result += np.asarray(
            cache[
                local_index,
                channel_index,
                role_slot(current_role),
                :length,
                :,
            ],
            dtype=np.float32,
        )
    return result


def make_patch_batch(
    target_rows: list[dict[str, Any]],
    target_lookup: dict[int, dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    case_to_local: dict[int, int],
    cache: np.memmap,
    mask_name: str,
    mode: str,
    condition: str,
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
    np.ndarray,
]:
    model_rows = [
        cases[int(row["target_case_index"])] for row in target_rows
    ]
    width = max(len(row["input_ids"]) for row in model_rows)
    ids = torch.full(
        (len(model_rows), width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(model_rows), width), dtype=torch.long
    )
    positions = torch.zeros(
        (len(model_rows), protocol.MAX_PATCH_TOKENS),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    payloads = np.zeros(
        (
            len(model_rows),
            protocol.MAX_PATCH_TOKENS,
            cache.shape[-1],
        ),
        dtype=np.float32,
    )
    pre_positions = torch.empty(len(model_rows), dtype=torch.long)
    sites = protocol.POSITION_MASKS[mask_name]

    for row_index, (target, model_row) in enumerate(
        zip(target_rows, model_rows)
    ):
        values = torch.tensor(model_row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        donor_case_index, donor_target = donor_spec(
            target, condition, target_lookup
        )
        donor_row = cases[donor_case_index]
        cursor = 0
        for site in sites:
            target_role = semantic_role(site, target)
            donor_role = semantic_role(site, donor_target)
            target_start, target_end = (
                int(value)
                for value in model_row["anchor_spans"][target_role]
            )
            donor_start, donor_end = (
                int(value)
                for value in donor_row["anchor_spans"][donor_role]
            )
            target_span = list(range(target_start, target_end + 1))
            donor_span = list(range(donor_start, donor_end + 1))
            if len(target_span) != len(donor_span):
                raise RuntimeError(
                    f"span mismatch {site}: "
                    f"{len(target_span)} != {len(donor_span)}"
                )
            if cursor + len(target_span) > protocol.MAX_PATCH_TOKENS:
                raise RuntimeError("patch token budget exceeded")
            target_value = cache_value(
                cache,
                case_to_local[int(target["target_case_index"])],
                target_role,
                len(target_span),
                mode,
            )
            donor_value = cache_value(
                cache,
                case_to_local[donor_case_index],
                donor_role,
                len(donor_span),
                mode,
            )
            positions[
                row_index, cursor:cursor + len(target_span)
            ] = torch.tensor(target_span, dtype=torch.long)
            masks[
                row_index, cursor:cursor + len(target_span)
            ] = True
            payloads[
                row_index, cursor:cursor + len(target_span), :
            ] = donor_value - target_value
            cursor += len(target_span)
        active_positions = positions[row_index, masks[row_index]].tolist()
        if len(active_positions) != len(set(active_positions)):
            raise RuntimeError(
                f"overlapping patch positions for {mask_name}"
            )
        pre_positions[row_index] = int(
            model_row["anchor_spans"]["pre_output"][1]
        )
    return (
        ids.to(device),
        attention_mask.to(device),
        positions,
        masks,
        pre_positions,
        torch.from_numpy(payloads),
        np.asarray(
            [int(row["discovery_index"]) for row in target_rows],
            dtype=np.int64,
        ),
    )


class AlliancePatch:
    def __init__(self, layer: Any):
        self.layer = layer
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
            raise RuntimeError("alliance patch context missing")
        hidden = output_tensor(output)
        patched = hidden.clone()
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        payloads = self.payloads.to(hidden.device, dtype=hidden.dtype)
        for slot in range(protocol.MAX_PATCH_TOKENS):
            active = torch.where(masks[:, slot])[0]
            if len(active) == 0:
                continue
            target_positions = positions[active, slot]
            patched[active, target_positions, :] += payloads[
                active, slot, :
            ]
        self.count += 1
        return replace_output(output, patched)

    def register(self) -> None:
        self.handle = self.layer.register_forward_hook(self._hook)

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
                f"alliance patch count drift: {self.count}"
            )
        self.positions = None
        self.masks = None
        self.payloads = None

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def behavior_summary(
    clean_logits: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = np.asarray(
        [int(row["expected_index"]) for row in cases], dtype=np.int64
    )
    finite = np.all(np.isfinite(clean_logits), axis=-1)
    prediction = np.argmax(
        np.where(np.isfinite(clean_logits), clean_logits, -np.inf),
        axis=-1,
    )
    return {
        "row_count": len(cases),
        "finite_rate": float(np.mean(finite)),
        "candidate_accuracy": float(np.mean(
            prediction == expected
        )),
    }


def intervention_index(mode: str, condition: str) -> int:
    return protocol.INTERVENTIONS.index((mode, condition))


def group_indices(
    targets: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    result = {"all": np.arange(len(targets), dtype=np.int64)}
    for template in (0, 1):
        for stratum in protocol.source.SURFACE_STRATA:
            result[f"template_{template}/{stratum}"] = np.asarray([
                index for index, row in enumerate(targets)
                if int(row["template_index"]) == template
                and row["surface_stratum"] == stratum
            ], dtype=np.int64)
    return result


def paired_metrics(
    patched_logits: np.ndarray,
    clean_logits: np.ndarray,
    targets: list[dict[str, Any]],
    case_to_local: dict[int, int],
) -> dict[str, Any]:
    target_family = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_family = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    target_case_local = np.asarray([
        case_to_local[int(row["target_case_index"])]
        for row in targets
    ], dtype=np.int64)
    clean_margin = (
        clean_logits[target_case_local, cross_family]
        - clean_logits[target_case_local, target_family]
    )
    rows = np.arange(len(targets), dtype=np.int64)
    shifts = np.empty(
        (
            len(targets),
            len(protocol.POSITION_MASKS),
            len(protocol.INTERVENTIONS),
        ),
        dtype=np.float32,
    )
    for mask_index in range(len(protocol.POSITION_MASKS)):
        current = patched_logits[:, mask_index]
        margin = (
            current[
                rows[:, None],
                np.arange(len(protocol.INTERVENTIONS))[None, :],
                cross_family[:, None],
            ]
            - current[
                rows[:, None],
                np.arange(len(protocol.INTERVENTIONS))[None, :],
                target_family[:, None],
            ]
        )
        shifts[:, mask_index, :] = margin - clean_margin[:, None]

    groups = group_indices(targets)
    result_rows = []
    mask_names = tuple(protocol.POSITION_MASKS)
    for mask_index, mask_name in enumerate(mask_names):
        full = shifts[
            :,
            mask_index,
            intervention_index("full_state", "cross_matched"),
        ]
        full_median = scalar_summary(full)["median"]
        for mode in (
            "attention_write",
            "mlp_write",
            "current_write",
        ):
            cross = shifts[
                :,
                mask_index,
                intervention_index(mode, "cross_matched"),
            ]
            same = (
                shifts[
                    :,
                    mask_index,
                    intervention_index(mode, "same_lexical"),
                ]
                if (mode, "same_lexical")
                in protocol.INTERVENTIONS
                else None
            )
            shuffled = (
                shifts[
                    :,
                    mask_index,
                    intervention_index(mode, "cross_shuffled"),
                ]
                if (mode, "cross_shuffled")
                in protocol.INTERVENTIONS
                else None
            )
            group_rows = {}
            for group, indices in groups.items():
                cross_summary = scalar_summary(cross[indices])
                same_abs = (
                    scalar_summary(np.abs(same[indices]))
                    if same is not None else None
                )
                shuffled_abs = (
                    scalar_summary(np.abs(shuffled[indices]))
                    if shuffled is not None else None
                )
                purity = None
                matched_to_shuffled = None
                if (
                    same_abs is not None
                    and same_abs["median"] is not None
                ):
                    purity = (
                        abs(float(cross_summary["median"]))
                        / (float(same_abs["median"]) + EPS)
                    )
                if (
                    shuffled_abs is not None
                    and shuffled_abs["median"] is not None
                ):
                    matched_to_shuffled = (
                        abs(float(cross_summary["median"]))
                        / (float(shuffled_abs["median"]) + EPS)
                    )
                group_rows[group] = {
                    "cross_shift": cross_summary,
                    "same_lexical_absolute_shift": same_abs,
                    "shuffled_absolute_shift": shuffled_abs,
                    "purity_ratio": purity,
                    "matched_to_shuffled_ratio": matched_to_shuffled,
                }
            all_cross = group_rows["all"]["cross_shift"]
            retention = (
                float(all_cross["median"])
                / (float(full_median) + EPS)
                if full_median is not None
                and abs(float(full_median)) > EPS
                and all_cross["median"] is not None
                else None
            )
            result_rows.append({
                "mask": mask_name,
                "mode": mode,
                "groups": group_rows,
                "full_state_cross_shift": scalar_summary(full),
                "full_state_retention": retention,
            })

    row_lookup = {
        (row["mask"], row["mode"]): row for row in result_rows
    }
    alliance_rows = []
    for mask_name, constituents in protocol.CONSTITUENTS.items():
        mask_index = mask_names.index(mask_name)
        for mode in ("mlp_write", "current_write"):
            joint = shifts[
                :,
                mask_index,
                intervention_index(mode, "cross_matched"),
            ]
            constituent_values = np.stack([
                shifts[
                    :,
                    mask_names.index(constituent),
                    intervention_index(mode, "cross_matched"),
                ]
                for constituent in constituents
            ], axis=1)
            best_gain = joint - np.max(constituent_values, axis=1)
            additivity = joint - np.sum(constituent_values, axis=1)
            baseline = shifts[
                :,
                mask_names.index("selected_concept"),
                intervention_index(mode, "cross_matched"),
            ]
            current = row_lookup[(mask_name, mode)]
            base_row = row_lookup[("selected_concept", mode)]
            current_purity = current["groups"]["all"]["purity_ratio"]
            base_purity = base_row["groups"]["all"]["purity_ratio"]
            alliance_rows.append({
                "mask": mask_name,
                "mode": mode,
                "constituents": list(constituents),
                "best_constituent_gain": scalar_summary(best_gain),
                "additivity_residual": scalar_summary(additivity),
                "cross_gain_over_selected_concept": scalar_summary(
                    joint - baseline
                ),
                "purity_gain_over_selected_concept": (
                    float(current_purity) - float(base_purity)
                    if current_purity is not None
                    and base_purity is not None
                    else None
                ),
            })

    role_specificity = {}
    for mode in ("mlp_write", "current_write"):
        selected = shifts[
            :,
            mask_names.index("selected_fact"),
            intervention_index(mode, "cross_matched"),
        ]
        unselected = shifts[
            :,
            mask_names.index("unselected_fact"),
            intervention_index(mode, "cross_matched"),
        ]
        role_specificity[mode] = scalar_summary(selected - unselected)

    return {
        "clean_margin": scalar_summary(clean_margin),
        "position_mode_rows": result_rows,
        "alliance_rows": alliance_rows,
        "selected_minus_unselected_fact": role_specificity,
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not all(protocol_audit["checks"].values()):
        raise RuntimeError("protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    targets = [
        {**row, "discovery_index": index}
        for index, row in enumerate(targets)
    ]
    target_lookup = {
        int(row["target_index"]): row for row in targets
    }
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"discovery_cases.{model_name}.jsonl"
    )
    cases = {
        int(row["case_index"]): row for row in cases_list
    }
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    physical_depth = int(
        prereg["model_physical_depth"][model_name]
    )
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
        layer = layers[physical_depth - 1]
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )

        cache = np.lib.format.open_memmap(
            atlas_dir / "clean_role_channels.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(protocol.CACHE_CHANNELS),
                len(protocol.ROLE_ORDER),
                protocol.MAX_ROLE_SPAN,
                info.d_model,
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "channel_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(protocol.ROLE_ORDER)),
        )
        clean_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(protocol.source.FAMILIES)),
        )
        cache[:] = np.nan
        closure[:] = np.nan
        clean_logits[:] = np.nan
        capture = ChannelCapture(
            layer, cache, closure, case_to_local
        )
        capture.register()
        try:
            for row_batch in chunks(
                cases_list, BATCH_SIZE[model_name]
            ):
                (
                    input_ids,
                    attention_mask,
                    positions,
                    masks,
                    pre_positions,
                    case_indices,
                ) = make_clean_batch(
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
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float()
                candidates = selected.index_select(
                    -1, candidate_ids.to(selected.device)
                )
                local = np.asarray([
                    case_to_local[int(value)] for value in case_indices
                ], dtype=np.int64)
                clean_logits[local] = candidates.detach().cpu().numpy()
                del output, logits, selected, candidates
        finally:
            capture.close()
        cache.flush()
        closure.flush()
        clean_logits.flush()

        patcher = AlliancePatch(layer)
        patcher.register()
        zero_identity = {}
        identity_reference: np.ndarray | None = None
        try:
            identity_targets = targets[
                :min(16, BATCH_SIZE[model_name])
            ]
            for mode in protocol.PATCH_MODES:
                (
                    input_ids,
                    attention_mask,
                    positions,
                    masks,
                    pre_positions,
                    payloads,
                    target_indices,
                ) = make_patch_batch(
                    identity_targets,
                    target_lookup,
                    cases,
                    case_to_local,
                    cache,
                    "all_facts_query_boundary",
                    mode,
                    "self",
                    pad_token_id=pad_token_id,
                    device=device,
                )
                if float(torch.max(torch.abs(payloads)).item()) != 0.0:
                    raise RuntimeError("self payload is not exactly zero")
                patcher.begin(positions, masks, payloads)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                patcher.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                selected_values = selected.detach().cpu().numpy()
                if identity_reference is None:
                    identity_reference = selected_values.copy()
                difference = selected_values - identity_reference
                zero_identity[mode] = {
                    "payload_max_absolute": float(
                        torch.max(torch.abs(payloads)).item()
                    ),
                    "max_absolute_logit_difference": float(
                        np.nanmax(np.abs(difference))
                    ),
                    "exact": bool(
                        np.all(
                            np.nan_to_num(difference, nan=0.0) == 0.0
                        )
                    ),
                }
                del output, logits, selected

            patched_logits = np.lib.format.open_memmap(
                atlas_dir / "patched_candidate_logits.fp32.npy",
                mode="w+",
                dtype=np.float32,
                shape=(
                    len(targets),
                    len(protocol.POSITION_MASKS),
                    len(protocol.INTERVENTIONS),
                    len(protocol.source.FAMILIES),
                ),
            )
            patched_logits[:] = np.nan
            for mask_index, mask_name in enumerate(
                protocol.POSITION_MASKS
            ):
                for intervention_index_value, (
                    mode,
                    condition,
                ) in enumerate(protocol.INTERVENTIONS):
                    for target_batch in chunks(
                        targets, BATCH_SIZE[model_name]
                    ):
                        (
                            input_ids,
                            attention_mask,
                            positions,
                            masks,
                            pre_positions,
                            payloads,
                            target_indices,
                        ) = make_patch_batch(
                            target_batch,
                            target_lookup,
                            cases,
                            case_to_local,
                            cache,
                            mask_name,
                            mode,
                            condition,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        patcher.begin(positions, masks, payloads)
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        patcher.end()
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
                        patched_logits[
                            target_indices,
                            mask_index,
                            intervention_index_value,
                            :,
                        ] = selected.detach().cpu().numpy()
                        del output, logits, selected
                    print(
                        f"[phase1041] {model_name} "
                        f"mask={mask_name} mode={mode} "
                        f"condition={condition}",
                        flush=True,
                    )
            patched_logits.flush()
        finally:
            patcher.close()

        metrics = {
            "schema_version": "phase1041_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "behavior": behavior_summary(clean_logits, cases_list),
            **paired_metrics(
                patched_logits,
                clean_logits,
                targets,
                case_to_local,
            ),
        }
        summary = {
            "schema_version": "phase1041_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": model.__class__.__name__,
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
            },
            "physical_depth": physical_depth,
            "sample_counts": {
                "clean_cases": len(cases_list),
                "discovery_targets": len(targets),
                "position_masks": len(protocol.POSITION_MASKS),
                "interventions_per_mask": len(
                    protocol.INTERVENTIONS
                ),
                "patched_forward_rows": (
                    len(targets)
                    * len(protocol.POSITION_MASKS)
                    * len(protocol.INTERVENTIONS)
                ),
            },
            "array_finiteness": {
                "clean_role_channels": finite_summary(cache),
                "channel_closure": finite_summary(closure),
                "clean_candidate_logits": finite_summary(clean_logits),
                "patched_candidate_logits": finite_summary(
                    patched_logits
                ),
            },
            "instrumentation_closure": scalar_summary(closure),
            "zero_delta_identity": zero_identity,
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
