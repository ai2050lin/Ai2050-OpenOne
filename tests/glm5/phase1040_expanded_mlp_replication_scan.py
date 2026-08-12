#!/usr/bin/env python3
"""Run the expanded early-MLP replication on all three local models."""

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
import phase1040_expanded_mlp_replication_protocol as protocol


BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8
MAX_SPAN = 2


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def role_index(role: str) -> int:
    return 0 if role == "concept_a" else 1


def patch_spec(
    target: dict[str, Any],
    condition: str,
) -> tuple[int, str, str]:
    if condition == "same_family_selected":
        return (
            int(target["same_family_case_index"]),
            str(target["selected_role"]),
            str(target["selected_role"]),
        )
    if condition == "cross_family_selected":
        return (
            int(target["cross_family_case_index"]),
            str(target["selected_role"]),
            str(target["selected_role"]),
        )
    if condition == "cross_family_unselected":
        return (
            int(target["cross_family_case_index"]),
            str(target["unselected_role"]),
            str(target["unselected_role"]),
        )
    if condition == "cross_family_wrong_target":
        return (
            int(target["cross_family_case_index"]),
            str(target["selected_role"]),
            str(target["unselected_role"]),
        )
    raise ValueError(condition)


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
        (len(rows), 2, MAX_SPAN), dtype=torch.long
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    pre_positions = torch.empty(len(rows), dtype=torch.long)
    for row_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        for role_slot, role in enumerate(("concept_a", "concept_b")):
            start, end = (
                int(value) for value in row["anchor_spans"][role]
            )
            span = list(range(start, end + 1))
            if len(span) not in (1, 2):
                raise RuntimeError("unexpected concept span length")
            positions[row_index, role_slot, :len(span)] = torch.tensor(
                span, dtype=torch.long
            )
            masks[row_index, role_slot, :len(span)] = True
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


class CleanSourceCapture:
    def __init__(
        self,
        layer: Any,
        cache: np.memmap,
    ):
        self.layer = layer
        self.cache = cache
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.case_indices: np.ndarray | None = None
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
        self.case_indices = case_indices
        self.counts = defaultdict(int)

    def _save(self, hidden: torch.Tensor, channel_index: int) -> None:
        if (
            self.positions is None
            or self.masks is None
            or self.case_indices is None
        ):
            raise RuntimeError("clean source capture context missing")
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None, None].expand_as(positions)
        values = hidden[batch, positions, :].clone()
        values = values.masked_fill(~masks[..., None], 0)
        self.cache[
            self.case_indices, channel_index, :, :, :
        ] = values.detach().to(
            "cpu", dtype=torch.float16
        ).numpy()

    def _mlp_hook(self, module, args, output):
        self._save(output_tensor(output), 0)
        self.counts["mlp_write"] += 1
        return output

    def _layer_hook(self, module, args, output):
        self._save(output_tensor(output), 1)
        self.counts["layer_output"] += 1
        return output

    def register(self) -> None:
        self.handles.append(
            self.layer.mlp.register_forward_hook(self._mlp_hook)
        )
        self.handles.append(
            self.layer.register_forward_hook(self._layer_hook)
        )

    def end(self) -> None:
        if dict(self.counts) != {
            "mlp_write": 1,
            "layer_output": 1,
        }:
            raise RuntimeError(
                f"clean source hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.case_indices = None

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


def make_patch_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean_cache: np.memmap,
    channel_index: int,
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
    patch_positions = torch.zeros(
        (len(model_rows), MAX_SPAN), dtype=torch.long
    )
    patch_masks = torch.zeros_like(patch_positions, dtype=torch.bool)
    pre_positions = torch.empty(len(model_rows), dtype=torch.long)
    payloads = np.zeros(
        (len(model_rows), MAX_SPAN, clean_cache.shape[-1]),
        dtype=np.float32,
    )
    for row_index, (target, model_row) in enumerate(
        zip(target_rows, model_rows)
    ):
        values = torch.tensor(model_row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        donor_case, donor_role, target_role = patch_spec(
            target, condition
        )
        target_start, target_end = (
            int(value)
            for value in model_row["anchor_spans"][target_role]
        )
        donor_row = cases[donor_case]
        donor_start, donor_end = (
            int(value)
            for value in donor_row["anchor_spans"][donor_role]
        )
        target_span = list(range(target_start, target_end + 1))
        donor_span = list(range(donor_start, donor_end + 1))
        if len(target_span) != len(donor_span):
            raise RuntimeError("donor/target span length mismatch")
        patch_positions[
            row_index, :len(target_span)
        ] = torch.tensor(target_span, dtype=torch.long)
        patch_masks[row_index, :len(target_span)] = True
        pre_positions[row_index] = int(
            model_row["anchor_spans"]["pre_output"][1]
        )
        donor = np.asarray(
            clean_cache[
                donor_case,
                channel_index,
                role_index(donor_role),
                :len(donor_span),
                :,
            ],
            dtype=np.float32,
        )
        if protocol.CHANNELS[channel_index] == "mlp_write":
            target_value = np.asarray(
                clean_cache[
                    int(target["target_case_index"]),
                    channel_index,
                    role_index(target_role),
                    :len(target_span),
                    :,
                ],
                dtype=np.float32,
            )
            donor = donor - target_value
        payloads[row_index, :len(target_span)] = donor
    return (
        ids.to(device),
        attention_mask.to(device),
        patch_positions,
        patch_masks,
        pre_positions,
        torch.from_numpy(payloads),
        np.asarray(
            [int(row["target_index"]) for row in target_rows],
            dtype=np.int64,
        ),
    )


class SpanPatch:
    def __init__(self, layer: Any):
        self.layer = layer
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.payloads: torch.Tensor | None = None
        self.additive = True
        self.count = 0
        self.handle = None

    def _hook(self, module, args, output):
        if (
            self.positions is None
            or self.masks is None
            or self.payloads is None
        ):
            raise RuntimeError("span patch context missing")
        hidden = output_tensor(output)
        patched = hidden.clone()
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        payloads = self.payloads.to(hidden.device, dtype=hidden.dtype)
        for span_slot in range(MAX_SPAN):
            active = torch.where(masks[:, span_slot])[0]
            if len(active) == 0:
                continue
            target_positions = positions[active, span_slot]
            if self.additive:
                patched[active, target_positions, :] += payloads[
                    active, span_slot, :
                ]
            else:
                patched[active, target_positions, :] = payloads[
                    active, span_slot, :
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
        *,
        additive: bool,
    ) -> None:
        self.positions = positions
        self.masks = masks
        self.payloads = payloads
        self.additive = additive
        self.count = 0

    def end(self) -> None:
        if self.count != 1:
            raise RuntimeError(f"span patch count drift: {self.count}")
        self.positions = None
        self.masks = None
        self.payloads = None

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    return metric_tools.scalar_summary(values)


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
    groups = {
        "all": np.arange(len(cases)),
        "template_0": np.asarray([
            i for i, row in enumerate(cases)
            if int(row["template_index"]) == 0
        ], dtype=np.int64),
        "template_1": np.asarray([
            i for i, row in enumerate(cases)
            if int(row["template_index"]) == 1
        ], dtype=np.int64),
        "single_token": np.asarray([
            i for i, row in enumerate(cases)
            if row["surface_stratum"] == "single_token"
        ], dtype=np.int64),
        "two_token": np.asarray([
            i for i, row in enumerate(cases)
            if row["surface_stratum"] == "two_token"
        ], dtype=np.int64),
    }
    return {
        group: {
            "row_count": int(len(indices)),
            "finite_rate": float(np.mean(finite[indices])),
            "candidate_accuracy": float(np.mean(
                prediction[indices] == expected[indices]
            )),
        }
        for group, indices in groups.items()
    }


def paired_metrics(
    patched_logits: np.ndarray,
    clean_logits: np.ndarray,
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
    indices = np.arange(len(targets))
    clean_margin = (
        clean_logits[
            np.asarray(
                [int(row["target_case_index"]) for row in targets]
            ),
            cross_family,
        ]
        - clean_logits[
            np.asarray(
                [int(row["target_case_index"]) for row in targets]
            ),
            target_family,
        ]
    )
    condition_slots = {
        condition: protocol.CONDITIONS.index(condition)
        for condition in protocol.CONDITIONS
    }
    groups = {
        "all": indices,
        "template_0": np.asarray([
            i for i, row in enumerate(targets)
            if int(row["template_index"]) == 0
        ], dtype=np.int64),
        "template_1": np.asarray([
            i for i, row in enumerate(targets)
            if int(row["template_index"]) == 1
        ], dtype=np.int64),
        "single_token": np.asarray([
            i for i, row in enumerate(targets)
            if row["surface_stratum"] == "single_token"
        ], dtype=np.int64),
        "two_token": np.asarray([
            i for i, row in enumerate(targets)
            if row["surface_stratum"] == "two_token"
        ], dtype=np.int64),
    }
    for template in (0, 1):
        for stratum in protocol.SURFACE_STRATA:
            groups[f"template_{template}/{stratum}"] = np.asarray([
                i for i, row in enumerate(targets)
                if int(row["template_index"]) == template
                and row["surface_stratum"] == stratum
            ], dtype=np.int64)

    mode_shifts: dict[str, np.ndarray] = {}
    for channel_index, channel in enumerate(protocol.CHANNELS):
        current = np.asarray(
            patched_logits[:, channel_index], dtype=np.float32
        )
        margins = (
            current[
                indices[:, None],
                np.arange(len(protocol.CONDITIONS))[None, :],
                cross_family[:, None],
            ]
            - current[
                indices[:, None],
                np.arange(len(protocol.CONDITIONS))[None, :],
                target_family[:, None],
            ]
        )
        mode_shifts[channel] = margins - clean_margin[:, None]

    rows = []
    for channel in protocol.CHANNELS:
        shifts = mode_shifts[channel]
        same = shifts[:, condition_slots["same_family_selected"]]
        cross = shifts[:, condition_slots["cross_family_selected"]]
        unselected = shifts[
            :, condition_slots["cross_family_unselected"]
        ]
        wrong = shifts[
            :, condition_slots["cross_family_wrong_target"]
        ]
        full_cross = mode_shifts["layer_output"][
            :, condition_slots["cross_family_selected"]
        ]
        for group, group_indices in groups.items():
            usable = group_indices[
                np.isfinite(same[group_indices])
                & np.isfinite(cross[group_indices])
                & np.isfinite(unselected[group_indices])
                & np.isfinite(wrong[group_indices])
                & np.isfinite(full_cross[group_indices])
            ]
            cross_abs = (
                float(np.median(np.abs(cross[usable])))
                if len(usable)
                else None
            )
            same_abs = (
                float(np.median(np.abs(same[usable])))
                if len(usable)
                else None
            )
            cross_median = (
                float(np.median(cross[usable])) if len(usable) else None
            )
            full_median = (
                float(np.median(full_cross[usable]))
                if len(usable)
                else None
            )
            rows.append({
                "channel": channel,
                "group": group,
                "row_count": int(len(group_indices)),
                "usable_count": int(len(usable)),
                "cross_selected_shift": scalar_summary(cross[usable]),
                "same_family_absolute_shift": scalar_summary(
                    np.abs(same[usable])
                ),
                "selected_minus_unselected": scalar_summary(
                    cross[usable] - unselected[usable]
                ),
                "selected_minus_wrong_target": scalar_summary(
                    cross[usable] - wrong[usable]
                ),
                "cross_to_same_absolute_ratio": (
                    float(cross_abs / max(same_abs, EPS))
                    if cross_abs is not None and same_abs is not None
                    else None
                ),
                "whole_state_effect_retention": (
                    float(cross_median / full_median)
                    if (
                        cross_median is not None
                        and full_median is not None
                        and abs(full_median) > EPS
                    )
                    else None
                ),
            })

    mlp_cross = mode_shifts["mlp_write"][
        :, condition_slots["cross_family_selected"]
    ]
    pair_rows = []
    for stratum in protocol.SURFACE_STRATA:
        for pair in sorted({row["ordered_pair"] for row in targets}):
            pair_indices = np.asarray([
                i for i, row in enumerate(targets)
                if row["surface_stratum"] == stratum
                and row["ordered_pair"] == pair
            ], dtype=np.int64)
            pair_rows.append({
                "surface_stratum": stratum,
                "ordered_pair": pair,
                "row_count": int(len(pair_indices)),
                "mlp_cross_selected_shift": scalar_summary(
                    mlp_cross[pair_indices]
                ),
            })
    pair_positive_rate = {
        stratum: float(np.mean([
            row["mlp_cross_selected_shift"]["median"] > 0
            for row in pair_rows
            if row["surface_stratum"] == stratum
        ]))
        for stratum in protocol.SURFACE_STRATA
    }
    return {
        "paired_rows": rows,
        "ordered_pair_rows": pair_rows,
        "ordered_pair_positive_median_rate": pair_positive_rate,
    }


def finite_summary(values: np.ndarray) -> dict[str, Any]:
    return metric_tools.finite_summary(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{args.model}.jsonl"
    )
    cases = {
        int(row["case_index"]): row for row in cases_list
    }
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    physical_depth = int(
        prereg["model_physical_depth"][args.model]
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
        layer = layers[physical_depth - 1]
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )

        clean_cache = np.lib.format.open_memmap(
            atlas_dir / "clean_source_channels.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(protocol.CHANNELS),
                2,
                MAX_SPAN,
                info.d_model,
            ),
        )
        clean_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(protocol.FAMILIES)),
        )
        clean_cache[:] = np.nan
        clean_logits[:] = np.nan
        capture = CleanSourceCapture(layer, clean_cache)
        capture.register()
        try:
            for batch_number, row_batch in enumerate(
                chunks(cases_list, BATCH_SIZE[args.model]), 1
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
                clean_logits[case_indices] = (
                    candidates.detach().cpu().numpy()
                )
                del output, logits, selected, candidates
                if batch_number % 64 == 0:
                    print(
                        f"[phase1040-clean] {args.model} "
                        f"cases={int(case_indices[-1]) + 1}/"
                        f"{len(cases_list)}",
                        flush=True,
                    )
        finally:
            capture.close()
        clean_cache.flush()
        clean_logits.flush()

        patched_logits = np.lib.format.open_memmap(
            atlas_dir / "patched_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.CHANNELS),
                len(protocol.CONDITIONS),
                len(protocol.FAMILIES),
            ),
        )
        patched_logits[:] = np.nan
        patcher = SpanPatch(layer)
        patcher.register()
        try:
            for channel_index, channel in enumerate(protocol.CHANNELS):
                for condition_index, condition in enumerate(
                    protocol.CONDITIONS
                ):
                    for target_batch in chunks(
                        targets, BATCH_SIZE[args.model]
                    ):
                        (
                            input_ids,
                            attention_mask,
                            patch_positions,
                            patch_masks,
                            pre_positions,
                            payloads,
                            target_indices,
                        ) = make_patch_batch(
                            target_batch,
                            cases,
                            clean_cache,
                            channel_index,
                            condition,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        patcher.begin(
                            patch_positions,
                            patch_masks,
                            payloads,
                            additive=(channel == "mlp_write"),
                        )
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
                        ].float()
                        candidates = selected.index_select(
                            -1, candidate_ids.to(selected.device)
                        )
                        patched_logits[
                            target_indices,
                            channel_index,
                            condition_index,
                            :,
                        ] = candidates.detach().cpu().numpy()
                        del output, logits, selected, candidates
                    print(
                        f"[phase1040-patch] {args.model} "
                        f"channel={channel} condition={condition}",
                        flush=True,
                    )
        finally:
            patcher.close()
        patched_logits.flush()

        paired = paired_metrics(
            patched_logits, clean_logits, targets
        )
        metrics = {
            "schema_version": "phase1040_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": args.model,
            "behavior": behavior_summary(clean_logits, cases_list),
            **paired,
        }
        summary = {
            "schema_version": "phase1040_model_summary.v1",
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
            "physical_depth": physical_depth,
            "normalized_depth_slot": 1,
            "sample_counts": {
                "cases": len(cases_list),
                "targets": len(targets),
                "channels": len(protocol.CHANNELS),
                "conditions": len(protocol.CONDITIONS),
                "patched_forward_rows": (
                    len(targets)
                    * len(protocol.CHANNELS)
                    * len(protocol.CONDITIONS)
                ),
            },
            "array_finiteness": {
                "clean_source_channels": finite_summary(clean_cache),
                "clean_candidate_logits": finite_summary(clean_logits),
                "patched_candidate_logits": finite_summary(
                    patched_logits
                ),
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
