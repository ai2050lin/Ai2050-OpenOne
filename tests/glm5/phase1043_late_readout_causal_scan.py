#!/usr/bin/env python3
"""Run Phase1043 late query-write causal confirmation."""

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
import phase1043_late_readout_causal_protocol as protocol


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
        (len(rows), protocol.MAX_SPAN), dtype=torch.long
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    pre_positions = torch.empty(len(rows), dtype=torch.long)
    for row_index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_index, :len(values)] = values
        attention_mask[row_index, :len(values)] = 1
        start, end = (
            int(value)
            for value in row["anchor_spans"][protocol.SITE]
        )
        span = list(range(start, end + 1))
        positions[row_index, :len(span)] = torch.tensor(
            span, dtype=torch.long
        )
        masks[row_index, :len(span)] = True
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


class DepthCacheCapture:
    def __init__(
        self,
        layers: list[Any],
        depths: list[int],
        cache: np.memmap,
        closure: np.memmap,
        case_to_local: dict[int, int],
    ):
        self.layers = layers
        self.depths = depths
        self.depth_slots = {
            depth: index for index, depth in enumerate(depths)
        }
        self.cache = cache
        self.closure = closure
        self.case_to_local = case_to_local
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.local_indices: np.ndarray | None = None
        self.current: dict[int, dict[str, torch.Tensor]] = {}
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
            raise RuntimeError("capture context missing")
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None].expand_as(positions)
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
            if self.local_indices is None or self.masks is None:
                raise RuntimeError("capture indices missing")
            current = self.current[depth]
            current["layer_output"] = self._states(
                output_tensor(output)
            )
            depth_slot = self.depth_slots[depth]
            for channel_slot, channel in enumerate(protocol.CHANNELS):
                self.cache[
                    self.local_indices,
                    depth_slot,
                    channel_slot,
                    :,
                    :,
                ] = current[channel].to(
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
            self.closure[self.local_indices, depth_slot] = (
                torch.nanmean(relative, dim=-1).cpu().numpy()
            )
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
                f"cache hook count drift: {dict(self.counts)}"
            )
        self.positions = None
        self.masks = None
        self.local_indices = None
        self.current = {}

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []


class ComponentPatch:
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
            raise RuntimeError("patch context missing")
        hidden = output_tensor(output)
        patched = hidden.clone()
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        payloads = self.payloads.to(hidden.device, dtype=hidden.dtype)
        for span_slot in range(protocol.MAX_SPAN):
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
            raise RuntimeError(f"patch hook count drift: {self.count}")
        self.positions = None
        self.masks = None
        self.payloads = None

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def make_patch_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    case_to_local: dict[int, int],
    cache: np.memmap,
    depth_slot: int,
    cache_channel: str,
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
    (
        ids,
        attention_mask,
        positions,
        masks,
        pre_positions,
        _,
    ) = make_clean_batch(
        model_rows, pad_token_id=pad_token_id, device=device
    )
    channel_slot = protocol.CHANNELS.index(cache_channel)
    payloads = np.zeros(
        (len(target_rows), protocol.MAX_SPAN, cache.shape[-1]),
        dtype=np.float32,
    )
    for row_index, target in enumerate(target_rows):
        target_case = int(target["target_case_index"])
        donor_case = protocol.donor_case(target, condition)
        target_row = cases[target_case]
        donor_row = cases[donor_case]
        target_start, target_end = (
            int(value)
            for value in target_row["anchor_spans"][protocol.SITE]
        )
        donor_start, donor_end = (
            int(value)
            for value in donor_row["anchor_spans"][protocol.SITE]
        )
        target_length = target_end - target_start + 1
        donor_length = donor_end - donor_start + 1
        if target_length != donor_length:
            raise RuntimeError("query-span length drift")
        if condition == "self_zero":
            continue
        target_value = np.asarray(
            cache[
                case_to_local[target_case],
                depth_slot,
                channel_slot,
                :target_length,
                :,
            ],
            dtype=np.float32,
        )
        donor_value = np.asarray(
            cache[
                case_to_local[donor_case],
                depth_slot,
                channel_slot,
                :donor_length,
                :,
            ],
            dtype=np.float32,
        )
        payloads[row_index, :target_length] = (
            donor_value - target_value
        )
    return (
        ids,
        attention_mask,
        positions,
        masks,
        pre_positions,
        torch.from_numpy(payloads),
        np.asarray(
            [int(row["confirmation_index"]) for row in target_rows],
            dtype=np.int64,
        ),
    )


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
    candidates: list[dict[str, Any]],
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
    clean_margin = (
        clean_logits[rows, cross_family]
        - clean_logits[rows, target_family]
    )
    margins = (
        patched_logits[
            rows[:, None, None],
            np.arange(len(candidates))[None, :, None],
            np.arange(len(protocol.INTERVENTIONS))[None, None, :],
            cross_family[:, None, None],
        ]
        - patched_logits[
            rows[:, None, None],
            np.arange(len(candidates))[None, :, None],
            np.arange(len(protocol.INTERVENTIONS))[None, None, :],
            target_family[:, None, None],
        ]
    )
    shifts = margins - clean_margin[:, None, None]
    groups = group_indices(targets)
    result_rows = []
    for candidate_index, candidate in enumerate(candidates):
        condition_values = {}
        for intervention_index, intervention in enumerate(
            protocol.INTERVENTIONS
        ):
            values = shifts[
                :, candidate_index, intervention_index
            ]
            condition_values["/".join(intervention)] = {
                group: scalar_summary(values[indices])
                for group, indices in groups.items()
            }
        cross = shifts[:, candidate_index, 0]
        same = shifts[:, candidate_index, 1]
        shuffled = shifts[:, candidate_index, 2]
        self_zero = shifts[:, candidate_index, 3]
        full = shifts[:, candidate_index, 4]
        cross_median = scalar_summary(cross)["median"]
        same_abs = scalar_summary(np.abs(same))["median"]
        shuffled_abs = scalar_summary(np.abs(shuffled))["median"]
        full_median = scalar_summary(full)["median"]
        result_rows.append({
            **candidate,
            "condition_groups": condition_values,
            "matched_to_same_absolute_ratio": (
                abs(float(cross_median))
                / (float(same_abs) + EPS)
                if cross_median is not None and same_abs is not None
                else None
            ),
            "matched_to_shuffled_absolute_ratio": (
                abs(float(cross_median))
                / (float(shuffled_abs) + EPS)
                if cross_median is not None
                and shuffled_abs is not None
                else None
            ),
            "full_state_retention": (
                float(cross_median) / (float(full_median) + EPS)
                if cross_median is not None
                and full_median is not None
                and abs(float(full_median)) > EPS
                else None
            ),
            "self_zero_shift": scalar_summary(self_zero),
        })
    return {
        "clean_margin": scalar_summary(clean_margin),
        "candidate_rows": result_rows,
    }


def behavior_summary(
    clean_logits: np.ndarray,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    finite = np.all(np.isfinite(clean_logits), axis=-1)
    prediction = np.argmax(
        np.where(np.isfinite(clean_logits), clean_logits, -np.inf),
        axis=-1,
    )
    return {
        "row_count": len(targets),
        "finite_rate": float(np.mean(finite)),
        "candidate_accuracy": float(np.mean(prediction == expected)),
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not all(audit["checks"].values()):
        raise RuntimeError("Phase1043 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    candidates = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "candidates.json"
    )
    cases = {int(row["case_index"]): row for row in cases_list}
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    unique_depths = sorted({
        int(row["physical_depths"][model_name])
        for row in candidates
    })
    depth_slots = {
        depth: index for index, depth in enumerate(unique_depths)
    }
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

        cache = np.lib.format.open_memmap(
            atlas_dir / "query_channels.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(unique_depths),
                len(protocol.CHANNELS),
                protocol.MAX_SPAN,
                info.d_model,
            ),
        )
        closure = np.lib.format.open_memmap(
            atlas_dir / "channel_closure.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(unique_depths)),
        )
        clean_all_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_all_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(cases_list),
                len(protocol.source.FAMILIES),
            ),
        )
        clean_target_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_target_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(targets), len(protocol.source.FAMILIES)),
        )
        patched_logits = np.lib.format.open_memmap(
            atlas_dir / "patched_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(candidates),
                len(protocol.INTERVENTIONS),
                len(protocol.source.FAMILIES),
            ),
        )
        cache[:] = np.nan
        closure[:] = np.nan
        clean_all_logits[:] = np.nan
        clean_target_logits[:] = np.nan
        patched_logits[:] = np.nan

        capture = DepthCacheCapture(
            layers,
            unique_depths,
            cache,
            closure,
            case_to_local,
        )
        capture.register()
        try:
            for row_batch in chunks(cases_list, BATCH_SIZE[model_name]):
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
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                local = np.asarray(
                    [case_to_local[int(value)] for value in case_indices],
                    dtype=np.int64,
                )
                clean_all_logits[local] = selected.detach().cpu().numpy()
                del output, logits, selected
        finally:
            capture.close()
        cache.flush()
        closure.flush()
        clean_all_logits.flush()

        target_model_rows = [
            cases[int(row["target_case_index"])] for row in targets
        ]
        for row_batch, target_batch in zip(
            chunks(target_model_rows, BATCH_SIZE[model_name]),
            chunks(targets, BATCH_SIZE[model_name]),
        ):
            (
                input_ids,
                attention_mask,
                _,
                _,
                pre_positions,
                _,
            ) = make_clean_batch(
                row_batch,
                pad_token_id=pad_token_id,
                device=device,
            )
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            logits = output.logits
            batch = torch.arange(logits.shape[0], device=logits.device)
            selected = logits[
                batch, pre_positions.to(logits.device), :
            ].float().index_select(
                -1, candidate_ids.to(logits.device)
            )
            indices = np.asarray(
                [
                    int(row["confirmation_index"])
                    for row in target_batch
                ],
                dtype=np.int64,
            )
            clean_target_logits[indices] = (
                selected.detach().cpu().numpy()
            )
            del output, logits, selected
        clean_target_logits.flush()

        for candidate_index, candidate in enumerate(candidates):
            depth = int(candidate["physical_depths"][model_name])
            depth_slot = depth_slots[depth]
            for intervention_index, (mode, condition) in enumerate(
                protocol.INTERVENTIONS
            ):
                cache_channel = (
                    candidate["channel"]
                    if mode == "candidate"
                    else "layer_output"
                )
                layer = layers[depth - 1]
                module = (
                    layer.self_attn
                    if cache_channel == "attention_write"
                    else layer.mlp
                    if cache_channel == "mlp_write"
                    else layer
                )
                patch = ComponentPatch(module)
                patch.register()
                try:
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
                            indices,
                        ) = make_patch_batch(
                            target_batch,
                            cases,
                            case_to_local,
                            cache,
                            depth_slot,
                            cache_channel,
                            condition,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        patch.begin(positions, masks, payloads)
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
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
                        patched_logits[
                            indices,
                            candidate_index,
                            intervention_index,
                            :,
                        ] = selected.detach().cpu().numpy()
                        del output, logits, selected
                finally:
                    patch.close()
                patched_logits.flush()
                print(
                    f"[phase1043] {model_name} candidate="
                    f"{candidate_index} depth={depth} "
                    f"{cache_channel}/{condition} complete",
                    flush=True,
                )

        metrics = {
            "schema_version": "phase1043_model_metrics.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "behavior": behavior_summary(
                clean_target_logits, targets
            ),
            **paired_metrics(
                patched_logits,
                clean_target_logits,
                targets,
                candidates,
            ),
            "zero_delta_identity": {
                "max_absolute_logit_difference": float(
                    np.nanmax(np.abs(
                        patched_logits[:, :, 3, :]
                        - clean_target_logits[:, None, :]
                    ))
                ),
                "exact": bool(np.array_equal(
                    np.asarray(patched_logits[:, :, 3, :]),
                    np.broadcast_to(
                        np.asarray(clean_target_logits)[:, None, :],
                        (
                            len(targets),
                            len(candidates),
                            len(protocol.source.FAMILIES),
                        ),
                    ),
                    equal_nan=True,
                )),
            },
        }
        summary = {
            "schema_version": "phase1043_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "class": info.model_class,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
            },
            "unique_depths": unique_depths,
            "sample_counts": {
                "targets": len(targets),
                "cases": len(cases_list),
                "candidates": len(candidates),
                "interventions": len(protocol.INTERVENTIONS),
                "patched_rows": (
                    len(targets)
                    * len(candidates)
                    * len(protocol.INTERVENTIONS)
                ),
            },
            "array_finiteness": {
                "query_channels": finite_summary(cache),
                "channel_closure": finite_summary(closure),
                "clean_target_logits": finite_summary(
                    clean_target_logits
                ),
                "patched_logits": finite_summary(patched_logits),
            },
            "instrumentation_closure": scalar_summary(closure),
            "zero_delta_identity": metrics["zero_delta_identity"],
            "elapsed_seconds": time.time() - started,
        }
        protocol.write_json(atlas_dir / "metrics.json", metrics)
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
