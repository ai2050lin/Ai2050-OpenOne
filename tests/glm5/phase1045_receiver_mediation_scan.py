#!/usr/bin/env python3
"""Run Phase1045 receiver reset/replay confirmation in native FP16."""

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
import phase1044_natural_recompute_trajectory_scan as tools
import phase1045_receiver_mediation_protocol as protocol


CLEAN_BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
TARGET_BATCH_SIZE = {"qwen3": 16, "glm4": 4, "deepseek7b": 4}
SOURCE_ROLES = ("concept_a", "concept_b")
OPERATIONS = ("none", "query_swap", "wrong_site_swap")
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
        (len(rows), len(SOURCE_ROLES), protocol.MAX_SOURCE_SPAN),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    pre_positions = torch.empty(len(rows), dtype=torch.long)
    for row_slot, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[row_slot, :len(values)] = values
        attention_mask[row_slot, :len(values)] = 1
        for role_slot, role in enumerate(SOURCE_ROLES):
            start, end = (
                int(value) for value in row["anchor_spans"][role]
            )
            span = list(range(start, end + 1))
            if len(span) > protocol.MAX_SOURCE_SPAN:
                raise RuntimeError(f"{role} span budget exceeded")
            positions[
                row_slot, role_slot, :len(span)
            ] = torch.tensor(span, dtype=torch.long)
            masks[row_slot, role_slot, :len(span)] = True
        pre_positions[row_slot] = int(
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


class SourceStateCapture:
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
        self.count = 0
        self.handle = None

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
        self.count = 0

    def _hook(self, module, args, output):
        if (
            self.positions is None
            or self.masks is None
            or self.local_indices is None
        ):
            raise RuntimeError("source state capture context missing")
        hidden = output_tensor(output)
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        batch = batch[:, None, None].expand_as(positions)
        values = hidden[batch, positions, :].clone()
        values = values.masked_fill(~masks[..., None], 0)
        self.cache[self.local_indices, :, :, :] = values.to(
            "cpu", dtype=torch.float16
        ).numpy()
        self.count += 1
        return output

    def register(self) -> None:
        self.handle = self.layer.register_forward_hook(self._hook)

    def end(self) -> None:
        if self.count != 1:
            raise RuntimeError(
                f"source state capture count drift: {self.count}"
            )
        self.positions = None
        self.masks = None
        self.local_indices = None

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ReceiverSwap:
    def __init__(
        self,
        layer: Any,
        response_norms: np.memmap,
    ):
        self.layer = layer
        self.response_norms = response_norms
        self.operation = ""
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None
        self.condition_slot = -1
        self.operation_slot = -1
        self.count = 0
        self.handle = None

    def begin(
        self,
        operation: str,
        positions: torch.Tensor,
        masks: torch.Tensor,
        target_indices: np.ndarray,
        condition_slot: int,
        operation_slot: int,
    ) -> None:
        if len(positions) != 2 * len(target_indices):
            raise RuntimeError("receiver pair batch drift")
        self.operation = operation
        self.positions = positions
        self.masks = masks
        self.target_indices = target_indices
        self.condition_slot = condition_slot
        self.operation_slot = operation_slot
        self.count = 0

    def _hook(self, module, args, output):
        if (
            self.positions is None
            or self.masks is None
            or self.target_indices is None
        ):
            raise RuntimeError("receiver swap context missing")
        hidden = output_tensor(output)
        positions = self.positions.to(hidden.device)
        masks = self.masks.to(hidden.device)
        even = torch.arange(
            0, hidden.shape[0], 2, device=hidden.device
        )
        odd = even + 1
        difference_sq = torch.zeros(
            len(even), device=hidden.device, dtype=torch.float32
        )
        token_count = torch.zeros_like(difference_sq)
        patched = hidden.clone()
        for span_slot in range(protocol.MAX_RECEIVER_SPAN):
            active = torch.where(masks[0::2, span_slot])[0]
            if len(active) == 0:
                continue
            even_rows = even[active]
            odd_rows = odd[active]
            even_pos = positions[even_rows, span_slot]
            odd_pos = positions[odd_rows, span_slot]
            even_value = hidden[even_rows, even_pos, :].clone()
            odd_value = hidden[odd_rows, odd_pos, :].clone()
            difference_sq[active] += torch.sum(
                (even_value.float() - odd_value.float()) ** 2,
                dim=-1,
            )
            token_count[active] += 1.0
            if self.operation != "none":
                patched[even_rows, even_pos, :] = odd_value
                patched[odd_rows, odd_pos, :] = even_value
        normalized = torch.sqrt(
            difference_sq / torch.clamp(token_count, min=1.0)
        )
        self.response_norms[
            self.target_indices,
            self.condition_slot,
            self.operation_slot,
        ] = normalized.cpu().numpy()
        self.count += 1
        return (
            replace_output(output, patched)
            if self.operation != "none"
            else output
        )

    def register(self) -> None:
        self.handle = self.layer.register_forward_hook(self._hook)

    def end(self) -> None:
        if self.count != 1:
            raise RuntimeError(
                f"receiver swap count drift: {self.count}"
            )
        self.operation = ""
        self.positions = None
        self.masks = None
        self.target_indices = None
        self.condition_slot = -1
        self.operation_slot = -1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def make_paired_batch(
    target_rows: list[dict[str, Any]],
    condition: str,
    operation: str,
    targets_by_index: dict[int, dict[str, Any]],
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
    donor_specs = []
    for target in target_rows:
        row = cases[int(target["target_case_index"])]
        model_rows.extend((row, row))
        donor_specs.append(
            protocol.donor_spec(target, condition, targets_by_index)
        )
    (
        ids,
        attention_mask,
        _,
        _,
        pre_positions,
        _,
    ) = make_clean_batch(
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
    receiver_positions = torch.zeros(
        (len(model_rows), protocol.MAX_RECEIVER_SPAN), dtype=torch.long
    )
    receiver_masks = torch.zeros_like(
        receiver_positions, dtype=torch.bool
    )
    source_norms = np.zeros(len(target_rows), dtype=np.float32)

    receiver_site = (
        protocol.WRONG_RECEIVER_SITE
        if operation == "wrong_site_swap"
        else protocol.RECEIVER_SITE
    )
    for target_slot, (target, donor_spec) in enumerate(
        zip(target_rows, donor_specs)
    ):
        patched_slot = 2 * target_slot
        zero_slot = patched_slot + 1
        target_case = int(target["target_case_index"])
        donor_case, donor_target, source_site = donor_spec
        target_row = cases[target_case]
        donor_row = cases[int(donor_case)]
        target_role = protocol.semantic_role(source_site, target)
        donor_role = protocol.semantic_role(
            source_site, donor_target
        )
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
            raise RuntimeError("source span mismatch")
        target_role_slot = SOURCE_ROLES.index(target_role)
        donor_role_slot = SOURCE_ROLES.index(donor_role)
        target_value = np.asarray(
            source_cache[
                case_to_local[target_case],
                target_role_slot,
                :len(target_span),
                :,
            ],
            dtype=np.float32,
        )
        donor_value = np.asarray(
            source_cache[
                case_to_local[int(donor_case)],
                donor_role_slot,
                :len(donor_span),
                :,
            ],
            dtype=np.float32,
        )
        payload = donor_value - target_value
        source_positions[
            patched_slot, :len(target_span)
        ] = torch.tensor(target_span, dtype=torch.long)
        source_masks[patched_slot, :len(target_span)] = True
        payloads[patched_slot, :len(target_span), :] = payload
        source_norms[target_slot] = float(
            np.linalg.norm(payload) / math.sqrt(len(target_span))
        )

        receiver_role = protocol.semantic_role(
            receiver_site, target
        )
        receiver_start, receiver_end = (
            int(value)
            for value in target_row["anchor_spans"][receiver_role]
        )
        receiver_span = list(
            range(receiver_start, receiver_end + 1)
        )
        if len(receiver_span) > protocol.MAX_RECEIVER_SPAN:
            raise RuntimeError("receiver span budget exceeded")
        for row_slot in (patched_slot, zero_slot):
            receiver_positions[
                row_slot, :len(receiver_span)
            ] = torch.tensor(receiver_span, dtype=torch.long)
            receiver_masks[row_slot, :len(receiver_span)] = True

    return (
        ids,
        attention_mask,
        pre_positions,
        source_positions,
        source_masks,
        torch.from_numpy(payloads),
        receiver_positions,
        receiver_masks,
        np.asarray(
            [int(row["confirmation_index"]) for row in target_rows],
            dtype=np.int64,
        ),
        source_norms,
    )


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
    return values[rows, cross_index] - values[rows, target_index]


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


def model_metrics(
    paired_logits: np.ndarray,
    targets: list[dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    condition_slot = {
        value: index
        for index, value in enumerate(protocol.SOURCE_CONDITIONS)
    }
    operation_slot = {
        value: index for index, value in enumerate(OPERATIONS)
    }
    cross = condition_slot["cross_selected"]
    none = operation_slot["none"]
    query = operation_slot["query_swap"]
    wrong = operation_slot["wrong_site_swap"]

    source_margin = margin_values(
        paired_logits[:, cross, none, 0, :], targets
    )
    zero_margin = margin_values(
        paired_logits[:, cross, none, 1, :], targets
    )
    reset_margin = margin_values(
        paired_logits[:, cross, query, 0, :], targets
    )
    replay_margin = margin_values(
        paired_logits[:, cross, query, 1, :], targets
    )
    wrong_reset_margin = margin_values(
        paired_logits[:, cross, wrong, 0, :], targets
    )
    wrong_replay_margin = margin_values(
        paired_logits[:, cross, wrong, 1, :], targets
    )

    source_shift = source_margin - zero_margin
    reset_shift = reset_margin - zero_margin
    replay_shift = replay_margin - zero_margin
    wrong_reset_shift = wrong_reset_margin - zero_margin
    wrong_replay_shift = wrong_replay_margin - zero_margin
    query_blocked = source_shift - reset_shift
    wrong_blocked = source_shift - wrong_reset_shift
    query_minus_wrong = query_blocked - wrong_blocked
    mediation_fraction = normalized_ratio(query_blocked, source_shift)
    replay_recovery = normalized_ratio(replay_shift, source_shift)

    controls = {}
    for condition in protocol.SOURCE_CONDITIONS:
        slot = condition_slot[condition]
        source = margin_values(
            paired_logits[:, slot, none, 0, :], targets
        )
        zero = margin_values(
            paired_logits[:, slot, none, 1, :], targets
        )
        controls[condition] = tools.scalar_summary(source - zero)

    summaries = {
        "source_shift": tools.scalar_summary(source_shift),
        "query_reset_shift": tools.scalar_summary(reset_shift),
        "query_replay_shift": tools.scalar_summary(replay_shift),
        "wrong_reset_shift": tools.scalar_summary(wrong_reset_shift),
        "wrong_replay_shift": tools.scalar_summary(wrong_replay_shift),
        "query_blocked_amount": tools.scalar_summary(query_blocked),
        "wrong_blocked_amount": tools.scalar_summary(wrong_blocked),
        "query_minus_wrong_blocked": tools.scalar_summary(
            query_minus_wrong
        ),
        "query_mediation_fraction": tools.scalar_summary(
            mediation_fraction
        ),
        "query_replay_recovery": tools.scalar_summary(replay_recovery),
        "source_control_shifts": controls,
    }
    gate = prereg["mediation_gate"]
    passed = (
        summaries["source_shift"]["median"]
        >= gate["source_shift_median_min"]
        and summaries["source_shift"]["positive_rate"]
        >= gate["source_positive_rate_min"]
        and summaries["query_blocked_amount"]["median"]
        >= gate["query_blocked_amount_median_min"]
        and summaries["query_blocked_amount"]["positive_rate"]
        >= gate["query_blocked_positive_rate_min"]
        and summaries["query_mediation_fraction"]["median"]
        >= gate["query_mediation_fraction_median_min"]
        and summaries["query_minus_wrong_blocked"]["median"]
        >= gate["query_minus_wrong_blocked_median_min"]
        and summaries["query_replay_shift"]["median"]
        >= gate["query_replay_shift_median_min"]
        and summaries["query_replay_shift"]["positive_rate"]
        >= gate["query_replay_positive_rate_min"]
        and summaries["query_replay_recovery"]["median"]
        >= gate["query_replay_recovery_median_min"]
    )
    summaries["mediation_gate_passed"] = bool(passed)
    return summaries


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1045 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    targets_by_index = {
        int(row["target_index"]): row for row in targets
    }
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    cases = {int(row["case_index"]): row for row in cases_list}
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    source_depth = int(
        prereg["model_depths"][model_name]["source_depth"]
    )
    receiver_depth = int(
        prereg["model_depths"][model_name]["receiver_depth"]
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
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )

        source_cache = np.lib.format.open_memmap(
            atlas_dir / "source_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(SOURCE_ROLES),
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
        paired_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_CONDITIONS),
                len(OPERATIONS),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        receiver_response_norms = np.lib.format.open_memmap(
            atlas_dir / "receiver_response_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_CONDITIONS),
                len(OPERATIONS),
            ),
        )
        source_payload_norms = np.lib.format.open_memmap(
            atlas_dir / "source_payload_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.SOURCE_CONDITIONS),
                len(OPERATIONS),
            ),
        )
        for array in (
            source_cache,
            clean_logits,
            paired_logits,
            receiver_response_norms,
            source_payload_norms,
        ):
            array[:] = np.nan

        capture = SourceStateCapture(
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
                clean_logits[local] = selected.detach().cpu().numpy()
                del output, logits, selected
        finally:
            capture.close()
        source_cache.flush()
        clean_logits.flush()

        source_patch = tools.SourcePatch(layers[source_depth - 1])
        receiver_swap = ReceiverSwap(
            layers[receiver_depth - 1], receiver_response_norms
        )
        source_patch.register()
        receiver_swap.register()
        try:
            for condition_slot, condition in enumerate(
                protocol.SOURCE_CONDITIONS
            ):
                for operation in protocol.OPERATIONS_BY_CONDITION[
                    condition
                ]:
                    operation_slot = OPERATIONS.index(operation)
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
                            receiver_positions,
                            receiver_masks,
                            target_indices,
                            source_norm,
                        ) = make_paired_batch(
                            target_batch,
                            condition,
                            operation,
                            targets_by_index,
                            cases,
                            case_to_local,
                            source_cache,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        source_patch.begin(
                            source_positions, source_masks, payloads
                        )
                        receiver_swap.begin(
                            operation,
                            receiver_positions,
                            receiver_masks,
                            target_indices,
                            condition_slot,
                            operation_slot,
                        )
                        with torch.inference_mode():
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                use_cache=False,
                                return_dict=True,
                            )
                        receiver_swap.end()
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
                        pair = selected.reshape(
                            len(target_batch),
                            2,
                            len(protocol.material.FAMILIES),
                        )
                        paired_logits[
                            target_indices,
                            condition_slot,
                            operation_slot,
                            :,
                            :,
                        ] = pair.detach().cpu().numpy()
                        source_payload_norms[
                            target_indices,
                            condition_slot,
                            operation_slot,
                        ] = source_norm
                        del output, logits, selected, pair
        finally:
            receiver_swap.close()
            source_patch.close()
        for array in (
            paired_logits,
            receiver_response_norms,
            source_payload_norms,
        ):
            array.flush()

        metrics = model_metrics(paired_logits, targets, prereg)
        expected = np.asarray(
            [int(row["expected_index"]) for row in cases_list],
            dtype=np.int64,
        )
        clean_values = np.asarray(clean_logits, dtype=np.float32)
        finite = np.all(np.isfinite(clean_values), axis=-1)
        prediction = np.full(len(cases_list), -1, dtype=np.int64)
        prediction[finite] = np.argmax(clean_values[finite], axis=-1)
        summary = {
            "schema_version": "phase1045_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
            },
            "source_depth": source_depth,
            "receiver_depth": receiver_depth,
            "behavior": {
                "case_count": len(cases_list),
                "finite_row_rate": float(np.mean(finite)),
                "candidate_accuracy": float(
                    np.mean(prediction == expected)
                ),
            },
            "source_cache_finite": tools.finite_summary(source_cache),
            "paired_logits_finite": tools.finite_summary(paired_logits),
            "receiver_response_finite": tools.finite_summary(
                receiver_response_norms[
                    np.isfinite(receiver_response_norms)
                ]
            ),
            "metrics": metrics,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "behavior_accuracy": summary["behavior"][
                "candidate_accuracy"
            ],
            "source_shift": metrics["source_shift"]["median"],
            "query_blocked": metrics[
                "query_blocked_amount"
            ]["median"],
            "query_replay": metrics[
                "query_replay_shift"
            ]["median"],
            "query_minus_wrong": metrics[
                "query_minus_wrong_blocked"
            ]["median"],
            "mediation_gate_passed": metrics[
                "mediation_gate_passed"
            ],
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
