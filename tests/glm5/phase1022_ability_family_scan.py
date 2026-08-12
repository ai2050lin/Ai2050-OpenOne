#!/usr/bin/env python3
"""Map Phase1022 ability, semantic-family, and generation-time differences.

The scan records residual stream, whole attention output, MLP output, real
pre-o-projection attention heads, and KV-cache states.  All comparisons stay
inside one model.  Cross-model analysis is limited to normalized scalar
profiles at relative depth.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model
from phase1014_bf16_precision_confirmation import load_bf16
from phase1018_language_pattern_scan import (
    BatchRoleHeadCapture,
    BatchRoleStateCapture,
    event_definitions,
)
import phase1022_ability_family_protocol as protocol


PAIR_BATCH = {"qwen3": 8, "glm4": 2, "deepseek7b": 2}
CASE_BATCH = {"qwen3": 16, "glm4": 4, "deepseek7b": 4}
EPSILON = 1e-12
ABILITY_TYPES = ("success_failure", "success_success", "failure_failure")
TIMELINE_CONTRASTS = (
    "pre_from_source",
    "output1_from_pre",
    "output2_from_pre",
    "outputlast_from_pre",
)
CROSS_MODEL_GROUPS = (
    "qwen_glm_success_ds_failure",
    "all_success",
    "all_failure",
)
MAX_CROSS_CASES_PER_PANEL = 48


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def normalized_directions(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values.astype(np.float64), axis=-1, keepdims=True)
    result = np.zeros_like(values, dtype=np.float32)
    np.divide(values, norms, out=result, where=norms > EPSILON)
    return result


def direction_consistency(
    unit_sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    squared = np.einsum(
        "...d,...d->...",
        unit_sums.astype(np.float64, copy=False),
        unit_sums.astype(np.float64, copy=False),
    )
    result = np.full(counts.shape, np.nan, dtype=np.float32)
    valid = counts >= 2
    result[valid] = (
        (squared[valid] - counts[valid])
        / (counts[valid] * (counts[valid] - 1.0))
    ).astype(np.float32)
    return result


class DeltaAccumulator:
    def __init__(self, role_count: int, event_count: int, width: int):
        shape = (role_count, event_count)
        self.sum_delta = np.zeros((*shape, width), dtype=np.float32)
        self.sum_unit = np.zeros((*shape, width), dtype=np.float32)
        self.magnitude_sum = np.zeros(shape, dtype=np.float64)
        self.magnitude_sq_sum = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.int32)

    def add(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        valid_roles: torch.Tensor,
    ) -> None:
        left = left.float().cpu()
        right = right.float().cpu()
        valid_roles = valid_roles.bool().cpu()
        delta = left - right
        delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        scale = 0.5 * (
            torch.linalg.vector_norm(left, dim=-1)
            + torch.linalg.vector_norm(right, dim=-1)
        )
        normalized_magnitude = delta_norm / torch.clamp(scale, min=EPSILON)
        valid = valid_roles[:, :, None] & (delta_norm > EPSILON)
        masked_delta = torch.where(valid[..., None], delta, 0.0)
        unit = torch.where(
            valid[..., None],
            delta / torch.clamp(delta_norm[..., None], min=EPSILON),
            0.0,
        )
        magnitude = torch.where(valid, normalized_magnitude, 0.0)
        self.sum_delta += masked_delta.sum(dim=0).numpy()
        self.sum_unit += unit.sum(dim=0).numpy()
        self.magnitude_sum += magnitude.sum(dim=0).double().numpy()
        self.magnitude_sq_sum += (
            magnitude.double().square().sum(dim=0).numpy()
        )
        self.count += valid.sum(dim=0).int().numpy()

    def arrays(self) -> dict[str, np.ndarray]:
        count_float = np.maximum(self.count.astype(np.float64), 1.0)
        mean = self.magnitude_sum / count_float
        variance = np.maximum(
            self.magnitude_sq_sum / count_float - mean * mean,
            0.0,
        )
        mean[self.count == 0] = np.nan
        variance[self.count == 0] = np.nan
        return {
            "mean_normalized_magnitude": mean.astype(np.float32),
            "sd_normalized_magnitude": np.sqrt(variance).astype(np.float32),
            "direction_consistency": direction_consistency(
                self.sum_unit, self.count
            ),
            "mean_direction": normalized_directions(
                self.sum_delta
            ).astype(np.float16),
            "count": self.count,
        }


def prefix_arrays(
    output: dict[str, np.ndarray],
    prefix: str,
    arrays: dict[str, np.ndarray],
) -> None:
    for name, value in arrays.items():
        output[f"{prefix}_{name}"] = value


def extended_case(
    case: dict[str, Any],
    behavior: dict[str, Any],
) -> dict[str, Any]:
    generated = [
        int(value) for value in behavior["generated_token_ids"]
    ]
    if not generated:
        raise RuntimeError(f"empty generated continuation: {case['case_key']}")
    prompt_length = len(case["input_ids"])
    positions = dict(case["role_positions"])
    positions.update({
        "output_1": prompt_length,
        "output_2": (
            prompt_length + 1 if len(generated) >= 2 else prompt_length
        ),
        "output_last": prompt_length + len(generated) - 1,
    })
    validity = {role: True for role in protocol.INTERNAL_ROLES}
    validity["output_2"] = len(generated) >= 2
    return {
        "case_key": case["case_key"],
        "input_ids": [*case["input_ids"], *generated],
        "role_positions": positions,
        "role_validity": validity,
        "generated_token_count": len(generated),
    }


def pad_cases(
    cases: list[dict[str, Any]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(case["input_ids"]) for case in cases)
    input_ids = torch.full(
        (len(cases), width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    positions = []
    validity = []
    for index, case in enumerate(cases):
        values = torch.tensor(
            case["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, :len(values)] = values
        attention_mask[index, :len(values)] = 1
        positions.append([
            int(case["role_positions"][role])
            for role in protocol.INTERNAL_ROLES
        ])
        validity.append([
            bool(case["role_validity"][role])
            for role in protocol.INTERNAL_ROLES
        ])
    return (
        input_ids,
        attention_mask,
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(validity, dtype=torch.bool),
    )


def legacy_cache(past_key_values: Any) -> Any:
    if past_key_values is None:
        raise RuntimeError("model did not return past_key_values")
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    if hasattr(past_key_values, "layers"):
        rows = []
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", getattr(layer, "key_cache", None))
            values = getattr(
                layer, "values", getattr(layer, "value_cache", None)
            )
            rows.append((keys, values))
        return rows
    raise RuntimeError(
        f"unsupported cache type: {type(past_key_values).__name__}"
    )


def canonical_cache_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim < 3:
        raise RuntimeError("cache entry is not a rank >=3 tensor")
    batch_axes = [
        axis for axis, size in enumerate(value.shape)
        if int(size) == batch_size
    ]
    batch_axis = 0 if value.shape[0] == batch_size else (
        batch_axes[0] if batch_axes else -1
    )
    if batch_axis < 0:
        raise RuntimeError(f"cannot find cache batch axis: {value.shape}")
    if batch_axis != 0:
        value = value.movedim(batch_axis, 0)

    if value.ndim >= 3 and int(value.shape[2]) == sequence_length:
        sequence_axis = 2
    elif int(value.shape[1]) == sequence_length:
        sequence_axis = 1
    else:
        candidates = [
            axis for axis in range(1, value.ndim)
            if int(value.shape[axis]) == sequence_length
        ]
        if not candidates:
            raise RuntimeError(
                f"cannot find cache sequence axis: {value.shape}, "
                f"sequence={sequence_length}"
            )
        sequence_axis = candidates[0]
    if sequence_axis != 1:
        value = value.movedim(sequence_axis, 1)
    return value


def cache_role_values(
    past_key_values: Any,
    positions: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = legacy_cache(past_key_values)
    key_rows = []
    value_rows = []
    for layer in cache:
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            key, value = layer[0], layer[1]
        else:
            key = getattr(layer, "keys", getattr(layer, "key_cache", None))
            value = getattr(
                layer, "values", getattr(layer, "value_cache", None)
            )
        key = canonical_cache_tensor(
            key,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        value = canonical_cache_tensor(
            value,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        local_positions = positions.to(key.device)
        batch = torch.arange(batch_size, device=key.device)[:, None]
        selected_key = key[batch, local_positions]
        local_positions = positions.to(value.device)
        batch = torch.arange(batch_size, device=value.device)[:, None]
        selected_value = value[batch, local_positions]
        key_rows.append(selected_key.reshape(
            batch_size, positions.shape[1], -1
        ).float().cpu())
        value_rows.append(selected_value.reshape(
            batch_size, positions.shape[1], -1
        ).float().cpu())
    return torch.stack(key_rows, dim=2), torch.stack(value_rows, dim=2)


def captured_values(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
    state_capture: BatchRoleStateCapture,
    head_capture: BatchRoleHeadCapture,
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    state_capture.begin(positions)
    head_capture.begin(positions)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    state_capture.validate()
    head_capture.validate()
    whole = torch.stack([
        state_capture.values[key].float().cpu() for key in whole_keys
    ], dim=2)
    head = torch.stack([
        head_capture.values[depth][:, :, head_index].float().cpu()
        for depth, head_index in head_keys
    ], dim=2)
    key, value = cache_role_values(
        output.past_key_values,
        positions.cpu(),
        batch_size=input_ids.shape[0],
        sequence_length=input_ids.shape[1],
    )
    del output
    state_capture.values = {}
    head_capture.values = {}
    return {
        "whole": whole,
        "head": head,
        "key": key,
        "value": value,
    }


def pair_values(
    values: dict[str, torch.Tensor],
    pair_count: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for name, tensor in values.items():
        tensor = tensor.reshape(pair_count, 2, *tensor.shape[1:])
        result[name] = (tensor[:, 0], tensor[:, 1])
    return result


def save_accumulators(
    path: Path,
    accumulators: dict[str, DeltaAccumulator],
    *,
    role_names: tuple[str, ...],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "role_names": np.asarray(role_names),
    }
    for name, accumulator in accumulators.items():
        prefix_arrays(arrays, name, accumulator.arrays())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def event_metadata(
    n_layers: int,
    physical_heads: int,
    whole_events: list[dict[str, Any]],
    kv_width: int,
) -> dict[str, Any]:
    whole_ids = [row["event_id"] for row in whole_events if row["head"] is None]
    head_ids = [row["event_id"] for row in whole_events if row["head"] is not None]
    return {
        "whole_event_ids": whole_ids,
        "head_event_ids": head_ids,
        "kv_event_ids": [
            f"kv_cache.d{depth:02d}" for depth in range(1, n_layers + 1)
        ],
        "n_layers": n_layers,
        "physical_heads": physical_heads,
        "kv_width": kv_width,
    }


def run_pair_panel(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    pairs: list[dict[str, Any]],
    case_by_key: dict[str, dict[str, Any]],
    behavior_by_key: dict[str, dict[str, Any]],
    state_capture: BatchRoleStateCapture,
    head_capture: BatchRoleHeadCapture,
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
    output_path: Path,
) -> dict[str, Any]:
    accumulators: dict[str, DeltaAccumulator] | None = None
    forward_count = 0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    started = time.time()
    for batch_pairs in chunks(pairs, PAIR_BATCH[model_name]):
        flat_cases = []
        validity = []
        for pair in batch_pairs:
            left = extended_case(
                case_by_key[pair["left_case_key"]],
                behavior_by_key[pair["left_case_key"]],
            )
            right = extended_case(
                case_by_key[pair["right_case_key"]],
                behavior_by_key[pair["right_case_key"]],
            )
            flat_cases.extend((left, right))
            validity.append([
                left["role_validity"][role]
                and right["role_validity"][role]
                for role in protocol.INTERNAL_ROLES
            ])
        input_ids, attention_mask, positions, _ = pad_cases(
            flat_cases, int(pad_id), device
        )
        values = captured_values(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
            state_capture=state_capture,
            head_capture=head_capture,
            whole_keys=whole_keys,
            head_keys=head_keys,
        )
        paired = pair_values(values, len(batch_pairs))
        if accumulators is None:
            accumulators = {
                name: DeltaAccumulator(
                    len(protocol.INTERNAL_ROLES),
                    tensor[0].shape[2],
                    tensor[0].shape[3],
                )
                for name, tensor in paired.items()
            }
        valid_tensor = torch.tensor(validity, dtype=torch.bool)
        for name, (left, right) in paired.items():
            accumulators[name].add(left, right, valid_tensor)
        forward_count += 1
        del (
            input_ids,
            attention_mask,
            positions,
            values,
            paired,
            flat_cases,
            validity,
            valid_tensor,
        )
    if accumulators is None:
        raise RuntimeError("empty pair panel")
    save_accumulators(
        output_path,
        accumulators,
        role_names=protocol.INTERNAL_ROLES,
    )
    return {
        "pair_count": len(pairs),
        "batched_forward_count": forward_count,
        "elapsed_seconds": time.time() - started,
        "mean_source_token_gap": float(np.mean([
            row["source_token_gap"] for row in pairs
        ])),
        "mean_generated_token_gap": float(np.mean([
            row["generated_token_gap"] for row in pairs
        ])),
        "mean_prompt_token_gap": float(np.mean([
            row["prompt_token_gap"] for row in pairs
        ])),
    }


def round_robin_cases(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["category"], f"{row['source_language']}_{row['target_language']}")].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: (row["template"], row["case_key"]))
    keys = sorted(grouped)
    result = []
    cursor = 0
    while len(result) < limit:
        added = False
        for key in keys:
            if cursor < len(grouped[key]) and len(result) < limit:
                result.append(grouped[key][cursor])
                added = True
        if not added:
            break
        cursor += 1
    return result


def run_timeline_panel(
    *,
    model,
    tokenizer,
    device,
    model_name: str,
    cases: list[dict[str, Any]],
    case_by_key: dict[str, dict[str, Any]],
    behavior_by_key: dict[str, dict[str, Any]],
    state_capture: BatchRoleStateCapture,
    head_capture: BatchRoleHeadCapture,
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
    output_path: Path,
) -> dict[str, Any]:
    accumulators: dict[str, DeltaAccumulator] | None = None
    forward_count = 0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    role_index = {
        role: index for index, role in enumerate(protocol.INTERNAL_ROLES)
    }
    left_roles = (
        role_index["pre_output"],
        role_index["output_1"],
        role_index["output_2"],
        role_index["output_last"],
    )
    right_roles = (
        role_index["source_end"],
        role_index["pre_output"],
        role_index["pre_output"],
        role_index["pre_output"],
    )
    started = time.time()
    for batch_rows in chunks(cases, CASE_BATCH[model_name]):
        extended = [
            extended_case(
                case_by_key[row["case_key"]],
                behavior_by_key[row["case_key"]],
            )
            for row in batch_rows
        ]
        input_ids, attention_mask, positions, validity = pad_cases(
            extended, int(pad_id), device
        )
        values = captured_values(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
            state_capture=state_capture,
            head_capture=head_capture,
            whole_keys=whole_keys,
            head_keys=head_keys,
        )
        if accumulators is None:
            accumulators = {
                name: DeltaAccumulator(
                    len(TIMELINE_CONTRASTS),
                    tensor.shape[2],
                    tensor.shape[3],
                )
                for name, tensor in values.items()
            }
        selected_validity = torch.stack([
            validity[:, role_index["pre_output"]]
            & validity[:, role_index["source_end"]],
            validity[:, role_index["output_1"]],
            validity[:, role_index["output_2"]],
            validity[:, role_index["output_last"]],
        ], dim=1)
        for name, tensor in values.items():
            left = torch.stack(
                [tensor[:, index] for index in left_roles], dim=1
            )
            right = torch.stack(
                [tensor[:, index] for index in right_roles], dim=1
            )
            accumulators[name].add(left, right, selected_validity)
        forward_count += 1
        del (
            extended,
            input_ids,
            attention_mask,
            positions,
            validity,
            values,
            selected_validity,
        )
    if accumulators is None:
        raise RuntimeError("empty timeline panel")
    save_accumulators(
        output_path,
        accumulators,
        role_names=TIMELINE_CONTRASTS,
    )
    return {
        "case_count": len(cases),
        "batched_forward_count": forward_count,
        "elapsed_seconds": time.time() - started,
    }


def run_model(model_name: str, *, resume: bool) -> dict[str, Any]:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    pairing = protocol.read_json(
        protocol.OUT_ROOT / "pairing" / "summary.json"
    )
    if pairing["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("pairing/protocol digest mismatch")
    if not pairing["translation_internal_authorized"]:
        raise RuntimeError("translation internal scan was not authorized")

    protocol_cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    behavior_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "behavior" / model_name / "formal.jsonl"
    )
    ability_pairs = protocol.read_jsonl(
        protocol.OUT_ROOT / "pairing" / f"ability_pairs.{model_name}.jsonl"
    )
    family_pairs = protocol.read_jsonl(
        protocol.OUT_ROOT / "pairing" / f"family_pairs.{model_name}.jsonl"
    )
    cross_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "pairing" / "cross_model_cases.jsonl"
    )
    case_by_key = {row["case_key"]: row for row in protocol_cases}
    behavior_by_key = {row["case_key"]: row for row in behavior_rows}

    output_root = protocol.OUT_ROOT / "internal_scan" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    state_capture = head_capture = None
    panel_summaries = []
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        info = get_model_info(model, model_name)
        layers = get_layers(model)
        physical_heads = int(model.config.num_attention_heads)
        events, whole_keys, head_keys = event_definitions(
            int(info.n_layers), physical_heads
        )
        state_capture = BatchRoleStateCapture(model, layers)
        head_capture = BatchRoleHeadCapture(layers, physical_heads)
        state_capture.register()
        head_capture.register()

        # A one-case probe freezes the actual cache width and verifies that KV
        # extraction works for this architecture before the formal panels.
        probe_behavior = next(
            row for row in behavior_rows
            if row["family"] == "translation"
            and row["generated_token_count"] > 0
        )
        probe_case = extended_case(
            case_by_key[probe_behavior["case_key"]], probe_behavior
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        probe_inputs = pad_cases([probe_case], int(pad_id), device)
        probe_values = captured_values(
            model=model,
            input_ids=probe_inputs[0],
            attention_mask=probe_inputs[1],
            positions=probe_inputs[2],
            state_capture=state_capture,
            head_capture=head_capture,
            whole_keys=whole_keys,
            head_keys=head_keys,
        )
        kv_width = int(probe_values["key"].shape[-1])
        value_width = int(probe_values["value"].shape[-1])
        if kv_width != value_width:
            raise RuntimeError("key/value cache widths differ")
        metadata = event_metadata(
            int(info.n_layers),
            physical_heads,
            events,
            kv_width,
        )
        protocol.write_json(output_root / "events.json", metadata)
        del probe_inputs, probe_values, probe_case

        grouped_ability: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in ability_pairs:
            grouped_ability[(row["pair_type"], row["split"])].append(row)
        for pair_type in ABILITY_TYPES:
            for split in protocol.SPLITS:
                rows = grouped_ability.get((pair_type, split), [])
                if not rows:
                    continue
                path = output_root / "ability" / pair_type / f"{split}.npz"
                summary_path = path.with_suffix(".summary.json")
                if resume and path.exists() and summary_path.exists():
                    summary = protocol.read_json(summary_path)
                else:
                    summary = run_pair_panel(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        model_name=model_name,
                        pairs=rows,
                        case_by_key=case_by_key,
                        behavior_by_key=behavior_by_key,
                        state_capture=state_capture,
                        head_capture=head_capture,
                        whole_keys=whole_keys,
                        head_keys=head_keys,
                        output_path=path,
                    )
                    summary.update({
                        "panel_kind": "ability_pair",
                        "pair_type": pair_type,
                        "split": split,
                    })
                    protocol.write_json(summary_path, summary)
                panel_summaries.append(summary)
                print(
                    f"[scan] {model_name} ability/{pair_type}/{split} "
                    f"pairs={summary['pair_count']}",
                    flush=True,
                )

        grouped_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in family_pairs:
            grouped_family[(row["split"], row["category"])].append(row)
        for split in protocol.SPLITS:
            for category in protocol.CATEGORIES:
                rows = grouped_family.get((split, category), [])
                if not rows:
                    continue
                path = output_root / "family" / category / f"{split}.npz"
                summary_path = path.with_suffix(".summary.json")
                if resume and path.exists() and summary_path.exists():
                    summary = protocol.read_json(summary_path)
                else:
                    summary = run_pair_panel(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        model_name=model_name,
                        pairs=rows,
                        case_by_key=case_by_key,
                        behavior_by_key=behavior_by_key,
                        state_capture=state_capture,
                        head_capture=head_capture,
                        whole_keys=whole_keys,
                        head_keys=head_keys,
                        output_path=path,
                    )
                    summary.update({
                        "panel_kind": "family_vs_other",
                        "category": category,
                        "split": split,
                    })
                    protocol.write_json(summary_path, summary)
                panel_summaries.append(summary)
                print(
                    f"[scan] {model_name} family/{category}/{split} "
                    f"pairs={summary['pair_count']}",
                    flush=True,
                )

        cross_by_key = {
            row["case_key"]: row for row in cross_rows
            if row["group"] in CROSS_MODEL_GROUPS
        }
        model_cross_rows = [
            {
                **cross_by_key[key],
                **{
                    field: behavior_by_key[key][field]
                    for field in (
                        "source_language",
                        "target_language",
                        "category",
                        "template",
                    )
                },
            }
            for key in sorted(cross_by_key)
            if key in behavior_by_key
            and behavior_by_key[key]["generated_token_count"] > 0
        ]
        grouped_cross: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in model_cross_rows:
            grouped_cross[(row["group"], row["split"])].append(row)
        for group in CROSS_MODEL_GROUPS:
            for split in protocol.SPLITS:
                rows = round_robin_cases(
                    grouped_cross.get((group, split), []),
                    MAX_CROSS_CASES_PER_PANEL,
                )
                if not rows:
                    continue
                path = output_root / "timeline" / group / f"{split}.npz"
                summary_path = path.with_suffix(".summary.json")
                if resume and path.exists() and summary_path.exists():
                    summary = protocol.read_json(summary_path)
                else:
                    summary = run_timeline_panel(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        model_name=model_name,
                        cases=rows,
                        case_by_key=case_by_key,
                        behavior_by_key=behavior_by_key,
                        state_capture=state_capture,
                        head_capture=head_capture,
                        whole_keys=whole_keys,
                        head_keys=head_keys,
                        output_path=path,
                    )
                    summary.update({
                        "panel_kind": "cross_model_timeline",
                        "group": group,
                        "split": split,
                    })
                    protocol.write_json(summary_path, summary)
                panel_summaries.append(summary)
                print(
                    f"[scan] {model_name} timeline/{group}/{split} "
                    f"cases={summary['case_count']}",
                    flush=True,
                )

        summary = {
            "schema_version": "phase1022_internal_scan_model.v1",
            "phase": protocol.PHASE,
            "protocol_revision": protocol.PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "pairing_digest": pairing["pairing_digest"],
            "model": model_name,
            "precision": "bf16",
            "quantization": "none",
            "placement": placement,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "physical_heads": physical_heads,
                "head_width": int(
                    layers[0].self_attn.o_proj.in_features // physical_heads
                ),
                "kv_width": kv_width,
            },
            "panel_count": len(panel_summaries),
            "pair_count": int(sum(
                row.get("pair_count", 0) for row in panel_summaries
            )),
            "timeline_case_count": int(sum(
                row.get("case_count", 0) for row in panel_summaries
            )),
            "batched_forward_count": int(sum(
                row["batched_forward_count"] for row in panel_summaries
            ) + 1),
            "panel_kinds": dict(Counter(
                row["panel_kind"] for row in panel_summaries
            )),
            "elapsed_seconds": time.time() - started,
            "claim_limits": [
                "All differences are observational measurements.",
                "Success/failure is not a randomized intervention.",
                "Generated-token roles contain consequences of generation.",
                "Cross-model hidden vectors and neuron IDs are not aligned.",
                "Physical reuse does not establish functional identity.",
            ],
        }
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if head_capture is not None:
            head_capture.close()
        if state_capture is not None:
            state_capture.close()
        if model is not None:
            release_model(model)
        del model, tokenizer, device, state_capture, head_capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_model(args.model, resume=args.resume)


if __name__ == "__main__":
    main()
