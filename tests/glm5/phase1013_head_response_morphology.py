#!/usr/bin/env python3
"""Map per-head remote responses inside Phase1012-frozen broad regions."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1011_native_semantic_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT as PHASE1011_ROOT,
    OUTPUT_MODES,
    PHASE as SOURCE_PROTOCOL_PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1011_native_semantic_scan import (
    OP_INDEX,
    SPLIT_INDEX,
    STATE_ORDER,
    case_tensors,
    direction_consistency,
    operation_deltas,
    operation_scales,
    stage_case,
    unit_qualification,
)


PHASE = 1013
PHASE1012_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1012_remote_receiver_tomography"
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1013_head_response_morphology"
)
CONTROL_OPERATIONS = ("E", "O", "N", "S")
TARGET_OPERATIONS = ("F", "Q", "FQ", "X")
EPSILON = 1e-12
MIN_N = 8
MIN_POOLS = 2
MIN_TEMPLATES = 2
DIRECTION_THRESHOLD = 0.50
PREVALENCE_THRESHOLD = 0.70
MAX_HEADS_PER_REGION = 5
HEAD_DIRECTION_AXES = (
    "all_units",
    "semantic_panel",
    "natural_rollout",
)


class HeadCapture:
    def __init__(
        self,
        layers,
        depths: list[int],
        head_count: int,
    ):
        self.layers = layers
        self.depths = depths
        self.head_count = head_count
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args):
            value = args[0]
            if self.positions is None:
                raise RuntimeError("positions not set")
            positions = self.positions.to(value.device)
            batch = torch.arange(
                value.shape[0], device=value.device
            )
            selected = value[batch, positions, :]
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("head width drift")
            self.values[depth] = selected.reshape(
                selected.shape[0],
                self.head_count,
                selected.shape[-1] // self.head_count,
            ).detach()
            self.counts[depth] += 1

        return hook

    def register(self) -> None:
        for depth in self.depths:
            self.handles.append(
                self.layers[depth - 1].self_attn.o_proj
                .register_forward_pre_hook(self._hook(depth))
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        missing = sorted(set(self.depths) - set(self.values))
        repeated = {
            depth: count
            for depth, count in self.counts.items()
            if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"head capture drift missing={missing} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


def selected_depth_operations(
    model_name: str,
) -> dict[int, set[str]]:
    selection = read_json(
        PHASE1012_ROOT / "discovery_selection.json"
    )
    result: dict[int, set[str]] = defaultdict(set)
    for row in selection["selections"]:
        if (
            row["model"] == model_name
            and row["component"] == "attention_output"
        ):
            result[int(row["depth"])].add(row["operation"])
    if not result:
        raise RuntimeError(f"{model_name}: no attention region")
    return result


def events(
    depths: list[int],
    head_count: int,
    n_layers: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    rows = []
    lookup = {}
    for depth in depths:
        for head in range(head_count):
            index = len(rows)
            lookup[(depth, head)] = index
            rows.append({
                "schema_version": "phase1013_head_event.v1",
                "phase": PHASE,
                "event_index": index,
                "event_id": f"attention.d{depth:02d}.h{head:02d}",
                "component": "attention_head_pre_o_proj",
                "depth": depth,
                "relative_depth": depth / max(n_layers, 1),
                "head": head,
                "receiver_role": "answer_boundary",
                "claim": "physical_head_response_only",
            })
    return rows, lookup


def scan_panel(
    *,
    model,
    device,
    capture: HeadCapture,
    pad_token_id: int,
    model_name: str,
    family: str,
    output_mode: str,
    units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
    event_rows: list[dict[str, Any]],
    event_lookup: dict[tuple[int, int], int],
    output_root: Path,
) -> dict[str, Any]:
    unit_count = len(units)
    operation_count = len(ANALYSIS_OPERATIONS)
    event_count = len(event_rows)
    raw = np.full(
        (unit_count, operation_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    normalized = np.full_like(raw, np.nan)
    qualification_arrays = {
        axis: np.zeros(
            (unit_count, operation_count), dtype=np.bool_
        )
        for axis in (
            "semantic_panel",
            "strict_teacher",
            "natural_rollout",
            "strict_rollout",
        )
    }
    qualification_arrays["all_units"] = np.ones(
        (unit_count, operation_count), dtype=np.bool_
    )
    head_dim = None
    direction_sum = None
    direction_count = np.zeros(
        (
            len(HEAD_DIRECTION_AXES),
            operation_count,
            len(SPLIT_INDEX),
            event_count,
        ),
        dtype=np.int32,
    )
    unit_rows = []
    started = time.time()
    for unit_index, unit in enumerate(units):
        qualifications = unit_qualification(
            unit, qualification_by_key
        )
        qualifications["all_units"] = np.ones(
            operation_count, dtype=np.bool_
        )
        for axis, values in qualifications.items():
            qualification_arrays[axis][unit_index] = values
        base = case_by_id[unit["case_ids"]["base"]]
        state_cases = [
            base,
            case_by_id[unit["case_ids"]["F"]],
            case_by_id[unit["case_ids"]["Q"]],
            case_by_id[unit["case_ids"]["FQ"]],
            case_by_id[unit["case_ids"]["E"]],
            case_by_id[unit["case_ids"]["O"]],
            case_by_id[unit["case_ids"]["N"]],
            case_by_id[unit["case_ids"]["S"]],
            dict(base),
        ]
        staged = [stage_case(case, "prompt") for case in state_cases]
        singleton_values: dict[
            int, list[torch.Tensor]
        ] = {
            depth: [] for depth in capture.depths
        }
        for case in staged:
            positions = torch.tensor(
                [
                    int(
                        case["scan_role_positions"][
                            "answer_boundary"
                        ]
                    )
                ],
                dtype=torch.long,
                device=device,
            )
            input_ids, attention, lengths = case_tensors(
                [case], device, pad_token_id
            )
            capture.begin(positions)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
            capture.validate()
            for depth, value in capture.values.items():
                singleton_values[depth].append(
                    value[0].detach()
                )
            del output, input_ids, attention, lengths, positions
            capture.values = {}
        split_index = SPLIT_INDEX[unit["split"]]
        for depth, state_values in singleton_values.items():
            values = torch.stack(state_values, dim=0)
            deltas = operation_deltas(values)
            scales = operation_scales(values)
            delta_stack = torch.stack(
                [
                    deltas[operation]
                    for operation in ANALYSIS_OPERATIONS
                ],
                dim=0,
            ).float()
            scale_stack = torch.stack(
                [
                    scales[operation]
                    for operation in ANALYSIS_OPERATIONS
                ],
                dim=0,
            ).float()
            raw_stack = torch.linalg.vector_norm(
                delta_stack, dim=-1
            )
            normalized_stack = raw_stack / torch.clamp(
                scale_stack, min=EPSILON
            )
            directions = delta_stack / torch.clamp(
                raw_stack[..., None], min=EPSILON
            )
            raw_cpu = raw_stack.detach().cpu().numpy()
            norm_cpu = normalized_stack.detach().cpu().numpy()
            direction_cpu = directions.detach().cpu().numpy()
            if head_dim is None:
                head_dim = int(direction_cpu.shape[-1])
                direction_sum = np.zeros(
                    (
                        len(HEAD_DIRECTION_AXES),
                        operation_count,
                        len(SPLIT_INDEX),
                        event_count,
                        head_dim,
                    ),
                    dtype=np.float32,
                )
            for head in range(capture.head_count):
                event_index = event_lookup[(depth, head)]
                raw[unit_index, :, event_index] = raw_cpu[:, head]
                normalized[
                    unit_index, :, event_index
                ] = norm_cpu[:, head]
                for operation_index in range(operation_count):
                    if raw_cpu[operation_index, head] <= EPSILON:
                        continue
                    for axis_index, axis in enumerate(
                        HEAD_DIRECTION_AXES
                    ):
                        if not qualifications[axis][operation_index]:
                            continue
                        direction_sum[
                            axis_index,
                            operation_index,
                            split_index,
                            event_index,
                        ] += direction_cpu[
                            operation_index, head
                        ].astype(np.float32, copy=False)
                        direction_count[
                            axis_index,
                            operation_index,
                            split_index,
                            event_index,
                        ] += 1
            del (
                delta_stack,
                scale_stack,
                raw_stack,
                normalized_stack,
                directions,
            )
        del singleton_values
        unit_rows.append({
            "schema_version": "phase1013_head_scan_unit.v1",
            "phase": PHASE,
            "model": model_name,
            "family": family,
            "output_mode": output_mode,
            "unit_index": unit_index,
            "unit_id": unit["unit_id"],
            "split": unit["split"],
            "template": int(unit["template"]),
            "name_pool": int(unit["name_pool"]),
            "world_index": int(unit["world_index"]),
        })
        if (unit_index + 1) % 8 == 0:
            print(
                f"[head-scan] {model_name}/{family}/"
                f"{output_mode} {unit_index + 1}/{unit_count}",
                flush=True,
            )
    if direction_sum is None:
        raise RuntimeError("head directions were not initialized")
    consistency = direction_consistency(
        direction_sum, direction_count
    )
    panel_root = output_root / family / output_mode
    panel_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_root / "response_scalars.npz",
        raw_magnitude=raw,
        normalized_magnitude=normalized,
        all_units_qualified=qualification_arrays["all_units"],
        semantic_panel_qualified=qualification_arrays[
            "semantic_panel"
        ],
        strict_teacher_qualified=qualification_arrays[
            "strict_teacher"
        ],
        natural_rollout_qualified=qualification_arrays[
            "natural_rollout"
        ],
        strict_rollout_qualified=qualification_arrays[
            "strict_rollout"
        ],
    )
    np.savez_compressed(
        panel_root / "direction_consistency.npz",
        direction_consistency=consistency,
        direction_count=direction_count,
    )
    write_jsonl(panel_root / "units.jsonl", unit_rows)
    identity = normalized[:, OP_INDEX["I"], :]
    summary = {
        "schema_version": "phase1013_head_scan_panel.v1",
        "phase": PHASE,
        "model": model_name,
        "family": family,
        "output_mode": output_mode,
        "unit_count": unit_count,
        "event_count": event_count,
        "scalar_measurement_count": int(raw.size),
        "head_dim": head_dim,
        "raw_hidden_tensors_persisted": 0,
        "state_forward_mode": "singleton_8bit",
        "direction_axes": list(HEAD_DIRECTION_AXES),
        "model_forward_count": int(unit_count * len(STATE_ORDER)),
        "identity_maximum": float(np.max(np.abs(identity))),
        "elapsed_seconds": time.time() - started,
    }
    write_json(panel_root / "summary.json", summary)
    del direction_sum
    gc.collect()
    return summary


def scan_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(PHASE1011_ROOT / "protocol" / "protocol.json")
    depth_operations = selected_depth_operations(model_name)
    depths = sorted(depth_operations)
    cases = read_jsonl(
        PHASE1011_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        PHASE1011_ROOT / "protocol" / model_name / "units.jsonl"
    )
    qualifications = read_jsonl(
        PHASE1011_ROOT
        / "behavior"
        / model_name
        / "pair_qualification.jsonl"
    )
    qualification_by_key = {
        (row["unit_id"], row["operation"]): row
        for row in qualifications
    }
    case_by_id = {case["record_id"]: case for case in cases}
    output_root = OUT_ROOT / "scan" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = device = capture = None
    panel_summaries = []
    try:
        model, tokenizer, device = load_model(
            model_name, use_8bit=True
        )
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        head_count = int(model.config.num_attention_heads)
        event_rows, event_lookup = events(
            depths, head_count, int(info.n_layers)
        )
        write_jsonl(output_root / "events.jsonl", event_rows)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else 0
            )
        capture = HeadCapture(layers, depths, head_count)
        capture.register()
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel_units = [
                    unit for unit in units
                    if unit["family"] == family
                    and unit["output_mode"] == output_mode
                ]
                panel_summaries.append(scan_panel(
                    model=model,
                    device=device,
                    capture=capture,
                    pad_token_id=int(pad_token_id),
                    model_name=model_name,
                    family=family,
                    output_mode=output_mode,
                    units=panel_units,
                    case_by_id=case_by_id,
                    qualification_by_key=qualification_by_key,
                    event_rows=event_rows,
                    event_lookup=event_lookup,
                    output_root=output_root,
                ))
        result = {
            "schema_version": "phase1013_head_scan_model.v1",
            "phase": PHASE,
            "source_protocol_phase": SOURCE_PROTOCOL_PHASE,
            "source_protocol_digest": protocol[
                "preregistration_digest"
            ],
            "model": model_name,
            "depth_operations": {
                str(depth): sorted(operations)
                for depth, operations in depth_operations.items()
            },
            "depths": depths,
            "head_count": head_count,
            "event_count": len(event_rows),
            "unit_count": sum(
                row["unit_count"] for row in panel_summaries
            ),
            "scalar_measurement_count": sum(
                row["scalar_measurement_count"]
                for row in panel_summaries
            ),
            "raw_hidden_tensors_persisted": 0,
            "state_forward_mode": "singleton_8bit",
            "direction_axes": list(HEAD_DIRECTION_AXES),
            "model_forward_count": sum(
                row["model_forward_count"]
                for row in panel_summaries
            ),
            "identity_maximum": max(
                row["identity_maximum"]
                for row in panel_summaries
            ),
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "head response morphology only; no causal edge or "
                "mechanism claim"
            ),
        }
        write_json(output_root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def profile_pass_at(
    row: dict[str, Any],
    split: str,
    *,
    direction_threshold: float,
    prevalence_threshold: float,
) -> bool:
    value = row["splits"][split]
    return bool(
        value["n"] >= MIN_N
        and len(value["name_pools"]) >= MIN_POOLS
        and len(value["templates"]) >= MIN_TEMPLATES
        and value["direction_consistency"] is not None
        and value["direction_consistency"] >= direction_threshold
        and value["control_envelope_prevalence"] is not None
        and value["control_envelope_prevalence"]
        >= prevalence_threshold
        and value["control_envelope_delta"] is not None
        and value["control_envelope_delta"] > 0
    )


def profile_pass(row: dict[str, Any], split: str) -> bool:
    return profile_pass_at(
        row,
        split,
        direction_threshold=DIRECTION_THRESHOLD,
        prevalence_threshold=PREVALENCE_THRESHOLD,
    )


def finalize() -> dict[str, Any]:
    all_profiles = []
    selection_source = read_json(
        PHASE1012_ROOT / "discovery_selection.json"
    )
    allowed = {
        (
            row["model"],
            row["operation"],
            int(row["depth"]),
        )
        for row in selection_source["selections"]
        if row["component"] == "attention_output"
    }
    for model_name in MODELS:
        scan_root = OUT_ROOT / "scan" / model_name
        event_rows = read_jsonl(scan_root / "events.jsonl")
        event_by_index = {
            int(row["event_index"]): row for row in event_rows
        }
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel_root = scan_root / family / output_mode
                units = read_jsonl(panel_root / "units.jsonl")
                scalar = np.load(
                    panel_root / "response_scalars.npz"
                )
                direction = np.load(
                    panel_root / "direction_consistency.npz"
                )
                values = scalar["normalized_magnitude"]
                directions = direction["direction_consistency"]
                counts = direction["direction_count"]
                split_masks = {
                    split: np.asarray(
                        [row["split"] == split for row in units],
                        dtype=np.bool_,
                    )
                    for split in SPLIT_INDEX
                }
                for axis_index, axis in enumerate(
                    HEAD_DIRECTION_AXES
                ):
                    qualified = scalar[f"{axis}_qualified"]
                    for event_index, event in event_by_index.items():
                        depth = int(event["depth"])
                        for operation in TARGET_OPERATIONS:
                            if (
                                model_name,
                                operation,
                                depth,
                            ) not in allowed:
                                continue
                            operation_index = OP_INDEX[operation]
                            split_rows = {}
                            for split, split_index in SPLIT_INDEX.items():
                                mask = (
                                    qualified[:, operation_index]
                                    & split_masks[split]
                                )
                                target = values[
                                    mask,
                                    operation_index,
                                    event_index,
                                ]
                                controls = values[
                                    mask, :, event_index
                                ][:, [
                                    OP_INDEX[name]
                                    for name in CONTROL_OPERATIONS
                                ]]
                                envelope = (
                                    np.max(controls, axis=1)
                                    if target.size
                                    else np.asarray([])
                                )
                                selected_units = [
                                    row
                                    for index, row in enumerate(units)
                                    if mask[index]
                                ]
                                split_rows[split] = {
                                    "n": len(selected_units),
                                    "name_pools": sorted({
                                        int(row["name_pool"])
                                        for row in selected_units
                                    }),
                                    "templates": sorted({
                                        int(row["template"])
                                        for row in selected_units
                                    }),
                                    "target_median": (
                                        None if not target.size
                                        else finite(np.median(target))
                                    ),
                                    "control_envelope_median": (
                                        None if not envelope.size
                                        else finite(np.median(envelope))
                                    ),
                                    "control_envelope_delta": (
                                        None if not target.size
                                        else finite(np.median(
                                            target - envelope
                                        ))
                                    ),
                                    "control_envelope_prevalence": (
                                        None if not target.size
                                        else finite(np.mean(
                                            target > envelope
                                        ))
                                    ),
                                    "direction_consistency": finite(
                                        directions[
                                            axis_index,
                                            operation_index,
                                            split_index,
                                            event_index,
                                        ]
                                    ),
                                    "direction_count": int(
                                        counts[
                                            axis_index,
                                            operation_index,
                                            split_index,
                                            event_index,
                                        ]
                                    ),
                                }
                            profile = {
                                "schema_version": (
                                    "phase1013_head_profile.v1"
                                ),
                                "phase": PHASE,
                                "model": model_name,
                                "family": family,
                                "output_mode": output_mode,
                                "qualification_axis": axis,
                                "operation": operation,
                                "event_id": event["event_id"],
                                "depth": depth,
                                "relative_depth": event[
                                    "relative_depth"
                                ],
                                "head": int(event["head"]),
                                "splits": split_rows,
                                "discovery_pass": False,
                                "confirmation_pass": False,
                                "claim": "head_response_only",
                            }
                            profile["discovery_pass"] = profile_pass(
                                profile, "discovery"
                            )
                            profile["confirmation_pass"] = profile_pass(
                                profile, "confirmation"
                            )
                            all_profiles.append(profile)
                scalar.close()
                direction.close()

    grouped: dict[tuple, dict[tuple, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in all_profiles:
        if not row["discovery_pass"]:
            continue
        physical = (
            row["model"],
            row["operation"],
            int(row["depth"]),
            int(row["head"]),
        )
        panel = (row["family"], row["output_mode"])
        grouped[physical][panel].add(
            row["qualification_axis"]
        )
    selections = []
    region_keys = sorted({
        (model, operation, depth)
        for model, operation, depth in allowed
    })
    for model, operation, depth in region_keys:
        candidates = []
        for physical, panels in grouped.items():
            if physical[:3] != (model, operation, depth):
                continue
            candidates.append({
                "head": int(physical[3]),
                "panel_count": len(panels),
                "all_axis_panel_count": sum(
                    len(axes) == len(HEAD_DIRECTION_AXES)
                    for axes in panels.values()
                ),
                "families": sorted({
                    panel[0] for panel in panels
                }),
                "output_modes": sorted({
                    panel[1] for panel in panels
                }),
            })
        if not candidates:
            continue
        maximum_panel = max(
            row["panel_count"] for row in candidates
        )
        candidates = [
            row for row in candidates
            if row["panel_count"] == maximum_panel
        ]
        maximum_all_axis = max(
            row["all_axis_panel_count"] for row in candidates
        )
        candidates = [
            row for row in candidates
            if row["all_axis_panel_count"] == maximum_all_axis
        ]
        candidates.sort(key=lambda row: row["head"])
        for row in candidates[:MAX_HEADS_PER_REGION]:
            confirmation_rows = [
                profile for profile in all_profiles
                if profile["model"] == model
                and profile["operation"] == operation
                and int(profile["depth"]) == depth
                and int(profile["head"]) == row["head"]
                and profile["confirmation_pass"]
            ]
            selections.append({
                "schema_version": (
                    "phase1013_discovery_frozen_head.v1"
                ),
                "phase": PHASE,
                "model": model,
                "operation": operation,
                "depth": depth,
                "head": row["head"],
                "selection_used_confirmation": False,
                "discovery_panel_count": row["panel_count"],
                "discovery_all_axis_panel_count": row[
                    "all_axis_panel_count"
                ],
                "discovery_families": row["families"],
                "discovery_output_modes": row["output_modes"],
                "confirmation_panel_count": len({
                    (profile["family"], profile["output_mode"])
                    for profile in confirmation_rows
                }),
                "confirmation_all_axis_panel_count": sum(
                    len({
                        profile["qualification_axis"]
                        for profile in confirmation_rows
                        if profile["family"] == family
                        and profile["output_mode"] == output_mode
                    }) == len(HEAD_DIRECTION_AXES)
                    for family in FAMILIES
                    for output_mode in OUTPUT_MODES
                ),
                "claim": "repeated_physical_head_response_only",
            })
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / "head_profiles.jsonl", all_profiles)
    write_jsonl(OUT_ROOT / "discovery_frozen_heads.jsonl", selections)
    sensitivity = []
    for direction_threshold in (0.30, 0.50, 0.70):
        for prevalence_threshold in (0.60, 0.70, 0.80):
            sensitivity.append({
                "direction_threshold": direction_threshold,
                "prevalence_threshold": prevalence_threshold,
                "discovery_pass_count": int(sum(
                    profile_pass_at(
                        row,
                        "discovery",
                        direction_threshold=direction_threshold,
                        prevalence_threshold=prevalence_threshold,
                    )
                    for row in all_profiles
                )),
                "confirmation_pass_count": int(sum(
                    profile_pass_at(
                        row,
                        "confirmation",
                        direction_threshold=direction_threshold,
                        prevalence_threshold=prevalence_threshold,
                    )
                    for row in all_profiles
                )),
                "both_split_pass_count": int(sum(
                    profile_pass_at(
                        row,
                        "discovery",
                        direction_threshold=direction_threshold,
                        prevalence_threshold=prevalence_threshold,
                    )
                    and profile_pass_at(
                        row,
                        "confirmation",
                        direction_threshold=direction_threshold,
                        prevalence_threshold=prevalence_threshold,
                    )
                    for row in all_profiles
                )),
            })
    summary = {
        "schema_version": "phase1013_head_response_summary.v1",
        "phase": PHASE,
        "source_phase": 1012,
        "profile_count": len(all_profiles),
        "discovery_pass_count": int(sum(
            row["discovery_pass"] for row in all_profiles
        )),
        "confirmation_pass_count": int(sum(
            row["confirmation_pass"] for row in all_profiles
        )),
        "both_split_pass_count": int(sum(
            row["discovery_pass"] and row["confirmation_pass"]
            for row in all_profiles
        )),
        "discovery_frozen_head_count": len(selections),
        "frozen_heads_confirming_any_panel": int(sum(
            row["confirmation_panel_count"] > 0
            for row in selections
        )),
        "by_model": {
            model: {
                "frozen_head_count": int(sum(
                    row["model"] == model for row in selections
                )),
                "confirming_head_count": int(sum(
                    row["model"] == model
                    and row["confirmation_panel_count"] > 0
                    for row in selections
                )),
            }
            for model in MODELS
        },
        "selection_used_confirmation": False,
        "threshold_sensitivity": sensitivity,
        "operational_thresholds_not_theory": {
            "direction": DIRECTION_THRESHOLD,
            "control_envelope_prevalence": PREVALENCE_THRESHOLD,
            "minimum_n": MIN_N,
            "minimum_name_pools": MIN_POOLS,
            "minimum_templates": MIN_TEMPLATES,
        },
        "claim_limits": [
            "head response is not head necessity or sufficiency",
            "head identifiers are model-local physical coordinates",
            "broad regions and heads were selected on discovery only",
            "confirmation updates evidence but does not create a closed "
            "mechanism",
        ],
    }
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(*MODELS, "finalize")
    )
    args = parser.parse_args()
    if args.command == "finalize":
        finalize()
    else:
        scan_model(args.command)


if __name__ == "__main__":
    main()
