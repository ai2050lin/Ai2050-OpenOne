#!/usr/bin/env python3
"""Measure Phase1011 native-task response fields without naming mechanisms.

The scan persists scalar response magnitudes and aggregate direction
consistency only. Hidden states and component tensors are never written.
"""
from __future__ import annotations

import argparse
import gc
import json
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
from phase1006_blind_source_and_behavior import eos_token_ids
from phase1008_global_response_atlas_scan import StateCapture
from phase1011_native_semantic_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_MODES,
    PAIR_OPERATIONS,
    PHASE,
    PROTOCOL_REVISION,
    TIME_STAGES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


STATE_ORDER = ("base", "F", "Q", "FQ", "E", "O", "N", "S", "identity")
STATE_INDEX = {name: index for index, name in enumerate(STATE_ORDER)}
OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLIT_INDEX = {"discovery": 0, "confirmation": 1}
DIRECTION_AXES = ("semantic_panel", "natural_rollout")
EPSILON = 1e-12


def case_tensors(
    cases: list[dict[str, Any]],
    device,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad natural variants while preserving every real token index."""
    lengths = torch.tensor(
        [len(case["input_ids"]) for case in cases],
        dtype=torch.long,
        device=device,
    )
    width = int(lengths.max().item())
    input_ids = torch.full(
        (len(cases), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    for index, case in enumerate(cases):
        values = torch.tensor(
            case["input_ids"], dtype=torch.long, device=device
        )
        input_ids[index, : values.numel()] = values
        attention[index, : values.numel()] = 1
    return input_ids, attention, lengths


def stage_case(case: dict[str, Any], stage: str) -> dict[str, Any]:
    row = dict(case)
    if stage == "prompt":
        suffix: list[int] = []
        role_positions = dict(case["role_positions"])
    elif stage == "after_answer":
        suffix = [int(value) for value in case["answer_token_ids"]]
        role_positions = {
            "decision_boundary": len(case["input_ids"]) + len(suffix) - 1
        }
    else:
        raise KeyError(stage)
    row["input_ids"] = [int(value) for value in case["input_ids"]] + suffix
    row["scan_role_positions"] = role_positions
    return row


def event_definitions(
    *,
    n_layers: int,
    prompt_role_classes: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[tuple, int]]:
    events: list[dict[str, Any]] = []
    lookup: dict[tuple, int] = {}
    for stage in TIME_STAGES:
        roles = (
            tuple(prompt_role_classes)
            if stage == "prompt"
            else ("decision_boundary",)
        )
        component_depths = (
            ("residual", range(0, n_layers + 1)),
            ("attention_output", range(1, n_layers + 1)),
            ("mlp_output", range(1, n_layers + 1)),
        )
        for component, depths in component_depths:
            for depth in depths:
                for role in roles:
                    key = (stage, component, int(depth), role)
                    event_index = len(events)
                    lookup[key] = event_index
                    role_class = (
                        prompt_role_classes[role]
                        if stage == "prompt"
                        else "decision_boundary"
                    )
                    events.append({
                        "schema_version": "phase1011_native_event.v1",
                        "phase": PHASE,
                        "event_index": event_index,
                        "event_id": (
                            f"{stage}.{component}.d{int(depth):02d}.{role}"
                        ),
                        "stage": stage,
                        "component": component,
                        "depth": int(depth),
                        "relative_depth": float(depth / max(n_layers, 1)),
                        "role": role,
                        "role_class": role_class,
                        "edge_claim_allowed_from_scan": "co_response_only",
                    })
    return events, lookup


def operation_deltas(values: torch.Tensor) -> dict[str, torch.Tensor]:
    base = values[STATE_INDEX["base"]]
    result = {
        operation: values[STATE_INDEX[operation]] - base
        for operation in ("F", "Q", "FQ", "E", "O", "N", "S")
    }
    result["I"] = values[STATE_INDEX["identity"]] - base
    result["X"] = (
        values[STATE_INDEX["FQ"]]
        - values[STATE_INDEX["F"]]
        - values[STATE_INDEX["Q"]]
        + base
    )
    return result


def operation_scales(values: torch.Tensor) -> dict[str, torch.Tensor]:
    norms = torch.linalg.vector_norm(values.float(), dim=-1)
    base = norms[STATE_INDEX["base"]]
    result = {
        operation: 0.5 * (norms[STATE_INDEX[operation]] + base)
        for operation in ("F", "Q", "FQ", "E", "O", "N", "S")
    }
    result["I"] = 0.5 * (norms[STATE_INDEX["identity"]] + base)
    result["X"] = 0.25 * (
        norms[STATE_INDEX["base"]]
        + norms[STATE_INDEX["F"]]
        + norms[STATE_INDEX["Q"]]
        + norms[STATE_INDEX["FQ"]]
    )
    return result


def direction_consistency(
    direction_sum: np.ndarray,
    direction_count: np.ndarray,
) -> np.ndarray:
    """Return mean pairwise cosine without storing any unit direction."""
    result = np.full(direction_count.shape, np.nan, dtype=np.float32)
    for axis in range(direction_sum.shape[0]):
        for operation in range(direction_sum.shape[1]):
            for split in range(direction_sum.shape[2]):
                counts = direction_count[axis, operation, split].astype(
                    np.float64
                )
                sums = direction_sum[
                    axis, operation, split
                ].astype(np.float64, copy=False)
                squared = np.einsum("ed,ed->e", sums, sums)
                valid = counts >= 2
                result[axis, operation, split, valid] = (
                    (squared[valid] - counts[valid])
                    / (counts[valid] * (counts[valid] - 1.0))
                ).astype(np.float32)
    return result


def unit_qualification(
    unit: dict[str, Any],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, np.ndarray]:
    result = {
        "semantic_panel": np.zeros(
            len(ANALYSIS_OPERATIONS), dtype=np.bool_
        ),
        "strict_teacher": np.zeros(
            len(ANALYSIS_OPERATIONS), dtype=np.bool_
        ),
        "natural_rollout": np.zeros(
            len(ANALYSIS_OPERATIONS), dtype=np.bool_
        ),
        "strict_rollout": np.zeros(
            len(ANALYSIS_OPERATIONS), dtype=np.bool_
        ),
    }
    source_fields = {
        "semantic_panel": "semantic_pair_qualified",
        "strict_teacher": "strict_teacher_pair_qualified",
        "natural_rollout": "rollout_pair_qualified",
        "strict_rollout": "strict_rollout_pair_qualified",
    }
    for operation in PAIR_OPERATIONS:
        row = qualification_by_key[(unit["unit_id"], operation)]
        for axis, field in source_fields.items():
            result[axis][OP_INDEX[operation]] = bool(row[field])
    for values in result.values():
        values[OP_INDEX["X"]] = all(
            values[OP_INDEX[operation]]
            for operation in ("F", "Q", "FQ")
        )
    return result


def output_metrics(
    *,
    logits: torch.Tensor,
    base_case: dict[str, Any],
    state_cases: list[dict[str, Any]],
    stage: str,
    effective_eos: set[int],
) -> list[dict[str, float]]:
    values = logits.float()
    rows: list[dict[str, float]] = []
    if stage == "prompt":
        base_gold = int(base_case["answer_token_ids"][0])
        base_foil = int(base_case["candidate_token_ids"][base_case["foil"]])
        fixed_ids = torch.tensor(
            [base_gold, base_foil],
            dtype=torch.long,
            device=values.device,
        )
        fixed_panel = values.index_select(-1, fixed_ids)
        fixed_probability = torch.softmax(fixed_panel, dim=-1)[:, 0]
        for index, case in enumerate(state_cases):
            gold = int(case["answer_token_ids"][0])
            foil = int(case["candidate_token_ids"][case["foil"]])
            rows.append({
                "controlled_correct_margin": float(
                    (values[index, gold] - values[index, foil]).item()
                ),
                "fixed_base_margin": float(
                    (fixed_panel[index, 0] - fixed_panel[index, 1]).item()
                ),
                "fixed_base_probability": float(
                    fixed_probability[index].item()
                ),
            })
    elif stage == "after_answer":
        eos_tensor = torch.tensor(
            sorted(effective_eos),
            dtype=torch.long,
            device=values.device,
        )
        eos_best = values.index_select(-1, eos_tensor).max(dim=-1).values
        non_eos = values.clone()
        non_eos.index_fill_(1, eos_tensor, -torch.inf)
        other_best = non_eos.max(dim=-1).values
        for index in range(len(state_cases)):
            rows.append({
                "eos_margin": float(
                    (eos_best[index] - other_best[index]).item()
                )
            })
        del non_eos, eos_best, other_best
    else:
        raise KeyError(stage)
    return rows


def scan_panel(
    *,
    model,
    layers,
    info,
    device,
    pad_token_id: int,
    effective_eos: set[int],
    model_name: str,
    family: str,
    output_mode: str,
    units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    first_case = case_by_id[units[0]["case_ids"]["base"]]
    events, event_lookup = event_definitions(
        n_layers=int(info.n_layers),
        prompt_role_classes=first_case["role_classes"],
    )
    operation_count = len(ANALYSIS_OPERATIONS)
    unit_count = len(units)
    event_count = len(events)
    raw_magnitude = np.full(
        (unit_count, operation_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    normalized_magnitude = np.full_like(raw_magnitude, np.nan)
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
    direction_sum = np.zeros(
        (
            len(DIRECTION_AXES),
            operation_count,
            len(SPLIT_INDEX),
            event_count,
            int(info.d_model),
        ),
        dtype=np.float32,
    )
    direction_count = np.zeros(
        (
            len(DIRECTION_AXES),
            operation_count,
            len(SPLIT_INDEX),
            event_count,
        ),
        dtype=np.int32,
    )
    capture = StateCapture(model, layers)
    capture.register()
    unit_rows = []
    pair_outputs = []
    started = time.time()
    try:
        for unit_index, unit in enumerate(units):
            qualifications = unit_qualification(
                unit, qualification_by_key
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
            split_index = SPLIT_INDEX[unit["split"]]
            unit_rows.append({
                "schema_version": "phase1011_native_scan_unit.v1",
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
                "qualification": {
                    axis: {
                        operation: bool(values[OP_INDEX[operation]])
                        for operation in ANALYSIS_OPERATIONS
                    }
                    for axis, values in qualifications.items()
                },
            })
            measurements = {
                state: {} for state in STATE_ORDER
            }

            for stage in TIME_STAGES:
                staged = [stage_case(case, stage) for case in state_cases]
                role_names = (
                    list(first_case["role_classes"])
                    if stage == "prompt"
                    else ["decision_boundary"]
                )
                positions = torch.tensor(
                    [
                        [
                            int(case["scan_role_positions"][role])
                            for role in role_names
                        ]
                        for case in staged
                    ],
                    dtype=torch.long,
                    device=device,
                )
                input_ids, attention, lengths = case_tensors(
                    staged, device, pad_token_id
                )
                capture.begin(positions)
                try:
                    with torch.inference_mode():
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention,
                            use_cache=False,
                            return_dict=True,
                        )
                    capture.validate()
                    batch_index = torch.arange(
                        len(staged), device=output.logits.device
                    )
                    selected_logits = output.logits[
                        batch_index, lengths - 1, :
                    ].detach()
                    metrics = output_metrics(
                        logits=selected_logits,
                        base_case=staged[0],
                        state_cases=staged,
                        stage=stage,
                        effective_eos=effective_eos,
                    )
                    for state_index, state in enumerate(STATE_ORDER):
                        measurements[state][stage] = metrics[state_index]

                    for (component, depth), values in capture.captured.items():
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
                        direction_stack = delta_stack / torch.clamp(
                            raw_stack[..., None], min=EPSILON
                        )
                        raw_cpu = raw_stack.detach().cpu().numpy()
                        normalized_cpu = (
                            normalized_stack.detach().cpu().numpy()
                        )
                        direction_cpu = (
                            direction_stack.detach().cpu().numpy()
                        )
                        for role_index, role in enumerate(role_names):
                            event_index = event_lookup[
                                (stage, component, int(depth), role)
                            ]
                            raw_magnitude[
                                unit_index, :, event_index
                            ] = raw_cpu[:, role_index]
                            normalized_magnitude[
                                unit_index, :, event_index
                            ] = normalized_cpu[:, role_index]
                            for operation_index in range(operation_count):
                                if raw_cpu[
                                    operation_index, role_index
                                ] <= EPSILON:
                                    continue
                                for axis_index, axis in enumerate(
                                    DIRECTION_AXES
                                ):
                                    if not qualifications[axis][
                                        operation_index
                                    ]:
                                        continue
                                    direction_sum[
                                        axis_index,
                                        operation_index,
                                        split_index,
                                        event_index,
                                    ] += direction_cpu[
                                        operation_index, role_index
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
                            direction_stack,
                        )
                    del output, selected_logits, batch_index
                finally:
                    del input_ids, attention, positions, lengths
                    capture.captured = {}

            base_measurements = measurements["base"]
            for operation in PAIR_OPERATIONS:
                variant_state = "base" if operation == "I" else operation
                variant_measurements = measurements[variant_state]
                base_probability = base_measurements["prompt"][
                    "fixed_base_probability"
                ]
                variant_probability = variant_measurements["prompt"][
                    "fixed_base_probability"
                ]
                pair_outputs.append({
                    "schema_version": "phase1011_native_output_pair.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "family": family,
                    "output_mode": output_mode,
                    "unit_id": unit["unit_id"],
                    "split": unit["split"],
                    "template": int(unit["template"]),
                    "name_pool": int(unit["name_pool"]),
                    "world_index": int(unit["world_index"]),
                    "operation": operation,
                    "expected_output_relation": (
                        "changes" if operation in ("F", "Q")
                        else "same_as_base"
                    ),
                    "base_fixed_choice_margin": base_measurements[
                        "prompt"
                    ]["fixed_base_margin"],
                    "variant_fixed_choice_margin": variant_measurements[
                        "prompt"
                    ]["fixed_base_margin"],
                    "delta_fixed_choice_margin": (
                        variant_measurements["prompt"]["fixed_base_margin"]
                        - base_measurements["prompt"]["fixed_base_margin"]
                    ),
                    "fixed_panel_probability_l1": float(
                        2.0
                        * abs(variant_probability - base_probability)
                    ),
                    "base_controlled_correct_margin": base_measurements[
                        "prompt"
                    ]["controlled_correct_margin"],
                    "variant_controlled_correct_margin": (
                        variant_measurements["prompt"][
                            "controlled_correct_margin"
                        ]
                    ),
                    "base_eos_margin": base_measurements["after_answer"][
                        "eos_margin"
                    ],
                    "variant_eos_margin": variant_measurements[
                        "after_answer"
                    ]["eos_margin"],
                })
            if (unit_index + 1) % 4 == 0 or unit_index + 1 == unit_count:
                print(
                    f"[scan] {model_name} {family}/{output_mode} "
                    f"{unit_index + 1}/{unit_count}",
                    flush=True,
                )

        consistency = direction_consistency(
            direction_sum, direction_count
        )
        panel_root = output_root / family / output_mode
        panel_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            panel_root / "response_scalars.npz",
            raw_magnitude=raw_magnitude,
            normalized_magnitude=normalized_magnitude,
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
        write_jsonl(panel_root / "events.jsonl", events)
        write_jsonl(panel_root / "units.jsonl", unit_rows)
        write_jsonl(panel_root / "output_pairs.jsonl", pair_outputs)
        identity = normalized_magnitude[:, OP_INDEX["I"], :]
        summary = {
            "schema_version": "phase1011_native_scan_panel_summary.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "family": family,
            "output_mode": output_mode,
            "unit_count": unit_count,
            "event_count": event_count,
            "operation_count": operation_count,
            "scalar_measurement_count": int(
                unit_count * operation_count * event_count
            ),
            "raw_hidden_tensors_persisted": 0,
            "operations": list(ANALYSIS_OPERATIONS),
            "stages": list(TIME_STAGES),
            "direction_axes": list(DIRECTION_AXES),
            "qualified_pair_counts": {
                axis: {
                    operation: int(np.sum(values[:, OP_INDEX[operation]]))
                    for operation in ANALYSIS_OPERATIONS
                }
                for axis, values in qualification_arrays.items()
            },
            "identity_normalized_floor": {
                "maximum": float(np.nanmax(identity)),
                "mean": float(np.nanmean(identity)),
                "nonzero_count": int(np.sum(identity > EPSILON)),
            },
            "direction_formula": (
                "(||sum_i u_i||^2-n)/(n(n-1)); computed separately "
                "for behavior-qualified discovery and confirmation pairs"
            ),
            "edge_claim_allowed": "co_response_only",
            "after_answer_caveat": (
                "contains the teacher-forced answer-token surface and is "
                "not by itself a semantic or causal mechanism"
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_json(panel_root / "summary.json", summary)
        return summary
    finally:
        capture.close()
        del direction_sum
        gc.collect()


def run_model(
    model_name: str,
    *,
    scope: str,
    limit_units: int | None,
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("protocol revision drift")
    behavior = read_json(
        OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    if behavior["protocol_digest"] != protocol["preregistration_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    protocol_root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(protocol_root / "cases.jsonl")
    units = read_jsonl(protocol_root / "units.jsonl")
    case_by_id = {case["record_id"]: case for case in cases}
    qualifications = read_jsonl(
        OUT_ROOT / "behavior" / model_name / "pair_qualification.jsonl"
    )
    qualification_by_key = {
        (row["unit_id"], row["operation"]): row
        for row in qualifications
    }
    output_root = (
        OUT_ROOT
        / ("scan" if scope == "formal" else "scan_smoke")
        / model_name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = device = None
    panel_summaries = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else 0
            )
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel_units = [
                    unit
                    for unit in units
                    if unit["family"] == family
                    and unit["output_mode"] == output_mode
                ]
                if limit_units is not None:
                    panel_units = panel_units[:limit_units]
                panel_summaries.append(scan_panel(
                    model=model,
                    layers=layers,
                    info=info,
                    device=device,
                    pad_token_id=int(pad_token_id),
                    effective_eos=effective_eos,
                    model_name=model_name,
                    family=family,
                    output_mode=output_mode,
                    units=panel_units,
                    case_by_id=case_by_id,
                    qualification_by_key=qualification_by_key,
                    output_root=output_root,
                ))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        summary = {
            "schema_version": "phase1011_native_scan_model_summary.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "scope": scope,
            "protocol_digest": protocol["preregistration_digest"],
            "model": model_name,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "model_class": info.model_class,
                "loaded_8bit": True,
            },
            "panel_count": len(panel_summaries),
            "unit_count": int(sum(
                row["unit_count"] for row in panel_summaries
            )),
            "scalar_measurement_count": int(sum(
                row["scalar_measurement_count"]
                for row in panel_summaries
            )),
            "raw_hidden_tensors_persisted": 0,
            "panels": panel_summaries,
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "descriptive repeated response fields only; no scan edge "
                "establishes transport, mediation, necessity, or sufficiency"
            ),
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--scope", choices=("smoke", "formal"), default="formal"
    )
    parser.add_argument("--limit-units", type=int)
    args = parser.parse_args()
    limit = args.limit_units
    if args.scope == "smoke" and limit is None:
        limit = 2
    run_model(args.model, scope=args.scope, limit_units=limit)


if __name__ == "__main__":
    main()
