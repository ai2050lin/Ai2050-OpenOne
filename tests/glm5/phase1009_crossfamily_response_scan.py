#!/usr/bin/env python3
"""Stream Phase1009 cross-family residual/attention/MLP response scalars."""
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
from phase1008_global_response_atlas_scan import (
    StateCapture,
    direction_consistency,
)
from phase1009_crossfamily_response_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    NATURAL_STATES,
    OUT_ROOT,
    PAIR_OPERATIONS,
    PHASE,
    ROLE_CLASSES,
    TIME_STAGES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


STATE_ORDER = NATURAL_STATES + ("identity",)
STATE_INDEX = {name: index for index, name in enumerate(STATE_ORDER)}
OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLIT_INDEX = {"discovery": 0, "confirmation": 1}
EPSILON = 1e-12


def stage_case(case: dict[str, Any], stage: str) -> dict[str, Any]:
    row = dict(case)
    if stage == "prompt":
        suffix: list[int] = []
        roles = dict(case["role_positions"])
    elif stage == "semantic0":
        suffix = [int(value) for value in case["protocol_prefix_ids"]]
        roles = {
            "decision_boundary": len(case["input_ids"]) + len(suffix) - 1
        }
    elif stage == "function0":
        semantic_id = int(
            case["answer_token_ids"][int(case["semantic_step"])]
        )
        suffix = (
            [int(value) for value in case["protocol_prefix_ids"]]
            + [semantic_id]
        )
        roles = {
            "decision_boundary": len(case["input_ids"]) + len(suffix) - 1
        }
    elif stage == "termination":
        suffix = [int(value) for value in case["answer_token_ids"]]
        roles = {
            "decision_boundary": len(case["input_ids"]) + len(suffix) - 1
        }
    else:
        raise KeyError(stage)
    row["input_ids"] = [int(value) for value in case["input_ids"]] + suffix
    row["scan_role_positions"] = roles
    return row


def case_tensors(cases: list[dict[str, Any]], device):
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift: {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def event_definitions(
    family: str,
    n_layers: int,
) -> tuple[list[dict[str, Any]], dict[tuple, int]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple, int] = {}
    for stage in TIME_STAGES:
        roles = (
            tuple(ROLE_CLASSES[family])
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
                    event_index = len(rows)
                    lookup[key] = event_index
                    rows.append({
                        "schema_version": "phase1009_event.v1",
                        "phase": PHASE,
                        "family": family,
                        "event_index": event_index,
                        "event_id": (
                            f"{family}.{stage}.{component}."
                            f"d{int(depth):02d}.{role}"
                        ),
                        "stage": stage,
                        "component": component,
                        "depth": int(depth),
                        "relative_depth": float(depth / max(n_layers, 1)),
                        "role": role,
                        "role_class": (
                            ROLE_CLASSES[family][role]
                            if stage == "prompt"
                            else "decision_boundary"
                        ),
                        "edge_claim_allowed_from_scan": "co_response_only",
                    })
    return rows, lookup


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


def unit_qualification(
    unit: dict[str, Any],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic = np.zeros(len(ANALYSIS_OPERATIONS), dtype=np.bool_)
    strict = np.zeros_like(semantic)
    rollout = np.zeros_like(semantic)
    for operation in PAIR_OPERATIONS:
        row = qualification_by_key[(unit["unit_id"], operation)]
        index = OP_INDEX[operation]
        semantic[index] = bool(row["semantic_pair_qualified"])
        strict[index] = bool(row["strict_teacher_pair_qualified"])
        rollout[index] = bool(row["rollout_pair_qualified"])
    for target in (semantic, strict, rollout):
        target[OP_INDEX["X"]] = all(
            target[OP_INDEX[operation]]
            for operation in ("F", "Q", "FQ")
        )
    return semantic, strict, rollout


def output_metrics(
    logits: torch.Tensor,
    state_cases: list[dict[str, Any]],
    stage: str,
    effective_eos: set[int],
) -> list[dict[str, Any]]:
    values = logits.float()
    if stage == "semantic0":
        base = state_cases[STATE_INDEX["base"]]
        base_gold = int(
            base["answer_token_ids"][int(base["semantic_step"])]
        )
        base_foil = int(base["candidate_name_ids"][base["foil"]])
        fixed_ids = torch.tensor(
            [base_gold, base_foil],
            dtype=torch.long,
            device=values.device,
        )
        fixed = values.index_select(-1, fixed_ids)
        probabilities = torch.softmax(fixed, dim=-1)
        rows = []
        for index, case in enumerate(state_cases):
            gold = int(
                case["answer_token_ids"][int(case["semantic_step"])]
            )
            foil = int(case["candidate_name_ids"][case["foil"]])
            rows.append({
                "correct_margin": float(
                    (values[index, gold] - values[index, foil]).item()
                ),
                "fixed_base_margin": float(
                    (fixed[index, 0] - fixed[index, 1]).item()
                ),
                "fixed_base_probability": float(
                    probabilities[index, 0].item()
                ),
            })
        return rows
    if stage == "function0":
        eos_ids = torch.tensor(
            sorted(effective_eos),
            dtype=torch.long,
            device=values.device,
        )
        rows = []
        for index, case in enumerate(state_cases):
            done = int(case["function_token_id"])
            eos_best = values[index].index_select(0, eos_ids).max()
            rows.append({
                "done_vs_eos_margin": float(
                    (values[index, done] - eos_best).item()
                ),
            })
        return rows
    if stage == "termination":
        eos_ids = torch.tensor(
            sorted(effective_eos),
            dtype=torch.long,
            device=values.device,
        )
        eos_best = values.index_select(-1, eos_ids).max(dim=-1).values
        non_eos = values.clone()
        non_eos.index_fill_(1, eos_ids, -torch.inf)
        other_best = non_eos.max(dim=-1).values
        rows = [
            {
                "eos_margin": float(
                    (eos_best[index] - other_best[index]).item()
                ),
            }
            for index in range(len(state_cases))
        ]
        del non_eos, eos_best, other_best
        return rows
    return [{} for _ in state_cases]


def capture_stage(
    *,
    model,
    capture: StateCapture,
    device,
    staged: list[dict[str, Any]],
    role_names: list[str],
) -> tuple[
    dict[tuple[str, int], torch.Tensor],
    torch.Tensor,
]:
    captured_states: dict[
        tuple[str, int],
        list[torch.Tensor | None],
    ] = {}
    logits_by_state: list[torch.Tensor | None] = [None] * len(staged)
    groups: dict[int, list[int]] = defaultdict(list)
    for state_index, case in enumerate(staged):
        groups[len(case["input_ids"])].append(state_index)
    for _, state_indices in sorted(groups.items()):
        group = [staged[index] for index in state_indices]
        positions = torch.tensor(
            [
                [
                    int(case["scan_role_positions"][role])
                    for role in role_names
                ]
                for case in group
            ],
            dtype=torch.long,
            device=device,
        )
        input_ids, attention = case_tensors(group, device)
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
            group_logits = output.logits[:, -1, :].detach().cpu()
            for local_index, state_index in enumerate(state_indices):
                logits_by_state[state_index] = group_logits[local_index]
            for key, values in capture.captured.items():
                slots = captured_states.setdefault(
                    key,
                    [None] * len(staged),
                )
                selected = values.detach().to("cpu")
                for local_index, state_index in enumerate(state_indices):
                    slots[state_index] = selected[local_index]
            del output, group_logits
        finally:
            del input_ids, attention, positions
            capture.captured = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if any(value is None for value in logits_by_state):
        raise RuntimeError("stage logits coverage drift")
    stacked_captures = {}
    for key, slots in captured_states.items():
        if any(value is None for value in slots):
            raise RuntimeError(f"capture state coverage drift at {key}")
        stacked_captures[key] = torch.stack(slots)  # type: ignore[arg-type]
    logits = torch.stack(logits_by_state)  # type: ignore[arg-type]
    return stacked_captures, logits


def scan_family(
    *,
    model,
    layers,
    info,
    device,
    capture: StateCapture,
    effective_eos: set[int],
    model_name: str,
    family: str,
    units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    events, event_lookup = event_definitions(family, int(info.n_layers))
    unit_count = len(units)
    operation_count = len(ANALYSIS_OPERATIONS)
    event_count = len(events)
    raw_magnitude = np.full(
        (unit_count, operation_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    normalized_magnitude = np.full_like(raw_magnitude, np.nan)
    semantic_qualified = np.zeros(
        (unit_count, operation_count),
        dtype=np.bool_,
    )
    strict_qualified = np.zeros_like(semantic_qualified)
    rollout_qualified = np.zeros_like(semantic_qualified)
    direction_sum = np.zeros(
        (
            operation_count,
            len(SPLIT_INDEX),
            event_count,
            int(info.d_model),
        ),
        dtype=np.float32,
    )
    direction_count = np.zeros(
        (operation_count, len(SPLIT_INDEX), event_count),
        dtype=np.int32,
    )
    unit_rows = []
    all_output_measurements: dict[str, dict[str, dict[str, Any]]] = {}
    started = time.time()
    for unit_index, unit in enumerate(units):
        unit_semantic, unit_strict, unit_rollout = unit_qualification(
            unit,
            qualification_by_key,
        )
        semantic_qualified[unit_index] = unit_semantic
        strict_qualified[unit_index] = unit_strict
        rollout_qualified[unit_index] = unit_rollout
        base = case_by_id[unit["case_ids"]["base"]]
        state_cases = [
            case_by_id[unit["case_ids"][state]]
            for state in NATURAL_STATES
        ] + [dict(base)]
        split_index = SPLIT_INDEX[unit["split"]]
        unit_rows.append({
            "schema_version": "phase1009_scan_unit.v1",
            "phase": PHASE,
            "model": model_name,
            "family": family,
            "unit_index": unit_index,
            "unit_id": unit["unit_id"],
            "split": unit["split"],
            "template": int(unit["template"]),
            "name_pool": int(unit["name_pool"]),
            "world_index": int(unit["world_index"]),
            "semantic_qualified": {
                operation: bool(unit_semantic[OP_INDEX[operation]])
                for operation in ANALYSIS_OPERATIONS
            },
            "strict_qualified": {
                operation: bool(unit_strict[OP_INDEX[operation]])
                for operation in ANALYSIS_OPERATIONS
            },
            "rollout_qualified": {
                operation: bool(unit_rollout[OP_INDEX[operation]])
                for operation in ANALYSIS_OPERATIONS
            },
        })
        all_output_measurements[unit["unit_id"]] = {
            state: {} for state in STATE_ORDER
        }
        for stage in TIME_STAGES:
            staged = [stage_case(case, stage) for case in state_cases]
            role_names = (
                list(ROLE_CLASSES[family])
                if stage == "prompt"
                else ["decision_boundary"]
            )
            captured, logits = capture_stage(
                model=model,
                capture=capture,
                device=device,
                staged=staged,
                role_names=role_names,
            )
            stage_outputs = output_metrics(
                logits,
                staged,
                stage,
                effective_eos,
            )
            for state_index, state in enumerate(STATE_ORDER):
                all_output_measurements[unit["unit_id"]][state][stage] = (
                    stage_outputs[state_index]
                )
            for (component, depth), values in captured.items():
                deltas = operation_deltas(values)
                scales = operation_scales(values)
                for role_index, role in enumerate(role_names):
                    event_index = event_lookup[
                        (stage, component, int(depth), role)
                    ]
                    for operation in ANALYSIS_OPERATIONS:
                        operation_index = OP_INDEX[operation]
                        delta = deltas[operation][role_index].float()
                        raw = torch.linalg.vector_norm(delta)
                        scale = scales[operation][role_index].float()
                        normalized = raw / torch.clamp(
                            scale,
                            min=EPSILON,
                        )
                        raw_value = float(raw.item())
                        raw_magnitude[
                            unit_index, operation_index, event_index
                        ] = raw_value
                        normalized_magnitude[
                            unit_index, operation_index, event_index
                        ] = float(normalized.item())
                        if raw_value > EPSILON:
                            direction = (
                                delta / torch.clamp(raw, min=EPSILON)
                            ).numpy()
                            direction_sum[
                                operation_index,
                                split_index,
                                event_index,
                            ] += direction.astype(np.float32, copy=False)
                            direction_count[
                                operation_index,
                                split_index,
                                event_index,
                            ] += 1
            del captured, logits
        if (unit_index + 1) % 4 == 0 or unit_index + 1 == unit_count:
            print(
                f"[scan] {model_name}/{family} "
                f"{unit_index + 1}/{unit_count} units",
                flush=True,
            )
    output_rows = []
    for unit in units:
        measurements = all_output_measurements[unit["unit_id"]]
        base = measurements["base"]
        for operation in PAIR_OPERATIONS:
            variant_state = "base" if operation == "I" else operation
            variant = measurements[variant_state]
            base_probability = base["semantic0"]["fixed_base_probability"]
            variant_probability = variant["semantic0"][
                "fixed_base_probability"
            ]
            output_rows.append({
                "schema_version": "phase1009_output_pair.v1",
                "phase": PHASE,
                "model": model_name,
                "family": family,
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
                "base_fixed_choice_margin": base["semantic0"][
                    "fixed_base_margin"
                ],
                "variant_fixed_choice_margin": variant["semantic0"][
                    "fixed_base_margin"
                ],
                "delta_fixed_choice_margin": (
                    variant["semantic0"]["fixed_base_margin"]
                    - base["semantic0"]["fixed_base_margin"]
                ),
                "fixed_panel_probability_l1": float(
                    2.0 * abs(variant_probability - base_probability)
                ),
                "base_correct_margin": base["semantic0"]["correct_margin"],
                "variant_correct_margin": variant["semantic0"][
                    "correct_margin"
                ],
                "base_done_vs_eos_margin": base["function0"][
                    "done_vs_eos_margin"
                ],
                "variant_done_vs_eos_margin": variant["function0"][
                    "done_vs_eos_margin"
                ],
                "base_eos_margin": base["termination"]["eos_margin"],
                "variant_eos_margin": variant["termination"]["eos_margin"],
            })
    family_root = output_root / family
    family_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        family_root / "response_scalars.npz",
        raw_magnitude=raw_magnitude,
        normalized_magnitude=normalized_magnitude,
        semantic_qualified=semantic_qualified,
        strict_qualified=strict_qualified,
        rollout_qualified=rollout_qualified,
    )
    np.savez_compressed(
        family_root / "direction_consistency.npz",
        direction_consistency=direction_consistency(
            direction_sum,
            direction_count,
        ),
        direction_count=direction_count,
    )
    write_jsonl(family_root / "events.jsonl", events)
    write_jsonl(family_root / "units.jsonl", unit_rows)
    write_jsonl(family_root / "output_pairs.jsonl", output_rows)
    identity = normalized_magnitude[:, OP_INDEX["I"], :]
    summary = {
        "schema_version": "phase1009_family_scan_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "family": family,
        "unit_count": unit_count,
        "event_count": event_count,
        "operation_count": operation_count,
        "scalar_measurement_count": int(
            unit_count * operation_count * event_count
        ),
        "raw_hidden_tensors_persisted": 0,
        "semantic_qualified_pair_counts": {
            operation: int(np.sum(
                semantic_qualified[:, OP_INDEX[operation]]
            ))
            for operation in ANALYSIS_OPERATIONS
        },
        "strict_qualified_pair_counts": {
            operation: int(np.sum(
                strict_qualified[:, OP_INDEX[operation]]
            ))
            for operation in ANALYSIS_OPERATIONS
        },
        "rollout_qualified_pair_counts": {
            operation: int(np.sum(
                rollout_qualified[:, OP_INDEX[operation]]
            ))
            for operation in ANALYSIS_OPERATIONS
        },
        "identity_normalized_floor": {
            "maximum": float(np.nanmax(identity)),
            "mean": float(np.nanmean(identity)),
            "nonzero_count": int(np.sum(identity > EPSILON)),
        },
        "edge_claim_allowed": "co_response_only",
        "elapsed_seconds": time.time() - started,
    }
    write_json(family_root / "summary.json", summary)
    return summary


def run_model(
    model_name: str,
    *,
    scope: str,
    limit_units_per_family: int | None,
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    behavior = read_json(OUT_ROOT / "behavior" / model_name / "summary.json")
    if behavior["protocol_digest"] != protocol["preregistration_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "units.jsonl"
    )
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
    started = time.time()
    model = tokenizer = device = capture = None
    summaries = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        capture = StateCapture(model, layers)
        capture.register()
        for family in FAMILIES:
            family_units = [
                unit for unit in units if unit["family"] == family
            ]
            if limit_units_per_family is not None:
                family_units = family_units[:limit_units_per_family]
            summaries.append(scan_family(
                model=model,
                layers=layers,
                info=info,
                device=device,
                capture=capture,
                effective_eos=effective_eos,
                model_name=model_name,
                family=family,
                units=family_units,
                case_by_id=case_by_id,
                qualification_by_key=qualification_by_key,
                output_root=output_root,
            ))
        summary = {
            "schema_version": "phase1009_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "scope": scope,
            "protocol_digest": protocol["preregistration_digest"],
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "model_class": info.model_class,
                "loaded_8bit": True,
            },
            "family_summaries": summaries,
            "unit_count": int(sum(row["unit_count"] for row in summaries)),
            "event_count_sum": int(
                sum(row["event_count"] for row in summaries)
            ),
            "scalar_measurement_count": int(sum(
                row["scalar_measurement_count"] for row in summaries
            )),
            "raw_hidden_tensors_persisted": 0,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--limit-units-per-family", type=int)
    args = parser.parse_args()
    limit = args.limit_units_per_family
    if args.scope == "smoke" and limit is None:
        limit = 1
    run_model(
        args.model,
        scope=args.scope,
        limit_units_per_family=limit,
    )


if __name__ == "__main__":
    main()
