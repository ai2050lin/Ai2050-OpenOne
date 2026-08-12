#!/usr/bin/env python3
"""Stream the Phase1008 residual/attention/MLP response atlas.

Only per-unit scalar measurements and aggregate direction consistency are
persisted. Full hidden-state and component tensors are never written.
"""
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
from phase1006_blind_source_and_behavior import eos_token_ids
from phase1008_global_response_atlas_protocol import (
    ANALYSIS_OPERATIONS,
    MODELS,
    OUT_ROOT,
    PAIR_OPERATIONS,
    PHASE,
    PROMPT_ROLES,
    TIME_STAGES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


STATE_ORDER = ("base", "B", "Q", "BQ", "E", "O", "N", "identity")
STATE_INDEX = {name: index for index, name in enumerate(STATE_ORDER)}
OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLIT_INDEX = {"discovery": 0, "confirmation": 1}
EPSILON = 1e-12


def semantic_answer_ids(case: dict[str, Any]) -> list[int]:
    return [
        int(case["answer_token_ids"][int(index)])
        for index in case["semantic_steps"]
    ]


def stage_case(case: dict[str, Any], stage: str) -> dict[str, Any]:
    row = dict(case)
    if stage == "prompt":
        suffix: list[int] = []
        role_positions = dict(case["role_positions"])
    elif stage == "semantic0":
        suffix = [int(value) for value in case["protocol_prefix_ids"]]
        if not suffix:
            raise RuntimeError("empty protocol prefix at semantic0")
        role_positions = {"decision_boundary": len(case["input_ids"]) + len(suffix) - 1}
    elif stage == "semantic1":
        suffix = (
            [int(value) for value in case["protocol_prefix_ids"]]
            + [semantic_answer_ids(case)[0]]
        )
        role_positions = {"decision_boundary": len(case["input_ids"]) + len(suffix) - 1}
    elif stage == "termination":
        suffix = [int(value) for value in case["answer_token_ids"]]
        role_positions = {"decision_boundary": len(case["input_ids"]) + len(suffix) - 1}
    else:
        raise KeyError(stage)
    row["input_ids"] = [int(value) for value in case["input_ids"]] + suffix
    row["scan_role_positions"] = role_positions
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


def event_definitions(n_layers: int) -> tuple[list[dict[str, Any]], dict[tuple, int]]:
    result: list[dict[str, Any]] = []
    lookup: dict[tuple, int] = {}
    for stage in TIME_STAGES:
        roles = PROMPT_ROLES if stage == "prompt" else ("decision_boundary",)
        component_depths = (
            ("residual", range(0, n_layers + 1)),
            ("attention_output", range(1, n_layers + 1)),
            ("mlp_output", range(1, n_layers + 1)),
        )
        for component, depths in component_depths:
            for depth in depths:
                for role in roles:
                    key = (stage, component, int(depth), role)
                    event_index = len(result)
                    lookup[key] = event_index
                    result.append({
                        "schema_version": "phase1008_event.v1",
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
                        "edge_claim_allowed_from_scan": "co_response_only",
                    })
    return result, lookup


class StateCapture:
    """Capture selected positions from all residual/component writes."""

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.captured: dict[tuple[str, int], torch.Tensor] = {}
        self.counts: dict[tuple[str, int], int] = defaultdict(int)
        self.handles = []

    def _selected(self, output) -> torch.Tensor:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise RuntimeError(f"unexpected hook output type/shape: {type(value)}")
        if self.positions is None:
            raise RuntimeError("capture positions were not set")
        positions = self.positions.to(value.device)
        batch_index = torch.arange(value.shape[0], device=value.device)[:, None]
        return value[batch_index, positions, :].detach()

    def _hook(self, component: str, depth: int):
        key = (component, depth)

        def hook(module, args, output):
            self.captured[key] = self._selected(output)
            self.counts[key] += 1
            return output

        return hook

    def register(self) -> None:
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(
                self._hook("residual", 0)
            )
        )
        for index, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(self._hook("residual", index))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook("attention_output", index)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._hook("mlp_output", index)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.captured = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        expected = {
            ("residual", 0),
            *{
                (component, depth)
                for depth in range(1, len(self.layers) + 1)
                for component in ("residual", "attention_output", "mlp_output")
            },
        }
        missing = sorted(expected - set(self.captured))
        repeated = {
            str(key): count
            for key, count in self.counts.items()
            if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"capture drift missing={missing[:5]} repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.captured = {}


def operation_deltas(values: torch.Tensor) -> dict[str, torch.Tensor]:
    base = values[STATE_INDEX["base"]]
    result = {
        operation: values[STATE_INDEX[operation]] - base
        for operation in ("B", "Q", "BQ", "E", "O", "N")
    }
    result["I"] = values[STATE_INDEX["identity"]] - base
    result["X"] = (
        values[STATE_INDEX["BQ"]]
        - values[STATE_INDEX["B"]]
        - values[STATE_INDEX["Q"]]
        + base
    )
    return result


def operation_scales(values: torch.Tensor) -> dict[str, torch.Tensor]:
    norms = torch.linalg.vector_norm(values.float(), dim=-1)
    base = norms[STATE_INDEX["base"]]
    result = {
        operation: 0.5 * (
            norms[STATE_INDEX[operation]] + base
        )
        for operation in ("B", "Q", "BQ", "E", "O", "N")
    }
    result["I"] = 0.5 * (
        norms[STATE_INDEX["identity"]] + base
    )
    result["X"] = 0.25 * (
        norms[STATE_INDEX["base"]]
        + norms[STATE_INDEX["B"]]
        + norms[STATE_INDEX["Q"]]
        + norms[STATE_INDEX["BQ"]]
    )
    return result


def direction_consistency(
    direction_sum: np.ndarray,
    direction_count: np.ndarray,
) -> np.ndarray:
    result = np.full(direction_count.shape, np.nan, dtype=np.float32)
    for operation in range(direction_sum.shape[0]):
        for split in range(direction_sum.shape[1]):
            counts = direction_count[operation, split].astype(np.float64)
            sums = direction_sum[operation, split].astype(np.float64, copy=False)
            squared = np.einsum("ed,ed->e", sums, sums)
            valid = counts >= 2
            result[operation, split, valid] = (
                (squared[valid] - counts[valid])
                / (counts[valid] * (counts[valid] - 1.0))
            ).astype(np.float32)
    return result


def fixed_panel_metrics(
    logits: torch.Tensor,
    base_case: dict[str, Any],
    state_cases: list[dict[str, Any]],
    stage: str,
    effective_eos: set[int],
) -> list[dict[str, Any]]:
    values = logits.float()
    rows = []
    if stage == "semantic0":
        base_gold = int(
            base_case["candidate_ids_by_step"][0][base_case["gold_parts"][0]]
        )
        base_foil = int(
            base_case["candidate_ids_by_step"][0][base_case["foil_parts"][0]]
        )
        fixed_ids = torch.tensor(
            [base_gold, base_foil],
            dtype=torch.long,
            device=values.device,
        )
        fixed_panel = values.index_select(-1, fixed_ids)
        fixed_prob = torch.softmax(fixed_panel, dim=-1)
        for index, case in enumerate(state_cases):
            gold = int(
                case["candidate_ids_by_step"][0][case["gold_parts"][0]]
            )
            foil = int(
                case["candidate_ids_by_step"][0][case["foil_parts"][0]]
            )
            rows.append({
                "correct_margin": float(
                    (values[index, gold] - values[index, foil]).item()
                ),
                "fixed_base_margin": float(
                    (fixed_panel[index, 0] - fixed_panel[index, 1]).item()
                ),
                "fixed_base_probability": float(fixed_prob[index, 0].item()),
            })
    elif stage == "semantic1":
        for index, case in enumerate(state_cases):
            gold = int(
                case["candidate_ids_by_step"][1][case["gold_parts"][1]]
            )
            foil = int(
                case["candidate_ids_by_step"][1][case["foil_parts"][1]]
            )
            rows.append({
                "correct_margin": float(
                    (values[index, gold] - values[index, foil]).item()
                ),
            })
    elif stage == "termination":
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
                ),
            })
        del non_eos, eos_best, other_best
    else:
        rows = [{} for _ in state_cases]
    return rows


def unit_qualification(
    unit: dict[str, Any],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.zeros(len(ANALYSIS_OPERATIONS), dtype=np.bool_)
    rollout = np.zeros(len(ANALYSIS_OPERATIONS), dtype=np.bool_)
    for operation in PAIR_OPERATIONS:
        row = qualification_by_key[(unit["unit_id"], operation)]
        semantic[OP_INDEX[operation]] = bool(row["semantic_pair_qualified"])
        rollout[OP_INDEX[operation]] = bool(row["rollout_pair_qualified"])
    for target, source in ((semantic, "semantic"), (rollout, "rollout")):
        target[OP_INDEX["X"]] = all(
            target[OP_INDEX[operation]]
            for operation in ("B", "Q", "BQ")
        )
    return semantic, rollout


def run_model(
    model_name: str,
    *,
    scope: str,
    limit_units: int | None,
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    behavior = read_json(OUT_ROOT / "behavior" / model_name / "summary.json")
    if behavior["protocol_digest"] != protocol["preregistration_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    protocol_root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(protocol_root / "cases.jsonl")
    units = read_jsonl(protocol_root / "units.jsonl")
    if limit_units is not None:
        units = units[:limit_units]
    case_by_id = {case["record_id"]: case for case in cases}
    qualification = read_jsonl(
        OUT_ROOT / "behavior" / model_name / "pair_qualification.jsonl"
    )
    qualification_by_key = {
        (row["unit_id"], row["operation"]): row for row in qualification
    }
    output_root = OUT_ROOT / ("scan" if scope == "formal" else "scan_smoke") / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = device = capture = None
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        events, event_lookup = event_definitions(info.n_layers)
        operation_count = len(ANALYSIS_OPERATIONS)
        unit_count = len(units)
        event_count = len(events)
        raw_magnitude = np.full(
            (unit_count, operation_count, event_count),
            np.nan,
            dtype=np.float32,
        )
        normalized_magnitude = np.full_like(raw_magnitude, np.nan)
        semantic_qualified = np.zeros(
            (unit_count, operation_count), dtype=np.bool_
        )
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
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        capture = StateCapture(model, layers)
        capture.register()
        unit_metadata = []
        output_measurements: dict[str, dict[str, dict[str, Any]]] = {}

        for unit_index, unit in enumerate(units):
            unit_semantic, unit_rollout = unit_qualification(
                unit, qualification_by_key
            )
            semantic_qualified[unit_index] = unit_semantic
            rollout_qualified[unit_index] = unit_rollout
            base = case_by_id[unit["case_ids"]["base"]]
            state_cases = [
                base,
                case_by_id[unit["case_ids"]["B"]],
                case_by_id[unit["case_ids"]["Q"]],
                case_by_id[unit["case_ids"]["BQ"]],
                case_by_id[unit["case_ids"]["E"]],
                case_by_id[unit["case_ids"]["O"]],
                case_by_id[unit["case_ids"]["N"]],
                dict(base),
            ]
            split_index = SPLIT_INDEX[unit["split"]]
            unit_metadata.append({
                "schema_version": "phase1008_scan_unit.v1",
                "phase": PHASE,
                "model": model_name,
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
                "rollout_qualified": {
                    operation: bool(unit_rollout[OP_INDEX[operation]])
                    for operation in ANALYSIS_OPERATIONS
                },
            })
            output_measurements[unit["unit_id"]] = {
                state: {} for state in STATE_ORDER
            }

            for stage in TIME_STAGES:
                staged = [stage_case(case, stage) for case in state_cases]
                role_names = (
                    list(PROMPT_ROLES)
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
                input_ids, attention = case_tensors(staged, device)
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
                    last_logits = output.logits[:, -1, :].detach()
                    stage_outputs = fixed_panel_metrics(
                        last_logits,
                        staged[0],
                        staged,
                        stage,
                        effective_eos,
                    )
                    for state_index, state in enumerate(STATE_ORDER):
                        output_measurements[unit["unit_id"]][state][stage] = (
                            stage_outputs[state_index]
                        )

                    for (component, depth), values in capture.captured.items():
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
                                    scale, min=EPSILON
                                )
                                raw_value = float(raw.item())
                                normalized_value = float(normalized.item())
                                raw_magnitude[
                                    unit_index, operation_index, event_index
                                ] = raw_value
                                normalized_magnitude[
                                    unit_index, operation_index, event_index
                                ] = normalized_value
                                if raw_value > EPSILON:
                                    direction = (
                                        delta / torch.clamp(raw, min=EPSILON)
                                    ).detach().cpu().numpy()
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
                                del delta, raw, scale, normalized
                    del output, last_logits
                finally:
                    del input_ids, attention, positions
                    capture.captured = {}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if (unit_index + 1) % 4 == 0 or unit_index + 1 == unit_count:
                print(
                    f"[scan] {model_name} {unit_index + 1}/{unit_count} units",
                    flush=True,
                )

        consistency = direction_consistency(direction_sum, direction_count)
        output_rows = []
        for unit in units:
            measurements = output_measurements[unit["unit_id"]]
            base = measurements["base"]
            for operation in PAIR_OPERATIONS:
                variant_state = "base" if operation == "I" else operation
                variant = measurements[variant_state]
                base_probability = base["semantic0"]["fixed_base_probability"]
                variant_probability = variant["semantic0"]["fixed_base_probability"]
                output_rows.append({
                    "schema_version": "phase1008_output_pair.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "unit_id": unit["unit_id"],
                    "split": unit["split"],
                    "template": int(unit["template"]),
                    "name_pool": int(unit["name_pool"]),
                    "world_index": int(unit["world_index"]),
                    "operation": operation,
                    "expected_output_relation": (
                        "changes" if operation in ("B", "Q")
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
                    "base_semantic0_correct_margin": base["semantic0"][
                        "correct_margin"
                    ],
                    "variant_semantic0_correct_margin": variant["semantic0"][
                        "correct_margin"
                    ],
                    "base_semantic1_correct_margin": base["semantic1"][
                        "correct_margin"
                    ],
                    "variant_semantic1_correct_margin": variant["semantic1"][
                        "correct_margin"
                    ],
                    "base_eos_margin": base["termination"]["eos_margin"],
                    "variant_eos_margin": variant["termination"]["eos_margin"],
                })

        np.savez_compressed(
            output_root / "response_scalars.npz",
            raw_magnitude=raw_magnitude,
            normalized_magnitude=normalized_magnitude,
            semantic_qualified=semantic_qualified,
            rollout_qualified=rollout_qualified,
        )
        np.savez_compressed(
            output_root / "direction_consistency.npz",
            direction_consistency=consistency,
            direction_count=direction_count,
        )
        write_jsonl(output_root / "events.jsonl", events)
        write_jsonl(output_root / "units.jsonl", unit_metadata)
        write_jsonl(output_root / "output_pairs.jsonl", output_rows)
        numerical = normalized_magnitude[:, OP_INDEX["I"], :]
        summary = {
            "schema_version": "phase1008_scan_summary.v1",
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
            "unit_count": unit_count,
            "event_count": event_count,
            "operation_count": operation_count,
            "scalar_measurement_count": int(
                unit_count * operation_count * event_count
            ),
            "raw_hidden_tensors_persisted": 0,
            "operations": list(ANALYSIS_OPERATIONS),
            "stages": list(TIME_STAGES),
            "semantic_qualified_pair_counts": {
                operation: int(np.sum(
                    semantic_qualified[:, OP_INDEX[operation]]
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
                "maximum": float(np.nanmax(numerical)),
                "mean": float(np.nanmean(numerical)),
                "nonzero_count": int(np.sum(numerical > EPSILON)),
            },
            "direction_formula": (
                "(||sum_i u_i||^2-n)/(n(n-1)); u_i is a unit paired "
                "response direction"
            ),
            "edge_claim_allowed": "co_response_only",
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
    parser.add_argument("--limit-units", type=int)
    args = parser.parse_args()
    limit = args.limit_units
    if args.scope == "smoke" and limit is None:
        limit = 2
    run_model(args.model, scope=args.scope, limit_units=limit)


if __name__ == "__main__":
    main()
