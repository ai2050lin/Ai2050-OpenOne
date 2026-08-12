#!/usr/bin/env python3
"""Singleton all-component scan for counterbalanced relative differences."""

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
from phase1008_global_response_atlas_scan import (
    StateCapture,
    direction_consistency,
)
from phase1014_relative_difference_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    NATURAL_STATES,
    OUT_ROOT,
    OUTPUT_MODES,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


STATE_ORDER = NATURAL_STATES + ("identity",)
STATE_INDEX = {
    state: index for index, state in enumerate(STATE_ORDER)
}
OP_INDEX = {
    operation: index
    for index, operation in enumerate(ANALYSIS_OPERATIONS)
}
TARGET_OPERATIONS = ("F", "Q")
TARGET_INDEX = {
    operation: index
    for index, operation in enumerate(TARGET_OPERATIONS)
}
SPLIT_INDEX = {"discovery": 0, "confirmation": 1}
DIRECTION_AXES = (
    "all_units",
    "singleton_panel",
    "natural_rollout",
)
DIRECTION_MODES = ("raw", "canonical")
EPSILON = 1e-12


class AllHeadCapture:
    """Capture every real pre-o_proj head at one receiver position."""

    def __init__(self, layers, head_count: int):
        self.layers = layers
        self.head_count = head_count
        self.position: int | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args):
            value = args[0]
            if self.position is None:
                raise RuntimeError("head capture position is not set")
            selected = value[:, self.position, :]
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
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._hook(depth)
                )
            )

    def begin(self, position: int) -> None:
        self.position = int(position)
        self.values = {}
        self.counts = defaultdict(int)

    def validate(self) -> None:
        expected = set(range(1, len(self.layers) + 1))
        missing = sorted(expected - set(self.values))
        repeated = {
            depth: count
            for depth, count in self.counts.items()
            if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"head capture drift missing={missing[:5]} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.position = None


def event_definitions(
    n_layers: int,
    head_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, int]],
    list[tuple[int, int]],
]:
    events = []
    whole_keys: list[tuple[str, int]] = []
    head_keys: list[tuple[int, int]] = []

    whole_keys.append(("residual", 0))
    for depth in range(1, n_layers + 1):
        whole_keys.extend((
            ("residual", depth),
            ("attention_output", depth),
            ("mlp_output", depth),
        ))
    for component, depth in whole_keys:
        events.append({
            "schema_version": "phase1014_relative_event.v1",
            "phase": PHASE,
            "event_index": len(events),
            "event_id": f"{component}.d{depth:02d}",
            "component": component,
            "depth": depth,
            "relative_depth": depth / max(n_layers, 1),
            "head": None,
            "receiver_role": "answer_boundary",
            "vector_space": "model_width",
            "claim": "response_only",
        })
    for depth in range(1, n_layers + 1):
        for head in range(head_count):
            head_keys.append((depth, head))
            events.append({
                "schema_version": "phase1014_relative_event.v1",
                "phase": PHASE,
                "event_index": len(events),
                "event_id": f"attention_head.d{depth:02d}.h{head:02d}",
                "component": "attention_head_pre_o_proj",
                "depth": depth,
                "relative_depth": depth / max(n_layers, 1),
                "head": head,
                "receiver_role": "answer_boundary",
                "vector_space": "head_width",
                "claim": "physical_head_response_only",
            })
    return events, whole_keys, head_keys


def operation_values(
    values: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base = values[STATE_INDEX["base"]]
    deltas = {
        operation: values[STATE_INDEX[operation]] - base
        for operation in ("F", "Q", "FQ", "E", "O", "N", "L")
    }
    deltas["I"] = values[STATE_INDEX["identity"]] - base
    deltas["X"] = (
        values[STATE_INDEX["FQ"]]
        - values[STATE_INDEX["F"]]
        - values[STATE_INDEX["Q"]]
        + base
    )
    norms = torch.linalg.vector_norm(values, dim=-1)
    scales = {
        operation: 0.5 * (
            norms[STATE_INDEX[operation]]
            + norms[STATE_INDEX["base"]]
        )
        for operation in ("F", "Q", "FQ", "E", "O", "N", "L")
    }
    scales["I"] = 0.5 * (
        norms[STATE_INDEX["identity"]]
        + norms[STATE_INDEX["base"]]
    )
    scales["X"] = 0.25 * (
        norms[STATE_INDEX["base"]]
        + norms[STATE_INDEX["F"]]
        + norms[STATE_INDEX["Q"]]
        + norms[STATE_INDEX["FQ"]]
    )
    return deltas, scales


def panel_prediction(
    logits: torch.Tensor,
    case: dict[str, Any],
) -> tuple[int, bool, float]:
    candidates = [
        int(value) for value in case["candidate_token_ids"].values()
    ]
    candidate_tensor = torch.tensor(
        candidates,
        dtype=torch.long,
        device=logits.device,
    )
    candidate_logits = logits.index_select(0, candidate_tensor)
    winner = int(
        candidate_tensor[candidate_logits.argmax()].item()
    )
    expected = int(case["answer_token_ids"][0])
    foil = int(case["candidate_token_ids"][case["foil"]])
    margin = float((logits[expected] - logits[foil]).item())
    return winner, winner == expected, margin


def all_unit_direction_consistency(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Apply the pairwise direction identity over the last vector axis."""
    expanded_counts = np.broadcast_to(
        counts[None, ...],
        sums.shape[:-1],
    )
    result = np.full(sums.shape[:-1], np.nan, dtype=np.float32)
    flat_sums = sums.reshape(-1, sums.shape[-2], sums.shape[-1])
    flat_counts = expanded_counts.reshape(-1, expanded_counts.shape[-1])
    flat_result = result.reshape(-1, result.shape[-1])
    for index in range(flat_sums.shape[0]):
        count = flat_counts[index].astype(np.float64)
        squared = np.einsum(
            "ed,ed->e",
            flat_sums[index].astype(np.float64, copy=False),
            flat_sums[index].astype(np.float64, copy=False),
        )
        valid = count >= 2
        flat_result[index, valid] = (
            (squared[valid] - count[valid])
            / (count[valid] * (count[valid] - 1.0))
        ).astype(np.float32)
    return result


def run_panel(
    *,
    model,
    layers,
    info,
    head_count: int,
    device,
    model_name: str,
    family: str,
    output_mode: str,
    panel_units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    rollout_qualification: dict[tuple[str, str], bool],
    output_root: Path,
    events: list[dict[str, Any]],
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
    state_capture: StateCapture,
    head_capture: AllHeadCapture,
) -> dict[str, Any]:
    unit_count = len(panel_units)
    event_count = len(events)
    whole_count = len(whole_keys)
    head_event_count = len(head_keys)
    operation_count = len(ANALYSIS_OPERATIONS)
    target_count = len(TARGET_OPERATIONS)
    axis_count = len(DIRECTION_AXES)
    mode_count = len(DIRECTION_MODES)
    split_count = len(SPLIT_INDEX)
    d_model = int(info.d_model)
    attention_width = int(
        layers[0].self_attn.o_proj.in_features
    )
    if attention_width % head_count:
        raise RuntimeError(
            f"pre-o_proj width {attention_width} is not divisible by "
            f"{head_count} heads"
        )
    head_width = attention_width // head_count

    raw_magnitude = np.full(
        (unit_count, operation_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    normalized_magnitude = np.full_like(raw_magnitude, np.nan)
    direction_sum_whole = np.zeros(
        (
            mode_count,
            axis_count,
            target_count,
            split_count,
            whole_count,
            d_model,
        ),
        dtype=np.float32,
    )
    direction_sum_head = np.zeros(
        (
            mode_count,
            axis_count,
            target_count,
            split_count,
            head_event_count,
            head_width,
        ),
        dtype=np.float32,
    )
    direction_count_whole = np.zeros(
        (
            axis_count,
            target_count,
            split_count,
            whole_count,
        ),
        dtype=np.int32,
    )
    direction_count_head = np.zeros(
        (
            axis_count,
            target_count,
            split_count,
            head_event_count,
        ),
        dtype=np.int32,
    )
    unit_rows = []
    identity_maximum = 0.0
    singleton_forward_count = 0
    started = time.time()

    for unit_index, unit in enumerate(panel_units):
        state_cases = [
            case_by_id[unit["case_ids"][state]]
            for state in NATURAL_STATES
        ]
        state_cases.append(dict(state_cases[0]))
        state_whole = []
        state_head = []
        state_hits = {}
        state_winners = {}
        state_margins = {}

        for state, case in zip(STATE_ORDER, state_cases):
            input_ids = torch.tensor(
                [case["input_ids"]],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            position = int(case["role_positions"]["answer_boundary"])
            positions = torch.tensor(
                [[position]],
                dtype=torch.long,
                device=device,
            )
            state_capture.begin(positions)
            head_capture.begin(position)
            try:
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                singleton_forward_count += 1
                state_capture.validate()
                head_capture.validate()
                whole = torch.stack([
                    state_capture.captured[key][0, 0].float().cpu()
                    for key in whole_keys
                ])
                heads = torch.stack([
                    head_capture.values[depth][0, head].float().cpu()
                    for depth, head in head_keys
                ])
                state_whole.append(whole)
                state_head.append(heads)
                winner, hit, margin = panel_prediction(
                    output.logits[0, -1],
                    case,
                )
                state_winners[state] = winner
                state_hits[state] = bool(hit)
                state_margins[state] = margin
                del output, whole, heads
            finally:
                state_capture.captured = {}
                head_capture.values = {}
                del input_ids, attention_mask, positions

        whole_values = torch.stack(state_whole)
        head_values = torch.stack(state_head)
        whole_deltas, whole_scales = operation_values(whole_values)
        head_deltas, head_scales = operation_values(head_values)
        split_index = SPLIT_INDEX[unit["split"]]

        singleton_pair = {}
        natural_pair = {}
        for operation in ANALYSIS_OPERATIONS:
            if operation == "X":
                singleton_pair[operation] = all(
                    state_hits[state]
                    for state in ("base", "F", "Q", "FQ")
                )
                natural_pair[operation] = all(
                    rollout_qualification.get(
                        (unit["unit_id"], source),
                        False,
                    )
                    for source in ("F", "Q", "FQ")
                )
            else:
                variant = "base" if operation == "I" else operation
                singleton_pair[operation] = bool(
                    state_hits["base"] and state_hits[variant]
                )
                natural_pair[operation] = bool(
                    rollout_qualification.get(
                        (unit["unit_id"], operation),
                        False,
                    )
                )

        unit_rows.append({
            "schema_version": "phase1014_relative_scan_unit.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "family": family,
            "output_mode": output_mode,
            "unit_index": unit_index,
            "unit_id": unit["unit_id"],
            "split": unit["split"],
            "template": int(unit["template"]),
            "name_pool": int(unit["name_pool"]),
            "world_index": int(unit["world_index"]),
            "base_fact_bit": int(unit["base_fact_bit"]),
            "base_query_bit": int(unit["base_query_bit"]),
            "counterbalance_cell": unit["counterbalance_cell"],
            "canonical_factor_signs": unit[
                "canonical_factor_signs"
            ],
            "edit_counts": unit["edit_counts"],
            "singleton_state_hits": state_hits,
            "singleton_state_winners": state_winners,
            "singleton_state_margins": state_margins,
            "singleton_panel_qualification": singleton_pair,
            "natural_rollout_qualification": natural_pair,
        })

        for operation_index, operation in enumerate(
            ANALYSIS_OPERATIONS
        ):
            whole_delta = whole_deltas[operation]
            head_delta = head_deltas[operation]
            whole_raw = torch.linalg.vector_norm(
                whole_delta, dim=-1
            )
            head_raw = torch.linalg.vector_norm(
                head_delta, dim=-1
            )
            whole_norm = whole_raw / torch.clamp(
                whole_scales[operation], min=EPSILON
            )
            head_norm = head_raw / torch.clamp(
                head_scales[operation], min=EPSILON
            )
            combined_raw = torch.cat((whole_raw, head_raw))
            combined_norm = torch.cat((whole_norm, head_norm))
            raw_magnitude[
                unit_index, operation_index
            ] = combined_raw.numpy()
            normalized_magnitude[
                unit_index, operation_index
            ] = combined_norm.numpy()
            if operation == "I":
                identity_maximum = max(
                    identity_maximum,
                    float(combined_raw.max().item()),
                )
            del (
                whole_raw,
                head_raw,
                whole_norm,
                head_norm,
                combined_raw,
                combined_norm,
            )

        qualifications = {
            "all_units": {
                operation: True
                for operation in TARGET_OPERATIONS
            },
            "singleton_panel": {
                operation: singleton_pair[operation]
                for operation in TARGET_OPERATIONS
            },
            "natural_rollout": {
                operation: natural_pair[operation]
                for operation in TARGET_OPERATIONS
            },
        }
        for operation in TARGET_OPERATIONS:
            target_index = TARGET_INDEX[operation]
            whole_delta = whole_deltas[operation]
            head_delta = head_deltas[operation]
            whole_raw = torch.linalg.vector_norm(
                whole_delta, dim=-1
            )
            head_raw = torch.linalg.vector_norm(
                head_delta, dim=-1
            )
            whole_direction = whole_delta / torch.clamp(
                whole_raw[:, None], min=EPSILON
            )
            head_direction = head_delta / torch.clamp(
                head_raw[:, None], min=EPSILON
            )
            whole_direction_np = whole_direction.numpy()
            head_direction_np = head_direction.numpy()
            whole_valid = whole_raw.numpy() > EPSILON
            head_valid = head_raw.numpy() > EPSILON
            sign = int(unit["canonical_factor_signs"][operation])
            for axis_index, axis in enumerate(DIRECTION_AXES):
                if not qualifications[axis][operation]:
                    continue
                direction_sum_whole[
                    0,
                    axis_index,
                    target_index,
                    split_index,
                ] += whole_direction_np
                direction_sum_head[
                    0,
                    axis_index,
                    target_index,
                    split_index,
                ] += head_direction_np
                direction_sum_whole[
                    1,
                    axis_index,
                    target_index,
                    split_index,
                ] += sign * whole_direction_np
                direction_sum_head[
                    1,
                    axis_index,
                    target_index,
                    split_index,
                ] += sign * head_direction_np
                direction_count_whole[
                    axis_index,
                    target_index,
                    split_index,
                ] += whole_valid.astype(np.int32)
                direction_count_head[
                    axis_index,
                    target_index,
                    split_index,
                ] += head_valid.astype(np.int32)
            del (
                whole_raw,
                head_raw,
                whole_direction,
                head_direction,
            )

        del (
            state_whole,
            state_head,
            whole_values,
            head_values,
            whole_deltas,
            whole_scales,
            head_deltas,
            head_scales,
        )
        if (unit_index + 1) % 8 == 0:
            print(
                f"[relative-scan] {model_name}/{family}/"
                f"{output_mode} {unit_index + 1}/{unit_count}",
                flush=True,
            )

    consistency_whole = all_unit_direction_consistency(
        direction_sum_whole,
        direction_count_whole,
    )
    consistency_head = all_unit_direction_consistency(
        direction_sum_head,
        direction_count_head,
    )
    direction_consistency_all = np.concatenate(
        (consistency_whole, consistency_head),
        axis=-1,
    )
    direction_count_all = np.concatenate(
        (direction_count_whole, direction_count_head),
        axis=-1,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "units.jsonl", unit_rows)
    np.savez_compressed(
        output_root / "response_scalars.npz",
        raw_magnitude=raw_magnitude,
        normalized_magnitude=normalized_magnitude,
    )
    np.savez_compressed(
        output_root / "direction_consistency.npz",
        direction_consistency=direction_consistency_all,
        direction_count=direction_count_all,
    )
    np.savez_compressed(
        output_root / "canonical_direction_sums.npz",
        whole=direction_sum_whole[
            1, DIRECTION_AXES.index("all_units")
        ],
        head=direction_sum_head[
            1, DIRECTION_AXES.index("all_units")
        ],
        whole_count=direction_count_whole[
            DIRECTION_AXES.index("all_units")
        ],
        head_count=direction_count_head[
            DIRECTION_AXES.index("all_units")
        ],
    )
    summary = {
        "schema_version": "phase1014_relative_scan_panel.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "family": family,
        "output_mode": output_mode,
        "unit_count": unit_count,
        "event_count": event_count,
        "whole_event_count": whole_count,
        "head_event_count": head_event_count,
        "singleton_forward_count": singleton_forward_count,
        "scalar_measurement_count": int(
            raw_magnitude.size + normalized_magnitude.size
        ),
        "identity_maximum": identity_maximum,
        "raw_hidden_tensors_persisted": 0,
        "aggregated_direction_sums_persisted": True,
        "elapsed_seconds": time.time() - started,
        "claim_limit": (
            "relative response morphology only; no transport, causal, "
            "or mechanism edge"
        ),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def run_model(
    model_name: str,
    smoke_units_per_panel: int = 0,
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1014 protocol revision drift")
    model_root = OUT_ROOT / "protocol" / model_name
    cases = read_jsonl(model_root / "cases.jsonl")
    units = read_jsonl(model_root / "units.jsonl")
    behavior_pairs = read_jsonl(
        OUT_ROOT
        / "behavior"
        / model_name
        / "pair_qualification.jsonl"
    )
    rollout_qualification = {
        (row["unit_id"], row["operation"]): bool(
            row["rollout_pair_qualified"]
        )
        for row in behavior_pairs
    }
    case_by_id = {row["record_id"]: row for row in cases}
    model = tokenizer = device = None
    state_capture = head_capture = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        head_count = int(model.config.num_attention_heads)
        attention_width = int(
            layers[0].self_attn.o_proj.in_features
        )
        if attention_width % head_count:
            raise RuntimeError(
                f"{model_name}: pre-o_proj width {attention_width} "
                f"is not divisible by {head_count} heads"
            )
        head_width = attention_width // head_count
        events, whole_keys, head_keys = event_definitions(
            int(info.n_layers),
            head_count,
        )
        scan_namespace = (
            f"scan_smoke_{smoke_units_per_panel}"
            if smoke_units_per_panel > 0
            else "scan"
        )
        output_root = OUT_ROOT / scan_namespace / model_name
        output_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_root / "events.jsonl", events)
        state_capture = StateCapture(model, layers)
        head_capture = AllHeadCapture(layers, head_count)
        state_capture.register()
        head_capture.register()
        panel_summaries = []
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel_units = [
                    unit for unit in units
                    if unit["family"] == family
                    and unit["output_mode"] == output_mode
                ]
                if smoke_units_per_panel > 0:
                    panel_units = panel_units[:smoke_units_per_panel]
                panel_summaries.append(run_panel(
                    model=model,
                    layers=layers,
                    info=info,
                    head_count=head_count,
                    device=device,
                    model_name=model_name,
                    family=family,
                    output_mode=output_mode,
                    panel_units=panel_units,
                    case_by_id=case_by_id,
                    rollout_qualification=rollout_qualification,
                    output_root=(
                        output_root / family / output_mode
                    ),
                    events=events,
                    whole_keys=whole_keys,
                    head_keys=head_keys,
                    state_capture=state_capture,
                    head_capture=head_capture,
                ))
        summary = {
            "schema_version": "phase1014_relative_scan_model.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": protocol["preregistration_digest"],
            "model": model_name,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": head_count,
                "pre_o_proj_width": attention_width,
                "head_width": head_width,
                "loaded_8bit": True,
            },
            "unit_count": sum(
                row["unit_count"] for row in panel_summaries
            ),
            "smoke_units_per_panel": smoke_units_per_panel,
            "formal_scan": smoke_units_per_panel == 0,
            "event_count": len(events),
            "singleton_forward_count": sum(
                row["singleton_forward_count"]
                for row in panel_summaries
            ),
            "scalar_measurement_count": sum(
                row["scalar_measurement_count"]
                for row in panel_summaries
            ),
            "identity_maximum": max(
                row["identity_maximum"]
                for row in panel_summaries
            ),
            "raw_hidden_tensors_persisted": 0,
            "panel_summaries": panel_summaries,
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "all-layer singleton relative-response atlas; formulas "
                "remain measurement definitions"
            ),
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if state_capture is not None:
            state_capture.close()
        if head_capture is not None:
            head_capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = state_capture = head_capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--smoke-units-per-panel",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    run_model(args.model, args.smoke_units_per_panel)


if __name__ == "__main__":
    main()
