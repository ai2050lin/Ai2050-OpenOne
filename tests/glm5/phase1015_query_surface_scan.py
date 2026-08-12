#!/usr/bin/env python3
"""Singleton multi-position scan for Phase1015 query-surface mapping."""

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
from phase1008_global_response_atlas_scan import StateCapture
from phase1014_bf16_precision_confirmation import load_bf16
from phase1015_query_surface_protocol import (
    ANALYSIS_OPERATIONS,
    CAPTURE_ROLES,
    FAMILIES,
    MODELS,
    NATURAL_STATES,
    OUT_ROOT,
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
DIRECTION_MODES = ("raw", "canonical")
ROLE_INDEX = {
    role: index for index, role in enumerate(CAPTURE_ROLES)
}
Q_PREFIX_ROLES = (
    "fact_source",
    "fact_relation",
    "fact_target",
    "lexical_control",
    "query_anchor",
)
EPSILON = 1e-12
PHASE1014_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)


class MultiPositionHeadCapture:
    """Capture all real pre-o_proj heads at several token positions."""

    def __init__(self, layers, head_count: int):
        self.layers = layers
        self.head_count = head_count
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args):
            value = args[0]
            if self.positions is None:
                raise RuntimeError("head capture positions are not set")
            positions = self.positions.to(value.device)
            selected = value[:, positions, :]
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("head width drift")
            self.values[depth] = selected.reshape(
                selected.shape[0],
                selected.shape[1],
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

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
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
        self.positions = None
        self.values = {}


def event_definitions(
    n_layers: int,
    head_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, int]],
    list[tuple[int, int]],
]:
    events = []
    whole_keys: list[tuple[str, int]] = [("residual", 0)]
    for depth in range(1, n_layers + 1):
        whole_keys.extend((
            ("residual", depth),
            ("attention_output", depth),
            ("mlp_output", depth),
        ))
    for component, depth in whole_keys:
        events.append({
            "schema_version": "phase1015_chain_event.v1",
            "phase": PHASE,
            "event_index": len(events),
            "event_id": f"{component}.d{depth:02d}",
            "component": component,
            "depth": int(depth),
            "relative_depth": depth / max(n_layers, 1),
            "head": None,
            "vector_space": "model_width",
            "claim": "role_conditioned_response_only",
        })
    head_keys = []
    for depth in range(1, n_layers + 1):
        for head in range(head_count):
            head_keys.append((depth, head))
            events.append({
                "schema_version": "phase1015_chain_event.v1",
                "phase": PHASE,
                "event_index": len(events),
                "event_id": (
                    f"attention_head.d{depth:02d}.h{head:02d}"
                ),
                "component": "attention_head_pre_o_proj",
                "depth": int(depth),
                "relative_depth": depth / max(n_layers, 1),
                "head": int(head),
                "vector_space": "head_width",
                "claim": "physical_head_role_response_only",
            })
    return events, whole_keys, head_keys


def load_phase1014_frozen(model_name: str) -> list[dict[str, Any]]:
    path = (
        PHASE1014_ROOT
        / "analysis"
        / "control_specific_shared_events.jsonl"
    )
    if not path.exists():
        return []
    rows = read_jsonl(path)
    return [
        row for row in rows
        if row["model"] == model_name
        and row["component"] == "attention_head_pre_o_proj"
    ]


def operation_values(
    values: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base = values[STATE_INDEX["base"]]
    deltas = {
        operation: values[STATE_INDEX[operation]] - base
        for operation in ("F", "Q", "FQ", "E", "N", "L")
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
        for operation in ("F", "Q", "FQ", "E", "N", "L")
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


def consistency_from_sums(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    result = np.full(sums.shape[:-1], np.nan, dtype=np.float32)
    squared = np.einsum(
        "...d,...d->...",
        sums.astype(np.float64, copy=False),
        sums.astype(np.float64, copy=False),
    )
    expanded = np.broadcast_to(counts[None, ...], squared.shape)
    valid = expanded >= 2
    result[valid] = (
        (squared[valid] - expanded[valid])
        / (expanded[valid] * (expanded[valid] - 1.0))
    ).astype(np.float32)
    return result


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


def run_panel(
    *,
    model,
    layers,
    info,
    head_count: int,
    device,
    model_name: str,
    family: str,
    surface: int,
    panel_units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    output_root: Path,
    events: list[dict[str, Any]],
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
    frozen_rows: list[dict[str, Any]],
    state_capture: StateCapture,
    head_capture: MultiPositionHeadCapture,
) -> dict[str, Any]:
    unit_count = len(panel_units)
    role_count = len(CAPTURE_ROLES)
    event_count = len(events)
    whole_count = len(whole_keys)
    head_event_count = len(head_keys)
    operation_count = len(ANALYSIS_OPERATIONS)
    target_count = len(TARGET_OPERATIONS)
    d_model = int(info.d_model)
    attention_width = int(layers[0].self_attn.o_proj.in_features)
    if attention_width % head_count:
        raise RuntimeError(
            f"pre-o_proj width {attention_width} is not divisible by "
            f"{head_count} heads"
        )
    head_width = attention_width // head_count
    frozen_head_indices = np.asarray([
        (int(row["depth"]) - 1) * head_count + int(row["head"])
        for row in frozen_rows
    ], dtype=np.int64)

    normalized_magnitude = np.full(
        (unit_count, operation_count, role_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    direction_sum_whole = np.zeros(
        (
            len(DIRECTION_MODES),
            target_count,
            role_count,
            whole_count,
            d_model,
        ),
        dtype=np.float32,
    )
    direction_sum_head = np.zeros(
        (
            len(DIRECTION_MODES),
            target_count,
            role_count,
            head_event_count,
            head_width,
        ),
        dtype=np.float32,
    )
    direction_count_whole = np.zeros(
        (target_count, role_count, whole_count),
        dtype=np.int32,
    )
    direction_count_head = np.zeros(
        (target_count, role_count, head_event_count),
        dtype=np.int32,
    )
    unit_rows = []
    identity_maximum = 0.0
    q_prefix_maximum = 0.0
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
            role_positions = [
                int(case["role_positions"][role])
                for role in CAPTURE_ROLES
            ]
            positions = torch.tensor(
                [role_positions],
                dtype=torch.long,
                device=device,
            )
            head_positions = torch.tensor(
                role_positions,
                dtype=torch.long,
                device=device,
            )
            state_capture.begin(positions)
            head_capture.begin(head_positions)
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
                    state_capture.captured[key][0].float().cpu()
                    for key in whole_keys
                ]).permute(1, 0, 2).contiguous()
                heads = torch.stack([
                    head_capture.values[depth][0, :, head].float().cpu()
                    for depth, head in head_keys
                ]).permute(1, 0, 2).contiguous()
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
                del (
                    input_ids,
                    attention_mask,
                    positions,
                    head_positions,
                )

        whole_values = torch.stack(state_whole)
        head_values = torch.stack(state_head)
        whole_deltas, whole_scales = operation_values(whole_values)
        head_deltas, head_scales = operation_values(head_values)
        unit_rows.append({
            "schema_version": "phase1015_chain_scan_unit.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "family": family,
            "query_surface": int(surface),
            "query_surface_name": unit["query_surface_name"],
            "query_surface_class": unit["query_surface_class"],
            "balanced_query_inventory": bool(
                unit["balanced_query_inventory"]
            ),
            "unit_index": int(unit_index),
            "unit_id": unit["unit_id"],
            "split": unit["split"],
            "template": int(unit["template"]),
            "name_pool": int(unit["name_pool"]),
            "world_index": int(unit["world_index"]),
            "canonical_factor_signs": unit[
                "canonical_factor_signs"
            ],
            "edit_counts": unit["edit_counts"],
            "singleton_state_hits": state_hits,
            "singleton_state_winners": state_winners,
            "singleton_state_margins": state_margins,
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
            normalized_magnitude[
                unit_index,
                operation_index,
                :,
                :whole_count,
            ] = whole_norm.numpy()
            normalized_magnitude[
                unit_index,
                operation_index,
                :,
                whole_count:,
            ] = head_norm.numpy()
            if operation == "I":
                identity_maximum = max(
                    identity_maximum,
                    float(whole_raw.max().item()),
                    float(head_raw.max().item()),
                )
            if operation == "Q":
                prefix = [ROLE_INDEX[role] for role in Q_PREFIX_ROLES]
                q_prefix_maximum = max(
                    q_prefix_maximum,
                    float(whole_raw[prefix].max().item()),
                    float(head_raw[prefix].max().item()),
                )
            del whole_raw, head_raw, whole_norm, head_norm

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
                whole_raw[..., None], min=EPSILON
            )
            head_direction = head_delta / torch.clamp(
                head_raw[..., None], min=EPSILON
            )
            whole_np = whole_direction.numpy()
            head_np = head_direction.numpy()
            whole_valid = whole_raw.numpy() > EPSILON
            head_valid = head_raw.numpy() > EPSILON
            sign = int(unit["canonical_factor_signs"][operation])
            direction_sum_whole[0, target_index] += whole_np
            direction_sum_head[0, target_index] += head_np
            direction_sum_whole[1, target_index] += sign * whole_np
            direction_sum_head[1, target_index] += sign * head_np
            direction_count_whole[target_index] += (
                whole_valid.astype(np.int32)
            )
            direction_count_head[target_index] += (
                head_valid.astype(np.int32)
            )
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
                f"[scan] {model_name}/{family}/surface{surface} "
                f"{unit_index + 1}/{unit_count}",
                flush=True,
            )

    whole_consistency = consistency_from_sums(
        direction_sum_whole,
        direction_count_whole,
    )
    head_consistency = consistency_from_sums(
        direction_sum_head,
        direction_count_head,
    )
    panel_root = output_root / model_name / family / f"surface_{surface}"
    panel_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_root / "response_scalars.npz",
        normalized_magnitude=normalized_magnitude,
        operation_names=np.asarray(ANALYSIS_OPERATIONS),
        role_names=np.asarray(CAPTURE_ROLES),
    )
    np.savez_compressed(
        panel_root / "direction_consistency.npz",
        whole=whole_consistency,
        head=head_consistency,
        whole_count=direction_count_whole,
        head_count=direction_count_head,
        mode_names=np.asarray(DIRECTION_MODES),
        operation_names=np.asarray(TARGET_OPERATIONS),
        role_names=np.asarray(CAPTURE_ROLES),
    )
    answer_role = ROLE_INDEX["answer_boundary"]
    np.savez_compressed(
        panel_root / "answer_head_direction_sums.npz",
        canonical_sum=direction_sum_head[
            1, :, answer_role
        ],
        count=direction_count_head[:, answer_role],
        operation_names=np.asarray(TARGET_OPERATIONS),
    )
    if len(frozen_head_indices):
        np.savez_compressed(
            panel_root / "phase1014_frozen_role_direction_sums.npz",
            canonical_sum=direction_sum_head[
                1, :, :, frozen_head_indices
            ],
            count=direction_count_head[
                :, :, frozen_head_indices
            ],
            frozen_head_indices=frozen_head_indices,
            frozen_event_ids=np.asarray([
                row["event_id"] for row in frozen_rows
            ]),
            frozen_operations=np.asarray([
                row["operation"] for row in frozen_rows
            ]),
            operation_names=np.asarray(TARGET_OPERATIONS),
            role_names=np.asarray(CAPTURE_ROLES),
        )
    write_jsonl(panel_root / "units.jsonl", unit_rows)
    summary = {
        "schema_version": "phase1015_chain_scan_panel.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "family": family,
        "query_surface": int(surface),
        "query_surface_name": panel_units[0][
            "query_surface_name"
        ],
        "query_surface_class": panel_units[0][
            "query_surface_class"
        ],
        "balanced_query_inventory": bool(
            panel_units[0]["balanced_query_inventory"]
        ),
        "split": panel_units[0]["split"],
        "unit_count": unit_count,
        "event_count": event_count,
        "whole_event_count": whole_count,
        "head_event_count": head_event_count,
        "role_count": role_count,
        "singleton_forward_count": singleton_forward_count,
        "identity_maximum": identity_maximum,
        "q_causal_prefix_maximum": q_prefix_maximum,
        "panel_state_hit_rates": {
            state: float(np.mean([
                row["singleton_state_hits"][state]
                for row in unit_rows
            ]))
            for state in STATE_ORDER
        },
        "elapsed_seconds": time.time() - started,
        "claim_limits": [
            "role order and depth response are co-response, not an edge",
            "Q prefix zero is an instrument and causal-mask audit",
            "pre-o_proj head response is not a residual write direction",
        ],
    }
    write_json(panel_root / "summary.json", summary)
    return summary


def run_model(
    model_name: str,
    *,
    output_namespace: str,
    max_panels: int | None,
    use_8bit: bool,
    resume: bool,
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1015 protocol revision drift")
    units = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "units.jsonl"
    )
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    case_by_id = {row["record_id"]: row for row in cases}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[(unit["family"], int(unit["query_surface"]))].append(unit)
    panel_items = sorted(grouped.items())
    if max_panels is not None:
        panel_items = panel_items[:max_panels]

    output_root = OUT_ROOT / output_namespace
    output_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    state_capture = None
    head_capture = None
    started = time.time()
    panel_summaries = []
    try:
        placement = {}
        if use_8bit:
            model, tokenizer, device = load_model(
                model_name,
                use_8bit=True,
            )
        else:
            model, tokenizer, device, placement = load_bf16(model_name)
        info = get_model_info(model, model_name)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        events, whole_keys, head_keys = event_definitions(
            int(info.n_layers),
            head_count,
        )
        write_jsonl(output_root / model_name / "events.jsonl", events)
        frozen_rows = load_phase1014_frozen(model_name)
        write_jsonl(
            output_root / model_name / "phase1014_frozen_events.jsonl",
            frozen_rows,
        )
        state_capture = StateCapture(model, layers)
        head_capture = MultiPositionHeadCapture(layers, head_count)
        state_capture.register()
        head_capture.register()
        for (family, surface), panel_units in panel_items:
            panel_root = (
                output_root
                / model_name
                / family
                / f"surface_{surface}"
            )
            panel_summary_path = panel_root / "summary.json"
            required_panel_files = (
                panel_root / "response_scalars.npz",
                panel_root / "direction_consistency.npz",
                panel_root / "answer_head_direction_sums.npz",
                panel_root / "units.jsonl",
            )
            if (
                resume
                and panel_summary_path.exists()
                and all(path.exists() for path in required_panel_files)
            ):
                existing = read_json(panel_summary_path)
                if (
                    int(existing["phase"]) == PHASE
                    and int(existing["protocol_revision"])
                    == PROTOCOL_REVISION
                    and existing["model"] == model_name
                    and existing["family"] == family
                    and int(existing["query_surface"]) == int(surface)
                    and int(existing["unit_count"]) == len(panel_units)
                    and int(existing["event_count"]) == len(events)
                ):
                    print(
                        f"[resume] {model_name}/{family}/"
                        f"surface{surface}",
                        flush=True,
                    )
                    panel_summaries.append(existing)
                    continue
            panel_summaries.append(run_panel(
                model=model,
                layers=layers,
                info=info,
                head_count=head_count,
                device=device,
                model_name=model_name,
                family=family,
                surface=surface,
                panel_units=panel_units,
                case_by_id=case_by_id,
                output_root=output_root,
                events=events,
                whole_keys=whole_keys,
                head_keys=head_keys,
                frozen_rows=frozen_rows,
                state_capture=state_capture,
                head_capture=head_capture,
            ))
        summary = {
            "schema_version": "phase1015_chain_scan_model.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": protocol["preregistration_digest"],
            "model": model_name,
            "output_namespace": output_namespace,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": head_count,
                "pre_o_proj_head_width": int(
                    layers[0].self_attn.o_proj.in_features
                    // head_count
                ),
                "model_class": info.model_class,
                "loaded_8bit": bool(use_8bit),
                "placement": placement,
            },
            "panel_count": len(panel_summaries),
            "event_count": len(events),
            "whole_event_count": len(whole_keys),
            "head_event_count": len(head_keys),
            "role_count": len(CAPTURE_ROLES),
            "phase1014_frozen_event_count": len(frozen_rows),
            "singleton_forward_count": sum(
                row["singleton_forward_count"]
                for row in panel_summaries
            ),
            "identity_maximum": max(
                row["identity_maximum"]
                for row in panel_summaries
            ),
            "q_causal_prefix_maximum": max(
                row["q_causal_prefix_maximum"]
                for row in panel_summaries
            ),
            "panel_summaries": panel_summaries,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / model_name / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if head_capture is not None:
            head_capture.close()
        if state_capture is not None:
            state_capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--output-namespace", default="scan")
    parser.add_argument("--max-panels", type=int)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="disable 8bit loading for a precision smoke test",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse fully written panels with matching protocol metadata",
    )
    args = parser.parse_args()
    run_model(
        args.model,
        output_namespace=args.output_namespace,
        max_panels=args.max_panels,
        use_8bit=not args.bf16,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
