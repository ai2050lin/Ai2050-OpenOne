#!/usr/bin/env python3
"""Scan Phase1018 branch responses in all real Transformer components."""

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

from model_utils import get_layers, get_model_info, release_model
from phase1014_bf16_precision_confirmation import load_bf16
from phase1018_language_pattern_protocol import (
    CAPTURE_ROLES,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    STATES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


STATE_INDEX = {state: index for index, state in enumerate(STATES)}
ROLE_INDEX = {role: index for index, role in enumerate(CAPTURE_ROLES)}
CONTRASTS = ("D", "D_L0", "D_L1", "S", "X", "I")
CONTRAST_INDEX = {name: index for index, name in enumerate(CONTRASTS)}
EPSILON = 1e-12


def event_definitions(
    n_layers: int,
    head_count: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, int]], list[tuple[int, int]]]:
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
            "schema_version": "phase1018_pattern_event.v1",
            "phase": PHASE,
            "event_index": len(events),
            "event_id": f"{component}.d{depth:02d}",
            "component": component,
            "depth": int(depth),
            "relative_depth": float(depth / max(n_layers, 1)),
            "head": None,
            "vector_space": "model_width",
        })
    head_keys = []
    for depth in range(1, n_layers + 1):
        for head in range(head_count):
            head_keys.append((depth, head))
            events.append({
                "schema_version": "phase1018_pattern_event.v1",
                "phase": PHASE,
                "event_index": len(events),
                "event_id": f"attention_head.d{depth:02d}.h{head:02d}",
                "component": "attention_head_pre_o_proj",
                "depth": int(depth),
                "relative_depth": float(depth / max(n_layers, 1)),
                "head": int(head),
                "vector_space": "head_width",
            })
    return events, whole_keys, head_keys


class BatchRoleStateCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.positions: torch.Tensor | None = None
        self.values: dict[tuple[str, int], torch.Tensor] = {}
        self.counts: dict[tuple[str, int], int] = defaultdict(int)
        self.handles = []

    def _hook(self, component: str, depth: int):
        key = (component, depth)

        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError("state capture was not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[key] = value[batch, positions, :].detach()
            self.counts[key] += 1
            return output

        return hook

    def register(self) -> None:
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(
                self._hook("residual", 0)
            )
        )
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(self._hook("residual", depth))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook("attention_output", depth)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._hook("mlp_output", depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
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
        missing = expected - set(self.values)
        repeated = {
            str(key): count for key, count in self.counts.items() if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"state capture drift missing={list(missing)[:4]} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


class BatchRoleHeadCapture:
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
                raise RuntimeError("head capture was not initialized")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            selected = value[batch, positions, :]
            if selected.shape[-1] % self.head_count:
                raise RuntimeError("attention width is not head aligned")
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
        missing = expected - set(self.values)
        repeated = {
            depth: count for depth, count in self.counts.items() if count != 1
        }
        if missing or repeated:
            raise RuntimeError(
                f"head capture drift missing={list(missing)[:4]} "
                f"repeated={repeated}"
            )

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


def contrast_values(
    values: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    b00 = values[STATE_INDEX["b0_l0"]].float()
    b10 = values[STATE_INDEX["b1_l0"]].float()
    b01 = values[STATE_INDEX["b0_l1"]].float()
    b11 = values[STATE_INDEX["b1_l1"]].float()
    identity = values[STATE_INDEX["identity"]].float()
    d_l0 = b10 - b00
    d_l1 = b11 - b01
    deltas = {
        "D": 0.5 * (d_l0 + d_l1),
        "D_L0": d_l0,
        "D_L1": d_l1,
        "S": 0.5 * ((b01 - b00) + (b11 - b10)),
        "X": d_l1 - d_l0,
        "I": identity - b00,
    }
    norms = {
        name: torch.stack([
            torch.linalg.vector_norm(state, dim=-1)
            for state in states
        ]).mean(dim=0)
        for name, states in {
            "D": (b00, b10, b01, b11),
            "D_L0": (b00, b10),
            "D_L1": (b01, b11),
            "S": (b00, b10, b01, b11),
            "X": (b00, b10, b01, b11),
            "I": (identity, b00),
        }.items()
    }
    return deltas, norms


def add_unit_directions(
    delta: torch.Tensor,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    norms = torch.linalg.vector_norm(delta, dim=-1)
    valid = norms > EPSILON
    units = torch.zeros_like(delta)
    units[valid] = delta[valid] / norms[valid, None]
    sums += units.cpu().numpy()
    counts += valid.cpu().numpy().astype(np.int32)


def direction_consistency(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    squared = np.einsum(
        "...d,...d->...",
        sums.astype(np.float64, copy=False),
        sums.astype(np.float64, copy=False),
    )
    result = np.full(counts.shape, np.nan, dtype=np.float32)
    valid = counts >= 2
    result[valid] = (
        (squared[valid] - counts[valid])
        / (counts[valid] * (counts[valid] - 1.0))
    ).astype(np.float32)
    return result


def cosine_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("...d,...d->...", a, b, dtype=np.float64)
    denominator = np.sqrt(
        np.einsum("...d,...d->...", a, a, dtype=np.float64)
        * np.einsum("...d,...d->...", b, b, dtype=np.float64)
    )
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = denominator > EPSILON
    result[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return result


def normalized_directions(sums: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(sums.astype(np.float64), axis=-1, keepdims=True)
    result = np.zeros_like(sums, dtype=np.float32)
    np.divide(sums, norms, out=result, where=norms > EPSILON)
    return result


def pad_states(
    cases: list[dict[str, Any]],
    pad_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(case["input_ids"]) for case in cases)
    input_ids = torch.full(
        (len(cases), width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, case in enumerate(cases):
        length = len(case["input_ids"])
        input_ids[index, :length] = torch.tensor(
            case["input_ids"],
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, :length] = 1
    positions = torch.tensor([
        [int(case["role_positions"][role]) for role in CAPTURE_ROLES]
        for case in cases
    ], dtype=torch.long, device=device)
    return input_ids, attention_mask, positions


def behavior_by_unit(model_name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(OUT_ROOT / "behavior" / model_name / "formal.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["unit_id"]].append(row)
    result = {}
    for unit_id, values in grouped.items():
        by_state = {row["state"]: row for row in values}
        ordered = [by_state[state] for state in STATES if state != "identity"]
        result[unit_id] = {
            "candidate_hit_count": int(sum(
                row["candidate_hit"] for row in ordered
            )),
            "candidate_all_hit": bool(all(
                row["candidate_hit"] for row in ordered
            )),
            "first_token_hit_count": int(sum(
                row["first_token_hit"] for row in ordered
            )),
            "first_token_all_hit": bool(all(
                row["first_token_hit"] for row in ordered
            )),
            "state_candidate_margin": {
                state: float(by_state[state]["candidate_margin"])
                for state in by_state
            },
        }
    return result


def run_panel(
    *,
    model,
    device,
    tokenizer,
    model_name: str,
    prompt_mode: str,
    family: str,
    item_id: str,
    split: str,
    units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    behavior: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
    state_capture: BatchRoleStateCapture,
    head_capture: BatchRoleHeadCapture,
    output_root: Path,
) -> dict[str, Any]:
    unit_count = len(units)
    role_count = len(CAPTURE_ROLES)
    whole_count = len(whole_keys)
    head_count_events = len(head_keys)
    event_count = len(events)
    d_model = int(model.config.hidden_size)
    attention_width = int(
        get_layers(model)[0].self_attn.o_proj.in_features
    )
    physical_heads = int(model.config.num_attention_heads)
    head_width = attention_width // physical_heads
    scalars = np.full(
        (unit_count, len(CONTRASTS), role_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    whole_sums = {
        name: np.zeros(
            (role_count, whole_count, d_model), dtype=np.float32
        )
        for name in ("D", "D_L0", "D_L1")
    }
    head_sums = {
        name: np.zeros(
            (role_count, head_count_events, head_width), dtype=np.float32
        )
        for name in ("D", "D_L0", "D_L1")
    }
    whole_counts = {
        name: np.zeros((role_count, whole_count), dtype=np.int32)
        for name in ("D", "D_L0", "D_L1")
    }
    head_counts = {
        name: np.zeros((role_count, head_count_events), dtype=np.int32)
        for name in ("D", "D_L0", "D_L1")
    }
    unit_rows = []
    identity_maximum = 0.0
    prefix_branch_maximum = 0.0
    started = time.time()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    for unit_index, unit in enumerate(units):
        cases = [case_by_id[unit["record_ids"][state]] for state in STATES]
        input_ids, attention_mask, positions = pad_states(
            cases, int(pad_id), device
        )
        state_capture.begin(positions)
        head_capture.begin(positions)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            state_capture.validate()
            head_capture.validate()
            whole_values = torch.stack([
                state_capture.values[key] for key in whole_keys
            ], dim=2)
            head_values = torch.stack([
                head_capture.values[depth][:, :, head]
                for depth, head in head_keys
            ], dim=2)
            whole_deltas, whole_scales = contrast_values(whole_values)
            head_deltas, head_scales = contrast_values(head_values)
            for name in CONTRASTS:
                contrast_index = CONTRAST_INDEX[name]
                whole_raw = torch.linalg.vector_norm(
                    whole_deltas[name], dim=-1
                )
                head_raw = torch.linalg.vector_norm(
                    head_deltas[name], dim=-1
                )
                scalars[
                    unit_index,
                    contrast_index,
                    :,
                    :whole_count,
                ] = (
                    whole_raw
                    / torch.clamp(whole_scales[name], min=EPSILON)
                ).cpu().numpy()
                scalars[
                    unit_index,
                    contrast_index,
                    :,
                    whole_count:,
                ] = (
                    head_raw
                    / torch.clamp(head_scales[name], min=EPSILON)
                ).cpu().numpy()
                if name == "I":
                    identity_maximum = max(
                        identity_maximum,
                        float(whole_raw.max().item()),
                        float(head_raw.max().item()),
                    )
                if name == "D":
                    prefix = ROLE_INDEX["prefix_anchor"]
                    prefix_branch_maximum = max(
                        prefix_branch_maximum,
                        float(whole_raw[prefix].max().item()),
                        float(head_raw[prefix].max().item()),
                    )
            for name in ("D", "D_L0", "D_L1"):
                add_unit_directions(
                    whole_deltas[name],
                    whole_sums[name],
                    whole_counts[name],
                )
                add_unit_directions(
                    head_deltas[name],
                    head_sums[name],
                    head_counts[name],
                )
            unit_rows.append({
                "schema_version": "phase1018_pattern_scan_unit.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model_name,
                "prompt_mode": prompt_mode,
                "family": family,
                "item_id": item_id,
                "subgroup": cases[0]["subgroup"],
                "split": split,
                "template": int(unit["template"]),
                "world": int(unit["world"]),
                "unit_id": unit["unit_id"],
                **behavior[unit["unit_id"]],
            })
            del (
                output,
                whole_values,
                head_values,
                whole_deltas,
                whole_scales,
                head_deltas,
                head_scales,
            )
        finally:
            state_capture.values = {}
            head_capture.values = {}
            del input_ids, attention_mask, positions

    whole_consistency = direction_consistency(
        whole_sums["D"], whole_counts["D"]
    )
    head_consistency = direction_consistency(
        head_sums["D"], head_counts["D"]
    )
    whole_surface_alignment = cosine_arrays(
        whole_sums["D_L0"], whole_sums["D_L1"]
    )
    head_surface_alignment = cosine_arrays(
        head_sums["D_L0"], head_sums["D_L1"]
    )
    whole_direction = normalized_directions(whole_sums["D"]).astype(
        np.float16
    )
    head_direction = normalized_directions(head_sums["D"]).astype(np.float16)

    panel_root = output_root / model_name / family / item_id / split
    panel_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_root / "response_scalars.npz",
        normalized_magnitude=scalars,
        contrast_names=np.asarray(CONTRASTS),
        role_names=np.asarray(CAPTURE_ROLES),
    )
    np.savez_compressed(
        panel_root / "direction_metrics.npz",
        whole_consistency=whole_consistency,
        head_consistency=head_consistency,
        whole_surface_alignment=whole_surface_alignment,
        head_surface_alignment=head_surface_alignment,
        whole_count=whole_counts["D"],
        head_count=head_counts["D"],
        role_names=np.asarray(CAPTURE_ROLES),
    )
    np.savez_compressed(
        panel_root / "directions.npz",
        whole_direction=whole_direction,
        head_direction=head_direction,
        role_names=np.asarray(CAPTURE_ROLES),
        storage_dtype="float16",
    )
    write_jsonl(panel_root / "units.jsonl", unit_rows)
    summary = {
        "schema_version": "phase1018_pattern_scan_panel.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "family": family,
        "item_id": item_id,
        "subgroup": unit_rows[0]["subgroup"],
        "split": split,
        "unit_count": unit_count,
        "batched_forward_count": unit_count,
        "state_case_count": unit_count * len(STATES),
        "event_count": event_count,
        "identity_maximum": identity_maximum,
        "prefix_branch_maximum": prefix_branch_maximum,
        "candidate_all_hit_count": int(sum(
            row["candidate_all_hit"] for row in unit_rows
        )),
        "first_token_all_hit_count": int(sum(
            row["first_token_all_hit"] for row in unit_rows
        )),
        "elapsed_seconds": time.time() - started,
        "claim_limits": [
            "D is a branch-response measurement, not a mechanism equation.",
            "Stored directions are aggregate response directions, not raw states.",
            "Repeated response does not establish causal use.",
        ],
    }
    write_json(panel_root / "summary.json", summary)
    return summary


def run_model(model_name: str, *, resume: bool) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    selection = read_json(OUT_ROOT / "behavior" / model_name / "selection.json")
    if selection["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    selected_modes = selection["selected_by_family"]
    cases = []
    units = []
    for family in FAMILIES:
        mode = selected_modes[family]
        family_cases = read_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.{mode}.jsonl"
        )
        family_units = read_jsonl(
            OUT_ROOT / "protocol" / f"units.{model_name}.{mode}.jsonl"
        )
        cases.extend([row for row in family_cases if row["family"] == family])
        units.extend([row for row in family_units if row["family"] == family])
    case_by_id = {row["record_id"]: row for row in cases}
    behavior = behavior_by_unit(model_name)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[(unit["family"], unit["item_id"], unit["split"])].append(unit)
    panel_items = sorted(grouped.items())

    output_root = OUT_ROOT / "formal_scan"
    model_root = output_root / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    state_capture = head_capture = None
    summaries = []
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        info = get_model_info(model, model_name)
        layers = get_layers(model)
        physical_heads = int(model.config.num_attention_heads)
        events, whole_keys, head_keys = event_definitions(
            int(info.n_layers), physical_heads
        )
        write_jsonl(model_root / "events.jsonl", events)
        state_capture = BatchRoleStateCapture(model, layers)
        head_capture = BatchRoleHeadCapture(layers, physical_heads)
        state_capture.register()
        head_capture.register()
        for panel_index, (key, panel_units) in enumerate(panel_items, 1):
            family, item_id, split = key
            panel_root = model_root / family / item_id / split
            required = (
                panel_root / "summary.json",
                panel_root / "response_scalars.npz",
                panel_root / "direction_metrics.npz",
                panel_root / "directions.npz",
                panel_root / "units.jsonl",
            )
            if resume and all(path.exists() for path in required):
                existing = read_json(panel_root / "summary.json")
                if (
                    int(existing["protocol_revision"]) == PROTOCOL_REVISION
                    and int(existing["unit_count"]) == len(panel_units)
                ):
                    summaries.append(existing)
                    print(
                        f"[resume] {model_name}/{family}/{item_id}/{split}",
                        flush=True,
                    )
                    continue
            summary = run_panel(
                model=model,
                device=device,
                tokenizer=tokenizer,
                model_name=model_name,
                prompt_mode=selected_modes[family],
                family=family,
                item_id=item_id,
                split=split,
                units=panel_units,
                case_by_id=case_by_id,
                behavior=behavior,
                events=events,
                whole_keys=whole_keys,
                head_keys=head_keys,
                state_capture=state_capture,
                head_capture=head_capture,
                output_root=output_root,
            )
            summaries.append(summary)
            print(
                f"[scan] {model_name} panel={panel_index}/{len(panel_items)} "
                f"{family}/{item_id}/{split}",
                flush=True,
            )
        model_summary = {
            "schema_version": "phase1018_pattern_scan_model.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "selected_modes": selected_modes,
            "precision": "bf16",
            "placement": placement,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": physical_heads,
                "head_width": int(
                    layers[0].self_attn.o_proj.in_features // physical_heads
                ),
            },
            "panel_count": len(summaries),
            "unit_count": int(sum(row["unit_count"] for row in summaries)),
            "batched_forward_count": int(sum(
                row["batched_forward_count"] for row in summaries
            )),
            "state_case_count": int(sum(
                row["state_case_count"] for row in summaries
            )),
            "identity_maximum": float(max(
                row["identity_maximum"] for row in summaries
            )),
            "prefix_branch_maximum": float(max(
                row["prefix_branch_maximum"] for row in summaries
            )),
            "by_family": {
                family: {
                    "panel_count": sum(
                        row["family"] == family for row in summaries
                    ),
                    "unit_count": sum(
                        row["unit_count"]
                        for row in summaries
                        if row["family"] == family
                    ),
                }
                for family in FAMILIES
            },
            "elapsed_seconds": time.time() - started,
        }
        write_json(model_root / "summary.json", model_summary)
        print(json.dumps(model_summary, ensure_ascii=False, indent=2))
        return model_summary
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
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_model(args.model, resume=args.resume)


if __name__ == "__main__":
    main()
