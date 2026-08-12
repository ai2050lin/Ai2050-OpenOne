#!/usr/bin/env python3
"""Behavior-stratified targeted rescan for frozen Phase1016 components."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model
from phase1008_global_response_atlas_scan import StateCapture
from phase1009_crossfamily_response_protocol import digest
from phase1014_bf16_precision_confirmation import load_bf16
from phase1016_query_factorial_protocol import (
    FACTORIAL_STATES,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1016_query_factorial_scan import prediction


ANALYSIS_ROOT = OUT_ROOT / "analysis"
TARGET_ROOT = OUT_ROOT / "targeted_behavior_scan"
TARGET_ROLES = ("query_operator", "answer_boundary")
POPULATIONS = ("all", "factorial_correct", "factorial_failed")
CONTRASTS = ("S", "S_L0", "S_L1")
EPSILON = 1e-12


class BatchHeadCapture:
    """Capture all real pre-o_proj heads at batch-specific positions."""

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
                raise RuntimeError("head positions are not set")
            positions = self.positions.to(value.device)
            batch_index = torch.arange(
                value.shape[0],
                device=value.device,
            )[:, None]
            selected = value[batch_index, positions, :]
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
        self.values = {}
        self.positions = None


def freeze_selection() -> dict[str, Any]:
    metrics = read_jsonl(ANALYSIS_ROOT / "event_role_metrics.jsonl")
    selections = []
    for model_name in MODELS:
        for role in TARGET_ROLES:
            grouped = defaultdict(list)
            for row in metrics:
                if (
                    row["model"] == model_name
                    and row["split"] == "discovery"
                    and row["role"] == role
                    and row["observation_fixed_orientation_candidate"]
                ):
                    grouped[row["event_id"]].append(row)
            eligible = []
            for event_id, rows in grouped.items():
                panels = {
                    (row["family"], row["template"]) for row in rows
                }
                families = {row["family"] for row in rows}
                if len(panels) < 4 or len(families) < 2:
                    continue
                exemplar = rows[0]
                eligible.append({
                    "schema_version": (
                        "phase1016_targeted_selection_event.v1"
                    ),
                    "phase": PHASE,
                    "protocol_revision": PROTOCOL_REVISION,
                    "model": model_name,
                    "event_id": event_id,
                    "component": exemplar["component"],
                    "depth": int(exemplar["depth"]),
                    "relative_depth": float(
                        exemplar["relative_depth"]
                    ),
                    "head": exemplar["head"],
                    "role": role,
                    "discovery_panel_count": len(panels),
                    "discovery_family_count": len(families),
                    "discovery_median_raw_consistency": float(
                        np.median([
                            row["raw_semantic_direction_consistency"]
                            for row in rows
                        ])
                    ),
                    "discovery_median_lexical_alignment": float(
                        np.median([
                            row["lexical_family_direction_alignment"]
                            for row in rows
                        ])
                    ),
                    "discovery_median_semantic_prevalence": float(
                        np.median([
                            row["semantic_over_lexical_prevalence"]
                            for row in rows
                        ])
                    ),
                    "discovery_median_semantic_minus_lexical": float(
                        np.median([
                            row["semantic_minus_lexical_median"]
                            for row in rows
                        ])
                    ),
                })
            def rank(row: dict[str, Any]) -> tuple[Any, ...]:
                return (
                    row["discovery_family_count"],
                    row["discovery_panel_count"],
                    row["discovery_median_raw_consistency"],
                    row["discovery_median_lexical_alignment"],
                    row["discovery_median_semantic_minus_lexical"],
                )

            heads = sorted(
                [
                    row for row in eligible
                    if row["component"] == "attention_head_pre_o_proj"
                ],
                key=rank,
                reverse=True,
            )[:4]
            whole = sorted(
                [
                    row for row in eligible
                    if row["component"] != "attention_head_pre_o_proj"
                ],
                key=rank,
                reverse=True,
            )[:2]
            selections.extend(heads + whole)
    payload = {
        "schema_version": "phase1016_targeted_selection.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "selection_source": "discovery_event_role_metrics_only",
        "confirmation_metrics_used": False,
        "behavior_labels_used": False,
        "selection_count": len(selections),
        "selection_count_by_model": dict(
            __import__("collections").Counter(
                row["model"] for row in selections
            )
        ),
        "selections": selections,
    }
    payload["selection_digest"] = digest(payload)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(TARGET_ROOT / "selection.json", payload)
    write_jsonl(TARGET_ROOT / "selection.jsonl", selections)
    return payload


def direction_consistency(total: np.ndarray, count: int) -> float | None:
    if count < 2:
        return None
    squared = float(np.dot(total, total))
    return (squared - count) / (count * (count - 1))


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return None
    return float(np.dot(left, right) / denominator)


def selected_value(
    selection: dict[str, Any],
    state_capture: StateCapture,
    head_capture: BatchHeadCapture,
    role_index: int,
) -> torch.Tensor:
    component = selection["component"]
    depth = int(selection["depth"])
    if component == "residual":
        value = state_capture.captured[("residual", depth)][
            :, role_index
        ]
    elif component == "attention_output":
        value = state_capture.captured[("attention_output", depth)][
            :, role_index
        ]
    elif component == "mlp_output":
        value = state_capture.captured[("mlp_output", depth)][
            :, role_index
        ]
    elif component == "attention_head_pre_o_proj":
        value = head_capture.values[depth][
            :, role_index, int(selection["head"])
        ]
    else:
        raise RuntimeError(f"unknown component {component}")
    return value.float().cpu()


def run_model(model_name: str) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    selection_payload = read_json(TARGET_ROOT / "selection.json")
    selections = [
        row for row in selection_payload["selections"]
        if row["model"] == model_name
    ]
    calibration = read_json(
        OUT_ROOT
        / "behavior_calibration"
        / model_name
        / "selection.json"
    )
    prompt_mode = calibration["selected_prompt_mode"]
    units = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"units.{model_name}.{prompt_mode}.jsonl"
    )
    cases = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    case_by_id = {row["record_id"]: row for row in cases}
    role_index = {role: index for index, role in enumerate(TARGET_ROLES)}
    sums: dict[tuple, np.ndarray] = {}
    counts = Counter()
    magnitudes = defaultdict(list)
    behavior_rows = []
    model = tokenizer = device = None
    state_capture = head_capture = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        head_count = int(model.config.num_attention_heads)
        state_capture = StateCapture(model, layers)
        head_capture = BatchHeadCapture(layers, head_count)
        state_capture.register()
        head_capture.register()
        for unit_index, unit in enumerate(units):
            state_cases = [
                case_by_id[unit["record_ids"][state]]
                for state in FACTORIAL_STATES
            ]
            widths = {len(row["input_ids"]) for row in state_cases}
            if len(widths) != 1:
                raise RuntimeError(f"{unit['unit_id']}: width drift")
            input_ids = torch.tensor(
                [row["input_ids"] for row in state_cases],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            positions = torch.tensor(
                [
                    [
                        int(case["role_positions"][role])
                        for role in TARGET_ROLES
                    ]
                    for case in state_cases
                ],
                dtype=torch.long,
                device=device,
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
                state_behavior = {
                    state: prediction(
                        output.logits[index, -1].float(),
                        state_cases[index],
                    )
                    for index, state in enumerate(FACTORIAL_STATES)
                }
                correct = all(
                    row["candidate_hit"]
                    for row in state_behavior.values()
                )
                population = (
                    "factorial_correct"
                    if correct
                    else "factorial_failed"
                )
                behavior_rows.append({
                    "schema_version": (
                        "phase1016_targeted_behavior_unit.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "unit_id": unit["unit_id"],
                    "split": unit["split"],
                    "family": unit["family"],
                    "template": int(unit["template"]),
                    "factorial_correct": bool(correct),
                    "state_behavior": state_behavior,
                })
                for target_index, selection in enumerate(selections):
                    values = selected_value(
                        selection,
                        state_capture,
                        head_capture,
                        role_index[selection["role"]],
                    )
                    h00, h10, h01, h11 = values
                    deltas = {
                        "S_L0": h10 - h00,
                        "S_L1": h11 - h01,
                    }
                    deltas["S"] = 0.5 * (
                        deltas["S_L0"] + deltas["S_L1"]
                    )
                    lexical = 0.5 * (
                        (h01 - h00) + (h11 - h10)
                    )
                    state_scale = torch.mean(
                        torch.linalg.vector_norm(values, dim=-1)
                    )
                    for contrast, delta in deltas.items():
                        norm = float(
                            torch.linalg.vector_norm(delta).item()
                        )
                        if norm <= EPSILON:
                            continue
                        direction = (
                            delta / norm
                        ).numpy().astype(np.float64, copy=False)
                        for family_scope in (
                            "all_families",
                            unit["family"],
                        ):
                            for pop in ("all", population):
                                key = (
                                    unit["split"],
                                    family_scope,
                                    pop,
                                    target_index,
                                    contrast,
                                )
                                if key not in sums:
                                    sums[key] = np.zeros_like(direction)
                                sums[key] += direction
                                counts[key] += 1
                    semantic_norm = float(
                        torch.linalg.vector_norm(deltas["S"]).item()
                        / max(float(state_scale.item()), EPSILON)
                    )
                    lexical_norm = float(
                        torch.linalg.vector_norm(lexical).item()
                        / max(float(state_scale.item()), EPSILON)
                    )
                    for family_scope in (
                        "all_families",
                        unit["family"],
                    ):
                        for pop in ("all", population):
                            magnitudes[(
                                unit["split"],
                                family_scope,
                                pop,
                                target_index,
                                "S",
                            )].append(semantic_norm)
                            magnitudes[(
                                unit["split"],
                                family_scope,
                                pop,
                                target_index,
                                "L",
                            )].append(lexical_norm)
                del output
            finally:
                state_capture.captured = {}
                head_capture.values = {}
                del input_ids, attention_mask, positions
            if (unit_index + 1) % 40 == 0:
                print(
                    f"[targeted] {model_name} "
                    f"{unit_index + 1}/{len(units)}",
                    flush=True,
                )

        result_rows = []
        for target_index, selection in enumerate(selections):
            for split in ("discovery", "confirmation"):
                for family_scope in ("all_families",) + FAMILIES:
                    correct_sum = sums.get((
                        split,
                        family_scope,
                        "factorial_correct",
                        target_index,
                        "S",
                    ))
                    failed_sum = sums.get((
                        split,
                        family_scope,
                        "factorial_failed",
                        target_index,
                        "S",
                    ))
                    correct_failed_cosine = (
                        None
                        if correct_sum is None or failed_sum is None
                        else cosine(correct_sum, failed_sum)
                    )
                    for population in POPULATIONS:
                        s_key = (
                            split,
                            family_scope,
                            population,
                            target_index,
                            "S",
                        )
                        l0_key = (
                            split,
                            family_scope,
                            population,
                            target_index,
                            "S_L0",
                        )
                        l1_key = (
                            split,
                            family_scope,
                            population,
                            target_index,
                            "S_L1",
                        )
                        s_count = counts[s_key]
                        l0_sum = sums.get(l0_key)
                        l1_sum = sums.get(l1_key)
                        s_values = magnitudes.get((
                            split,
                            family_scope,
                            population,
                            target_index,
                            "S",
                        ), [])
                        l_values = magnitudes.get((
                            split,
                            family_scope,
                            population,
                            target_index,
                            "L",
                        ), [])
                        result_rows.append({
                        **selection,
                        "schema_version": (
                            "phase1016_targeted_direction_result.v1"
                        ),
                        "phase": PHASE,
                        "protocol_revision": PROTOCOL_REVISION,
                        "split": split,
                        "family_scope": family_scope,
                        "population": population,
                        "n": int(s_count),
                        "semantic_direction_consistency": (
                            None
                            if s_key not in sums
                            else direction_consistency(
                                sums[s_key],
                                s_count,
                            )
                        ),
                        "lexical_family_direction_alignment": (
                            None
                            if l0_sum is None or l1_sum is None
                            else cosine(l0_sum, l1_sum)
                        ),
                        "semantic_median": (
                            None
                            if not s_values
                            else float(np.median(s_values))
                        ),
                        "lexical_median": (
                            None
                            if not l_values
                            else float(np.median(l_values))
                        ),
                        "semantic_over_lexical_prevalence": (
                            None
                            if not s_values
                            else float(np.mean(
                                np.asarray(s_values)
                                > np.asarray(l_values)
                            ))
                        ),
                        "correct_failed_mean_direction_cosine": (
                            correct_failed_cosine
                        ),
                        "selection_used_behavior": False,
                        "causal_claim": False,
                        })
        output_root = TARGET_ROOT / model_name
        output_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_root / "direction_results.jsonl", result_rows)
        write_jsonl(output_root / "behavior_units.jsonl", behavior_rows)
        summary = {
            "schema_version": "phase1016_targeted_model_summary.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "selection_digest": selection_payload["selection_digest"],
            "model": model_name,
            "prompt_mode": prompt_mode,
            "precision": "bf16",
            "selection_count": len(selections),
            "unit_count": len(units),
            "batched_forward_count": len(units),
            "factorial_case_count": len(units) * len(FACTORIAL_STATES),
            "factorial_correct_count": int(sum(
                row["factorial_correct"] for row in behavior_rows
            )),
            "result_row_count": len(result_rows),
            "placement": placement,
            "model_info": {
                "layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": head_count,
            },
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if head_capture is not None:
            head_capture.close()
        if state_capture is not None:
            state_capture.close()
        if model is not None:
            release_model(model)
        del model, tokenizer, state_capture, head_capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-selection", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.freeze_selection:
        payload = freeze_selection()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.model is None:
        parser.error("--model is required unless --freeze-selection is used")
    run_model(args.model)


if __name__ == "__main__":
    main()
