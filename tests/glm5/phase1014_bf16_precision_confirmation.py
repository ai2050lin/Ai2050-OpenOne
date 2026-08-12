#!/usr/bin/env python3
"""BF16 held-out confirmation of discovery-frozen relative directions."""

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

from model_utils import MODEL_CONFIGS, get_layers, load_model
from phase1008_global_response_atlas_scan import StateCapture
from phase1014_relative_difference_scan import AllHeadCapture


SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
PROTOCOL_ROOT = SOURCE_ROOT / "precision_protocol"
OUT_ROOT = SOURCE_ROOT / "precision_bf16"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
OUTPUT_MODES = ("entity", "property", "binary")
STATE_ORDER = ("base", "F", "Q", "E", "N", "L", "identity")
MATCHED_CONTROLS = {"F": ("E", "N"), "Q": ("L",)}
TARGET_INDEX = {"F": 0, "Q": 1}
CONFIRMATION_SPLIT_INDEX = 1
EPSILON = 1e-12
DIRECTION_THRESHOLD = 0.50
ORIENTATION_GAIN = 0.30
PREVALENCE_THRESHOLD = 0.70
PANEL_COUNT_THRESHOLD = 4
CROSS_PANEL_THRESHOLD = 0.30
PRECISION_COSINE_THRESHOLD = 0.50


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def direction_consistency(total: np.ndarray, count: int) -> float | None:
    if count < 2:
        return None
    return finite(
        (float(np.dot(total, total)) - count)
        / (count * (count - 1))
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPSILON:
        return None
    return finite(float(np.dot(left, right)) / denominator)


def pairwise_consistency(vectors: list[np.ndarray]) -> float | None:
    normalized = []
    for value in vectors:
        norm = float(np.linalg.norm(value))
        if norm > EPSILON:
            normalized.append(value.astype(np.float64, copy=False) / norm)
    count = len(normalized)
    if count < 2:
        return None
    total = np.sum(normalized, axis=0)
    return direction_consistency(total, count)


def load_bf16(model_name: str):
    if model_name == "qwen3":
        model, tokenizer, device = load_model(
            model_name,
            dtype=torch.bfloat16,
            use_8bit=False,
        )
        return model, tokenizer, device, {
            "placement": "full_cuda",
            "max_memory": None,
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = MODEL_CONFIGS[model_name]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_memory = {0: "11GiB", "cpu": "24GiB"}
    print(
        f"[precision-bf16] loading {model_name} with {max_memory}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    placement = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    return model, tokenizer, device, {
        "placement": "accelerate_auto_cpu_gpu",
        "max_memory": {"cuda:0": "11GiB", "cpu": "24GiB"},
        "device_map": placement,
    }


def selected_value(
    selection: dict[str, Any],
    state_capture: StateCapture,
    head_capture: AllHeadCapture,
) -> torch.Tensor:
    component = selection["component"]
    depth = int(selection["depth"])
    if component == "residual_stream":
        value = state_capture.captured[("residual", depth)][0, 0]
    elif component == "attention_output":
        value = state_capture.captured[("attention", depth)][0, 0]
    elif component == "mlp_output":
        value = state_capture.captured[("mlp", depth)][0, 0]
    elif component == "attention_head_pre_o_proj":
        value = head_capture.values[depth][
            0, int(selection["head"])
        ]
    else:
        raise RuntimeError(f"unknown selected component {component}")
    return value.float().cpu()


def normalized_delta(
    variant: torch.Tensor,
    base: torch.Tensor,
) -> tuple[np.ndarray, float]:
    delta = variant - base
    raw = float(torch.linalg.vector_norm(delta).item())
    scale = 0.5 * (
        float(torch.linalg.vector_norm(variant).item())
        + float(torch.linalg.vector_norm(base).item())
    )
    normalized = raw / max(scale, EPSILON)
    direction = (
        (delta / max(raw, EPSILON)).numpy()
        if raw > EPSILON
        else np.zeros(delta.shape, dtype=np.float32)
    )
    return direction.astype(np.float32, copy=False), normalized


def eight_bit_panel_mean(
    *,
    model: str,
    family: str,
    output_mode: str,
    selection: dict[str, Any],
) -> np.ndarray | None:
    panel_root = (
        SOURCE_ROOT / "scan" / model / family / output_mode
    )
    panel_summary = read_json(panel_root / "summary.json")
    bundle = np.load(panel_root / "canonical_direction_sums.npz")
    operation_index = TARGET_INDEX[selection["operation"]]
    event_index = int(selection["event_index"])
    whole_count = int(panel_summary["whole_event_count"])
    if event_index < whole_count:
        value = bundle["whole"][
            operation_index,
            CONFIRMATION_SPLIT_INDEX,
            event_index,
        ]
        count = int(bundle["whole_count"][
            operation_index,
            CONFIRMATION_SPLIT_INDEX,
            event_index,
        ])
    else:
        local_index = event_index - whole_count
        value = bundle["head"][
            operation_index,
            CONFIRMATION_SPLIT_INDEX,
            local_index,
        ]
        count = int(bundle["head_count"][
            operation_index,
            CONFIRMATION_SPLIT_INDEX,
            local_index,
        ])
    result = (
        value.astype(np.float32, copy=False) / count
        if count > 0
        else None
    )
    bundle.close()
    return result


def run_model(model_name: str) -> dict[str, Any]:
    precision_protocol = read_json(PROTOCOL_ROOT / "protocol.json")
    selections = read_jsonl(PROTOCOL_ROOT / model_name / "events.jsonl")
    selected_units = read_jsonl(
        PROTOCOL_ROOT / model_name / "units.jsonl"
    )
    source_cases = read_jsonl(
        SOURCE_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    case_by_id = {row["record_id"]: row for row in source_cases}
    output_root = OUT_ROOT / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    if not selections:
        result = {
            "schema_version": "phase1014_bf16_precision_model.v1",
            "phase": 1014,
            "model": model_name,
            "selected_event_count": 0,
            "skipped": True,
            "reason": "no discovery-frozen control-specific event",
        }
        write_json(output_root / "summary.json", result)
        return result

    selected_by_id = {
        row["event_id"]: row for row in selections
    }
    selected_operations = {
        row["operation"] for row in selections
    }
    required_state_set = {"base", "identity"}
    for operation in selected_operations:
        required_state_set.add(operation)
        required_state_set.update(MATCHED_CONTROLS[operation])
    required_states = tuple(
        state for state in STATE_ORDER if state in required_state_set
    )
    accumulators: dict[tuple[str, str, str], dict[str, Any]] = {}
    for selection in selections:
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                accumulators[
                    (selection["event_id"], family, output_mode)
                ] = {
                    "raw_sum": None,
                    "canonical_sum": None,
                    "count": 0,
                    "targets": [],
                    "matched_controls": [],
                    "bf16_directions": [],
                }

    model = tokenizer = device = None
    state_capture = head_capture = None
    unit_rows = []
    identity_maximum = 0.0
    started = time.time()
    placement = {}
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        state_capture = StateCapture(model, layers)
        head_capture = AllHeadCapture(layers, head_count)
        state_capture.register()
        head_capture.register()
        for unit_index, unit in enumerate(selected_units):
            state_cases = {
                state: (
                    case_by_id[unit["case_ids"][state]]
                    if state != "identity"
                    else case_by_id[unit["case_ids"]["base"]]
                )
                for state in required_states
            }
            captured: dict[str, dict[str, torch.Tensor]] = {
                event_id: {} for event_id in selected_by_id
            }
            for state in required_states:
                case = state_cases[state]
                input_ids = torch.tensor(
                    [case["input_ids"]],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                position = int(
                    case["role_positions"]["answer_boundary"]
                )
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
                    state_capture.validate()
                    head_capture.validate()
                    for event_id, selection in selected_by_id.items():
                        captured[event_id][state] = selected_value(
                            selection,
                            state_capture,
                            head_capture,
                        )
                    del output
                finally:
                    state_capture.captured = {}
                    head_capture.values = {}
                    del input_ids, attention_mask, positions

            for event_id, selection in selected_by_id.items():
                values = captured[event_id]
                base = values["base"]
                identity_raw = float(torch.linalg.vector_norm(
                    values["identity"] - base
                ).item())
                identity_maximum = max(identity_maximum, identity_raw)
                operation = selection["operation"]
                direction, target = normalized_delta(
                    values[operation], base
                )
                control_values = []
                for control in MATCHED_CONTROLS[operation]:
                    _, normalized = normalized_delta(
                        values[control], base
                    )
                    control_values.append(normalized)
                matched = max(control_values)
                sign = int(
                    unit["canonical_factor_signs"][operation]
                )
                key = (
                    event_id,
                    unit["family"],
                    unit["output_mode"],
                )
                accumulator = accumulators[key]
                if accumulator["raw_sum"] is None:
                    accumulator["raw_sum"] = np.zeros_like(
                        direction, dtype=np.float64
                    )
                    accumulator["canonical_sum"] = np.zeros_like(
                        direction, dtype=np.float64
                    )
                if float(np.linalg.norm(direction)) > EPSILON:
                    accumulator["raw_sum"] += direction
                    accumulator["canonical_sum"] += sign * direction
                    accumulator["count"] += 1
                    accumulator["bf16_directions"].append(
                        sign * direction
                    )
                accumulator["targets"].append(target)
                accumulator["matched_controls"].append(matched)
                unit_rows.append({
                    "schema_version": (
                        "phase1014_bf16_precision_unit.v1"
                    ),
                    "phase": 1014,
                    "model": model_name,
                    "unit_id": unit["unit_id"],
                    "family": unit["family"],
                    "output_mode": unit["output_mode"],
                    "template": int(unit["template"]),
                    "name_pool": int(unit["name_pool"]),
                    "world_index": int(unit["world_index"]),
                    "event_id": event_id,
                    "operation": operation,
                    "target_normalized_magnitude": target,
                    "matched_control_normalized_magnitude": matched,
                    "target_exceeds_matched_control": target > matched,
                    "identity_raw_magnitude": identity_raw,
                })
            if (unit_index + 1) % 8 == 0:
                print(
                    f"[precision-bf16] {model_name} "
                    f"{unit_index + 1}/{len(selected_units)}",
                    flush=True,
                )

        cells = []
        candidate_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        candidate_precision_cosines: dict[str, list[float]] = (
            defaultdict(list)
        )
        for key, accumulator in accumulators.items():
            event_id, family, output_mode = key
            selection = selected_by_id[event_id]
            count = int(accumulator["count"])
            raw_consistency = direction_consistency(
                accumulator["raw_sum"], count
            )
            canonical_consistency = direction_consistency(
                accumulator["canonical_sum"], count
            )
            orientation_gain = (
                canonical_consistency - raw_consistency
                if canonical_consistency is not None
                and raw_consistency is not None
                else None
            )
            target = np.asarray(
                accumulator["targets"], dtype=np.float64
            )
            matched = np.asarray(
                accumulator["matched_controls"], dtype=np.float64
            )
            prevalence = finite(np.mean(target > matched))
            median_delta = finite(np.median(target - matched))
            direction_pass = bool(
                canonical_consistency is not None
                and canonical_consistency >= DIRECTION_THRESHOLD
                and orientation_gain is not None
                and orientation_gain >= ORIENTATION_GAIN
            )
            specificity_pass = bool(
                prevalence is not None
                and prevalence >= PREVALENCE_THRESHOLD
                and median_delta is not None
                and median_delta > 0
            )
            bf16_mean = (
                accumulator["canonical_sum"] / count
                if count > 0 else None
            )
            eight_bit_mean = eight_bit_panel_mean(
                model=model_name,
                family=family,
                output_mode=output_mode,
                selection=selection,
            )
            precision_cosine = (
                cosine(bf16_mean, eight_bit_mean)
                if bf16_mean is not None
                and eight_bit_mean is not None
                else None
            )
            if direction_pass and bf16_mean is not None:
                candidate_vectors[event_id].append(bf16_mean)
            if precision_cosine is not None:
                candidate_precision_cosines[event_id].append(
                    precision_cosine
                )
            cells.append({
                "schema_version": (
                    "phase1014_bf16_precision_cell.v1"
                ),
                "phase": 1014,
                "model": model_name,
                "event_id": event_id,
                "operation": selection["operation"],
                "family": family,
                "output_mode": output_mode,
                "n": len(target),
                "direction_count": count,
                "raw_direction_consistency": raw_consistency,
                "canonical_direction_consistency": (
                    canonical_consistency
                ),
                "orientation_gain": orientation_gain,
                "matched_control_prevalence": prevalence,
                "matched_control_median_delta": median_delta,
                "direction_pass": direction_pass,
                "specificity_pass": specificity_pass,
                "eight_bit_bf16_direction_cosine": precision_cosine,
            })

        candidate_rows = []
        for selection in selections:
            event_id = selection["event_id"]
            event_cells = [
                row for row in cells if row["event_id"] == event_id
            ]
            direction_panel_count = sum(
                row["direction_pass"] for row in event_cells
            )
            specificity_panel_count = sum(
                row["specificity_pass"] for row in event_cells
            )
            both_panel_count = sum(
                row["direction_pass"] and row["specificity_pass"]
                for row in event_cells
            )
            cross_panel = pairwise_consistency(
                candidate_vectors[event_id]
            )
            cosines = candidate_precision_cosines[event_id]
            median_precision_cosine = (
                finite(np.median(cosines)) if cosines else None
            )
            precision_supported = bool(
                both_panel_count >= PANEL_COUNT_THRESHOLD
                and cross_panel is not None
                and cross_panel >= CROSS_PANEL_THRESHOLD
                and median_precision_cosine is not None
                and median_precision_cosine
                >= PRECISION_COSINE_THRESHOLD
            )
            candidate_rows.append({
                **selection,
                "schema_version": (
                    "phase1014_bf16_precision_candidate.v1"
                ),
                "phase": 1014,
                "model": model_name,
                "bf16_direction_panel_count": direction_panel_count,
                "bf16_specificity_panel_count": specificity_panel_count,
                "bf16_both_panel_count": both_panel_count,
                "bf16_cross_panel_direction_consistency": cross_panel,
                "eight_bit_bf16_median_direction_cosine": (
                    median_precision_cosine
                ),
                "precision_supported": precision_supported,
                "claim": (
                    "held-out precision replication only; no causal "
                    "necessity or sufficiency"
                ),
            })

        write_jsonl(output_root / "unit_metrics.jsonl", unit_rows)
        write_jsonl(output_root / "cells.jsonl", cells)
        write_jsonl(
            output_root / "candidate_summary.jsonl",
            candidate_rows,
        )
        result = {
            "schema_version": "phase1014_bf16_precision_model.v1",
            "phase": 1014,
            "model": model_name,
            "source_precision_protocol_digest": precision_protocol[
                "precision_protocol_digest"
            ],
            "precision": "bfloat16",
            "placement": placement,
            "selected_event_count": len(selections),
            "confirmation_unit_count": len(selected_units),
            "singleton_forward_count": (
                len(selected_units) * len(required_states)
            ),
            "state_forward_order": list(required_states),
            "cell_count": len(cells),
            "precision_supported_event_count": sum(
                row["precision_supported"] for row in candidate_rows
            ),
            "identity_maximum": identity_maximum,
            "elapsed_seconds": time.time() - started,
            "operational_thresholds_not_theory": {
                "canonical_direction": DIRECTION_THRESHOLD,
                "orientation_gain": ORIENTATION_GAIN,
                "matched_control_prevalence": PREVALENCE_THRESHOLD,
                "panel_count": PANEL_COUNT_THRESHOLD,
                "cross_panel_direction": CROSS_PANEL_THRESHOLD,
                "eight_bit_bf16_cosine": PRECISION_COSINE_THRESHOLD,
            },
            "claim_limit": (
                "precision and held-out response replication only; "
                "no causal or transport claim"
            ),
        }
        write_json(output_root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if state_capture is not None:
            state_capture.close()
        if head_capture is not None:
            head_capture.close()
        model = tokenizer = device = state_capture = head_capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
