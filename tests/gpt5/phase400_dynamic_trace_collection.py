#!/usr/bin/env python3
"""Collect Phase400 interval, direction, readout, and raw-anchor trajectories."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase399_dynamic_trace_collection as p399  # noqa: E402
from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase267_multifamily_continuation_physical_path_trace import get_final_norm  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks, module_attr  # noqa: E402
from phase400_dynamic_protocol import MODELS, OUT  # noqa: E402


PRIVATE = OUT / "dynamic_trace/protocol/private"
STAGES = ("instrument", "discovery", "calibration", "physical_holdout")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase400 non-finite scalar: {value}")
    return round(value, 9)


def source_path(stage: str) -> Path:
    name = (
        "phase400_instrument_dynamic_trace_cases.jsonl"
        if stage == "instrument"
        else f"phase400_{stage}_dynamic_trace_cases.jsonl"
    )
    return PRIVATE / name


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def effect_vectors(
    collected: list[dict[str, Any]], event_id: str, layer_count: int
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    fields = p399.EFFECTS["ROQ"]
    for axis in ("X", "Y"):
        items = sorted(
            [item for item in collected if item["case"]["axis_private"] == axis],
            key=lambda item: item["case"]["anonymous_condition_slot"],
        )
        if len(items) != 8:
            raise RuntimeError("Phase400 raw effect axis is not an eight-cell factorial")
        layers = []
        for layer_index in range(layer_count):
            layers.append(
                torch.stack(
                    [
                        item["events"][event_id][layer_index]
                        * p399.sign(item["case"], fields)
                        for item in items
                    ]
                ).mean(dim=0)
            )
        result[axis] = torch.stack(layers)
    return result


def transition_cosines(vectors: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for axis, tensor in vectors.items():
        values = F.cosine_similarity(tensor[:-1], tensor[1:], dim=-1, eps=1e-8)
        output[axis] = [clean(float(value.item())) for value in values]
    output["min_axis"] = [
        min(x, y) for x, y in zip(output["X"], output["Y"], strict=True)
    ]
    return output


def fill_single_gaps(values: list[bool]) -> list[bool]:
    result = list(values)
    for index in range(1, len(values) - 1):
        if not values[index] and values[index - 1] and values[index + 1]:
            result[index] = True
    return result


def intervals(values: list[bool], minimum: int = 2) -> list[list[int]]:
    found: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate([*values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                found.append([start, index - 1])
            start = None
    return found


def interval_descriptor(row: dict[str, Any], transitions: dict[str, list[float]]) -> dict[str, Any]:
    protocol = read_json(OUT / "phase400_partial_order_protocol.json")
    gate = protocol["per_group_layer_gate"]
    contract = protocol["interval_contract"]
    norms = row["interaction_trajectories"]["ROQ"]["min_axis_normalized_norm"]
    cosines = row["interaction_trajectories"]["ROQ"]["cross_axis_cosine"]
    ratios = row["roq_to_strongest_competing_interaction"]
    qualified_raw = [
        norm >= gate["roq_min_axis_normalized_norm_min"]
        and cosine >= gate["roq_cross_axis_cosine_min"]
        and ratio >= gate["roq_to_competing_interaction_min"]
        for norm, cosine, ratio in zip(norms, cosines, ratios, strict=True)
    ]
    qualified = fill_single_gaps(qualified_raw)
    runs = intervals(qualified, contract["minimum_consecutive_qualified_layers"])
    active = [index for start, end in runs for index in range(start, end + 1)]
    amplification = []
    for index in range(1, len(norms)):
        delta = norms[index] - norms[index - 1]
        relative = delta / max(abs(norms[index - 1]), 1e-8)
        if (
            delta >= contract["amplification_absolute_delta_min"]
            and relative >= contract["amplification_relative_delta_min"]
        ):
            amplification.append(index)
    flips = [
        index + 1
        for index, value in enumerate(transitions["min_axis"])
        if value <= contract["flip_transition_cosine_max"]
        and qualified[index]
        and qualified[index + 1]
    ]
    return {
        "qualified_layers_before_gap_fill": [
            index for index, value in enumerate(qualified_raw) if value
        ],
        "qualified_layers": [index for index, value in enumerate(qualified) if value],
        "qualified_intervals": runs,
        "interval_present": bool(runs),
        "onset_layer": runs[0][0] if runs else None,
        "offset_layer": runs[-1][1] if runs else None,
        "duration_layers": len(active),
        "amplification_layers": amplification,
        "peak_layer": row["peak_roq_layer"],
        "flip_layers": flips,
        "active_median_roq_norm": (
            clean(float(torch.tensor([norms[index] for index in active]).median().item()))
            if active
            else 0.0
        ),
        "active_median_cross_axis_cosine": (
            clean(float(torch.tensor([cosines[index] for index in active]).median().item()))
            if active
            else 0.0
        ),
        "active_median_specificity_ratio": (
            clean(float(torch.tensor([ratios[index] for index in active]).median().item()))
            if active
            else 0.0
        ),
    }


@torch.inference_mode()
def readout_margins(
    loaded: Any,
    vectors: list[torch.Tensor],
    target_id: int,
    distractor_ids: list[int],
) -> list[float]:
    stack = torch.stack(vectors).to(
        device=loaded.input_device,
        dtype=next(loaded.model.parameters()).dtype,
    )
    final_norm = get_final_norm(loaded.model)
    if final_norm is not None:
        stack = final_norm(stack)
    output = loaded.model.get_output_embeddings()
    ids = torch.tensor(
        [target_id, *distractor_ids], dtype=torch.long, device=output.weight.device
    )
    weights = output.weight.index_select(0, ids).to(dtype=stack.dtype)
    scores = stack.to(weights.device) @ weights.transpose(0, 1)
    bias = getattr(output, "bias", None)
    if bias is not None:
        scores = scores + bias.index_select(0, ids)
    margins = scores[:, 0] - scores[:, 1:].max(dim=-1).values
    return [clean(float(value.item())) for value in margins.detach().float().cpu()]


def add_prediction_readouts(loaded: Any, collected: dict[str, Any]) -> None:
    case = collected["case"]
    target = int(case["target_first_token_id_private"])
    distractors = [int(value) for value in case["distractor_first_token_ids_private"]]
    coordinates = ("query_end", "first_answer", "target_completion", "post_target")
    collected["prediction_readouts"] = {
        coordinate: readout_margins(
            loaded,
            collected["events"][f"state:{coordinate}:layer_output"],
            target,
            distractors,
        )
        for coordinate in coordinates
    }


def summarize_group(
    model: str,
    stage: str,
    collected: list[dict[str, Any]],
    layer_count: int,
    save_anchor: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    rows, audit = p399.summarize_group(model, stage, collected, layer_count)
    transformed: list[dict[str, Any]] = []
    anchor_events: dict[str, Any] = {}
    for row in rows:
        vectors = effect_vectors(collected, row["event_id"], layer_count)
        transitions = transition_cosines(vectors)
        row.update(
            {
                "schema_version": "74.5.0",
                "phase_id": "Phase400-DynamicTraceCollection",
                "phase400_public_parallel_group_id": row.pop("public_parallel_group_id"),
                "roq_layer_transition_cosine": transitions,
                "partial_order_descriptor": interval_descriptor(row, transitions),
                "raw_vectors_persisted": save_anchor,
            }
        )
        transformed.append(row)
        if save_anchor:
            anchor_events[row["event_id"]] = {
                axis: tensor.to(dtype=torch.float16).contiguous()
                for axis, tensor in vectors.items()
            }
    audit.update(
        {
            "schema_version": "74.5.0",
            "phase_id": "Phase400-DynamicTraceGroupAudit",
            "phase400_public_parallel_group_id": audit.pop("public_parallel_group_id"),
        }
    )
    anchor = None
    if save_anchor:
        base = collected[0]["case"]
        anchor = {
            "schema_version": "phase400.raw_roq_anchor.v1",
            "phase_id": "Phase400-RawROQAnchor",
            "model": model,
            "surface": base["task_surface_private"],
            "public_parallel_group_id": base["phase400_public_parallel_group_id"],
            "layer_count": layer_count,
            "stored_dtype": "float16",
            "events": anchor_events,
            "private_only": True,
        }
    return transformed, audit, anchor


def prediction_rows(collected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in collected:
        case = item["case"]
        rows.append(
            {
                "schema_version": "74.5.0",
                "phase_id": "Phase400-CasePredictionTrajectory",
                "created_at": now(),
                "model": case["private_execution_model"],
                "stage": case["phase400_split"],
                "public_parallel_group_id": case["phase400_public_parallel_group_id"],
                "surface_private": case["task_surface_private"],
                "blind_case_id_private": case["blind_case_id"],
                "condition_private": case["anonymous_condition_slot"],
                "target_token_id_private": case["target_first_token_id_private"],
                "distractor_token_ids_private": case["distractor_first_token_ids_private"],
                "target_minus_distractor_margin_by_coordinate": item["prediction_readouts"],
                "first_answer_replay_match": item["first_answer_replay_match"],
                "causal_intervention": False,
            }
        )
    return rows


def run(
    model: str,
    stage: str,
    shard_index: int | None = None,
    shard_size: int = 2,
) -> dict[str, Any]:
    cases = [
        row for row in read_jsonl(source_path(stage)) if row["private_execution_model"] == model
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["anonymous_parallel_group_id"]].append(case)
    all_group_ids = sorted(grouped)
    if shard_index is not None:
        if shard_index < 0 or shard_size <= 0:
            raise ValueError("Phase400 shard index and size must be positive")
        start = shard_index * shard_size
        chosen_group_ids = all_group_ids[start : start + shard_size]
        if not chosen_group_ids:
            raise RuntimeError(f"Phase400 empty shard {shard_index} for {model}/{stage}")
    else:
        chosen_group_ids = all_group_ids

    trace_protocol = read_json(OUT / "phase400_dynamic_trace_protocol.json")
    anchor_groups = set(trace_protocol["raw_anchor_group_ids_private"].values())
    loaded = None
    handles: list[Any] = []
    value_handles: list[Any] = []
    event_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    anchor_records: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], Any] = {}
        handles = install_hooks(layers, captures)
        for layer_index, layer in enumerate(layers):
            value_proj = module_attr(layer.self_attn, ("v_proj", "value"))

            def value_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                idx: int = layer_index,
            ) -> None:
                captures[("value_projection", idx)] = output.detach()

            value_handles.append(value_proj.register_forward_hook(value_post))

        for group_index, group_id in enumerate(chosen_group_ids, 1):
            group_cases = sorted(
                grouped[group_id], key=lambda row: row["anonymous_condition_slot"]
            )
            collected = []
            for case in group_cases:
                compatible_case = {
                    **case,
                    "phase399_public_parallel_group_id": case[
                        "phase400_public_parallel_group_id"
                    ],
                }
                item = p399.collect_case(loaded, layers, captures, compatible_case)
                item["case"] = compatible_case
                add_prediction_readouts(loaded, item)
                collected.append(item)
            save_anchor = stage == "discovery" and group_id in anchor_groups
            rows, audit, anchor = summarize_group(
                model, stage, collected, len(layers), save_anchor
            )
            event_rows.extend(rows)
            group_rows.append(audit)
            case_rows.extend(prediction_rows(collected))
            if anchor is not None:
                anchor_path = (
                    OUT
                    / "dynamic_trace/discovery/private/raw_anchors"
                    / model
                    / f"{anchor['surface']}.pt"
                )
                anchor_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(anchor, anchor_path)
                anchor_records.append(
                    {
                        "surface": anchor["surface"],
                        "path": str(anchor_path.relative_to(OUT)),
                        "byte_count": anchor_path.stat().st_size,
                        "sha256": sha256(anchor_path),
                    }
                )
            del collected
            gc.collect()
            print(
                f"[{model}/{stage}] Phase400 group {group_index}/{len(chosen_group_ids)} "
                f"gate={audit['quality_gate_pass']} anchor={save_anchor}",
                flush=True,
            )

        model_root = OUT / "dynamic_trace" / stage / "private/models" / model
        if shard_index is not None:
            model_root = model_root / "shards" / f"shard_{shard_index:03d}"
        write_jsonl(model_root / "event_trajectory_rows.jsonl", event_rows)
        write_jsonl(model_root / "group_audit_rows.jsonl", group_rows)
        write_jsonl(model_root / "case_prediction_rows.jsonl", case_rows)
        payload = {
            "schema_version": "74.5.0",
            "phase_id": "Phase400-DynamicTraceCollection",
            "created_at": now(),
            "model": model,
            "stage": stage,
            "shard_index": shard_index,
            "shard_size": shard_size if shard_index is not None else None,
            "stage_total_group_count": len(all_group_ids),
            "selected_group_ids": chosen_group_ids,
            "case_count": len(chosen_group_ids) * 16,
            "group_count": len(chosen_group_ids),
            "layer_count": len(layers),
            "event_trajectory_row_count": len(event_rows),
            "case_prediction_row_count": len(case_rows),
            "quality_group_count": sum(row["quality_gate_pass"] for row in group_rows),
            "max_block_relative_error": max(row["max_block_relative_error"] for row in group_rows),
            "max_attention_replay_relative_error": max(
                row["max_attention_replay_relative_error"] for row in group_rows
            ),
            "max_probability_sum_error": max(
                row["max_probability_sum_error"] for row in group_rows
            ),
            "raw_anchor_records": anchor_records,
            "valid": bool(chosen_group_ids)
            and all(len(grouped[group]) == 16 for group in chosen_group_ids)
            and all(row["quality_gate_pass"] for row in group_rows),
        }
        write_json(model_root / "complete.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload
    finally:
        for handle in value_handles + handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-size", type=int, default=2)
    args = parser.parse_args()
    run(args.model, args.stage, args.shard_index, args.shard_size)
