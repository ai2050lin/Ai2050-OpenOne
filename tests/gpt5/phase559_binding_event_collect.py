#!/usr/bin/env python3
"""Trace fixed-identity binding-swap events without persisting hidden vectors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402


PHASE = "Phase559"
MODEL = "qwen3"
SPLITS = ("path_discovery", "path_confirmation")
POSITIONS = (
    "source_object_end",
    "source_color_end",
    "source_fact_end",
    "nontarget_fact_end",
    "query_relation_end",
    "query_object_end",
    "answer_boundary",
)
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PATH_ROWS = OUT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHOR_REGISTRY = OUT_DIR / "phase559_path_anchor_registry.json"
EVENT_ROWS = OUT_DIR / "phase559_binding_event_rows.jsonl"
EVENT_SUMMARY = OUT_DIR / "phase559_binding_event_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm().item() * right.norm().item())
    if denominator < 1e-12:
        return 0.0
    return finite(float(torch.dot(left, right).item()) / denominator)


def char_end_to_token(offsets: list[tuple[int, int]], char_end: int) -> int:
    for index, (start, end) in enumerate(offsets):
        if start < char_end <= end:
            return index
    candidates = [index for index, (_start, end) in enumerate(offsets) if 0 < end <= char_end]
    if not candidates:
        raise ValueError(f"No token covers character boundary {char_end}")
    return max(candidates)


def child_span(prompt: str, parent: str, child: str, *, last_child: bool = False) -> tuple[int, int]:
    parent_start = prompt.index(parent)
    child_offset = parent.rfind(child) if last_child else parent.find(child)
    if child_offset < 0:
        raise ValueError(f"{child!r} is absent from {parent!r}")
    start = parent_start + child_offset
    return start, start + len(child)


def semantic_positions(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], dict[str, int]]:
    prompt = row["prompt"]
    encoded = tokenizer(
        prompt, add_special_tokens=True, return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    source_object = child_span(prompt, row["source_fact"], row["query_object"])
    source_color = child_span(prompt, row["source_fact"], row["target"])
    source_fact_start = prompt.index(row["source_fact"])
    nontarget_fact_start = prompt.index(row["nontarget_fact"])
    query_object = child_span(prompt, row["question"], row["query_object"], last_child=True)
    query_relation = child_span(prompt, row["question"], "color")
    positions = {
        "source_object_end": char_end_to_token(offsets, source_object[1]),
        "source_color_end": char_end_to_token(offsets, source_color[1]),
        "source_fact_end": char_end_to_token(
            offsets, source_fact_start + len(row["source_fact"])
        ),
        "nontarget_fact_end": char_end_to_token(
            offsets, nontarget_fact_start + len(row["nontarget_fact"])
        ),
        "query_relation_end": char_end_to_token(offsets, query_relation[1]),
        "query_object_end": char_end_to_token(offsets, query_object[1]),
        "answer_boundary": len(ids) - 1,
    }
    return ids, positions


def canonical_pair_delta(
    vectors: torch.Tensor,
    pair_rows: list[tuple[int, dict[str, Any]]],
) -> tuple[torch.Tensor, float]:
    by_binding = {int(row["binding"]): (index, row) for index, row in pair_rows}
    if set(by_binding) != {0, 1}:
        raise RuntimeError("Phase559 event batch lost a binding pair")
    index0, row0 = by_binding[0]
    index1, _row1 = by_binding[1]
    delta = vectors[index1] - vectors[index0]
    if int(row0["query_object_index"]) == 1:
        delta = -delta
    state_scale = float((vectors[index0].norm() + vectors[index1].norm()).item() / 2.0)
    return delta, state_scale


def run(batch_size: int, restart: bool) -> Path:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Phase559 event batch size must be a positive even number")
    registry = read_json(ANCHOR_REGISTRY)
    if registry["authorized_models"] != [MODEL] or registry["sealed_split_read"]:
        raise RuntimeError("Phase559 internal event collection is not authorized")
    authorized_anchors = {
        row["anchor_id"] for row in registry["anchors"]
        if row["authorized_for_internal_collection"]
    }
    cases = [
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] in SPLITS
        and row["anchor_id"] in authorized_anchors
        and row["semantic_correct"]
    ]
    if len(cases) != 4096 or len(authorized_anchors) != 128:
        raise RuntimeError("Phase559 event denominator drift")
    if any(row["sealed"] for row in cases):
        raise RuntimeError("Phase559 event denominator contains sealed rows")
    if restart:
        EVENT_ROWS.unlink(missing_ok=True)
        EVENT_SUMMARY.unlink(missing_ok=True)

    worlds: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        worlds[(row["split"], row["anchor_id"])].append(row)
    if any(len(rows) != 32 for rows in worlds.values()):
        raise RuntimeError("Phase559 event worlds are incomplete")

    loaded = None
    handles: list[Any] = []
    captures: dict[str, list[dict[str, torch.Tensor]]] = {component: [] for component in COMPONENTS}
    current_indices: dict[str, torch.Tensor] = {}
    metric_sums: defaultdict[tuple[str, tuple[str, int, str], str], float] = defaultdict(float)
    metric_counts: defaultdict[tuple[str, tuple[str, int, str]], int] = defaultdict(int)
    color_group_sums: dict[tuple[str, str, tuple[str, int, str]], torch.Tensor] = {}
    color_group_counts: defaultdict[tuple[str, str, tuple[str, int, str]], int] = defaultdict(int)
    max_ledger_error = 0.0
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        if not getattr(loaded.tokenizer, "is_fast", False):
            raise RuntimeError("Phase559 exact semantic positions require a fast tokenizer")
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase559 event collection requires BF16, got {run_dtype}")

        def select_positions(value: torch.Tensor) -> dict[str, torch.Tensor]:
            batch_index = torch.arange(value.shape[0], device=value.device)
            return {
                position: value[batch_index, indices.to(value.device), :].detach().float().cpu()
                for position, indices in current_indices.items()
            }

        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            captures["layer_input"].append(select_positions(inputs[0]))

        def make_hook(component: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                captures[component].append(select_positions(tensor_from_output(output)))
            return hook

        for layer in layers:
            handles.append(layer.register_forward_pre_hook(pre_hook))
            handles.append(layer.self_attn.register_forward_hook(make_hook("attention_output")))
            handles.append(layer.mlp.register_forward_hook(make_hook("mlp_output")))
            handles.append(layer.register_forward_hook(make_hook("layer_output")))

        for world_number, ((split, anchor_id), world_rows) in enumerate(sorted(worlds.items()), start=1):
            query_means: dict[int, dict[tuple[str, int, str], torch.Tensor]] = {}
            query_relative: dict[int, dict[tuple[str, int, str], float]] = {}
            query_stability: dict[int, dict[tuple[str, int, str], float]] = {}
            for query_index in (0, 1):
                query_rows = [
                    row for row in world_rows if int(row["query_object_index"]) == query_index
                ]
                pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in query_rows:
                    pairs[row["pair_id"]].append(row)
                ordered_pairs = [sorted(rows, key=lambda row: int(row["binding"])) for _, rows in sorted(pairs.items())]
                coord_delta_sums: dict[tuple[str, int, str], torch.Tensor] = {}
                coord_unit_sums: dict[tuple[str, int, str], torch.Tensor] = {}
                coord_relative_sums: defaultdict[tuple[str, int, str], float] = defaultdict(float)
                coord_counts: defaultdict[tuple[str, int, str], int] = defaultdict(int)

                pair_batch_width = batch_size // 2
                for pair_start in range(0, len(ordered_pairs), pair_batch_width):
                    pair_batch = ordered_pairs[pair_start:pair_start + pair_batch_width]
                    batch_rows = [row for pair in pair_batch for row in pair]
                    for values in captures.values():
                        values.clear()
                    individual = [semantic_positions(loaded.tokenizer, row) for row in batch_rows]
                    encoded = loaded.tokenizer(
                        [row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                        truncation=True, max_length=256,
                    )
                    sequence_length = int(encoded["input_ids"].shape[1])
                    positions: dict[str, list[int]] = {position: [] for position in POSITIONS}
                    for row_index, (ids, semantic) in enumerate(individual):
                        mask_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                        if [int(value) for value in mask_ids] != ids:
                            raise RuntimeError("Phase559 individual/batch tokenization drift")
                        offset = sequence_length - len(ids)
                        for position in POSITIONS:
                            positions[position].append(offset + semantic[position])
                    current_indices.clear()
                    current_indices.update({
                        position: torch.tensor(indices, dtype=torch.long)
                        for position, indices in positions.items()
                    })
                    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        result = loaded.model(**encoded, use_cache=False)
                    if any(len(captures[component]) != len(layers) for component in COMPONENTS):
                        raise RuntimeError("Phase559 event hook count mismatch")

                    batch_pair_indices = [
                        [(2 * local, pair[0]), (2 * local + 1, pair[1])]
                        for local, pair in enumerate(pair_batch)
                    ]
                    for layer_index in range(len(layers)):
                        for position in POSITIONS:
                            residual = (
                                captures["layer_output"][layer_index][position]
                                - captures["layer_input"][layer_index][position]
                                - captures["attention_output"][layer_index][position]
                                - captures["mlp_output"][layer_index][position]
                            )
                            denominator = captures["layer_output"][layer_index][position].norm(dim=1).clamp_min(1e-12)
                            max_ledger_error = max(
                                max_ledger_error,
                                float((residual.norm(dim=1) / denominator).max().item()),
                            )
                            for component in COMPONENTS:
                                coord = (position, layer_index, component)
                                vectors = captures[component][layer_index][position]
                                for pair_rows in batch_pair_indices:
                                    delta, scale = canonical_pair_delta(vectors, pair_rows)
                                    norm = float(delta.norm().item())
                                    if coord not in coord_delta_sums:
                                        coord_delta_sums[coord] = torch.zeros_like(delta)
                                        coord_unit_sums[coord] = torch.zeros_like(delta)
                                    coord_delta_sums[coord] += delta
                                    coord_unit_sums[coord] += delta / max(norm, 1e-12)
                                    coord_relative_sums[coord] += norm / max(scale, 1e-12)
                                    coord_counts[coord] += 1
                    del result, encoded

                query_means[query_index] = {
                    coord: value / coord_counts[coord] for coord, value in coord_delta_sums.items()
                }
                query_relative[query_index] = {
                    coord: coord_relative_sums[coord] / coord_counts[coord]
                    for coord in coord_delta_sums
                }
                query_stability[query_index] = {
                    coord: finite(float((coord_unit_sums[coord] / coord_counts[coord]).norm().item()))
                    for coord in coord_delta_sums
                }

            color_key = f"{world_rows[0]['color_a']}|{world_rows[0]['color_b']}"
            for coord in query_means[0]:
                role_cosine = cosine(query_means[0][coord], query_means[1][coord])
                world_mean = (query_means[0][coord] + query_means[1][coord]) / 2.0
                world_norm = float(world_mean.norm().item())
                relative = (query_relative[0][coord] + query_relative[1][coord]) / 2.0
                stability = (query_stability[0][coord] + query_stability[1][coord]) / 2.0
                metric_sums[(split, coord, "mean_relative_binding_delta_norm")] += relative
                metric_sums[(split, coord, "mean_surface_order_direction_stability")] += stability
                metric_sums[(split, coord, "mean_query_role_direction_cosine")] += role_cosine
                metric_sums[(split, coord, "mean_world_binding_delta_norm")] += world_norm
                metric_counts[(split, coord)] += 1
                group_key = (split, color_key, coord)
                normalized = world_mean / max(world_norm, 1e-12)
                if group_key not in color_group_sums:
                    color_group_sums[group_key] = torch.zeros_like(normalized)
                color_group_sums[group_key] += normalized
                color_group_counts[group_key] += 1
            if world_number == 1 or world_number == len(worlds) or world_number % 8 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] qwen3 Phase559 binding events "
                    f"{world_number}/{len(worlds)} worlds",
                    flush=True,
                )

        output_rows: list[dict[str, Any]] = []
        for layer_index in range(len(layers)):
            for component in COMPONENTS:
                for position in POSITIONS:
                    coord = (position, layer_index, component)
                    split_metrics: dict[str, dict[str, Any]] = {}
                    for split in SPLITS:
                        count = metric_counts[(split, coord)]
                        color_stabilities = []
                        for key, value in color_group_sums.items():
                            key_split, _color, key_coord = key
                            if key_split == split and key_coord == coord:
                                color_stabilities.append(
                                    finite(float((value / color_group_counts[key]).norm().item()))
                                )
                        split_metrics[split] = {
                            "world_count": count,
                            "mean_relative_binding_delta_norm": finite(
                                metric_sums[(split, coord, "mean_relative_binding_delta_norm")] / count
                            ),
                            "mean_surface_order_direction_stability": finite(
                                metric_sums[(split, coord, "mean_surface_order_direction_stability")] / count
                            ),
                            "mean_query_role_direction_cosine": finite(
                                metric_sums[(split, coord, "mean_query_role_direction_cosine")] / count
                            ),
                            "mean_world_binding_delta_norm": finite(
                                metric_sums[(split, coord, "mean_world_binding_delta_norm")] / count
                            ),
                            "repeated_color_pair_count": len(color_stabilities),
                            "mean_repeated_color_pair_direction_stability": finite(
                                sum(color_stabilities) / len(color_stabilities)
                            ) if color_stabilities else 0.0,
                        }
                    shared_colors = sorted({
                        color for split, color, key_coord in color_group_sums
                        if split == SPLITS[0] and key_coord == coord
                    } & {
                        color for split, color, key_coord in color_group_sums
                        if split == SPLITS[1] and key_coord == coord
                    })
                    cross_split = [
                        cosine(
                            color_group_sums[(SPLITS[0], color, coord)],
                            color_group_sums[(SPLITS[1], color, coord)],
                        )
                        for color in shared_colors
                    ]
                    output_rows.append({
                        "schema_version": "phase559_binding_event.v1",
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": MODEL,
                        "torch_dtype": run_dtype,
                        "layer": layer_index,
                        "layer_count": len(layers),
                        "relative_depth": layer_index / max(1, len(layers) - 1),
                        "component": component,
                        "semantic_position": position,
                        "split_metrics": split_metrics,
                        "shared_color_pair_count": len(shared_colors),
                        "mean_cross_split_matched_color_pair_direction_cosine": finite(
                            sum(cross_split) / len(cross_split)
                        ) if cross_split else 0.0,
                        "minimum_cross_split_matched_color_pair_direction_cosine": finite(
                            min(cross_split)
                        ) if cross_split else 0.0,
                        "counterfactual_unit": "same_objects_same_colors_same_query_binding_swap",
                        "full_vector_persisted": False,
                        "observer_only": True,
                        "causal": False,
                        "compute_edge": False,
                        "sealed": False,
                    })
        write_jsonl(EVENT_ROWS, output_rows)
        summary = {
            "schema_version": "phase559_binding_event_summary.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "world_count": len(worlds),
            "split_world_counts": {
                split: sum(key[0] == split for key in worlds) for split in SPLITS
            },
            "counterfactual_pair_count": len(cases) // 2,
            "layer_count": len(layers),
            "components": list(COMPONENTS),
            "semantic_positions": list(POSITIONS),
            "event_row_count": len(output_rows),
            "max_component_ledger_relative_error": finite(max_ledger_error),
            "runtime_seconds": time.monotonic() - started,
            "rows_path": str(EVENT_ROWS.relative_to(ROOT)),
            "rows_sha256": sha256_file(EVENT_ROWS),
            "full_vectors_persisted": False,
            "causal_intervention_executed": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(EVENT_SUMMARY, summary)
        print(EVENT_SUMMARY)
        return EVENT_SUMMARY
    finally:
        for handle in handles:
            handle.remove()
        for values in captures.values():
            values.clear()
        current_indices.clear()
        color_group_sums.clear()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.restart)
