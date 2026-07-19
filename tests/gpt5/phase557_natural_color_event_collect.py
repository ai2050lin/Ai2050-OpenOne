#!/usr/bin/env python3
"""Collect multi-position Phase557 natural-color event ledgers.

The qualified natural-color branch is observational at this stage. Full hidden
vectors remain in memory; persisted rows contain object-level and group-level
geometry only. Discovery and confirmation are collected in one model load so
their direction agreement can be measured without serializing hidden states.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
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


PHASE = "Phase557"
MODELS = ("qwen3", "glm4")
SPLITS = ("behavior_discovery", "behavior_confirmation")
POSITIONS = ("object_source_end", "relation_request_end", "answer_boundary")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
CASES_PATH = OUT_DIR / "phase557_open_cases.jsonl"
BEHAVIOR_SUMMARY_PATH = OUT_DIR / "phase557_behavior_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rows_path(model: str) -> Path:
    return OUT_DIR / "natural_color_events" / model / "phase557_natural_color_event_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / "natural_color_events" / model / "phase557_natural_color_event_summary.json"


def observer_prompt(model: str, prompt: str) -> str:
    return prompt + "\n" if model == "glm4" else prompt


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


def pair_mean(values: list[float]) -> float | None:
    return finite(sum(values) / len(values)) if values else None


def find_last_subsequence(sequence: list[int], candidates: list[list[int]]) -> int:
    matches: list[int] = []
    for candidate in candidates:
        if not candidate:
            continue
        width = len(candidate)
        matches.extend(
            index + width - 1
            for index in range(len(sequence) - width + 1)
            if sequence[index:index + width] == candidate
        )
    if not matches:
        raise ValueError("Semantic token span not found in rendered prompt")
    return max(matches)


def token_candidates(tokenizer: Any, text: str) -> list[list[int]]:
    variants = (text, " " + text, "\n" + text, ": " + text)
    unique: list[list[int]] = []
    for variant in variants:
        ids = [int(value) for value in tokenizer(variant, add_special_tokens=False)["input_ids"]]
        if ids and ids not in unique:
            unique.append(ids)
        # A punctuation prefix can merge separately; its suffix is still useful.
        for start in range(1, len(ids)):
            suffix = ids[start:]
            if suffix and suffix not in unique:
                unique.append(suffix)
    return unique


def semantic_positions(tokenizer: Any, model: str, row: dict[str, Any]) -> tuple[list[int], dict[str, int]]:
    text = observer_prompt(model, row["prompt"])
    ids = [int(value) for value in tokenizer(text, add_special_tokens=True)["input_ids"]]
    positions = {
        "object_source_end": find_last_subsequence(ids, token_candidates(tokenizer, row["object_label"])),
        "relation_request_end": find_last_subsequence(ids, token_candidates(tokenizer, "color")),
        "answer_boundary": len(ids) - 1,
    }
    return ids, positions


def authorized(model: str) -> bool:
    summary = read_json(BEHAVIOR_SUMMARY_PATH)
    reports = {row["model"]: row for row in summary["model_reports"]}
    return "color" in reports.get(model, {}).get("authorized_natural_relations", [])


def aggregate_geometry(
    object_vectors: dict[str, torch.Tensor],
    object_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
    labels = sorted(object_vectors)
    fruits = [label for label in labels if object_meta[label]["is_fruit"]]
    controls = [label for label in labels if not object_meta[label]["is_fruit"]]
    fruit_centroid = torch.stack([object_vectors[label] for label in fruits]).mean(dim=0)
    control_centroid = torch.stack([object_vectors[label] for label in controls]).mean(dim=0)
    category_direction = fruit_centroid - control_centroid
    grand_norm = float(torch.stack([object_vectors[label].norm() for label in labels]).mean().item())

    fruit_pair_cosines = [
        cosine(object_vectors[left], object_vectors[right])
        for left, right in itertools.combinations(fruits, 2)
    ]
    control_pair_cosines = [
        cosine(object_vectors[left], object_vectors[right])
        for left, right in itertools.combinations(controls, 2)
    ]
    same_color_pairs: list[float] = []
    different_color_pairs: list[float] = []
    cross_category_same_color_pairs: list[float] = []
    for left, right in itertools.combinations(labels, 2):
        value = cosine(object_vectors[left], object_vectors[right])
        same_color = object_meta[left]["color"] == object_meta[right]["color"]
        if same_color:
            same_color_pairs.append(value)
            if object_meta[left]["is_fruit"] != object_meta[right]["is_fruit"]:
                cross_category_same_color_pairs.append(value)
        elif object_meta[left]["is_fruit"] and object_meta[right]["is_fruit"]:
            different_color_pairs.append(value)

    color_centroids: dict[str, torch.Tensor] = {}
    for color in sorted({object_meta[label]["color"] for label in labels}):
        members = [object_vectors[label] for label in labels if object_meta[label]["color"] == color]
        color_centroids[color] = torch.stack(members).mean(dim=0)
    grand_centroid = torch.stack([object_vectors[label] for label in labels]).mean(dim=0)
    color_directions = {color: value - grand_centroid for color, value in color_centroids.items()}

    same_mean = pair_mean(same_color_pairs)
    different_mean = pair_mean(different_color_pairs)
    return ({
        "object_count": len(labels),
        "fruit_object_count": len(fruits),
        "control_object_count": len(controls),
        "category_contrast_norm": finite(float(category_direction.norm().item())),
        "category_contrast_relative_norm": finite(float(category_direction.norm().item()) / max(grand_norm, 1e-12)),
        "fruit_pair_cosine_mean": pair_mean(fruit_pair_cosines),
        "control_pair_cosine_mean": pair_mean(control_pair_cosines),
        "same_color_pair_count": len(same_color_pairs),
        "same_color_pair_cosine_mean": same_mean,
        "different_color_fruit_pair_count": len(different_color_pairs),
        "different_color_fruit_pair_cosine_mean": different_mean,
        "same_minus_different_color_cosine": (
            finite(same_mean - different_mean)
            if same_mean is not None and different_mean is not None else None
        ),
        "cross_category_same_color_pair_count": len(cross_category_same_color_pairs),
        "cross_category_same_color_cosine_mean": pair_mean(cross_category_same_color_pairs),
    }, category_direction, color_directions)


def run(model_key: str, batch_size: int, restart: bool) -> Path:
    if model_key not in MODELS:
        raise ValueError(f"Unsupported model: {model_key}")
    if not authorized(model_key):
        raise RuntimeError(f"Phase557 natural color branch is not behavior-authorized for {model_key}")
    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_key
        and row["split"] in SPLITS
        and row["case_type"] == "natural_parametric"
        and row["natural_relation"] == "color"
    ]
    if len(cases) != 128:
        raise RuntimeError(f"Expected 128 open natural-color rows for {model_key}, got {len(cases)}")
    output = rows_path(model_key)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_key).unlink(missing_ok=True)

    loaded = None
    handles: list[Any] = []
    captures: dict[str, list[dict[str, torch.Tensor]]] = {component: [] for component in COMPONENTS}
    current_indices: dict[str, torch.Tensor] = {}
    started = time.monotonic()
    max_ledger_error = 0.0
    try:
        loaded = load_probe_model(model_key)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase557 natural-color observer requires BF16, got {run_dtype}")

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

        sums: dict[tuple[str, str, int, str, str], torch.Tensor] = {}
        unit_sums: dict[tuple[str, str, int, str, str], torch.Tensor] = {}
        counts: defaultdict[tuple[str, str, int, str, str], int] = defaultdict(int)
        norm_sums: defaultdict[tuple[str, str, int, str, str], float] = defaultdict(float)
        object_meta: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in SPLITS}

        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            for values in captures.values():
                values.clear()
            individual = [semantic_positions(loaded.tokenizer, model_key, row) for row in batch_rows]
            texts = [observer_prompt(model_key, row["prompt"]) for row in batch_rows]
            encoded = loaded.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            sequence_length = int(encoded["input_ids"].shape[1])
            batch_positions: dict[str, list[int]] = {position: [] for position in POSITIONS}
            for row_index, (ids, positions) in enumerate(individual):
                mask_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                if [int(value) for value in mask_ids] != ids:
                    raise RuntimeError("Phase557 individual/batch tokenization drift")
                offset = sequence_length - len(ids)
                for position in POSITIONS:
                    batch_positions[position].append(offset + positions[position])
            current_indices.clear()
            current_indices.update({
                position: torch.tensor(indices, dtype=torch.long)
                for position, indices in batch_positions.items()
            })
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            if any(len(captures[component]) != len(layers) for component in COMPONENTS):
                raise RuntimeError("Phase557 natural-color hook count mismatch")

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
                        vectors = captures[component][layer_index][position]
                        for row_index, row in enumerate(batch_rows):
                            label = row["object_label"]
                            key = (row["split"], position, layer_index, component, label)
                            vector = vectors[row_index]
                            if key not in sums:
                                sums[key] = torch.zeros_like(vector)
                                unit_sums[key] = torch.zeros_like(vector)
                            norm = float(vector.norm().item())
                            sums[key] += vector
                            unit_sums[key] += vector / max(norm, 1e-12)
                            counts[key] += 1
                            norm_sums[key] += norm
                            object_meta[row["split"]][label] = {
                                "is_fruit": bool(row["is_fruit"]),
                                "color": row["target"],
                            }
            del result, encoded
            if batch_start == 0 or batch_start + batch_size >= len(cases) or (batch_start // batch_size) % 4 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_key} Phase557 natural-color events "
                    f"{min(batch_start + batch_size, len(cases))}/{len(cases)}",
                    flush=True,
                )

        output_rows: list[dict[str, Any]] = []
        for layer_index in range(len(layers)):
            for component in COMPONENTS:
                for position in POSITIONS:
                    split_vectors: dict[str, dict[str, torch.Tensor]] = {}
                    split_unit_stability: dict[str, dict[str, float]] = {}
                    split_geometry: dict[str, dict[str, Any]] = {}
                    category_directions: dict[str, torch.Tensor] = {}
                    color_directions: dict[str, dict[str, torch.Tensor]] = {}
                    object_metrics: dict[str, dict[str, Any]] = {}
                    for split in SPLITS:
                        labels = sorted(object_meta[split])
                        split_vectors[split] = {}
                        split_unit_stability[split] = {}
                        object_metrics[split] = {}
                        for label in labels:
                            key = (split, position, layer_index, component, label)
                            count = counts[key]
                            centroid = sums[key] / count
                            split_vectors[split][label] = centroid
                            stability = float((unit_sums[key] / count).norm().item())
                            split_unit_stability[split][label] = finite(stability)
                            object_metrics[split][label] = {
                                **object_meta[split][label],
                                "surface_order_count": count,
                                "mean_state_norm": finite(norm_sums[key] / count),
                                "surface_direction_stability": finite(stability),
                            }
                        geometry, category_direction, color_direction = aggregate_geometry(
                            split_vectors[split], object_meta[split]
                        )
                        geometry["mean_object_surface_direction_stability"] = finite(
                            sum(split_unit_stability[split].values()) / len(split_unit_stability[split])
                        )
                        split_geometry[split] = geometry
                        category_directions[split] = category_direction
                        color_directions[split] = color_direction

                    shared_colors = sorted(set(color_directions[SPLITS[0]]) & set(color_directions[SPLITS[1]]))
                    shared_color_alignment = {
                        color: cosine(color_directions[SPLITS[0]][color], color_directions[SPLITS[1]][color])
                        for color in shared_colors
                    }
                    output_rows.append({
                        "schema_version": "phase557_natural_color_event.v1",
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": model_key,
                        "torch_dtype": run_dtype,
                        "layer": layer_index,
                        "layer_count": len(layers),
                        "relative_depth": layer_index / max(1, len(layers) - 1),
                        "component": component,
                        "semantic_position": position,
                        "split_geometry": split_geometry,
                        "object_metrics": object_metrics,
                        "cross_split_category_direction_cosine": cosine(
                            category_directions[SPLITS[0]], category_directions[SPLITS[1]]
                        ),
                        "cross_split_shared_color_direction_cosines": shared_color_alignment,
                        "mean_cross_split_shared_color_direction_cosine": pair_mean(
                            list(shared_color_alignment.values())
                        ),
                        "full_vector_persisted": False,
                        "observer_only": True,
                        "causal": False,
                        "compute_edge": False,
                        "sealed": False,
                    })
        write_jsonl(output, output_rows)
        summary = {
            "schema_version": "phase557_natural_color_event_summary.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model_key,
            "torch_dtype": run_dtype,
            "behavior_authorized_relation": "color",
            "case_count": len(cases),
            "split_case_counts": {
                split: sum(row["split"] == split for row in cases) for split in SPLITS
            },
            "layer_count": len(layers),
            "component_count": len(COMPONENTS),
            "semantic_positions": list(POSITIONS),
            "event_row_count": len(output_rows),
            "max_component_ledger_relative_error": finite(max_ledger_error),
            "runtime_seconds": time.monotonic() - started,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "full_vectors_persisted": False,
            "causal_intervention_executed": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_key), summary)
        print(summary_path(model_key))
        return summary_path(model_key)
    finally:
        for handle in handles:
            handle.remove()
        for values in captures.values():
            values.clear()
        current_indices.clear()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.restart)


if __name__ == "__main__":
    main()
