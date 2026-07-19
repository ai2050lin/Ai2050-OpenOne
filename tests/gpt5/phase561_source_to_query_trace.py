#!/usr/bin/env python3
"""Trace where an L3 source-color intervention first changes query computation."""

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
from phase557_natural_color_source_intervention import word_scores  # noqa: E402
from phase559_binding_event_collect import semantic_positions, tensor_from_output  # noqa: E402
from phase559_causal_screen import replace_primary  # noqa: E402


MODEL = "qwen3"
POSITIONS = ("query_object_end", "answer_boundary")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CONTRACT_PATH = OUT_DIR / "phase561_source_to_query_trace_frozen_contract.json"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ROWS_PATH = OUT_DIR / "phase561_source_to_query_trace_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase561_source_to_query_trace_execution_summary.json"


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


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def run(batch_size: int, restart: bool) -> Path:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Phase561 batch size must be positive and even")
    contract = read_json(CONTRACT_PATH)
    selected = set(contract["selected_anchor_ids"])
    cases = [
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == contract["split"] and row["anchor_id"] in selected
    ]
    if len(cases) != contract["case_count"] or any(not row["semantic_correct"] for row in cases):
        raise RuntimeError("Phase561 trace denominator drift")
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        pair_groups[row["pair_id"]].append(row)
    pairs = [sorted(rows, key=lambda row: int(row["binding"])) for _, rows in sorted(pair_groups.items())]
    if len(pairs) != contract["counterfactual_pair_count"] or any(len(pair) != 2 for pair in pairs):
        raise RuntimeError("Phase561 counterfactual pairs are incomplete")
    if restart:
        ROWS_PATH.unlink(missing_ok=True)
        SUMMARY_PATH.unlink(missing_ok=True)

    loaded = None
    handles: list[Any] = []
    captures: dict[str, list[dict[str, torch.Tensor]]] = {component: [] for component in COMPONENTS}
    current_indices: dict[str, torch.Tensor] = {}
    metric_sums: defaultdict[tuple[str, int, str, str], float] = defaultdict(float)
    metric_counts: defaultdict[tuple[str, int, str], int] = defaultdict(int)
    cell_sums: defaultdict[tuple[str, int, str, str], float] = defaultdict(float)
    cell_counts: defaultdict[tuple[str, int, str, str], int] = defaultdict(int)
    behavior_count = 0
    behavior_donor_wins = 0
    behavior_switch_effect_sum = 0.0
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError(f"Phase561 model drift: {run_dtype}/{len(layers)}")

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
            handles.extend((
                layer.register_forward_pre_hook(pre_hook),
                layer.self_attn.register_forward_hook(make_hook("attention_output")),
                layer.mlp.register_forward_hook(make_hook("mlp_output")),
                layer.register_forward_hook(make_hook("layer_output")),
            ))

        pair_width = batch_size // 2
        processed_pairs = 0
        for pair_start in range(0, len(pairs), pair_width):
            pair_batch = pairs[pair_start:pair_start + pair_width]
            batch_rows = [row for pair in pair_batch for row in pair]
            donor_indices = [index ^ 1 for index in range(len(batch_rows))]
            individual = [semantic_positions(loaded.tokenizer, row) for row in batch_rows]
            encoded = loaded.tokenizer(
                [row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            sequence_length = int(encoded["input_ids"].shape[1])
            positions: dict[str, list[int]] = {position: [] for position in POSITIONS}
            source_positions = []
            for row_index, (ids, semantic) in enumerate(individual):
                batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                if [int(value) for value in batch_ids] != ids:
                    raise RuntimeError("Phase561 tokenization drift")
                offset = sequence_length - len(ids)
                for position in POSITIONS:
                    positions[position].append(offset + semantic[position])
                source_positions.append(offset + semantic["source_color_end"])
            current_indices.clear()
            current_indices.update({
                position: torch.tensor(indices, dtype=torch.long)
                for position, indices in positions.items()
            })
            source_positions_tensor = torch.tensor(
                source_positions, dtype=torch.long, device=loaded.input_device
            )
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            for values in captures.values():
                values.clear()
            source_states: torch.Tensor | None = None

            def source_capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                nonlocal source_states
                value = tensor_from_output(output)
                batch_index = torch.arange(value.shape[0], device=value.device)
                source_states = value[batch_index, source_positions_tensor, :].detach().float().cpu()

            source_handle = layers[3].register_forward_hook(source_capture)
            with torch.inference_mode():
                natural_result = loaded.model(**encoded, use_cache=False)
            source_handle.remove()
            if source_states is None:
                raise RuntimeError("Phase561 source capture failed")
            natural = {
                component: [dict(values) for values in captures[component]]
                for component in COMPONENTS
            }
            natural_logits = natural_result.logits[:, -1, :].detach().float().cpu()
            replacement = source_states[donor_indices].to(
                device=loaded.input_device, dtype=next(loaded.model.parameters()).dtype
            )
            for values in captures.values():
                values.clear()

            def source_patch(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                primary = tensor_from_output(output).clone()
                batch_index = torch.arange(primary.shape[0], device=primary.device)
                primary[batch_index, source_positions_tensor, :] = replacement
                return replace_primary(output, primary)

            patch_handle = layers[3].register_forward_hook(source_patch)
            with torch.inference_mode():
                intervention_result = loaded.model(**encoded, use_cache=False)
            patch_handle.remove()
            intervention_logits = intervention_result.logits[:, -1, :].detach().float().cpu()

            for index, recipient in enumerate(batch_rows):
                donor = batch_rows[donor_indices[index]]
                baseline_scores = word_scores(
                    natural_logits[index], loaded.tokenizer, recipient["all_candidates"]
                )
                intervention_scores = word_scores(
                    intervention_logits[index], loaded.tokenizer, recipient["all_candidates"]
                )
                baseline_margin = baseline_scores[donor["target"]] - baseline_scores[recipient["target"]]
                intervention_margin = (
                    intervention_scores[donor["target"]] - intervention_scores[recipient["target"]]
                )
                behavior_count += 1
                behavior_donor_wins += max(intervention_scores, key=intervention_scores.get) == donor["target"]
                behavior_switch_effect_sum += intervention_margin - baseline_margin
                cell = (
                    f"query{recipient['query_object_index']}_surface{recipient['surface_id']}_"
                    f"order{recipient['fact_order']}"
                )
                for layer_index in range(len(layers)):
                    for component in COMPONENTS:
                        for position in POSITIONS:
                            recipient_natural = natural[component][layer_index][position][index]
                            donor_natural = natural[component][layer_index][position][donor_indices[index]]
                            recipient_intervention = captures[component][layer_index][position][index]
                            target_delta = donor_natural - recipient_natural
                            causal_delta = recipient_intervention - recipient_natural
                            target_norm = float(target_delta.norm().item())
                            causal_norm = float(causal_delta.norm().item())
                            target_norm_sq = float(torch.dot(target_delta, target_delta).item())
                            projection = (
                                float(torch.dot(causal_delta, target_delta).item()) / target_norm_sq
                                if target_norm_sq > 1e-12 else 0.0
                            )
                            cosine_denominator = target_norm * causal_norm
                            alignment = (
                                float(torch.dot(causal_delta, target_delta).item()) / cosine_denominator
                                if cosine_denominator > 1e-12 else 0.0
                            )
                            ratio = causal_norm / max(target_norm, 1e-12)
                            coord = (position, layer_index, component)
                            metric_sums[(*coord, "mean_causal_to_natural_norm_ratio")] += ratio
                            metric_sums[(*coord, "mean_causal_projection_to_natural")] += projection
                            metric_sums[(*coord, "mean_causal_natural_direction_cosine")] += alignment
                            metric_sums[(*coord, "mean_causal_delta_norm")] += causal_norm
                            metric_sums[(*coord, "mean_natural_pair_delta_norm")] += target_norm
                            metric_counts[coord] += 1
                            cell_key = (*coord, cell)
                            cell_sums[cell_key] += ratio
                            cell_counts[cell_key] += 1
            processed_pairs += len(pair_batch)
            del natural_result, intervention_result, encoded, natural, replacement
            if pair_start == 0 or processed_pairs == len(pairs) or processed_pairs % 40 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] qwen3 Phase561 causal trace "
                    f"{processed_pairs}/{len(pairs)} pairs",
                    flush=True,
                )

        output_rows = []
        for layer_index in range(len(layers)):
            for component in COMPONENTS:
                for position in POSITIONS:
                    coord = (position, layer_index, component)
                    count = metric_counts[coord]
                    cell_ratios = {
                        cell: finite(cell_sums[(*coord, cell)] / cell_counts[(*coord, cell)])
                        for cell in sorted({key[-1] for key in cell_counts if key[:3] == coord})
                    }
                    output_rows.append({
                        "schema_version": "phase561_source_to_query_trace.v1",
                        "phase_id": "Phase561",
                        "created_at": now(),
                        "model": MODEL,
                        "torch_dtype": run_dtype,
                        "source_intervention_layer": 3,
                        "source_intervention_component": "layer_output",
                        "source_intervention_position": "source_color_end",
                        "layer": layer_index,
                        "relative_depth": layer_index / max(1, len(layers) - 1),
                        "component": component,
                        "semantic_position": position,
                        "case_count": count,
                        "mean_causal_to_natural_norm_ratio": finite(
                            metric_sums[(*coord, "mean_causal_to_natural_norm_ratio")] / count
                        ),
                        "mean_causal_projection_to_natural": finite(
                            metric_sums[(*coord, "mean_causal_projection_to_natural")] / count
                        ),
                        "mean_causal_natural_direction_cosine": finite(
                            metric_sums[(*coord, "mean_causal_natural_direction_cosine")] / count
                        ),
                        "mean_causal_delta_norm": finite(
                            metric_sums[(*coord, "mean_causal_delta_norm")] / count
                        ),
                        "mean_natural_pair_delta_norm": finite(
                            metric_sums[(*coord, "mean_natural_pair_delta_norm")] / count
                        ),
                        "factorial_cell_causal_to_natural_norm_ratios": cell_ratios,
                        "minimum_factorial_cell_causal_to_natural_norm_ratio": min(
                            cell_ratios.values(), default=0.0
                        ),
                        "intervention_conditioned_observation": True,
                        "reader_compute_edge": False,
                        "full_vector_persisted": False,
                        "sealed": False,
                    })
        write_jsonl(ROWS_PATH, output_rows)
        summary = {
            "schema_version": "phase561_source_to_query_trace_execution_summary.v1",
            "phase_id": "Phase561",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "counterfactual_pair_count": len(pairs),
            "event_row_count": len(output_rows),
            "source_patch_donor_win_rate": behavior_donor_wins / behavior_count,
            "source_patch_mean_donor_switch_effect": behavior_switch_effect_sum / behavior_count,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "full_vectors_persisted": False,
            "reader_intervention_executed": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        print(SUMMARY_PATH)
        return SUMMARY_PATH
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.restart)
