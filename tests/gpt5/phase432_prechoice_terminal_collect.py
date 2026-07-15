#!/usr/bin/env python3
"""Collect Phase432 behavior and pre-choice terminal observer traces."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase431_position_time_collect as p431  # noqa: E402
from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase432_prechoice_terminal_protocol import (  # noqa: E402
    LANGUAGE_MODEL,
    MODELS,
    OPEN_SPLIT,
    OUT,
    PHASE_ID as PROTOCOL_PHASE_ID,
    ROLES,
    ROUTE_MODES,
    SCHEMA_VERSION,
    SCORABLE_ROUTES,
    SEALED_SPLIT,
    freeze,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE_ID = "Phase432-PrechoiceTerminalCollection"
BEHAVIOR_BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}
PHYSICAL_BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 16}
BEHAVIOR_CHECKPOINT = 256
PHYSICAL_CHECKPOINT = 256
POSITION_ROLES = ("question_end", "instruction_end", "prompt_terminal")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase432 non-finite scalar: {value}")
    return round(float(value), 9)


def read_jsonl_any(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return read_jsonl(path)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def groups_for_stage(stage: str) -> list[dict[str, Any]]:
    if stage == "open":
        return read_jsonl(OUT / "phase432_groups_open.jsonl")
    if stage != "sealed":
        raise ValueError(stage)
    gate_path = OUT / "phase432_open_gate.json"
    if not gate_path.exists() or not read_json(gate_path).get("sealed_unlock"):
        raise RuntimeError("Phase432 sealed denominator is not authorized")
    return read_jsonl(OUT / "sealed/phase432_groups_sealed.jsonl")


def materialize_groups(loaded: Any, stage: str) -> list[dict[str, Any]]:
    rows = []
    for group in groups_for_stage(stage):
        for role in ROLES:
            for route_mode in ROUTE_MODES:
                row = p431.materialize_condition(group, role, route_mode, loaded)
                row["schema_version"] = SCHEMA_VERSION
                row["phase_id"] = PHASE_ID
                rows.append(row)
    rows.sort(key=lambda row: row["condition_id"])
    return rows


def transform_behavior(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "schema_version": SCHEMA_VERSION, "phase_id": PHASE_ID}


def collect_behavior(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = OUT / stage / model / "behavior"
    complete_path = root / "phase432_behavior_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase432_behavior_part_*.jsonl"))
    existing = [row for path in existing_paths for row in read_jsonl(path)]
    completed = {row["condition_id"] for row in existing}
    pending = [row for row in rows if row["condition_id"] not in completed]
    part = len(existing_paths)
    buffer: list[dict[str, Any]] = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase432 behavior] {stage} {model}; conditions={len(rows)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), BEHAVIOR_BATCH_SIZE[model]):
        batch = pending[start : start + BEHAVIOR_BATCH_SIZE[model]]
        buffer.extend(transform_behavior(row) for row in p431.collect_behavior_batch(loaded, batch))
        processed += len(batch)
        if processed % BEHAVIOR_CHECKPOINT < len(batch) or processed == len(rows):
            write_jsonl(
                checkpoint_root / f"phase432_behavior_part_{part:05d}.jsonl", buffer
            )
            buffer.clear()
            part += 1
        if processed == len(batch) or processed % 512 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase432 behavior] {stage} {model} {processed}/{len(rows)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    if buffer:
        write_jsonl(checkpoint_root / f"phase432_behavior_part_{part:05d}.jsonl", buffer)
    collected = [
        row
        for path in sorted(checkpoint_root.glob("phase432_behavior_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    unique = {row["condition_id"]: row for row in collected}
    final_rows = [unique[key] for key in sorted(unique)]
    if len(final_rows) != len(rows):
        raise RuntimeError(f"Incomplete Phase432 behavior: {len(final_rows)} != {len(rows)}")
    write_jsonl(root / "phase432_behavior_rows.jsonl", final_rows)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": stage,
        "model": model,
        "condition_count": len(rows),
        "behavior_row_count": len(final_rows),
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in final_rows)),
        "elapsed_seconds": clean(time.monotonic() - started),
        "all_rows_complete": len(final_rows) == len(rows),
        "sealed_read": stage == "sealed",
    }
    write_json(complete_path, complete)
    return complete


def physical_rows(
    rows: list[dict[str, Any]], behavior: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    behavior_map = {row["condition_id"]: row for row in behavior}
    selected = []
    for row in rows:
        if row["candidate"] and row["route_mode"] not in SCORABLE_ROUTES:
            continue
        observed = behavior_map.get(row["condition_id"])
        if observed is None:
            raise RuntimeError(f"Missing behavior row: {row['condition_id']}")
        selected.append(
            {
                **row,
                "actual_choice": observed["actual_choice"],
                "registered_source_choice": observed["registered_source_choice"],
                "natural_target_first": observed["natural_target_first"],
                "natural_opposite_first": observed["natural_opposite_first"],
                "natural_interface_valid": observed["natural_interface_valid"],
                "natural_revision": observed["natural_revision"],
                "natural_boundary": observed["natural_boundary"],
                "natural_stop": observed["natural_stop"],
                "natural_censoring": observed["natural_censoring"],
                "teacher_sequence_correct": observed["teacher_sequence_correct"],
            }
        )
    return selected


def component_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(type(value).__name__)


@torch.inference_mode()
def collect_physical_batch(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registered = [p431.register_positions(fast_tokenizer, row) for row in rows]
    ids = [p431.prompt_ids(loaded, row) for row in rows]
    input_ids, attention_mask, pads = p431.padded_batch(
        ids, int(loaded.tokenizer.pad_token_id), loaded.input_device
    )
    hook_layer = 26 if loaded.key == LANGUAGE_MODEL else round((26 / 35) * (len(layers) - 1))
    hook_captures: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            hook_captures[layer_index] = component_tensor(output).detach()

        return hook

    for layer_index in sorted({hook_layer, len(layers) - 1}):
        handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))
    try:
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    hidden_states = output.hidden_states
    if hidden_states is None or len(hidden_states) != len(layers) + 1:
        raise RuntimeError("Unexpected hidden-state ledger length")
    if set(hook_captures) != {hook_layer, len(layers) - 1}:
        raise RuntimeError("Phase432 registered hook coordinates were not captured")
    hook_error = (
        hook_captures[hook_layer].float() - hidden_states[hook_layer + 1].float()
    ).abs().max().item()
    final_norm, output_head = p431.final_norm_and_head(loaded)
    output_weight = output_head.weight
    directions = torch.stack(
        [
            output_weight[int(row["source_1_first_token_id"])].float()
            - output_weight[int(row["source_2_first_token_id"])].float()
            for row in rows
        ]
    )
    batch_axis = torch.arange(len(rows), device=loaded.input_device)
    terminal = torch.tensor(
        [pads[index] + registered[index]["position_roles"]["prompt_terminal"] for index in range(len(rows))],
        dtype=torch.long,
        device=loaded.input_device,
    )
    native_logits = output.logits[batch_axis, terminal]
    # Hugging Face returns the final hidden-state entry after the model's
    # terminal norm. Earlier entries are raw block outputs.
    reconstructed_logits = output_head(hidden_states[-1][batch_axis, terminal])
    native_top1 = native_logits.argmax(dim=-1)
    reconstructed_top1 = reconstructed_logits.argmax(dim=-1)
    identity_equal = native_top1 == reconstructed_top1
    output_rows = []
    for layer_index in range(len(layers)):
        layer_positions: dict[str, list[float]] = {}
        post = (
            hook_captures[layer_index]
            if layer_index == len(layers) - 1
            else hidden_states[layer_index + 1]
        )
        for role in POSITION_ROLES:
            positions = torch.tensor(
                [pads[index] + registered[index]["position_roles"][role] for index in range(len(rows))],
                dtype=torch.long,
                device=loaded.input_device,
            )
            state = post[batch_axis, positions]
            normalized = final_norm(state)
            margins = (normalized.float() * directions).sum(dim=-1)
            layer_positions[role] = margins.detach().float().cpu().tolist()
        for batch_index, row in enumerate(rows):
            output_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": loaded.key,
                    "stage": "sealed" if row["split"] == SEALED_SPLIT else "open",
                    "split": row["split"],
                    "condition_id": row["condition_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "paired_group_id": row["paired_group_id"],
                    "candidate": row["candidate"],
                    "block_id": row["block_id"],
                    "role": row["role"],
                    "route_mode": row["route_mode"],
                    "source_1": row["source_1"],
                    "source_2": row["source_2"],
                    "actual_choice": row["actual_choice"],
                    "registered_source_choice": row["registered_source_choice"],
                    "layer": layer_index,
                    "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                    "position_metrics": {
                        role: {
                            "source_1_minus_source_2_margin": clean(values[batch_index]),
                            "absolute_token_index": int(registered[batch_index]["position_roles"][role]),
                        }
                        for role, values in layer_positions.items()
                    },
                    "terminal_native_top1_equal": bool(identity_equal[batch_index].item()),
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                }
            )
    diagnostics = {
        "hook_layer": hook_layer,
        "hook_hidden_state_max_abs_error": clean(hook_error),
        "terminal_native_top1_equal": int(identity_equal.sum().item()),
        "terminal_native_top1_total": len(rows),
        "terminal_logit_max_abs_error": clean(
            (native_logits.float() - reconstructed_logits.float()).abs().max().item()
        ),
    }
    return output_rows, diagnostics


def collect_physical(
    loaded: Any,
    model: str,
    stage: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    root = OUT / stage / model / "physical"
    complete_path = root / "phase432_physical_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    behavior = read_jsonl(OUT / stage / model / "behavior/phase432_behavior_rows.jsonl")
    selected = physical_rows(rows, behavior)
    fast_tokenizer = AutoTokenizer.from_pretrained(
        str(loaded.spec.local_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    layers = get_layers(loaded.model)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase432_physical_part_*.jsonl.gz"))
    existing = [row for path in existing_paths for row in read_jsonl_any(path)]
    counts = Counter(row["condition_id"] for row in existing)
    completed = {key for key, value in counts.items() if value == len(layers)}
    pending = [row for row in selected if row["condition_id"] not in completed]
    diagnostics: list[dict[str, Any]] = []
    part = len(existing_paths)
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase432 physical] {stage} {model}; conditions={len(selected)}; "
        f"pending={len(pending)}; layers={len(layers)}",
        flush=True,
    )
    for start in range(0, len(pending), PHYSICAL_BATCH_SIZE[model]):
        batch = pending[start : start + PHYSICAL_BATCH_SIZE[model]]
        traced, diagnostic = collect_physical_batch(
            loaded, fast_tokenizer, layers, batch
        )
        diagnostics.append(diagnostic)
        write_jsonl_gz(
            checkpoint_root / f"phase432_physical_part_{part:05d}.jsonl.gz", traced
        )
        part += 1
        processed += len(batch)
        if processed == len(batch) or processed % PHYSICAL_CHECKPOINT < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase432 physical] {stage} {model} {processed}/{len(selected)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    final_rows = [
        row
        for path in sorted(checkpoint_root.glob("phase432_physical_part_*.jsonl.gz"))
        for row in read_jsonl_any(path)
    ]
    unique = {(row["condition_id"], row["layer"]): row for row in final_rows}
    ordered = [unique[key] for key in sorted(unique)]
    expected = len(selected) * len(layers)
    if len(ordered) != expected:
        raise RuntimeError(f"Incomplete Phase432 physical: {len(ordered)} != {expected}")
    write_jsonl_gz(root / "phase432_physical_rows.jsonl.gz", ordered)
    all_hook_errors = [row["hook_hidden_state_max_abs_error"] for row in diagnostics]
    all_identity_success = sum(row["terminal_native_top1_equal"] for row in diagnostics)
    all_identity_total = sum(row["terminal_native_top1_total"] for row in diagnostics)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "stage": stage,
        "model": model,
        "condition_count": len(selected),
        "layer_count": len(layers),
        "trace_row_count": len(ordered),
        "hook_hidden_state_max_abs_error": max(all_hook_errors, default=0.0),
        "terminal_native_top1_equal": all_identity_success,
        "terminal_native_top1_total": all_identity_total,
        "all_rows_complete": len(ordered) == expected,
        "elapsed_seconds": clean(time.monotonic() - started),
        "sealed_read": stage == "sealed",
    }
    write_json(complete_path, complete)
    return complete


def collect(model: str, stage: str) -> dict[str, Any]:
    freeze()
    if stage == "sealed" and model != LANGUAGE_MODEL:
        raise RuntimeError("Phase432 sealed replication is registered for Qwen3 only")
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        rows = materialize_groups(loaded, stage)
        root = OUT / stage / model
        write_jsonl(root / "phase432_materialized_conditions.jsonl", rows)
        behavior = collect_behavior(loaded, model, stage, rows)
        physical = collect_physical(loaded, model, stage, rows)
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "protocol_phase_id": PROTOCOL_PHASE_ID,
            "created_at": now(),
            "stage": stage,
            "model": model,
            "behavior": behavior,
            "physical": physical,
            "elapsed_seconds": clean(time.monotonic() - started),
            "released_after_run": True,
        }
        write_json(root / "phase432_collection_complete.json", complete)
        return complete
    finally:
        if loaded is not None:
            release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("open", "sealed"), default="open")
    args = parser.parse_args()
    print(json.dumps(collect(args.model, args.stage), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
