#!/usr/bin/env python3
"""Collect Phase433 behavior, event trajectories and source-write ledgers."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import statistics
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
from phase429_typed_route_protocol import render_chat  # noqa: E402
from phase433_shared_prefix_protocol import (  # noqa: E402
    LANGUAGE_MODEL,
    MAIN_ROUTES,
    MODELS,
    OPEN_SPLITS,
    OUT,
    PHASE_ID as PROTOCOL_PHASE_ID,
    ROLES,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    STRESS_ROUTES,
    TRACE_SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    route_tags,
    write_json,
    write_jsonl,
)


PHASE_ID = "Phase433-SharedPrefixCollection"
BEHAVIOR_BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}
PHYSICAL_BATCH_SIZE = {"qwen3": 24, "glm4": 6, "deepseek7b": 12}
COMPONENT_BATCH_SIZE = 8
BEHAVIOR_CHECKPOINT = 256
PHYSICAL_CHECKPOINT = 192
QUESTION_LINE = "Question: Which item is selected?"
INSTRUCTION_LINE = "Output exactly the selected item and then stop."
POSITION_ROLES = (
    "source_1_end",
    "source_2_end",
    "question_end",
    "instruction_start",
    "instruction_mid",
    "instruction_end",
    "assistant_boundary",
    "prompt_terminal",
    "teacher_branch_boundary",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase433 non-finite scalar: {value}")
    return round(float(value), 9)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def read_jsonl_any(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return read_jsonl(path)


def digest_ids(ids: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in ids).encode("ascii")
    ).hexdigest()


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def contextual_continuation_ids(
    tokenizer: Any, rendered_prompt: str, prompt_ids: list[int], continuation: str
) -> tuple[list[int], bool]:
    joined = [
        int(value)
        for value in tokenizer(
            rendered_prompt + continuation, add_special_tokens=False
        )["input_ids"]
    ]
    prefix_exact = joined[: len(prompt_ids)] == prompt_ids
    if prefix_exact:
        return joined[len(prompt_ids) :], True
    fallback = [
        int(value)
        for value in tokenizer(continuation, add_special_tokens=False)["input_ids"]
    ]
    return fallback, False


def groups_for_stage(stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if stage == "open":
        return (
            read_jsonl(OUT / "phase433_groups_open.jsonl"),
            read_jsonl(OUT / "phase433_groups_stress_open.jsonl"),
        )
    if stage != "sealed":
        raise ValueError(stage)
    gate_path = OUT / "phase433_open_gate.json"
    if not gate_path.exists() or not read_json(gate_path).get("sealed_unlock"):
        raise RuntimeError("Phase433 sealed denominator is not authorized")
    return read_jsonl(OUT / "sealed/phase433_groups_sealed.jsonl"), []


def materialize_condition(
    group: dict[str, Any], role: str, route_mode: str, loaded: Any
) -> dict[str, Any]:
    before, after, source_role, query_role = route_tags(route_mode, role)
    if group["candidate"]:
        semantic_target = group["role_targets"][role]
        opposite_role = "b" if role == "a" else "a"
        semantic_opposite = group["role_targets"][opposite_role]
        query_semantic = (
            semantic_opposite if route_mode == "conflict" else semantic_target
        )
        normative = route_mode in MAIN_ROUTES
    else:
        semantic_target = group["stable_target"]
        semantic_opposite = group["decoy"]
        query_semantic = semantic_target
        normative = route_mode in MAIN_ROUTES
    before_line = f"Before-source selector slot: {before}."
    after_line = f"After-source selector slot: {after}."
    content = "\n".join(
        (
            "Follow this deterministic selector task.",
            group["task"],
            before_line,
            group["record"],
            after_line,
            QUESTION_LINE,
            INSTRUCTION_LINE,
        )
    )
    rendered = render_chat(loaded.tokenizer, loaded.key, content)
    prompt_ids = [
        int(value)
        for value in loaded.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]
    source_1_ids, source_1_context_exact = contextual_continuation_ids(
        loaded.tokenizer, rendered, prompt_ids, group["source_1"]
    )
    source_2_ids, source_2_context_exact = contextual_continuation_ids(
        loaded.tokenizer, rendered, prompt_ids, group["source_2"]
    )
    common_length = common_prefix_length(source_1_ids, source_2_ids)
    if (
        common_length < 2
        or common_length >= len(source_1_ids)
        or common_length >= len(source_2_ids)
    ):
        raise RuntimeError(
            f"Invalid shared-prefix token contract: {group['semantic_group_id']} "
            f"{len(source_1_ids)=} {len(source_2_ids)=} {common_length=}"
        )
    target_ids = source_1_ids if semantic_target == group["source_1"] else source_2_ids
    opposite_ids = source_2_ids if semantic_target == group["source_1"] else source_1_ids
    target_source = "source_1" if semantic_target == group["source_1"] else "source_2"
    condition_id = (
        f"{group['semantic_group_id']}__r{role}__route_{route_mode}__{loaded.key}"
    )
    return {
        **{
            key: value
            for key, value in group.items()
            if key not in {"task", "record", "history_text", "role_targets"}
        },
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": loaded.key,
        "condition_id": condition_id,
        "interface": "direct_item",
        "role": role,
        "route_mode": route_mode,
        "source_role": source_role,
        "query_role": query_role,
        "source_route_target": semantic_target,
        "query_route_target": query_semantic,
        "content_prompt": content,
        "rendered_prompt": rendered,
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": digest_ids(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "semantic_target": semantic_target,
        "semantic_opposite": semantic_opposite,
        "semantic_target_source": target_source,
        "target": semantic_target,
        "opposite_target": semantic_opposite,
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "source_1_sequence_token_ids": source_1_ids,
        "source_2_sequence_token_ids": source_2_ids,
        "common_prefix_token_ids": source_1_ids[:common_length],
        "common_prefix_token_count": common_length,
        "source_1_branch_token_id": source_1_ids[common_length],
        "source_2_branch_token_id": source_2_ids[common_length],
        "source_1_token_count": len(source_1_ids),
        "source_2_token_count": len(source_2_ids),
        "same_first_token": source_1_ids[0] == source_2_ids[0],
        "contextual_tokenization_exact": bool(
            source_1_context_exact and source_2_context_exact
        ),
        "natural_generation_max_new_tokens": 16,
        "normative_target": normative,
        "descriptive_none_only": route_mode == "none",
        "descriptive_conflict_only": route_mode == "conflict",
        "before_selector": before,
        "after_selector": after,
        "before_line": before_line,
        "after_line": after_line,
        "question_line": QUESTION_LINE,
        "instruction_line": INSTRUCTION_LINE,
        "record_line": group["record"],
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def materialize_groups(loaded: Any, stage: str) -> list[dict[str, Any]]:
    main_groups, stress_groups = groups_for_stage(stage)
    rows: list[dict[str, Any]] = []
    for group in main_groups:
        for role in ROLES:
            for route in MAIN_ROUTES:
                rows.append(materialize_condition(group, role, route, loaded))
    for group in stress_groups:
        for role in ROLES:
            for route in STRESS_ROUTES:
                rows.append(materialize_condition(group, role, route, loaded))
    rows.sort(key=lambda row: row["condition_id"])
    return rows


def enrich_behavior(source: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    generated = [int(value) for value in source["natural_generated_token_ids"]]
    prefix = [int(value) for value in row["common_prefix_token_ids"]]
    prefix_exact = generated[: len(prefix)] == prefix
    reaches_branch = prefix_exact and len(generated) > len(prefix)
    observed_branch = generated[len(prefix)] if reaches_branch else None
    if observed_branch == int(row["source_1_branch_token_id"]):
        branch_choice = "source_1"
    elif observed_branch == int(row["source_2_branch_token_id"]):
        branch_choice = "source_2"
    else:
        branch_choice = "other"
    return {
        **source,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "semantic_target_source": row["semantic_target_source"],
        "stress_only": row["stress_only"],
        "common_prefix_token_count": row["common_prefix_token_count"],
        "same_first_token": row["same_first_token"],
        "contextual_tokenization_exact": row["contextual_tokenization_exact"],
        "natural_common_prefix_exact": prefix_exact,
        "natural_reaches_branch_boundary": reaches_branch,
        "natural_observed_branch_token_id": observed_branch,
        "natural_branch_choice": branch_choice,
        "natural_branch_correct": branch_choice == row["semantic_target_source"],
        "natural_complete_event_correct": bool(
            source["natural_exact_target_contract"]
            and source["actual_choice"] == row["semantic_target_source"]
        ),
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def collect_behavior(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = OUT / stage / model / "behavior"
    complete_path = root / "phase433_behavior_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase433_behavior_part_*.jsonl"))
    existing = [row for path in existing_paths for row in read_jsonl(path)]
    completed = {row["condition_id"] for row in existing}
    pending = [row for row in rows if row["condition_id"] not in completed]
    part = len(existing_paths)
    buffer: list[dict[str, Any]] = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase433 behavior] {stage} {model}; conditions={len(rows)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), BEHAVIOR_BATCH_SIZE[model]):
        batch = pending[start : start + BEHAVIOR_BATCH_SIZE[model]]
        raw = p431.collect_behavior_batch(loaded, batch)
        buffer.extend(enrich_behavior(item, row) for item, row in zip(raw, batch))
        processed += len(batch)
        if processed % BEHAVIOR_CHECKPOINT < len(batch) or processed == len(rows):
            write_jsonl(
                checkpoint_root / f"phase433_behavior_part_{part:05d}.jsonl",
                buffer,
            )
            buffer.clear()
            part += 1
        if processed == len(batch) or processed % 512 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase433 behavior] {stage} {model} {processed}/{len(rows)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    if buffer:
        write_jsonl(
            checkpoint_root / f"phase433_behavior_part_{part:05d}.jsonl", buffer
        )
    collected = [
        row
        for path in sorted(checkpoint_root.glob("phase433_behavior_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    unique = {row["condition_id"]: row for row in collected}
    final_rows = [unique[key] for key in sorted(unique)]
    if len(final_rows) != len(rows):
        raise RuntimeError(f"Incomplete Phase433 behavior: {len(final_rows)} != {len(rows)}")
    write_jsonl(root / "phase433_materialized_conditions.jsonl", rows)
    write_jsonl(root / "phase433_behavior_rows.jsonl", final_rows)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "condition_count": len(rows),
        "main_condition_count": sum(not row["stress_only"] for row in rows),
        "stress_condition_count": sum(row["stress_only"] for row in rows),
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in final_rows)),
        "common_prefix_length_range": [
            min(row["common_prefix_token_count"] for row in rows),
            max(row["common_prefix_token_count"] for row in rows),
        ],
        "token_contract_valid_count": sum(
            row["same_first_token"]
            and row["contextual_tokenization_exact"]
            and row["common_prefix_token_count"] >= 2
            for row in rows
        ),
        "elapsed_seconds": clean(time.monotonic() - started),
        "all_rows_complete": len(final_rows) == len(rows),
        "sealed_read": stage == "sealed",
    }
    write_json(complete_path, complete)
    return complete


def register_positions(fast_tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    base = p431.register_positions(fast_tokenizer, row)
    rendered = row["rendered_prompt"]
    encoded = fast_tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    instruction_start_char = rendered.find(row["instruction_line"])
    instruction = p431.token_positions(
        rendered, offsets, row["instruction_line"], instruction_start_char
    )
    content_start = rendered.find(row["content_prompt"])
    content_end = content_start + len(row["content_prompt"])
    template = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left >= content_end
    ]
    common_count = int(row["common_prefix_token_count"])
    prompt_count = len(base["input_ids"])
    roles = dict(base["position_roles"])
    roles.update(
        {
            "instruction_start": instruction[0],
            "instruction_mid": instruction[len(instruction) // 2],
            "teacher_branch_boundary": prompt_count + common_count - 1,
        }
    )
    partitions = dict(base["source_partitions"])
    partitions["assistant_template"] = template
    partitions["generated_common_prefix"] = list(
        range(prompt_count, prompt_count + common_count)
    )
    return {
        "input_ids": base["input_ids"],
        "position_roles": roles,
        "source_partitions": partitions,
    }


def physical_input_rows(
    rows: list[dict[str, Any]], behavior: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    behavior_map = {row["condition_id"]: row for row in behavior}
    selected = []
    for row in rows:
        if row.get("stress_only", row["split"] == "conflict_stress"):
            continue
        observed = behavior_map[row["condition_id"]]
        selected.append(
            {
                **row,
                "actual_choice": observed["actual_choice"],
                "registered_source_choice": observed["registered_source_choice"],
                "teacher_sequence_correct": observed["teacher_sequence_correct"],
                "teacher_sequence_logprob_margin": observed[
                    "teacher_sequence_logprob_margin"
                ],
                "natural_common_prefix_exact": observed[
                    "natural_common_prefix_exact"
                ],
                "natural_reaches_branch_boundary": observed[
                    "natural_reaches_branch_boundary"
                ],
                "natural_branch_correct": observed["natural_branch_correct"],
                "natural_complete_event_correct": observed[
                    "natural_complete_event_correct"
                ],
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
    registered = [register_positions(fast_tokenizer, row) for row in rows]
    prompt_ids = [p431.prompt_ids(loaded, row) for row in rows]
    sequences = [
        [*ids, *[int(value) for value in row["common_prefix_token_ids"]]]
        for ids, row in zip(prompt_ids, rows)
    ]
    input_ids, attention_mask, pads = p431.padded_batch(
        sequences, int(loaded.tokenizer.pad_token_id), loaded.input_device
    )
    candidate_hook = (
        26
        if loaded.key == LANGUAGE_MODEL
        else round((26 / 35) * (len(layers) - 1))
    )
    hook_layers = sorted({candidate_hook, len(layers) - 1})
    captures: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            captures[index] = component_tensor(output).detach()

        return hook

    for index in hook_layers:
        handles.append(layers[index].register_forward_hook(make_hook(index)))
    try:
        result = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    hidden_states = result.hidden_states
    if hidden_states is None or len(hidden_states) != len(layers) + 1:
        raise RuntimeError("Unexpected Phase433 hidden-state ledger")
    final_norm, output_head = p431.final_norm_and_head(loaded)
    weight = output_head.weight
    directions = torch.stack(
        [
            weight[int(row["source_1_branch_token_id"])].float()
            - weight[int(row["source_2_branch_token_id"])].float()
            for row in rows
        ]
    )
    batch_axis = torch.arange(len(rows), device=loaded.input_device)
    prompt_terminal = torch.tensor(
        [
            pads[index] + registered[index]["position_roles"]["prompt_terminal"]
            for index in range(len(rows))
        ],
        dtype=torch.long,
        device=loaded.input_device,
    )
    branch_boundary = torch.tensor(
        [
            pads[index]
            + registered[index]["position_roles"]["teacher_branch_boundary"]
            for index in range(len(rows))
        ],
        dtype=torch.long,
        device=loaded.input_device,
    )
    terminal_native = result.logits[batch_axis, prompt_terminal]
    branch_native = result.logits[batch_axis, branch_boundary]
    terminal_rebuilt = output_head(hidden_states[-1][batch_axis, prompt_terminal])
    branch_rebuilt = output_head(hidden_states[-1][batch_axis, branch_boundary])
    terminal_equal = terminal_native.argmax(-1) == terminal_rebuilt.argmax(-1)
    branch_equal = branch_native.argmax(-1) == branch_rebuilt.argmax(-1)
    rows_out: list[dict[str, Any]] = []
    for layer_index in range(len(layers)):
        post = (
            captures[layer_index]
            if layer_index == len(layers) - 1
            else hidden_states[layer_index + 1]
        )
        position_values: dict[str, list[float]] = {}
        for role in POSITION_ROLES:
            positions = torch.tensor(
                [
                    pads[index] + registered[index]["position_roles"][role]
                    for index in range(len(rows))
                ],
                dtype=torch.long,
                device=loaded.input_device,
            )
            state = post[batch_axis, positions]
            margins = (final_norm(state).float() * directions).sum(dim=-1)
            position_values[role] = margins.detach().float().cpu().tolist()
        for index, row in enumerate(rows):
            rows_out.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
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
                    "actual_choice": row["actual_choice"],
                    "semantic_target_source": row["semantic_target_source"],
                    "registered_source_choice": row["registered_source_choice"],
                    "common_prefix_token_count": row["common_prefix_token_count"],
                    "natural_common_prefix_exact": row[
                        "natural_common_prefix_exact"
                    ],
                    "natural_reaches_branch_boundary": row[
                        "natural_reaches_branch_boundary"
                    ],
                    "natural_branch_correct": row["natural_branch_correct"],
                    "natural_complete_event_correct": row[
                        "natural_complete_event_correct"
                    ],
                    "teacher_sequence_correct": row["teacher_sequence_correct"],
                    "teacher_sequence_logprob_margin": clean(
                        row["teacher_sequence_logprob_margin"]
                    ),
                    "layer": layer_index,
                    "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                    "position_metrics": {
                        role: {
                            "source_1_minus_source_2_branch_margin": clean(
                                values[index]
                            ),
                            "absolute_token_index": int(
                                registered[index]["position_roles"][role]
                            ),
                            "time_status": (
                                "teacher_forced_pre_divergence"
                                if role == "teacher_branch_boundary"
                                else "natural_prompt_pre_generation"
                            ),
                        }
                        for role, values in position_values.items()
                    },
                    "terminal_native_top1_equal": bool(terminal_equal[index].item()),
                    "branch_native_top1_equal": bool(branch_equal[index].item()),
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                }
            )
    hook_error = (
        captures[candidate_hook].float() - hidden_states[candidate_hook + 1].float()
    ).abs().max().item()
    diagnostic = {
        "hook_layer": candidate_hook,
        "hook_hidden_state_max_abs_error": clean(hook_error),
        "terminal_native_top1_equal": int(terminal_equal.sum().item()),
        "branch_native_top1_equal": int(branch_equal.sum().item()),
        "identity_total": len(rows),
    }
    return rows_out, diagnostic


def collect_physical(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = OUT / stage / model / "physical"
    complete_path = root / "phase433_physical_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    behavior = read_jsonl(
        OUT / stage / model / "behavior/phase433_behavior_rows.jsonl"
    )
    selected = physical_input_rows(rows, behavior)
    fast_tokenizer = AutoTokenizer.from_pretrained(
        str(loaded.spec.local_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    layers = get_layers(loaded.model)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase433_physical_part_*.jsonl.gz"))
    existing = [row for path in existing_paths for row in read_jsonl_any(path)]
    counts = Counter(row["condition_id"] for row in existing)
    completed = {key for key, value in counts.items() if value == len(layers)}
    pending = [row for row in selected if row["condition_id"] not in completed]
    part = len(existing_paths)
    diagnostics = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase433 physical] {stage} {model}; conditions={len(selected)}; "
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
            checkpoint_root / f"phase433_physical_part_{part:05d}.jsonl.gz",
            traced,
        )
        part += 1
        processed += len(batch)
        if processed == len(batch) or processed % PHYSICAL_CHECKPOINT < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase433 physical] {stage} {model} {processed}/{len(selected)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    final_rows = [
        row
        for path in sorted(checkpoint_root.glob("phase433_physical_part_*.jsonl.gz"))
        for row in read_jsonl_any(path)
    ]
    unique = {(row["condition_id"], row["layer"]): row for row in final_rows}
    ordered = [unique[key] for key in sorted(unique)]
    expected = len(selected) * len(layers)
    if len(ordered) != expected:
        raise RuntimeError(f"Incomplete Phase433 physical: {len(ordered)} != {expected}")
    write_jsonl_gz(root / "phase433_physical_rows.jsonl.gz", ordered)
    complete = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "condition_count": len(selected),
        "layer_count": len(layers),
        "trace_row_count": len(ordered),
        "hook_hidden_state_max_abs_error": max(
            (item["hook_hidden_state_max_abs_error"] for item in diagnostics),
            default=0.0,
        ),
        "terminal_native_top1_equal": sum(
            item["terminal_native_top1_equal"] for item in diagnostics
        ),
        "branch_native_top1_equal": sum(
            item["branch_native_top1_equal"] for item in diagnostics
        ),
        "identity_total": sum(item["identity_total"] for item in diagnostics),
        "all_rows_complete": len(ordered) == expected,
        "elapsed_seconds": clean(time.monotonic() - started),
        "sealed_read": stage == "sealed",
    }
    write_json(complete_path, complete)
    return complete


def install_component_hooks(
    layers: list[Any], selected_layers: list[int], captures: dict[tuple[str, int], torch.Tensor]
) -> list[Any]:
    handles = []
    for layer_index in selected_layers:
        layer = layers[layer_index]
        value_projection = p431.module_attr(layer.self_attn, ("v_proj", "value"))

        def pre_hook(_module: Any, args: tuple[Any, ...], idx: int = layer_index) -> None:
            captures[("pre", idx)] = args[0].detach()

        def value_hook(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("value", idx)] = component_tensor(output).detach()

        def attention_hook(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("attention", idx)] = component_tensor(output).detach()
            if isinstance(output, (tuple, list)) and len(output) > 1 and torch.is_tensor(output[1]):
                captures[("probabilities", idx)] = output[1].detach()

        def mlp_hook(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("mlp", idx)] = component_tensor(output).detach()

        def post_hook(
            _module: Any, _args: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            captures[("post", idx)] = component_tensor(output).detach()

        handles.extend(
            (
                layer.register_forward_pre_hook(pre_hook),
                value_projection.register_forward_hook(value_hook),
                layer.self_attn.register_forward_hook(attention_hook),
                layer.mlp.register_forward_hook(mlp_hook),
                layer.register_forward_hook(post_hook),
            )
        )
    return handles


@torch.inference_mode()
def collect_component_batch(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    selected_layers: list[int],
) -> list[dict[str, Any]]:
    registered = [register_positions(fast_tokenizer, row) for row in rows]
    prompts = [p431.prompt_ids(loaded, row) for row in rows]
    sequences = [
        [*prompt, *[int(value) for value in row["common_prefix_token_ids"]]]
        for prompt, row in zip(prompts, rows)
    ]
    input_ids, attention_mask, pads = p431.padded_batch(
        sequences, int(loaded.tokenizer.pad_token_id), loaded.input_device
    )
    captures: dict[tuple[str, int], torch.Tensor] = {}
    handles = install_component_hooks(layers, selected_layers, captures)
    try:
        loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    final_norm, output_head = p431.final_norm_and_head(loaded)
    directions = torch.stack(
        [
            output_head.weight[int(row["source_1_branch_token_id"])].float()
            - output_head.weight[int(row["source_2_branch_token_id"])].float()
            for row in rows
        ]
    )
    sequence_width = input_ids.shape[1]
    sequence_axis = torch.arange(sequence_width, device=loaded.input_device)
    receiver_names = ("prompt_terminal", "teacher_branch_boundary")
    output_rows: list[dict[str, Any]] = []
    for layer_index in selected_layers:
        probabilities = captures[("probabilities", layer_index)].float()
        value_raw = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        attention_module = layers[layer_index].self_attn
        kv_heads = int(
            getattr(attention_module, "num_key_value_heads", 0)
            or getattr(attention_module.config, "num_key_value_heads", 0)
        )
        head_dim = int(value_raw.shape[-1] // kv_heads)
        values = value_raw.view(
            value_raw.shape[0], value_raw.shape[1], kv_heads, head_dim
        ).permute(0, 2, 1, 3)
        if kv_heads != head_count:
            values = values.repeat_interleave(head_count // kv_heads, dim=1)
        output_projection = p431.module_attr(attention_module, ("o_proj", "dense"))
        output_blocks = output_projection.weight.float().view(
            output_projection.weight.shape[0], head_count, head_dim
        )
        bias = (
            output_projection.bias.float()
            if output_projection.bias is not None
            else None
        )
        receiver_payloads: list[dict[str, Any]] = [{} for _ in rows]
        for receiver_name in receiver_names:
            receiver = torch.tensor(
                [
                    pads[index] + registered[index]["position_roles"][receiver_name]
                    for index in range(len(rows))
                ],
                dtype=torch.long,
                device=loaded.input_device,
            )
            local_axis = torch.arange(len(rows), device=loaded.input_device)
            receiver_probabilities = probabilities[local_axis, :, receiver, :]
            causal_mask = torch.zeros(
                (len(rows), sequence_width), dtype=torch.bool, device=loaded.input_device
            )
            for index in range(len(rows)):
                causal_mask[index, pads[index] : int(receiver[index].item()) + 1] = True
            occupied = torch.zeros_like(causal_mask)
            masks: dict[str, torch.Tensor] = {}
            partition_names = tuple(registered[0]["source_partitions"])
            for partition_name in partition_names:
                mask = torch.zeros_like(causal_mask)
                for index, item in enumerate(registered):
                    for position in item["source_partitions"][partition_name]:
                        shifted = pads[index] + int(position)
                        if shifted <= int(receiver[index].item()):
                            mask[index, shifted] = True
                mask &= causal_mask
                masks[partition_name] = mask
                occupied |= mask
            masks["other_positions"] = causal_mask & ~occupied
            replay = torch.zeros(
                (len(rows), output_blocks.shape[0]),
                dtype=torch.float32,
                device=loaded.input_device,
            )
            partitions: dict[str, list[dict[str, float | int]]] = {}
            for partition_name, mask in masks.items():
                alpha = receiver_probabilities * mask.unsqueeze(1)
                weighted = torch.einsum("nhs,nhsd->nhd", alpha, values.float())
                head_writes = torch.einsum("nhd,ohd->nho", weighted, output_blocks)
                write = head_writes.sum(dim=1)
                replay += write
                mass = alpha.sum(dim=-1).mean(dim=-1)
                norm = torch.linalg.vector_norm(write, dim=-1)
                margin = (write * directions).sum(dim=-1)
                partitions[partition_name] = [
                    {
                        "token_count": int(mask[index].sum().item()),
                        "attention_mass_mean": clean(mass[index].item()),
                        "write_norm": clean(norm[index].item()),
                        "branch_margin_write": clean(margin[index].item()),
                    }
                    for index in range(len(rows))
                ]
            if bias is not None:
                replay += bias
            actual_attention = captures[("attention", layer_index)][
                local_axis, receiver
            ].float()
            replay_error = torch.linalg.vector_norm(
                actual_attention - replay, dim=-1
            ) / torch.linalg.vector_norm(actual_attention, dim=-1).clamp_min(1e-8)
            pre = captures[("pre", layer_index)][local_axis, receiver]
            mlp = captures[("mlp", layer_index)][local_axis, receiver]
            post = captures[("post", layer_index)][local_axis, receiver]
            residual_margin = (final_norm(post).float() * directions).sum(dim=-1)
            attention_margin = (actual_attention * directions).sum(dim=-1)
            mlp_margin = (mlp.float() * directions).sum(dim=-1)
            for index in range(len(rows)):
                receiver_payloads[index][receiver_name] = {
                    "absolute_token_index": int(
                        registered[index]["position_roles"][receiver_name]
                    ),
                    "residual_pre_rms": clean(
                        torch.sqrt(torch.mean(pre[index].float() ** 2)).item()
                    ),
                    "residual_post_branch_margin": clean(residual_margin[index].item()),
                    "attention_branch_margin_write": clean(attention_margin[index].item()),
                    "mlp_branch_margin_write": clean(mlp_margin[index].item()),
                    "attention_replay_relative_error": clean(replay_error[index].item()),
                    "source_partition": {
                        name: values_for_name[index]
                        for name, values_for_name in partitions.items()
                    },
                }
        for index, row in enumerate(rows):
            output_rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": loaded.key,
                    "stage": "open",
                    "split": row["split"],
                    "condition_id": row["condition_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "paired_group_id": row["paired_group_id"],
                    "candidate": row["candidate"],
                    "role": row["role"],
                    "route_mode": row["route_mode"],
                    "actual_choice": row["actual_choice"],
                    "layer": layer_index,
                    "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                    "receiver_metrics": receiver_payloads[index],
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                }
            )
    return output_rows


def collect_components(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = OUT / stage / model / "components"
    complete_path = root / "phase433_component_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    if model != LANGUAGE_MODEL or stage != "open":
        skipped = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "stage": stage,
            "skipped": True,
            "reason": "component source-write ledger preregistered only for qwen3 open physical calibration",
            "all_rows_complete": True,
        }
        write_json(complete_path, skipped)
        return skipped
    behavior = read_jsonl(
        OUT / stage / model / "behavior/phase433_behavior_rows.jsonl"
    )
    all_physical = physical_input_rows(rows, behavior)
    selected = [
        row
        for row in all_physical
        if row["split"] == "physical_calibration" and row["route_mode"] == "consistent"
    ]
    fast_tokenizer = AutoTokenizer.from_pretrained(
        str(loaded.spec.local_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    layers = get_layers(loaded.model)
    selected_layers = [24, 25, 26, 27, 28, 29]
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase433_component_part_*.jsonl.gz"))
    existing = [row for path in existing_paths for row in read_jsonl_any(path)]
    counts = Counter(row["condition_id"] for row in existing)
    completed = {
        key for key, value in counts.items() if value == len(selected_layers)
    }
    pending = [row for row in selected if row["condition_id"] not in completed]
    part = len(existing_paths)
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase433 components] open qwen3; conditions={len(selected)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), COMPONENT_BATCH_SIZE):
        batch = pending[start : start + COMPONENT_BATCH_SIZE]
        traced = collect_component_batch(
            loaded, fast_tokenizer, layers, batch, selected_layers
        )
        write_jsonl_gz(
            checkpoint_root / f"phase433_component_part_{part:05d}.jsonl.gz",
            traced,
        )
        part += 1
        processed += len(batch)
        if processed == len(batch) or processed % 128 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase433 components] {processed}/{len(selected)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    final_rows = [
        row
        for path in sorted(checkpoint_root.glob("phase433_component_part_*.jsonl.gz"))
        for row in read_jsonl_any(path)
    ]
    unique = {(row["condition_id"], row["layer"]): row for row in final_rows}
    ordered = [unique[key] for key in sorted(unique)]
    expected = len(selected) * len(selected_layers)
    if len(ordered) != expected:
        raise RuntimeError(f"Incomplete component ledger: {len(ordered)} != {expected}")
    write_jsonl_gz(root / "phase433_component_rows.jsonl.gz", ordered)
    errors = [
        receiver["attention_replay_relative_error"]
        for row in ordered
        for receiver in row["receiver_metrics"].values()
    ]
    complete = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "condition_count": len(selected),
        "selected_layers": selected_layers,
        "trace_row_count": len(ordered),
        "attention_replay_relative_error_median": clean(statistics.median(errors)),
        "attention_replay_relative_error_max": clean(max(errors)),
        "elapsed_seconds": clean(time.monotonic() - started),
        "all_rows_complete": len(ordered) == expected,
        "head_channel_neuron_scan": False,
        "intervention": False,
        "causal": False,
    }
    write_json(complete_path, complete)
    return complete


def collect(model: str, stage: str, mode: str) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(model)
    if not (OUT / "phase433_protocol.json").exists():
        protocol = freeze()
    else:
        protocol = read_json(OUT / "phase433_protocol.json")
    if protocol["phase_id"] != PROTOCOL_PHASE_ID:
        raise RuntimeError("Phase433 protocol identity mismatch")
    loaded = None
    try:
        loaded = load_probe_model(model)
        rows = materialize_groups(loaded, stage)
        output: dict[str, Any] = {"model": model, "stage": stage, "mode": mode}
        if mode in {"behavior", "all"}:
            output["behavior"] = collect_behavior(loaded, model, stage, rows)
        if mode in {"physical", "component", "all"}:
            behavior_path = (
                OUT / stage / model / "behavior/phase433_behavior_complete.json"
            )
            if not behavior_path.exists():
                raise RuntimeError("Behavior must complete before Phase433 physical collection")
        if mode in {"physical", "all"}:
            output["physical"] = collect_physical(loaded, model, stage, rows)
        if mode in {"component", "all"}:
            output["components"] = collect_components(loaded, model, stage, rows)
        allocated, reserved = vram_gb()
        output["vram_before_release_gb"] = {
            "allocated": clean(allocated),
            "reserved": clean(reserved),
        }
        return output
    finally:
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("open", "sealed"), default="open")
    parser.add_argument(
        "--mode", choices=("behavior", "physical", "component", "all"), default="all"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.model, args.stage, args.mode),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
