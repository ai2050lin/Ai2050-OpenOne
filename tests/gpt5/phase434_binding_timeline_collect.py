#!/usr/bin/env python3
"""Collect Phase434 natural behavior and label-blind state sketches."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
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
from phase429_typed_route_protocol import render_chat  # noqa: E402
from phase433_shared_prefix_collect import (  # noqa: E402
    common_prefix_length,
    contextual_continuation_ids,
    digest_ids,
)
from phase434_binding_timeline_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    LANGUAGE_MODEL,
    MAPPINGS,
    MODELS,
    NEUTRAL_CUE,
    OUT,
    PHYSICAL_SPLIT,
    PHASE_ID as PROTOCOL_PHASE_ID,
    RECORD_ORDERS,
    ROLES,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    STRESS_SPLIT,
    TIMINGS,
    TRACE_SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE_ID = "Phase434-BindingTimelineCollection"
BEHAVIOR_BATCH_SIZE = {"qwen3": 8, "glm4": 6, "deepseek7b": 8}
PHYSICAL_BATCH_SIZE = {"qwen3": 20, "glm4": 5, "deepseek7b": 10}
BEHAVIOR_CHECKPOINT = 256
PHYSICAL_CHECKPOINT = 160
QUESTION_LINE = "Question: Which item is selected?"
INSTRUCTION_LINE = "Output exactly the selected item and then stop."
RECORD_FOOTER_LINE = "End of the two-item record."
POSITION_ROLES = (
    "selector_slot_end",
    "role_a_result_end",
    "role_b_result_end",
    "after_records_end",
    "question_end",
    "instruction_end",
    "assistant_boundary",
    "prompt_terminal",
    "teacher_branch_boundary",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase434 non-finite scalar: {value}")
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


def group_path(split: str) -> Path:
    if split == SEALED_SPLIT:
        return OUT / "sealed/phase434_groups_sealed.jsonl"
    return OUT / f"phase434_groups_{split}.jsonl"


def groups_for_stage(stage: str) -> list[dict[str, Any]]:
    if stage == "behavior":
        return [
            row
            for split in (*BEHAVIOR_SPLITS, STRESS_SPLIT)
            for row in read_jsonl(group_path(split))
        ]
    if stage == "physical":
        gate = read_json(OUT / "phase434_behavior_gate.json")
        if not gate.get("eligible_models"):
            raise RuntimeError("Phase434 has no behavior-qualified physical model")
        return read_jsonl(group_path(PHYSICAL_SPLIT))
    if stage == "sealed":
        gate = read_json(OUT / "phase434_open_gate.json")
        if not gate.get("sealed_unlock"):
            raise RuntimeError("Phase434 sealed denominator is not authorized")
        return read_jsonl(group_path(SEALED_SPLIT))
    raise ValueError(stage)


def role_targets(group: dict[str, Any], mapping: str) -> dict[str, str]:
    if mapping == "direct":
        return {"a": group["source_1"], "b": group["source_2"]}
    if mapping == "swapped":
        return {"a": group["source_2"], "b": group["source_1"]}
    raise ValueError(mapping)


def selector_slots(
    timing: str, role: str, cue_aliases: dict[str, str]
) -> dict[str, str]:
    slots = {name: NEUTRAL_CUE for name in TIMINGS}
    slots[timing] = cue_aliases[role]
    return slots


def stress_slots(
    mode: str, role: str, cue_aliases: dict[str, str]
) -> dict[str, str]:
    slots = {name: NEUTRAL_CUE for name in TIMINGS}
    if mode == "conflict_slots":
        opposite = "b" if role == "a" else "a"
        slots["before_records"] = cue_aliases[role]
        slots["near_query"] = cue_aliases[opposite]
    elif mode != "neutral_only":
        raise ValueError(mode)
    return slots


def materialize_condition(
    group: dict[str, Any],
    role: str,
    timing: str,
    record_order: str,
    mapping: str,
    loaded: Any,
    *,
    stress_mode: str | None = None,
) -> dict[str, Any]:
    aliases = group["role_aliases"]
    cues = group["cue_aliases"]
    targets = role_targets(group, mapping)
    effective_cues = cues if group["candidate"] else {"a": "cue-ok", "b": "cue-ok"}
    if stress_mode is None:
        slots = selector_slots(timing, role, effective_cues)
    else:
        slots = stress_slots(stress_mode, role, effective_cues)

    if group["candidate"]:
        target = targets[role]
        opposite = group["source_2"] if target == group["source_1"] else group["source_1"]
        record_targets = targets
        task = (
            f"{cues['a']} selects the result recorded for {aliases['a']}; "
            f"{cues['b']} selects the result recorded for {aliases['b']}; "
            f"{NEUTRAL_CUE} carries no selection."
        )
    else:
        target_source = group["control_target_source"]
        target = group[target_source]
        opposite = group["source_2"] if target_source == "source_1" else group["source_1"]
        record_targets = {"a": target, "b": opposite}
        task = (
            f"{effective_cues['a']} selects the stable item; "
            f"{NEUTRAL_CUE} also leaves the stable item unchanged. Never select the decoy."
        )

    order = ("a", "b") if record_order == "ab" else ("b", "a")
    record_lines = {
        semantic_role: (
            f"Record item for {aliases[semantic_role]}: "
            f"{'result' if group['candidate'] else 'stable item' if semantic_role == 'a' else 'decoy item'} "
            f"{record_targets[semantic_role]}."
        )
        for semantic_role in ROLES
    }
    slot_lines = {
        "before_records": f"Selector slot before records: {slots['before_records']}.",
        "after_role_a": f"Selector slot after {aliases['a']}: {slots['after_role_a']}.",
        "after_role_b": f"Selector slot after {aliases['b']}: {slots['after_role_b']}.",
        "after_records": f"Selector slot after records: {slots['after_records']}.",
        "near_query": f"Selector slot near query: {slots['near_query']}.",
    }
    content_lines = [
        "Follow this deterministic two-item relation task.",
        task,
        slot_lines["before_records"],
    ]
    for semantic_role in order:
        content_lines.append(record_lines[semantic_role])
        content_lines.append(slot_lines[f"after_role_{semantic_role}"])
    content_lines.extend(
        (
            RECORD_FOOTER_LINE,
            slot_lines["after_records"],
            QUESTION_LINE,
            slot_lines["near_query"],
            INSTRUCTION_LINE,
        )
    )
    content = "\n".join(content_lines)
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
    target_ids = source_1_ids if target == group["source_1"] else source_2_ids
    opposite_ids = source_2_ids if target == group["source_1"] else source_1_ids
    suffix_1 = source_1_ids[common_length:]
    suffix_2 = source_2_ids[common_length:]
    mismatch_count = sum(left != right for left, right in zip(suffix_1, suffix_2))
    mismatch_count += abs(len(suffix_1) - len(suffix_2))
    if (
        common_length < 2
        or len(suffix_1) < 2
        or len(suffix_2) < 2
        or mismatch_count < 2
    ):
        raise RuntimeError(
            f"Invalid Phase434 event token contract: {group['semantic_group_id']} "
            f"{len(source_1_ids)=} {len(source_2_ids)=} {common_length=} "
            f"{mismatch_count=}"
        )
    target_source = "source_1" if target == group["source_1"] else "source_2"
    route = stress_mode or timing
    condition_id = (
        f"{group['semantic_group_id']}__ord_{record_order}__map_{mapping}__"
        f"r{role}__t_{route}__{loaded.key}"
    )
    active_count = sum(value != NEUTRAL_CUE for value in slots.values())
    return {
        **{
            key: value
            for key, value in group.items()
            if key not in {"role_aliases", "cue_aliases"}
        },
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": loaded.key,
        "condition_id": condition_id,
        "interface": "direct_item",
        "role": role,
        "route_mode": route,
        "timing": timing,
        "record_order": record_order,
        "mapping": mapping,
        "source_role": role if timing != "near_query" else "none",
        "query_role": role,
        "role_aliases": aliases,
        "cue_aliases": effective_cues,
        "selector_slots": slots,
        "active_selector_count": active_count,
        "content_prompt": content,
        "rendered_prompt": rendered,
        "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": digest_ids(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "semantic_target": target,
        "semantic_opposite": opposite,
        "semantic_target_source": target_source,
        "target": target,
        "opposite_target": opposite,
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
        "post_divergence_mismatch_count": mismatch_count,
        "same_first_token": source_1_ids[0] == source_2_ids[0],
        "same_surface_length": len(group["source_1"]) == len(group["source_2"]),
        "contextual_tokenization_exact": bool(source_1_context_exact and source_2_context_exact),
        "natural_generation_max_new_tokens": 24,
        "normative_target": stress_mode is None,
        "descriptive_none_only": stress_mode == "neutral_only",
        "descriptive_conflict_only": stress_mode == "conflict_slots",
        "question_line": QUESTION_LINE,
        "instruction_line": INSTRUCTION_LINE,
        "record_footer_line": RECORD_FOOTER_LINE,
        "record_lines": record_lines,
        "record_targets": record_targets,
        "slot_lines": slot_lines,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
    }


def materialize_groups(loaded: Any, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups_for_stage(stage):
        if group["stress_only"]:
            for role in ROLES:
                for mode in ("conflict_slots", "neutral_only"):
                    rows.append(
                        materialize_condition(
                            group,
                            role,
                            "before_records",
                            group["baseline_record_order"],
                            group["baseline_mapping"],
                            loaded,
                            stress_mode=mode,
                        )
                    )
            continue
        if group["candidate"]:
            variants = (
                (order, mapping)
                for order in RECORD_ORDERS
                for mapping in MAPPINGS
            )
        else:
            variants = ((group["baseline_record_order"], group["baseline_mapping"]),)
        for order, mapping in variants:
            role_order = ROLES if group["query_execution_order"] == "ab" else tuple(reversed(ROLES))
            for role in role_order:
                for timing in TIMINGS:
                    rows.append(
                        materialize_condition(group, role, timing, order, mapping, loaded)
                    )
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
    retained = {
        key: row[key]
        for key in (
            "timing",
            "record_order",
            "mapping",
            "semantic_target_source",
            "stress_only",
            "role_alias_index",
            "cue_alias_index",
            "factor_cell",
            "replicate_index",
            "active_selector_count",
            "common_prefix_token_count",
            "post_divergence_mismatch_count",
            "same_first_token",
            "same_surface_length",
            "contextual_tokenization_exact",
        )
    }
    return {
        **source,
        **retained,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
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


def stage_root(stage: str, model: str) -> Path:
    return OUT / stage / model


def collect_behavior(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = stage_root(stage, model) / "behavior"
    complete_path = root / "phase434_behavior_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase434_behavior_part_*.jsonl"))
    existing = [row for path in existing_paths for row in read_jsonl(path)]
    completed = {row["condition_id"] for row in existing}
    pending = [row for row in rows if row["condition_id"] not in completed]
    part = len(existing_paths)
    buffer: list[dict[str, Any]] = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase434 behavior] {stage} {model}; conditions={len(rows)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), BEHAVIOR_BATCH_SIZE[model]):
        batch = pending[start : start + BEHAVIOR_BATCH_SIZE[model]]
        raw = p431.collect_behavior_batch(loaded, batch)
        buffer.extend(enrich_behavior(item, row) for item, row in zip(raw, batch))
        processed += len(batch)
        if processed % BEHAVIOR_CHECKPOINT < len(batch) or processed == len(rows):
            write_jsonl(
                checkpoint_root / f"phase434_behavior_part_{part:05d}.jsonl",
                buffer,
            )
            buffer.clear()
            part += 1
        if processed == len(batch) or processed % 512 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase434 behavior] {stage} {model} {processed}/{len(rows)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    if buffer:
        write_jsonl(
            checkpoint_root / f"phase434_behavior_part_{part:05d}.jsonl", buffer
        )
    collected = [
        row
        for path in sorted(checkpoint_root.glob("phase434_behavior_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    unique = {row["condition_id"]: row for row in collected}
    final_rows = [unique[key] for key in sorted(unique)]
    if len(final_rows) != len(rows):
        raise RuntimeError(f"Incomplete Phase434 behavior: {len(final_rows)} != {len(rows)}")
    write_jsonl(root / "phase434_materialized_conditions.jsonl", rows)
    write_jsonl(root / "phase434_behavior_rows.jsonl", final_rows)
    valid = sum(
        row["same_first_token"]
        and row["same_surface_length"]
        and row["contextual_tokenization_exact"]
        and row["common_prefix_token_count"] >= 2
        and row["post_divergence_mismatch_count"] >= 2
        for row in final_rows
    )
    complete = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "condition_count": len(rows),
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in final_rows)),
        "token_contract_valid_count": valid,
        "common_prefix_length_range": [
            min(row["common_prefix_token_count"] for row in rows),
            max(row["common_prefix_token_count"] for row in rows),
        ],
        "elapsed_seconds": clean(time.monotonic() - started),
        "all_rows_complete": len(final_rows) == len(rows),
        "sealed_read": stage == "sealed",
    }
    write_json(complete_path, complete)
    return complete


def component_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(type(value).__name__)


def register_positions(fast_tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = row["rendered_prompt"]
    encoded = fast_tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if digest_ids(ids) != row["prompt_token_ids_sha256"]:
        raise RuntimeError(f"Fast tokenizer disagreement: {row['condition_id']}")

    def span(value: str, start: int = 0) -> list[int]:
        return p431.token_positions(rendered, offsets, value, start)

    record_a_start = rendered.find(row["record_lines"]["a"])
    record_b_start = rendered.find(row["record_lines"]["b"])
    source_a = span(row["record_targets"]["a"], record_a_start)
    source_b = span(row["record_targets"]["b"], record_b_start)
    active_timing = row["timing"] if row["active_selector_count"] == 1 else "near_query"
    active_line = row["slot_lines"][active_timing]
    active_start = rendered.find(active_line)
    active_slot = span(active_line, active_start)
    after_records_line = row["slot_lines"]["after_records"]
    after_records = span(after_records_line, rendered.find(after_records_line))
    question = span(row["question_line"], rendered.find(row["question_line"]))
    instruction = span(row["instruction_line"], rendered.find(row["instruction_line"]))
    content_start = rendered.find(row["content_prompt"])
    content_end = content_start + len(row["content_prompt"])
    boundary = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left >= content_end
    ]
    prompt_terminal = len(ids) - 1
    common_count = int(row["common_prefix_token_count"])
    return {
        "input_ids": ids,
        "position_roles": {
            "selector_slot_end": active_slot[-1],
            "role_a_result_end": source_a[-1],
            "role_b_result_end": source_b[-1],
            "after_records_end": after_records[-1],
            "question_end": question[-1],
            "instruction_end": instruction[-1],
            "assistant_boundary": boundary[-1] if boundary else prompt_terminal,
            "prompt_terminal": prompt_terminal,
            "teacher_branch_boundary": prompt_terminal + common_count,
        },
    }


def deterministic_projection(hidden_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4340715 + hidden_size)
    matrix = torch.randn(hidden_size, 16, generator=generator, dtype=torch.float32)
    matrix /= math.sqrt(hidden_size)
    return matrix.to(device)


@torch.inference_mode()
def collect_physical_batch(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    projection: torch.Tensor,
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
    hook_layer = round(0.75 * (len(layers) - 1))
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _args: Any, output: Any) -> None:
        captured["state"] = component_tensor(output).detach()

    handle = layers[hook_layer].register_forward_hook(hook)
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
        handle.remove()
    hidden_states = result.hidden_states
    if hidden_states is None or len(hidden_states) != len(layers) + 1:
        raise RuntimeError("Unexpected Phase434 hidden-state ledger")
    hook_error = (
        captured["state"].float() - hidden_states[hook_layer + 1].float()
    ).abs().max().item()
    batch_axis = torch.arange(len(rows), device=loaded.input_device)
    output_rows: list[dict[str, Any]] = []
    for layer_index in range(len(layers)):
        states = hidden_states[layer_index + 1]
        position_payload: dict[str, tuple[list[list[float]], list[float]]] = {}
        for role_name in POSITION_ROLES:
            positions = torch.tensor(
                [
                    pads[index] + registered[index]["position_roles"][role_name]
                    for index in range(len(rows))
                ],
                dtype=torch.long,
                device=loaded.input_device,
            )
            state = states[batch_axis, positions].float()
            sketches = (state @ projection).cpu().tolist()
            norms = state.norm(dim=-1).cpu().tolist()
            position_payload[role_name] = (sketches, norms)
        for index, row in enumerate(rows):
            output_rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": loaded.key,
                    "stage": "sealed" if row["split"] == SEALED_SPLIT else "physical",
                    "split": row["split"],
                    "condition_id": row["condition_id"],
                    "semantic_group_id": row["semantic_group_id"],
                    "paired_group_id": row["paired_group_id"],
                    "group_index": row["group_index"],
                    "role_alias_index": row["role_alias_index"],
                    "cue_alias_index": row["cue_alias_index"],
                    "candidate": row["candidate"],
                    "block_id": row["block_id"],
                    "role": row["role"],
                    "timing": row["timing"],
                    "record_order": row["record_order"],
                    "mapping": row["mapping"],
                    "semantic_target_source": row["semantic_target_source"],
                    "actual_choice": row["actual_choice"],
                    "natural_complete_event_correct": row["natural_complete_event_correct"],
                    "layer": layer_index,
                    "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                    "position_metrics": {
                        role_name: {
                            "state_sketch": [clean(value) for value in values[0][index]],
                            "state_norm": clean(values[1][index]),
                            "absolute_token_index": int(registered[index]["position_roles"][role_name]),
                            "output_label_blind": True,
                            "time_status": (
                                "teacher_forced_pre_divergence"
                                if role_name == "teacher_branch_boundary"
                                else "natural_prompt_pre_generation"
                            ),
                        }
                        for role_name, values in position_payload.items()
                    },
                    "physical": True,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "single_neuron": False,
                }
            )
    return output_rows, {
        "hook_layer": hook_layer,
        "hook_hidden_state_max_abs_error": clean(hook_error),
        "identity_total": len(rows),
    }


def physical_input_rows(
    materialized: list[dict[str, Any]], behavior: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    behavior_map = {row["condition_id"]: row for row in behavior}
    return [
        {
            **row,
            "actual_choice": behavior_map[row["condition_id"]]["actual_choice"],
            "natural_complete_event_correct": behavior_map[row["condition_id"]]["natural_complete_event_correct"],
        }
        for row in materialized
        if not row["stress_only"]
    ]


def collect_physical(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = stage_root(stage, model) / "physical"
    complete_path = root / "phase434_physical_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    behavior = read_jsonl(stage_root(stage, model) / "behavior/phase434_behavior_rows.jsonl")
    selected = physical_input_rows(rows, behavior)
    fast_tokenizer = AutoTokenizer.from_pretrained(
        str(loaded.spec.local_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    layers = get_layers(loaded.model)
    hidden_size = int(loaded.model.config.hidden_size)
    projection = deterministic_projection(hidden_size, loaded.input_device)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase434_physical_part_*.jsonl.gz"))
    existing = [row for path in existing_paths for row in read_jsonl_any(path)]
    counts = Counter(row["condition_id"] for row in existing)
    completed = {key for key, value in counts.items() if value == len(layers)}
    pending = [row for row in selected if row["condition_id"] not in completed]
    part = len(existing_paths)
    diagnostics: list[dict[str, Any]] = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase434 physical] {stage} {model}; conditions={len(selected)}; "
        f"pending={len(pending)}; layers={len(layers)}",
        flush=True,
    )
    for start in range(0, len(pending), PHYSICAL_BATCH_SIZE[model]):
        batch = pending[start : start + PHYSICAL_BATCH_SIZE[model]]
        traced, diagnostic = collect_physical_batch(
            loaded, fast_tokenizer, layers, batch, projection
        )
        diagnostics.append(diagnostic)
        write_jsonl_gz(
            checkpoint_root / f"phase434_physical_part_{part:05d}.jsonl.gz",
            traced,
        )
        part += 1
        processed += len(batch)
        if processed == len(batch) or processed % PHYSICAL_CHECKPOINT < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase434 physical] {stage} {model} {processed}/{len(selected)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    final_rows = [
        row
        for path in sorted(checkpoint_root.glob("phase434_physical_part_*.jsonl.gz"))
        for row in read_jsonl_any(path)
    ]
    unique = {(row["condition_id"], row["layer"]): row for row in final_rows}
    ordered = [unique[key] for key in sorted(unique)]
    expected = len(selected) * len(layers)
    if len(ordered) != expected:
        raise RuntimeError(f"Incomplete Phase434 physical: {len(ordered)} != {expected}")
    write_jsonl_gz(root / "phase434_physical_rows.jsonl.gz", ordered)
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
        "finite_sketch_count": sum(
            all(
                math.isfinite(value)
                for metric in row["position_metrics"].values()
                for value in [*metric["state_sketch"], metric["state_norm"]]
            )
            for row in ordered
        ),
        "all_rows_complete": len(ordered) == expected,
        "elapsed_seconds": clean(time.monotonic() - started),
        "sealed_read": stage == "sealed",
    }
    write_json(complete_path, complete)
    return complete


def collect(model: str, stage: str, mode: str) -> dict[str, Any]:
    protocol = freeze()
    if model not in MODELS:
        raise ValueError(model)
    if stage == "physical":
        gate = read_json(OUT / "phase434_behavior_gate.json")
        if model not in gate.get("eligible_models", []):
            return {
                "model": model,
                "stage": stage,
                "skipped": True,
                "reason": "model_failed_natural_behavior_qualification",
            }
    if stage == "sealed" and model != LANGUAGE_MODEL:
        raise RuntimeError("Phase434 sealed execution is frozen to Qwen3")
    loaded = None
    try:
        loaded = load_probe_model(model)
        rows = materialize_groups(loaded, stage)
        expected = protocol["denominator_audit"]["conditions_by_split_per_model"]
        expected_count = (
            expected["behavior_discovery"] + expected["behavior_holdout"] + expected[STRESS_SPLIT]
            if stage == "behavior"
            else expected[PHYSICAL_SPLIT]
            if stage == "physical"
            else expected[SEALED_SPLIT]
        )
        if len(rows) != expected_count:
            raise RuntimeError(f"Phase434 materialized denominator {len(rows)} != {expected_count}")
        output: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "model": model,
            "stage": stage,
            "condition_count": len(rows),
            "behavior": None,
            "physical": None,
        }
        if mode in {"behavior", "all"}:
            output["behavior"] = collect_behavior(loaded, model, stage, rows)
        if mode in {"physical", "all"}:
            behavior_path = stage_root(stage, model) / "behavior/phase434_behavior_complete.json"
            if not behavior_path.exists():
                output["behavior"] = collect_behavior(loaded, model, stage, rows)
            output["physical"] = collect_physical(loaded, model, stage, rows)
        return output
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
    parser.add_argument("--stage", choices=("behavior", "physical", "sealed"), required=True)
    parser.add_argument("--mode", choices=("behavior", "physical", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(collect(args.model, args.stage, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
