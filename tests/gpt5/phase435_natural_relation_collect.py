#!/usr/bin/env python3
"""Collect Phase435 interface, behavior, and compact physical ledgers."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import shutil
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
from phase433_shared_prefix_collect import contextual_continuation_ids, digest_ids  # noqa: E402
from phase433_shared_prefix_collect import common_prefix_length  # noqa: E402
from phase435_natural_relation_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    CONTRACTS,
    DTYPES,
    FAMILY_CONFIG,
    GENERIC_CONTROL,
    INTERFACE_SIMPLICITY,
    INTERFACE_SPLIT,
    INTERFACES,
    MAPPINGS,
    MODELS,
    OUT,
    PHYSICAL_SPLIT,
    QUERY_ROLES,
    RECORD_ORDERS,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    TRACE_SCHEMA_VERSION,
    freeze,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE_ID = "Phase435-NaturalRelationCollection"
BEHAVIOR_BATCH_SIZE = {"qwen3": 8, "glm4": 6, "deepseek7b": 8}
PHYSICAL_BATCH_SIZE = {"qwen3": 6, "glm4": 2, "deepseek7b": 4}
BEHAVIOR_CHECKPOINT = 256
PHYSICAL_CHECKPOINT = 48
GENERATION_HORIZON = 16
PHYSICAL_GENERATION_STEPS = 8
POSITION_ROLES = (
    "first_record_object_end",
    "first_record_relation_end",
    "first_record_value_end",
    "second_record_object_end",
    "second_record_relation_end",
    "second_record_value_end",
    "query_object_end",
    "query_relation_end",
    "question_end",
    "instruction_end",
    "assistant_boundary",
    "prompt_terminal",
    "first_answer_token",
    "complete_answer_boundary",
)
RECEIVER_ROLES = ("question_end", "prompt_terminal", "complete_answer_boundary")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase435 non-finite scalar: {value}")
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


def group_path(split: str) -> Path:
    if split == SEALED_SPLIT:
        return OUT / "sealed/phase435_groups_sealed.jsonl"
    return OUT / f"phase435_groups_{split}.jsonl"


def selected_interface(model: str) -> str:
    path = OUT / "phase435_interface_freeze.json"
    if not path.exists():
        raise RuntimeError("Phase435 interface freeze is missing")
    return str(read_json(path)["models"][model]["selected_interface"])


def interface_qualified(model: str) -> bool:
    path = OUT / "phase435_interface_freeze.json"
    return bool(
        path.exists()
        and read_json(path)["models"][model]["calibration_qualified"]
    )


def eligible_contracts(model: str) -> list[str]:
    path = OUT / "phase435_behavior_gate.json"
    if not path.exists():
        return []
    gate = read_json(path)
    return sorted(
        row["contract"]
        for row in gate.get("eligible_model_contracts", [])
        if row["model"] == model
    )


def authorized_sealed_contracts(model: str) -> list[str]:
    path = OUT / "phase435_open_gate.json"
    if not path.exists() or not read_json(path).get("sealed_unlock"):
        raise RuntimeError("Phase435 sealed denominator is not authorized")
    return sorted(
        row["contract"]
        for row in read_json(path).get("sealed_authorized_model_contracts", [])
        if row["model"] == model
    )


def mapping_values(group: dict[str, Any], mapping: str) -> dict[str, str]:
    if mapping == "direct":
        return {"a": group["value_1"], "b": group["value_2"]}
    if mapping == "swapped":
        return {"a": group["value_2"], "b": group["value_1"]}
    raise ValueError(mapping)


def value_source(group: dict[str, Any], value: str) -> str:
    if value == group["value_1"]:
        return "source_1"
    if value == group["value_2"]:
        return "source_2"
    raise ValueError(value)


def interface_payload(interface: str, target_value: str, opposite_value: str) -> dict[str, Any]:
    if interface == "direct_value":
        return {
            "target": target_value,
            "opposite": opposite_value,
            "instruction": "Reply with only the requested value and no added punctuation.",
        }
    if interface == "answer_field":
        return {
            "target": f"Answer: {target_value}",
            "opposite": f"Answer: {opposite_value}",
            "instruction": "Reply exactly as 'Answer: VALUE', replacing VALUE with the requested value.",
        }
    if interface == "result_field":
        target = json.dumps({"result": target_value}, ensure_ascii=True, separators=(",", ":"))
        opposite = json.dumps({"result": opposite_value}, ensure_ascii=True, separators=(",", ":"))
        return {
            "target": target,
            "opposite": opposite,
            "instruction": 'Reply with exactly {"result":"VALUE"}, replacing VALUE with the requested value.',
        }
    if interface == "natural_sentence":
        return {
            "target": f"The requested value is {target_value}.",
            "opposite": f"The requested value is {opposite_value}.",
            "instruction": "Reply exactly as 'The requested value is VALUE.'",
        }
    raise ValueError(interface)


def contract_surface(
    group: dict[str, Any], contract: str, role_values: dict[str, str], record_order: str, query_role: str
) -> dict[str, Any]:
    config = FAMILY_CONFIG[group["relation_family"]]
    entities = {"a": group["entity_a"], "b": group["entity_b"]}
    order = ("a", "b") if record_order == "ab" else ("b", "a")
    entries = []
    for role in order:
        entity = entities[role]
        value = role_values[role]
        if contract == "field_extract":
            relation_surface = group["value_label"]
            line = f"{group['entity_label']}: {entity}; {relation_surface}: {value}."
        elif contract == "natural_qa":
            relation_surface = str(config["statement_relation_surface"])
            line = str(config["statement"]).format(entity=entity, value=value)
        elif contract == "relation_rewrite":
            relation_surface = group["relation_label"]
            line = f"The registry links {entity} with {value} through the {relation_surface} relation."
        elif contract == GENERIC_CONTROL:
            relation_surface = "pairs"
            line = f"The registry pairs {entity} with the value {value}."
        else:
            raise ValueError(contract)
        entries.append(
            {
                "semantic_role": role,
                "entity": entity,
                "value": value,
                "value_source": value_source(group, value),
                "relation_surface": relation_surface,
                "line": line,
            }
        )
    query_entity = entities[query_role]
    if contract == "field_extract":
        query_relation_surface = group["value_label"]
        question = f"For {query_entity}, what is the {query_relation_surface}?"
    elif contract == "natural_qa":
        query_relation_surface = str(config["question_relation_surface"])
        question = str(config["question"]).format(entity=query_entity)
    elif contract == "relation_rewrite":
        query_relation_surface = group["relation_label"]
        question = f"Give the value linked to {query_entity} by the {query_relation_surface} relation."
    else:
        query_relation_surface = "paired with"
        question = f"Which value is paired with {query_entity}?"
    return {
        "record_entries": entries,
        "question_line": f"Question: {question}",
        "query_entity": query_entity,
        "query_relation_surface": query_relation_surface,
    }


def materialize_condition(
    group: dict[str, Any],
    contract: str,
    interface: str,
    record_order: str,
    mapping: str,
    query_role: str,
    loaded: Any,
) -> dict[str, Any]:
    role_values = mapping_values(group, mapping)
    target_value = role_values[query_role]
    opposite_role = "b" if query_role == "a" else "a"
    opposite_value = role_values[opposite_role]
    surface = contract_surface(group, contract, role_values, record_order, query_role)
    output = interface_payload(interface, target_value, opposite_value)
    content = "\n".join(
        (
            "Use only the two registry records below. Treat them as the complete context.",
            *[entry["line"] for entry in surface["record_entries"]],
            surface["question_line"],
            str(output["instruction"]),
        )
    )
    rendered = render_chat(loaded.tokenizer, loaded.key, content)
    prompt_ids = [
        int(value)
        for value in loaded.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]
    target_ids, target_prefix_exact = contextual_continuation_ids(
        loaded.tokenizer, rendered, prompt_ids, str(output["target"])
    )
    opposite_ids, opposite_prefix_exact = contextual_continuation_ids(
        loaded.tokenizer, rendered, prompt_ids, str(output["opposite"])
    )
    common_length = common_prefix_length(target_ids, opposite_ids)
    if common_length >= len(target_ids) or common_length >= len(opposite_ids):
        raise RuntimeError(
            f"Target continuations do not branch for {group['semantic_group_id']}"
        )
    value_1_ids = [
        int(value)
        for value in loaded.tokenizer(group["value_1"], add_special_tokens=False)["input_ids"]
    ]
    value_2_ids = [
        int(value)
        for value in loaded.tokenizer(group["value_2"], add_special_tokens=False)["input_ids"]
    ]
    first_role = "a" if record_order == "ab" else "b"
    target_position = "first" if query_role == first_role else "second"
    condition_id = (
        f"{group['semantic_group_id']}__c_{contract}__i_{interface}__o_{record_order}"
        f"__m_{mapping}__q_{query_role}__{loaded.key}"
    )
    return {
        **group,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": loaded.key,
        "condition_id": condition_id,
        "contract": contract,
        "condition_kind": "generic_control" if contract == GENERIC_CONTROL else "candidate",
        "interface": interface,
        "record_order": record_order,
        "mapping": mapping,
        "query_role": query_role,
        "target_position": target_position,
        "role_values": role_values,
        "target_value": target_value,
        "opposite_value": opposite_value,
        "semantic_target_source": value_source(group, target_value),
        "semantic_opposite_source": value_source(group, opposite_value),
        "source_1": group["value_1"],
        "source_2": group["value_2"],
        "semantic_target": target_value,
        "semantic_opposite": opposite_value,
        "target": str(output["target"]),
        "opposite_target": str(output["opposite"]),
        "target_sequence_token_ids": target_ids,
        "opposite_sequence_token_ids": opposite_ids,
        "output_common_prefix_token_count": common_length,
        "target_branch_token_id": target_ids[common_length],
        "opposite_branch_token_id": opposite_ids[common_length],
        "target_contextual_tokenization_exact": target_prefix_exact,
        "opposite_contextual_tokenization_exact": opposite_prefix_exact,
        "source_1_first_token_id": value_1_ids[0],
        "source_2_first_token_id": value_2_ids[0],
        "natural_generation_max_new_tokens": GENERATION_HORIZON,
        "content_prompt": content,
        "rendered_prompt": rendered,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": digest_ids(prompt_ids),
        "record_entries": surface["record_entries"],
        "question_line": surface["question_line"],
        "query_entity": surface["query_entity"],
        "query_relation_surface": surface["query_relation_surface"],
        "instruction_line": str(output["instruction"]),
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
    }


def materialize_groups(loaded: Any, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage == "interface":
        for group in read_jsonl(group_path(INTERFACE_SPLIT)):
            contract = group["contract_variants"][0]
            for interface in INTERFACES:
                rows.append(
                    materialize_condition(
                        group,
                        contract,
                        interface,
                        group["baseline_record_order"],
                        group["baseline_mapping"],
                        group["baseline_query_role"],
                        loaded,
                    )
                )
    elif stage == "behavior":
        interface = selected_interface(loaded.key)
        for split in BEHAVIOR_SPLITS:
            for group in read_jsonl(group_path(split)):
                for contract in CONTRACTS:
                    for order in RECORD_ORDERS:
                        for mapping in MAPPINGS:
                            for query_role in QUERY_ROLES:
                                rows.append(
                                    materialize_condition(
                                        group, contract, interface, order, mapping, query_role, loaded
                                    )
                                )
    elif stage in {"physical", "sealed"}:
        contracts = (
            eligible_contracts(loaded.key)
            if stage == "physical"
            else authorized_sealed_contracts(loaded.key)
        )
        split = PHYSICAL_SPLIT if stage == "physical" else SEALED_SPLIT
        interface = selected_interface(loaded.key)
        for group in read_jsonl(group_path(split)):
            for contract in (*contracts, GENERIC_CONTROL):
                for order in RECORD_ORDERS:
                    for mapping in MAPPINGS:
                        for query_role in QUERY_ROLES:
                            rows.append(
                                materialize_condition(
                                    group, contract, interface, order, mapping, query_role, loaded
                                )
                            )
    else:
        raise ValueError(stage)
    rows.sort(key=lambda row: row["condition_id"])
    return rows


def enrich_behavior(source: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    retained = {
        key: row[key]
        for key in (
            "contract",
            "condition_kind",
            "interface",
            "record_order",
            "mapping",
            "query_role",
            "target_position",
            "relation_family",
            "semantic_target_source",
            "semantic_opposite_source",
            "target_contextual_tokenization_exact",
            "opposite_contextual_tokenization_exact",
            "physical_fold",
            "pipeline_sealed",
        )
    }
    content_good = bool(
        source["teacher_sequence_correct"]
        and source["actual_choice"] == row["semantic_target_source"]
        and source["natural_target_first"]
        and not source["natural_opposite_first"]
        and source["natural_interface_valid"]
        and source["natural_exact_target_contract"]
        and not source["natural_revision"]
    )
    return {
        **source,
        **retained,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "natural_first_answer_good": bool(
            source["natural_target_first"] and not source["natural_opposite_first"]
        ),
        "natural_complete_answer_good": bool(source["natural_exact_target_contract"]),
        "natural_content_good": content_good,
        "natural_stop_good": bool(source["natural_stop"] and not source["natural_censoring"]),
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
    }


def stage_root(stage: str, model: str) -> Path:
    return OUT / stage / model


def collect_behavior(loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = stage_root(stage, model) / "behavior"
    complete_path = root / "phase435_behavior_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase435_behavior_part_*.jsonl"))
    existing = [row for path in existing_paths for row in read_jsonl(path)]
    completed = {row["condition_id"] for row in existing}
    pending = [row for row in rows if row["condition_id"] not in completed]
    part = len(existing_paths)
    buffer: list[dict[str, Any]] = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase435 behavior] {stage} {model}; conditions={len(rows)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), BEHAVIOR_BATCH_SIZE[model]):
        batch = pending[start : start + BEHAVIOR_BATCH_SIZE[model]]
        raw = p431.collect_behavior_batch(loaded, batch)
        buffer.extend(enrich_behavior(item, row) for item, row in zip(raw, batch))
        processed += len(batch)
        if processed % BEHAVIOR_CHECKPOINT < len(batch) or processed == len(rows):
            write_jsonl(checkpoint_root / f"phase435_behavior_part_{part:05d}.jsonl", buffer)
            buffer.clear()
            part += 1
        if processed == len(batch) or processed % 512 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase435 behavior] {stage} {model} {processed}/{len(rows)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    if buffer:
        write_jsonl(checkpoint_root / f"phase435_behavior_part_{part:05d}.jsonl", buffer)
    collected = [
        row
        for path in sorted(checkpoint_root.glob("phase435_behavior_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    unique = {row["condition_id"]: row for row in collected}
    final_rows = [unique[key] for key in sorted(unique)]
    if len(final_rows) != len(rows):
        raise RuntimeError(f"Incomplete Phase435 behavior: {len(final_rows)} != {len(rows)}")
    write_jsonl(root / "phase435_materialized_conditions.jsonl", rows)
    write_jsonl(root / "phase435_behavior_rows.jsonl", final_rows)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "condition_count": len(rows),
        "unique_condition_id_count": len(unique),
        "content_good_count": sum(row["natural_content_good"] for row in final_rows),
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in final_rows)),
        "all_rows_complete": len(final_rows) == len(rows),
        "sealed_read": stage == "sealed",
        "elapsed_seconds": clean(time.monotonic() - started),
    }
    write_json(complete_path, complete)
    return complete


def token_positions(
    rendered: str, offsets: list[tuple[int, int]], value: str, start_at: int = 0
) -> list[int]:
    left = rendered.find(value, start_at)
    if left < 0:
        raise RuntimeError(f"Registered span not found: {value!r}")
    right = left + len(value)
    return [
        index
        for index, (token_left, token_right) in enumerate(offsets)
        if token_right > token_left and token_left < right and token_right > left
    ]


def register_positions(fast_tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = row["rendered_prompt"]
    encoded = fast_tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if digest_ids(ids) != row["prompt_token_ids_sha256"]:
        raise RuntimeError(f"Fast tokenizer disagreement: {row['condition_id']}")
    entry_payload = []
    for entry in row["record_entries"]:
        start = rendered.find(entry["line"])
        line_positions = token_positions(rendered, offsets, entry["line"], start)
        entry_payload.append(
            {
                **entry,
                "line_positions": line_positions,
                "entity_positions": token_positions(rendered, offsets, entry["entity"], start),
                "relation_positions": token_positions(
                    rendered, offsets, entry["relation_surface"], start
                ),
                "value_positions": token_positions(rendered, offsets, entry["value"], start),
            }
        )
    question_start = rendered.find(row["question_line"])
    instruction_start = rendered.find(row["instruction_line"])
    question_positions = token_positions(rendered, offsets, row["question_line"], question_start)
    instruction_positions = token_positions(
        rendered, offsets, row["instruction_line"], instruction_start
    )
    query_entity = token_positions(rendered, offsets, row["query_entity"], question_start)
    query_relation = token_positions(
        rendered, offsets, row["query_relation_surface"], question_start
    )
    first, second = entry_payload
    content_start = rendered.find(row["content_prompt"])
    content_end = content_start + len(row["content_prompt"])
    boundary = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left >= content_end
    ]
    source_partitions: dict[str, list[int]] = {
        "source_1_record": [],
        "source_2_record": [],
        "question": question_positions,
        "instruction": instruction_positions,
    }
    for entry in entry_payload:
        source_partitions[f"{entry['value_source']}_record"] = entry["line_positions"]
    return {
        "input_ids": ids,
        "position_roles": {
            "first_record_object_end": first["entity_positions"][-1],
            "first_record_relation_end": first["relation_positions"][-1],
            "first_record_value_end": first["value_positions"][-1],
            "second_record_object_end": second["entity_positions"][-1],
            "second_record_relation_end": second["relation_positions"][-1],
            "second_record_value_end": second["value_positions"][-1],
            "query_object_end": query_entity[-1],
            "query_relation_end": query_relation[-1],
            "question_end": question_positions[-1],
            "instruction_end": instruction_positions[-1],
            "assistant_boundary": boundary[-1] if boundary else len(ids) - 1,
            "prompt_terminal": len(ids) - 1,
        },
        "source_partitions": source_partitions,
    }


def deterministic_projection(hidden_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4350715 + hidden_size)
    matrix = torch.randn(hidden_size, 8, generator=generator, dtype=torch.float32)
    matrix /= math.sqrt(hidden_size)
    return matrix.to(device)


@torch.inference_mode()
def collect_physical_batch(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    projection: torch.Tensor,
) -> tuple[list[dict[str, Any]], float]:
    registered = [register_positions(fast_tokenizer, row) for row in rows]
    prompt_ids = [p431.prompt_ids(loaded, row) for row in rows]
    generated_ids = [
        [int(value) for value in row["natural_generated_token_ids"]][
            :PHYSICAL_GENERATION_STEPS
        ]
        for row in rows
    ]
    sequences = [prompt + generated for prompt, generated in zip(prompt_ids, generated_ids)]
    input_ids, attention_mask, pads = p431.padded_batch(
        sequences, int(loaded.tokenizer.pad_token_id), loaded.input_device
    )
    position_maps: list[dict[str, int]] = []
    partition_maps: list[dict[str, list[int]]] = []
    for index, item in enumerate(registered):
        pad = pads[index]
        positions = {
            role: pad + int(position) for role, position in item["position_roles"].items()
        }
        if generated_ids[index]:
            positions["first_answer_token"] = pad + len(prompt_ids[index])
            positions["complete_answer_boundary"] = (
                pad + len(prompt_ids[index]) + len(generated_ids[index]) - 1
            )
        else:
            positions["first_answer_token"] = positions["prompt_terminal"]
            positions["complete_answer_boundary"] = positions["prompt_terminal"]
        position_maps.append(positions)
        partition_maps.append(
            {
                role: [pad + int(value) for value in values]
                for role, values in item["source_partitions"].items()
            }
        )
    captures: dict[tuple[str, int], torch.Tensor] = {}
    handles = p431.install_compact_hooks(layers, captures)
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
    output_weight = output_head.weight
    directions = torch.stack(
        [
            (
                output_weight[int(row["target_branch_token_id"])].float()
                - output_weight[int(row["opposite_branch_token_id"])].float()
            )
            * (1.0 if row["semantic_target_source"] == "source_1" else -1.0)
            for row in rows
        ]
    )
    batch_size = len(rows)
    sequence_width = input_ids.shape[1]
    sequence_axis = torch.arange(sequence_width, device=loaded.input_device)
    condition_layers: list[list[dict[str, Any]]] = [[] for _ in rows]
    max_reconstruction_error = 0.0
    for layer_index, layer in enumerate(layers):
        expected = {"pre", "value", "attention", "probabilities", "mlp", "post"}
        actual = {name for name, index in captures if index == layer_index}
        if not expected.issubset(actual):
            raise RuntimeError(f"Missing Phase435 captures at layer {layer_index}: {actual}")
        probabilities = captures[("probabilities", layer_index)].float()
        value_raw = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        attention_module = layer.self_attn
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
        bias = output_projection.bias.float() if output_projection.bias is not None else None
        layer_positions: list[dict[str, Any]] = [{} for _ in rows]
        for role in POSITION_ROLES:
            positions = torch.tensor(
                [mapping[role] for mapping in position_maps],
                dtype=torch.long,
                device=loaded.input_device,
            )
            batch_axis = torch.arange(batch_size, device=loaded.input_device)
            pre = captures[("pre", layer_index)][batch_axis, positions]
            attention = captures[("attention", layer_index)][batch_axis, positions]
            mlp = captures[("mlp", layer_index)][batch_axis, positions]
            post = captures[("post", layer_index)][batch_axis, positions]
            transition = post - pre
            reconstructed = attention + mlp
            errors = torch.linalg.vector_norm(transition.float() - reconstructed.float(), dim=-1) / torch.linalg.vector_norm(
                transition.float(), dim=-1
            ).clamp_min(1e-8)
            max_reconstruction_error = max(max_reconstruction_error, float(errors.max().item()))
            sketches = (post.float() @ projection).detach().cpu().tolist()
            metrics = torch.stack(
                [
                    torch.sqrt(torch.mean(pre.float() ** 2, dim=-1).clamp_min(1e-20)),
                    torch.sqrt(torch.mean(attention.float() ** 2, dim=-1).clamp_min(1e-20)),
                    torch.sqrt(torch.mean(mlp.float() ** 2, dim=-1).clamp_min(1e-20)),
                    torch.sqrt(torch.mean(post.float() ** 2, dim=-1).clamp_min(1e-20)),
                    errors,
                ],
                dim=-1,
            ).detach().cpu().tolist()
            absolute = positions.detach().cpu().tolist()
            for batch_index in range(batch_size):
                layer_positions[batch_index][role] = {
                    "absolute_token_index": int(absolute[batch_index] - pads[batch_index]),
                    "residual_pre_rms": clean(metrics[batch_index][0]),
                    "attention_write_rms": clean(metrics[batch_index][1]),
                    "mlp_write_rms": clean(metrics[batch_index][2]),
                    "residual_post_rms": clean(metrics[batch_index][3]),
                    "block_reconstruction_relative_error": clean(metrics[batch_index][4]),
                    "state_sketch": [clean(value) for value in sketches[batch_index]],
                    "output_label_blind": True,
                }
        layer_receivers: list[dict[str, Any]] = [{} for _ in rows]
        for receiver_role in RECEIVER_ROLES:
            receiver = torch.tensor(
                [mapping[receiver_role] for mapping in position_maps],
                dtype=torch.long,
                device=loaded.input_device,
            )
            local_axis = torch.arange(batch_size, device=loaded.input_device)
            receiver_probabilities = probabilities[local_axis, :, receiver, :]
            causal_mask = torch.stack(
                [
                    (sequence_axis >= pads[index]) & (sequence_axis <= receiver[index])
                    for index in range(batch_size)
                ]
            )
            masks: dict[str, torch.Tensor] = {}
            occupied = torch.zeros_like(causal_mask)
            for source_role in ("source_1_record", "source_2_record", "question", "instruction"):
                mask = torch.zeros_like(causal_mask)
                for batch_index in range(batch_size):
                    positions = [
                        value
                        for value in partition_maps[batch_index][source_role]
                        if value <= int(receiver[batch_index].item())
                    ]
                    if positions:
                        mask[batch_index, positions] = True
                mask &= causal_mask
                masks[source_role] = mask
                occupied |= mask
            masks["other_positions"] = causal_mask & ~occupied
            replay = torch.zeros(
                (batch_size, output_blocks.shape[0]),
                dtype=torch.float32,
                device=loaded.input_device,
            )
            source_payload: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for source_role, mask in masks.items():
                alpha = receiver_probabilities * mask.unsqueeze(1)
                weighted = torch.einsum("bhs,bhsd->bhd", alpha, values.float())
                head_writes = torch.einsum("bhd,ohd->bho", weighted, output_blocks)
                write = head_writes.sum(dim=1)
                replay += write
                source_payload[source_role] = (
                    mask.sum(dim=-1),
                    alpha.sum(dim=-1).mean(dim=-1),
                    torch.linalg.vector_norm(write, dim=-1),
                    (write * directions).sum(dim=-1),
                )
            if bias is not None:
                replay += bias
            actual_attention = captures[("attention", layer_index)][
                local_axis, receiver
            ].float()
            replay_error = torch.linalg.vector_norm(actual_attention - replay, dim=-1) / torch.linalg.vector_norm(
                actual_attention, dim=-1
            ).clamp_min(1e-8)
            for batch_index in range(batch_size):
                layer_receivers[batch_index][receiver_role] = {
                    "absolute_token_index": int(receiver[batch_index].item() - pads[batch_index]),
                    "source_partition": {
                        source_role: {
                            "token_count": int(values_tuple[0][batch_index].item()),
                            "attention_mass_mean": clean(values_tuple[1][batch_index].item()),
                            "write_norm": clean(values_tuple[2][batch_index].item()),
                            "source_1_minus_source_2_margin_write": clean(
                                values_tuple[3][batch_index].item()
                            ),
                        }
                        for source_role, values_tuple in source_payload.items()
                    },
                    "attention_replay_relative_error": clean(replay_error[batch_index].item()),
                }
        for batch_index in range(batch_size):
            condition_layers[batch_index].append(
                {
                    "layer": layer_index,
                    "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                    "position_metrics": layer_positions[batch_index],
                    "receiver_metrics": layer_receivers[batch_index],
                }
            )
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": loaded.key,
                "condition_id": row["condition_id"],
                "semantic_group_id": row["semantic_group_id"],
                "paired_group_id": row["paired_group_id"],
                "split": row["split"],
                "physical_fold": row["physical_fold"],
                "contract": row["contract"],
                "condition_kind": row["condition_kind"],
                "relation_family": row["relation_family"],
                "interface": row["interface"],
                "record_order": row["record_order"],
                "mapping": row["mapping"],
                "query_role": row["query_role"],
                "target_position": row["target_position"],
                "semantic_target_source": row["semantic_target_source"],
                "actual_choice": row["actual_choice"],
                "natural_content_good": row["natural_content_good"],
                "layers": condition_layers[index],
                "physical": True,
                "observer": True,
                "predictive": False,
                "causal": False,
                "single_neuron": False,
                "pipeline_sealed": row["pipeline_sealed"],
            }
        )
    del input_ids, attention_mask, captures
    return output, max_reconstruction_error


def collect_physical(loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = stage_root(stage, model) / "physical"
    complete_path = root / "phase435_physical_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    behavior_root = stage_root(stage, model) / "behavior"
    behavior = {
        row["condition_id"]: row
        for row in read_jsonl(behavior_root / "phase435_behavior_rows.jsonl")
    }
    selected = [
        {
            **row,
            "actual_choice": behavior[row["condition_id"]]["actual_choice"],
            "natural_content_good": behavior[row["condition_id"]]["natural_content_good"],
            "natural_generated_token_ids": behavior[row["condition_id"]]["natural_generated_token_ids"],
        }
        for row in rows
    ]
    fast_tokenizer = AutoTokenizer.from_pretrained(
        str(loaded.spec.local_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    layers = get_layers(loaded.model)
    projection = deterministic_projection(int(loaded.model.config.hidden_size), loaded.input_device)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase435_physical_part_*.jsonl.gz"))
    existing_ids = {
        row["condition_id"] for path in existing_paths for row in read_jsonl_any(path)
    }
    pending = [row for row in selected if row["condition_id"] not in existing_ids]
    part = len(existing_paths)
    buffer: list[dict[str, Any]] = []
    processed = len(existing_ids)
    max_reconstruction_error = 0.0
    started = time.monotonic()
    print(
        f"[Phase435 physical] {stage} {model}; conditions={len(selected)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), PHYSICAL_BATCH_SIZE[model]):
        batch = pending[start : start + PHYSICAL_BATCH_SIZE[model]]
        traced, reconstruction_error = collect_physical_batch(
            loaded, fast_tokenizer, layers, batch, projection
        )
        buffer.extend(traced)
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
        processed += len(batch)
        if processed % PHYSICAL_CHECKPOINT < len(batch) or processed == len(selected):
            write_jsonl_gz(
                checkpoint_root / f"phase435_physical_part_{part:05d}.jsonl.gz",
                buffer,
            )
            buffer.clear()
            part += 1
        if processed == len(batch) or processed % 96 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase435 physical] {stage} {model} {processed}/{len(selected)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    if buffer:
        write_jsonl_gz(
            checkpoint_root / f"phase435_physical_part_{part:05d}.jsonl.gz", buffer
        )
    existing_paths = sorted(checkpoint_root.glob("phase435_physical_part_*.jsonl.gz"))
    all_ids = [row["condition_id"] for path in existing_paths for row in read_jsonl_any(path)]
    if len(all_ids) != len(selected) or len(set(all_ids)) != len(selected):
        raise RuntimeError(
            f"Incomplete Phase435 physical: rows={len(all_ids)} unique={len(set(all_ids))} expected={len(selected)}"
        )
    final_path = root / "phase435_physical_rows.jsonl.gz"
    with final_path.open("wb") as target:
        for path in existing_paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target)
    complete = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "stage": stage,
        "condition_count": len(selected),
        "unique_condition_id_count": len(set(all_ids)),
        "layer_count": len(layers),
        "max_block_reconstruction_relative_error": clean(max_reconstruction_error),
        "all_rows_complete": len(set(all_ids)) == len(selected),
        "sealed_read": stage == "sealed",
        "elapsed_seconds": clean(time.monotonic() - started),
    }
    write_json(complete_path, complete)
    return complete


def collect(model: str, stage: str, mode: str) -> dict[str, Any]:
    freeze()
    if stage == "behavior" and not interface_qualified(model):
        return {
            "model": model,
            "stage": stage,
            "skipped": True,
            "reason": "interface_calibration_gate_failed",
        }
    if stage == "physical" and not eligible_contracts(model):
        return {
            "model": model,
            "stage": stage,
            "skipped": True,
            "reason": "no_behavior_eligible_model_contract",
        }
    if stage == "sealed" and not authorized_sealed_contracts(model):
        return {
            "model": model,
            "stage": stage,
            "skipped": True,
            "reason": "no_sealed_authorized_model_contract",
        }
    loaded = None
    try:
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != DTYPES[model]:
            raise RuntimeError(f"Execution dtype mismatch: {actual_dtype} != {DTYPES[model]}")
        rows = materialize_groups(loaded, stage)
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
            behavior_path = stage_root(stage, model) / "behavior/phase435_behavior_complete.json"
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
    parser.add_argument("--stage", choices=("interface", "behavior", "physical", "sealed"), required=True)
    parser.add_argument("--mode", choices=("behavior", "physical", "all"), default="behavior")
    args = parser.parse_args()
    print(json.dumps(collect(args.model, args.stage, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
