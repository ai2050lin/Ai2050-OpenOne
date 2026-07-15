#!/usr/bin/env python3
"""Collect Phase437 factorized behavior and gate-authorized physical ledgers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase431_position_time_collect as p431  # noqa: E402
import phase435_natural_relation_collect as engine  # noqa: E402
import phase435_natural_relation_protocol as p435  # noqa: E402
from hf_probe_env import load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase429_typed_route_protocol import render_chat  # noqa: E402
from phase433_shared_prefix_collect import (  # noqa: E402
    common_prefix_length,
    contextual_continuation_ids,
    digest_ids,
)
from phase437_position_factor_protocol import (  # noqa: E402
    BEHAVIOR_SPLITS,
    BEHAVIOR_VARIANTS,
    CONTRACTS,
    DTYPES,
    FROZEN_INTERFACES,
    MATCHED_VARIANTS,
    MODELS,
    OBSERVER_SPLIT,
    OUT,
    PHYSICAL_SPLIT,
    PHYSICAL_VARIANTS,
    SCHEMA_VERSION,
    SEALED_SPLIT,
    freeze,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE_ID = "Phase437-PositionFactorCollection"
TRACE_SCHEMA_VERSION = "phase437_position_factor_trace.v1"
BEHAVIOR_BATCH_SIZE = {"qwen3": 8, "glm4": 6, "deepseek7b": 8}
BEHAVIOR_CHECKPOINT = 256
GENERATION_HORIZON = 16

engine.OUT = OUT
engine.PHASE_ID = PHASE_ID
engine.SCHEMA_VERSION = SCHEMA_VERSION
engine.TRACE_SCHEMA_VERSION = TRACE_SCHEMA_VERSION


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase437 non-finite scalar: {value}")
    return round(float(value), 9)


def group_path(split: str) -> Path:
    if split == SEALED_SPLIT:
        return OUT / "sealed/phase437_groups_sealed.jsonl"
    return OUT / f"phase437_groups_{split}.jsonl"


def observer_contracts(model: str) -> list[str]:
    path = OUT / "phase437_observer_freeze.json"
    if not path.exists():
        return []
    return sorted(
        contract
        for contract, payload in read_json(path)["models"][model]["contracts"].items()
        if payload["observer_qualified"]
    )


def behavior_contracts(model: str) -> list[str]:
    path = OUT / "phase437_behavior_gate.json"
    if not path.exists():
        return []
    return sorted(
        row["contract"]
        for row in read_json(path).get("eligible_model_contracts", [])
        if row["model"] == model
    )


def sealed_contracts(model: str) -> list[str]:
    path = OUT / "phase437_open_gate.json"
    if not path.exists() or not read_json(path).get("sealed_unlock"):
        return []
    return sorted(
        row["contract"]
        for row in read_json(path).get("sealed_authorized_model_contracts", [])
        if row["model"] == model
    )


def variant_spec(variant: str) -> dict[str, str]:
    if variant not in BEHAVIOR_VARIANTS:
        raise ValueError(variant)
    return {
        "target_position": "first" if variant.startswith("first_") else "second",
        "recency_control": "matched" if "_matched_" in variant else "natural",
        "post_gap": "far" if variant.endswith("_far") else "near",
    }


def record_order(query_role: str, target_position: str) -> str:
    if query_role == "a":
        return "ab" if target_position == "first" else "ba"
    return "ba" if target_position == "first" else "ab"


def boundary_suffix(boundary: str) -> str:
    return {
        "period": ".",
        "semicolon": ";",
        "newline": "",
        "field_delimiter": " |",
    }[boundary]


def relation_first_core(group: dict[str, Any], entity: str, value: str) -> tuple[str, str]:
    relation = group["relation_label"]
    return (
        f"Under the {relation} relation, {value} is linked to {entity}",
        relation,
    )


def record_core(
    group: dict[str, Any], contract: str, entity: str, value: str
) -> tuple[str, str]:
    config = p435.FAMILY_CONFIG[group["relation_family"]]
    if group["label_order"] == "relation_first":
        return relation_first_core(group, entity, value)
    if contract == "field_extract":
        relation = group["value_label"]
        return f"{group['entity_label']}: {entity}; {relation}: {value}", relation
    if contract == "natural_qa":
        relation = str(config["statement_relation_surface"])
        return str(config["statement"]).format(entity=entity, value=value).rstrip("."), relation
    relation = group["relation_label"]
    return (
        f"The registry links {entity} with {value} through the {relation} relation",
        relation,
    )


def build_record(
    group: dict[str, Any], contract: str, role: str, entity: str, value: str
) -> dict[str, Any]:
    core, relation_surface = record_core(group, contract, entity, value)
    if group["record_length"] == "long":
        core = f"According to the current registry entry, {core}, and this entry remains active"
    line = core + boundary_suffix(group["boundary"])
    return {
        "semantic_role": role,
        "entity": entity,
        "value": value,
        "value_source": engine.value_source(group, value),
        "relation_surface": relation_surface,
        "line": line,
    }


def neutral_words(word_count: int) -> str:
    words = (
        "neutral audit note adds no entity relation value evidence or instruction and only controls distance"
    ).split()
    selected = [words[index % len(words)] for index in range(max(4, word_count))]
    return "Distance-control note: " + " ".join(selected) + "."


def post_gap_text(post_gap: str) -> str:
    if post_gap == "near":
        return ""
    return (
        "A neutral ledger note contains no entity, relation, or value. "
        "It does not revise either registry record. "
        "The two records above remain the complete evidence."
    )


def connector_text(connector: str) -> str:
    return {
        "parallel": "Also,",
        "separate": "In a separate record,",
        "none": "",
    }[connector]


def query_surface(group: dict[str, Any], contract: str, query_entity: str) -> tuple[str, str]:
    config = p435.FAMILY_CONFIG[group["relation_family"]]
    if contract == "field_extract":
        relation = group["value_label"]
        question = f"For {query_entity}, what is the {relation}?"
    elif contract == "natural_qa":
        relation = str(config["question_relation_surface"])
        question = str(config["question"]).format(entity=query_entity)
    else:
        relation = group["relation_label"]
        question = f"Give the value linked to {query_entity} by the {relation} relation."
    return f"Question: {question}", relation


def build_surface(
    group: dict[str, Any], contract: str, role_values: dict[str, str], variant: str
) -> dict[str, Any]:
    spec = variant_spec(variant)
    query_role = group["query_role"]
    order_name = record_order(query_role, spec["target_position"])
    entities = {"a": group["entity_a"], "b": group["entity_b"]}
    order = ("a", "b") if order_name == "ab" else ("b", "a")
    entries = [
        build_record(group, contract, role, entities[role], role_values[role])
        for role in order
    ]
    second_prefix = connector_text(group["connector"])
    record_lines = [entries[0]["line"]]
    record_lines.append(
        f"{second_prefix} {entries[1]['line']}".strip()
        if second_prefix
        else entries[1]["line"]
    )
    spacer = ""
    if spec["recency_control"] == "matched":
        spacer = neutral_words(len(entries[0]["line"].split()))
    gap = post_gap_text(spec["post_gap"])
    question_line, query_relation = query_surface(group, contract, entities[query_role])
    return {
        "record_entries": entries,
        "record_block": "\n".join(record_lines),
        "distance_spacer": spacer,
        "post_gap_text": gap,
        "question_line": question_line,
        "query_entity": entities[query_role],
        "query_relation_surface": query_relation,
        "record_order": order_name,
        **spec,
    }


def alternate_variant(variant: str, control_type: str) -> str:
    """Return a registered surface intervention for factor controls."""
    spec = variant_spec(variant)
    if control_type == "order_swap_control":
        prefix = "second" if spec["target_position"] == "first" else "first"
        return f"{prefix}_natural_{spec['post_gap']}"
    if control_type == "distance_swap_control":
        if spec["target_position"] == "second":
            recency = "natural" if spec["recency_control"] == "matched" else "matched"
            return f"second_{recency}_{spec['post_gap']}"
        post_gap = "far" if spec["post_gap"] == "near" else "near"
        return f"first_natural_{post_gap}"
    return variant


def rebuild_record_block(group: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    second_prefix = connector_text(group["connector"])
    second = (
        f"{second_prefix} {entries[1]['line']}".strip()
        if second_prefix
        else entries[1]["line"]
    )
    return "\n".join((entries[0]["line"], second))


def controlled_surface(
    group: dict[str, Any],
    contract: str,
    role_values: dict[str, str],
    variant: str,
    condition_kind: str,
) -> tuple[dict[str, Any], dict[str, str], str, dict[str, Any]]:
    """Materialize a genuinely changed prompt for every registered control."""
    surface_group = dict(group)
    surface_values = dict(role_values)
    control_type = (
        condition_kind.removeprefix("control_")
        if condition_kind.startswith("control_")
        else None
    )
    surface_variant = alternate_variant(variant, control_type or "")
    if control_type == "boundary_swap_control":
        boundaries = ("period", "semicolon", "newline", "field_delimiter")
        surface_group["boundary"] = boundaries[(boundaries.index(group["boundary"]) + 1) % len(boundaries)]
    if control_type == "wrong_value_mapping":
        surface_values = {"a": role_values["b"], "b": role_values["a"]}

    surface = build_surface(surface_group, contract, surface_values, surface_variant)
    if control_type in {"no_relation", "wrong_relation"}:
        entries = []
        for entry in surface["record_entries"]:
            item = dict(entry)
            if control_type == "no_relation":
                item["relation_surface"] = "independent mention"
                item["line"] = (
                    f"Independent mention: {item['entity']}; independent value mention: "
                    f"{item['value']}{boundary_suffix(surface_group['boundary'])}"
                )
            else:
                item["relation_surface"] = "unrelated checksum"
                item["line"] = (
                    f"The unrelated checksum for {item['entity']} is {item['value']}"
                    f"{boundary_suffix(surface_group['boundary'])}"
                )
            entries.append(item)
        surface["record_entries"] = entries
        surface["record_block"] = rebuild_record_block(surface_group, entries)
    elif control_type == "wrong_query_entity":
        control_entity = f"Unlisted Control Entity {group['group_index']}"
        surface["query_entity"] = control_entity
        surface["question_line"] = (
            f"Question: For {control_entity}, what is the "
            f"{surface['query_relation_surface']}?"
        )
    elif control_type == "wrong_query_relation":
        surface["query_relation_surface"] = "unrelated checksum"
        surface["question_line"] = (
            f"Question: For {surface['query_entity']}, what is the unrelated checksum?"
        )

    design = {
        "control_type": control_type,
        "surface_variant": surface_variant,
        "surface_boundary": surface_group["boundary"],
        "surface_mapping": "swapped_from_candidate" if control_type == "wrong_value_mapping" else "candidate",
        "prompt_changed": control_type is not None,
    }
    return surface, surface_values, surface_variant, design


def materialize_condition(
    group: dict[str, Any],
    interface: str,
    variant: str,
    loaded: Any,
    fast_tokenizer: Any,
    *,
    condition_kind: str = "candidate",
) -> dict[str, Any]:
    contract = group["contract"]
    role_values = engine.mapping_values(group, group["mapping"])
    query_role = group["query_role"]
    opposite_role = "b" if query_role == "a" else "a"
    target_value = role_values[query_role]
    opposite_value = role_values[opposite_role]
    surface, surface_role_values, surface_variant, control_design = controlled_surface(
        group, contract, role_values, variant, condition_kind
    )
    output = engine.interface_payload(interface, target_value, opposite_value)
    lines = [
        "Use only the two registry records below. Treat them as the complete context.",
        surface["record_block"],
    ]
    if surface["distance_spacer"]:
        lines.append(surface["distance_spacer"])
    if surface["post_gap_text"]:
        lines.append(surface["post_gap_text"])
    lines.extend((surface["question_line"], str(output["instruction"])))
    content = "\n".join(lines)
    rendered = render_chat(loaded.tokenizer, loaded.key, content)
    prompt_ids = [
        int(value)
        for value in loaded.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    ]
    target_ids, target_exact = contextual_continuation_ids(
        loaded.tokenizer, rendered, prompt_ids, str(output["target"])
    )
    opposite_ids, opposite_exact = contextual_continuation_ids(
        loaded.tokenizer, rendered, prompt_ids, str(output["opposite"])
    )
    common_length = common_prefix_length(target_ids, opposite_ids)
    if common_length >= len(target_ids) or common_length >= len(opposite_ids):
        raise RuntimeError(f"Phase437 continuations do not branch: {group['semantic_group_id']}")
    value_1_ids = loaded.tokenizer(group["value_1"], add_special_tokens=False)["input_ids"]
    value_2_ids = loaded.tokenizer(group["value_2"], add_special_tokens=False)["input_ids"]
    condition_id = (
        f"{group['semantic_group_id']}__v_{variant}__i_{interface}"
        f"__kind_{condition_kind}__{loaded.key}"
    )
    row = {
        **group,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "model": loaded.key,
        "condition_id": condition_id,
        "condition_kind": condition_kind,
        "interface": interface,
        "variant": variant,
        "surface_variant": surface_variant,
        "control_design": control_design,
        "record_order": surface["record_order"],
        "query_role": query_role,
        "target_position": surface["target_position"],
        "recency_control": surface["recency_control"],
        "post_gap": surface["post_gap"],
        "role_values": role_values,
        "surface_role_values": surface_role_values,
        "target_value": target_value,
        "opposite_value": opposite_value,
        "semantic_target_source": engine.value_source(group, target_value),
        "semantic_opposite_source": engine.value_source(group, opposite_value),
        "source_1": group["value_1"],
        "source_2": group["value_2"],
        "semantic_target": target_value,
        "semantic_opposite": opposite_value,
        "target": str(output["target"]),
        "opposite_target": str(output["opposite"]),
        "target_sequence_token_ids": [int(value) for value in target_ids],
        "opposite_sequence_token_ids": [int(value) for value in opposite_ids],
        "output_common_prefix_token_count": common_length,
        "target_branch_token_id": int(target_ids[common_length]),
        "opposite_branch_token_id": int(opposite_ids[common_length]),
        "target_contextual_tokenization_exact": target_exact,
        "opposite_contextual_tokenization_exact": opposite_exact,
        "source_1_first_token_id": int(value_1_ids[0]),
        "source_2_first_token_id": int(value_2_ids[0]),
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
        "distance_spacer": surface["distance_spacer"],
        "post_gap_text": surface["post_gap_text"],
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
    }
    return {**row, **token_distance_metrics(fast_tokenizer, row)}


def token_distance_metrics(fast_tokenizer: Any, row: dict[str, Any]) -> dict[str, int]:
    encoded = fast_tokenizer(
        row["rendered_prompt"], add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if digest_ids(ids) != row["prompt_token_ids_sha256"]:
        raise RuntimeError(f"Phase437 fast tokenizer disagreement: {row['condition_id']}")
    entries = row["record_entries"]
    target_entry = next(entry for entry in entries if entry["semantic_role"] == row["query_role"])
    distractor_entry = next(entry for entry in entries if entry["semantic_role"] != row["query_role"])

    def positions(entry: dict[str, Any]) -> list[int]:
        start = row["rendered_prompt"].find(entry["line"])
        return engine.token_positions(
            row["rendered_prompt"], offsets, entry["value"], start
        )

    question_start = row["rendered_prompt"].find(row["question_line"])
    question = engine.token_positions(
        row["rendered_prompt"], offsets, row["question_line"], question_start
    )
    target = positions(target_entry)
    distractor = positions(distractor_entry)
    question_token = min(question)
    terminal = len(ids) - 1
    return {
        "target_value_token_end": max(target),
        "distractor_value_token_end": max(distractor),
        "question_token_start": question_token,
        "prompt_terminal_index": terminal,
        "target_to_question_token_distance": question_token - max(target) - 1,
        "distractor_to_question_token_distance": question_token - max(distractor) - 1,
        "target_to_terminal_token_distance": terminal - max(target),
        "distractor_to_terminal_token_distance": terminal - max(distractor),
    }


def interface_for(model: str, contract: str) -> str:
    return FROZEN_INTERFACES[model][contract]


def materialize_stage(loaded: Any, fast_tokenizer: Any, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage == "observer":
        for group in read_jsonl(group_path(OBSERVER_SPLIT)):
            rows.append(
                materialize_condition(
                    group,
                    interface_for(loaded.key, group["contract"]),
                    group["observer_variant"],
                    loaded,
                    fast_tokenizer,
                )
            )
    elif stage == "behavior":
        allowed = set(observer_contracts(loaded.key))
        for split in BEHAVIOR_SPLITS:
            for group in read_jsonl(group_path(split)):
                if group["contract"] not in allowed:
                    continue
                for variant in BEHAVIOR_VARIANTS:
                    rows.append(
                        materialize_condition(
                            group,
                            interface_for(loaded.key, group["contract"]),
                            variant,
                            loaded,
                            fast_tokenizer,
                        )
                    )
    elif stage in {"physical", "sealed"}:
        allowed = set(
            behavior_contracts(loaded.key)
            if stage == "physical"
            else sealed_contracts(loaded.key)
        )
        split = PHYSICAL_SPLIT if stage == "physical" else SEALED_SPLIT
        for group in read_jsonl(group_path(split)):
            if group["contract"] not in allowed:
                continue
            interface = interface_for(loaded.key, group["contract"])
            for variant in PHYSICAL_VARIANTS:
                rows.append(
                    materialize_condition(group, interface, variant, loaded, fast_tokenizer)
                )
                rows.append(
                    materialize_condition(
                        group,
                        interface,
                        variant,
                        loaded,
                        fast_tokenizer,
                        condition_kind=f"control_{group['control_type']}",
                    )
                )
    else:
        raise ValueError(stage)
    return sorted(rows, key=lambda row: row["condition_id"])


RETAINED_FIELDS = (
    "contract",
    "condition_kind",
    "interface",
    "variant",
    "record_order",
    "mapping",
    "query_role",
    "target_position",
    "recency_control",
    "post_gap",
    "relation_family",
    "boundary",
    "connector",
    "record_length",
    "label_order",
    "semantic_target_source",
    "semantic_opposite_source",
    "target_contextual_tokenization_exact",
    "opposite_contextual_tokenization_exact",
    "physical_fold",
    "pipeline_sealed",
    "target_to_question_token_distance",
    "distractor_to_question_token_distance",
    "target_to_terminal_token_distance",
    "distractor_to_terminal_token_distance",
)


def enrich_behavior(source: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    semantic_good = bool(
        source["teacher_sequence_correct"]
        and source["actual_choice"] == row["semantic_target_source"]
        and source["natural_target_first"]
        and not source["natural_opposite_first"]
        and not source["natural_revision"]
    )
    return {
        **source,
        **{key: row[key] for key in RETAINED_FIELDS},
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "semantic_content_good": semantic_good,
        "natural_first_answer_good": bool(
            source["natural_target_first"] and not source["natural_opposite_first"]
        ),
        "natural_complete_answer_good": bool(source["natural_exact_target_contract"]),
        "natural_content_good": semantic_good,
        "natural_stop_good": bool(source["natural_stop"] and not source["natural_censoring"]),
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "single_neuron": False,
    }


def stage_root(stage: str, model: str) -> Path:
    return OUT / stage / model


def collect_behavior(
    loaded: Any, model: str, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    root = stage_root(stage, model) / "behavior"
    complete_path = root / "phase435_behavior_complete.json"
    if complete_path.exists() and read_json(complete_path).get("all_rows_complete"):
        return read_json(complete_path)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(checkpoint_root.glob("phase437_behavior_part_*.jsonl"))
    existing = [row for path in existing_paths for row in read_jsonl(path)]
    completed = {row["condition_id"] for row in existing}
    pending = [row for row in rows if row["condition_id"] not in completed]
    part = len(existing_paths)
    buffer: list[dict[str, Any]] = []
    processed = len(completed)
    started = time.monotonic()
    print(
        f"[Phase437 behavior] {stage} {model}; conditions={len(rows)}; pending={len(pending)}",
        flush=True,
    )
    for start in range(0, len(pending), BEHAVIOR_BATCH_SIZE[model]):
        batch = pending[start : start + BEHAVIOR_BATCH_SIZE[model]]
        raw = p431.collect_behavior_batch(loaded, batch)
        buffer.extend(enrich_behavior(item, row) for item, row in zip(raw, batch))
        processed += len(batch)
        if processed % BEHAVIOR_CHECKPOINT < len(batch) or processed == len(rows):
            write_jsonl(
                checkpoint_root / f"phase437_behavior_part_{part:05d}.jsonl",
                buffer,
            )
            buffer.clear()
            part += 1
        if processed == len(batch) or processed % 512 < len(batch):
            allocated, reserved = vram_gb()
            print(
                f"[Phase437 behavior] {stage} {model} {processed}/{len(rows)}; "
                f"VRAM={allocated:.2f}/{reserved:.2f} GiB",
                flush=True,
            )
    if buffer:
        write_jsonl(
            checkpoint_root / f"phase437_behavior_part_{part:05d}.jsonl", buffer
        )
    collected = [
        row
        for path in sorted(checkpoint_root.glob("phase437_behavior_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    unique = {row["condition_id"]: row for row in collected}
    final_rows = [unique[key] for key in sorted(unique)]
    if len(final_rows) != len(rows):
        raise RuntimeError(f"Incomplete Phase437 behavior: {len(final_rows)} != {len(rows)}")
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
        "semantic_content_good_count": sum(row["semantic_content_good"] for row in final_rows),
        "actual_choice_counts": dict(Counter(row["actual_choice"] for row in final_rows)),
        "all_rows_complete": len(final_rows) == len(rows),
        "sealed_read": stage == "sealed",
        "elapsed_seconds": clean(time.monotonic() - started),
    }
    write_json(complete_path, complete)
    return complete


def collect(model: str, stage: str, mode: str) -> dict[str, Any]:
    freeze()
    if stage == "behavior" and not observer_contracts(model):
        return {"model": model, "stage": stage, "skipped": True, "reason": "no_qualified_observer"}
    if stage == "physical" and not behavior_contracts(model):
        return {"model": model, "stage": stage, "skipped": True, "reason": "no_behavior_eligible_contract"}
    if stage == "sealed" and not sealed_contracts(model):
        return {"model": model, "stage": stage, "skipped": True, "reason": "sealed_not_authorized"}
    loaded = None
    try:
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != DTYPES[model]:
            raise RuntimeError(f"Execution dtype mismatch: {actual_dtype} != {DTYPES[model]}")
        fast_tokenizer = AutoTokenizer.from_pretrained(
            str(loaded.spec.local_dir),
            trust_remote_code=loaded.spec.trust_remote_code,
            local_files_only=True,
            use_fast=True,
        )
        rows = materialize_stage(loaded, fast_tokenizer, stage)
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
            output["physical"] = engine.collect_physical(loaded, model, stage, rows)
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
    parser.add_argument("--stage", choices=("observer", "behavior", "physical", "sealed"), required=True)
    parser.add_argument("--mode", choices=("behavior", "physical", "all"), default="behavior")
    args = parser.parse_args()
    print(json.dumps(collect(args.model, args.stage, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
