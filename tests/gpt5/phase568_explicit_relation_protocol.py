#!/usr/bin/env python3
"""Freeze an independent explicit-key retest of the Phase567 ontology.

Phase567 exposed a systematic surface-vs-marker default. This protocol keeps
the 3x3x2 ontology and 108-cell denominator, but uses explicit body_color and
tag_color keys. Qualification is defined on counterfactual triplets rather
than an all-108-cells world event whose meaning changes with cell count.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase567_multi_relation_binding_protocol as base  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase568"
SCHEMA_VERSION = "phase568_explicit_relation_binding.v1"
MODELS = base.MODELS
SPLIT_WORLD_COUNTS = {
    "gate_discovery": 48,
    "gate_confirmation": 48,
    "role_discovery": 32,
    "role_confirmation": 32,
    "unseen_recombination": 48,
    "sealed": 48,
}
SPLITS = tuple(SPLIT_WORLD_COUNTS)
OPEN_SPLITS = tuple(split for split in SPLITS if split != "sealed")
BINDINGS = base.BINDINGS
QUERY_OBJECTS = base.QUERY_OBJECTS
QUERY_RELATIONS = ("body", "tag")
SURFACES = base.SURFACES
FACT_ORDERS = base.FACT_ORDERS
ROWS_PER_WORLD = 108
CELLS = tuple(
    f"binding{binding}_query{query}_relation{relation}_surface{surface}_order{order}"
    for binding, query, relation, surface, order in itertools.product(
        BINDINGS, QUERY_OBJECTS, QUERY_RELATIONS, SURFACES, FACT_ORDERS
    )
)
TEMPLATE_SPLIT = {
    "gate_discovery": "behavior_discovery",
    "gate_confirmation": "behavior_confirmation",
    "role_discovery": "role_discovery",
    "role_confirmation": "role_confirmation",
    "unseen_recombination": "unseen_recombination",
    "sealed": "sealed",
}

OUT_DIR = ROOT / "tests/gpt5/result/phase568_explicit_relation_binding"
OPEN_CASES_PATH = OUT_DIR / "phase568_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase568_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase568_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase568_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase568_static_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalized_word_multiset(text: str) -> list[str]:
    return sorted(re.findall(r"[A-Za-z]+", text.casefold()))


def world_spec(split: str, world_index: int) -> dict[str, Any]:
    template = TEMPLATE_SPLIT[split]
    # Keep this retest's pseudo-object lexicon outside every Phase567 split
    # range, including the split-specific offsets applied by pseudo_word().
    object_offset = 100000 + world_index * 3
    objects = tuple(base.pseudo_word(object_offset + index, template) for index in range(3))
    heldout = (
        split in {"unseen_recombination", "sealed"}
        and world_index >= SPLIT_WORLD_COUNTS[split] // 2
    )
    pool = base.HELDOUT_VALUES if heldout else base.CORE_VALUES
    values = base.distinct_triple(pool, world_index + list(SPLITS).index(split) * 31, 7)
    offset = (world_index + list(SPLITS).index(split)) % 3
    body_map = tuple((index + offset) % 3 for index in range(3))
    tag_map = tuple((value + 1) % 3 for value in body_map)
    return {
        "objects": objects,
        "values": values,
        "value_regime": "heldout_value_labels" if heldout else "core_value_labels",
        "base_maps": {"body": body_map, "tag": tag_map},
    }


def relation_map(world: dict[str, Any], relation: str, binding: int, query_relation: str) -> tuple[int, ...]:
    base_map = world["base_maps"][relation]
    if relation != query_relation:
        return tuple(base_map)
    return tuple((value + binding) % 3 for value in base_map)


def relation_label(surface: int, relation: str) -> str:
    if surface == 0:
        return f"{relation}_color"
    if surface == 1:
        return f"{relation}-color"
    if surface == 2:
        return f"{relation.upper()}_COLOR"
    raise ValueError(surface)


def render_fact(surface: int, obj: str, relation: str, value: str) -> str:
    label = relation_label(surface, relation)
    if surface == 0:
        return f"record(object={obj}, key={label}, value={value})"
    if surface == 1:
        return f"{obj} | {label} | {value}"
    if surface == 2:
        return f"{label}[{obj}] = {value}"
    raise ValueError(surface)


def render_context(surface: int, records: list[dict[str, Any]]) -> str:
    facts = [record["text"] for record in records]
    if surface == 0:
        return "Temporary typed ledger:\n" + "\n".join(facts)
    if surface == 1:
        return "Temporary typed ledger:\n" + "\n".join(facts)
    if surface == 2:
        return "Temporary typed ledger: " + "; ".join(facts) + "."
    raise ValueError(surface)


def render_question(surface: int, obj: str, relation: str) -> tuple[str, str]:
    label = relation_label(surface, relation)
    if surface == 0:
        return f"lookup(object={obj}, key={label})", label
    if surface == 1:
        return f"Look up key {label} for object {obj}.", label
    if surface == 2:
        return f"Return {label}[{obj}].", label
    raise ValueError(surface)


def ordered_records(
    world: dict[str, Any], query_relation: str, binding: int, surface: int, fact_order: int
) -> list[dict[str, Any]]:
    records = []
    for relation in QUERY_RELATIONS:
        mapping = relation_map(world, relation, binding, query_relation)
        for object_index, value_index in enumerate(mapping):
            obj = world["objects"][object_index]
            value = world["values"][value_index]
            records.append({
                "object_index": object_index,
                "relation": relation,
                "value_index": value_index,
                "object": obj,
                "value": value,
                "relation_label": relation_label(surface, relation),
                "text": render_fact(surface, obj, relation, value),
            })
    if fact_order == 0:
        return records
    by_key = {(row["relation"], row["object_index"]): row for row in records}
    return [
        by_key[(relation, object_index)]
        for object_index in (2, 1, 0)
        for relation in ("tag", "body")
    ]


def controlled_case(
    split: str, world_index: int, binding: int, query_index: int,
    query_relation: str, surface: int, fact_order: int,
) -> dict[str, Any]:
    world = world_spec(split, world_index)
    records = ordered_records(world, query_relation, binding, surface, fact_order)
    context = render_context(surface, records)
    query_object = world["objects"][query_index]
    question, query_label = render_question(surface, query_object, query_relation)
    instruction = (
        "Use the exact requested key. Reply with exactly one lowercase color word "
        "and no explanation."
    )
    raw_prompt = f"{context}\nQuery: {question}\nInstruction: {instruction}"
    target_record = next(
        row for row in records
        if row["relation"] == query_relation and row["object_index"] == query_index
    )
    same_relation_other = [
        row for row in records
        if row["relation"] == query_relation and row["object_index"] != query_index
    ]
    other_relation = [row for row in records if row["relation"] != query_relation]
    target = target_record["value"]
    anchor_id = f"phase568_{split}_{world_index:03d}"
    triplet_id = (
        f"{anchor_id}_query{query_index}_relation{query_relation}_surface{surface}_order{fact_order}"
    )
    raw_facts = " ".join(row["text"] for row in records)
    return {
        "raw_prompt": raw_prompt,
        "context": context,
        "question": question,
        "instruction": instruction,
        "objects": list(world["objects"]),
        "values": list(world["values"]),
        "value_regime": world["value_regime"],
        "relation_maps": {
            relation: list(relation_map(world, relation, binding, query_relation))
            for relation in QUERY_RELATIONS
        },
        "binding": binding,
        "query_object_index": query_index,
        "query_object": query_object,
        "query_relation": query_relation,
        "other_relation": "tag" if query_relation == "body" else "body",
        "query_relation_label": query_label,
        "target": target,
        "target_aliases": [target],
        "distractors": [value for value in world["values"] if value != target],
        "all_candidates": list(world["values"]),
        "surface_id": surface,
        "fact_order": fact_order,
        "anchor_id": anchor_id,
        "triplet_id": triplet_id,
        "factorial_cell": (
            f"binding{binding}_query{query_index}_relation{query_relation}_"
            f"surface{surface}_order{fact_order}"
        ),
        "factorial_cell_without_binding": (
            f"query{query_index}_relation{query_relation}_surface{surface}_order{fact_order}"
        ),
        "fact_token_multiset_key": stable_hash(normalized_word_multiset(raw_facts)),
        "prompt_token_multiset_key": stable_hash(normalized_word_multiset(raw_prompt)),
        "fact_records": records,
        "semantic_fragments": {
            "target_fact": target_record["text"],
            "target_fact_object": target_record["object"],
            "target_fact_value": target_record["value"],
            "target_fact_relation": target_record["relation_label"],
            "same_relation_other_facts": [row["text"] for row in same_relation_other],
            "other_relation_facts": [row["text"] for row in other_relation],
            "query_relation": query_label,
            "query_object": query_object,
        },
    }


def materialize(model: str, tokenizer: Any, row: dict[str, Any], split: str) -> dict[str, Any]:
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
    distractor_ids = {
        value: tokenizer(value, add_special_tokens=False)["input_ids"]
        for value in row["distractors"]
    }
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "target_token_ids": [int(value) for value in target_ids],
        "distractor_token_ids": {
            key: [int(value) for value in values] for key, values in distractor_ids.items()
        },
        "case_id": (
            f"{row['anchor_id']}_{model}_binding{row['binding']}_query{row['query_object_index']}_"
            f"relation{row['query_relation']}_surface{row['surface_id']}_order{row['fact_order']}"
        ),
        "split": split,
        "sealed": split == "sealed",
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for split, count in SPLIT_WORLD_COUNTS.items():
        for world_index in range(count):
            semantic_rows = [
                controlled_case(split, world_index, binding, query, relation, surface, order)
                for binding, query, relation, surface, order in itertools.product(
                    BINDINGS, QUERY_OBJECTS, QUERY_RELATIONS, SURFACES, FACT_ORDERS
                )
            ]
            for model in MODELS:
                target = sealed_rows if split == "sealed" else open_rows
                target.extend(materialize(model, tokenizers[model], row, split) for row in semantic_rows)
    return open_rows, sealed_rows


def audit_rows(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = open_rows + sealed_rows
    failures = []
    expected_open = sum(SPLIT_WORLD_COUNTS[split] for split in OPEN_SPLITS) * 108 * 3
    expected_sealed = SPLIT_WORLD_COUNTS["sealed"] * 108 * 3
    if len(open_rows) != expected_open:
        failures.append("open_count")
    if len(sealed_rows) != expected_sealed:
        failures.append("sealed_count")
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        failures.append("case_id_collision")
    by_world: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_triplet: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_world[(row["model"], row["anchor_id"])].append(row)
        by_triplet[(row["model"], row["triplet_id"])].append(row)
    world_sizes = sorted({len(group) for group in by_world.values()})
    triplet_sizes = sorted({len(group) for group in by_triplet.values()})
    if world_sizes != [108]:
        failures.append("world_size")
    if triplet_sizes != [3]:
        failures.append("triplet_size")
    invariant_failures = 0
    for group in by_triplet.values():
        ordered = sorted(group, key=lambda row: row["binding"])
        fixed = (
            "objects", "values", "query_object", "query_relation", "surface_id", "fact_order",
            "fact_token_multiset_key", "prompt_token_multiset_key",
        )
        if any(row[key] != ordered[0][key] for row in ordered[1:] for key in fixed):
            invariant_failures += 1
        elif {row["target"] for row in ordered} != set(ordered[0]["values"]):
            invariant_failures += 1
        elif len({tuple(row["relation_maps"][row["other_relation"]]) for row in ordered}) != 1:
            invariant_failures += 1
    if invariant_failures:
        failures.append("triplet_invariants")
    phase567_objects: set[str] = set()
    for line in base.OPEN_CASES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prior_row = json.loads(line)
        if prior_row["model"] == "qwen3":
            phase567_objects.update(prior_row["objects"])
    phase568_objects = {
        value for row in open_rows if row["model"] == "qwen3" for value in row["objects"]
    }
    prior_overlap = len(phase567_objects & phase568_objects)
    if prior_overlap:
        failures.append("prior_object_overlap")
    return {
        "schema_version": "phase568_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": not failures,
        "status": "static_pass_no_model_run" if not failures else "static_fail",
        "failures": failures,
        "registered_case_count": len(all_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "open_case_count_by_model": dict(sorted(Counter(row["model"] for row in open_rows).items())),
        "rows_per_world": world_sizes,
        "rows_per_triplet": triplet_sizes,
        "triplet_count": len(by_triplet),
        "triplet_invariant_failure_count": invariant_failures,
        "phase567_object_overlap_count": prior_overlap,
        "sealed_split_read": False,
        "model_execution_performed": False,
    }


def freeze() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = audit_rows(open_rows, sealed_rows)
    if not audit["valid"]:
        raise RuntimeError(f"Phase568 static audit failed: {audit['failures']}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(SEALED_COMMITMENT_PATH, {
        "schema_version": "phase568_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    })
    protocol = {
        "schema_version": "phase568_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "open_splits": list(OPEN_SPLITS),
        "rows_per_world": 108,
        "factorial_cells": list(CELLS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "behavior_gate": {
            "semantic_accuracy_min_per_gate_split": 0.98,
            "all_three_bindings_correct_triplet_rate_min": 0.95,
            "minimum_axis_wilson_95_lcb": 0.95,
            "minimum_factorial_cell_accuracy": 0.80,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "minimum_all_correct_role_triplets_per_split": 900,
        },
        "evidence_policy": {
            "new_object_lexicon_after_phase567_diagnosis": True,
            "same_three_object_three_value_two_relation_ontology": True,
            "explicit_relation_keys_replace_ambiguous_natural_labels": True,
            "world_all_cells_rate_reported_not_used_as_gate": True,
            "triplet_gate_scales_with_counterfactual_unit": True,
            "phase567_frozen_result_not_reclassified": True,
            "role_coordinates_must_be_disjoint": True,
            "fine_scan_before_coarse_role_replication": False,
            "sealed_split_read": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    write_json(AUDIT_PATH, {
        **audit,
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
    })
    print(json.dumps({
        "registered": protocol["registered_case_count"],
        "open": protocol["open_case_count"],
        "sealed": protocol["sealed_case_count"],
        "per_model_open": audit["open_case_count_by_model"],
        "valid": audit["valid"],
    }, ensure_ascii=False, indent=2))
    return protocol


if __name__ == "__main__":
    freeze()
