#!/usr/bin/env python3
"""Freeze Phase569 non-colliding relation-competition phenotypes.

The queried relation cycles over three values while the same object's other
relation uses a fourth value. This makes target-vs-other-relation selection
observable in every row, unlike a three-value/three-binding construction.
"""

from __future__ import annotations

import hashlib
import gzip
import itertools
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402
from phase567_multi_relation_binding_protocol import (  # noqa: E402
    SYLLABLE_A,
    SYLLABLE_B,
    alpha_code,
)


PHASE = "Phase569"
SCHEMA_VERSION = "phase569_relation_competition.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLIT_WORLD_COUNTS = {
    "phenotype_discovery": 128,
    "phenotype_confirmation": 128,
    "path_discovery": 96,
    "path_confirmation": 96,
    "sealed": 96,
}
SPLITS = tuple(SPLIT_WORLD_COUNTS)
OPEN_SPLITS = tuple(split for split in SPLITS if split != "sealed")
BINDINGS = (0, 1, 2)
QUERY_OBJECTS = (0, 1, 2)
QUERY_RELATIONS = ("body", "tag")
SURFACES = (0, 1, 2)
FACT_ORDERS = (0, 1)
ROWS_PER_WORLD = 108
CORE_VALUES = ("alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "lambda")
HELDOUT_VALUES = ("amber", "silver", "mint", "tan", "rust")
CELLS = tuple(
    f"binding{binding}_query{query}_relation{relation}_surface{surface}_order{order}"
    for binding, query, relation, surface, order in itertools.product(
        BINDINGS, QUERY_OBJECTS, QUERY_RELATIONS, SURFACES, FACT_ORDERS
    )
)

OUT_DIR = ROOT / "tests/gpt5/result/phase569_relation_competition"
OPEN_CASES_PATH = OUT_DIR / "phase569_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase569_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase569_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase569_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase569_static_audit.json"


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
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def pseudo_word(index: int, split: str) -> str:
    shifted = 300000 + list(SPLITS).index(split) * 12000 + index
    stem = (
        SYLLABLE_A[shifted % len(SYLLABLE_A)]
        + SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
        + SYLLABLE_A[(shifted * 17 + 5) % len(SYLLABLE_A)]
    )
    return (stem + alpha_code(shifted)).capitalize()


def distinct_quad(pool: tuple[str, ...], index: int, stride: int) -> tuple[str, str, str, str]:
    selected: list[str] = []
    cursor = 0
    while len(selected) < 4:
        value = pool[(index * stride + cursor * 3 + 1) % len(pool)]
        if value not in selected:
            selected.append(value)
        cursor += 1
        if cursor > len(pool) * 4:
            raise RuntimeError("Could not construct a distinct value quadruple")
    return selected[0], selected[1], selected[2], selected[3]


def world_spec(split: str, world_index: int) -> dict[str, Any]:
    # The fourth object is a never-queried balancing object. It keeps all four
    # labels present under every binding so triplet token multisets stay fixed.
    objects = tuple(pseudo_word(world_index * 4 + offset, split) for offset in range(4))
    heldout = split == "sealed" or (
        split == "path_confirmation" and world_index >= SPLIT_WORLD_COUNTS[split] // 2
    )
    pool = HELDOUT_VALUES if heldout else CORE_VALUES
    values = distinct_quad(pool, world_index + list(SPLITS).index(split) * 137, 5)
    return {
        "objects": objects,
        "values": values,
        "value_regime": "heldout_labels" if heldout else "core_labels",
        "map_offset": (world_index + list(SPLITS).index(split)) % 4,
    }


def relation_map(
    world: dict[str, Any], relation: str, binding: int, query_relation: str
) -> tuple[int, ...]:
    offset = int(world["map_offset"])
    if relation == query_relation:
        return tuple((object_index + offset + binding + 1) % 4 for object_index in range(4))
    return tuple((object_index + offset) % 4 for object_index in range(4))


def relation_label(surface: int, relation: str) -> str:
    if surface == 0:
        return f"{relation}_field"
    if surface == 1:
        return f"{relation} label"
    if surface == 2:
        return f"{relation.upper()}_FIELD"
    raise ValueError(surface)


def render_fact(surface: int, obj: str, relation: str, value: str) -> str:
    label = relation_label(surface, relation)
    if surface == 0:
        return f"item={obj}; field={label}; label={value}"
    if surface == 1:
        return f"For entity {obj}, the {label} is {value}."
    if surface == 2:
        return f"{obj} || {label} || {value}"
    raise ValueError(surface)


def render_context(surface: int, records: list[dict[str, Any]]) -> str:
    facts = [record["text"] for record in records]
    if surface == 2:
        return "Independent temporary registry:\n" + "\n".join(facts)
    return "Independent temporary registry:\n" + "\n".join(facts)


def render_question(surface: int, obj: str, relation: str) -> tuple[str, str]:
    label = relation_label(surface, relation)
    if surface == 0:
        return f"read(item={obj}, field={label})?", label
    if surface == 1:
        return f"Which label is the {label} for entity {obj}?", label
    if surface == 2:
        return f"LOOKUP {obj} || {label} || ?", label
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
        for object_index in (3, 2, 1, 0)
        for relation in ("tag", "body")
    ]


def controlled_case(
    split: str,
    world_index: int,
    binding: int,
    query_index: int,
    query_relation: str,
    surface: int,
    fact_order: int,
) -> dict[str, Any]:
    world = world_spec(split, world_index)
    records = ordered_records(world, query_relation, binding, surface, fact_order)
    context = render_context(surface, records)
    query_object = world["objects"][query_index]
    question, query_label = render_question(surface, query_object, query_relation)
    instruction = "Return exactly one lowercase label from the registry and no other text."
    raw_prompt = f"{context}\nQuery: {question}\nInstruction: {instruction}"
    target_record = next(
        row for row in records
        if row["relation"] == query_relation and row["object_index"] == query_index
    )
    other_relation = "tag" if query_relation == "body" else "body"
    other_record = next(
        row for row in records
        if row["relation"] == other_relation and row["object_index"] == query_index
    )
    target = target_record["value"]
    other_target = other_record["value"]
    if target == other_target:
        raise RuntimeError("Phase569 relation competition collapsed to the same value")
    anchor_id = f"phase569_{split}_{world_index:03d}"
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
        "query_object_count": 3,
        "balancing_object_index": 3,
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
        "other_relation": other_relation,
        "query_relation_label": query_label,
        "target": target,
        "other_relation_target": other_target,
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
            "target_fact_relation": target_record["relation_label"],
            "target_fact_value": target_record["value"],
            "other_fact": other_record["text"],
            "other_fact_object": other_record["object"],
            "other_fact_relation": other_record["relation_label"],
            "other_fact_value": other_record["value"],
            "query_relation": query_label,
            "query_object": query_object,
        },
    }


def materialize_semantic(
    tokenizers: dict[str, Any], row: dict[str, Any], split: str
) -> dict[str, Any]:
    candidate_ids_by_model = {}
    prompt_token_count_by_model = {}
    for model, tokenizer in tokenizers.items():
        prompt = render_chat(tokenizer, model, row["raw_prompt"])
        prompt_token_count_by_model[model] = len(
            tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )
        candidate_ids_by_model[model] = {
            value: [int(token) for token in tokenizer(value, add_special_tokens=False)["input_ids"]]
            for value in row["all_candidates"]
        }
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "prompt_token_count_by_model": prompt_token_count_by_model,
        "candidate_token_ids_by_model": candidate_ids_by_model,
        "semantic_case_id": (
            f"{row['anchor_id']}_binding{row['binding']}_query{row['query_object_index']}_"
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
            target_rows = sealed_rows if split == "sealed" else open_rows
            target_rows.extend(
                materialize_semantic(tokenizers, row, split) for row in semantic_rows
            )
    return open_rows, sealed_rows


def prior_objects(path: Path) -> set[str]:
    if not path.exists():
        return set()
    values = set()
    for row in read_jsonl(path):
        if row.get("model") in {None, "qwen3"}:
            values.update(row["objects"])
    return values


def audit_rows(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = open_rows + sealed_rows
    failures = []
    expected_open = sum(SPLIT_WORLD_COUNTS[split] for split in OPEN_SPLITS) * 108
    expected_sealed = SPLIT_WORLD_COUNTS["sealed"] * 108
    if len(open_rows) != expected_open:
        failures.append("open_count")
    if len(sealed_rows) != expected_sealed:
        failures.append("sealed_count")
    if len({row["semantic_case_id"] for row in all_rows}) != len(all_rows):
        failures.append("case_id_collision")
    if any(row["target"] == row["other_relation_target"] for row in all_rows):
        failures.append("target_other_relation_collision")
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_triplet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_world[row["anchor_id"]].append(row)
        by_triplet[row["triplet_id"]].append(row)
    world_sizes = sorted({len(group) for group in by_world.values()})
    triplet_sizes = sorted({len(group) for group in by_triplet.values()})
    if world_sizes != [108]:
        failures.append("world_size")
    if triplet_sizes != [3]:
        failures.append("triplet_size")
    triplet_invariant_failures = 0
    for group in by_triplet.values():
        ordered = sorted(group, key=lambda row: row["binding"])
        fixed = (
            "objects", "values", "query_object", "query_relation", "surface_id", "fact_order",
            "other_relation_target", "fact_token_multiset_key", "prompt_token_multiset_key",
        )
        if any(row[key] != ordered[0][key] for row in ordered[1:] for key in fixed):
            triplet_invariant_failures += 1
        elif len({row["target"] for row in ordered}) != 3:
            triplet_invariant_failures += 1
        elif ordered[0]["other_relation_target"] in {row["target"] for row in ordered}:
            triplet_invariant_failures += 1
    if triplet_invariant_failures:
        failures.append("triplet_invariants")
    non_single_token_rows = sum(
        any(
            len(ids) != 1
            for model_ids in row["candidate_token_ids_by_model"].values()
            for ids in model_ids.values()
        )
        for row in all_rows
    )
    token_collision_rows = sum(
        any(len({tuple(ids) for ids in model_ids.values()}) != 4
            for model_ids in row["candidate_token_ids_by_model"].values())
        for row in all_rows
    )
    if non_single_token_rows:
        failures.append("candidate_not_single_token")
    if token_collision_rows:
        failures.append("candidate_token_collision")
    split_objects = {
        split: {
            value
            for row in all_rows
            if row["split"] == split
            for value in row["objects"]
        }
        for split in SPLITS
    }
    split_overlap = 0
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            split_overlap += len(split_objects[left] & split_objects[right])
    if split_overlap:
        failures.append("split_object_overlap")
    phase567_path = ROOT / "tests/gpt5/result/phase567_multi_relation_binding/phase567_open_cases.jsonl"
    phase568_path = ROOT / "tests/gpt5/result/phase568_explicit_relation_binding/phase568_open_cases.jsonl"
    current_objects = set().union(*(split_objects.values()))
    prior_overlap = len(current_objects & (prior_objects(phase567_path) | prior_objects(phase568_path)))
    if prior_overlap:
        failures.append("prior_object_overlap")
    return {
        "schema_version": "phase569_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": not failures,
        "status": "static_pass_no_model_run" if not failures else "static_fail",
        "failures": failures,
        "registered_case_count": len(all_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "registered_semantic_case_count": len(all_rows),
        "open_semantic_case_count": len(open_rows),
        "sealed_semantic_case_count": len(sealed_rows),
        "registered_model_evaluation_count": len(all_rows) * len(MODELS),
        "open_model_evaluation_count": len(open_rows) * len(MODELS),
        "sealed_model_evaluation_count": len(sealed_rows) * len(MODELS),
        "open_case_count_by_model": {model: len(open_rows) for model in MODELS},
        "rows_per_world": world_sizes,
        "rows_per_triplet": triplet_sizes,
        "triplet_count": len(by_triplet),
        "triplet_invariant_failure_count": triplet_invariant_failures,
        "target_other_relation_collision_count": sum(
            row["target"] == row["other_relation_target"] for row in all_rows
        ),
        "candidate_non_single_token_row_count": non_single_token_rows,
        "candidate_token_collision_row_count": token_collision_rows,
        "split_object_overlap_count": split_overlap,
        "phase567_568_object_overlap_count": prior_overlap,
        "sealed_split_read": False,
        "model_execution_performed": False,
    }


def freeze() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = audit_rows(open_rows, sealed_rows)
    if not audit["valid"]:
        raise RuntimeError(f"Phase569 static audit failed: {audit['failures']}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(SEALED_COMMITMENT_PATH, {
        "schema_version": "phase569_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_semantic_case_count": len(sealed_rows),
        "sealed_model_evaluation_count": len(sealed_rows) * len(MODELS),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    })
    protocol = {
        "schema_version": "phase569_frozen_protocol.v1",
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
        "registered_semantic_case_count": len(open_rows) + len(sealed_rows),
        "open_semantic_case_count": len(open_rows),
        "sealed_semantic_case_count": len(sealed_rows),
        "registered_model_evaluation_count": (
            len(open_rows) + len(sealed_rows)
        ) * len(MODELS),
        "open_model_evaluation_count": len(open_rows) * len(MODELS),
        "sealed_model_evaluation_count": len(sealed_rows) * len(MODELS),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "phenotype_gate": {
            "stable_correct_cell_accuracy_min": 0.95,
            "stable_confusion_cell_rate_min": 0.20,
            "stable_confusion_share_of_registered_errors_min": 0.80,
            "stable_confusion_cell_accuracy_max": 0.75,
            "minimum_path_cases_per_phenotype_per_split": 64,
            "minimum_path_distinct_cells_per_phenotype": 2,
            "minimum_path_target_other_pairs_per_phenotype": 8,
        },
        "evidence_policy": {
            "new_objects_after_phase568": True,
            "new_value_labels_after_phase568": True,
            "new_surface_templates_after_phase568": True,
            "four_values_remove_target_other_relation_collision": True,
            "behavior_failure_is_new_phenotype_not_phase568_reclassification": True,
            "phenotype_discovery_and_confirmation_both_required": True,
            "path_splits_are_independent_from_phenotype_cell_selection": True,
            "single_token_candidates_required_for_internal_margin": True,
            "whole_world_rate_is_report_only": True,
            "no_fine_scan_before_coarse_event_trace": True,
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
        "semantic_registered": protocol["registered_semantic_case_count"],
        "semantic_open": protocol["open_semantic_case_count"],
        "semantic_sealed": protocol["sealed_semantic_case_count"],
        "model_evaluations_registered": protocol["registered_model_evaluation_count"],
        "model_evaluations_open": protocol["open_model_evaluation_count"],
        "model_evaluations_sealed": protocol["sealed_model_evaluation_count"],
        "per_model_open": audit["open_case_count_by_model"],
        "target_other_collision": audit["target_other_relation_collision_count"],
        "valid": audit["valid"],
    }, ensure_ascii=False, indent=2))
    return protocol


if __name__ == "__main__":
    freeze()
