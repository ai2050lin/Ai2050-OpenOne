#!/usr/bin/env python3
"""Freeze the Phase576 natural fruit reuse/difference denominator.

The protocol deliberately contains no internal-coordinate or intervention
choice.  It first asks whether each model has a stable natural-knowledge
denominator for a shared category and an object-varying visible attribute.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase576"
MODELS = ("qwen3", "glm4", "deepseek7b")
STRUCTURE_SPLITS = (
    "structure_discovery",
    "structure_confirmation",
    "heldout_objects",
)
CAUSAL_SPLITS = ("causal_discovery", "causal_confirmation")
SEALED_SPLIT = "sealed"
SPLITS = STRUCTURE_SPLITS + CAUSAL_SPLITS + (SEALED_SPLIT,)
RELATIONS = ("category", "outer_color")
SURFACES_PER_SPLIT = 8
NOOP_REPEATS = ("noop1", "noop2")
FIXED_BATCH_SIZE = 32
MIN_STABLE_SURFACES_PER_RELATION = 6
MIN_QUALIFIED_FRUITS_PER_SPLIT = 8
MIN_QUALIFIED_CONTROLS_PER_SPLIT = 3
TRACE_PAIRS_PER_SPLIT = 48

OUT_DIR = ROOT / "tests/gpt5/result/phase576_natural_fruit"
OPEN_CASES_PATH = OUT_DIR / "phase576_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase576_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase576_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase576_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase576_static_audit.json"
PUBLIC_OBJECTS_PATH = OUT_DIR / "phase576_public_object_bank.json"


def obj(
    object_id: str,
    label: str,
    kind: str,
    colors: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "label": label,
        "kind": kind,
        "is_fruit": kind == "fruit",
        "category_aliases": [kind],
        "outer_color_aliases": list(colors),
    }


OBJECT_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "A": (
        obj("apple", "apple", "fruit", ("red", "green", "yellow")),
        obj("banana", "banana", "fruit", ("yellow",)),
        obj("orange", "orange", "fruit", ("orange",)),
        obj("lemon", "lemon", "fruit", ("yellow",)),
        obj("strawberry", "strawberry", "fruit", ("red",)),
        obj("blueberry", "blueberry", "fruit", ("blue",)),
        obj("cherry", "cherry", "fruit", ("red",)),
        obj("pear", "pear", "fruit", ("green", "yellow")),
        obj("grape", "grape", "fruit", ("purple", "green", "red")),
        obj("watermelon", "watermelon", "fruit", ("green",)),
        obj("carrot", "carrot", "vegetable", ("orange",)),
        obj("broccoli", "broccoli", "vegetable", ("green",)),
        obj("beet", "beet", "vegetable", ("red", "purple")),
        obj("eggplant", "eggplant", "vegetable", ("purple",)),
    ),
    "B": (
        obj("lime", "lime", "fruit", ("green",)),
        obj("raspberry", "raspberry", "fruit", ("red",)),
        obj("blackberry", "blackberry", "fruit", ("black", "purple")),
        obj("plum", "plum", "fruit", ("purple", "red")),
        obj("peach", "peach", "fruit", ("orange", "pink", "yellow")),
        obj("pineapple", "pineapple", "fruit", ("brown", "yellow")),
        obj("kiwi", "kiwi", "fruit", ("brown",)),
        obj("coconut", "coconut", "fruit", ("brown",)),
        obj("pomegranate", "pomegranate", "fruit", ("red",)),
        obj("cantaloupe", "cantaloupe", "fruit", ("tan",)),
        obj("potato", "potato", "vegetable", ("brown",)),
        obj("celery", "celery", "vegetable", ("green",)),
        obj("radish", "radish", "vegetable", ("red",)),
        obj("cauliflower", "cauliflower", "vegetable", ("white",)),
    ),
    "C": (
        obj("mango", "mango", "fruit", ("green", "yellow", "red")),
        obj("papaya", "papaya", "fruit", ("green", "yellow")),
        obj("avocado", "avocado", "fruit", ("green",)),
        obj("apricot", "apricot", "fruit", ("orange",)),
        obj("cranberry", "cranberry", "fruit", ("red",)),
        obj("grapefruit", "grapefruit", "fruit", ("yellow", "pink")),
        obj("guava", "guava", "fruit", ("green", "yellow")),
        obj("nectarine", "nectarine", "fruit", ("red", "orange")),
        obj("tangerine", "tangerine", "fruit", ("orange",)),
        obj("honeydew", "honeydew", "fruit", ("green",)),
        obj("spinach", "spinach", "vegetable", ("green",)),
        obj("onion", "onion", "vegetable", ("yellow", "white")),
        obj("turnip", "turnip", "vegetable", ("white", "purple")),
        obj("cabbage", "cabbage", "vegetable", ("green", "purple")),
    ),
    "D": (
        obj("fig", "fig", "fruit", ("purple", "green")),
        obj("date", "date", "fruit", ("brown",)),
        obj("lychee", "lychee", "fruit", ("red", "pink")),
        obj("persimmon", "persimmon", "fruit", ("orange",)),
        obj("passionfruit", "passion fruit", "fruit", ("purple", "yellow")),
        obj("dragonfruit", "dragon fruit", "fruit", ("pink", "red")),
        obj("gooseberry", "gooseberry", "fruit", ("green",)),
        obj("mulberry", "mulberry", "fruit", ("black", "purple")),
        obj("boysenberry", "boysenberry", "fruit", ("purple",)),
        obj("starfruit", "star fruit", "fruit", ("yellow",)),
        obj("lettuce", "lettuce", "vegetable", ("green",)),
        obj("asparagus", "asparagus", "vegetable", ("green",)),
        obj("pumpkin", "pumpkin", "vegetable", ("orange",)),
        obj("sweet_potato", "sweet potato", "vegetable", ("orange", "brown")),
    ),
}


SPLIT_GROUP = {
    "structure_discovery": "A",
    "structure_confirmation": "A",
    "heldout_objects": "B",
    "causal_discovery": "C",
    "causal_confirmation": "C",
    "sealed": "D",
}


SURFACE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("object_first", "Object: {object}. Requested field: {field}."),
    ("object_first", "Item: {object}. Give its {field}."),
    ("object_first", "Subject = {object}; report the {field}."),
    ("object_first", "Consider {object}. What is its {field}?"),
    ("relation_first", "Requested field: {field}. Object: {object}."),
    ("relation_first", "Give the {field} of {object}."),
    ("relation_first", "{field} lookup for {object}:"),
    ("relation_first", "Field = {field}; subject = {object}."),
    ("object_first", "Knowledge object {object}. Supply the {field}."),
    ("object_first", "For {object}, return its {field}."),
    ("object_first", "Entry {object}; requested property: {field}."),
    ("object_first", "Think about {object}. Name the {field}."),
    ("relation_first", "The requested property is {field}; the item is {object}."),
    ("relation_first", "What is the {field} for the item {object}?"),
    ("relation_first", "Property {field}; knowledge subject {object}."),
    ("relation_first", "Look up {field}({object})."),
    ("object_first", "Everyday-knowledge subject: {object}. Answer its {field}."),
    ("object_first", "Take the item {object}; identify the {field}."),
    ("object_first", "Record about {object}: fill the {field}."),
    ("object_first", "The object is {object}. State the {field}."),
    ("relation_first", "First identify the {field}; target object: {object}."),
    ("relation_first", "Question field {field}. The subject is {object}."),
    ("relation_first", "For the property {field}, answer about {object}."),
    ("relation_first", "Complete this knowledge slot: {field} of {object}."),
)


SPLIT_SURFACES = {
    "structure_discovery": tuple(range(0, 8)),
    "structure_confirmation": tuple(range(8, 16)),
    "heldout_objects": tuple(range(16, 24)),
    "causal_discovery": tuple(range(0, 8)),
    "causal_confirmation": tuple(range(8, 16)),
    "sealed": tuple(range(16, 24)),
}


FIELD_PHRASES = {
    "category": (
        "everyday broad food category",
        "ordinary food category",
        "common broad category",
        "everyday kind of food",
    ),
    "outer_color": (
        "typical outer color",
        "usual visible color",
        "common exterior color",
        "typical outside color",
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def surface_for(split: str, surface_id: int) -> tuple[str, str]:
    if surface_id not in SPLIT_SURFACES[split]:
        raise ValueError(f"surface {surface_id} is not frozen for {split}")
    return SURFACE_TEMPLATES[surface_id]


def field_phrase(relation: str, surface_id: int) -> str:
    phrases = FIELD_PHRASES[relation]
    return phrases[surface_id % len(phrases)]


def aliases_for(item: dict[str, Any], relation: str) -> list[str]:
    return list(item[f"{relation}_aliases"])


def prompt_for(item: dict[str, Any], relation: str, split: str, surface_id: int) -> tuple[str, str, str]:
    order, template = surface_for(split, surface_id)
    phrase = field_phrase(relation, surface_id)
    body = template.format(object=item["label"], field=phrase)
    raw_prompt = (
        "Use common everyday knowledge, not a fictional definition. "
        + body
        + " Reply with one answer word only and no explanation."
    )
    return raw_prompt, order, phrase


def candidate_bank() -> list[str]:
    candidates = {"fruit", "vegetable"}
    for group in OBJECT_GROUPS.values():
        for item in group:
            candidates.update(item["outer_color_aliases"])
    return sorted(candidates, key=lambda value: (-len(value), value))


def materialize_row(
    tokenizers: dict[str, Any],
    item: dict[str, Any],
    relation: str,
    split: str,
    surface_id: int,
) -> dict[str, Any]:
    raw_prompt, order, phrase = prompt_for(item, relation, split, surface_id)
    target_aliases = aliases_for(item, relation)
    other_relation = "outer_color" if relation == "category" else "category"
    other_aliases = aliases_for(item, other_relation)
    all_candidates = candidate_bank()
    token_ids: dict[str, dict[str, list[int]]] = {}
    prompt_counts: dict[str, int] = {}
    for model, tokenizer in tokenizers.items():
        prompt = render_chat(tokenizer, model, raw_prompt)
        prompt_counts[model] = len(
            tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )
        token_ids[model] = {
            candidate: [
                int(token)
                for token in tokenizer(candidate, add_special_tokens=False)["input_ids"]
            ]
            for candidate in all_candidates
        }
    pair_id = f"phase576_{split}_{item['object_id']}_surface{surface_id:02d}"
    return {
        "schema_version": "phase576_natural_fruit_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": f"{pair_id}_{relation}",
        "pair_id": pair_id,
        "split": split,
        "object_group": SPLIT_GROUP[split],
        "object_id": item["object_id"],
        "object_label": item["label"],
        "is_fruit": item["is_fruit"],
        "object_category": item["kind"],
        "relation": relation,
        "other_relation": other_relation,
        "surface_id": surface_id,
        "surface_order": order,
        "field_phrase": phrase,
        "raw_prompt": raw_prompt,
        "target": target_aliases[0],
        "target_aliases": target_aliases,
        "other_relation_aliases": other_aliases,
        "all_candidates": all_candidates,
        "candidate_token_ids_by_model": token_ids,
        "prompt_token_count_by_model": prompt_counts,
        "semantic_fragments": {
            "object": item["label"],
            "relation": phrase,
        },
        "natural_parametric_knowledge": True,
        "context_fact_supplied": False,
        "observer": False,
        "causal": False,
        "sealed": split == SEALED_SPLIT,
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        destination = sealed_rows if split == SEALED_SPLIT else open_rows
        for item in OBJECT_GROUPS[SPLIT_GROUP[split]]:
            for surface_id in SPLIT_SURFACES[split]:
                for relation in RELATIONS:
                    destination.append(
                        materialize_row(tokenizers, item, relation, split, surface_id)
                    )
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    group_objects = {
        group: {item["object_id"] for item in items}
        for group, items in OBJECT_GROUPS.items()
    }
    cross_group_overlap = sum(
        len(group_objects[left] & group_objects[right])
        for index, left in enumerate(sorted(group_objects))
        for right in sorted(group_objects)[index + 1:]
    )
    expected_per_split = 14 * len(RELATIONS) * SURFACES_PER_SPLIT
    audit = {
        "schema_version": "phase576_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "object_count_by_group": {key: len(value) for key, value in group_objects.items()},
        "fruit_count_by_group": {
            key: sum(item["is_fruit"] for item in value)
            for key, value in OBJECT_GROUPS.items()
        },
        "control_count_by_group": {
            key: sum(not item["is_fruit"] for item in value)
            for key, value in OBJECT_GROUPS.items()
        },
        "pair_count": len(by_pair),
        "incomplete_pair_count": sum(
            {row["relation"] for row in pair} != set(RELATIONS)
            for pair in by_pair.values()
        ),
        "pair_row_count_values": sorted({len(pair) for pair in by_pair.values()}),
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows) - len(
            {(row["split"], row["raw_prompt"]) for row in rows}
        ),
        "cross_group_object_overlap_count": cross_group_overlap,
        "target_other_alias_overlap_count": sum(
            bool(
                {value.casefold() for value in row["target_aliases"]}
                & {value.casefold() for value in row["other_relation_aliases"]}
            )
            for row in rows
        ),
        "missing_candidate_token_count": sum(
            not token_ids
            for row in rows
            for model_map in row["candidate_token_ids_by_model"].values()
            for token_ids in model_map.values()
        ),
        "max_prompt_token_count": max(
            count for row in rows for count in row["prompt_token_count_by_model"].values()
        ),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
        "expected_case_count_per_split": expected_per_split,
    }
    audit["valid"] = bool(
        audit["registered_case_count"] == expected_per_split * len(SPLITS)
        and audit["open_case_count"] == expected_per_split * (len(SPLITS) - 1)
        and audit["sealed_case_count"] == expected_per_split
        and set(audit["case_count_by_split"].values()) == {expected_per_split}
        and audit["object_count_by_group"] == {key: 14 for key in OBJECT_GROUPS}
        and audit["fruit_count_by_group"] == {key: 10 for key in OBJECT_GROUPS}
        and audit["control_count_by_group"] == {key: 4 for key in OBJECT_GROUPS}
        and audit["pair_row_count_values"] == [2]
        and audit["max_prompt_token_count"] <= 160
        and all(
            audit[key] == 0
            for key in (
                "incomplete_pair_count",
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
                "cross_group_object_overlap_count",
                "target_other_alias_overlap_count",
                "missing_candidate_token_count",
                "open_contains_sealed_count",
                "sealed_flag_missing_count",
            )
        )
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = validate(open_rows, sealed_rows)
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(
        SEALED_COMMITMENT_PATH,
        {
            "schema_version": "phase576_sealed_commitment.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "sealed_case_count": len(sealed_rows),
            "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
            "sealed_split_read_for_analysis": False,
        },
    )
    write_json(
        PUBLIC_OBJECTS_PATH,
        {
            "schema_version": "phase576_public_object_bank.v1",
            "created_at": now(),
            "open_groups": {key: value for key, value in OBJECT_GROUPS.items() if key != "D"},
            "sealed_group_object_count": len(OBJECT_GROUPS["D"]),
            "relations": list(RELATIONS),
        },
    )
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": "phase576_natural_fruit_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Natural fruit shared-category and object-color physical atlas",
        "models_in_required_execution_order": list(MODELS),
        "splits": list(SPLITS),
        "structure_splits": list(STRUCTURE_SPLITS),
        "causal_splits": list(CAUSAL_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "relations": list(RELATIONS),
        "surfaces_per_split": SURFACES_PER_SPLIT,
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_gate": {
            "minimum_stable_surfaces_per_relation": MIN_STABLE_SURFACES_PER_RELATION,
            "minimum_qualified_fruits_per_structure_split": MIN_QUALIFIED_FRUITS_PER_SPLIT,
            "minimum_qualified_controls_per_structure_split": MIN_QUALIFIED_CONTROLS_PER_SPLIT,
            "trace_pairs_per_structure_split": TRACE_PAIRS_PER_SPLIT,
            "exact_repeat_required": True,
            "model_specific_qualification": True,
        },
        "internal_policy": {
            "full_depth_natural_trace_before_intervention": True,
            "no_layer_component_head_or_neuron_preselection": True,
            "natural_event_must_repeat_in_all_structure_splits": True,
            "causal_operator_defined_only_after_event_freeze": True,
        },
        "evidence_policy": {
            "formulas_describe_measurements_only_after_discovery": True,
            "sealed_split_read": False,
            "phase575_seal_reused": False,
            "strict_mechanism_closure_claimed": False,
        },
        "open_cases_path": str(OPEN_CASES_PATH.relative_to(ROOT)),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_path": str(SEALED_COMMITMENT_PATH.relative_to(ROOT)),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
