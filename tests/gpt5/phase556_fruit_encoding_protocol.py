#!/usr/bin/env python3
"""Freeze the Phase556 single-category reuse/difference protocol.

The open protocol separates four factors in controlled local worlds and keeps
natural parametric fruit knowledge in a separate ledger.  Sealed rows are
written once to a private file and exposed only through a commitment.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase556"
SCHEMA_VERSION = "phase556_fruit_encoding.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "independent_confirmation", "sealed")
OPEN_SPLITS = SPLITS[:2]
WORLDS_PER_SPLIT = 96
FACTORS = ("entity", "category", "query", "binding")
CELLS = tuple(
    f"entity{entity}_category{category}_query{query}_binding{binding}"
    for entity, category, query, binding in itertools.product((0, 1), repeat=4)
)
NATURAL_RELATIONS = ("category", "subclass", "color", "taste", "part")
NATURAL_SURFACES = tuple(range(4))
NATURAL_ORDERS = (0, 1)

OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
OPEN_CASES_PATH = OUT_DIR / "phase556_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase556_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase556_sealed_commitment.json"
FACT_BANK_PATH = OUT_DIR / "phase556_public_fact_bank.json"
PROTOCOL_PATH = OUT_DIR / "phase556_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase556_static_audit.json"

CONTROL_CATEGORIES = ("tool", "vegetable", "mineral", "instrument", "furniture", "animal")
ATTRIBUTE_POOLS = {
    "color": ("red", "yellow", "green", "purple", "orange", "blue", "white", "black"),
    "taste": ("sweet", "sour", "bitter", "mild", "tart", "savory", "earthy", "plain"),
    "part": ("core", "peel", "pit", "rind", "stem", "skin", "seed", "pulp"),
}

SYLLABLE_A = ("ba", "ce", "di", "fo", "ga", "hu", "ji", "ke", "lu", "mi", "no", "pa", "ri")
SYLLABLE_B = ("lan", "mer", "tin", "vor", "sen", "dak", "pel", "rin", "sol", "wen", "yas", "kor", "zen")


NATURAL_OBJECTS: tuple[dict[str, Any], ...] = (
    # Discovery: one fruit from every frozen subclass and four matched controls.
    {"split": "discovery", "id": "apple", "label": "apple", "is_fruit": True, "category": ["fruit"], "subclass": ["tree fruit", "pome"], "color": ["red", "green"], "taste": ["sweet", "tart"], "part": ["core", "skin"]},
    {"split": "discovery", "id": "orange", "label": "orange", "is_fruit": True, "category": ["fruit"], "subclass": ["citrus"], "color": ["orange"], "taste": ["sweet", "tart"], "part": ["peel", "rind"]},
    {"split": "discovery", "id": "strawberry", "label": "strawberry", "is_fruit": True, "category": ["fruit"], "subclass": ["berry"], "color": ["red"], "taste": ["sweet", "tart"], "part": ["seed", "seeds"]},
    {"split": "discovery", "id": "banana", "label": "banana", "is_fruit": True, "category": ["fruit"], "subclass": ["tropical fruit", "tropical"], "color": ["yellow"], "taste": ["sweet"], "part": ["peel", "skin"]},
    {"split": "discovery", "id": "peach", "label": "peach", "is_fruit": True, "category": ["fruit"], "subclass": ["stone fruit", "drupe"], "color": ["pink", "orange", "yellow"], "taste": ["sweet"], "part": ["pit", "stone"]},
    {"split": "discovery", "id": "watermelon", "label": "watermelon", "is_fruit": True, "category": ["fruit"], "subclass": ["melon"], "color": ["green", "red"], "taste": ["sweet"], "part": ["rind", "seed", "seeds"]},
    {"split": "discovery", "id": "carrot", "label": "carrot", "is_fruit": False, "category": ["vegetable"], "subclass": ["root vegetable"], "color": ["orange"], "taste": ["earthy", "sweet"], "part": ["root", "stem"]},
    {"split": "discovery", "id": "bread", "label": "bread", "is_fruit": False, "category": ["food"], "subclass": ["baked food"], "color": ["brown", "white"], "taste": ["savory", "mild"], "part": ["crust", "crumb"]},
    {"split": "discovery", "id": "ball", "label": "ball", "is_fruit": False, "category": ["object"], "subclass": ["toy"], "color": ["varied"], "taste": ["inedible"], "part": ["surface", "cover"]},
    {"split": "discovery", "id": "canary", "label": "canary", "is_fruit": False, "category": ["animal", "bird"], "subclass": ["bird"], "color": ["yellow"], "taste": ["inedible"], "part": ["feather", "wing"]},
    # Independent confirmation.
    {"split": "independent_confirmation", "id": "pear", "label": "pear", "is_fruit": True, "category": ["fruit"], "subclass": ["tree fruit", "pome"], "color": ["green", "yellow"], "taste": ["sweet"], "part": ["core", "skin"]},
    {"split": "independent_confirmation", "id": "lemon", "label": "lemon", "is_fruit": True, "category": ["fruit"], "subclass": ["citrus"], "color": ["yellow"], "taste": ["sour", "tart"], "part": ["peel", "rind"]},
    {"split": "independent_confirmation", "id": "blueberry", "label": "blueberry", "is_fruit": True, "category": ["fruit"], "subclass": ["berry"], "color": ["blue", "purple"], "taste": ["sweet", "tart"], "part": ["skin", "seed"]},
    {"split": "independent_confirmation", "id": "mango", "label": "mango", "is_fruit": True, "category": ["fruit"], "subclass": ["tropical fruit", "tropical"], "color": ["yellow", "orange", "green"], "taste": ["sweet"], "part": ["pit", "skin"]},
    {"split": "independent_confirmation", "id": "cherry", "label": "cherry", "is_fruit": True, "category": ["fruit"], "subclass": ["stone fruit", "drupe"], "color": ["red"], "taste": ["sweet", "tart"], "part": ["pit", "stem"]},
    {"split": "independent_confirmation", "id": "cantaloupe", "label": "cantaloupe", "is_fruit": True, "category": ["fruit"], "subclass": ["melon"], "color": ["orange", "tan"], "taste": ["sweet"], "part": ["rind", "seed", "seeds"]},
    {"split": "independent_confirmation", "id": "potato", "label": "potato", "is_fruit": False, "category": ["vegetable"], "subclass": ["tuber"], "color": ["brown", "yellow"], "taste": ["earthy", "starchy"], "part": ["skin", "eye"]},
    {"split": "independent_confirmation", "id": "rice", "label": "rice", "is_fruit": False, "category": ["food", "grain"], "subclass": ["grain"], "color": ["white", "brown"], "taste": ["mild", "plain"], "part": ["grain", "husk"]},
    {"split": "independent_confirmation", "id": "marble", "label": "marble", "is_fruit": False, "category": ["object", "toy"], "subclass": ["toy"], "color": ["varied"], "taste": ["inedible"], "part": ["surface"]},
    {"split": "independent_confirmation", "id": "tiger", "label": "tiger", "is_fruit": False, "category": ["animal", "mammal"], "subclass": ["mammal", "cat"], "color": ["orange", "black"], "taste": ["inedible"], "part": ["stripe", "fur"]},
    # Sealed stress set.
    {"split": "sealed", "id": "quince", "label": "quince", "is_fruit": True, "category": ["fruit"], "subclass": ["tree fruit", "pome"], "color": ["yellow"], "taste": ["tart"], "part": ["core", "skin"]},
    {"split": "sealed", "id": "lime", "label": "lime", "is_fruit": True, "category": ["fruit"], "subclass": ["citrus"], "color": ["green"], "taste": ["sour", "tart"], "part": ["peel", "rind"]},
    {"split": "sealed", "id": "raspberry", "label": "raspberry", "is_fruit": True, "category": ["fruit"], "subclass": ["berry"], "color": ["red"], "taste": ["sweet", "tart"], "part": ["seed", "seeds"]},
    {"split": "sealed", "id": "pineapple", "label": "pineapple", "is_fruit": True, "category": ["fruit"], "subclass": ["tropical fruit", "tropical"], "color": ["yellow", "brown"], "taste": ["sweet", "tart"], "part": ["rind", "crown"]},
    {"split": "sealed", "id": "plum", "label": "plum", "is_fruit": True, "category": ["fruit"], "subclass": ["stone fruit", "drupe"], "color": ["purple", "red"], "taste": ["sweet", "tart"], "part": ["pit", "skin"]},
    {"split": "sealed", "id": "honeydew", "label": "honeydew", "is_fruit": True, "category": ["fruit"], "subclass": ["melon"], "color": ["green"], "taste": ["sweet"], "part": ["rind", "seed", "seeds"]},
    {"split": "sealed", "id": "cucumber", "label": "cucumber", "is_fruit": False, "category": ["vegetable"], "subclass": ["gourd", "vegetable"], "color": ["green"], "taste": ["mild"], "part": ["skin", "seed"]},
    {"split": "sealed", "id": "cheese", "label": "cheese", "is_fruit": False, "category": ["food", "dairy"], "subclass": ["dairy"], "color": ["yellow", "white"], "taste": ["savory"], "part": ["rind"]},
    {"split": "sealed", "id": "traffic_cone", "label": "traffic cone", "is_fruit": False, "category": ["object"], "subclass": ["safety device"], "color": ["orange"], "taste": ["inedible"], "part": ["base", "surface"]},
    {"split": "sealed", "id": "frog", "label": "frog", "is_fruit": False, "category": ["animal", "amphibian"], "subclass": ["amphibian"], "color": ["green"], "taste": ["inedible"], "part": ["leg", "skin"]},
)


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
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pseudo_word(index: int, split: str) -> str:
    split_offset = {"discovery": 0, "independent_confirmation": 2000, "sealed": 4000}[split]
    shifted = index + split_offset
    a = SYLLABLE_A[shifted % len(SYLLABLE_A)]
    b = SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
    c = SYLLABLE_A[(shifted * 7 + 5) % len(SYLLABLE_A)]
    return (a + b + c).capitalize()


def cell_factors(cell: str) -> dict[str, int]:
    return {factor: int(cell.split(factor, 1)[1][0]) for factor in FACTORS}


def controlled_world(split: str, world_index: int, cell: str) -> dict[str, Any]:
    factors = cell_factors(cell)
    entity_a = pseudo_word(world_index * 2, split)
    entity_b = pseudo_word(world_index * 2 + 1, split)
    control_category = CONTROL_CATEGORIES[world_index % len(CONTROL_CATEGORIES)]
    categories = ("fruit", control_category)
    relation = tuple(ATTRIBUTE_POOLS)[world_index % len(ATTRIBUTE_POOLS)]
    pool = ATTRIBUTE_POOLS[relation]
    value_a = pool[(world_index * 3 + 1) % len(pool)]
    value_b = pool[(world_index * 5 + 4) % len(pool)]
    if value_a == value_b:
        value_b = pool[(pool.index(value_b) + 1) % len(pool)]

    category_a, category_b = categories
    if factors["category"]:
        category_a, category_b = category_b, category_a
    bound_a, bound_b = value_a, value_b
    if factors["binding"]:
        bound_a, bound_b = bound_b, bound_a
    selected = entity_a if factors["entity"] == 0 else entity_b
    selected_category = category_a if factors["entity"] == 0 else category_b
    selected_value = bound_a if factors["entity"] == 0 else bound_b
    query_kind = "category" if factors["query"] == 0 else "attribute"
    target = selected_category if query_kind == "category" else selected_value

    facts = [
        f"{entity_a} belongs to category {category_a}",
        f"{entity_b} belongs to category {category_b}",
        f"the {relation} of {entity_a} is {bound_a}",
        f"the {relation} of {entity_b} is {bound_b}",
    ]
    order = world_index % 2
    if order:
        facts = list(reversed(facts))
    surface = world_index % 4
    if surface == 0:
        context = ". ".join(facts) + "."
        question = (
            f"What category is assigned to {selected}?"
            if query_kind == "category"
            else f"What {relation} is assigned to {selected}?"
        )
    elif surface == 1:
        context = "Local record: " + "; ".join(facts) + "."
        question = (
            f"Return the category recorded for {selected}."
            if query_kind == "category"
            else f"Return the recorded {relation} for {selected}."
        )
    elif surface == 2:
        context = "Use only these stated facts. " + ". ".join(facts) + "."
        question = (
            f"Lookup category({selected})."
            if query_kind == "category"
            else f"Lookup {relation}({selected})."
        )
    else:
        context = "Fact ledger:\n- " + "\n- ".join(facts)
        question = (
            f"Requested field for {selected}: category."
            if query_kind == "category"
            else f"Requested field for {selected}: {relation}."
        )
    raw_prompt = f"{context}\n{question}\nReply with exactly one answer word and no explanation."
    all_candidates = sorted({category_a, category_b, value_a, value_b})
    return {
        "raw_prompt": raw_prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": [value for value in all_candidates if value != target],
        "all_candidates": all_candidates,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "selected_entity": selected,
        "category_a": category_a,
        "category_b": category_b,
        "attribute_relation": relation,
        "attribute_a": bound_a,
        "attribute_b": bound_b,
        "query_kind": query_kind,
        "surface_id": surface,
        "fact_order": order,
        "factor_values": factors,
        "fact_token_multiset_key": stable_hash(sorted(" ".join(facts).lower().split())),
        "semantic_fragments": {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "category_a": category_a,
            "category_b": category_b,
            "attribute_a": bound_a,
            "attribute_b": bound_b,
            "selected_entity": selected,
            "query_relation": "category" if query_kind == "category" else relation,
        },
    }


def relation_phrase(relation: str) -> str:
    return {
        "category": "category",
        "subclass": "subclass",
        "color": "usual color",
        "taste": "usual taste",
        "part": "notable part",
    }[relation]


def natural_prompt(obj: dict[str, Any], relation: str, surface: int, order: int) -> str:
    label = obj["label"]
    phrase = relation_phrase(relation)
    if surface == 0:
        body = f"Use common everyday knowledge. What is the {phrase} of a {label}?"
    elif surface == 1:
        body = f"Complete this familiar fact: the {phrase} of a {label} is"
    elif surface == 2:
        body = f"Knowledge check about {label}: give its {phrase}."
    else:
        body = f"Object: {label}. Requested field: {phrase}."
    if order:
        reversed_surfaces = {
            0: f"Requested field: {phrase}. Object: {label}. Use common everyday knowledge.",
            1: f"Blank to complete: {phrase}. Subject: {label}. Supply the familiar fact.",
            2: f"Field first: {phrase}. Knowledge-check object: {label}.",
            3: f"Requested={phrase}; object={label}; source=everyday knowledge.",
        }
        body = reversed_surfaces[surface]
    return body + " Reply with only one short answer and no explanation."


def natural_distractors(obj: dict[str, Any], relation: str) -> list[str]:
    candidates: list[str] = []
    target = {value.casefold() for value in obj[relation]}
    for other in NATURAL_OBJECTS:
        if other["split"] != obj["split"] or other["id"] == obj["id"]:
            continue
        for value in other[relation]:
            if value.casefold() not in target and value not in candidates:
                candidates.append(value)
    return candidates[:4]


def materialize_row(model: str, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    target_token_ids = [int(value) for value in tokenizer(row["target"], add_special_tokens=False)["input_ids"]]
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "target_first_token_id_private": target_token_ids[0] if target_token_ids else None,
        "strict_expected": row["target"],
        "sealed": row["split"] == "sealed",
        "behavior_only": True,
        "observer": False,
        "causal": False,
        "single_neuron": False,
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for split in SPLITS:
            destination = sealed_rows if split == "sealed" else open_rows
            for world_index in range(WORLDS_PER_SPLIT):
                anchor_id = f"phase556_controlled_{split}_{world_index:03d}"
                for cell in CELLS:
                    spec = controlled_world(split, world_index, cell)
                    row = materialize_row(model, tokenizer, {
                        "case_id": f"{anchor_id}_{model}_{cell}",
                        "anchor_id": anchor_id,
                        "case_type": "controlled_factorial",
                        "split": split,
                        "world_index": world_index,
                        "factorial_cell": cell,
                        **spec,
                    })
                    destination.append(row)
            for obj in (item for item in NATURAL_OBJECTS if item["split"] == split):
                for relation in NATURAL_RELATIONS:
                    for surface in NATURAL_SURFACES:
                        for order in NATURAL_ORDERS:
                            anchor_id = f"phase556_natural_{split}_{obj['id']}_{relation}"
                            aliases = list(obj[relation])
                            row = materialize_row(model, tokenizer, {
                                "case_id": f"{anchor_id}_{model}_surface{surface}_order{order}",
                                "anchor_id": anchor_id,
                                "case_type": "natural_knowledge",
                                "split": split,
                                "object_id": obj["id"],
                                "object_label": obj["label"],
                                "is_fruit": obj["is_fruit"],
                                "natural_relation": relation,
                                "surface_id": surface,
                                "fact_order": order,
                                "raw_prompt": natural_prompt(obj, relation, surface, order),
                                "target": aliases[0],
                                "target_aliases": aliases,
                                "distractors": natural_distractors(obj, relation),
                                "all_candidates": aliases + natural_distractors(obj, relation),
                                "semantic_fragments": {
                                    "selected_entity": obj["label"],
                                    "query_relation": relation,
                                },
                            })
                            destination.append(row)
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = open_rows + sealed_rows
    controlled = [row for row in all_rows if row["case_type"] == "controlled_factorial"]
    natural = [row for row in all_rows if row["case_type"] == "natural_knowledge"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in controlled:
        groups[(row["model"], row["anchor_id"])].append(row)
    factor_errors = 0
    for group in groups.values():
        if {row["factorial_cell"] for row in group} != set(CELLS):
            factor_errors += 1
            continue
        if len(group) != 16:
            factor_errors += 1
    natural_by_split = Counter((row["model"], row["split"]) for row in natural)
    audit = {
        "schema_version": "phase556_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(all_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "controlled_case_count": len(controlled),
        "natural_case_count": len(natural),
        "model_case_counts": dict(Counter(row["model"] for row in all_rows)),
        "open_model_case_counts": dict(Counter(row["model"] for row in open_rows)),
        "sealed_model_case_counts": dict(Counter(row["model"] for row in sealed_rows)),
        "controlled_anchor_count": len(groups),
        "controlled_rows_per_anchor": sorted({len(group) for group in groups.values()}),
        "natural_case_counts_by_model_split": {
            f"{model}:{split}": natural_by_split[(model, split)]
            for model in MODELS for split in SPLITS
        },
        "factorial_error_count": factor_errors,
        "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
        "duplicate_model_prompt_count": len(all_rows) - len({(row["model"], row["prompt"]) for row in all_rows}),
        "missing_target_token_count": sum(row["target_first_token_id_private"] is None for row in all_rows),
        "max_prompt_token_count": max(row["prompt_token_count"] for row in all_rows),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    audit["valid"] = bool(
        audit["registered_case_count"] == 17424
        and audit["open_case_count"] == 11616
        and audit["sealed_case_count"] == 5808
        and audit["model_case_counts"] == {model: 5808 for model in MODELS}
        and audit["open_model_case_counts"] == {model: 3872 for model in MODELS}
        and audit["sealed_model_case_counts"] == {model: 1936 for model in MODELS}
        and audit["controlled_rows_per_anchor"] == [16]
        and audit["max_prompt_token_count"] <= 512
        and all(audit[key] == 0 for key in (
            "factorial_error_count", "duplicate_case_id_count", "duplicate_model_prompt_count",
            "missing_target_token_count", "open_contains_sealed_count", "sealed_flag_missing_count",
        ))
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = validate(open_rows, sealed_rows)
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    commitment = {
        "schema_version": "phase556_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, commitment)
    public_fact_bank = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now(),
        "open_objects": [row for row in NATURAL_OBJECTS if row["split"] in OPEN_SPLITS],
        "sealed_object_count": sum(row["split"] == "sealed" for row in NATURAL_OBJECTS),
        "controlled_relations": list(ATTRIBUTE_POOLS),
        "factor_names": list(FACTORS),
    }
    write_json(FACT_BANK_PATH, public_fact_bank)
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Single-category fruit reuse, difference, attribute, and binding recovery",
        "models_in_required_execution_order": list(MODELS),
        "splits": list(SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "worlds_per_split": WORLDS_PER_SPLIT,
        "factor_names": list(FACTORS),
        "factorial_cells": list(CELLS),
        "natural_relations": list(NATURAL_RELATIONS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_gate": {
            "controlled_all_16_anchor_rate_min_per_open_split": 0.80,
            "controlled_each_cell_accuracy_min_per_open_split": 0.90,
            "controlled_unrecoverable_rate_max_per_open_split": 0.05,
            "natural_relation_accuracy_min_per_open_split": 0.80,
            "natural_surface_accuracy_min": 0.70,
            "model_specific_qualification": True,
        },
        "internal_gate": {
            "behavior_pass_required": True,
            "full_layer_direction_preserving_ledger": True,
            "factor_ledgers": list(FACTORS),
            "causal_search_starts_at_terminal_semantic_event": True,
            "head_channel_neuron_scan_before_path_gate": False,
        },
        "evidence_policy": {
            "open_cases_only_before_sealed_authorization": True,
            "sealed_cases_path_not_exposed": True,
            "sealed_split_read": False,
            "similarity_is_not_causality": True,
            "observer_is_not_compute_edge": True,
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
