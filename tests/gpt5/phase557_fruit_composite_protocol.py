#!/usr/bin/env python3
"""Freeze the Phase557 fruit reuse/difference composite causal protocol."""

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


PHASE = "Phase557"
SCHEMA_VERSION = "phase557_fruit_composite.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = (
    "behavior_discovery",
    "behavior_confirmation",
    "path_discovery",
    "path_confirmation",
    "unseen_recombination",
    "sealed",
)
OPEN_SPLITS = SPLITS[:-1]
NATURAL_SPLITS = (
    "behavior_discovery", "behavior_confirmation", "unseen_recombination", "sealed"
)
WORLDS_PER_SPLIT = 48
FACTORS = ("object", "category", "attribute", "binding")
QUERY_STRATA = ("category", "attribute")
FACTOR_CELLS = tuple(
    f"object{o}_category{c}_attribute{a}_binding{b}"
    for o, c, a, b in itertools.product((0, 1), repeat=4)
)
CELLS = tuple(f"{query}__{cell}" for query in QUERY_STRATA for cell in FACTOR_CELLS)
NATURAL_RELATIONS = ("category", "color")
NATURAL_SURFACES = tuple(range(4))
NATURAL_ORDERS = (0, 1)

OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
OPEN_CASES_PATH = OUT_DIR / "phase557_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase557_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase557_sealed_commitment.json"
FACT_BANK_PATH = OUT_DIR / "phase557_public_fact_bank.json"
PROTOCOL_PATH = OUT_DIR / "phase557_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase557_static_audit.json"

CONTROL_CATEGORIES = ("tool", "vegetable", "mineral", "instrument", "furniture", "animal")
ATTRIBUTE_POOLS = {
    "color": ("red", "green", "blue", "yellow", "purple", "orange", "white", "black"),
    "taste": ("sweet", "sour", "bitter", "mild", "tart", "savory", "earthy", "plain"),
    "texture": ("smooth", "rough", "soft", "firm", "crisp", "silky", "grainy", "waxy"),
    "shape": ("round", "long", "oval", "flat", "curved", "square", "narrow", "wide"),
}
SYLLABLE_A = ("ba", "ce", "di", "fo", "ga", "hu", "ji", "ke", "lu", "mi", "no", "pa", "ri")
SYLLABLE_B = ("lan", "mer", "tin", "vor", "sen", "dak", "pel", "rin", "sol", "wen", "yas", "kor", "zen")


NATURAL_OBJECTS: tuple[dict[str, Any], ...] = (
    {"split": "behavior_discovery", "id": "apple", "label": "apple", "is_fruit": True, "category": "fruit", "color": "red"},
    {"split": "behavior_discovery", "id": "banana", "label": "banana", "is_fruit": True, "category": "fruit", "color": "yellow"},
    {"split": "behavior_discovery", "id": "orange", "label": "orange", "is_fruit": True, "category": "fruit", "color": "orange"},
    {"split": "behavior_discovery", "id": "strawberry", "label": "strawberry", "is_fruit": True, "category": "fruit", "color": "red"},
    {"split": "behavior_discovery", "id": "blueberry", "label": "blueberry", "is_fruit": True, "category": "fruit", "color": "blue"},
    {"split": "behavior_discovery", "id": "lime", "label": "lime", "is_fruit": True, "category": "fruit", "color": "green"},
    {"split": "behavior_discovery", "id": "carrot", "label": "carrot", "is_fruit": False, "category": "vegetable", "color": "orange"},
    {"split": "behavior_discovery", "id": "canary", "label": "canary", "is_fruit": False, "category": "animal", "color": "yellow"},
    {"split": "behavior_confirmation", "id": "pear", "label": "pear", "is_fruit": True, "category": "fruit", "color": "green"},
    {"split": "behavior_confirmation", "id": "lemon", "label": "lemon", "is_fruit": True, "category": "fruit", "color": "yellow"},
    {"split": "behavior_confirmation", "id": "cherry", "label": "cherry", "is_fruit": True, "category": "fruit", "color": "red"},
    {"split": "behavior_confirmation", "id": "plum", "label": "plum", "is_fruit": True, "category": "fruit", "color": "purple"},
    {"split": "behavior_confirmation", "id": "kiwi", "label": "kiwi", "is_fruit": True, "category": "fruit", "color": "green"},
    {"split": "behavior_confirmation", "id": "coconut", "label": "coconut", "is_fruit": True, "category": "fruit", "color": "brown"},
    {"split": "behavior_confirmation", "id": "potato", "label": "potato", "is_fruit": False, "category": "vegetable", "color": "brown"},
    {"split": "behavior_confirmation", "id": "coal", "label": "coal", "is_fruit": False, "category": "mineral", "color": "black"},
    {"split": "unseen_recombination", "id": "raspberry", "label": "raspberry", "is_fruit": True, "category": "fruit", "color": "red"},
    {"split": "unseen_recombination", "id": "pineapple", "label": "pineapple", "is_fruit": True, "category": "fruit", "color": "yellow"},
    {"split": "unseen_recombination", "id": "watermelon", "label": "watermelon", "is_fruit": True, "category": "fruit", "color": "green"},
    {"split": "unseen_recombination", "id": "grape", "label": "grape", "is_fruit": True, "category": "fruit", "color": "purple"},
    {"split": "unseen_recombination", "id": "peach", "label": "peach", "is_fruit": True, "category": "fruit", "color": "pink"},
    {"split": "unseen_recombination", "id": "mango", "label": "mango", "is_fruit": True, "category": "fruit", "color": "yellow"},
    {"split": "unseen_recombination", "id": "cucumber", "label": "cucumber", "is_fruit": False, "category": "vegetable", "color": "green"},
    {"split": "unseen_recombination", "id": "marble", "label": "marble", "is_fruit": False, "category": "object", "color": "white"},
    {"split": "sealed", "id": "quince", "label": "quince", "is_fruit": True, "category": "fruit", "color": "yellow"},
    {"split": "sealed", "id": "grapefruit", "label": "grapefruit", "is_fruit": True, "category": "fruit", "color": "pink"},
    {"split": "sealed", "id": "blackberry", "label": "blackberry", "is_fruit": True, "category": "fruit", "color": "black"},
    {"split": "sealed", "id": "papaya", "label": "papaya", "is_fruit": True, "category": "fruit", "color": "orange"},
    {"split": "sealed", "id": "apricot", "label": "apricot", "is_fruit": True, "category": "fruit", "color": "orange"},
    {"split": "sealed", "id": "fig", "label": "fig", "is_fruit": True, "category": "fruit", "color": "purple"},
    {"split": "sealed", "id": "celery", "label": "celery", "is_fruit": False, "category": "vegetable", "color": "green"},
    {"split": "sealed", "id": "sapphire", "label": "sapphire", "is_fruit": False, "category": "mineral", "color": "blue"},
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
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pseudo_word(index: int, split: str) -> str:
    split_offset = {name: position * 2000 for position, name in enumerate(SPLITS)}[split]
    shifted = index + split_offset
    return (
        SYLLABLE_A[shifted % len(SYLLABLE_A)]
        + SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
        + SYLLABLE_A[(shifted * 7 + 5) % len(SYLLABLE_A)]
    ).capitalize()


def split_cell(cell: str) -> tuple[str, str]:
    query, factor_cell = cell.split("__", 1)
    return query, factor_cell


def cell_factors(cell: str) -> dict[str, int]:
    _, factor_cell = split_cell(cell)
    return {factor: int(factor_cell.split(factor, 1)[1][0]) for factor in FACTORS}


def controlled_world(split: str, world_index: int, cell: str) -> dict[str, Any]:
    query_stratum, _ = split_cell(cell)
    factors = cell_factors(cell)
    entity_a = pseudo_word(world_index * 2, split)
    entity_b = pseudo_word(world_index * 2 + 1, split)
    categories = ("fruit", CONTROL_CATEGORIES[world_index % len(CONTROL_CATEGORIES)])
    relation = tuple(ATTRIBUTE_POOLS)[world_index % len(ATTRIBUTE_POOLS)]
    pool = ATTRIBUTE_POOLS[relation]
    offset = (world_index * 3) % len(pool)
    primary = (pool[offset], pool[(offset + 1) % len(pool)])
    alternate = (pool[(offset + 3) % len(pool)], pool[(offset + 5) % len(pool)])
    if len(set(primary + alternate)) != 4:
        raise RuntimeError("Phase557 attribute pool did not produce four unique values")

    category_a, category_b = categories
    if factors["category"]:
        category_a, category_b = category_b, category_a

    active_pair = primary if factors["attribute"] == 0 else alternate
    reference_pair = alternate if factors["attribute"] == 0 else primary
    active_a, active_b = active_pair
    if factors["binding"]:
        active_a, active_b = active_b, active_a
    reference_a, reference_b = reference_pair

    selected_entity = entity_a if factors["object"] == 0 else entity_b
    selected_category = category_a if factors["object"] == 0 else category_b
    selected_attribute = active_a if factors["object"] == 0 else active_b
    target = selected_category if query_stratum == "category" else selected_attribute

    facts = [
        f"{entity_a} belongs to category {category_a}",
        f"{entity_b} belongs to category {category_b}",
        f"the {relation} of {entity_a} is {active_a}",
        f"the {relation} of {entity_b} is {active_b}",
        f"the marker of {entity_a} is {reference_a}",
        f"the marker of {entity_b} is {reference_b}",
    ]
    fact_order = world_index % 2
    if fact_order:
        facts = list(reversed(facts))
    surface = world_index % 4
    if surface == 0:
        context = ". ".join(facts) + "."
        question = (
            f"What category is assigned to {selected_entity}?"
            if query_stratum == "category"
            else f"What {relation} is assigned to {selected_entity}?"
        )
    elif surface == 1:
        context = "Local record: " + "; ".join(facts) + "."
        question = (
            f"Return the category recorded for {selected_entity}."
            if query_stratum == "category"
            else f"Return the recorded {relation} for {selected_entity}."
        )
    elif surface == 2:
        context = "Use only these facts. " + ". ".join(facts) + "."
        question = (
            f"Lookup category({selected_entity})."
            if query_stratum == "category"
            else f"Lookup {relation}({selected_entity})."
        )
    else:
        context = "Fact ledger:\n- " + "\n- ".join(facts)
        question = (
            f"Requested field for {selected_entity}: category."
            if query_stratum == "category"
            else f"Requested field for {selected_entity}: {relation}."
        )
    raw_prompt = f"{context}\n{question}\nReply with exactly one answer word and no explanation."
    all_candidates = sorted(set(categories + primary + alternate))
    return {
        "raw_prompt": raw_prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": [value for value in all_candidates if value != target],
        "all_candidates": all_candidates,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "selected_entity": selected_entity,
        "category_a": category_a,
        "category_b": category_b,
        "attribute_relation": relation,
        "active_attribute_a": active_a,
        "active_attribute_b": active_b,
        "reference_attribute_a": reference_a,
        "reference_attribute_b": reference_b,
        "primary_attribute_pair": list(primary),
        "alternate_attribute_pair": list(alternate),
        "query_stratum": query_stratum,
        "surface_id": surface,
        "fact_order": fact_order,
        "factor_values": factors,
        "fact_token_multiset_key": stable_hash(sorted(" ".join(facts).lower().split())),
        "target_factor_dependencies": (
            ["object", "category"]
            if query_stratum == "category" else ["object", "attribute", "binding"]
        ),
        "semantic_fragments": {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "category_a": category_a,
            "category_b": category_b,
            "active_attribute_a": active_a,
            "active_attribute_b": active_b,
            "selected_entity": selected_entity,
            "query_relation": query_stratum if query_stratum == "category" else relation,
        },
    }


def natural_prompt(obj: dict[str, Any], relation: str, surface: int, order: int) -> str:
    label = obj["label"]
    phrase = "category" if relation == "category" else "typical color"
    if surface == 0:
        body = f"Using common everyday knowledge, what is the {phrase} of {label}?"
    elif surface == 1:
        body = f"Complete this familiar fact: the {phrase} of {label} is"
    elif surface == 2:
        body = f"Knowledge check about {label}. Give its {phrase}."
    else:
        body = f"Object: {label}. Requested field: {phrase}."
    if order:
        body = {
            0: f"Requested field: {phrase}. Object: {label}. Use common everyday knowledge.",
            1: f"Familiar fact requested: {phrase}. Subject: {label}.",
            2: f"Field first: {phrase}. Knowledge-check object: {label}.",
            3: f"Requested={phrase}; object={label}; source=everyday knowledge.",
        }[surface]
    return body + " Reply with exactly one answer word and no explanation."


def natural_distractors(obj: dict[str, Any], relation: str) -> list[str]:
    candidates = []
    for other in NATURAL_OBJECTS:
        if other["split"] != obj["split"] or other["id"] == obj["id"]:
            continue
        value = other[relation]
        if value != obj[relation] and value not in candidates:
            candidates.append(value)
    return candidates[:4]


def materialize_row(model: str, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    target_ids = [int(value) for value in tokenizer(row["target"], add_special_tokens=False)["input_ids"]]
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "target_first_token_id_private": target_ids[0] if target_ids else None,
        "strict_expected": row["target"],
        "sealed": row["split"] == "sealed",
        "behavior_only": True,
        "observer": False,
        "causal": False,
        "compute_edge": False,
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
                anchor_id = f"phase557_controlled_{split}_{world_index:03d}"
                for cell in CELLS:
                    destination.append(materialize_row(model, tokenizer, {
                        "case_id": f"{anchor_id}_{model}_{cell}",
                        "anchor_id": anchor_id,
                        "case_type": "controlled_factorial",
                        "split": split,
                        "world_index": world_index,
                        "factorial_cell": cell,
                        **controlled_world(split, world_index, cell),
                    }))
            if split in NATURAL_SPLITS:
                for obj in (item for item in NATURAL_OBJECTS if item["split"] == split):
                    for relation in NATURAL_RELATIONS:
                        for surface in NATURAL_SURFACES:
                            for order in NATURAL_ORDERS:
                                anchor_id = f"phase557_natural_{split}_{obj['id']}_{relation}"
                                target = obj[relation]
                                distractors = natural_distractors(obj, relation)
                                destination.append(materialize_row(model, tokenizer, {
                                    "case_id": f"{anchor_id}_{model}_surface{surface}_order{order}",
                                    "anchor_id": anchor_id,
                                    "case_type": "natural_parametric",
                                    "split": split,
                                    "object_id": obj["id"],
                                    "object_label": obj["label"],
                                    "is_fruit": obj["is_fruit"],
                                    "natural_relation": relation,
                                    "surface_id": surface,
                                    "fact_order": order,
                                    "raw_prompt": natural_prompt(obj, relation, surface, order),
                                    "target": target,
                                    "target_aliases": [target],
                                    "distractors": distractors,
                                    "all_candidates": [target] + distractors,
                                    "semantic_fragments": {
                                        "selected_entity": obj["label"],
                                        "query_relation": relation,
                                    },
                                }))
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = open_rows + sealed_rows
    controlled = [row for row in all_rows if row["case_type"] == "controlled_factorial"]
    natural = [row for row in all_rows if row["case_type"] == "natural_parametric"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in controlled:
        groups[(row["model"], row["anchor_id"])].append(row)
    factor_errors = 0
    target_dependency_errors = 0
    for group in groups.values():
        if len(group) != 32 or {row["factorial_cell"] for row in group} != set(CELLS):
            factor_errors += 1
            continue
        by_cell = {row["factorial_cell"]: row for row in group}
        for row in group:
            query, factor_cell = split_cell(row["factorial_cell"])
            factors = cell_factors(row["factorial_cell"])
            for factor in FACTORS:
                flipped = dict(factors)
                flipped[factor] = 1 - flipped[factor]
                other_factor_cell = "_".join(f"{name}{flipped[name]}" for name in FACTORS)
                other = by_cell[f"{query}__{other_factor_cell}"]
                should_change = factor in row["target_factor_dependencies"]
                if (row["target"] != other["target"]) != should_change:
                    target_dependency_errors += 1
    audit = {
        "schema_version": "phase557_static_audit.v1",
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
        "factorial_error_count": factor_errors,
        "target_dependency_error_count": target_dependency_errors,
        "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
        "duplicate_model_prompt_count": len(all_rows) - len({(row["model"], row["prompt"]) for row in all_rows}),
        "missing_target_token_count": sum(row["target_first_token_id_private"] is None for row in all_rows),
        "max_prompt_token_count": max(row["prompt_token_count"] for row in all_rows),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    expected_per_model = 9728
    expected_open_per_model = 8064
    expected_sealed_per_model = 1664
    audit["valid"] = bool(
        audit["registered_case_count"] == 29184
        and audit["open_case_count"] == 24192
        and audit["sealed_case_count"] == 4992
        and audit["model_case_counts"] == {model: expected_per_model for model in MODELS}
        and audit["open_model_case_counts"] == {model: expected_open_per_model for model in MODELS}
        and audit["sealed_model_case_counts"] == {model: expected_sealed_per_model for model in MODELS}
        and audit["controlled_rows_per_anchor"] == [32]
        and audit["max_prompt_token_count"] <= 512
        and all(audit[key] == 0 for key in (
            "factorial_error_count", "target_dependency_error_count", "duplicate_case_id_count",
            "duplicate_model_prompt_count", "missing_target_token_count",
            "open_contains_sealed_count", "sealed_flag_missing_count",
        ))
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = validate(open_rows, sealed_rows)
    if not audit["valid"]:
        write_json(AUDIT_PATH, audit)
        raise RuntimeError(f"Phase557 static protocol failed: {audit}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    commitment = {
        "schema_version": "phase557_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, commitment)
    write_json(FACT_BANK_PATH, {
        "schema_version": "phase557_public_fact_bank.v1",
        "phase_id": PHASE,
        "factors": list(FACTORS),
        "query_strata": list(QUERY_STRATA),
        "worlds_per_split": WORLDS_PER_SPLIT,
        "controlled_cells_per_world": len(CELLS),
        "natural_objects": [
            {key: value for key, value in row.items() if key != "split"}
            for row in NATURAL_OBJECTS if row["split"] != "sealed"
        ],
    })
    protocol = {
        "schema_version": "phase557_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "splits": list(SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "natural_splits": list(NATURAL_SPLITS),
        "worlds_per_split": WORLDS_PER_SPLIT,
        "factors": list(FACTORS),
        "query_strata": list(QUERY_STRATA),
        "factorial_cells": list(CELLS),
        "natural_relations": list(NATURAL_RELATIONS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "behavior_gate": {
            "world_all_32_rate_min_per_behavior_split": 0.70,
            "query_all_16_rate_min_per_behavior_split": 0.80,
            "each_cell_accuracy_min_per_behavior_split": 0.90,
            "controlled_unrecoverable_rate_max_per_behavior_split": 0.05,
            "path_anchor_requires_all_32_correct": True,
            "natural_relation_accuracy_min_per_behavior_split": 0.80,
            "natural_surface_accuracy_min": 0.65,
        },
        "evidence_policy": {
            "parametric_and_contextual_ledgers_separate": True,
            "complete_state_replacement_is_state_sufficiency_only": True,
            "additive_parent_sum_is_state_reconstruction_only": True,
            "compute_edge_requires_source_recompute_intervention": True,
            "parameter_scan_requires_replicated_compute_edge": True,
            "sealed_split_read": False,
            "single_neuron_scan_before_compute_edge": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    register()
