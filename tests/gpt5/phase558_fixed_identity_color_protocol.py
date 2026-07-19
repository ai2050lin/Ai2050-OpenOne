#!/usr/bin/env python3
"""Freeze the Phase558 fixed-identity color binding protocol.

Every counterfactual pair keeps object tokens, color tokens, query, answer set,
surface, and fact order fixed. Only the object-color assignment is swapped.
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

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase558"
SCHEMA_VERSION = "phase558_fixed_identity_color.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLIT_WORLD_COUNTS = {
    "behavior_discovery": 64,
    "behavior_confirmation": 64,
    "path_discovery": 48,
    "path_confirmation": 48,
    "unseen_recombination": 64,
    "sealed": 64,
}
SPLITS = tuple(SPLIT_WORLD_COUNTS)
OPEN_SPLITS = tuple(split for split in SPLITS if split != "sealed")
BINDINGS = (0, 1)
QUERY_OBJECTS = (0, 1)
SURFACES = tuple(range(4))
FACT_ORDERS = (0, 1)
CELLS = tuple(
    f"binding{binding}_query{query}_surface{surface}_order{order}"
    for binding, query, surface, order in itertools.product(
        BINDINGS, QUERY_OBJECTS, SURFACES, FACT_ORDERS
    )
)

CORE_COLORS = (
    "red", "green", "blue", "yellow", "orange", "purple", "black", "white",
    "brown", "pink", "gray", "gold",
)
HELDOUT_COLORS = (
    "teal", "silver", "violet", "amber", "crimson", "ivory", "navy", "maroon",
)
SYLLABLE_A = ("ba", "ce", "di", "fo", "ga", "hu", "ji", "ke", "lu", "mi", "no", "pa", "ri")
SYLLABLE_B = ("lan", "mer", "tin", "vor", "sen", "dak", "pel", "rin", "sol", "wen", "yas", "kor", "zen")

OUT_DIR = ROOT / "tests/gpt5/result/phase558_fixed_identity_color"
OPEN_CASES_PATH = OUT_DIR / "phase558_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase558_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase558_sealed_commitment.json"
PUBLIC_WORLD_BANK_PATH = OUT_DIR / "phase558_public_world_bank.json"
PROTOCOL_PATH = OUT_DIR / "phase558_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase558_static_audit.json"


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


def alpha_code(value: int) -> str:
    """Encode a non-negative integer as a collision-free lowercase suffix."""
    if value < 0:
        raise ValueError(value)
    letters = []
    current = value
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("a") + remainder))
        if current == 0:
            return "".join(reversed(letters))


def pseudo_word(index: int, split: str) -> str:
    split_offset = {name: position * 3000 for position, name in enumerate(SPLITS)}[split]
    shifted = index + split_offset
    stem = (
        SYLLABLE_A[shifted % len(SYLLABLE_A)]
        + SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
        + SYLLABLE_A[(shifted * 7 + 5) % len(SYLLABLE_A)]
    )
    return (stem + alpha_code(shifted)).capitalize()


def normalized_word_multiset(text: str) -> list[str]:
    return sorted(re.findall(r"[A-Za-z]+", text.casefold()))


def distinct_pair(pool: tuple[str, ...], index: int, stride: int) -> tuple[str, str]:
    left = pool[index % len(pool)]
    right = pool[(index * stride + 3) % len(pool)]
    cursor = 0
    while right == left:
        cursor += 1
        right = pool[(index * stride + 3 + cursor) % len(pool)]
    return left, right


def world_spec(split: str, world_index: int) -> dict[str, Any]:
    object_a = pseudo_word(world_index * 2, split)
    object_b = pseudo_word(world_index * 2 + 1, split)
    if split == "unseen_recombination" and world_index >= SPLIT_WORLD_COUNTS[split] // 2:
        colors = distinct_pair(HELDOUT_COLORS, world_index, 5)
        color_regime = "heldout_color_labels"
    elif split == "sealed" and world_index >= SPLIT_WORLD_COUNTS[split] // 2:
        colors = distinct_pair(HELDOUT_COLORS, world_index + 19, 7)
        color_regime = "sealed_heldout_color_labels"
    else:
        split_shift = {name: position * 17 for position, name in enumerate(SPLITS)}[split]
        colors = distinct_pair(CORE_COLORS, world_index + split_shift, 7)
        color_regime = "core_color_labels"
    return {
        "object_a": object_a,
        "object_b": object_b,
        "color_a": colors[0],
        "color_b": colors[1],
        "color_regime": color_regime,
    }


def render_facts(surface: int, assignments: list[tuple[str, str]]) -> tuple[list[str], str]:
    if surface == 0:
        facts = [f"{obj} has color {color}" for obj, color in assignments]
        context = "Temporary color ledger. " + ". ".join(facts) + "."
    elif surface == 1:
        facts = [f"{obj} | color | {color}" for obj, color in assignments]
        context = "Temporary color ledger:\n" + "\n".join(facts)
    elif surface == 2:
        facts = [f"color({obj}) = {color}" for obj, color in assignments]
        context = "Temporary color ledger: " + "; ".join(facts) + "."
    elif surface == 3:
        facts = [f"{color} is assigned as the color of {obj}" for obj, color in assignments]
        context = "Temporary color ledger. " + "; ".join(facts) + "."
    else:
        raise ValueError(surface)
    return facts, context


def render_question(surface: int, query_object: str) -> str:
    if surface == 0:
        return f"According to the ledger, what color is {query_object}?"
    if surface == 1:
        return f"Look up the recorded color for {query_object}."
    if surface == 2:
        return f"Return color({query_object})."
    if surface == 3:
        return f"Which color is assigned to {query_object}?"
    raise ValueError(surface)


def controlled_case(
    split: str,
    world_index: int,
    binding: int,
    query_index: int,
    surface: int,
    fact_order: int,
) -> dict[str, Any]:
    world = world_spec(split, world_index)
    objects = (world["object_a"], world["object_b"])
    colors = (world["color_a"], world["color_b"])
    assignments = [(objects[0], colors[binding]), (objects[1], colors[1 - binding])]
    if fact_order:
        assignments = list(reversed(assignments))
    facts, context = render_facts(surface, assignments)
    query_object = objects[query_index]
    target = colors[query_index ^ binding]
    nontarget_object = objects[1 - query_index]
    nontarget_color = colors[(1 - query_index) ^ binding]
    question = render_question(surface, query_object)
    instruction = (
        "Use only this temporary ledger. Reply with exactly one lowercase color word "
        "and no explanation."
    )
    raw_prompt = f"{context}\nQuestion: {question}\nInstruction: {instruction}"
    source_fact = next(fact for fact, assignment in zip(facts, assignments) if assignment[0] == query_object)
    nontarget_fact = next(
        fact for fact, assignment in zip(facts, assignments) if assignment[0] == nontarget_object
    )
    pair_id = (
        f"phase558_{split}_{world_index:03d}_query{query_index}_surface{surface}_order{fact_order}"
    )
    return {
        "raw_prompt": raw_prompt,
        "context": context,
        "question": question,
        "instruction": instruction,
        "facts": facts,
        "source_fact": source_fact,
        "nontarget_fact": nontarget_fact,
        "object_a": objects[0],
        "object_b": objects[1],
        "color_a": colors[0],
        "color_b": colors[1],
        "color_regime": world["color_regime"],
        "binding": binding,
        "query_object_index": query_index,
        "query_object": query_object,
        "nontarget_object": nontarget_object,
        "target": target,
        "target_aliases": [target],
        "nontarget_color": nontarget_color,
        "distractors": [nontarget_color],
        "all_candidates": list(colors),
        "surface_id": surface,
        "fact_order": fact_order,
        "pair_id": pair_id,
        "factorial_cell": (
            f"binding{binding}_query{query_index}_surface{surface}_order{fact_order}"
        ),
        "fact_token_multiset_key": stable_hash(normalized_word_multiset(" ".join(facts))),
        "prompt_token_multiset_key": stable_hash(normalized_word_multiset(raw_prompt)),
        "semantic_fragments": {
            "source_fact": source_fact,
            "nontarget_fact": nontarget_fact,
            "query_object": query_object,
            "target_color": target,
            "nontarget_color": nontarget_color,
            "query_relation": "color",
        },
    }


def materialize_row(model: str, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    target_ids = [int(value) for value in tokenizer(row["target"], add_special_tokens=False)["input_ids"]]
    distractor_ids = [
        int(value)
        for value in tokenizer(row["distractors"][0], add_special_tokens=False)["input_ids"]
    ]
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "target_first_token_id_private": target_ids[0] if target_ids else None,
        "distractor_first_token_id_private": distractor_ids[0] if distractor_ids else None,
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
        for split, world_count in SPLIT_WORLD_COUNTS.items():
            destination = sealed_rows if split == "sealed" else open_rows
            for world_index in range(world_count):
                anchor_id = f"phase558_{split}_{world_index:03d}"
                for binding, query_index, surface, fact_order in itertools.product(
                    BINDINGS, QUERY_OBJECTS, SURFACES, FACT_ORDERS
                ):
                    spec = controlled_case(
                        split, world_index, binding, query_index, surface, fact_order
                    )
                    destination.append(materialize_row(model, tokenizer, {
                        "case_id": f"{anchor_id}_{model}_{spec['factorial_cell']}",
                        "anchor_id": anchor_id,
                        "case_type": "fixed_identity_color_binding",
                        "split": split,
                        "world_index": world_index,
                        **spec,
                    }))
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = open_rows + sealed_rows
    worlds: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        worlds[(row["model"], row["anchor_id"])].append(row)
        pairs[(row["model"], row["pair_id"])].append(row)

    world_errors = 0
    pair_errors = 0
    for rows in worlds.values():
        if len(rows) != len(CELLS) or {row["factorial_cell"] for row in rows} != set(CELLS):
            world_errors += 1
    for rows in pairs.values():
        if len(rows) != 2 or {row["binding"] for row in rows} != set(BINDINGS):
            pair_errors += 1
            continue
        left, right = sorted(rows, key=lambda row: row["binding"])
        invariant_keys = (
            "object_a", "object_b", "color_a", "color_b", "query_object",
            "surface_id", "fact_order", "fact_token_multiset_key", "prompt_token_multiset_key",
        )
        if any(left[key] != right[key] for key in invariant_keys):
            pair_errors += 1
        if left["target"] == right["target"] or left["nontarget_color"] == right["nontarget_color"]:
            pair_errors += 1
        if left["target"] != right["nontarget_color"] or right["target"] != left["nontarget_color"]:
            pair_errors += 1

    object_sets = {
        split: {
            value
            for row in all_rows if row["split"] == split
            for value in (row["object_a"], row["object_b"])
        }
        for split in SPLITS
    }
    cross_split_object_overlap = sum(
        len(object_sets[left] & object_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    )
    expected_per_model = sum(SPLIT_WORLD_COUNTS.values()) * len(CELLS)
    expected_open_per_model = sum(
        SPLIT_WORLD_COUNTS[split] for split in OPEN_SPLITS
    ) * len(CELLS)
    expected_sealed_per_model = SPLIT_WORLD_COUNTS["sealed"] * len(CELLS)
    audit = {
        "schema_version": "phase558_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(all_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "model_case_counts": dict(Counter(row["model"] for row in all_rows)),
        "open_model_case_counts": dict(Counter(row["model"] for row in open_rows)),
        "sealed_model_case_counts": dict(Counter(row["model"] for row in sealed_rows)),
        "world_count": len(worlds),
        "pair_count": len(pairs),
        "rows_per_world": sorted({len(rows) for rows in worlds.values()}),
        "rows_per_counterfactual_pair": sorted({len(rows) for rows in pairs.values()}),
        "world_error_count": world_errors,
        "counterfactual_pair_error_count": pair_errors,
        "cross_split_object_overlap_count": cross_split_object_overlap,
        "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
        "duplicate_model_prompt_count": len(all_rows) - len({(row["model"], row["prompt"]) for row in all_rows}),
        "missing_target_token_count": sum(row["target_first_token_id_private"] is None for row in all_rows),
        "first_token_collision_count": sum(
            row["target_first_token_id_private"] == row["distractor_first_token_id_private"]
            for row in all_rows
        ),
        "max_prompt_token_count": max(row["prompt_token_count"] for row in all_rows),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    audit["valid"] = bool(
        audit["registered_case_count"] == expected_per_model * len(MODELS)
        and audit["open_case_count"] == expected_open_per_model * len(MODELS)
        and audit["sealed_case_count"] == expected_sealed_per_model * len(MODELS)
        and audit["model_case_counts"] == {model: expected_per_model for model in MODELS}
        and audit["open_model_case_counts"] == {model: expected_open_per_model for model in MODELS}
        and audit["sealed_model_case_counts"] == {model: expected_sealed_per_model for model in MODELS}
        and audit["rows_per_world"] == [32]
        and audit["rows_per_counterfactual_pair"] == [2]
        and audit["max_prompt_token_count"] <= 256
        and all(audit[key] == 0 for key in (
            "world_error_count", "counterfactual_pair_error_count",
            "cross_split_object_overlap_count", "duplicate_case_id_count",
            "duplicate_model_prompt_count", "missing_target_token_count",
            "first_token_collision_count", "open_contains_sealed_count",
            "sealed_flag_missing_count",
        ))
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = validate(open_rows, sealed_rows)
    if not audit["valid"]:
        write_json(AUDIT_PATH, audit)
        raise RuntimeError(f"Phase558 static protocol failed: {audit}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    commitment = {
        "schema_version": "phase558_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, commitment)
    write_json(PUBLIC_WORLD_BANK_PATH, {
        "schema_version": "phase558_public_world_bank.v1",
        "phase_id": PHASE,
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "cells_per_world": len(CELLS),
        "core_colors": list(CORE_COLORS),
        "heldout_colors": list(HELDOUT_COLORS),
        "sealed_object_labels_omitted": True,
        "sealed_rows_omitted": True,
    })
    protocol = {
        "schema_version": "phase558_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "splits": list(SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "cells": list(CELLS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "behavior_gate": {
            "world_all_32_rate_min_per_behavior_split": 0.80,
            "minimum_cell_wilson_95_lcb": 0.90,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "minimum_all_correct_path_worlds_per_split": 24,
            "discovery_and_confirmation_both_required": True,
        },
        "evidence_policy": {
            "object_identity_fixed_within_counterfactual_pair": True,
            "color_set_fixed_within_counterfactual_pair": True,
            "query_and_answer_set_fixed_within_counterfactual_pair": True,
            "complete_state_replacement_is_state_sufficiency_only": True,
            "compute_edge_requires_source_delete_restore_and_exclusion": True,
            "cross_position_parent_before_head_scan": True,
            "parameter_scan_requires_replicated_compute_edge": True,
            "single_neuron_scan_before_compute_edge": False,
            "sealed_split_read": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    register()
