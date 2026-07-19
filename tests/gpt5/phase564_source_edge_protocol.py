#!/usr/bin/env python3
"""Freeze the Phase564 source-conditioned attention-edge protocol."""

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
from phase558_fixed_identity_color_protocol import (  # noqa: E402
    CORE_COLORS,
    FACT_ORDERS,
    HELDOUT_COLORS,
    QUERY_OBJECTS,
    SURFACES,
    alpha_code,
    distinct_pair,
    normalized_word_multiset,
    render_facts,
    render_question,
    stable_hash,
)


PHASE = "Phase564"
SCHEMA_VERSION = "phase564_source_edge_protocol.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLIT_WORLD_COUNTS = {
    "behavior_discovery": 128,
    "behavior_confirmation": 128,
    "edge_discovery": 48,
    "edge_confirmation": 64,
    "edge_unseen": 64,
    "sealed": 96,
}
SPLITS = tuple(SPLIT_WORLD_COUNTS)
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_confirmation")
EDGE_SPLITS = ("edge_discovery", "edge_confirmation", "edge_unseen")
OPEN_SPLITS = tuple(split for split in SPLITS if split != "sealed")
BINDINGS = (0, 1)
CELLS = tuple(
    f"binding{binding}_query{query}_surface{surface}_order{order}"
    for binding, query, surface, order in itertools.product(
        BINDINGS, QUERY_OBJECTS, SURFACES, FACT_ORDERS
    )
)
SYLLABLE_A = ("xa", "be", "ci", "do", "fu", "gi", "ha", "jo", "ku", "li", "me", "no", "pu")
SYLLABLE_B = ("vax", "bir", "cys", "dun", "fex", "gop", "hyl", "jor", "kem", "luz", "miv", "nax", "pyr")

OUT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
OPEN_CASES_PATH = OUT_DIR / "phase564_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase564_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase564_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase564_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase564_static_audit.json"
PRIOR_OPEN_PATHS = (
    ROOT / "tests/gpt5/result/phase558_fixed_identity_color/phase558_open_cases.jsonl",
    ROOT / "tests/gpt5/result/phase559_fixed_identity_replication/phase559_open_cases.jsonl",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pseudo_word(index: int, split: str) -> str:
    split_index = SPLITS.index(split)
    shifted = 250_000 + split_index * 20_000 + index
    stem = (
        SYLLABLE_A[shifted % len(SYLLABLE_A)]
        + SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
        + SYLLABLE_A[(shifted * 7 + 5) % len(SYLLABLE_A)]
    )
    return (stem + alpha_code(shifted)).capitalize()


def world_spec(split: str, world_index: int) -> dict[str, Any]:
    objects = (pseudo_word(world_index * 2, split), pseudo_word(world_index * 2 + 1, split))
    if split == "edge_unseen" and world_index >= 32:
        colors = distinct_pair(HELDOUT_COLORS, world_index + 101, 5)
        regime = "heldout_color_labels"
    elif split == "sealed" and world_index >= 48:
        colors = distinct_pair(HELDOUT_COLORS, world_index + 151, 7)
        regime = "sealed_heldout_color_labels"
    else:
        colors = distinct_pair(CORE_COLORS, world_index + SPLITS.index(split) * 41, 7)
        regime = "core_color_labels"
    return {
        "object_a": objects[0],
        "object_b": objects[1],
        "color_a": colors[0],
        "color_b": colors[1],
        "color_regime": regime,
    }


def case_spec(
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
    pair_id = f"phase564_{split}_{world_index:03d}_query{query_index}_surface{surface}_order{fact_order}"
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
        "factorial_cell": f"binding{binding}_query{query_index}_surface{surface}_order{fact_order}",
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


def materialize(model: str, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    target_ids = [int(value) for value in tokenizer(row["target"], add_special_tokens=False)["input_ids"]]
    distractor_ids = [
        int(value) for value in tokenizer(row["distractors"][0], add_special_tokens=False)["input_ids"]
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
        "sealed": split_is_sealed(row["split"]),
        "behavior_only": row["split"] in BEHAVIOR_SPLITS,
        "observer": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
    }


def split_is_sealed(split: str) -> bool:
    return split == "sealed"


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for split, world_count in SPLIT_WORLD_COUNTS.items():
            destination = sealed_rows if split_is_sealed(split) else open_rows
            for world_index in range(world_count):
                anchor_id = f"phase564_{split}_{world_index:03d}"
                for binding, query_index, surface, order in itertools.product(
                    BINDINGS, QUERY_OBJECTS, SURFACES, FACT_ORDERS
                ):
                    spec = case_spec(split, world_index, binding, query_index, surface, order)
                    destination.append(materialize(model, tokenizer, {
                        "case_id": f"{anchor_id}_{model}_{spec['factorial_cell']}",
                        "anchor_id": anchor_id,
                        "case_type": "fixed_identity_color_source_edge",
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
    world_errors = sum(
        len(rows) != 32 or {row["factorial_cell"] for row in rows} != set(CELLS)
        for rows in worlds.values()
    )
    pair_errors = 0
    for rows in pairs.values():
        if len(rows) != 2 or {row["binding"] for row in rows} != {0, 1}:
            pair_errors += 1
            continue
        left, right = sorted(rows, key=lambda row: row["binding"])
        fixed = (
            "object_a", "object_b", "color_a", "color_b", "query_object", "surface_id",
            "fact_order", "fact_token_multiset_key", "prompt_token_multiset_key",
        )
        if any(left[key] != right[key] for key in fixed):
            pair_errors += 1
        if left["target"] != right["nontarget_color"] or right["target"] != left["nontarget_color"]:
            pair_errors += 1

    new_objects = {value for row in all_rows for value in (row["object_a"], row["object_b"])}
    prior_open_objects = {
        value
        for path in PRIOR_OPEN_PATHS
        for row in read_jsonl(path)
        for value in (row["object_a"], row["object_b"])
    }
    expected_per_model = sum(SPLIT_WORLD_COUNTS.values()) * 32
    expected_open_per_model = sum(SPLIT_WORLD_COUNTS[split] for split in OPEN_SPLITS) * 32
    expected_sealed_per_model = SPLIT_WORLD_COUNTS["sealed"] * 32
    behavior_per_model = sum(SPLIT_WORLD_COUNTS[split] for split in BEHAVIOR_SPLITS) * 32
    audit = {
        "schema_version": "phase564_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(all_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_case_count_per_model": behavior_per_model,
        "model_case_counts": dict(Counter(row["model"] for row in all_rows)),
        "open_model_case_counts": dict(Counter(row["model"] for row in open_rows)),
        "sealed_model_case_counts": dict(Counter(row["model"] for row in sealed_rows)),
        "rows_per_world": sorted({len(rows) for rows in worlds.values()}),
        "rows_per_pair": sorted({len(rows) for rows in pairs.values()}),
        "world_error_count": world_errors,
        "pair_error_count": pair_errors,
        "prior_open_object_overlap_count": len(new_objects & prior_open_objects),
        "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
        "duplicate_model_prompt_count": len(all_rows) - len({(row["model"], row["prompt"]) for row in all_rows}),
        "first_token_collision_count": sum(
            row["target_first_token_id_private"] == row["distractor_first_token_id_private"]
            for row in all_rows
        ),
        "max_prompt_token_count": max(row["prompt_token_count"] for row in all_rows),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
        "sealed_rows_read_for_analysis": False,
    }
    zero_fields = (
        "world_error_count", "pair_error_count", "prior_open_object_overlap_count",
        "duplicate_case_id_count", "duplicate_model_prompt_count", "first_token_collision_count",
        "open_contains_sealed_count", "sealed_flag_missing_count",
    )
    audit["valid"] = bool(
        audit["registered_case_count"] == expected_per_model * len(MODELS)
        and audit["open_case_count"] == expected_open_per_model * len(MODELS)
        and audit["sealed_case_count"] == expected_sealed_per_model * len(MODELS)
        and audit["model_case_counts"] == {model: expected_per_model for model in MODELS}
        and audit["open_model_case_counts"] == {model: expected_open_per_model for model in MODELS}
        and audit["sealed_model_case_counts"] == {model: expected_sealed_per_model for model in MODELS}
        and audit["rows_per_world"] == [32]
        and audit["rows_per_pair"] == [2]
        and audit["max_prompt_token_count"] <= 256
        and all(audit[key] == 0 for key in zero_fields)
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = validate(open_rows, sealed_rows)
    write_json(AUDIT_PATH, audit)
    if not audit["valid"]:
        raise RuntimeError(f"Phase564 static protocol failed: {audit}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(SEALED_COMMITMENT_PATH, {
        "schema_version": "phase564_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    })
    protocol = {
        "schema_version": "phase564_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "splits": list(SPLITS),
        "behavior_splits": list(BEHAVIOR_SPLITS),
        "edge_splits": list(EDGE_SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "cells": list(CELLS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_case_count_per_model": audit["behavior_case_count_per_model"],
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "parent_contract": "Phase559 exact fixed-identity 32-cell behavior contract",
        "behavior_gate": {
            "world_all_32_rate_min_per_behavior_split": 0.80,
            "minimum_cell_wilson_95_lcb": 0.90,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "discovery_and_confirmation_both_required": True,
        },
        "edge_design": {
            "candidate_layers": list(range(4, 11)),
            "target_roles": ["query_object_end", "answer_boundary"],
            "source_role": "source_color_end",
            "discovery_conditions": [
                "same_case_restore", "source_edge_remove", "paired_donor_edge_replace",
                "nontarget_source_edge_replace",
            ],
            "confirmation_conditions": [
                "same_case_restore", "source_edge_remove", "paired_donor_edge_replace",
                "nontarget_source_edge_replace", "wrong_target_donor_replace",
                "wrong_depth_donor_replace", "channel_roll_donor_replace",
            ],
            "maximum_frozen_confirmation_candidates": 4,
            "candidate_selection_uses_edge_discovery_only": True,
            "confirmation_and_unseen_are_not_used_for_selection": True,
            "wrong_relation_control_available": False,
            "wrong_relation_reason": "the frozen denominator contains only the color relation",
        },
        "edge_gate": {
            "same_restore_max_abs_effect": 0.05,
            "donor_win_rate_min": 0.80,
            "minimum_factorial_cell_donor_win_rate": 0.60,
            "removal_mean_damage_min": 0.50,
            "donor_mean_effect_min": 1.00,
            "donor_effect_must_exceed_each_wrong_control": True,
            "discovery_confirmation_and_unseen_required": True,
        },
        "evidence_policy": {
            "behavior_authorization_before_internal_collection": True,
            "aggregate_heads_before_any_head_scan": True,
            "source_edge_is_post_softmax_value_contribution": True,
            "key_effect_not_identified_by_this_intervention": True,
            "compute_edge_requires_necessity_restore_sufficiency_specificity_and_holdout": True,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    register()
