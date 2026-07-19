#!/usr/bin/env python3
"""Freeze the Phase559 exact-contract, larger-denominator replication."""

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


PHASE = "Phase559"
SCHEMA_VERSION = "phase559_fixed_identity_replication.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLIT_WORLD_COUNTS = {
    "behavior_discovery": 128,
    "behavior_confirmation": 128,
    "path_discovery": 64,
    "path_confirmation": 64,
    "unseen_recombination": 96,
    "sealed": 96,
}
SPLITS = tuple(SPLIT_WORLD_COUNTS)
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_confirmation")
PATH_SPLITS = ("path_discovery", "path_confirmation")
OPEN_SPLITS = tuple(split for split in SPLITS if split != "sealed")
BINDINGS = (0, 1)
CELLS = tuple(
    f"binding{binding}_query{query}_surface{surface}_order{order}"
    for binding, query, surface, order in itertools.product(
        BINDINGS, QUERY_OBJECTS, SURFACES, FACT_ORDERS
    )
)
SYLLABLE_A = ("sa", "te", "vi", "wo", "zu", "ka", "le", "mo", "nu", "pe", "ra", "si", "to")
SYLLABLE_B = ("bar", "cen", "dor", "fel", "gan", "hir", "jon", "kel", "lum", "nar", "pos", "riv", "tan")

OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
OPEN_CASES_PATH = OUT_DIR / "phase559_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase559_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase559_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase559_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase559_static_audit.json"
PRIOR_OPEN_CASES_PATH = ROOT / "tests/gpt5/result/phase558_fixed_identity_color/phase558_open_cases.jsonl"


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
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pseudo_word(index: int, split: str) -> str:
    split_index = SPLITS.index(split)
    shifted = 50_000 + split_index * 10_000 + index
    stem = (
        SYLLABLE_A[shifted % len(SYLLABLE_A)]
        + SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
        + SYLLABLE_A[(shifted * 5 + 3) % len(SYLLABLE_A)]
    )
    return (stem + alpha_code(shifted)).capitalize()


def world_spec(split: str, world_index: int) -> dict[str, Any]:
    objects = (pseudo_word(world_index * 2, split), pseudo_word(world_index * 2 + 1, split))
    if split == "unseen_recombination" and world_index >= 48:
        colors = distinct_pair(HELDOUT_COLORS, world_index + 31, 5)
        regime = "heldout_color_labels"
    elif split == "sealed" and world_index >= 48:
        colors = distinct_pair(HELDOUT_COLORS, world_index + 61, 7)
        regime = "sealed_heldout_color_labels"
    else:
        colors = distinct_pair(CORE_COLORS, world_index + SPLITS.index(split) * 29, 7)
        regime = "core_color_labels"
    return {
        "object_a": objects[0], "object_b": objects[1],
        "color_a": colors[0], "color_b": colors[1], "color_regime": regime,
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
    nontarget_fact = next(fact for fact, assignment in zip(facts, assignments) if assignment[0] == nontarget_object)
    pair_id = f"phase559_{split}_{world_index:03d}_query{query_index}_surface{surface}_order{fact_order}"
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
    distractor_ids = [int(value) for value in tokenizer(row["distractors"][0], add_special_tokens=False)["input_ids"]]
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
        "behavior_only": row["split"] in BEHAVIOR_SPLITS,
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
                anchor_id = f"phase559_{split}_{world_index:03d}"
                for binding, query_index, surface, order in itertools.product(
                    BINDINGS, QUERY_OBJECTS, SURFACES, FACT_ORDERS
                ):
                    spec = case_spec(split, world_index, binding, query_index, surface, order)
                    destination.append(materialize(model, tokenizer, {
                        "case_id": f"{anchor_id}_{model}_{spec['factorial_cell']}",
                        "anchor_id": anchor_id,
                        "case_type": "fixed_identity_color_binding_replication",
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
    phase559_objects = {value for row in all_rows for value in (row["object_a"], row["object_b"])}
    prior_open_objects = {
        value for row in read_jsonl(PRIOR_OPEN_CASES_PATH)
        for value in (row["object_a"], row["object_b"])
    }
    expected_per_model = sum(SPLIT_WORLD_COUNTS.values()) * 32
    expected_open_per_model = sum(SPLIT_WORLD_COUNTS[split] for split in OPEN_SPLITS) * 32
    expected_sealed_per_model = SPLIT_WORLD_COUNTS["sealed"] * 32
    behavior_per_model = sum(SPLIT_WORLD_COUNTS[split] for split in BEHAVIOR_SPLITS) * 32
    audit = {
        "schema_version": "phase559_static_audit.v1",
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
        "phase558_open_object_overlap_count": len(phase559_objects & prior_open_objects),
        "duplicate_case_id_count": len(all_rows) - len({row["case_id"] for row in all_rows}),
        "duplicate_model_prompt_count": len(all_rows) - len({(row["model"], row["prompt"]) for row in all_rows}),
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
        and audit["rows_per_pair"] == [2]
        and audit["max_prompt_token_count"] <= 256
        and all(audit[key] == 0 for key in (
            "world_error_count", "pair_error_count", "phase558_open_object_overlap_count",
            "duplicate_case_id_count", "duplicate_model_prompt_count", "first_token_collision_count",
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
        raise RuntimeError(f"Phase559 static protocol failed: {audit}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(SEALED_COMMITMENT_PATH, {
        "schema_version": "phase559_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    })
    protocol = {
        "schema_version": "phase559_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "splits": list(SPLITS),
        "behavior_splits": list(BEHAVIOR_SPLITS),
        "path_splits": list(PATH_SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "cells": list(CELLS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_case_count_per_model": audit["behavior_case_count_per_model"],
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "phase558_contract_changes": {
            "surface_templates_changed": False,
            "fact_order_conditions_changed": False,
            "classifier_changed": False,
            "behavior_thresholds_changed": False,
            "only_new_disjoint_objects_and_larger_denominator": True,
        },
        "behavior_gate": {
            "world_all_32_rate_min_per_behavior_split": 0.80,
            "minimum_cell_wilson_95_lcb": 0.90,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "discovery_and_confirmation_both_required": True,
        },
        "evidence_policy": {
            "phase558_failures_not_used_as_training_data": True,
            "internal_collection_requires_phase559_behavior_pass": True,
            "path_behavior_runs_only_after_model_authorization": True,
            "complete_state_is_sufficiency_only": True,
            "compute_edge_requires_source_delete_restore_and_exclusion": True,
            "fine_scan_requires_replicated_compute_edge": True,
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
