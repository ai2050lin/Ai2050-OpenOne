#!/usr/bin/env python3
"""Freeze a 2x2 route-by-answer factorial follow-up to Phase548."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase548_shared_attention_compute_protocol import (  # noqa: E402
    ATTRIBUTE_WORDS,
    CATEGORY_WORDS,
    MODELS,
    PAIR_UNITS_PER_SPLIT,
    SPLITS,
    SYLLABLE_A,
    SYLLABLE_B,
    WINDOWS,
    render_chat,
    three_distinct,
    token_edit_distance,
    wilson,
)


PHASE = "Phase549"
SCHEMA_VERSION = "phase549_route_answer_factorial.v1"
MECHANISMS = ("category", "negated_attribute")
CELLS = ("route0_answer_a", "route1_answer_a", "route0_answer_b", "route1_answer_b")
OUT_DIR = ROOT / "tests/gpt5/result/phase549_route_answer_factorial"
CASES_PATH = OUT_DIR / "phase549_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase549_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase549_static_audit.json"


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


def entity_word(index: int, split: str) -> str:
    shifted = index + (1700 if split == "independent_confirmation" else 900)
    a = SYLLABLE_A[shifted % len(SYLLABLE_A)]
    b = SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
    c = SYLLABLE_A[(shifted // (len(SYLLABLE_A) * len(SYLLABLE_B))) % len(SYLLABLE_A)]
    return "X" + a + b + c


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def case_spec(mechanism: str, split: str, pair_index: int, cell: str) -> dict[str, Any]:
    split_offset = 0 if split == "discovery" else PAIR_UNITS_PER_SPLIT
    base = pair_index + split_offset
    entity_a = entity_word(base * 2, split)
    entity_b = entity_word(base * 2 + 1, split)
    route = 0 if cell.startswith("route0") else 1
    answer_role = "a" if cell.endswith("answer_a") else "b"
    if mechanism == "category":
        answer_a, answer_b, _unused = three_distinct(CATEGORY_WORDS, base + 37)
        if route == 0:
            mapping = f"{entity_a} is a {answer_a}, and {entity_b} is a {answer_b}"
            selected = entity_a if answer_role == "a" else entity_b
        else:
            mapping = f"{entity_a} is a {answer_b}, and {entity_b} is a {answer_a}"
            selected = entity_b if answer_role == "a" else entity_a
        context = (
            f"In the stated local taxonomy, {mapping}. The selected entry is {selected}."
        )
        question = "What category belongs to the selected entry?"
        target = answer_a if answer_role == "a" else answer_b
        distractors = [answer_b if answer_role == "a" else answer_a]
        operation = "factorial_category_selection"
    elif mechanism == "negated_attribute":
        answer_a, answer_b, _unused = three_distinct(ATTRIBUTE_WORDS, base + 53)
        if route == 0:
            relation = f"{answer_a} is applicable and {answer_b} is inapplicable"
            question = (
                "Which attribute is applicable?" if answer_role == "a"
                else "Which attribute is inapplicable?"
            )
        else:
            relation = f"{answer_a} is inapplicable and {answer_b} is applicable"
            question = (
                "Which attribute is inapplicable?" if answer_role == "a"
                else "Which attribute is applicable?"
            )
        context = f"In the attribute record for {entity_a}, {relation}."
        target = answer_a if answer_role == "a" else answer_b
        distractors = [answer_b if answer_role == "a" else answer_a]
        operation = "factorial_negation_selection"
    else:
        raise KeyError(mechanism)
    instruction = "Return only the requested answer word and no explanation."
    raw_prompt = f"Context: {context}\nQuestion: {question}\nInstruction: {instruction}"
    return {
        "raw_prompt": raw_prompt,
        "source_fragment": f"Context: {context}",
        "query_fragment": f"Question: {question}",
        "target": target,
        "target_aliases": [target],
        "distractors": distractors,
        "all_candidates": [answer_a, answer_b],
        "route_factor": route,
        "answer_factor": answer_role,
        "operation": operation,
        "entity_key": f"{entity_a}:{entity_b}",
    }


def build_rows() -> list[dict[str, Any]]:
    rows = []
    created_at = now()
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for mechanism in MECHANISMS:
            for split in SPLITS:
                for pair_index in range(PAIR_UNITS_PER_SPLIT):
                    anchor_id = f"phase549_{mechanism}_{split}_{pair_index:03d}"
                    anchor_rows = []
                    for cell in CELLS:
                        spec = case_spec(mechanism, split, pair_index, cell)
                        prompt = render_chat(tokenizer, model, spec["raw_prompt"])
                        ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
                        row = {
                            "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                            "created_at": created_at, "case_id": f"{anchor_id}_{model}_{cell}",
                            "anchor_id": anchor_id, "model": model,
                            "family_id": "content_knowledge", "mechanism_id": mechanism,
                            "split": split, "pair_index": pair_index, "factorial_cell": cell,
                            "route_factor": spec["route_factor"], "answer_factor": spec["answer_factor"],
                            "raw_prompt": spec["raw_prompt"], "prompt": prompt,
                            "source_fragment": spec["source_fragment"], "query_fragment": spec["query_fragment"],
                            "target": spec["target"], "target_aliases": spec["target_aliases"],
                            "distractors": spec["distractors"], "all_candidates": spec["all_candidates"],
                            "strict_expected": spec["target"], "strict_kind": "plain",
                            "operation": spec["operation"], "entity_key": spec["entity_key"],
                            "prompt_token_count": len(ids), "semantic_event_is_natural_answer": True,
                            "arbitrary_label_output": False, "sealed": False,
                            "observer_only": True, "compute_edge": False, "causal": False,
                            "single_neuron": False,
                        }
                        anchor_rows.append((row, ids))
                    reference_ids = anchor_rows[0][1]
                    for row, ids in anchor_rows:
                        row["token_edit_distance_from_route0_answer_a"] = token_edit_distance(reference_ids, ids)
                        row["token_length_delta_from_route0_answer_a"] = len(ids) - len(reference_ids)
                        rows.append(row)
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(MODELS) * len(MECHANISMS) * len(SPLITS) * PAIR_UNITS_PER_SPLIT * len(CELLS)
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mechanism_id"], row["split"], row["pair_index"])].append(row)
    factorial_errors = 0
    for group in groups.values():
        by_cell = {row["factorial_cell"]: row for row in group}
        if set(by_cell) != set(CELLS):
            factorial_errors += 1
            continue
        if not (
            by_cell["route0_answer_a"]["target"] == by_cell["route1_answer_a"]["target"]
            and by_cell["route0_answer_b"]["target"] == by_cell["route1_answer_b"]["target"]
            and by_cell["route0_answer_a"]["target"] != by_cell["route0_answer_b"]["target"]
        ):
            factorial_errors += 1
    p548 = ROOT / "tests/gpt5/result/phase548_shared_attention_compute/phase548_registered_cases.jsonl"
    old_entities = {
        json.loads(line)["entity_key"] for line in p548.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    new_entities = {row["entity_key"] for row in rows}
    perfect_lcb, _ = wilson(PAIR_UNITS_PER_SPLIT, PAIR_UNITS_PER_SPLIT)
    _, zero_ucb = wilson(0, PAIR_UNITS_PER_SPLIT)
    audit = {
        "schema_version": "phase549_static_audit.v1", "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "expected_case_count": expected,
        "model_case_counts": dict(Counter(row["model"] for row in rows)),
        "anchor_group_count": len(groups), "rows_per_anchor": sorted({len(group) for group in groups.values()}),
        "factorial_relation_error_count": factorial_errors,
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_prompt_count": len(rows) - len({(row["model"], row["prompt"]) for row in rows}),
        "phase548_entity_overlap_count": len(old_entities & new_entities),
        "prompt_token_count_range_by_model": {
            model: [
                min(row["prompt_token_count"] for row in rows if row["model"] == model),
                max(row["prompt_token_count"] for row in rows if row["model"] == model),
            ] for model in MODELS
        },
        "perfect_anchor_lcb95": perfect_lcb, "zero_unrecoverable_ucb95": zero_ucb,
        "sealed_row_count": sum(bool(row["sealed"]) for row in rows),
    }
    audit["valid"] = (
        len(rows) == expected
        and set(audit["model_case_counts"].values()) == {expected // len(MODELS)}
        and audit["rows_per_anchor"] == [4]
        and max(maximum for _minimum, maximum in audit["prompt_token_count_range_by_model"].values()) <= 512
        and all(audit[key] == 0 for key in (
            "factorial_relation_error_count", "duplicate_case_id_count", "duplicate_prompt_count",
            "phase548_entity_overlap_count", "sealed_row_count",
        ))
        and perfect_lcb >= 0.90 and zero_ucb <= 0.05
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    rows = build_rows()
    write_jsonl(CASES_PATH, rows)
    audit = validate(rows)
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "title": "Route-by-answer factorial decomposition of late-attention geometry",
        "models_in_required_execution_order": list(MODELS), "mechanisms": list(MECHANISMS),
        "splits": list(SPLITS), "independent_worlds_per_mechanism_split": PAIR_UNITS_PER_SPLIT,
        "factorial_cells": list(CELLS), "frozen_windows": WINDOWS,
        "primary_comparison": {
            "route_effect": "same answer, different route",
            "answer_identity_effect": "same route, different answer",
            "no_layer_or_threshold_selection": True,
        },
        "behavior_gate": {
            "all_four_cells_correct_lcb95_min": 0.90,
            "unrecoverable_anchor_ucb95_max": 0.05,
            "discovery_and_independent_confirmation_required": True,
        },
        "observer_interpretation": {
            "route_dominance_fraction_min": 0.70,
            "answer_dominance_fraction_min": 0.70,
            "one_sided_sign_flip_permutation_p_max": 0.01,
            "permutation_count": 1024,
            "compute_intervention_authorized": False,
        },
        "evidence_boundaries": {
            "phase548_shared_compute_route_closed": True,
            "factorial_observation_is_compute_edge": False,
            "new_sealed_split_read": False,
            "head_channel_neuron_search": False,
        },
        "registered_cases_path": str(CASES_PATH.relative_to(ROOT)),
        "registered_cases_sha256": sha256_file(CASES_PATH),
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(register(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
