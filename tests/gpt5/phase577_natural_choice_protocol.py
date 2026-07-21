#!/usr/bin/env python3
"""Freeze a natural two-choice fruit knowledge denominator for Phase577."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402
import phase576_natural_fruit_protocol as p576  # noqa: E402


PHASE = "Phase577"
MODELS = p576.MODELS
STRUCTURE_SPLITS = p576.STRUCTURE_SPLITS
CAUSAL_SPLITS = p576.CAUSAL_SPLITS
SEALED_SPLIT = p576.SEALED_SPLIT
SPLITS = p576.SPLITS
RELATIONS = p576.RELATIONS
VARIANTS = ("target_first", "target_second")
NOOP_REPEATS = p576.NOOP_REPEATS
FIXED_BATCH_SIZE = 32
MIN_STABLE_SURFACES_PER_RELATION = 7
MIN_QUALIFIED_FRUITS_PER_SPLIT = 9
MIN_QUALIFIED_CONTROLS_PER_SPLIT = 4
TRACE_WORLDS_PER_SPLIT = 96

OUT_DIR = ROOT / "tests/gpt5/result/phase577_natural_choice"
OPEN_CASES_PATH = OUT_DIR / "phase577_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase577_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase577_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase577_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase577_static_audit.json"
PUBLIC_OBJECTS_PATH = OUT_DIR / "phase577_public_object_bank.json"


COLOR_FOILS = {
    "red": "blue",
    "green": "red",
    "yellow": "purple",
    "orange": "blue",
    "blue": "orange",
    "purple": "yellow",
    "black": "yellow",
    "brown": "blue",
    "tan": "purple",
    "white": "black",
    "pink": "green",
}
COLOR_FALLBACKS = (
    "blue", "purple", "orange", "yellow", "red", "green", "black", "white",
)


CHOICE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("object_first", "Object: {object}. Question: {question} Choices: {left} or {right}."),
    ("object_first", "Item {object}. {question} Choose between {left} and {right}."),
    ("object_first", "Subject = {object}. {question} Options are {left} / {right}."),
    ("object_first", "Consider {object}. {question} Answer from: {left}, {right}."),
    ("relation_first", "{question} The object is {object}. Choices: {left} or {right}."),
    ("relation_first", "Requested judgment: {question} Item: {object}. Pick {left} or {right}."),
    ("relation_first", "Question first: {question} Subject: {object}. Options {left}; {right}."),
    ("relation_first", "Decide {question} for {object}. Candidate words: {left}, {right}."),
    ("object_first", "Knowledge object {object}. {question} Select {left} or {right}."),
    ("object_first", "For {object}, answer this: {question} Alternatives: {left} and {right}."),
    ("object_first", "Entry {object}. {question} Allowed answers: {left} | {right}."),
    ("object_first", "Think about {object}. {question} Use either {left} or {right}."),
    ("relation_first", "Decision field: {question} Target item: {object}. {left} versus {right}."),
    ("relation_first", "{question} About the item {object}, choose {left} or {right}."),
    ("relation_first", "Knowledge test: {question} Object = {object}. Options = {left}, {right}."),
    ("relation_first", "Resolve {question} with subject {object}; answers {left} / {right}."),
    ("object_first", "Everyday-knowledge subject {object}. {question} Choices {left} and {right}."),
    ("object_first", "Take the item {object}. {question} Select from {left}, {right}."),
    ("object_first", "Record about {object}. {question} Candidate one {left}; candidate two {right}."),
    ("object_first", "The object is {object}. {question} Pick the word {left} or {right}."),
    ("relation_first", "First resolve: {question} The target object is {object}. {left} or {right}."),
    ("relation_first", "Question: {question} Knowledge subject: {object}. Choose {left} / {right}."),
    ("relation_first", "For this property, {question} Answer about {object}: {left} or {right}."),
    ("relation_first", "Complete the judgment {question} for {object}; use {left} or {right}."),
)


QUESTION_PHRASES = {
    "category": (
        "Which everyday broad food category applies?",
        "Is its ordinary food category fruit or vegetable?",
        "Which broad everyday category is correct?",
        "What common food class fits it?",
    ),
    "outer_color": (
        "Which is its typical outer color?",
        "What is its usual visible color?",
        "Which common exterior color fits?",
        "What typical outside color is correct?",
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


def target_and_foil(item: dict[str, Any], relation: str) -> tuple[str, str]:
    if relation == "category":
        target = item["category_aliases"][0]
        foil = "vegetable" if target == "fruit" else "fruit"
        return target, foil
    target = item["outer_color_aliases"][0]
    foil = COLOR_FOILS[target]
    if foil in item["outer_color_aliases"]:
        foil = next(
            candidate
            for candidate in COLOR_FALLBACKS
            if candidate not in item["outer_color_aliases"]
        )
    return target, foil


def render_case(
    item: dict[str, Any],
    relation: str,
    surface_id: int,
    variant: str,
) -> tuple[str, str, str, str, str]:
    order, template = CHOICE_TEMPLATES[surface_id]
    question = QUESTION_PHRASES[relation][surface_id % 4]
    target, foil = target_and_foil(item, relation)
    left, right = (target, foil) if variant == "target_first" else (foil, target)
    body = template.format(
        object=item["label"], question=question, left=left, right=right
    )
    prompt = (
        "Use common everyday knowledge; no fictional facts are supplied. "
        + body
        + " Reply with exactly one of the two candidate words and no explanation."
    )
    return prompt, order, question, left, right


def materialize_row(
    tokenizers: dict[str, Any],
    item: dict[str, Any],
    relation: str,
    split: str,
    surface_id: int,
    variant: str,
) -> dict[str, Any]:
    raw_prompt, order, question, left, right = render_case(
        item, relation, surface_id, variant
    )
    target, foil = target_and_foil(item, relation)
    prompt_counts = {}
    candidate_ids = {}
    for model, tokenizer in tokenizers.items():
        prompt = render_chat(tokenizer, model, raw_prompt)
        prompt_counts[model] = len(
            tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )
        candidate_ids[model] = {
            candidate: [
                int(token)
                for token in tokenizer(candidate, add_special_tokens=False)["input_ids"]
            ]
            for candidate in (target, foil)
        }
    world_id = f"phase577_{split}_{item['object_id']}_{relation}_surface{surface_id:02d}"
    return {
        "schema_version": "phase577_natural_choice_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": f"{world_id}_{variant}",
        "world_id": world_id,
        "split": split,
        "object_group": p576.SPLIT_GROUP[split],
        "object_id": item["object_id"],
        "object_label": item["label"],
        "is_fruit": item["is_fruit"],
        "object_category": item["kind"],
        "relation": relation,
        "surface_id": surface_id,
        "surface_order": order,
        "variant": variant,
        "raw_prompt": raw_prompt,
        "question_phrase": question,
        "left_option": left,
        "right_option": right,
        "target": target,
        "foil": foil,
        "target_aliases": [target],
        "all_candidates": [target, foil],
        "candidate_token_ids_by_model": candidate_ids,
        "prompt_token_count_by_model": prompt_counts,
        "semantic_fragments": {
            "object": item["label"],
            "relation": question,
            "target_option": target,
            "foil_option": foil,
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
        for item in p576.OBJECT_GROUPS[p576.SPLIT_GROUP[split]]:
            for surface_id in p576.SPLIT_SURFACES[split]:
                for relation in RELATIONS:
                    for variant in VARIANTS:
                        destination.append(
                            materialize_row(
                                tokenizers, item, relation, split, surface_id, variant
                            )
                        )
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_world[row["world_id"]].append(row)
    expected_per_split = 14 * len(RELATIONS) * 8 * len(VARIANTS)
    audit = {
        "schema_version": "phase577_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "world_count": len(by_world),
        "world_row_count_values": sorted({len(value) for value in by_world.values()}),
        "incomplete_world_count": sum(
            {row["variant"] for row in value} != set(VARIANTS)
            for value in by_world.values()
        ),
        "option_swap_error_count": sum(
            not (
                value[0]["left_option"] == value[1]["right_option"]
                and value[0]["right_option"] == value[1]["left_option"]
            )
            for value in by_world.values()
        ),
        "target_foil_overlap_count": sum(row["target"] == row["foil"] for row in rows),
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows) - len(
            {(row["split"], row["raw_prompt"]) for row in rows}
        ),
        "missing_candidate_token_count": sum(
            not ids
            for row in rows
            for model_map in row["candidate_token_ids_by_model"].values()
            for ids in model_map.values()
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
        and audit["world_row_count_values"] == [2]
        and audit["max_prompt_token_count"] <= 192
        and all(
            audit[key] == 0
            for key in (
                "incomplete_world_count",
                "option_swap_error_count",
                "target_foil_overlap_count",
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
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
            "schema_version": "phase577_sealed_commitment.v1",
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
            "schema_version": "phase577_public_object_bank.v1",
            "created_at": now(),
            "open_groups": {
                key: value for key, value in p576.OBJECT_GROUPS.items() if key != "D"
            },
            "sealed_group_object_count": len(p576.OBJECT_GROUPS["D"]),
            "relations": list(RELATIONS),
            "candidate_semantics": "answer with the natural semantic option word, not a label",
        },
    )
    write_json(AUDIT_PATH, audit)
    frozen = {
        "schema_version": "phase577_natural_choice_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Natural fruit category/color choice and option-order counterfactual",
        "models_in_required_execution_order": list(MODELS),
        "splits": list(SPLITS),
        "structure_splits": list(STRUCTURE_SPLITS),
        "causal_splits": list(CAUSAL_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "relations": list(RELATIONS),
        "variants": list(VARIANTS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_gate": {
            "minimum_stable_surfaces_per_relation": MIN_STABLE_SURFACES_PER_RELATION,
            "minimum_qualified_fruits_per_structure_split": MIN_QUALIFIED_FRUITS_PER_SPLIT,
            "minimum_qualified_controls_per_structure_split": MIN_QUALIFIED_CONTROLS_PER_SPLIT,
            "trace_worlds_per_structure_split": TRACE_WORLDS_PER_SPLIT,
            "both_option_orders_required": True,
            "exact_repeat_required": True,
        },
        "internal_policy": {
            "full_depth_natural_trace_before_intervention": True,
            "all_layers_and_major_components_observed": True,
            "no_layer_head_channel_or_neuron_preselection": True,
            "natural_event_must_preserve_target_under_option_swap": True,
            "causal_operator_defined_only_after_event_freeze": True,
        },
        "evidence_policy": {
            "phase576_open_results_used_only_for_contract_repair": True,
            "phase576_internal_state_read": False,
            "sealed_split_read": False,
            "strict_mechanism_closure_claimed": False,
        },
        "open_cases_path": str(OPEN_CASES_PATH.relative_to(ROOT)),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_path": str(SEALED_COMMITMENT_PATH.relative_to(ROOT)),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
