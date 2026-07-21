#!/usr/bin/env python3
"""Freeze typed, no-candidate category behavior after Phase580 contract failure."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402
import phase580_open_category_protocol as p580  # noqa: E402


PHASE = "Phase581"
MODELS = p580.MODELS
OPEN_SPLITS = p580.OPEN_SPLITS
SEALED_SPLIT = p580.SEALED_SPLIT
SPLITS = OPEN_SPLITS + (SEALED_SPLIT,)
NOOP_REPEATS = p580.NOOP_REPEATS
SURFACES_PER_SPLIT = 8
FIXED_BATCH_SIZE = 32
MAX_NEW_TOKENS = 24
RELATIONS = ("culinary_produce_group", "functional_artifact_group")
RELATION_CATEGORIES = {
    "culinary_produce_group": ("fruit", "vegetable"),
    "functional_artifact_group": ("tool", "vehicle"),
}
MIN_SEMANTIC_ACCURACY = 0.90
MIN_STABLE_CASE_RATE = 0.90
MIN_REPEAT_EXACT_RATE = 0.99
MIN_STABLE_SURFACES_PER_OBJECT = 6
MIN_QUALIFIED_BY_RELATION_CATEGORY = {
    "culinary_produce_group": {"fruit": 10, "vegetable": 5},
    "functional_artifact_group": {"tool": 5, "vehicle": 5},
}

OUT_DIR = ROOT / "tests/gpt5/result/phase581_typed_category"
OPEN_CASES_PATH = OUT_DIR / "phase581_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase581_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase581_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase581_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase581_static_audit.json"


SPLIT_GROUP = p580.SPLIT_GROUP
OBJECT_GROUPS = p580.OBJECT_GROUPS
CATEGORY_ALIASES = p580.CATEGORY_ALIASES


PRODUCE_SURFACES: tuple[str, ...] = (
    "In ordinary cooking and grocery classification, which produce group does {object} belong to? Reply with one noun only.",
    "Classify {object} by its standard culinary produce group. Give one English noun.",
    "Complete with the broad grocery produce class: {object} is a",
    "Within everyday culinary produce taxonomy, the group for {object} is",
    "Name the usual cooking-and-shopping produce class of {object}. One noun only.",
    "For grocery organization, give the broad produce kind of {object} in one word.",
    "Fill one word: the ordinary culinary group containing {object} is ____.",
    "What broad produce kind is {object} in everyday food use? Answer with one noun.",
    "Use the normal supermarket produce grouping for {object}. Return one class noun.",
    "At the broad culinary level, identify the produce family of {object} with one noun.",
    "One-word grocery classification: the produce type of {object} is",
    "Complete the everyday food-group statement: {object} is a type of",
    "Give only the ordinary culinary produce category for {object}.",
    "In common cooking language, which produce class contains {object}? One noun.",
    "Supply the missing broad produce-group word: {object} belongs with ____.",
    "Classify {object} at the usual grocery produce level, using one noun only.",
    "What is the standard broad produce category of {object} in everyday shopping?",
    "Return one noun for the culinary produce group that includes {object}.",
    "Everyday grocery ontology: {object} has the broad produce type",
    "Complete with a common produce-group noun, not a variety name: {object} is a",
    "Name the broad cooking category containing {object}; use one word only.",
    "For an ordinary grocery list, the produce class of {object} is",
    "Give the parent culinary produce group for {object} as one English noun.",
    "Fill the category blank at grocery level: {object} -> ____.",
)


ARTIFACT_SURFACES: tuple[str, ...] = (
    "In ordinary functional classification of human-made objects, which broad group does {object} belong to? Reply with one noun only.",
    "Classify {object} by its standard broad functional artifact group. Give one English noun.",
    "Complete with the broad human-made object class: {object} is a",
    "Within everyday functional artifact taxonomy, the group for {object} is",
    "Name the usual broad use-based class of {object}. One noun only.",
    "For practical organization of human-made objects, give the broad kind of {object} in one word.",
    "Fill one word: the ordinary functional group containing {object} is ____.",
    "What broad human-made-object kind is {object} in everyday use? Answer with one noun.",
    "Use the normal functional grouping for {object}. Return one class noun.",
    "At the broad practical level, identify the artifact family of {object} with one noun.",
    "One-word functional classification: the broad type of {object} is",
    "Complete the everyday use-group statement: {object} is a type of",
    "Give only the ordinary broad functional category for {object}.",
    "In common practical language, which artifact class contains {object}? One noun.",
    "Supply the missing broad function-group word: {object} belongs with ____.",
    "Classify {object} at the usual broad functional level, using one noun only.",
    "What is the standard broad artifact category of {object} in everyday use?",
    "Return one noun for the functional object group that includes {object}.",
    "Everyday artifact ontology: {object} has the broad functional type",
    "Complete with a common function-group noun, not a subtype name: {object} is a",
    "Name the broad practical category containing {object}; use one word only.",
    "For an ordinary inventory, the functional class of {object} is",
    "Give the parent functional artifact group for {object} as one English noun.",
    "Fill the category blank at broad use level: {object} -> ____.",
)


SURFACES_BY_RELATION = {
    "culinary_produce_group": PRODUCE_SURFACES,
    "functional_artifact_group": ARTIFACT_SURFACES,
}
SPLIT_SURFACES = p580.SPLIT_SURFACES


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


def relation_for(category: str) -> str:
    for relation, categories in RELATION_CATEGORIES.items():
        if category in categories:
            return relation
    raise KeyError(category)


def prompt_for(item: dict[str, Any], surface_id: int) -> str:
    relation = relation_for(item["category"])
    return SURFACES_BY_RELATION[relation][surface_id].format(object=item["label"])


def materialize_row(
    tokenizers: dict[str, Any],
    item: dict[str, Any],
    split: str,
    surface_id: int,
) -> dict[str, Any]:
    category = item["category"]
    relation = relation_for(category)
    raw_prompt = prompt_for(item, surface_id)
    prompt_counts = {}
    target_token_ids = {}
    for model, tokenizer in tokenizers.items():
        prompt = render_chat(tokenizer, model, raw_prompt)
        prompt_counts[model] = len(
            tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )
        target_token_ids[model] = {
            alias: [
                int(token)
                for token in tokenizer(alias, add_special_tokens=False)["input_ids"]
            ]
            for alias in CATEGORY_ALIASES[category]
        }
    case_id = f"phase581_{split}_{item['object_id']}_surface{surface_id:02d}"
    return {
        "schema_version": "phase581_typed_category_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": case_id,
        "split": split,
        "object_group": SPLIT_GROUP[split],
        "object_id": item["object_id"],
        "object_label": item["label"],
        "relation": relation,
        "target_category": category,
        "target_aliases": list(CATEGORY_ALIASES[category]),
        "all_category_aliases": {
            key: list(value) for key, value in CATEGORY_ALIASES.items()
        },
        "surface_id": surface_id,
        "raw_prompt": raw_prompt,
        "prompt_token_count_by_model": prompt_counts,
        "target_token_ids_by_model": target_token_ids,
        "answer_word_present_in_raw_prompt": any(
            re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", raw_prompt, re.I)
            for aliases in CATEGORY_ALIASES.values()
            for alias in aliases
        ),
        "parent_domain_supplied": True,
        "answer_candidates_supplied": False,
        "context_fact_supplied": False,
        "observer_only": True,
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
                destination.append(
                    materialize_row(tokenizers, item, split, surface_id)
                )
    return open_rows, sealed_rows


def validate(
    open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    expected_per_split = 30 * SURFACES_PER_SPLIT
    audit = {
        "schema_version": "phase581_typed_category_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "relation_count_by_split": {
            split: dict(Counter(row["relation"] for row in rows if row["split"] == split))
            for split in SPLITS
        },
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows) - len(
            {(row["split"], row["raw_prompt"]) for row in rows}
        ),
        "answer_word_in_prompt_count": sum(
            row["answer_word_present_in_raw_prompt"] for row in rows
        ),
        "relation_category_mismatch_count": sum(
            row["target_category"] not in RELATION_CATEGORIES[row["relation"]]
            for row in rows
        ),
        "missing_target_token_count": sum(
            not ids
            for row in rows
            for model_map in row["target_token_ids_by_model"].values()
            for ids in model_map.values()
        ),
        "max_prompt_token_count": max(
            count
            for row in rows
            for count in row["prompt_token_count_by_model"].values()
        ),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
        "expected_case_count_per_split": expected_per_split,
    }
    expected_relations = {
        "culinary_produce_group": 18 * SURFACES_PER_SPLIT,
        "functional_artifact_group": 12 * SURFACES_PER_SPLIT,
    }
    audit["valid"] = bool(
        audit["registered_case_count"] == expected_per_split * len(SPLITS)
        and audit["open_case_count"] == expected_per_split * len(OPEN_SPLITS)
        and audit["sealed_case_count"] == expected_per_split
        and set(audit["case_count_by_split"].values()) == {expected_per_split}
        and all(
            counts == expected_relations
            for counts in audit["relation_count_by_split"].values()
        )
        and audit["max_prompt_token_count"] <= 128
        and all(
            audit[key] == 0
            for key in (
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
                "answer_word_in_prompt_count",
                "relation_category_mismatch_count",
                "missing_target_token_count",
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
            "schema_version": "phase581_typed_category_sealed_commitment.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "sealed_case_count": len(sealed_rows),
            "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
            "sealed_split_read_for_analysis": False,
        },
    )
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": "phase581_typed_category_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Typed-parent-domain category behavior without answer candidates",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "relations": list(RELATIONS),
        "relation_categories": {
            key: list(value) for key, value in RELATION_CATEGORIES.items()
        },
        "surfaces_per_split": SURFACES_PER_SPLIT,
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_gate": {
            "minimum_semantic_accuracy_each_split_relation": MIN_SEMANTIC_ACCURACY,
            "minimum_stable_case_rate_each_split_relation": MIN_STABLE_CASE_RATE,
            "minimum_repeat_exact_rate_each_split_relation": MIN_REPEAT_EXACT_RATE,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_by_relation_category_each_split": MIN_QUALIFIED_BY_RELATION_CATEGORY,
            "all_three_open_splits_must_pass_per_relation": True,
            "model_relation_specific_qualification": True,
        },
        "evidence_policy": {
            "phase580_results_not_relabelled": True,
            "parent_domain_supplied_but_answer_candidates_absent": True,
            "behavior_before_internal_state": True,
            "behavior_does_not_localize_parametric_storage": True,
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
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
