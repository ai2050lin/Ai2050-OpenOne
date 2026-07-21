#!/usr/bin/env python3
"""Freeze the Phase580 no-candidate immediate-category behavior denominator."""

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


PHASE = "Phase580"
MODELS = ("qwen3", "glm4", "deepseek7b")
OPEN_SPLITS = ("behavior_discovery", "behavior_confirmation", "heldout_objects")
SEALED_SPLIT = "sealed"
SPLITS = OPEN_SPLITS + (SEALED_SPLIT,)
SURFACES_PER_SPLIT = 8
NOOP_REPEATS = ("noop1", "noop2")
FIXED_BATCH_SIZE = 32
MAX_NEW_TOKENS = 8
MIN_SEMANTIC_ACCURACY = 0.90
MIN_STABLE_CASE_RATE = 0.90
MIN_REPEAT_EXACT_RATE = 0.99
MIN_STABLE_SURFACES_PER_OBJECT = 6
MIN_QUALIFIED_BY_CATEGORY = {
    "fruit": 10,
    "vegetable": 5,
    "tool": 5,
    "vehicle": 5,
}

OUT_DIR = ROOT / "tests/gpt5/result/phase580_open_category"
OPEN_CASES_PATH = OUT_DIR / "phase580_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase580_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase580_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase580_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase580_static_audit.json"
PUBLIC_OBJECTS_PATH = OUT_DIR / "phase580_public_object_bank.json"


def items(category: str, labels: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "object_id": label.replace(" ", "_"),
            "label": label,
            "category": category,
        }
        for label in labels
    )


OBJECT_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "A": (
        *items("fruit", (
            "apple", "banana", "orange", "lemon", "strawberry", "blueberry",
            "cherry", "pear", "grape", "watermelon", "mango", "peach",
        )),
        *items("vegetable", (
            "carrot", "broccoli", "celery", "spinach", "onion", "cabbage",
        )),
        *items("tool", (
            "hammer", "screwdriver", "wrench", "pliers", "chisel", "saw",
        )),
        *items("vehicle", (
            "bicycle", "airplane", "truck", "bus", "train", "motorcycle",
        )),
    ),
    "B": (
        *items("fruit", (
            "lime", "raspberry", "blackberry", "plum", "pineapple", "kiwi",
            "pomegranate", "papaya", "apricot", "cranberry", "nectarine",
            "tangerine",
        )),
        *items("vegetable", (
            "potato", "cauliflower", "radish", "lettuce", "asparagus", "turnip",
        )),
        *items("tool", (
            "drill", "shovel", "rake", "scissors", "axe", "crowbar",
        )),
        *items("vehicle", (
            "car", "scooter", "helicopter", "boat", "van", "taxi",
        )),
    ),
    "C": (
        *items("fruit", (
            "avocado", "guava", "cantaloupe", "honeydew", "mulberry",
            "gooseberry", "boysenberry", "durian", "kumquat", "quince",
            "persimmon", "mandarin",
        )),
        *items("vegetable", (
            "beet", "kale", "leek", "garlic", "parsnip", "artichoke",
        )),
        *items("tool", (
            "mallet", "file", "clamp", "trowel", "level", "vise",
        )),
        *items("vehicle", (
            "sedan", "tram", "ferry", "canoe", "ambulance", "bulldozer",
        )),
    ),
}


SPLIT_GROUP = {
    "behavior_discovery": "A",
    "behavior_confirmation": "A",
    "heldout_objects": "B",
    "sealed": "C",
}


SURFACE_TEMPLATES: tuple[str, ...] = (
    "Complete this everyday classification with one noun: {object} is a kind of",
    "Name the nearest common category for {object}. Reply with one noun only.",
    "Taxonomic completion, using one ordinary noun: {object} belongs to the category",
    "Classify {object} by its usual immediate kind. Answer with one noun.",
    "What ordinary kind of thing is {object}? Return only the category noun.",
    "Give the direct everyday class of {object}, using one noun and no explanation.",
    "Dictionary-style class lookup for {object}: provide its nearest common kind.",
    "Fill the blank with one ordinary category noun: {object} is a type of ___.",
    "State the usual immediate category of {object}. Use one noun only.",
    "One-noun classification task. The item is {object}; its common kind is",
    "Identify the closest everyday class for {object}, without an explanation.",
    "Complete with a direct category noun rather than a description: {object} is a",
    "For {object}, return its ordinary immediate type as one noun.",
    "Common-knowledge taxonomy: the nearest category containing {object} is",
    "Write only the direct class noun that normally describes {object}.",
    "Categorize {object} at the nearest familiar level. One noun:",
    "Give a one-word hypernym for {object} at the usual everyday level.",
    "Everyday ontology lookup: {object} has the immediate class",
    "Supply the ordinary parent category of {object}; answer with one noun.",
    "Use one familiar class noun to complete: {object} is an example of",
    "What is the nearest everyday parent kind of {object}? One noun only.",
    "Return the direct common category for the item {object}, with no explanation.",
    "Immediate-kind completion for {object}: answer using one ordinary noun.",
    "At the nearest common taxonomy level, {object} is classified as a",
)


SPLIT_SURFACES = {
    "behavior_discovery": tuple(range(0, 8)),
    "behavior_confirmation": tuple(range(8, 16)),
    "heldout_objects": tuple(range(16, 24)),
    "sealed": tuple(range(0, 8)),
}


CATEGORY_ALIASES = {
    "fruit": ("fruit", "fruits"),
    "vegetable": ("vegetable", "vegetables"),
    "tool": ("tool", "tools"),
    "vehicle": ("vehicle", "vehicles", "transport", "transportation"),
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


def prompt_for(item: dict[str, Any], surface_id: int) -> str:
    return SURFACE_TEMPLATES[surface_id].format(object=item["label"])


def materialize_row(
    tokenizers: dict[str, Any],
    item: dict[str, Any],
    split: str,
    surface_id: int,
) -> dict[str, Any]:
    raw_prompt = prompt_for(item, surface_id)
    category = item["category"]
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
    case_id = f"phase580_{split}_{item['object_id']}_surface{surface_id:02d}"
    return {
        "schema_version": "phase580_open_category_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": case_id,
        "split": split,
        "object_group": SPLIT_GROUP[split],
        "object_id": item["object_id"],
        "object_label": item["label"],
        "relation": "immediate_everyday_category",
        "target_category": category,
        "target_aliases": list(CATEGORY_ALIASES[category]),
        "all_category_aliases": {
            key: list(value) for key, value in CATEGORY_ALIASES.items()
        },
        "surface_id": surface_id,
        "raw_prompt": raw_prompt,
        "prompt_token_count_by_model": prompt_counts,
        "target_token_ids_by_model": target_token_ids,
        "answer_word_present_in_raw_prompt": bool(
            re.search(rf"(?<!\w){re.escape(category)}(?!\w)", raw_prompt, re.I)
        ),
        "natural_behavior_without_supplied_fact": True,
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
        group = OBJECT_GROUPS[SPLIT_GROUP[split]]
        for item in group:
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
    group_objects = {
        group: {item["object_id"] for item in values}
        for group, values in OBJECT_GROUPS.items()
    }
    audit = {
        "schema_version": "phase580_open_category_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "category_count_by_group": {
            group: dict(Counter(item["category"] for item in values))
            for group, values in OBJECT_GROUPS.items()
        },
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows) - len(
            {(row["split"], row["raw_prompt"]) for row in rows}
        ),
        "answer_word_in_prompt_count": sum(
            row["answer_word_present_in_raw_prompt"] for row in rows
        ),
        "missing_target_token_count": sum(
            not ids
            for row in rows
            for model_map in row["target_token_ids_by_model"].values()
            for ids in model_map.values()
        ),
        "cross_group_object_overlap_count": sum(
            len(group_objects[left] & group_objects[right])
            for index, left in enumerate(sorted(group_objects))
            for right in sorted(group_objects)[index + 1:]
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
    expected_categories = {"fruit": 12, "vegetable": 6, "tool": 6, "vehicle": 6}
    audit["valid"] = bool(
        audit["registered_case_count"] == expected_per_split * len(SPLITS)
        and audit["open_case_count"] == expected_per_split * len(OPEN_SPLITS)
        and audit["sealed_case_count"] == expected_per_split
        and set(audit["case_count_by_split"].values()) == {expected_per_split}
        and all(
            counts == expected_categories
            for counts in audit["category_count_by_group"].values()
        )
        and audit["max_prompt_token_count"] <= 128
        and all(
            audit[key] == 0
            for key in (
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
                "answer_word_in_prompt_count",
                "missing_target_token_count",
                "cross_group_object_overlap_count",
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
            "schema_version": "phase580_open_category_sealed_commitment.v1",
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
            "schema_version": "phase580_open_category_public_objects.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "open_groups": {"A": OBJECT_GROUPS["A"], "B": OBJECT_GROUPS["B"]},
            "sealed_group_object_count": len(OBJECT_GROUPS["C"]),
            "category_counts": {"fruit": 12, "vegetable": 6, "tool": 6, "vehicle": 6},
        },
    )
    write_json(AUDIT_PATH, audit)
    protocol = {
        "schema_version": "phase580_open_category_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "No-candidate immediate-category behavior denominator",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "surfaces_per_split": SURFACES_PER_SPLIT,
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_gate": {
            "minimum_semantic_accuracy_each_split": MIN_SEMANTIC_ACCURACY,
            "minimum_stable_case_rate_each_split": MIN_STABLE_CASE_RATE,
            "minimum_repeat_exact_rate_each_split": MIN_REPEAT_EXACT_RATE,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_by_category_each_split": MIN_QUALIFIED_BY_CATEGORY,
            "all_three_open_splits_must_pass": True,
            "model_specific_qualification": True,
        },
        "evidence_policy": {
            "behavior_before_internal_state": True,
            "answer_candidates_supplied": False,
            "context_fact_supplied": False,
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
