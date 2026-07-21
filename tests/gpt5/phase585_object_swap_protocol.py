#!/usr/bin/env python3
"""Freeze the Phase585 label-free natural object-response denominator."""

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


PHASE = "Phase585"
MODELS = ("qwen3", "glm4", "deepseek7b")
OPEN_SPLITS = ("behavior_discovery", "behavior_confirmation", "heldout_objects")
SEALED_SPLIT = "sealed"
SPLITS = OPEN_SPLITS + (SEALED_SPLIT,)
RELATIONS = ("ordinary_origin", "primary_function")
SURFACES_PER_SPLIT = 8
NOOP_REPEATS = ("noop1", "noop2")
FIXED_BATCH_SIZE = 32
MAX_NEW_TOKENS = 14
MIN_SEMANTIC_ACCURACY = 0.85
MIN_STABLE_CASE_RATE = 0.80
MIN_REPEAT_EXACT_RATE = 0.99
MIN_STABLE_SURFACES_PER_OBJECT = 6
MIN_QUALIFIED_BY_SPLIT_GROUP = {
    "behavior_discovery": {"fruit": 8, "near_food_plant": 4, "tool": 4, "vehicle": 4},
    "behavior_confirmation": {"fruit": 8, "near_food_plant": 4, "tool": 4, "vehicle": 4},
    "heldout_objects": {"fruit": 3, "near_food_plant": 1, "tool": 1, "vehicle": 1},
}

OUT_DIR = ROOT / "tests/gpt5/result/phase585_object_swap"
OPEN_CASES_PATH = OUT_DIR / "phase585_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase585_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase585_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase585_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase585_static_audit.json"
PUBLIC_OBJECTS_PATH = OUT_DIR / "phase585_public_object_bank.json"


ORIGIN_NATURAL = (
    "plant",
    "grow",
    "grown",
    "growth",
    "crop",
    "agricultur",
    "farm",
    "tree",
    "vine",
    "soil",
    "ground",
)
ORIGIN_MANUFACTURED = (
    "manufactur",
    "factory",
    "made",
    "built",
    "construct",
    "assembl",
    "human-made",
    "human made",
    "people",
)
FUNCTION_FOOD = (
    "eat",
    "eaten",
    "eating",
    "food",
    "cook",
    "cooking",
    "consume",
    "consumption",
    "ingredient",
    "nutrition",
)
FUNCTION_TRANSPORT = (
    "transport",
    "travel",
    "ride",
    "move",
    "carry",
    "commut",
    "passenger",
)


def obj(
    object_id: str,
    label: str,
    semantic_group: str,
    function_answer: str,
    function_aliases: tuple[str, ...],
) -> dict[str, Any]:
    natural = semantic_group in {"fruit", "near_food_plant"}
    return {
        "object_id": object_id,
        "label": label,
        "semantic_group": semantic_group,
        "origin_answer": "grown by a plant" if natural else "manufactured by people",
        "origin_aliases": list(ORIGIN_NATURAL if natural else ORIGIN_MANUFACTURED),
        "origin_forbidden_aliases": list(ORIGIN_MANUFACTURED if natural else ORIGIN_NATURAL),
        "function_answer": function_answer,
        "function_aliases": list(function_aliases),
    }


def fruit(object_id: str, label: str | None = None) -> dict[str, Any]:
    return obj(object_id, label or object_id, "fruit", "eaten as food", FUNCTION_FOOD)


def near_food(object_id: str, label: str | None = None) -> dict[str, Any]:
    return obj(
        object_id,
        label or object_id,
        "near_food_plant",
        "eaten as food",
        FUNCTION_FOOD,
    )


def tool(
    object_id: str, answer: str, aliases: tuple[str, ...], label: str | None = None
) -> dict[str, Any]:
    return obj(object_id, label or object_id, "tool", answer, aliases)


def vehicle(
    object_id: str, answer: str, aliases: tuple[str, ...], label: str | None = None
) -> dict[str, Any]:
    return obj(
        object_id,
        label or object_id,
        "vehicle",
        answer,
        tuple(dict.fromkeys((*FUNCTION_TRANSPORT, *aliases))),
    )


OBJECT_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "A": (
        fruit("apple"), fruit("banana"), fruit("orange"), fruit("lemon"),
        fruit("strawberry"), fruit("grape"), fruit("watermelon"),
        fruit("pineapple"), fruit("mango"), fruit("pear"),
        near_food("carrot"), near_food("potato"), near_food("broccoli"),
        near_food("spinach"), near_food("onion"),
        tool("hammer", "driving nails", ("nail", "strik", "pound")),
        tool("screwdriver", "turning screws", ("turn", "fasten", "tighten", "loosen")),
        tool("wrench", "turning nuts and bolts", ("bolt", "nut", "tighten", "loosen", "turn")),
        tool("saw", "cutting material", ("cut",)),
        tool("shovel", "digging soil", ("dig", "soil", "scoop")),
        vehicle("car", "transporting people", ("driv", "road")),
        vehicle("bus", "transporting passengers", ("public", "route")),
        vehicle("bicycle", "riding for transport", ("cycl", "pedal")),
        vehicle("train", "rail transport", ("rail",)),
        vehicle("airplane", "air travel", ("fly", "flight")),
    ),
    "B": (
        fruit("lime"), fruit("peach"), fruit("cherry"), fruit("blueberry"),
        near_food("celery"), near_food("cabbage"),
        tool("scissors", "cutting material", ("cut", "trim", "snip")),
        tool("drill", "making holes", ("hole", "bor")),
        vehicle("boat", "water transport", ("sail", "water")),
        vehicle("truck", "transporting goods", ("cargo", "goods", "haul")),
    ),
    "C": (
        fruit("plum"), fruit("kiwi"), fruit("papaya"), fruit("pomegranate"),
        near_food("radish"), near_food("lettuce"),
        tool("pliers", "gripping objects", ("grip", "hold", "bend", "wire")),
        tool("axe", "chopping wood", ("chop", "split", "cut", "wood")),
        vehicle("van", "transporting people or goods", ("driv", "cargo", "goods")),
        vehicle("helicopter", "air transport", ("fly", "flight", "air")),
    ),
}


SPLIT_GROUP = {
    "behavior_discovery": "A",
    "behavior_confirmation": "A",
    "heldout_objects": "B",
    "sealed": "C",
}


SURFACE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("short_answer", "Object: {object}. Requested aspect: {field}. Give a short answer."),
    ("short_answer", "Using ordinary real-world knowledge, what is the {field} of {object}? Answer briefly."),
    ("short_answer", "For {object}, state its {field} in at most five words."),
    ("completion", "Complete this everyday-knowledge entry with a short phrase: {field} of {object}:"),
    ("short_answer", "Consider the ordinary item {object}. What is its {field}? No explanation."),
    ("completion", "Everyday fact completion. Item = {object}; {field} ="),
    ("short_answer", "Give only a concise phrase for the {field} of {object}."),
    ("completion", "Fill the requested fact using common knowledge: {object} — {field}:"),
    ("short_answer", "State, without explanation, the {field} normally associated with {object}."),
    ("completion", "Common-knowledge record for {object}. Complete its {field}:"),
    ("short_answer", "What short phrase best describes the {field} of {object}?"),
    ("completion", "Finish this factual slot: object {object}; requested {field}; answer"),
    ("short_answer", "In ordinary life, identify the {field} of {object} using a few words."),
    ("completion", "Knowledge lookup: {field}({object}) ="),
    ("short_answer", "Take {object} as the subject. Report its {field} concisely."),
    ("completion", "One short factual completion is required. {object}; {field}:"),
    ("short_answer", "Using everyday knowledge rather than a special definition, give the {field} of {object}."),
    ("completion", "Complete the ordinary fact about {object}: its {field} is"),
    ("short_answer", "For the subject {object}, what is the usual {field}? Use a short phrase."),
    ("completion", "Factual field completion — subject: {object}; field: {field}; value:"),
    ("short_answer", "Answer in no more than five words: what is the {field} of {object}?"),
    ("completion", "Ordinary knowledge says the {field} of {object} is"),
    ("short_answer", "Name the typical {field} for {object}, without adding an explanation."),
    ("completion", "Supply a concise value for this record: {object} / {field} /"),
)


FIELD_PHRASES = {
    "ordinary_origin": (
        "broad source",
        "usual origin",
        "ordinary production path",
        "way it normally comes into existence",
    ),
    "primary_function": (
        "main practical purpose",
        "ordinary use",
        "typical role",
        "common function",
    ),
}


SPLIT_SURFACES = {
    "behavior_discovery": tuple(range(0, 8)),
    "behavior_confirmation": tuple(range(8, 16)),
    "heldout_objects": tuple(range(16, 24)),
    "sealed": tuple(range(0, 8)),
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


def aliases_for(item: dict[str, Any], relation: str) -> tuple[list[str], list[str]]:
    if relation == "ordinary_origin":
        return item["origin_aliases"], item["origin_forbidden_aliases"]
    return item["function_aliases"], []


def canonical_answer(item: dict[str, Any], relation: str) -> str:
    return str(item["origin_answer" if relation == "ordinary_origin" else "function_answer"])


def fragment_present(text: str, fragment: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(fragment.casefold())}\w*",
            text.casefold(),
        )
    )


def prompt_for(
    item: dict[str, Any], relation: str, split: str, surface_id: int
) -> tuple[str, str, str]:
    if surface_id not in SPLIT_SURFACES[split]:
        raise ValueError(f"Surface {surface_id} is not frozen for {split}")
    interface, template = SURFACE_TEMPLATES[surface_id]
    field = FIELD_PHRASES[relation][surface_id % len(FIELD_PHRASES[relation])]
    prompt = template.format(object=item["label"], field=field)
    return prompt, interface, field


def materialize_row(
    tokenizers: dict[str, Any],
    item: dict[str, Any],
    relation: str,
    split: str,
    surface_id: int,
) -> dict[str, Any]:
    raw_prompt, interface, field = prompt_for(item, relation, split, surface_id)
    target_aliases, forbidden_aliases = aliases_for(item, relation)
    prompt_counts: dict[str, int] = {}
    object_token_ids: dict[str, list[int]] = {}
    for model, tokenizer in tokenizers.items():
        rendered = render_chat(tokenizer, model, raw_prompt)
        prompt_counts[model] = len(tokenizer(rendered, add_special_tokens=True)["input_ids"])
        object_token_ids[model] = [
            int(token)
            for token in tokenizer(item["label"], add_special_tokens=False)["input_ids"]
        ]
    case_id = (
        f"phase585_{split}_{item['object_id']}_{relation}_surface{surface_id:02d}"
    )
    prompt_without_object = re.sub(
        rf"(?<!\w){re.escape(item['label'])}(?!\w)",
        "<object>",
        raw_prompt,
        flags=re.I,
    )
    return {
        "schema_version": "phase585_object_swap_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": case_id,
        "split": split,
        "object_group": SPLIT_GROUP[split],
        "object_id": item["object_id"],
        "object_label": item["label"],
        "semantic_group": item["semantic_group"],
        "relation": relation,
        "surface_id": surface_id,
        "interface": interface,
        "field_phrase": field,
        "raw_prompt": raw_prompt,
        "canonical_answer": canonical_answer(item, relation),
        "target_aliases": list(target_aliases),
        "forbidden_aliases": list(forbidden_aliases),
        "target_alias_in_prompt": any(
            fragment_present(prompt_without_object, alias) for alias in target_aliases
        ),
        "category_label_in_prompt": bool(
            re.search(r"(?<!\w)(fruit|vegetable|tool|vehicle)(?!\w)", raw_prompt, re.I)
        ),
        "prompt_token_count_by_model": prompt_counts,
        "object_token_ids_by_model": object_token_ids,
        "natural_behavior_without_supplied_fact": True,
        "category_label_used_as_internal_coordinate": False,
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
            for relation in RELATIONS:
                for surface_id in SPLIT_SURFACES[split]:
                    destination.append(
                        materialize_row(tokenizers, item, relation, split, surface_id)
                    )
    return open_rows, sealed_rows


def validate(
    open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    expected = {
        "behavior_discovery": 25 * len(RELATIONS) * SURFACES_PER_SPLIT,
        "behavior_confirmation": 25 * len(RELATIONS) * SURFACES_PER_SPLIT,
        "heldout_objects": 10 * len(RELATIONS) * SURFACES_PER_SPLIT,
        "sealed": 10 * len(RELATIONS) * SURFACES_PER_SPLIT,
    }
    group_objects = {
        group: {item["object_id"] for item in items}
        for group, items in OBJECT_GROUPS.items()
    }
    audit = {
        "schema_version": "phase585_object_swap_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "expected_case_count_by_split": expected,
        "object_count_by_group": {group: len(items) for group, items in OBJECT_GROUPS.items()},
        "semantic_group_count_by_group": {
            group: dict(Counter(item["semantic_group"] for item in items))
            for group, items in OBJECT_GROUPS.items()
        },
        "relation_count": dict(Counter(row["relation"] for row in rows)),
        "interface_count": dict(Counter(row["interface"] for row in rows)),
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows)
        - len({(row["split"], row["raw_prompt"]) for row in rows}),
        "target_alias_in_prompt_count": sum(row["target_alias_in_prompt"] for row in rows),
        "category_label_in_prompt_count": sum(row["category_label_in_prompt"] for row in rows),
        "empty_alias_count": sum(not row["target_aliases"] for row in rows),
        "empty_object_tokenization_count": sum(
            not ids for row in rows for ids in row["object_token_ids_by_model"].values()
        ),
        "cross_group_object_overlap_count": sum(
            len(group_objects[left] & group_objects[right])
            for index, left in enumerate(sorted(group_objects))
            for right in sorted(group_objects)[index + 1 :]
        ),
        "max_prompt_token_count": max(
            count
            for row in rows
            for count in row["prompt_token_count_by_model"].values()
        ),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    expected_group_counts = {
        "A": {"fruit": 10, "near_food_plant": 5, "tool": 5, "vehicle": 5},
        "B": {"fruit": 4, "near_food_plant": 2, "tool": 2, "vehicle": 2},
        "C": {"fruit": 4, "near_food_plant": 2, "tool": 2, "vehicle": 2},
    }
    audit["valid"] = bool(
        audit["registered_case_count"] == 1120
        and audit["open_case_count"] == 960
        and audit["sealed_case_count"] == 160
        and audit["case_count_by_split"] == expected
        and audit["semantic_group_count_by_group"] == expected_group_counts
        and audit["max_prompt_token_count"] <= 128
        and all(
            audit[key] == 0
            for key in (
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
                "target_alias_in_prompt_count",
                "category_label_in_prompt_count",
                "empty_alias_count",
                "empty_object_tokenization_count",
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
            "schema_version": "phase585_object_swap_sealed_commitment.v1",
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
            "schema_version": "phase585_object_swap_public_objects.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "open_groups": {"A": OBJECT_GROUPS["A"], "B": OBJECT_GROUPS["B"]},
            "sealed_group_object_count": len(OBJECT_GROUPS["C"]),
        },
    )
    write_json(AUDIT_PATH, audit)
    frozen = {
        "schema_version": "phase585_object_swap_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Label-free natural object response denominator",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "relations": list(RELATIONS),
        "surfaces_per_split": SURFACES_PER_SPLIT,
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "behavior_gate": {
            "relation_specific_qualification": True,
            "minimum_semantic_accuracy_each_split_relation": MIN_SEMANTIC_ACCURACY,
            "minimum_stable_case_rate_each_split_relation": MIN_STABLE_CASE_RATE,
            "minimum_repeat_exact_rate_each_split_relation": MIN_REPEAT_EXACT_RATE,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_by_split_group": MIN_QUALIFIED_BY_SPLIT_GROUP,
            "all_three_open_splits_must_pass": True,
        },
        "evidence_policy": {
            "behavior_before_internal_state": True,
            "category_label_used_as_internal_coordinate": False,
            "answer_candidates_supplied": False,
            "context_fact_supplied": False,
            "automatic_alias_gate_only": True,
            "unrecoverable_outputs_are_not_posthoc_relabelled": True,
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
    write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
