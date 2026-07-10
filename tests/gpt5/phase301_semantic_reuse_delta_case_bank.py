#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase301_semantic_reuse_delta_case_bank"
PHASE = "Phase301"
SCHEMA_VERSION = "2.28.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]


OBJECTS = [
    {"object_id": "apple", "label": "apple", "category": "fruit", "subclass": "tree_fruit", "color": "red", "shape": "round", "taste": "sweet", "texture": "crisp", "part": "core", "use": "pie", "features": ["entity", "plant", "food", "fruit", "tree_fruit", "round", "crisp", "red_green", "pie"]},
    {"object_id": "banana", "label": "banana", "category": "fruit", "subclass": "tropical", "color": "yellow", "shape": "curved", "taste": "sweet", "texture": "soft", "part": "peel", "use": "smoothie", "features": ["entity", "plant", "food", "fruit", "tropical", "yellow", "curved", "soft", "peel"]},
    {"object_id": "orange", "label": "orange", "category": "fruit", "subclass": "citrus", "color": "orange", "shape": "round", "taste": "sweet-sour", "texture": "juicy", "part": "peel", "use": "juice", "features": ["entity", "plant", "food", "fruit", "citrus", "orange_color", "round", "juicy", "peel", "juice"]},
    {"object_id": "lemon", "label": "lemon", "category": "fruit", "subclass": "citrus", "color": "yellow", "shape": "oval", "taste": "sour", "texture": "juicy", "part": "peel", "use": "seasoning", "features": ["entity", "plant", "food", "fruit", "citrus", "yellow", "oval", "sour", "juicy", "peel", "seasoning"]},
    {"object_id": "lime", "label": "lime", "category": "fruit", "subclass": "citrus", "color": "green", "shape": "round", "taste": "sour", "texture": "juicy", "part": "peel", "use": "seasoning", "features": ["entity", "plant", "food", "fruit", "citrus", "green", "round", "sour", "juicy", "peel", "seasoning"]},
    {"object_id": "grape", "label": "grape", "category": "fruit", "subclass": "berry_like", "color": "purple", "shape": "round", "taste": "sweet", "texture": "juicy", "part": "skin", "use": "wine", "features": ["entity", "plant", "food", "fruit", "berry_like", "purple_green", "round", "juicy", "wine"]},
    {"object_id": "strawberry", "label": "strawberry", "category": "fruit", "subclass": "berry_like", "color": "red", "shape": "heart-shaped", "taste": "sweet", "texture": "soft", "part": "seeds", "use": "dessert", "features": ["entity", "plant", "food", "fruit", "berry_like", "red", "heart_shaped", "soft", "seeds", "dessert"]},
    {"object_id": "blueberry", "label": "blueberry", "category": "fruit", "subclass": "berry_like", "color": "blue", "shape": "round", "taste": "sweet", "texture": "soft", "part": "skin", "use": "muffin", "features": ["entity", "plant", "food", "fruit", "berry_like", "blue", "round", "soft", "muffin"]},
    {"object_id": "pear", "label": "pear", "category": "fruit", "subclass": "tree_fruit", "color": "green", "shape": "pear-shaped", "taste": "sweet", "texture": "grainy", "part": "core", "use": "snack", "features": ["entity", "plant", "food", "fruit", "tree_fruit", "green", "pear_shaped", "grainy", "core"]},
    {"object_id": "peach", "label": "peach", "category": "fruit", "subclass": "stone_fruit", "color": "pink", "shape": "round", "taste": "sweet", "texture": "soft", "part": "pit", "use": "cobbler", "features": ["entity", "plant", "food", "fruit", "stone_fruit", "pink", "round", "soft", "pit", "cobbler"]},
    {"object_id": "mango", "label": "mango", "category": "fruit", "subclass": "tropical", "color": "yellow", "shape": "oval", "taste": "sweet", "texture": "juicy", "part": "pit", "use": "smoothie", "features": ["entity", "plant", "food", "fruit", "tropical", "yellow", "oval", "sweet", "juicy", "pit"]},
    {"object_id": "pineapple", "label": "pineapple", "category": "fruit", "subclass": "tropical", "color": "yellow", "shape": "spiky", "taste": "sweet-sour", "texture": "juicy", "part": "rind", "use": "juice", "features": ["entity", "plant", "food", "fruit", "tropical", "yellow", "spiky", "sweet_sour", "juicy", "rind"]},
    {"object_id": "watermelon", "label": "watermelon", "category": "fruit", "subclass": "melon", "color": "green", "shape": "large", "taste": "sweet", "texture": "watery", "part": "rind", "use": "summer snack", "features": ["entity", "plant", "food", "fruit", "melon", "green", "large", "watery", "rind"]},
    {"object_id": "cherry", "label": "cherry", "category": "fruit", "subclass": "stone_fruit", "color": "red", "shape": "round", "taste": "sweet", "texture": "juicy", "part": "pit", "use": "dessert", "features": ["entity", "plant", "food", "fruit", "stone_fruit", "red", "round", "juicy", "pit", "dessert"]},
    {"object_id": "kiwi", "label": "kiwi", "category": "fruit", "subclass": "tropical", "color": "green", "shape": "oval", "taste": "sweet-sour", "texture": "soft", "part": "fuzzy skin", "use": "fruit salad", "features": ["entity", "plant", "food", "fruit", "tropical", "green", "oval", "sweet_sour", "soft", "fuzzy_skin"]},
    {"object_id": "carrot", "label": "carrot", "category": "vegetable", "subclass": "root_vegetable", "color": "orange", "shape": "long", "taste": "earthy", "texture": "crunchy", "part": "root", "use": "soup", "features": ["entity", "plant", "food", "vegetable", "root_vegetable", "orange_color", "long", "crunchy"]},
    {"object_id": "potato", "label": "potato", "category": "vegetable", "subclass": "tuber", "color": "brown", "shape": "round", "taste": "starchy", "texture": "soft", "part": "skin", "use": "fries", "features": ["entity", "plant", "food", "vegetable", "tuber", "brown", "round", "starchy"]},
    {"object_id": "chair", "label": "chair", "category": "furniture", "subclass": "seat", "color": "varied", "shape": "upright", "taste": "inedible", "texture": "hard", "part": "legs", "use": "sitting", "features": ["entity", "object", "furniture", "seat", "hard", "legs", "sitting"]},
    {"object_id": "stone", "label": "stone", "category": "mineral", "subclass": "rock", "color": "gray", "shape": "irregular", "taste": "inedible", "texture": "hard", "part": "surface", "use": "building", "features": ["entity", "object", "mineral", "rock", "gray", "hard", "irregular"]},
    {"object_id": "knife", "label": "knife", "category": "tool", "subclass": "cutting_tool", "color": "metal", "shape": "sharp", "taste": "inedible", "texture": "hard", "part": "blade", "use": "cutting", "features": ["entity", "object", "tool", "cutting_tool", "metal", "sharp", "blade"]},
]

PROMPT_TYPES = [
    {"prompt_type": "category", "field": "category", "template": "An {label} is a type of ___. Answer with one word only."},
    {"prompt_type": "subclass", "field": "subclass", "template": "Within food categories, an {label} is most associated with ___. Answer briefly."},
    {"prompt_type": "color", "field": "color", "template": "The usual color of a {label} is ___. Answer with the color only."},
    {"prompt_type": "shape", "field": "shape", "template": "A {label} is usually shaped like ___. Answer briefly."},
    {"prompt_type": "taste", "field": "taste", "template": "A {label} tastes mostly ___. Answer with one short phrase."},
    {"prompt_type": "texture", "field": "texture", "template": "The texture of a {label} is usually ___. Answer briefly."},
    {"prompt_type": "part", "field": "part", "template": "A notable part of a {label} is its ___. Answer briefly."},
    {"prompt_type": "use", "field": "use", "template": "A common use of a {label} is ___. Answer briefly."},
]

CONTRAST_PAIRS = [
    ("orange", "lemon", "shared", "citrus"),
    ("lemon", "lime", "shared", "citrus"),
    ("apple", "pear", "shared", "tree fruit"),
    ("strawberry", "blueberry", "shared", "berry"),
    ("banana", "mango", "shared", "tropical"),
    ("apple", "banana", "difference", "curved"),
    ("orange", "lemon", "difference", "sweet"),
    ("banana", "apple", "difference", "yellow"),
    ("fruit", "chair", "difference", "food"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def aliases(value: str) -> list[str]:
    text = str(value)
    parts = [text, text.replace("_", " "), text.replace("-", " ")]
    if text == "sweet-sour":
        parts += ["sour", "sweet and sour", "sweet-sour"]
    if text == "tree_fruit":
        parts += ["tree fruit"]
    if text == "berry_like":
        parts += ["berry", "berries"]
    if text == "tropical":
        parts += ["tropical fruit", "tropical"]
    if text == "fruit":
        parts += ["fruit"]
    if text == "yellow":
        parts += ["yellow"]
    if text == "orange":
        parts += ["orange"]
    return sorted(set(p for p in parts if p))


def object_by_id() -> dict[str, dict[str, Any]]:
    return {row["object_id"]: row for row in OBJECTS}


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    created = now()
    object_rows = [{**obj, "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": created} for obj in OBJECTS]
    attribute_rows: list[dict[str, Any]] = []
    for obj in OBJECTS:
        for prompt in PROMPT_TYPES:
            target = str(obj[prompt["field"]])
            case_id = f"phase301:attribute:{obj['object_id']}:{prompt['prompt_type']}"
            attribute_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": created,
                    "case_id": case_id,
                    "case_type": "semantic_attribute",
                    "object_id": obj["object_id"],
                    "object_label": obj["label"],
                    "category_id": obj["category"],
                    "subclass_id": obj["subclass"],
                    "attribute_type": prompt["prompt_type"],
                    "prompt_type": prompt["prompt_type"],
                    "prompt": prompt["template"].format(**obj),
                    "target": target,
                    "target_aliases": aliases(target),
                    "expected_pattern": "short",
                    "semantic_field": prompt["field"],
                }
            )
    objects = object_by_id()
    contrast_rows: list[dict[str, Any]] = []
    for left, right, mode, target in CONTRAST_PAIRS:
        if left == "fruit":
            left_label = "fruit"
        else:
            left_label = objects[left]["label"]
        right_label = objects[right]["label"]
        prompt = (
            f"An {left_label} and a {right_label} are both ___. Answer briefly."
            if mode == "shared"
            else f"Compared with a {right_label}, a {left_label} is more associated with ___. Answer briefly."
        )
        case_id = f"phase301:contrast:{left}:{right}:{mode}"
        contrast_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": created,
                "case_id": case_id,
                "case_type": "semantic_contrast",
                "object_id": left,
                "contrast_object_id": right,
                "object_label": left_label,
                "contrast_object_label": right_label,
                "category_id": objects[left]["category"] if left in objects else "fruit",
                "contrast_category_id": objects[right]["category"],
                "attribute_type": mode,
                "prompt_type": f"contrast_{mode}",
                "prompt": prompt,
                "target": target,
                "target_aliases": aliases(target),
                "expected_pattern": "short",
                "semantic_field": mode,
            }
        )
    plan_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for row in attribute_rows + contrast_rows:
            plan_rows.append({**row, "model": model, "plan_id": f"phase301:plan:{model}:{row['case_id']}"})
    return object_rows, attribute_rows, contrast_rows, plan_rows


def main() -> None:
    object_rows, attribute_rows, contrast_rows, plan_rows = build_rows()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "object_rows": len(object_rows),
        "attribute_case_rows": len(attribute_rows),
        "contrast_case_rows": len(contrast_rows),
        "full_test_plan_rows": len(plan_rows),
        "models": MODELS,
        "note": "Semantic reuse-delta case bank only; no model execution in this phase.",
    }
    write_jsonl(V2 / "phase301_semantic_object_rows.jsonl", object_rows)
    write_jsonl(V2 / "phase301_semantic_attribute_case_rows.jsonl", attribute_rows)
    write_jsonl(V2 / "phase301_semantic_contrast_case_rows.jsonl", contrast_rows)
    write_jsonl(V2 / "phase301_semantic_full_test_plan_rows.jsonl", plan_rows)
    write_json(V2 / "phase301_semantic_reuse_delta_case_bank_summary.json", summary)
    write_jsonl(OUT / "phase301_semantic_object_rows.jsonl", object_rows)
    write_jsonl(OUT / "phase301_semantic_attribute_case_rows.jsonl", attribute_rows)
    write_jsonl(OUT / "phase301_semantic_contrast_case_rows.jsonl", contrast_rows)
    write_jsonl(OUT / "phase301_semantic_full_test_plan_rows.jsonl", plan_rows)
    write_json(OUT / "phase301_semantic_reuse_delta_case_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
