#!/usr/bin/env python3
"""Register a fresh 16-task copy-boundary protocol matrix."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase330_nine_family_case_bank import MODELS  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402


PHASE = "Phase343"
SCHEMA_VERSION = "19.0.0"
ROUND_DEFAULT = "copy_boundary_protocol_qualification"
OUT = ROOT / "tests/gpt5/result/phase343_copy_boundary_protocol"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("format_a", "format_b", "format_c")
TASKS = (
    ("random_label_copy", "explicit_copy"),
    ("digit_copy", "explicit_copy"),
    ("arbitrary_symbol_relay", "explicit_copy"),
    ("cross_sentence_pointer", "explicit_copy"),
    ("multi_token_phrase_copy", "explicit_copy"),
    ("delayed_copy", "explicit_copy"),
    ("key_value_read", "copy_neighbor"),
    ("object_name_relay", "copy_neighbor"),
    ("field_extraction", "copy_neighbor"),
    ("material_relation_binding", "noncopy_control"),
    ("attribute_relation_binding", "noncopy_control"),
    ("semantic_classification", "noncopy_control"),
    ("singular_agreement", "noncopy_control"),
    ("direct_entailment", "noncopy_control"),
    ("token_transformation", "noncopy_control"),
    ("non_source_answer", "noncopy_control"),
)

OBJECTS = (
    "anchor", "barrel", "camera", "drill", "envelope", "furnace", "guitar", "hammer", "island",
    "jar", "kite", "ladder", "magnet", "nozzle", "oven", "piano", "quill", "router",
)
LABELS = (
    "varda", "bexon", "clyra", "dorin", "elvex", "faryn", "goral", "huxen", "ivora",
    "jalen", "krynn", "lumar", "mivon", "narel", "orvix", "pelyn", "quorin", "rhyza",
)
DIGITS = (
    "4817", "2059", "7364", "9182", "3506", "6421", "1748", "8295", "5630",
    "2974", "6843", "1359", "7420", "9061", "3185", "5702", "8649", "4216",
)
SYMBOLS = (
    "K7Q", "V2M", "R9X", "B4Z", "T6N", "F3W", "J8C", "P5L", "D1Y",
    "H7S", "Q2A", "M9E", "X4K", "C6R", "N3V", "G8P", "L5T", "Z1F",
)
PHRASES = (
    "silver pine", "quiet harbor", "amber field", "winter bridge", "copper moon", "gentle river",
    "open valley", "bright cedar", "hidden lake", "soft thunder", "green lantern", "northern trail",
    "crystal shore", "silent meadow", "golden compass", "rapid current", "blue orchard", "stone garden",
)
MATERIALS = (
    "steel", "cedar", "glass", "rubber", "paper", "iron", "maple", "bronze", "granite",
    "clay", "nylon", "aluminum", "cobalt", "copper", "ceramic", "oak", "silver", "plastic",
)
ATTRIBUTES = (
    "heavy", "sealed", "digital", "corded", "folded", "heated", "tuned", "balanced", "remote",
    "glazed", "flying", "extended", "magnetic", "narrow", "baking", "musical", "pointed", "wireless",
)
CATEGORIES = (
    "tool", "container", "device", "tool", "document", "machine", "instrument", "tool", "place",
    "container", "toy", "structure", "object", "part", "appliance", "instrument", "tool", "device",
)
NAMES = (
    "Ari", "Bea", "Cole", "Dina", "Eli", "Fia", "Gus", "Hope", "Ivan",
    "Jade", "Kian", "Lena", "Milo", "Nia", "Oren", "Paz", "Rami", "Sia",
)
NUMBER_WORDS = (
    "four", "six", "eight", "five", "seven", "nine", "three", "ten", "eleven",
    "twelve", "fourteen", "fifteen", "sixteen", "eighteen", "twenty", "two", "thirteen", "seventeen",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def split_for(index: int) -> str:
    if index < 9:
        return "discovery"
    if index < 13:
        return "calibration"
    if index < 16:
        return "heldout"
    return "private_heldout"


def task_for(task_id: str, i: int) -> dict[str, Any]:
    obj, label, digit, symbol = OBJECTS[i], LABELS[i], DIGITS[i], SYMBOLS[i]
    instruction = "Return exactly the requested answer and nothing else."
    if task_id == "random_label_copy":
        context, question, target, distractors = f"The random label assigned to the {obj} is {label}.", f"What is the random label for the {obj}?", label, [LABELS[(i + 1) % 18], "unknown"]
    elif task_id == "digit_copy":
        context, question, target, distractors = f"The serial number for the {obj} is {digit}.", f"Return the serial number for the {obj}.", digit, [DIGITS[(i + 1) % 18], "0000"]
    elif task_id == "arbitrary_symbol_relay":
        context, question, target, distractors = f"The relay symbol attached to the {obj} is {symbol}.", f"Relay the symbol for the {obj}.", symbol, [SYMBOLS[(i + 1) % 18], "NONE"]
    elif task_id == "cross_sentence_pointer":
        context = f"The primary pointer is {label}. A secondary note mentions {LABELS[(i + 1) % 18]}. The active pointer remains the primary pointer."
        question, target, distractors = "What is the active pointer?", label, [LABELS[(i + 1) % 18], "unknown"]
    elif task_id == "multi_token_phrase_copy":
        context, question, target, distractors = f"The registered phrase for the {obj} is '{PHRASES[i]}'.", f"Return the full registered phrase for the {obj}.", PHRASES[i], [PHRASES[(i + 1) % 18], "unknown phrase"]
    elif task_id == "delayed_copy":
        context = f"Remember the code {symbol}. The {obj} is {ATTRIBUTES[i]}. The unrelated serial is {DIGITS[(i + 1) % 18]}."
        question, target, distractors = "What code were you asked to remember?", symbol, [DIGITS[(i + 1) % 18], "NONE"]
    elif task_id == "key_value_read":
        context = f"In the map, key-{i + 31} has value {label}, while key-{i + 32} has value {LABELS[(i + 1) % 18]}."
        question, target, distractors = f"What value belongs to key-{i + 31}?", label, [LABELS[(i + 1) % 18], "unknown"]
    elif task_id == "object_name_relay":
        context, question, target, distractors = f"Object {i + 51} is named {PHRASES[i]}.", f"What is the full name of object {i + 51}?", PHRASES[i], [PHRASES[(i + 1) % 18], "unknown object"]
    elif task_id == "field_extraction":
        context = f"Record {i + 71}: label={label}; serial={digit}; symbol={symbol}."
        question, target, distractors = f"What is the label field in record {i + 71}?", label, [digit, symbol]
    elif task_id == "material_relation_binding":
        context, question, target, distractors = f"The {obj} is made from {MATERIALS[i]} and is {ATTRIBUTES[i]}.", f"What material is the {obj} made from?", MATERIALS[i], [ATTRIBUTES[i], "unknown"]
    elif task_id == "attribute_relation_binding":
        context, question, target, distractors = f"The {obj} is made from {MATERIALS[i]} and is {ATTRIBUTES[i]}.", f"What attribute describes the {obj}?", ATTRIBUTES[i], [MATERIALS[i], "unknown"]
    elif task_id == "semantic_classification":
        context, question, target, distractors = f"A {obj} is being considered by its ordinary meaning and use.", f"Which category best fits a {obj}?", CATEGORIES[i], ["animal", "emotion"]
    elif task_id == "singular_agreement":
        context, question, target, distractors = f"Write a standard sentence about one {obj}.", f"Complete: The {obj} ___ ready.", "is", ["are", "unknown"]
    elif task_id == "direct_entailment":
        group = f"circle{i + 91}"
        context, question, target, distractors = f"Every member of {group} is calm. {NAMES[i]} belongs to {group}.", f"Is {NAMES[i]} calm?", "yes", ["no", "unknown"]
    elif task_id == "token_transformation":
        context, question, target, distractors = f"The source token is {label}.", "Return the source token in uppercase.", label.upper(), [label, LABELS[(i + 1) % 18].upper()]
    elif task_id == "non_source_answer":
        left = i % 7 + 1
        right = int(NUMBER_WORDS[i] in {"four", "six", "eight", "five", "seven", "nine", "three"})
        sums = ("four", "six", "eight", "five", "seven", "nine", "three", "ten", "eleven", "twelve", "fourteen", "fifteen", "sixteen", "eighteen", "twenty", "two", "thirteen", "seventeen")
        target = sums[i]
        # The arithmetic expression is fixed per item; the answer word is absent from the prompt.
        pairs = ((1, 3), (2, 4), (3, 5), (1, 4), (2, 5), (4, 5), (1, 2), (4, 6), (5, 6), (5, 7), (6, 8), (7, 8), (7, 9), (8, 10), (9, 11), (1, 1), (6, 7), (8, 9))
        left, right = pairs[i]
        context, question, distractors = f"Compute {left} plus {right} without copying a supplied answer.", "What is the sum, written as an English word?", ["zero", "one"]
    else:
        raise KeyError(task_id)
    return {
        "context": context, "question": question, "instruction": instruction,
        "target": target, "target_aliases": [target], "distractors": distractors,
        "source_fragment": context, "query_fragment": question,
    }


def render(task: dict[str, Any], template: str) -> str:
    if template == "format_a":
        return f"Information: {task['context']}\nTask: {task['question']}\nRule: {task['instruction']}\nResponse:"
    if template == "format_b":
        return f"Use this information: {task['context']}\n{task['instruction']}\nPrompt: {task['question']}\nResult:"
    if template == "format_c":
        return f"Reference: {task['context']}\nQuery: {task['question']}\nOutput rule: {task['instruction']}\nFinal answer:"
    raise KeyError(template)


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    rows = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for task_id, task_class in TASKS:
            for item_index in range(18):
                task = task_for(task_id, item_index)
                for template in TEMPLATES:
                    raw = render(task, template)
                    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw, INTERFACE)
                    rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                        "case_id": f"phase343_{model}_{task_id}_{item_index:02d}_{template}",
                        "semantic_case_id": f"phase343_{task_id}_{item_index:02d}_{template}",
                        "model": model, "family_id": task_class, "mechanism_id": task_id,
                        "task_class": task_class, "item_index": item_index,
                        "split": split_for(item_index), "template_id": template,
                        "interface": INTERFACE, "answer_phase": answer_phase,
                        "prompt": prompt, "raw_prompt": raw,
                        "tokenization_add_special_tokens": add_special,
                        "source_fragment": task["source_fragment"], "query_fragment": task["query_fragment"],
                        "target": task["target"], "target_aliases": task["target_aliases"],
                        "distractors": task["distractors"],
                        "official_execution_mode": "b1_left_cache0",
                        "baseline_only": True, "internal_intervention_allowed": False,
                    })
    if len(rows) != 2592 or len({row["case_id"] for row in rows}) != 2592:
        raise RuntimeError(f"Invalid Phase343 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Qualify explicit-copy, copy-neighbor, and noncopy tasks on the Phase342 official path.",
        "registered_case_count": len(rows), "items_per_task": 18,
        "templates": list(TEMPLATES),
        "tasks": [{"task_id": task, "task_class": klass} for task, klass in TASKS],
        "official_execution_mode": "b1_left_cache0",
        "thresholds": {
            "split_baseline_accuracy_min": 0.8,
            "split_phrase_valid_rate_min": 1.0,
            "glm4_explicit_copy_qualified_min": 3,
            "glm4_copy_neighbor_qualified_min": 1,
            "glm4_noncopy_control_qualified_min": 4,
        },
        "claim_boundaries": [
            "Phase343 is baseline qualification only and uses single-case left-padded execution.",
            "Task qualification is not copy-mechanism evidence.",
            "No task may be dropped using causal outcomes because no causal intervention is run.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "task_count": len(TASKS),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout", "private_heldout")
        },
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase343_registered_cases.jsonl", rows)
    write_json(root / "phase343_registered_protocol.json", protocol)
    write_json(root / "phase343_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
