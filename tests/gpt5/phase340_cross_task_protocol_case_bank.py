#!/usr/bin/env python3
"""Register an independent baseline-only repair matrix after Phase339."""

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


PHASE = "Phase340"
SCHEMA_VERSION = "16.0.0"
ROUND_DEFAULT = "fresh_cross_task_protocol_repair"
OUT = ROOT / "tests/gpt5/result/phase340_cross_task_protocol"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("template_b", "template_c")
TASKS = (
    ("material_relation_binding", "relation_binding"),
    ("attribute_relation_binding", "relation_binding"),
    ("part_relation_binding", "relation_binding"),
    ("location_relation_binding", "relation_binding"),
    ("identity_copy", "source_operation"),
    ("source_span_extraction", "source_operation"),
    ("singular_agreement", "cross_family"),
    ("direct_entailment", "cross_family"),
    ("answer_only_protocol", "cross_family"),
)

OBJECTS = (
    "anvil", "basket", "compass", "decanter", "engine", "fan", "goblet", "helmet", "insulator",
    "jug", "kettle", "lantern", "mirror", "needle", "organ", "pump", "quilt", "radio",
)
MATERIALS = (
    "steel", "willow", "bronze", "crystal", "iron", "nylon", "silver", "leather", "ceramic",
    "clay", "copper", "paper", "glass", "titanium", "oak", "rubber", "cotton", "aluminum",
)
ATTRIBUTES = (
    "heavy", "woven", "magnetic", "sealed", "powered", "rotating", "engraved", "padded", "heated",
    "glazed", "boiling", "bright", "reflective", "sharp", "musical", "pressurized", "stitched", "portable",
)
PARTS = (
    "plate", "handle", "needle", "stopper", "piston", "blade", "stem", "visor", "core",
    "spout", "whistle", "wick", "frame", "eye", "keyboard", "valve", "lining", "dial",
)
LOCATIONS = (
    "foundry", "pantry", "cockpit", "cellar", "factory", "office", "museum", "locker", "kitchen",
    "shelf", "stove", "hallway", "bedroom", "toolbox", "theater", "basement", "closet", "garage",
)
CODEWORDS = (
    "apricot", "birch", "canyon", "dawn", "echo", "fern", "granite", "honey", "island",
    "jade", "kiwi", "lake", "maple", "navy", "olive", "plum", "quartz", "river",
)
SPAN_WORDS = (
    "acorn", "brook", "cloud", "dune", "earth", "field", "grass", "hill", "inlet",
    "jetty", "knoll", "leaf", "marsh", "nest", "ocean", "pine", "ridge", "stone",
)
NAMES = (
    "Ava", "Ben", "Cara", "Dale", "Evan", "Faye", "Gina", "Hugo", "Iris",
    "Jake", "Kara", "Liam", "Mona", "Noah", "Omar", "Pia", "Ravi", "Sara",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def task_for(task_id: str, index: int) -> dict[str, Any]:
    obj = OBJECTS[index]
    if task_id == "material_relation_binding":
        context = f"The {obj} is made from {MATERIALS[index]} and is {ATTRIBUTES[index]}."
        question, target, distractors = f"What material is the {obj} made from?", MATERIALS[index], [ATTRIBUTES[index], "unknown"]
    elif task_id == "attribute_relation_binding":
        context = f"The {obj} is made from {MATERIALS[index]} and is {ATTRIBUTES[index]}."
        question, target, distractors = f"What attribute describes the {obj}?", ATTRIBUTES[index], [MATERIALS[index], "unknown"]
    elif task_id == "part_relation_binding":
        context = f"The {obj} contains a {PARTS[index]} and is {ATTRIBUTES[index]}."
        question, target, distractors = f"What part does the {obj} contain?", PARTS[index], [ATTRIBUTES[index], "unknown"]
    elif task_id == "location_relation_binding":
        context = f"The {obj} is stored in the {LOCATIONS[index]} and is {ATTRIBUTES[index]}."
        question, target, distractors = f"Where is the {obj} stored?", LOCATIONS[index], [ATTRIBUTES[index], "unknown"]
    elif task_id == "identity_copy":
        context = f"Record {index + 81} has identifier {CODEWORDS[index]}."
        question, target, distractors = "What is the identifier?", CODEWORDS[index], [SPAN_WORDS[index], "unknown"]
    elif task_id == "source_span_extraction":
        context = f"The marked sequence is north {SPAN_WORDS[index]} south."
        question, target, distractors = "What is the middle word?", SPAN_WORDS[index], ["north", "south"]
    elif task_id == "singular_agreement":
        context = f"Complete a standard sentence about one {obj}."
        question, target, distractors = f"The {obj} ___ ready.", "is", ["are", "unknown"]
    elif task_id == "direct_entailment":
        group = f"team{index + 91}"
        context = f"Every member of {group} is calm. {NAMES[index]} is a member of {group}."
        question, target, distractors = f"Is {NAMES[index]} calm?", "yes", ["no", "unknown"]
    elif task_id == "answer_only_protocol":
        context = f"The required response word is {CODEWORDS[index]}."
        question, target, distractors = "What is the required response word?", CODEWORDS[index], [SPAN_WORDS[index], "unknown"]
    else:
        raise KeyError(task_id)
    return {
        "context": context, "question": question, "target": target,
        "target_aliases": [target], "distractors": distractors,
        "source_fragment": context, "query_fragment": question,
    }


def render(task: dict[str, Any], template: str) -> str:
    if template == "template_b":
        return f"Information: {task['context']}\nTask: {task['question']}\nRule: Answer with one word only.\nResponse:"
    if template == "template_c":
        return f"Use this information: {task['context']}\nAnswer with one word only.\nPrompt: {task['question']}\nResult:"
    raise KeyError(template)


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    rows: list[dict[str, Any]] = []
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
                    prompt, add_special, answer_phase = interface_prompt(
                        tokenizer, model, raw, INTERFACE
                    )
                    rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                        "created_at": now(),
                        "case_id": f"phase340_{model}_{task_id}_{item_index:02d}_{template}",
                        "semantic_case_id": f"phase340_{task_id}_{item_index:02d}_{template}",
                        "model": model, "family_id": task_class,
                        "mechanism_id": task_id, "task_class": task_class,
                        "item_index": item_index, "split": split_for(item_index),
                        "template_id": template, "interface": INTERFACE,
                        "answer_phase": answer_phase, "prompt": prompt, "raw_prompt": raw,
                        "tokenization_add_special_tokens": add_special,
                        "source_fragment": task["source_fragment"],
                        "query_fragment": task["query_fragment"],
                        "target": task["target"], "target_aliases": task["target_aliases"],
                        "distractors": task["distractors"],
                        "baseline_only": True, "internal_intervention_allowed": False,
                        "selection_updates_allowed": False,
                    })
    if len(rows) != 972 or len({row["case_id"] for row in rows}) != 972:
        raise RuntimeError(f"Invalid Phase340 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Independently qualify a repaired two-template cross-task baseline denominator.",
        "registered_case_count": len(rows), "models": list(MODELS),
        "tasks": [{"task_id": task, "task_class": klass} for task, klass in TASKS],
        "templates": list(TEMPLATES), "items_per_task": 18,
        "thresholds": {
            "split_baseline_accuracy_min": 0.8,
            "split_phrase_score_valid_rate_min": 1.0,
            "glm4_relation_neighbor_qualified_min": 1,
            "glm4_source_control_qualified_min": 1,
            "glm4_cross_control_qualified_min": 1,
        },
        "entry_gate": (
            "GLM4 material plus at least one other relation, one source control, "
            "and one cross-family control must qualify on all four splits."
        ),
        "claim_boundaries": [
            "Phase340 is baseline protocol qualification only.",
            "No activation intervention is executed or inferred.",
            "Fresh items prevent Phase339 heldout outcomes from serving as the new causal denominator.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "task_case_count": {task: sum(row["mechanism_id"] == task for row in rows) for task, _ in TASKS},
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout", "private_heldout")
        },
        "valid": True,
    }
    write_jsonl(root / "phase340_registered_cases.jsonl", rows)
    write_json(root / "phase340_registered_protocol.json", protocol)
    write_json(root / "phase340_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
