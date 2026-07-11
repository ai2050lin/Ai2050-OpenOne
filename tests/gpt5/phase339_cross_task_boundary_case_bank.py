#!/usr/bin/env python3
"""Freeze Phase339 cross-task scope cases for Phase338 coarse blocks."""

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
from phase330_nine_family_case_bank import MODELS, TEMPLATES  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402


PHASE = "Phase339"
SCHEMA_VERSION = "15.0.0"
ROUND_DEFAULT = "early_source_cross_task_boundary"
OUT = ROOT / "tests/gpt5/result/phase339_cross_task_boundary"
INTERFACE = "answer_aligned_chat"

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
    "caliper", "drum", "easel", "flask", "grille", "harp", "inkwell", "jacket", "keypad",
    "loom", "mallet", "notebook", "obelisk", "paddle", "quiver", "ruler", "shovel", "telescope",
)
MATERIALS = (
    "cobalt", "cedar", "slate", "hemp", "pewter", "ivory", "magnesium", "rattan", "felt",
    "concrete", "brass", "beech", "latex", "satin", "alabaster", "chromium", "denim", "graphite",
)
ATTRIBUTES = (
    "ribbed", "lacquered", "hinged", "stoppered", "perforated", "tuned", "capped", "padded", "backlit",
    "tensioned", "balanced", "bound", "tapered", "varnished", "inscribed", "plated", "stitched", "aligned",
)
PARTS = (
    "dial", "membrane", "brace", "nozzle", "lattice", "bridge", "nib", "lining", "sensor",
    "shuttle", "handle", "spine", "base", "blade", "shaft", "scale", "grip", "lens",
)
LOCATIONS = (
    "alcove", "studio", "gallery", "cabinet", "workshop", "auditorium", "archive", "closet", "console",
    "mill", "shed", "library", "plaza", "boathouse", "armory", "laboratory", "depot", "observatory",
)
CODEWORDS = (
    "amber", "bison", "coral", "delta", "ember", "frost", "grove", "hazel", "indigo",
    "juniper", "kelp", "lotus", "mango", "nectar", "onyx", "pearl", "reed", "spruce",
)
SPAN_WORDS = (
    "anchor", "beacon", "cinder", "drizzle", "elm", "flint", "glacier", "harvest", "iris",
    "jasmine", "kernel", "lichen", "meadow", "north", "orbit", "prairie", "raven", "summit",
)
NAMES = (
    "Arin", "Borin", "Celia", "Darin", "Elin", "Faron", "Galen", "Hira", "Ilan",
    "Jora", "Kelin", "Liora", "Maren", "Nerin", "Orin", "Pela", "Quin", "Rina",
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
    obj, material, attribute = OBJECTS[index], MATERIALS[index], ATTRIBUTES[index]
    if task_id == "material_relation_binding":
        context = f"The record states that the {obj} is made from {material} and is {attribute}."
        question, target, distractors = f"What material is the {obj} made from?", material, [attribute, "unknown"]
        source = f"made from {material}"
    elif task_id == "attribute_relation_binding":
        context = f"The record states that the {obj} is made from {material} and is {attribute}."
        question, target, distractors = f"Which stated attribute describes the {obj}?", attribute, [material, "unknown"]
        source = f"is {attribute}"
    elif task_id == "part_relation_binding":
        context = f"The technical record says the {obj} contains a {PARTS[index]} and is {attribute}."
        question, target, distractors = f"Which part does the {obj} contain?", PARTS[index], [attribute, "unknown"]
        source = f"contains a {PARTS[index]}"
    elif task_id == "location_relation_binding":
        context = f"The inventory says the {obj} is stored in the {LOCATIONS[index]} and is {attribute}."
        question, target, distractors = f"Where is the {obj} stored?", LOCATIONS[index], [attribute, "unknown"]
        source = f"stored in the {LOCATIONS[index]}"
    elif task_id == "identity_copy":
        context = f"The identifier assigned to record {index + 41} is {CODEWORDS[index]}."
        question, target, distractors = "What is the assigned identifier?", CODEWORDS[index], [SPAN_WORDS[index], "unknown"]
        source = f"is {CODEWORDS[index]}"
    elif task_id == "source_span_extraction":
        context = f"The quoted three-word span is: alpha {SPAN_WORDS[index]} omega."
        question, target, distractors = "Return the middle word of the quoted span.", SPAN_WORDS[index], ["alpha", "omega"]
        source = f"alpha {SPAN_WORDS[index]} omega"
    elif task_id == "singular_agreement":
        context = f"Use standard singular subject-verb agreement for a sentence about the {obj}."
        question, target, distractors = f"Complete: The {obj} ___ ready.", "is", ["are", "unknown"]
        source = f"the {obj}"
    elif task_id == "direct_entailment":
        group = f"group{index + 61}"
        context = f"Every {group} member is calm. {NAMES[index]} is a {group} member."
        question, target, distractors = f"Is {NAMES[index]} calm?", "yes", ["no", "unknown"]
        source = f"{NAMES[index]} is a {group} member"
    elif task_id == "answer_only_protocol":
        context = f"For this protocol check, the required answer token is {CODEWORDS[index]}."
        question, target, distractors = "Return only the required answer token.", CODEWORDS[index], [SPAN_WORDS[index], "unknown"]
        source = f"answer token is {CODEWORDS[index]}"
    else:
        raise KeyError(task_id)
    return {
        "context": context, "question": question,
        "instruction": "Answer with the one registered word only.",
        "target": target, "target_aliases": [target], "distractors": distractors,
        "source_fragment": source, "query_fragment": question,
    }


def render(task: dict[str, Any], template: str) -> str:
    if template == "template_a":
        return f"{task['context']}\n{task['question']}\n{task['instruction']}\nAnswer:"
    if template == "template_b":
        return f"Information: {task['context']}\nTask: {task['question']}\nRule: {task['instruction']}\nResponse:"
    if template == "template_c":
        return f"Use this information: {task['context']}\n{task['instruction']}\nPrompt: {task['question']}\nResult:"
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
                        "case_id": f"phase339_{model}_{task_id}_{item_index:02d}_{template}",
                        "semantic_case_id": f"phase339_{task_id}_{item_index:02d}_{template}",
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
                        "selection_updates_allowed": False,
                        "block_reselection_allowed": False,
                        "layer_shrink_allowed": False,
                        "single_unit_intervention_allowed": False,
                    })
    if len(rows) != 1458:
        raise RuntimeError(f"Expected 1458 cases, got {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase339 case id")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Classify Phase338 frozen blocks across relation, source, and cross-family tasks.",
        "models": list(MODELS), "interface": INTERFACE,
        "tasks": [{"task_id": task, "task_class": klass} for task, klass in TASKS],
        "items_per_task": 18, "templates": list(TEMPLATES),
        "registered_case_count": len(rows),
        "frozen_block_source": (
            "tests/gpt5/result/phase338_block_causal_screen/material_relation_block_screen/"
            "models/{model}/phase338_frozen_heldout_block.jsonl"
        ),
        "phrase_conditions": [
            "baseline", "correct_zero", "correct_half", "correct_permutation",
            "wrong_depth_zero", "wrong_position_zero",
        ],
        "rollout_conditions": [
            "baseline", "correct_zero", "wrong_depth_zero", "wrong_position_zero",
        ],
        "thresholds": {
            "task_baseline_capability_rate_min": 0.8,
            "task_phrase_score_valid_rate_min": 1.0,
            "task_correct_behavior_loss_rate_min": 0.5,
            "task_wrong_control_behavior_loss_rate_max": 0.1,
            "task_phrase_control_superiority_min": 0.05,
            "relation_reuse_task_pass_min": 3,
            "unrelated_mean_behavior_loss_rate_max": 0.1,
        },
        "claim_boundaries": [
            "No block is reselected on Phase339 tasks.",
            "Relation-neighbor tasks are real mechanisms, not null controls.",
            "Layer-group and neuron shrinking remain closed in Phase339.",
            "Task selectivity is a scope classification, not mechanism closure.",
        ],
    }
    write_jsonl(root / "phase339_registered_cases.jsonl", rows)
    write_json(root / "phase339_registered_protocol.json", protocol)
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "task_count": len(TASKS),
        "model_case_count": {
            model: sum(row["model"] == model for row in rows) for model in MODELS
        },
        "task_case_count": {
            task: sum(row["mechanism_id"] == task for row in rows) for task, _ in TASKS
        },
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout", "private_heldout")
        },
        "valid": True,
    }
    write_json(root / "phase339_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
