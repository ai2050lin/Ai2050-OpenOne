#!/usr/bin/env python3
"""Register a 12-task knowledge/reasoning/grammar protocol matrix."""

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


PHASE = "Phase345"
SCHEMA_VERSION = "21.0.0"
ROUND_DEFAULT = "three_core_protocol_qualification"
OUT = ROOT / "tests/gpt5/result/phase345_three_core_protocol"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("format_a", "format_b", "format_c")
TASKS = (
    ("context_relation_binding", "knowledge_network"),
    ("parameter_knowledge_retrieval", "knowledge_network"),
    ("explicit_copy_control", "knowledge_network"),
    ("missing_condition_check", "reasoning"),
    ("two_hop_entailment", "reasoning"),
    ("direct_fact_control", "reasoning"),
    ("sentence_number_agreement", "grammar"),
    ("sentence_past_tense", "grammar"),
    ("no_morphology_control", "grammar"),
    ("answer_only_protocol", "protocol_control"),
    ("multi_token_natural_answer", "protocol_control"),
    ("no_source_answer", "protocol_control"),
)

OBJECTS = (
    "anchor", "basket", "camera", "drum", "engine", "flask", "guitar", "helmet",
    "inkwell", "jacket", "kettle", "lantern", "mirror", "notebook", "oven", "paddle",
    "quiver", "radio", "shovel", "telescope", "umbrella", "violin", "wheel", "yacht",
)
LABELS = (
    "arven", "brylo", "cestin", "dovar", "elmar", "fynel", "grest", "halvo",
    "invar", "joren", "kelto", "lurin", "mavor", "nexil", "orlan", "pyrin",
    "quast", "revik", "sorel", "tavin", "ulmar", "vexor", "wylan", "zorim",
)
MATERIALS = (
    "steel", "willow", "glass", "maple", "iron", "ceramic", "cedar", "leather",
    "clay", "cotton", "copper", "paper", "silver", "linen", "stone", "rubber",
    "bronze", "plastic", "aluminum", "brass", "nylon", "spruce", "oak", "fiberglass",
)
ATTRIBUTES = (
    "heavy", "woven", "digital", "tuned", "powered", "sealed", "musical", "padded",
    "capped", "lined", "heated", "bright", "reflective", "bound", "baking", "floating",
    "filled", "wireless", "sharp", "distant", "folded", "acoustic", "round", "moored",
)
CAPITAL_FACTS = (
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Egypt", "Cairo"),
    ("India", "New Delhi"), ("Brazil", "Brasilia"), ("China", "Beijing"), ("Greece", "Athens"),
    ("Portugal", "Lisbon"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
    ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Mexico", "Mexico City"),
    ("Argentina", "Buenos Aires"), ("Kenya", "Nairobi"), ("Thailand", "Bangkok"),
    ("South Korea", "Seoul"), ("Turkey", "Ankara"),
)
NAMES = (
    "Ari", "Bea", "Cole", "Dina", "Eli", "Fia", "Gus", "Hope",
    "Ivan", "Jade", "Kian", "Lena", "Milo", "Nia", "Oren", "Paz",
    "Rami", "Sia", "Tao", "Uma", "Vik", "Wren", "Xena", "Yuri",
)
VERBS = (
    ("walk", "walked"), ("jump", "jumped"), ("play", "played"), ("open", "opened"),
    ("close", "closed"), ("paint", "painted"), ("call", "called"), ("visit", "visited"),
    ("clean", "cleaned"), ("watch", "watched"), ("help", "helped"), ("start", "started"),
    ("finish", "finished"), ("move", "moved"), ("turn", "turned"), ("listen", "listened"),
    ("carry", "carried"), ("study", "studied"), ("try", "tried"), ("stop", "stopped"),
    ("plan", "planned"), ("smile", "smiled"), ("dance", "danced"), ("arrive", "arrived"),
)
PHRASES = (
    "quiet narrow", "bright copper", "soft green", "clear northern", "small wooden", "deep blue",
    "warm gentle", "open silver", "hidden amber", "rapid eastern", "calm winter", "light golden",
    "silent crystal", "wide central", "smooth white", "long coastal", "fresh alpine", "dark violet",
    "round stone", "distant red", "folded black", "musical cedar", "heavy iron", "moored white",
)
ARITHMETIC = (
    (1, 3, "four"), (2, 4, "six"), (3, 5, "eight"), (1, 4, "five"),
    (2, 5, "seven"), (4, 5, "nine"), (1, 2, "three"), (4, 6, "ten"),
    (5, 6, "eleven"), (5, 7, "twelve"), (6, 8, "fourteen"), (7, 8, "fifteen"),
    (7, 9, "sixteen"), (8, 10, "eighteen"), (9, 11, "twenty"), (1, 1, "two"),
    (6, 7, "thirteen"), (8, 9, "seventeen"), (10, 11, "twenty-one"),
    (10, 12, "twenty-two"), (11, 12, "twenty-three"), (11, 13, "twenty-four"),
    (12, 13, "twenty-five"), (12, 14, "twenty-six"),
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
    if index < 12:
        return "discovery"
    if index < 17:
        return "calibration"
    if index < 21:
        return "heldout"
    return "private_heldout"


def task_for(task_id: str, i: int) -> dict[str, Any]:
    obj, label = OBJECTS[i], LABELS[i]
    instruction = "Return exactly the requested answer and nothing else."
    if task_id == "context_relation_binding":
        context, question, target, distractors = f"The {obj} is made from {MATERIALS[i]} and is {ATTRIBUTES[i]}.", f"What material is the {obj} made from?", MATERIALS[i], [ATTRIBUTES[i], "unknown"]
    elif task_id == "parameter_knowledge_retrieval":
        country, capital = CAPITAL_FACTS[i]
        context, question, target, distractors = "Use ordinary world knowledge; no answer is supplied in the prompt.", f"What is the capital of {country}?", capital, [CAPITAL_FACTS[(i + 1) % 24][1], "unknown"]
    elif task_id == "explicit_copy_control":
        context, question, target, distractors = f"The explicit label for the {obj} is {label}.", f"What is the label for the {obj}?", label, [LABELS[(i + 1) % 24], "unknown"]
    elif task_id == "missing_condition_check":
        context = f"Rule: if the {obj} is {ATTRIBUTES[i]} and locked, then it is approved. Fact: the {obj} is {ATTRIBUTES[i]}. No fact says it is locked."
        question, target, distractors = "Based only on the facts, is approval proven?", "no", ["yes", "unknown"]
    elif task_id == "two_hop_entailment":
        group_a, group_b, group_c = f"group{i + 31}", f"class{i + 31}", f"set{i + 31}"
        context = f"Every {group_a} is a {group_b}. Every {group_b} is a {group_c}. {NAMES[i]} is a {group_a}."
        question, target, distractors = f"Is {NAMES[i]} a {group_c}?", "yes", ["no", "unknown"]
    elif task_id == "direct_fact_control":
        group = f"set{i + 61}"
        context, question, target, distractors = f"The fact states that {NAMES[i]} is a {group}.", f"Is {NAMES[i]} a {group}?", "yes", ["no", "unknown"]
    elif task_id == "sentence_number_agreement":
        plural = i % 2 == 1
        subject = f"the {obj}s" if plural else f"the {obj}"
        target = "are" if plural else "is"
        context, question, distractors = "Use standard English subject-verb agreement in a complete sentence.", f"Complete the sentence: {subject} ___ ready today.", ["is" if plural else "are", "unknown"]
    elif task_id == "sentence_past_tense":
        verb, past = VERBS[i]
        context, question, target, distractors = f"Yesterday {NAMES[i]} performed the action '{verb}'.", f"Complete: Yesterday {NAMES[i]} ___ before noon.", past, [verb, f"{verb}s"]
    elif task_id == "no_morphology_control":
        verb, _past = VERBS[i]
        context, question, target, distractors = f"The exact unchanged verb token is {verb}.", "Return the verb without changing its form.", verb, [f"{verb}s", "unknown"]
    elif task_id == "answer_only_protocol":
        context, question, target, distractors = f"The required one-word response is {label}.", "What is the required response?", label, [LABELS[(i + 1) % 24], "unknown"]
    elif task_id == "multi_token_natural_answer":
        first, second = PHRASES[i].split()
        context, question, target, distractors = f"The scene is described as {first} and {second}.", "Give the two-word description in the same order.", PHRASES[i], [f"{second} {first}", "unknown scene"]
    elif task_id == "no_source_answer":
        left, right, target = ARITHMETIC[i]
        context, question, distractors = f"Compute {left} plus {right}; the answer is not supplied.", "What is the sum, written as an English word?", ["zero", "one"]
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
            for item_index in range(24):
                task = task_for(task_id, item_index)
                for template in TEMPLATES:
                    raw = render(task, template)
                    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw, INTERFACE)
                    rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                        "case_id": f"phase345_{model}_{task_id}_{item_index:02d}_{template}",
                        "semantic_case_id": f"phase345_{task_id}_{item_index:02d}_{template}",
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
        raise RuntimeError(f"Invalid Phase345 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Qualify orthogonal knowledge, reasoning, grammar, and protocol tasks before physical tracing.",
        "registered_case_count": len(rows), "items_per_task": 24,
        "templates": list(TEMPLATES),
        "tasks": [{"task_id": task, "task_class": klass} for task, klass in TASKS],
        "official_execution_mode": "b1_left_cache0",
        "thresholds": {
            "split_baseline_accuracy_min": 0.8,
            "split_phrase_valid_rate_min": 1.0,
            "physical_trace_family_qualified_min": 2,
            "physical_trace_protocol_qualified_min": 2,
        },
        "claim_boundaries": [
            "Phase345 is baseline qualification only; MCUE and causal interventions are not run.",
            "All behavior and phrase scoring use the Phase342 single-case official path.",
            "Task-family coverage is a graph denominator, not a mechanism closure claim.",
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
    write_jsonl(root / "phase345_registered_cases.jsonl", rows)
    write_json(root / "phase345_registered_protocol.json", protocol)
    write_json(root / "phase345_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
