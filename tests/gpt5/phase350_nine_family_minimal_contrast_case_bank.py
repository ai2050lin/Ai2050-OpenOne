#!/usr/bin/env python3
"""Register one manually controlled A/B/C/D contrast contract per language family."""

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
from phase330_nine_family_case_bank import EN_ZH, MODELS, NAMES, SYNONYMS  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase345_three_core_protocol_case_bank import (  # noqa: E402
    ATTRIBUTES, LABELS, MATERIALS, OBJECTS, PHRASES, VERBS,
)


PHASE = "Phase350"
SCHEMA_VERSION = "26.0.0"
ROUND_DEFAULT = "nine_family_minimal_contrast_qualification"
OUT = ROOT / "tests/gpt5/result/phase350_nine_family_minimal_contrast"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("format_a", "format_b")
CONDITIONS = ("A_operation_lex_x", "B_control_lex_x", "C_operation_lex_y", "D_control_lex_y")
FAMILY_SPECS = {
    "content_knowledge": ("negated_attribute", "negate"),
    "output_protocol": ("answer_only", "stop"),
    "reasoning_constraint": ("missing_condition_control", "condition_check"),
    "syntax_structure": ("past_tense", "morph_transform"),
    "language_action": ("transform", "content_transform"),
    "cross_lingual": ("translation", "content_transform"),
    "readout_competition": ("target_vs_wrong", "compare"),
    "state_drift": ("entity_drift", "role_bind"),
    "closure": ("multi_token_completion", "continue"),
}


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
    if index < 6:
        return "physical_discovery"
    if index < 8:
        return "physical_calibration"
    if index < 10:
        return "physical_heldout"
    return "causal_sealed"


def task_for(family: str, lexical_index: int, control: bool) -> dict[str, Any]:
    obj, label = OBJECTS[lexical_index], LABELS[lexical_index]
    if family == "content_knowledge":
        target = ATTRIBUTES[lexical_index]
        wrong = ATTRIBUTES[(lexical_index + 5) % 24]
        context = (
            f"The {obj} is not {wrong}; it is {target}."
            if not control else
            f"For the {obj}, the registered answer is {target}; {wrong} is only a distractor."
        )
        question, distractors = f"Which stated attribute applies to the {obj}?", [wrong, "unknown"]
    elif family == "output_protocol":
        target = label
        context = f"The answer to the item is {target}."
        question, distractors = "What is the answer?", [LABELS[(lexical_index + 1) % 24], "unknown"]
    elif family == "reasoning_constraint":
        name = NAMES[lexical_index]
        first, second = ATTRIBUTES[lexical_index], MATERIALS[lexical_index]
        target = "unknown"
        context = (
            f"If something is {first} and {second}, it is approved. {name} is {first}. No fact states whether {name} is {second}."
            if not control else
            f"If something is {first} and {second}, it is approved. {name} is {first}. The registered verdict is unknown."
        )
        question, distractors = f"Is {name} definitely approved?", ["yes", "no"]
    elif family == "syntax_structure":
        base, past = VERBS[lexical_index]
        target = past
        context = (
            f"Yesterday {NAMES[lexical_index]} performed the action '{base}'."
            if not control else
            f"The required past-tense answer for '{base}' is '{past}'."
        )
        question, distractors = "Return the correct past-tense verb.", [base, base + "s"]
    elif family == "language_action":
        source = SYNONYMS[lexical_index][0]
        target = source.upper()
        context = (
            f"The source word is {source}."
            if not control else
            f"The source word is {source}; the required uppercase answer is {target}."
        )
        question, distractors = "Return the source word in uppercase.", [source, source.capitalize()]
    elif family == "cross_lingual":
        source, target = EN_ZH[lexical_index]
        context = (
            f"The English source word is '{source}'."
            if not control else
            f"The bilingual record states that '{source}' means '{target}'."
        )
        question, distractors = "Return the Chinese translation only.", [EN_ZH[(lexical_index + 1) % 24][1], source]
    elif family == "readout_competition":
        target = label
        wrong = LABELS[(lexical_index + 1) % 24]
        context = (
            f"The correct label is {target}; a competing wrong label is {wrong}."
            if not control else
            f"The required response is {target}."
        )
        question, distractors = "Return the correct label.", [wrong, "unknown"]
    elif family == "state_drift":
        target = NAMES[lexical_index]
        distractor = NAMES[(lexical_index + 7) % 24]
        context = (
            f"The original owner is {target}. Later, an unrelated note mentions {distractor}. Keep the original owner fixed."
            if not control else
            f"The registered original owner answer is {target}; ignore the unrelated name {distractor}."
        )
        question, distractors = "Who is the original owner?", [distractor, "unknown"]
    elif family == "closure":
        target = PHRASES[lexical_index]
        words = target.split()
        context = (
            f"The complete response phrase is '{target}'."
            if not control else
            f"The exact registered answer is '{target}'."
        )
        question, distractors = "Return the complete two-word response.", [words[0], f"{words[1]} {words[0]}"]
    else:
        raise KeyError(family)
    if family == "output_protocol":
        instruction = (
            "Return only the answer and nothing else."
            if not control else "Answer the question correctly; extra explanation is permitted."
        )
    elif family == "closure":
        instruction = (
            "Return the complete phrase and stop immediately."
            if not control else "Return the registered answer."
        )
    else:
        instruction = "Return exactly the requested answer and nothing else."
    return {
        "context": context, "question": question, "instruction": instruction,
        "target": target, "target_aliases": [target], "distractors": distractors,
        "source_fragment": context, "query_fragment": question,
    }


def render(task: dict[str, Any], template: str) -> str:
    if template == "format_a":
        return f"Information: {task['context']}\nTask: {task['question']}\nRule: {task['instruction']}\nResponse:"
    if template == "format_b":
        return f"Reference: {task['context']}\nOutput rule: {task['instruction']}\nQuery: {task['question']}\nFinal answer:"
    raise KeyError(template)


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    rows = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for family, (mechanism, operation) in FAMILY_SPECS.items():
            for item_index in range(12):
                split = split_for(item_index)
                for condition in CONDITIONS:
                    lexical_set = "x" if condition[0] in {"A", "B"} else "y"
                    control = condition[0] in {"B", "D"}
                    lexical_index = item_index if lexical_set == "x" else item_index + 12
                    task = task_for(family, lexical_index, control)
                    for template in TEMPLATES:
                        raw = render(task, template)
                        prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw, INTERFACE)
                        rows.append({
                            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                            "case_id": f"phase350_{model}_{family}_{mechanism}_{item_index:02d}_{condition}_{template}",
                            "semantic_case_id": f"phase350_{family}_{mechanism}_{item_index:02d}_{condition}_{template}",
                            "contrast_group_id": f"phase350_{family}_{mechanism}_{item_index:02d}_{template}",
                            "model": model, "family_id": family, "mechanism_id": mechanism,
                            "candidate_primary_operation": operation, "item_index": item_index,
                            "lexical_index": lexical_index, "lexical_set": lexical_set,
                            "contrast_condition": condition,
                            "operation_demanded": not control,
                            "control_type": "explicit_shortcut_or_relaxed_protocol" if control else "target_operation_demand",
                            "split": split, "template_id": template, "interface": INTERFACE,
                            "answer_phase": answer_phase, "prompt": prompt, "raw_prompt": raw,
                            "tokenization_add_special_tokens": add_special,
                            "source_fragment": task["source_fragment"], "query_fragment": task["query_fragment"],
                            "target": task["target"], "target_aliases": task["target_aliases"],
                            "distractors": task["distractors"],
                            "official_execution_mode": "b1_left_cache0",
                            "baseline_only": True, "internal_intervention_allowed": False,
                            "single_unit_intervention_allowed": False,
                        })
    if len(rows) != 2592 or len({row["case_id"] for row in rows}) != 2592:
        raise RuntimeError(f"Invalid Phase350 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Qualify one manually controlled paired contrast per family before signed physical tracing.",
        "registered_case_count": len(rows), "families": list(FAMILY_SPECS),
        "conditions": list(CONDITIONS), "items_per_family": 12,
        "templates": list(TEMPLATES), "models": list(MODELS),
        "split_contract": {"physical_discovery": 6, "physical_calibration": 2, "physical_heldout": 2, "causal_sealed": 2},
        "thresholds": {"split_semantic_accuracy_min": 0.8, "split_phrase_valid_rate_min": 1.0},
        "claim_boundaries": [
            "The control may use an explicit shortcut or relaxed protocol; it is not a pure operation-off state.",
            "Qualification tests behavior contracts only and cannot identify physical mechanisms.",
            "Causal-sealed cases may be behavior-qualified but their internal traces remain sealed.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "family_count": len(FAMILY_SPECS),
        "contrast_group_count": len({row["contrast_group_id"] for row in rows}),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "split_case_count": {split: sum(row["split"] == split for row in rows) for split in protocol["split_contract"]},
        "within_pair_target_mismatch_count": sum(
            len({row["target"] for row in rows if row["contrast_group_id"] == group and row["model"] == model and row["lexical_set"] == lexical}) != 1
            for group in {row["contrast_group_id"] for row in rows}
            for model in MODELS for lexical in ("x", "y")
        ),
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase350_registered_cases.jsonl", rows)
    write_json(root / "phase350_registered_protocol.json", protocol)
    write_json(root / "phase350_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
