#!/usr/bin/env python3
"""Rebuild seven rejected language-family contracts without model-effect feedback."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase330_nine_family_case_bank import EN_ZH, NAMES  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase345_three_core_protocol_case_bank import ATTRIBUTES, LABELS, MATERIALS, OBJECTS, VERBS  # noqa: E402
from phase353_family_contract_case_bank import record, render, split_for  # noqa: E402


PHASE = "Phase361"
SCHEMA_VERSION = "37.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
OUT = ROOT / "tests/gpt5/result/phase361_contract_repair"
ROUND_DEFAULT = "seven_contract_repair"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("format_a", "format_b")
CONDITIONS = ("A_target_lex_x", "B_control_lex_x", "C_target_lex_y", "D_control_lex_y")
CONTRACTS = {
    "reasoning_constraint": {
        "missing_condition": ("rule_evaluation_vs_direct_verdict", "verdict"),
        "two_hop_entailment": ("two_hop_vs_direct_relation", "verdict"),
    },
    "syntax_structure": {
        "past_tense_sentence": ("sentence_context_vs_vocabulary_context", "morpheme"),
        "number_agreement": ("sentence_agreement_vs_explicit_number", "morpheme"),
    },
    "language_action": {
        "case_transform": ("uppercase_transform_vs_uppercase_copy", "transformed_content"),
        "field_extract": ("field_extract_vs_delimited_copy", "extracted_content"),
    },
    "cross_lingual": {
        "translation": ("cross_language_choice_vs_marked_choice", "translated_content"),
    },
}
LEXICAL_OVERLAP_MIN = 0.65
MAX_PROMPT_TOKEN_DELTA = 24


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


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))


def overlap(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def repaired_pair(family: str, mechanism: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    name = NAMES[index]
    exact = "Return exactly the requested answer and nothing else."
    if mechanism == "missing_condition":
        first, second = ATTRIBUTES[index], MATERIALS[index]
        common = f"Approval rule: a record is approved only when it has both {first} and {second}. The record for {name} has {first}."
        target = record(
            f"{common} Its {second} status is not stated, and no direct verdict is supplied.",
            f"Based only on this record, is {name} definitely approved?",
            "unknown", ["yes", "no"], exact,
        )
        control = record(
            f"{common} Its final verdict is explicitly marked unresolved, and no additional condition is supplied.",
            f"Based only on this record, is {name} definitely approved?",
            "unknown", ["yes", "no"], exact, aliases=["unresolved"],
        )
    elif mechanism == "two_hop_entailment":
        a, b, c = f"group-{LABELS[index]}", f"class-{LABELS[index]}", f"set-{LABELS[index]}"
        common = f"Relation notes use {a}, {b}, and {c}. Every {a} is a {b}. Every {b} is a {c}. {name} is a {a}."
        target = record(
            f"{common} No direct relation verdict is supplied.",
            f"Based on the relation notes, is {name} a {c}?",
            "yes", ["no", "unknown"], exact,
        )
        control = record(
            f"{common} A direct relation note also marks {name} as a {c}.",
            f"Based on the relation notes, is {name} a {c}?",
            "yes", ["no", "unknown"], exact,
        )
    elif mechanism == "past_tense_sentence":
        base, past = VERBS[index]
        common = f"Grammar card: the base verb is '{base}' and the subject is {name}. The required output is the regular past-tense verb form."
        target = record(
            f"{common} Sentence context: 'Yesterday {name} ___ before noon.' No answer form is printed.",
            "Return the required verb form for the grammar card.",
            past, [base, base + "s"], exact,
        )
        control = record(
            f"{common} Vocabulary context: convert the listed base verb to regular past tense. No answer form is printed.",
            "Return the required verb form for the grammar card.",
            past, [base, base + "s"], exact,
        )
    elif mechanism == "number_agreement":
        plural = index % 2 == 0
        subject, target_word, wrong, number = (
            (f"the {OBJECTS[index]}s", "are", "is", "plural")
            if plural else (f"the {OBJECTS[index]}", "is", "are", "singular")
        )
        common = f"Grammar card: the subject is '{subject}'. Choose the matching present verb from 'is' or 'are'."
        target = record(
            f"{common} Sentence context: '{subject.capitalize()} ___ ready.' No answer is marked.",
            "Return the verb that agrees with the subject.",
            target_word, [wrong, "unknown"], exact,
        )
        control = record(
            f"{common} Number context: the subject is explicitly labeled {number}. No answer is marked.",
            "Return the verb that agrees with the subject.",
            target_word, [wrong, "unknown"], exact,
        )
    elif mechanism == "case_transform":
        source = f"code{LABELS[index]}"
        transformed = source.upper()
        target = record(
            f"Transformation card: source text is '{source}'. Preserve every letter and change lowercase letters to uppercase. No separate output is printed.",
            "Return the required transformed text.",
            transformed, [source, source.capitalize()], exact,
        )
        control = record(
            f"Transformation card: source text is '{transformed}'. Preserve every letter and copy the existing uppercase text. No separate output is printed.",
            "Return the required transformed text.",
            transformed, [source, source.capitalize()], exact,
        )
    elif mechanism == "field_extract":
        code = f"K{300 + index}"
        common = f"Record card: owner={name}; code={code}; status=active. Return one value from this record and ignore the other fields."
        target = record(
            f"{common} The requested field name is code.",
            "Return the requested code value from the record.",
            code, [name, "active"], exact,
        )
        control = record(
            f"{common} The requested delimiter is 'code=' and its following value must be copied.",
            "Return the requested code value from the record.",
            code, [name, "active"], exact,
        )
    elif mechanism == "translation":
        source, translated = EN_ZH[index]
        wrong = EN_ZH[(index + 1) % len(EN_ZH)][1]
        letter = "A" if index % 2 == 0 else "B"
        options = f"A={translated}; B={wrong}" if letter == "A" else f"A={wrong}; B={translated}"
        common = f"Bilingual card: the English source is '{source}'. The Chinese choices are {options}. Return only one Chinese choice."
        target = record(
            f"{common} Use the cross-language meaning of the English source to choose.",
            "Return the requested Chinese choice from the bilingual card.",
            translated, [wrong, source], exact, language="en_zh",
        )
        control = record(
            f"{common} The card directly marks choice {letter} as the approved choice to copy.",
            "Return the requested Chinese choice from the bilingual card.",
            translated, [wrong, source], exact, language="en_zh",
        )
    else:
        raise KeyError((family, mechanism))
    return target, control


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tokenizers = {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizer = AutoTokenizer.from_pretrained(
                str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
                local_files_only=True, use_fast=False,
            )
            tokenizers[model] = tokenizer
            for family, mechanisms in CONTRACTS.items():
                for mechanism, (variable, outcome) in mechanisms.items():
                    for item_index in range(12):
                        split = split_for(item_index)
                        for lexical_set, lexical_index in (("x", item_index), ("y", item_index + 12)):
                            demanded_task, control_task = repaired_pair(family, mechanism, lexical_index)
                            pairs = (
                                (True, demanded_task, "A" if lexical_set == "x" else "C"),
                                (False, control_task, "B" if lexical_set == "x" else "D"),
                            )
                            for demanded, task, letter in pairs:
                                condition = next(value for value in CONDITIONS if value.startswith(letter))
                                for template in TEMPLATES:
                                    raw = render(task, template)
                                    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw, INTERFACE)
                                    target_ids = tokenizer(task["target"], add_special_tokens=False)["input_ids"]
                                    prompt_ids = tokenizer(prompt, add_special_tokens=add_special)["input_ids"]
                                    rows.append({
                                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                                        "case_id": f"phase361_{model}_{family}_{mechanism}_{item_index:02d}_{condition}_{template}",
                                        "contract_group_id": f"phase361_{family}_{mechanism}_{item_index:02d}_{template}",
                                        "model": model, "family_id": family, "mechanism_id": mechanism,
                                        "manipulated_variable": variable, "outcome_type": outcome,
                                        "item_index": item_index, "lexical_index": lexical_index,
                                        "lexical_set": lexical_set, "contrast_condition": condition,
                                        "operation_demanded": demanded, "split": split, "template_id": template,
                                        "interface": INTERFACE, "answer_phase": answer_phase,
                                        "prompt": prompt, "raw_prompt": raw,
                                        "tokenization_add_special_tokens": add_special,
                                        "prompt_token_count": len(prompt_ids),
                                        "context": task["context"], "question": task["question"],
                                        "source_fragment": task["context"],
                                        "query_fragment": task["question"],
                                        "instruction": task["instruction"], "target": task["target"],
                                        "target_aliases": task["target_aliases"], "distractors": task["distractors"],
                                        "language": task["language"],
                                        "target_visible": task["target"].lower() in raw.lower(),
                                        "target_token_count": len(target_ids),
                                        "target_output_position_role": "first_answer_token",
                                        "counterfactual_natural_by_design": True,
                                        "official_execution_mode": "b1_left_cache0",
                                        "model_effect_used_for_contract": False,
                                        "internal_intervention_allowed": False,
                                    })
        expected = len(MODELS) * 7 * 12 * 2 * 2 * len(TEMPLATES)
        if len(rows) != expected or len({row["case_id"] for row in rows}) != expected:
            raise RuntimeError(f"Invalid repaired denominator: {len(rows)} != {expected}")

        metrics: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for model in MODELS:
            for family, mechanisms in CONTRACTS.items():
                for mechanism in mechanisms:
                    selected = [
                        row for row in rows
                        if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism
                    ]
                    by_key = {(row["item_index"], row["template_id"], row["contrast_condition"][0]): row for row in selected}
                    for item in range(12):
                        for template in TEMPLATES:
                            for left, right in (("A", "B"), ("C", "D")):
                                a, b = by_key[(item, template, left)], by_key[(item, template, right)]
                                metric = metrics[(family, mechanism)]
                                metric["target_match"].append(a["target"] == b["target"])
                                metric["visibility_match"].append(a["target_visible"] == b["target_visible"])
                                metric["target_length_match"].append(a["target_token_count"] == b["target_token_count"])
                                metric["position_match"].append(a["target_output_position_role"] == b["target_output_position_role"])
                                metric["language_match"].append(a["language"] == b["language"])
                                metric["prompt_length_match"].append(abs(a["prompt_token_count"] - b["prompt_token_count"]) <= MAX_PROMPT_TOKEN_DELTA)
                                metric["lexical_overlap"].append(overlap(a["context"] + " " + a["question"], b["context"] + " " + b["question"]))
        contracts = []
        for family, mechanisms in CONTRACTS.items():
            for mechanism, (variable, outcome) in mechanisms.items():
                metric = metrics[(family, mechanism)]
                values = {key: mean(value) for key, value in metric.items()}
                strict = bool(
                    values["target_match"] == 1
                    and values["visibility_match"] == 1
                    and values["target_length_match"] == 1
                    and values["position_match"] == 1
                    and values["language_match"] == 1
                    and values["prompt_length_match"] == 1
                    and values["lexical_overlap"] >= LEXICAL_OVERLAP_MIN
                )
                contracts.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "family_id": family, "mechanism_id": mechanism,
                    "manipulated_variable": variable, "outcome_type": outcome,
                    **{f"{key}_rate" if key != "lexical_overlap" else "mean_lexical_overlap": round(value, 7) for key, value in values.items()},
                    "strict_contract_gate_pass": strict,
                    "admission_state": "pending_behavior" if strict else "rejected_contract",
                    "model_effect_used_for_contract": False,
                })
        root = OUT / round_name
        write_jsonl(root / "phase361_registered_cases.jsonl", rows)
        write_jsonl(root / "phase361_repaired_contract_registry.jsonl", contracts)
        protocol = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "registered_case_count": len(rows), "repaired_contract_count": len(contracts),
            "split_contract": {"physical_discovery": 6, "physical_calibration": 2, "physical_heldout": 2, "causal_sealed": 2},
            "thresholds": {"mean_lexical_overlap_min": LEXICAL_OVERLAP_MIN, "max_prompt_token_delta": MAX_PROMPT_TOKEN_DELTA},
            "model_effect_used_for_contract": False,
            "physical_heldout_trace_revealed": False,
            "causal_sealed_trace_revealed": False,
        }
        summary = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "denominator": {"registered_case_count": len(rows), "repaired_contract_count": 7, "model_count": 3},
            "results": {
                "strict_contract_pass_count": sum(row["strict_contract_gate_pass"] for row in contracts),
                "model_execution_started": False,
            },
            "entry_decision": "run_repaired_behavior_qualification" if all(row["strict_contract_gate_pass"] for row in contracts) else "stop_and_repair_contracts",
        }
        write_json(root / "phase361_registered_protocol.json", protocol)
        write_json(root / "phase361_contract_summary.json", summary)
        return summary
    finally:
        tokenizers.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))
