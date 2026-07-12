#!/usr/bin/env python3
"""Compile 18 family-specific counterfactual contracts into a frozen case bank."""

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
from phase330_nine_family_case_bank import EN_ZH, MODELS, NAMES, SYNONYMS  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase345_three_core_protocol_case_bank import (  # noqa: E402
    ATTRIBUTES, LABELS, MATERIALS, OBJECTS, PHRASES, VERBS,
)


PHASE = "Phase353"
SCHEMA_VERSION = "29.0.0"
ROUND_DEFAULT = "family_specific_contract_compiler"
OUT = ROOT / "tests/gpt5/result/phase353_family_contracts"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("format_a", "format_b")
CONDITIONS = ("A_target_lex_x", "B_control_lex_x", "C_target_lex_y", "D_control_lex_y")
CONTRACTS = {
    "content_knowledge": {
        "negated_attribute": ("attribute_polarity", False, "semantic_answer"),
        "relation_binding": ("relation_path_length", False, "semantic_answer"),
    },
    "reasoning_constraint": {
        "missing_condition": ("condition_completeness", False, "verdict"),
        "two_hop_entailment": ("inference_path_length", False, "verdict"),
    },
    "syntax_structure": {
        "past_tense_sentence": ("sentence_morphology_context", False, "morpheme"),
        "number_agreement": ("agreement_context", False, "morpheme"),
    },
    "output_protocol": {
        "answer_only": ("output_explanation_permission", True, "protocol_and_semantic"),
        "json_format": ("output_serialization", True, "protocol_and_semantic"),
    },
    "language_action": {
        "case_transform": ("transformation_demand", False, "transformed_content"),
        "field_extract": ("structured_extraction_demand", False, "extracted_content"),
    },
    "cross_lingual": {
        "translation": ("cross_language_mapping_demand", False, "translated_content"),
        "language_routing": ("requested_output_language", True, "routed_content"),
    },
    "readout_competition": {
        "target_vs_wrong": ("wrong_candidate_presence", False, "semantic_answer"),
        "target_vs_continue": ("continuation_permission", True, "protocol_and_semantic"),
    },
    "state_drift": {
        "entity_recency": ("distractor_recency", False, "retained_entity"),
        "role_surface": ("active_passive_surface", False, "retained_role"),
    },
    "closure": {
        "multi_token_stop": ("stop_after_complete_phrase", True, "sequence_and_stop"),
        "continue_suppression": ("continuation_permission", True, "sequence_and_stop"),
    },
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
    return "physical_discovery" if index < 6 else "physical_calibration" if index < 8 else "physical_heldout" if index < 10 else "causal_sealed"


def record(context: str, question: str, target: str, distractors: list[str], instruction: str, *, aliases: list[str] | None = None, language: str = "en") -> dict[str, Any]:
    return {
        "context": context, "question": question, "target": target,
        "target_aliases": list(dict.fromkeys([target, *(aliases or [])])),
        "distractors": distractors, "instruction": instruction, "language": language,
    }


def pair_for(family: str, mechanism: str, i: int) -> tuple[dict[str, Any], dict[str, Any]]:
    obj, label, name = OBJECTS[i], LABELS[i], NAMES[i]
    exact = "Return exactly the requested answer and nothing else."
    if mechanism == "negated_attribute":
        target, wrong = ATTRIBUTES[i], ATTRIBUTES[(i + 5) % 24]
        target_case = record(f"The {obj} is not {wrong}; it is {target}.", f"Which attribute applies to the {obj}?", target, [wrong, "unknown"], exact)
        control = record(f"The {obj} is {target}; the alternative {wrong} does not apply.", f"Which attribute applies to the {obj}?", target, [wrong, "unknown"], exact)
    elif mechanism == "relation_binding":
        target, batch = MATERIALS[i], f"batch-{label}"
        target_case = record(f"The {obj} belongs to {batch}. Every item in {batch} is made from {target}.", f"What material is the {obj} made from?", target, [MATERIALS[(i + 1) % 24], "unknown"], exact)
        control = record(f"The {obj} is made from {target}. It is listed in {batch}.", f"What material is the {obj} made from?", target, [MATERIALS[(i + 1) % 24], "unknown"], exact)
    elif mechanism == "missing_condition":
        first, second = ATTRIBUTES[i], MATERIALS[i]
        target_case = record(f"If something is {first} and {second}, it is approved. {name} is {first}. No fact gives the second condition.", f"Is {name} definitely approved?", "unknown", ["yes", "no"], exact)
        control = record(f"The approval status of {name} is unresolved. The notes mention {first} and omit {second}.", f"Is {name} definitely approved?", "unknown", ["yes", "no"], exact)
    elif mechanism == "two_hop_entailment":
        a, b, c = f"group-{label}", f"class-{label}", f"set-{label}"
        target_case = record(f"Every {a} is a {b}. Every {b} is a {c}. {name} is a {a}.", f"Is {name} a {c}?", "yes", ["no", "unknown"], exact)
        control = record(f"The fact directly states that {name} is a {c}. The labels {a} and {b} are also mentioned.", f"Is {name} a {c}?", "yes", ["no", "unknown"], exact)
    elif mechanism == "past_tense_sentence":
        base, target = VERBS[i]
        target_case = record(f"Yesterday {name} performed the action '{base}'.", f"Complete: Yesterday {name} ___ before noon.", target, [base, base + "s"], exact)
        control = record(f"The base action is '{base}'. Choose its regular past form for a vocabulary record.", "Which form is required?", target, [base, base + "s"], exact)
    elif mechanism == "number_agreement":
        plural = i % 2 == 0
        noun, target, wrong = (f"{obj}s", "are", "is") if plural else (obj, "is", "are")
        target_case = record(f"The subject is 'the {noun}'.", f"Complete: The {noun} ___ ready.", target, [wrong, "unknown"], exact)
        control = record(f"The grammatical number belongs to 'the {noun}'.", "Which English verb agrees: is or are?", target, [wrong, "unknown"], exact)
    elif mechanism == "answer_only":
        base = f"The required semantic answer is {label}."
        target_case = record(base, "What is the semantic answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Return only the answer and nothing else.")
        control = record(base, "What is the semantic answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Answer correctly; one short explanation is permitted.")
    elif mechanism == "json_format":
        base = f"The required semantic answer is {label}."
        json_alias = json.dumps({"answer": label}, ensure_ascii=False, separators=(",", ":"))
        target_case = record(base, "What is the semantic answer?", label, [LABELS[(i + 1) % 24], "unknown"], 'Return JSON only with key "answer".', aliases=[json_alias])
        control = record(base, "What is the semantic answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Return the plain answer only.")
    elif mechanism == "case_transform":
        source = SYNONYMS[i][0]
        target = source.upper()
        target_case = record(f"The source word is {source}.", "Convert the source to uppercase.", target, [source, source.capitalize()], exact)
        control = record(f"The source word is already uppercase: {target}.", "Return the source without changing it.", target, [source, source.capitalize()], exact)
    elif mechanism == "field_extract":
        target = f"K{300 + i}"
        target_case = record(f"Record: owner={name}; code={target}; status=active.", "Extract the code field.", target, [name, "active"], exact)
        control = record(f"The code is {target}. The owner is {name} and status is active.", "Return the code.", target, [name, "active"], exact)
    elif mechanism == "translation":
        source, target = EN_ZH[i]
        target_case = record(f"The English source word is '{source}'.", "Translate it to Chinese.", target, [EN_ZH[(i + 1) % 24][1], source], exact, language="en_zh")
        control = record(f"The bilingual entry is '{source}' / '{target}'.", "Return the Chinese side of the entry.", target, [EN_ZH[(i + 1) % 24][1], source], exact, language="en_zh")
    elif mechanism == "language_routing":
        en, zh = EN_ZH[i]
        target_case = record(f"The concept is {en} / {zh}.", "Answer using the English form only.", en, [zh, EN_ZH[(i + 1) % 24][0]], exact, language="mixed")
        control = record(f"The concept is {en} / {zh}.", "Copy the English form.", en, [zh, EN_ZH[(i + 1) % 24][0]], exact, language="mixed")
    elif mechanism == "target_vs_wrong":
        wrong = LABELS[(i + 1) % 24]
        target_case = record(f"The correct label is {label}; a competing wrong label is {wrong}.", "Return the correct label.", label, [wrong, "unknown"], exact)
        control = record(f"The correct label is {label}; no competing label is supplied.", "Return the correct label.", label, [wrong, "unknown"], exact)
    elif mechanism == "target_vs_continue":
        base = f"The answer is {label}."
        target_case = record(base, "What is the answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Return the answer and stop immediately.")
        control = record(base, "What is the answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Return the answer, then add one short explanation.")
    elif mechanism == "entity_recency":
        other = NAMES[(i + 7) % 24]
        target_case = record(f"The designated owner is {name}. Later an unrelated note mentions {other}.", "Who is the designated owner?", name, [other, "unknown"], exact)
        control = record(f"An unrelated note mentions {other}. The designated owner is {name}.", "Who is the designated owner?", name, [other, "unknown"], exact)
    elif mechanism == "role_surface":
        other, thing = NAMES[(i + 5) % 24], OBJECTS[(i + 3) % 24]
        target_case = record(f"{name} handed the {thing} to {other}.", "Who handed over the object?", name, [other, thing], exact)
        control = record(f"The {thing} was handed to {other} by {name}.", "Who handed over the object?", name, [other, thing], exact)
    elif mechanism == "multi_token_stop":
        target = PHRASES[i]
        words = target.split()
        base = f"The complete answer phrase is '{target}'."
        target_case = record(base, "What is the complete phrase?", target, [words[0], f"{words[1]} {words[0]}"], "Return the complete phrase and stop immediately.")
        control = record(base, "What is the complete phrase?", target, [words[0], f"{words[1]} {words[0]}"], "Return the complete phrase, then add one short note.")
    elif mechanism == "continue_suppression":
        base = f"The answer is {label}."
        target_case = record(base, "What is the answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Return the answer and do not continue.")
        control = record(base, "What is the answer?", label, [LABELS[(i + 1) % 24], "unknown"], "Return the answer and continue with one sentence.")
    else:
        raise KeyError((family, mechanism))
    return target_case, control


def render(task: dict[str, Any], template: str) -> str:
    if template == "format_a":
        return f"Information: {task['context']}\nTask: {task['question']}\nRule: {task['instruction']}\nResponse:"
    if template == "format_b":
        return f"Reference: {task['context']}\nOutput rule: {task['instruction']}\nQuery: {task['question']}\nFinal answer:"
    raise KeyError(template)


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))


def overlap(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    rows, tokenizers = [], {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizer = AutoTokenizer.from_pretrained(str(spec.local_dir), trust_remote_code=spec.trust_remote_code, local_files_only=True, use_fast=False)
            tokenizers[model] = tokenizer
            for family, mechanisms in CONTRACTS.items():
                for mechanism, (variable, protocol_change, outcome) in mechanisms.items():
                    for item_index in range(12):
                        split = split_for(item_index)
                        for lexical_set, lexical_index in (("x", item_index), ("y", item_index + 12)):
                            target_task, control_task = pair_for(family, mechanism, lexical_index)
                            for demanded, task, letter in ((True, target_task, "A" if lexical_set == "x" else "C"), (False, control_task, "B" if lexical_set == "x" else "D")):
                                condition = next(value for value in CONDITIONS if value.startswith(letter))
                                for template in TEMPLATES:
                                    raw = render(task, template)
                                    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw, INTERFACE)
                                    target_ids = tokenizer(task["target"], add_special_tokens=False)["input_ids"]
                                    rows.append({
                                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                                        "case_id": f"phase353_{model}_{family}_{mechanism}_{item_index:02d}_{condition}_{template}",
                                        "semantic_case_id": f"phase353_{family}_{mechanism}_{item_index:02d}_{condition}_{template}",
                                        "contract_group_id": f"phase353_{family}_{mechanism}_{item_index:02d}_{template}",
                                        "model": model, "family_id": family, "mechanism_id": mechanism,
                                        "manipulated_variable": variable, "protocol_change_allowed": protocol_change,
                                        "outcome_type": outcome, "item_index": item_index, "lexical_index": lexical_index,
                                        "lexical_set": lexical_set, "contrast_condition": condition,
                                        "operation_demanded": demanded, "split": split, "template_id": template,
                                        "interface": INTERFACE, "answer_phase": answer_phase,
                                        "prompt": prompt, "raw_prompt": raw, "tokenization_add_special_tokens": add_special,
                                        "context": task["context"], "question": task["question"], "instruction": task["instruction"],
                                        "source_fragment": task["context"], "query_fragment": task["question"],
                                        "target": task["target"], "target_aliases": task["target_aliases"],
                                        "distractors": task["distractors"], "language": task["language"],
                                        "target_visible": task["target"].lower() in raw.lower(),
                                        "target_token_count": len(target_ids), "counterfactual_natural_by_design": True,
                                        "official_execution_mode": "b1_left_cache0", "baseline_only": True,
                                        "internal_intervention_allowed": False, "single_unit_intervention_allowed": False,
                                    })
        if len(rows) != 5184 or len({row["case_id"] for row in rows}) != 5184:
            raise RuntimeError(f"Invalid Phase353 denominator: {len(rows)}")
        pair_metrics: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for model in MODELS:
            for family, mechanisms in CONTRACTS.items():
                for mechanism in mechanisms:
                    selected = [row for row in rows if row["model"] == model and row["family_id"] == family and row["mechanism_id"] == mechanism]
                    by_key = {(row["item_index"], row["template_id"], row["contrast_condition"][0]): row for row in selected}
                    for item in range(12):
                        for template in TEMPLATES:
                            for left, right in (("A", "B"), ("C", "D")):
                                a, b = by_key[(item, template, left)], by_key[(item, template, right)]
                                metric = pair_metrics[(family, mechanism)]
                                metric["target_match"].append(a["target"] == b["target"])
                                metric["language_match"].append(a["language"] == b["language"])
                                metric["visibility_match"].append(a["target_visible"] == b["target_visible"])
                                metric["token_count_match"].append(a["target_token_count"] == b["target_token_count"])
                                metric["lexical_overlap"].append(overlap(a["context"] + " " + a["question"], b["context"] + " " + b["question"]))
        contracts = []
        for family, mechanisms in CONTRACTS.items():
            for mechanism, (variable, protocol_change, outcome) in mechanisms.items():
                metric = pair_metrics[(family, mechanism)]
                values = {
                    "target_match_rate": mean(metric["target_match"]),
                    "language_match_rate": mean(metric["language_match"]),
                    "target_visibility_match_rate": mean(metric["visibility_match"]),
                    "target_token_count_match_rate": mean(metric["token_count_match"]),
                    "mean_lexical_overlap": mean(metric["lexical_overlap"]),
                }
                strict = bool(
                    values["target_match_rate"] == 1 and values["language_match_rate"] == 1
                    and values["target_token_count_match_rate"] == 1
                    and values["target_visibility_match_rate"] >= 0.75
                    and values["mean_lexical_overlap"] >= 0.55
                )
                contracts.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "contract_id": f"phase353:{family}:{mechanism}", "family_id": family,
                    "mechanism_id": mechanism, "manipulated_variable": variable,
                    "protocol_change_allowed": protocol_change, "outcome_type": outcome,
                    **{key: round(value, 7) for key, value in values.items()},
                    "strict_contract_gate_pass": strict,
                    "mapping_status": "family_contract_qualified" if strict else "family_contract_repair_required",
                    "model_effect_used_for_contract": False, "single_unit_causal": False,
                })
        ready = sum(row["strict_contract_gate_pass"] for row in contracts)
        protocol = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "registered_case_count": len(rows), "contract_count": len(contracts),
            "entry_requires_qualified_contract_count": 6,
            "thresholds": {"target_match_rate": 1.0, "language_match_rate": 1.0, "target_token_count_match_rate": 1.0, "target_visibility_match_rate_min": 0.75, "mean_lexical_overlap_min": 0.55},
            "split_contract": {"physical_discovery": 6, "physical_calibration": 2, "physical_heldout": 2, "causal_sealed": 2},
            "claim_boundaries": [
                "Mechanical contract checks do not prove that the manipulated variable is isolated.",
                "Protocol and closure contracts may intentionally change protocol while preserving semantic content.",
                "No internal trace or intervention may run for rejected contracts.",
            ],
        }
        summary = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "denominator": {"registered_case_count": len(rows), "family_count": 9, "contract_count": 18, "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS}},
            "results": {"strict_contract_count": ready, "repair_required_contract_count": 18 - ready, "model_execution_started": False, "internal_intervention_executed_count": 0, "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0},
            "entry_decision": "run_baseline_qualification" if ready >= 6 else "repair_contracts_before_model_execution",
            "language_encoding_mechanism_closed": False, "intelligent_theory_experimentally_closed": False,
        }
        root = OUT / round_name
        write_jsonl(root / "phase353_registered_cases.jsonl", rows)
        write_jsonl(root / "phase353_contract_registry.jsonl", contracts)
        write_json(root / "phase353_registered_protocol.json", protocol)
        write_json(root / "phase353_contract_summary.json", summary)
        return summary
    finally:
        tokenizers.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
