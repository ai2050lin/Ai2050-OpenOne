#!/usr/bin/env python3
"""Freeze a larger common-noun number-agreement expansion for Phase369."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase369_protocol_and_case_bank import (  # noqa: E402
    CONDITIONS, INTERFACE, MODELS, PHASE, SCHEMA_VERSION, TEMPLATES,
    digest, now, write_json, write_jsonl,
)


OUT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister_number_agreement_expansion"
INITIAL = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister"
REPLACEMENT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister_number_agreement_replacement"
NOUNS = (
    ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("car", "cars"),
    ("teacher", "teachers"), ("student", "students"), ("key", "keys"),
    ("cup", "cups"), ("chair", "chairs"), ("bird", "birds"),
    ("apple", "apples"), ("train", "trains"), ("doctor", "doctors"),
    ("parent", "parents"), ("table", "tables"), ("phone", "phones"),
    ("pen", "pens"), ("tree", "trees"), ("flower", "flowers"),
    ("boat", "boats"), ("plane", "planes"), ("shirt", "shirts"),
    ("shoe", "shoes"), ("house", "houses"), ("window", "windows"),
    ("door", "doors"), ("clock", "clocks"), ("lamp", "lamps"),
    ("picture", "pictures"), ("bottle", "bottles"),
)
PREPOSITIONS = ("beside", "near", "behind", "in front of", "next to")


def split_for_expansion(group_index: int) -> str:
    if group_index < 18:
        return "fresh_discovery"
    if group_index < 27:
        return "fresh_calibration"
    return "physical_holdout_sealed"


def task(group_index: int, lexical_slot: str, demanded: bool) -> dict[str, Any]:
    offset = 0 if lexical_slot == "x" else 11
    index = (group_index + offset) % len(NOUNS)
    plural = (group_index + offset) % 2 == 0
    singular, plural_form = NOUNS[index]
    head = plural_form if plural else singular
    target, wrong = ("are", "is") if plural else ("is", "are")
    attr_singular, attr_plural = NOUNS[(index + 7) % len(NOUNS)]
    attractor = attr_singular if plural else attr_plural
    code = f"p369x-num-{group_index:02d}-{lexical_slot}"
    if demanded:
        subject = f"the {head} {PREPOSITIONS[group_index % len(PREPOSITIONS)]} the {attractor}"
        context = f"Case label {code} is metadata. Use the head noun '{head}' to control the verb."
    else:
        subject = f"the {head}"
        context = f"Case label {code} is metadata. Apply ordinary English subject-verb agreement."
    return {
        "context": context,
        "question": f"Fill the blank: {subject.capitalize()} ___ ready.",
        "instruction": "Answer with exactly is or are.",
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
        "language": "en",
    }


def render(item: dict[str, Any], group_index: int) -> str:
    return TEMPLATES[group_index % len(TEMPLATES)].format(
        context=item["context"], question=item["question"], instruction=item["instruction"],
    )


def existing_hashes() -> set[str]:
    paths = (
        INITIAL / "phase369_blind_case_registry.jsonl",
        REPLACEMENT / "phase369_number_agreement_blind_cases.jsonl",
    )
    values = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                values.add(digest(row["raw_prompt"]))
                values.add(digest(row["prompt"]))
    return values


def main() -> None:
    old_hashes = existing_hashes()
    execution_rows = []
    blind_rows = []
    label_rows = []
    prompt_hashes = set()
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for group_index in range(30):
            split = split_for_expansion(group_index)
            semantic_group = f"phase369x_syntax_structure_number_agreement_{group_index:02d}"
            parallel_group = "parallel_x_" + digest(semantic_group)[:18]
            model_group = "group_x_" + digest(f"{model}:{semantic_group}")[:18]
            items = {
                "A": task(group_index, "x", True),
                "B": task(group_index, "x", False),
                "C": task(group_index, "y", True),
                "D": task(group_index, "y", False),
            }
            for condition in CONDITIONS:
                item = items[condition[0]]
                raw_prompt = render(item, group_index)
                prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw_prompt, INTERFACE)
                prompt_hash = digest(prompt)
                if digest(raw_prompt) in old_hashes or prompt_hash in old_hashes:
                    raise RuntimeError("Expansion prompt overlaps a prior Phase369 bank")
                if prompt_hash in prompt_hashes:
                    raise RuntimeError("Duplicate expansion prompt")
                prompt_hashes.add(prompt_hash)
                blind_case_id = "p369x_" + digest(f"{model}:{semantic_group}:{condition}")[:23]
                common = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "blind_case_id": blind_case_id,
                    "anonymous_model_id": "am369_" + digest(model)[:12],
                    "anonymous_parallel_group_id": parallel_group,
                    "anonymous_group_id": model_group,
                    "anonymous_condition_slot": "slot_x_" + digest(f"{model_group}:{condition}")[:10],
                    "phase369_split": split,
                    "prompt": prompt,
                    "raw_prompt": raw_prompt,
                    "source_fragment": item["context"],
                    "query_fragment": item["question"],
                    "tokenization_add_special_tokens": add_special,
                    "prompt_token_count": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                    "interface": INTERFACE,
                    "answer_phase": answer_phase,
                }
                execution_rows.append({
                    **common,
                    "private_execution_model": model,
                    "family_id": "syntax_structure",
                    "mechanism_id": "number_agreement",
                    "semantic_group_id": semantic_group,
                    "contrast_condition": condition,
                    "operation_demanded": condition[0] in {"A", "C"},
                    "target": item["target"],
                    "target_aliases": item["target_aliases"],
                    "distractors": item["distractors"],
                    "language": item["language"],
                    "instruction": item["instruction"],
                    "question": item["question"],
                    "semantic_labels_available_to_collector": False,
                    "target_specific_competition_available_to_collector": False,
                })
                blind_rows.append({
                    **common,
                    "semantic_label_used_for_selection": False,
                    "target_or_distractor_exported": False,
                })
                label_rows.append({
                    "blind_case_id": blind_case_id,
                    "model": model,
                    "family_id": "syntax_structure",
                    "mechanism_id": "number_agreement",
                    "semantic_group_id": semantic_group,
                    "contrast_condition": condition,
                    "phase369_split": split,
                    "target": item["target"],
                    "target_aliases": item["target_aliases"],
                    "distractors": item["distractors"],
                })
    split_counts = Counter(row["phase369_split"] for row in execution_rows)
    if len(execution_rows) != 360 or len(prompt_hashes) != 360:
        raise RuntimeError("Invalid number-agreement expansion denominator")
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "reason": "increase_prequalified_common_noun_denominator_after_independent_replacement_remained_underpowered",
        "decision_basis": "behavior_only_no_internal_trace_no_physical_holdout",
        "group_counts": {"fresh_discovery": 18, "fresh_calibration": 9, "physical_holdout_sealed": 3},
        "all_four_conditions_all_three_models_gate_unchanged": True,
        "physical_holdout_execution": False,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_count": len(execution_rows),
        "model_count": 3,
        "parallel_group_count": 30,
        "fresh_discovery_case_count": split_counts["fresh_discovery"],
        "fresh_calibration_case_count": split_counts["fresh_calibration"],
        "physical_holdout_case_count": split_counts["physical_holdout_sealed"],
        "unique_rendered_prompt_count": len(prompt_hashes),
        "prior_prompt_overlap_count": 0,
        "physical_holdout_opened": False,
    }
    write_json(OUT / "phase369_number_agreement_expansion_protocol.json", protocol)
    write_json(OUT / "phase369_number_agreement_expansion_summary.json", summary)
    write_jsonl(OUT / "phase369_number_agreement_expansion_blind_cases.jsonl", blind_rows)
    write_jsonl(OUT / "private/phase369_number_agreement_expansion_execution_cases.jsonl", execution_rows)
    write_jsonl(OUT / "private/phase369_number_agreement_expansion_label_key.jsonl", label_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
