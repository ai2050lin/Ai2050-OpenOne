#!/usr/bin/env python3
"""Freeze an independent natural-form replacement for Phase369 number agreement."""

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
    digest, now, split_for, write_json, write_jsonl,
)


OUT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister_number_agreement_replacement"
INITIAL = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister"
NOUNS = (
    ("lantern", "lanterns"), ("tablet", "tablets"), ("compass", "compasses"),
    ("parcel", "parcels"), ("cabinet", "cabinets"), ("medallion", "medallions"),
    ("vessel", "vessels"), ("tripod", "tripods"), ("goblet", "goblets"),
    ("satchel", "satchels"), ("flask", "flasks"), ("casket", "caskets"),
)
ATTRACTORS = (
    ("cabinet", "cabinets"), ("curator", "curators"), ("archive", "archives"),
    ("shelf", "shelves"), ("workshop", "workshops"), ("registry", "registries"),
)


def task(group_index: int, lexical_slot: str, demanded: bool) -> dict[str, Any]:
    offset = 0 if lexical_slot == "x" else 5
    index = (group_index + offset) % len(NOUNS)
    plural = (group_index + offset) % 2 == 0
    singular, plural_form = NOUNS[index]
    head = plural_form if plural else singular
    target, wrong = ("are", "is") if plural else ("is", "are")
    attr_singular, attr_plural = ATTRACTORS[(group_index + offset) % len(ATTRACTORS)]
    opposite_attractor = attr_singular if plural else attr_plural
    code = f"p369r-num-{group_index:02d}-{lexical_slot}"
    if demanded:
        subject = f"the {head} near the {opposite_attractor}"
        context = (
            f"Record {code} marks the whole phrase '{subject}' as the grammatical subject. "
            "The noun after 'near' does not control the verb."
        )
    else:
        subject = f"the {head}"
        context = f"Record {code} marks '{subject}' as the complete grammatical subject."
    return {
        "context": context,
        "question": f"Fill only the blank with is or are: {subject.capitalize()} ___ ready.",
        "instruction": "Return exactly one word: is or are.",
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
        "language": "en",
    }


def render(item: dict[str, Any], group_index: int) -> str:
    return TEMPLATES[group_index % len(TEMPLATES)].format(
        context=item["context"], question=item["question"], instruction=item["instruction"],
    )


def main() -> None:
    old_blind = [
        json.loads(line) for line in (INITIAL / "phase369_blind_case_registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_hashes = {digest(row["raw_prompt"]) for row in old_blind} | {digest(row["prompt"]) for row in old_blind}
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
        for group_index in range(12):
            split = split_for(group_index)
            semantic_group = f"phase369r_syntax_structure_number_agreement_{group_index:02d}"
            parallel_group = "parallel_r_" + digest(semantic_group)[:18]
            model_group = "group_r_" + digest(f"{model}:{semantic_group}")[:18]
            condition_items = {
                "A": task(group_index, "x", True),
                "B": task(group_index, "x", False),
                "C": task(group_index, "y", True),
                "D": task(group_index, "y", False),
            }
            for condition in CONDITIONS:
                item = condition_items[condition[0]]
                raw_prompt = render(item, group_index)
                prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw_prompt, INTERFACE)
                if digest(raw_prompt) in old_hashes or digest(prompt) in old_hashes:
                    raise RuntimeError("Replacement prompt overlaps the initial Phase369 bank")
                if digest(prompt) in prompt_hashes:
                    raise RuntimeError("Duplicate replacement rendered prompt")
                prompt_hashes.add(digest(prompt))
                blind_case_id = "p369r_" + digest(f"{model}:{semantic_group}:{condition}")[:23]
                common = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "blind_case_id": blind_case_id,
                    "anonymous_model_id": "am369_" + digest(model)[:12],
                    "anonymous_parallel_group_id": parallel_group,
                    "anonymous_group_id": model_group,
                    "anonymous_condition_slot": "slot_r_" + digest(f"{model_group}:{condition}")[:10],
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
    if len(execution_rows) != 144 or len(prompt_hashes) != 144:
        raise RuntimeError("Invalid replacement denominator")
    amendment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "reason": "initial_number_agreement_subjects_used_non_natural_hyphenated_plural_forms",
        "decision_basis": "behavior_qualification_only_no_internal_trace_or_holdout_was_opened",
        "initial_number_agreement_contract_status": "retired_from_phase369_admissible_denominator_evidence_retained",
        "replacement_contract": "natural_head_noun_with_optional_opposite_number_attractor",
        "replacement_is_independent": True,
        "physical_holdout_execution": False,
        "qualification_gate_unchanged": {
            "all_four_conditions_all_three_models": True,
            "minimum_discovery_parallel_groups": 4,
            "minimum_calibration_parallel_groups": 2,
        },
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_count": len(execution_rows),
        "model_count": 3,
        "parallel_group_count": 12,
        "fresh_discovery_case_count": split_counts["fresh_discovery"],
        "fresh_calibration_case_count": split_counts["fresh_calibration"],
        "physical_holdout_case_count": split_counts["physical_holdout_sealed"],
        "unique_rendered_prompt_count": len(prompt_hashes),
        "initial_prompt_overlap_count": 0,
        "physical_holdout_opened": False,
        "next_decision": "run_replacement_behavior_qualification_sequentially_qwen3_glm4_deepseek7b",
    }
    write_json(OUT / "phase369_number_agreement_protocol_amendment.json", amendment)
    write_json(OUT / "phase369_number_agreement_case_bank_summary.json", summary)
    write_jsonl(OUT / "phase369_number_agreement_blind_cases.jsonl", blind_rows)
    write_jsonl(OUT / "private/phase369_number_agreement_execution_cases.jsonl", execution_rows)
    write_jsonl(OUT / "private/phase369_number_agreement_label_key.jsonl", label_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
