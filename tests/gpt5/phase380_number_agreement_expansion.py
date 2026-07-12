#!/usr/bin/env python3
"""Freeze a behavior-only number-agreement expansion for Phase380."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase380_independent_factorial_case_bank import (  # noqa: E402
    CONDITIONS,
    MODELS,
    now,
)


OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
EXPANSION = OUT / "number_agreement_expansion"
GROUP_COUNT = 72
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
    ("picture", "pictures"), ("bottle", "bottles"), ("desk", "desks"),
    ("road", "roads"), ("garden", "gardens"), ("river", "rivers"),
    ("worker", "workers"), ("driver", "drivers"),
)
PREPOSITIONS = ("near", "beside", "behind", "opposite", "alongside", "beyond")
TEMPLATES = (
    "Grammar evidence: {context}\nQuestion: {question}\nConstraint: {instruction}\nAnswer:",
    "Use this agreement record only. {context}\n{instruction}\nTask: {question}\nValue:",
    "Fresh syntax card: {context}\nFill the requested field. {question}\n{instruction}\nResult:",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def task(group_index: int, lexical_slot: str, operation: bool) -> dict[str, Any]:
    offset = 0 if lexical_slot == "x" else 17
    index = (group_index + offset) % len(NOUNS)
    singular, plural = NOUNS[index]
    plural_head = (group_index + offset) % 2 == 0
    head = plural if plural_head else singular
    target, wrong = ("are", "is") if plural_head else ("is", "are")
    attr_singular, attr_plural = NOUNS[(index + 11) % len(NOUNS)]
    attractor = attr_singular if plural_head else attr_plural
    code = f"p380x-num-{group_index:02d}-{lexical_slot}"
    if operation:
        subject = f"the {head} {PREPOSITIONS[group_index % len(PREPOSITIONS)]} the {attractor}"
        context = (
            f"Record {code}: the complete grammatical subject is '{subject}'. "
            f"The head noun is '{head}'. The noun after the preposition cannot control the verb."
        )
    else:
        subject = f"the {head}"
        context = (
            f"Record {code}: the complete grammatical subject is '{subject}'. "
            f"The head noun is '{head}'."
        )
    return {
        "context": context,
        "question": f"Fill only the blank: {subject.capitalize()} ___ ready.",
        "instruction": "Return exactly one word: is or are.",
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
        "language": "en",
    }


def main() -> None:
    execution_rows = []
    blind_rows = []
    hashes = set()
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for group_index in range(GROUP_COUNT):
            semantic_group = f"phase380x_syntax_structure_number_agreement_{group_index:02d}"
            parallel = "parallel380x_" + digest(semantic_group)[:18]
            model_group = "group380x_" + digest(f"{model}:{semantic_group}")[:18]
            items = {
                "A": task(group_index, "x", True),
                "B": task(group_index, "x", False),
                "C": task(group_index, "y", True),
                "D": task(group_index, "y", False),
            }
            for condition in CONDITIONS:
                item = items[condition[0]]
                raw_prompt = TEMPLATES[group_index % len(TEMPLATES)].format(**item)
                prompt, add_special, answer_phase = interface_prompt(
                    tokenizer, model, raw_prompt, "answer_aligned_chat"
                )
                prompt_hash = digest(prompt)
                if prompt_hash in hashes:
                    raise RuntimeError("Duplicate Phase380 expansion prompt")
                hashes.add(prompt_hash)
                case_id = "p380x_" + digest(
                    f"{model}:{semantic_group}:{condition}"
                )[:22]
                common = {
                    "schema_version": "53.4.0",
                    "phase_id": "Phase380-AgreementExpansion",
                    "created_at": now(),
                    "blind_case_id": case_id,
                    "anonymous_model_id": "am380_" + digest(model)[:11],
                    "anonymous_parallel_group_id": parallel,
                    "anonymous_group_id": model_group,
                    "anonymous_condition_slot": "slot380x_"
                    + digest(f"{model_group}:{condition}")[:10],
                    "phase380_split": "independent_residual_validation",
                    "prompt": prompt,
                    "raw_prompt": raw_prompt,
                    "source_fragment": item["context"],
                    "query_fragment": item["question"],
                    "tokenization_add_special_tokens": add_special,
                    "prompt_token_count": len(
                        tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    ),
                    "interface": "answer_aligned_chat",
                    "answer_phase": answer_phase,
                }
                execution_rows.append(
                    {
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
                    }
                )
                blind_rows.append(
                    {
                        **common,
                        "semantic_label_used_for_validation": False,
                        "target_or_distractor_exported": False,
                    }
                )
    if len(execution_rows) != GROUP_COUNT * 4 * len(MODELS):
        raise RuntimeError("Invalid Phase380 agreement expansion denominator")
    write_jsonl(EXPANSION / "private/phase380x_execution_cases.jsonl", execution_rows)
    write_jsonl(EXPANSION / "phase380x_blind_case_registry.jsonl", blind_rows)
    protocol = {
        "schema_version": "53.4.0",
        "phase_id": "Phase380-AgreementExpansion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "original_behavior_only_number_agreement_groups_failed_the_frozen_eight_group_gate",
        "decision_basis": "behavior_only_before_any_phase380_internal_trace",
        "group_count": GROUP_COUNT,
        "case_count": len(execution_rows),
        "original_number_agreement_groups_retired_as_a_complete_cohort": True,
        "failed_groups_replaced": False,
        "internal_trace_opened": False,
        "threshold_changed": False,
    }
    write_json(EXPANSION / "phase380x_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
