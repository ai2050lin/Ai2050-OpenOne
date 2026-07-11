#!/usr/bin/env python3
"""Register fresh protocol controls after the Phase345 contract failures."""

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


PHASE = "Phase346"
SCHEMA_VERSION = "22.0.0"
ROUND_DEFAULT = "three_core_protocol_repair"
OUT = ROOT / "tests/gpt5/result/phase346_protocol_repair"
INTERFACE = "answer_aligned_chat"
TEMPLATES = ("format_a", "format_b", "format_c")
TASKS = ("contiguous_multi_token_answer", "simple_no_source_answer")
PHRASES = (
    "silver harbor", "quiet forest", "amber valley", "winter garden", "copper bridge", "gentle river",
    "open meadow", "bright cedar", "hidden island", "soft thunder", "green lantern", "northern trail",
    "crystal shore", "silent station", "golden compass", "rapid current", "blue orchard", "stone tower",
    "white mountain", "red canyon", "black feather", "musical chamber", "heavy anchor", "calm ocean",
)
ARITHMETIC = (
    (1, 1, "two"), (1, 2, "three"), (2, 2, "four"), (2, 3, "five"),
    (3, 3, "six"), (3, 4, "seven"), (4, 4, "eight"), (4, 5, "nine"),
    (5, 5, "ten"), (1, 3, "four"), (1, 4, "five"), (1, 5, "six"),
    (2, 4, "six"), (2, 5, "seven"), (2, 6, "eight"), (3, 5, "eight"),
    (3, 6, "nine"), (3, 7, "ten"), (4, 6, "ten"), (1, 6, "seven"),
    (1, 7, "eight"), (2, 7, "nine"), (1, 8, "nine"), (2, 8, "ten"),
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


def task_for(task_id: str, index: int) -> dict[str, Any]:
    if task_id == "contiguous_multi_token_answer":
        target = PHRASES[index]
        context = f"The official route name is '{target}'."
        question = "What is the exact two-word route name?"
        distractors = [PHRASES[(index + 1) % 24], "unknown route"]
    else:
        left, right, target = ARITHMETIC[index]
        context = f"Solve the arithmetic problem {left} plus {right}; no answer is supplied."
        question = "What is the result, written as one English word?"
        distractors = ["zero", "one"]
    return {
        "context": context, "question": question,
        "instruction": "Return exactly the requested answer and nothing else.",
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
        for task_id in TASKS:
            for item_index in range(24):
                task = task_for(task_id, item_index)
                for template in TEMPLATES:
                    raw = render(task, template)
                    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, raw, INTERFACE)
                    rows.append({
                        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                        "case_id": f"phase346_{model}_{task_id}_{item_index:02d}_{template}",
                        "semantic_case_id": f"phase346_{task_id}_{item_index:02d}_{template}",
                        "model": model, "family_id": "protocol_control", "mechanism_id": task_id,
                        "task_class": "protocol_control", "item_index": item_index,
                        "split": split_for(item_index), "template_id": template,
                        "interface": INTERFACE, "answer_phase": answer_phase,
                        "prompt": prompt, "raw_prompt": raw,
                        "tokenization_add_special_tokens": add_special,
                        "source_fragment": task["source_fragment"], "query_fragment": task["query_fragment"],
                        "target": task["target"], "target_aliases": task["target_aliases"],
                        "distractors": task["distractors"], "official_execution_mode": "b1_left_cache0",
                        "baseline_only": True, "internal_intervention_allowed": False,
                    })
    if len(rows) != 432 or len({row["case_id"] for row in rows}) != 432:
        raise RuntimeError(f"Invalid Phase346 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Repair Phase345 protocol controls with contiguous phrases and simple no-source answers.",
        "registered_case_count": len(rows), "tasks": list(TASKS), "items_per_task": 24,
        "templates": list(TEMPLATES), "official_execution_mode": "b1_left_cache0",
        "thresholds": {"split_baseline_accuracy_min": 0.8, "split_phrase_valid_rate_min": 1.0},
        "trace_entry_requires_repaired_task_pass_min": 1,
        "claim_boundaries": [
            "Phase346 repairs the protocol denominator; it does not alter Phase345 results.",
            "No causal or neuron intervention is run.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows),
        "model_case_count": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase346_registered_cases.jsonl", rows)
    write_json(root / "phase346_registered_protocol.json", protocol)
    write_json(root / "phase346_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
