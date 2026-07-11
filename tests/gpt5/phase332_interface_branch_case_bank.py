#!/usr/bin/env python3
"""Freeze the Phase332 interface-branch and answer-phase denominator."""

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


PHASE = "Phase332"
SCHEMA_VERSION = "10.0.0"
ROUND_DEFAULT = "interface_branch_atlas"
OUT = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas"
INTERFACES = ("raw_completion", "native_chat", "chat_no_think", "answer_aligned_chat")

MECHANISMS = (
    ("language_action", "summarize", "positive", "rewrite"),
    ("language_action", "rewrite", "matched_negative_control", "summarize"),
    ("reasoning_constraint", "missing_condition_control", "positive", "two_hop_blocked"),
    ("reasoning_constraint", "two_hop_blocked", "matched_negative_control", "missing_condition_control"),
)

EXCHANGE_CONDITIONS = (
    "baseline",
    "shared_skeleton_correct",
    "interface_branch_correct",
    "shared_plus_branch_correct",
    "shared_plus_branch_wrong_item",
    "matched_random_units_correct",
)

TOPICS = (
    "astronomy", "ecology", "finance", "cooking",
    "education", "architecture", "medicine", "robotics",
)

REWRITES = (
    ("swift", "rapid"),
    ("silent", "quiet"),
    ("begin", "start"),
    ("assist", "help"),
    ("purchase", "buy"),
    ("difficult", "hard"),
    ("tiny", "small"),
    ("select", "choose"),
)

NAMES = ("Ari", "Bela", "Cyra", "Davi", "Enzo", "Fara", "Gino", "Hera")
PROPERTIES = (
    ("amber", "round", "marked"),
    ("calm", "bright", "approved"),
    ("solid", "warm", "stable"),
    ("quiet", "open", "ready"),
    ("green", "smooth", "listed"),
    ("young", "swift", "selected"),
    ("blue", "large", "flagged"),
    ("kind", "clear", "accepted"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def task_for(mechanism: str, item_index: int) -> dict[str, Any]:
    if mechanism == "summarize":
        topic = TOPICS[item_index]
        return {
            "context": (
                f"The passage describes several examples from {topic}, then compares their causes, "
                f"uses, and practical effects without changing its central subject."
            ),
            "question": "Summarize the central topic in one word.",
            "instruction": "Return only the topic word.",
            "target": topic,
            "distractors": ["history", "weather"],
            "target_class": "present",
        }
    if mechanism == "rewrite":
        source, target = REWRITES[item_index]
        return {
            "context": f"The source word is {source}.",
            "question": "Rewrite it using the registered common synonym.",
            "instruction": "Return only the replacement word.",
            "target": target,
            "distractors": [source, "other"],
            "target_class": "transformed",
        }
    name = NAMES[item_index]
    prop_a, prop_b, conclusion = PROPERTIES[item_index]
    if mechanism == "missing_condition_control":
        return {
            "context": (
                f"If something is {prop_a} and {prop_b}, it is {conclusion}. "
                f"{name} is {prop_a}, but no information about being {prop_b} is given."
            ),
            "question": f"Can we conclude that {name} is {conclusion}?",
            "instruction": "Answer yes or no only.",
            "target": "no",
            "distractors": ["yes"],
            "target_class": "absent",
        }
    if mechanism == "two_hop_blocked":
        cls = f"group{item_index + 1}"
        return {
            "context": (
                f"Every {cls} is {prop_a}. No {prop_a} thing is {prop_b}. "
                f"{name} is a {cls}."
            ),
            "question": f"Can we conclude that {name} is {prop_b}?",
            "instruction": "Answer yes or no only.",
            "target": "no",
            "distractors": ["yes"],
            "target_class": "absent",
        }
    raise KeyError(mechanism)


def render_raw(task: dict[str, Any], template_id: str) -> str:
    if template_id == "template_a":
        return f"{task['context']}\n{task['question']}\n{task['instruction']}\nAnswer:"
    if template_id == "template_b":
        return (
            f"Context: {task['context']}\nTask: {task['question']}\n"
            f"Instruction: {task['instruction']}\nResponse:"
        )
    if template_id == "template_c":
        return f"{task['context']}\n{task['instruction']}\nQuestion: {task['question']}\nFinal:"
    raise KeyError(template_id)


def native_chat(tokenizer: Any, raw_prompt: str, **kwargs: Any) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def interface_prompt(
    tokenizer: Any, model: str, raw_prompt: str, interface: str
) -> tuple[str, bool, str, str | None]:
    if interface == "raw_completion":
        return raw_prompt, True, "answer_start", None
    native = native_chat(tokenizer, raw_prompt)
    if interface == "native_chat":
        phase = "think_start" if model == "deepseek7b" else "assistant_start"
        return native, False, phase, None
    if interface == "chat_no_think":
        if model == "qwen3":
            prompt = native_chat(tokenizer, raw_prompt, enable_thinking=False)
            return prompt, False, "visible_answer_start", None
        if model == "deepseek7b":
            if not native.endswith("<think>\n"):
                raise RuntimeError("Unexpected DeepSeek native chat suffix")
            return native + "</think>\n", False, "visible_answer_start", None
        return native, False, "assistant_start", "native_chat"
    if interface == "answer_aligned_chat":
        if model == "deepseek7b" and native.endswith("<think>\n"):
            return native + "</think>\nFinal answer:", False, "visible_answer_start", None
        return native + "Final answer:", False, "visible_answer_start", None
    raise KeyError(interface)


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for family, mechanism, cohort, paired in MECHANISMS:
            for item_index in range(8):
                task = task_for(mechanism, item_index)
                split = "discovery" if item_index < 4 else "heldout"
                for template_id in TEMPLATES:
                    raw_prompt = render_raw(task, template_id)
                    for interface in INTERFACES:
                        prompt, add_special, answer_phase, equivalent = interface_prompt(
                            tokenizer, model, raw_prompt, interface
                        )
                        rows.append({
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "created_at": now(),
                            "case_id": (
                                f"phase332_{model}_{family}_{mechanism}_{item_index:02d}_"
                                f"{template_id}_{interface}"
                            ),
                            "semantic_case_id": f"phase332_{family}_{mechanism}_{item_index:02d}_{template_id}",
                            "item_id": f"phase332_{family}_{mechanism}_{item_index:02d}",
                            "model": model,
                            "family_id": family,
                            "mechanism_id": mechanism,
                            "cohort": cohort,
                            "paired_mechanism_id": paired,
                            "item_index": item_index,
                            "split": split,
                            "template_id": template_id,
                            "interface": interface,
                            "interface_equivalent_to": equivalent,
                            "answer_phase": answer_phase,
                            "prompt": prompt,
                            "raw_prompt": raw_prompt,
                            "tokenization_add_special_tokens": add_special,
                            "source_fragments": [task["context"]],
                            "query_fragment": task["question"],
                            "target": task["target"],
                            "target_aliases": [task["target"]],
                            "distractors": task["distractors"],
                            "target_bucket": "yes_no" if task["target"] == "no" else "lexical",
                            "target_class": task["target_class"],
                            "target_absent_from_prompt": task["target"].lower() not in raw_prompt.lower(),
                            "language": "en",
                            "protocol": "short",
                            "expected_structure": "plain",
                            "selection_eligible": split == "discovery",
                            "selection_updates_allowed": False,
                            "single_unit_intervention_gate_open": False,
                        })
    expected = len(MODELS) * len(MECHANISMS) * 8 * len(TEMPLATES) * len(INTERFACES)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} cases, got {len(rows)}")
    write_jsonl(root / "phase332_registered_cases.jsonl", rows)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "purpose": "Map interface-conditioned natural paths and test answer-phase-aligned path exchange.",
        "positive_mechanisms": ["language_action/summarize", "reasoning_constraint/missing_condition_control"],
        "matched_negative_controls": ["language_action/rewrite", "reasoning_constraint/two_hop_blocked"],
        "new_item_count_per_mechanism": 8,
        "discovery_items": [0, 1, 2, 3],
        "heldout_items": [4, 5, 6, 7],
        "templates": list(TEMPLATES),
        "interfaces": list(INTERFACES),
        "models": list(MODELS),
        "registered_interface_case_count": expected,
        "unique_prompt_case_count": expected - 96,
        "glm4_native_no_think_equivalent_case_count": 96,
        "component_selection": {
            "split": "discovery_only",
            "item_sign_consistency_min": 0.75,
            "top_fraction_per_component_role": 0.10,
            "max_members_per_component_role": 12,
            "shared_skeleton": "exact physical unit intersection across unique interfaces within one model",
            "interface_branch": "stable top units in one interface absent from all other interface top sets",
        },
        "exchange_directions": ["raw_to_answer_aligned", "answer_aligned_to_raw"],
        "exchange_conditions": list(EXCHANGE_CONDITIONS),
        "max_new_tokens": 64,
        "thresholds": {
            "heldout_item_direction_consistency_min": 0.75,
            "phrase_logprob_improvement_min": 0.10,
            "behavior_gain_min": 0.10,
            "behavior_side_effect_max": 0.10,
            "protocol_side_effect_max": 0.10,
        },
        "success_gate": [
            "shared_skeleton_stable", "interface_branch_specific", "path_exchange_effective",
            "full_string_improved", "free_generation_improved", "low_side_effect", "cross_model",
        ],
        "selection_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
        "theory_update_gate_open": False,
    }
    write_json(root / "phase332_registered_protocol.json", protocol)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_count": len(rows),
        "semantic_prompt_count": len({row["semantic_case_id"] for row in rows}),
        "new_item_count": len({(row["family_id"], row["mechanism_id"], row["item_index"]) for row in rows}),
        "model_counts": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("discovery", "heldout")},
        "equivalent_interface_case_count": sum(row["interface_equivalent_to"] is not None for row in rows),
        "valid": len(rows) == 1152,
    }
    write_json(root / "phase332_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
