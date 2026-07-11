#!/usr/bin/env python3
"""Freeze the Phase333 dynamic-sequence and residual-block denominator."""

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


PHASE = "Phase333"
SCHEMA_VERSION = "11.0.0"
ROUND_DEFAULT = "dynamic_path_atlas"
OUT = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas"
INTERFACES = ("raw_completion", "native_chat", "answer_aligned_chat")
MECHANISMS = (
    ("missing_condition_control", "positive", "two_hop_blocked"),
    ("two_hop_blocked", "matched_negative_control", "missing_condition_control"),
)
BLOCK_CONDITIONS = (
    "baseline",
    "correct_block_1",
    "correct_block_2",
    "correct_block_4",
    "wrong_object_block_4",
    "wrong_interface_block_4",
    "wrong_time_block_4",
    "moment_matched_permutation_block_4",
    "matched_control_block_4",
)

NAMES = (
    "Ilan", "Jora", "Kemi", "Luan", "Mira", "Niko",
    "Orin", "Pela", "Ravi", "Sena", "Toma", "Vela",
)
PROPERTIES = (
    ("bronze", "narrow", "certified"),
    ("gentle", "opaque", "stored"),
    ("dense", "cool", "inspected"),
    ("plain", "flexible", "usable"),
    ("violet", "rough", "tagged"),
    ("mature", "rapid", "admitted"),
    ("silver", "short", "verified"),
    ("polite", "sharp", "permitted"),
    ("hollow", "dry", "recorded"),
    ("level", "soft", "reserved"),
    ("scarlet", "thin", "accepted"),
    ("steady", "pale", "released"),
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


def split_for(item_index: int) -> str:
    if item_index < 6:
        return "discovery"
    if item_index < 9:
        return "calibration"
    return "heldout"


def task_for(mechanism: str, item_index: int) -> dict[str, Any]:
    name = NAMES[item_index]
    prop_a, prop_b, conclusion = PROPERTIES[item_index]
    if mechanism == "missing_condition_control":
        return {
            "context": (
                f"If something is {prop_a} and {prop_b}, then it is {conclusion}. "
                f"{name} is {prop_a}. No fact states whether {name} is {prop_b}."
            ),
            "question": f"Is {name} definitely {conclusion}?",
            "instruction": "Answer yes, no, or unknown only.",
            "target": "unknown",
            "target_aliases": ["unknown", "cannot determine", "not enough information"],
            "distractors": ["yes", "no"],
            "target_class": "missing_premise",
        }
    if mechanism == "two_hop_blocked":
        group = f"class{item_index + 21}"
        return {
            "context": (
                f"Every {group} is {prop_a}. No {prop_a} thing is {prop_b}. "
                f"{name} is a {group}."
            ),
            "question": f"Is {name} {prop_b}?",
            "instruction": "Answer yes, no, or unknown only.",
            "target": "no",
            "target_aliases": ["no", "false", "contradicted"],
            "distractors": ["yes", "unknown"],
            "target_class": "contradicted",
        }
    raise KeyError(mechanism)


def render_raw(task: dict[str, Any], template_id: str) -> str:
    if template_id == "template_a":
        return f"{task['context']}\n{task['question']}\n{task['instruction']}\nAnswer:"
    if template_id == "template_b":
        return (
            f"Facts: {task['context']}\nDecision: {task['question']}\n"
            f"Output rule: {task['instruction']}\nResponse:"
        )
    if template_id == "template_c":
        return f"{task['context']}\n{task['instruction']}\nQuery: {task['question']}\nVerdict:"
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
) -> tuple[str, bool, str]:
    if interface == "raw_completion":
        return raw_prompt, True, "answer_start"
    native = native_chat(tokenizer, raw_prompt)
    if interface == "native_chat":
        phase = "think_start" if model in {"qwen3", "deepseek7b"} else "assistant_start"
        return native, False, phase
    if interface == "answer_aligned_chat":
        if model == "qwen3":
            aligned = native_chat(tokenizer, raw_prompt, enable_thinking=False)
            return aligned + "Final answer:", False, "visible_answer_start"
        if model == "deepseek7b" and native.endswith("<think>\n"):
            return native + "</think>\nFinal answer:", False, "visible_answer_start"
        return native + "Final answer:", False, "visible_answer_start"
    raise KeyError(interface)


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    rows = []
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        for mechanism, cohort, paired in MECHANISMS:
            for item_index in range(12):
                task = task_for(mechanism, item_index)
                split = split_for(item_index)
                for template_id in TEMPLATES:
                    raw_prompt = render_raw(task, template_id)
                    for interface in INTERFACES:
                        prompt, add_special, answer_phase = interface_prompt(
                            tokenizer, model, raw_prompt, interface
                        )
                        rows.append({
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "created_at": now(),
                            "case_id": (
                                f"phase333_{model}_{mechanism}_{item_index:02d}_"
                                f"{template_id}_{interface}"
                            ),
                            "semantic_case_id": f"phase333_{mechanism}_{item_index:02d}_{template_id}",
                            "item_id": f"phase333_{mechanism}_{item_index:02d}",
                            "model": model,
                            "family_id": "reasoning_constraint",
                            "mechanism_id": mechanism,
                            "cohort": cohort,
                            "paired_mechanism_id": paired,
                            "item_index": item_index,
                            "split": split,
                            "template_id": template_id,
                            "interface": interface,
                            "answer_phase": answer_phase,
                            "prompt": prompt,
                            "raw_prompt": raw_prompt,
                            "tokenization_add_special_tokens": add_special,
                            "source_fragments": [task["context"]],
                            "query_fragment": task["question"],
                            "target": task["target"],
                            "target_aliases": task["target_aliases"],
                            "distractors": task["distractors"],
                            "target_class": task["target_class"],
                            "language": "en",
                            "protocol": "short_verdict",
                            "expected_structure": "plain",
                            "selection_eligible": split == "discovery",
                            "selection_updates_allowed": False,
                            "single_unit_intervention_gate_open": False,
                        })
    if len(rows) != 648:
        raise RuntimeError(f"Expected 648 cases, got {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate Phase333 case id")
    write_jsonl(root / "phase333_registered_cases.jsonl", rows)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "purpose": "Test dynamic event order, contiguous residual-state blocks, and lagged compensation.",
        "positive_mechanism": "reasoning_constraint/missing_condition_control",
        "matched_negative_control": "reasoning_constraint/two_hop_blocked",
        "new_item_count_per_mechanism": 12,
        "discovery_items": list(range(6)),
        "calibration_items": [6, 7, 8],
        "heldout_items": [9, 10, 11],
        "templates": list(TEMPLATES),
        "interfaces": list(INTERFACES),
        "models": list(MODELS),
        "registered_case_count": 648,
        "max_new_tokens": 64,
        "dynamic_components": [
            "residual_input", "normalized_input", "attention_output", "mlp_output",
        ],
        "functional_events": [
            "answer_start", "target_pressure_formation", "competitor_overtake",
            "target_first_appearance", "error_first_appearance", "stop", "final_readout",
        ],
        "block_lengths": [1, 2, 4],
        "block_patch_target": "contiguous transformer-layer residual outputs at one frozen functional time",
        "block_conditions": list(BLOCK_CONDITIONS),
        "thresholds": {
            "object_event_presence_min": 0.6666667,
            "relative_depth_tolerance": 0.15,
            "phrase_logprob_improvement_min": 0.10,
            "target_rank_improvement_min": 1.0,
            "behavior_gain_min": 0.10,
            "control_superiority_min": 0.05,
            "side_effect_max": 0.10,
            "compensation_explained_rate_min": 0.6666667,
        },
        "success_gate": [
            "dynamic_sequence_stable", "state_block_effective", "competition_consistent",
            "compensation_explained", "free_generation_improved", "matched_controls_clean",
            "cross_model",
        ],
        "selection_updates_allowed": False,
        "theory_update_gate_open": False,
        "single_unit_intervention_gate_open": False,
    }
    write_json(root / "phase333_registered_protocol.json", protocol)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "case_count": len(rows),
        "model_counts": {model: sum(row["model"] == model for row in rows) for model in MODELS},
        "mechanism_counts": {
            mechanism: sum(row["mechanism_id"] == mechanism for row in rows)
            for mechanism, _cohort, _paired in MECHANISMS
        },
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout")
        },
        "interface_counts": {
            interface: sum(row["interface"] == interface for row in rows)
            for interface in INTERFACES
        },
        "valid": len(rows) == 648,
    }
    write_json(root / "phase333_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
