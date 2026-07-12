#!/usr/bin/env python3
"""Freeze fresh blind groups for joint upstream-state causal validation."""

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


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
P380 = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("reasoning_constraint", "relation_binding"),
    ("state_drift", "entity_recency"),
    ("readout_competition", "target_vs_wrong"),
)
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
GROUPS_PER_MECHANISM = 24
TEMPLATES = (
    "Independent ledger: {context}\nRequested field: {question}\nReply rule: {instruction}\nReply:",
    "Consult only this sealed note. {context}\nQuestion: {question}\n{instruction}\nAnswer:",
    "New audit record: {context}\nExtraction request: {question}\nConstraint: {instruction}\nOutput:",
    "Isolated evidence card. {context}\nTask: {question}\nFormat: {instruction}\nValue:",
)
OBJECTS = (
    "anchor", "badge", "compass", "drum", "envelope", "funnel", "goblet", "helmet",
    "inkwell", "jacket", "kettle", "loom", "mirror", "notebook", "oar", "pillow",
    "quill", "ribbon", "saddle", "tripod", "uniform", "vase", "whistle", "yoke",
)
MATERIALS = (
    "bronze", "cotton", "basalt", "cedar", "nickel", "porcelain", "felt", "maple",
    "tin", "hemp", "quartz", "bone", "aluminum", "cardboard", "latex", "crystal",
    "birch", "resin", "velvet", "plaster", "pewter", "satin", "sandstone", "obsidian",
)
NAMES = (
    "Adrian", "Bianca", "Cyrus", "Delia", "Emil", "Freya", "Gavin", "Helena",
    "Imani", "Jasper", "Keira", "Leander", "Mina", "Nico", "Oriana", "Priya",
    "Rafael", "Selene", "Tobias", "Una", "Vera", "Wyatt", "Yara", "Zane",
)
LABELS = (
    "apricot", "blue", "cream", "crimson", "forest", "gray", "lavender", "magenta",
    "ochre", "pink", "ruby", "turquoise", "wine", "yellow", "bronze", "chrome",
    "ebony", "green", "orange", "purple", "saffron", "sepia", "snow", "charcoal",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def task(mechanism: str, group_index: int, lexical_slot: str, operation: bool) -> dict[str, Any]:
    index = (group_index + (0 if lexical_slot == "x" else 11)) % 24
    code = f"p381-{mechanism[:3]}-{group_index:02d}-{lexical_slot}"
    if mechanism == "relation_binding":
        obj = f"{OBJECTS[index]}-{code}"
        site = f"archive-{code}"
        target = MATERIALS[index]
        wrong = MATERIALS[(index + 7) % 24]
        context = (
            f"The {obj} is filed in {site}. Every item filed in {site} is made of {target}."
            if operation
            else f"The {obj} is made of {target}. Its filing site is {site}."
        )
        question = f"What is the {obj} made of?"
        instruction = "Return only the material without explanation."
    elif mechanism == "entity_recency":
        target = NAMES[index]
        wrong = NAMES[(index + 9) % 24]
        context = (
            f"The certified inspector for {code} is {target}. A later unrelated receipt mentions {wrong}."
            if operation
            else f"An unrelated receipt mentions {wrong}. The certified inspector for {code} is {target}."
        )
        question = f"Who is the certified inspector for {code}?"
        instruction = "Return only the person's name without explanation."
    elif mechanism == "target_vs_wrong":
        target = LABELS[index]
        wrong = LABELS[(index + 13) % 24]
        context = (
            f"Protocol {code} explicitly approves {target} and explicitly excludes {wrong}."
            if operation
            else f"Protocol {code} approves {target}; excluded labels are irrelevant."
        )
        question = f"Which label does protocol {code} approve?"
        instruction = "Return only the approved label without explanation."
    else:
        raise KeyError(mechanism)
    return {
        "context": context,
        "question": question,
        "instruction": instruction,
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
    }


def main() -> None:
    prior_hashes = {
        digest(row["prompt"])
        for row in read_jsonl(P380 / "private/phase380_execution_cases.jsonl")
    }
    execution_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for family, mechanism in MECHANISMS:
            for group_index in range(GROUPS_PER_MECHANISM):
                semantic_group = f"phase381_{family}_{mechanism}_{group_index:02d}"
                parallel_group = "parallel381_" + digest(semantic_group)[:18]
                model_group = "group381_" + digest(f"{model}:{semantic_group}")[:18]
                items = {
                    "A": task(mechanism, group_index, "x", True),
                    "B": task(mechanism, group_index, "x", False),
                    "C": task(mechanism, group_index, "y", True),
                    "D": task(mechanism, group_index, "y", False),
                }
                for condition in CONDITIONS:
                    letter = condition[0]
                    item = items[letter]
                    raw_prompt = TEMPLATES[group_index % len(TEMPLATES)].format(**item)
                    prompt, add_special, answer_phase = interface_prompt(
                        tokenizer, model, raw_prompt, "answer_aligned_chat"
                    )
                    prompt_hash = digest(prompt)
                    if prompt_hash in prompt_hashes:
                        raise RuntimeError("Duplicate Phase381 rendered prompt")
                    prompt_hashes.add(prompt_hash)
                    case_id = "p381_" + digest(f"{model}:{semantic_group}:{condition}")[:23]
                    common = {
                        "schema_version": "54.0.0",
                        "phase_id": "Phase381",
                        "created_at": now(),
                        "blind_case_id": case_id,
                        "anonymous_model_id": "am381_" + digest(model)[:11],
                        "anonymous_parallel_group_id": parallel_group,
                        "anonymous_group_id": model_group,
                        "anonymous_condition_slot": "slot381_"
                        + digest(f"{model_group}:{condition}")[:10],
                        "phase381_split": "fresh_joint_state_physical_validation",
                        "prompt": prompt,
                        "raw_prompt": raw_prompt,
                        "source_fragment": item["context"],
                        "query_fragment": item["question"],
                        "prompt_token_count": len(
                            tokenizer(prompt, add_special_tokens=False)["input_ids"]
                        ),
                        "tokenization_add_special_tokens": add_special,
                        "interface": "answer_aligned_chat",
                        "answer_phase": answer_phase,
                    }
                    execution_rows.append(
                        {
                            **common,
                            "private_execution_model": model,
                            "family_id": family,
                            "mechanism_id": mechanism,
                            "semantic_group_id": semantic_group,
                            "contrast_condition": condition,
                            "operation_demanded": letter in {"A", "C"},
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "language": "en",
                        }
                    )
                    blind_rows.append(
                        {
                            **common,
                            "semantic_label_used_for_validation": False,
                            "target_or_distractor_exported": False,
                        }
                    )
                    label_rows.append(
                        {
                            "blind_case_id": case_id,
                            "model": model,
                            "family_id": family,
                            "mechanism_id": mechanism,
                            "semantic_group_id": semantic_group,
                            "contrast_condition": condition,
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                        }
                    )
    expected = len(MODELS) * len(MECHANISMS) * GROUPS_PER_MECHANISM * 4
    prior_overlap = len(prompt_hashes & prior_hashes)
    if len(execution_rows) != expected or prior_overlap:
        raise RuntimeError(
            f"Invalid Phase381 denominator cases={len(execution_rows)} overlap={prior_overlap}"
        )
    write_jsonl(OUT / "private/phase381_execution_cases.jsonl", execution_rows)
    write_jsonl(OUT / "phase381_blind_case_registry.jsonl", blind_rows)
    write_jsonl(OUT / "private/phase381_label_key.jsonl", label_rows)
    protocol = {
        "schema_version": "54.0.0",
        "phase_id": "Phase381-Protocol",
        "created_at": now(),
        "objective": "distinguish_single_position_failure_from_joint_distributed_upstream_state",
        "denominator": {
            "model_count": 3,
            "mechanism_count": 3,
            "groups_per_mechanism": GROUPS_PER_MECHANISM,
            "conditions_per_group": 4,
            "case_count": len(execution_rows),
        },
        "role_sets": ["source", "query", "current", "source_query_current"],
        "scan_grid": {
            "relative_depths": ["early", "middle_early", "middle", "middle_late", "late"],
            "component_boundaries": ["layer_input", "attention_output", "mlp_output", "layer_output"],
            "conditions": ["natural_swap", "equal_energy_permutation"],
        },
        "frozen_joint_gates": {
            "minimum_natural_transfer_gain": 0.10,
            "minimum_gain_over_equal_energy": 0.05,
            "minimum_terminal_transfer_share": 0.02,
            "minimum_share_over_equal_energy": 0.01,
            "minimum_gain_over_best_single_position": 0.05,
            "minimum_gain_over_cyclic_wrong_depth": 0.05,
            "minimum_gain_over_cyclic_wrong_component": 0.05,
            "minimum_transfer_to_offtarget_rms_ratio": 0.05,
            "minimum_groups_all_four_directions": 6,
        },
        "claim_boundary": {
            "joint_state_cell_is_complete_path": False,
            "joint_state_cell_is_same_neurons_across_models": False,
            "terminal_interface_is_search_target": False,
            "single_neuron_scan": False,
            "language_encoding_mechanism_closed": False,
        },
        "execution_order": list(MODELS),
        "authorization": {
            "run_behavior_sequentially": True,
            "run_trace_before_behavior_gate": False,
            "run_joint_scan_before_group_freeze": False,
            "reuse_phase380_cases": False,
            "open_single_neuron_scan": False,
        },
    }
    summary = {
        "schema_version": "54.0.0",
        "phase_id": "Phase381-CaseBank",
        "created_at": now(),
        "case_count": len(execution_rows),
        "parallel_group_count": len(MECHANISMS) * GROUPS_PER_MECHANISM,
        "model_group_count": len(MODELS) * len(MECHANISMS) * GROUPS_PER_MECHANISM,
        "unique_rendered_prompt_count": len(prompt_hashes),
        "phase380_prompt_overlap_count": prior_overlap,
        "internal_trace_started": False,
    }
    write_json(OUT / "phase381_protocol.json", protocol)
    write_json(OUT / "phase381_case_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
