#!/usr/bin/env python3
"""Freeze a fresh independent denominator for the Phase380 layout metric."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402


PHASE = "Phase380"
SCHEMA = "53.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("content_knowledge", "relation_binding"),
    ("state_drift", "entity_recency"),
    ("syntax_structure", "number_agreement"),
    ("readout_competition", "target_vs_wrong"),
)
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
GROUPS_PER_MECHANISM = 24
OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
TEMPLATES = (
    "Evidence dossier: {context}\nQuestion under test: {question}\nResponse contract: {instruction}\nAnswer:",
    "Use only this fresh record. {context}\nRequested decision: {question}\n{instruction}\nDecision:",
    "Case evidence follows. {context}\nOutput rule: {instruction}\nQuery: {question}\nValue:",
    "Fresh benchmark card: {context}\nTask: {question}\nConstraint: {instruction}\nResult:",
)
OBJECTS = (
    "beacon", "cabinet", "diary", "easel", "flask", "gong", "harpoon", "inbox",
    "jar", "lantern", "medal", "net", "orb", "plaque", "reel", "shield",
    "tablet", "urn", "visor", "wheel", "arch", "bowl", "crate", "dome",
)
MATERIALS = (
    "copper", "linen", "granite", "bamboo", "silver", "ceramic", "leather", "pine",
    "brass", "denim", "slate", "ivory", "steel", "paper", "rubber", "glass",
    "oak", "plastic", "wool", "canvas", "iron", "silk", "clay", "marble",
)
NAMES = (
    "Alina", "Bruno", "Cassia", "Darius", "Esther", "Felix", "Greta", "Hector",
    "Iris", "Jonas", "Kara", "Linus", "Marta", "Nolan", "Opal", "Pavel",
    "Rosa", "Silas", "Tessa", "Ulric", "Willa", "Xenia", "Yusuf", "Zelda",
)
NOUNS = (
    ("rabbit", "rabbits"), ("painter", "painters"), ("parcel", "parcels"),
    ("sailor", "sailors"), ("monument", "monuments"), ("violinist", "violinists"),
    ("lantern", "lanterns"), ("cabinet", "cabinets"), ("scholar", "scholars"),
    ("vessel", "vessels"), ("reporter", "reporters"), ("workshop", "workshops"),
    ("traveler", "travelers"), ("portrait", "portraits"), ("gardener", "gardeners"),
    ("packet", "packets"), ("merchant", "merchants"), ("temple", "temples"),
    ("musician", "musicians"), ("archive", "archives"), ("carpenter", "carpenters"),
    ("harbor", "harbors"), ("researcher", "researchers"), ("theater", "theaters"),
)
LABELS = (
    "amber", "azure", "beige", "coral", "emerald", "indigo", "lilac", "maroon",
    "olive", "pearl", "scarlet", "teal", "violet", "umber", "gold", "silver",
    "copper", "jade", "navy", "plum", "rose", "tan", "white", "black",
)
PREPOSITIONS = ("beside", "behind", "near", "opposite", "beyond", "alongside")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def task(
    mechanism: str, group_index: int, lexical_slot: str, operation: bool
) -> dict[str, Any]:
    offset = 0 if lexical_slot == "x" else 13
    index = (group_index + offset) % 24
    code = f"p380-{mechanism[:3]}-{group_index:02d}-{lexical_slot}"
    if mechanism == "relation_binding":
        obj = f"{OBJECTS[index]}-{code}"
        depot = f"vault-{code}"
        target = MATERIALS[index]
        wrong = MATERIALS[(index + 7) % 24]
        context = (
            f"The {obj} is cataloged in {depot}. Every artifact cataloged in {depot} is made of {target}."
            if operation
            else f"The {obj} is made of {target}. Its catalog location is {depot}."
        )
        question = f"What is the {obj} made of?"
        instruction = "Return only the material without explanation."
    elif mechanism == "entity_recency":
        target = NAMES[index]
        wrong = NAMES[(index + 9) % 24]
        context = (
            f"The authorized reviewer for {code} is {target}. A later unrelated travel note names {wrong}."
            if operation
            else f"An unrelated travel note names {wrong}. The authorized reviewer for {code} is {target}."
        )
        question = f"Who is the authorized reviewer for {code}?"
        instruction = "Return only the person's name without explanation."
    elif mechanism == "number_agreement":
        singular, plural = NOUNS[index]
        plural_head = index % 2 == 0
        head = plural if plural_head else singular
        target, wrong = ("are", "is") if plural_head else ("is", "are")
        attr_singular, attr_plural = NOUNS[(index + 7) % 24]
        attractor = attr_singular if plural_head else attr_plural
        subject = (
            f"the {head} {PREPOSITIONS[group_index % len(PREPOSITIONS)]} the {attractor}"
            if operation
            else f"the {head}"
        )
        context = (
            f"Identifier {code} is metadata. The head noun '{head}' controls agreement; the later noun does not."
            if operation
            else f"Identifier {code} is metadata. Apply ordinary agreement to the head noun '{head}'."
        )
        question = f"Fill the blank: {subject.capitalize()} ___ ready."
        instruction = "Return exactly one word: is or are."
    elif mechanism == "target_vs_wrong":
        target = LABELS[index]
        wrong = LABELS[(index + 11) % 24]
        context = (
            f"Registry {code} explicitly accepts {target} and explicitly rejects {wrong}."
            if operation
            else f"Registry {code} accepts {target}; no rejected label is relevant."
        )
        question = f"Which label does registry {code} accept?"
        instruction = "Return only the accepted label without explanation."
    else:
        raise KeyError(mechanism)
    return {
        "context": context,
        "question": question,
        "instruction": instruction,
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
        "language": "en",
    }


def main() -> None:
    execution_rows = []
    blind_rows = []
    label_rows = []
    prompt_hashes = set()
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
                semantic_group = f"phase380_{family}_{mechanism}_{group_index:02d}"
                parallel_group = "parallel380_" + digest(semantic_group)[:18]
                model_group = "group380_" + digest(f"{model}:{semantic_group}")[:18]
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
                        raise RuntimeError("Duplicate Phase380 rendered prompt")
                    prompt_hashes.add(prompt_hash)
                    case_id = "p380_" + digest(
                        f"{model}:{semantic_group}:{condition}"
                    )[:23]
                    common = {
                        "schema_version": SCHEMA,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "blind_case_id": case_id,
                        "anonymous_model_id": "am380_" + digest(model)[:11],
                        "anonymous_parallel_group_id": parallel_group,
                        "anonymous_group_id": model_group,
                        "anonymous_condition_slot": "slot380_"
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
                            "family_id": family,
                            "mechanism_id": mechanism,
                            "semantic_group_id": semantic_group,
                            "contrast_condition": condition,
                            "operation_demanded": letter in {"A", "C"},
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
    counts = Counter(row["mechanism_id"] for row in execution_rows)
    expected = len(MODELS) * len(MECHANISMS) * GROUPS_PER_MECHANISM * 4
    if len(execution_rows) != expected or set(counts.values()) != {288}:
        raise RuntimeError(f"Invalid Phase380 denominator: {len(execution_rows)}")
    execution_path = OUT / "private/phase380_execution_cases.jsonl"
    write_jsonl(execution_path, execution_rows)
    write_jsonl(OUT / "phase380_blind_case_registry.jsonl", blind_rows)
    write_jsonl(OUT / "private/phase380_label_key.jsonl", label_rows)
    protocol = {
        "schema_version": SCHEMA,
        "phase_id": "Phase380-Protocol",
        "created_at": now(),
        "objective": "independently_validate_backbone_residual_function_layout_before_any_causal_scan",
        "denominator": {
            "model_count": 3,
            "mechanism_count": 4,
            "groups_per_mechanism": GROUPS_PER_MECHANISM,
            "condition_count_per_group": 4,
            "case_count": len(execution_rows),
        },
        "frozen_metric": {
            "raw_cell_weight": "min(1,event_delta_norm/terminal_delta_norm)*abs(cosine(event_delta,terminal_delta))",
            "common_backbone": "per_model_per_axis_cellwise_mean_across_four_mechanisms",
            "function_residual": "mechanism_profile-common_backbone",
            "discovery_validation_profile_cosine_gate": 0.60,
            "heterogeneous_crossmodel_residual_cosine_gate": 0.40,
            "minimum_residual_norm_fraction": 0.05,
            "threshold_retuning_allowed": False,
        },
        "trace_contract": {
            "semantic_decision_aligned": True,
            "all_layers": True,
            "all_four_natural_boundaries": True,
            "source_query_current_roles": True,
            "all_six_condition_pairs": True,
            "top_k_used": False,
        },
        "authorization": {
            "run_behavior_sequentially": True,
            "run_internal_trace_before_behavior_gate": False,
            "run_causal_scan_before_residual_validation": False,
            "open_prior_physical_holdout": False,
            "run_single_neuron_scan": False,
        },
    }
    summary = {
        "schema_version": SCHEMA,
        "phase_id": PHASE,
        "created_at": now(),
        "case_count": len(execution_rows),
        "parallel_group_count": len(MECHANISMS) * GROUPS_PER_MECHANISM,
        "model_group_count": len(MODELS) * len(MECHANISMS) * GROUPS_PER_MECHANISM,
        "unique_rendered_prompt_count": len(prompt_hashes),
        "all_prompts_fresh_by_phase380_identifier": True,
        "internal_trace_started": False,
        "physical_holdout_opened": False,
    }
    write_json(OUT / "phase380_protocol.json", protocol)
    write_json(OUT / "phase380_case_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
