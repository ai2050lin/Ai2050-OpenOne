#!/usr/bin/env python3
"""Freeze Phase407 event-horizon condition-response cases.

Phase407 separates semantic completion, sentence boundary, and model stop.
It also compares the same finite state when the state is carried inline in the
current turn or by a semantically equivalent prior turn.  Literal responses
are normalized per family-specific interface before any transfer gate.
"""

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
from phase386_multitime_protocol import NAMES  # noqa: E402
from phase404_direct_state_protocol import (  # noqa: E402
    FROZEN_DTYPES,
    MODELS,
    SURFACE_REPLICAS,
)


OUT = ROOT / "tests/gpt5/result/phase407_event_horizon_kernel"
SCHEMA_VERSION = "81.0.0"
FAMILIES = ("knowledge_binding", "rule_reasoning", "grammar_constraint")
SPLIT_GROUP_COUNTS = {
    "discovery": 12,
    "calibration": 6,
    "behavioral_holdout": 6,
    "physical_holdout": 6,
}
STATE_IDS = {
    "knowledge_binding": (
        "green_yellow",
        "yellow_green",
        "green_green",
        "yellow_yellow",
    ),
    "rule_reasoning": ("holder_a", "holder_b"),
    "grammar_constraint": (
        "singular_present",
        "plural_present",
        "singular_past",
        "plural_past",
    ),
}
INTERFACES = {
    "knowledge_binding": ("natural_qa", "ordered_cloze"),
    "rule_reasoning": ("conclusion_completion", "truth_condition"),
    "grammar_constraint": ("minimal_contrast", "syntax_completion"),
}
HISTORY_MODES = ("inline_no_prior_turn", "prior_turn_carried_state")
MAX_NEW_TOKENS = 48
TOP_K = 8
QUALIFICATION_CASES_PER_FAMILY_MODEL = 8

FRESH_NOUNS = (
    ("anchor", "anchors"),
    ("basket", "baskets"),
    ("candle", "candles"),
    ("drawer", "drawers"),
    ("envelope", "envelopes"),
    ("feather", "feathers"),
    ("guitar", "guitars"),
    ("helmet", "helmets"),
    ("island", "islands"),
    ("jar", "jars"),
    ("kite", "kites"),
    ("ladder", "ladders"),
    ("magnet", "magnets"),
    ("notebook", "notebooks"),
    ("orchard", "orchards"),
    ("pillow", "pillows"),
    ("quilt", "quilts"),
    ("radio", "radios"),
    ("spoon", "spoons"),
    ("tower", "towers"),
    ("umbrella", "umbrellas"),
    ("vessel", "vessels"),
    ("window", "windows"),
    ("xylophone", "xylophones"),
    ("yacht", "yachts"),
    ("zipper", "zippers"),
    ("bottle", "bottles"),
    ("camera", "cameras"),
    ("desk", "desks"),
    ("fountain", "fountains"),
    ("glove", "gloves"),
    ("harbor", "harbors"),
    ("igloo", "igloos"),
    ("jacket", "jackets"),
    ("kettle", "kettles"),
    ("lamp", "lamps"),
    ("mirror", "mirrors"),
    ("napkin", "napkins"),
    ("oven", "ovens"),
    ("paddle", "paddles"),
    ("rocket", "rockets"),
    ("saddle", "saddles"),
    ("table", "tables"),
    ("uniform", "uniforms"),
    ("violin", "violins"),
    ("wheel", "wheels"),
    ("box", "boxes"),
    ("cabin", "cabins"),
    ("door", "doors"),
    ("flower", "flowers"),
    ("globe", "globes"),
    ("hat", "hats"),
    ("insect", "insects"),
    ("kennel", "kennels"),
    ("leaf", "leaves"),
    ("mountain", "mountains"),
    ("necklace", "necklaces"),
    ("package", "packages"),
    ("robot", "robots"),
    ("shelf", "shelves"),
)
TRIGGERS = (
    "password",
    "coin",
    "stamp",
    "ribbon",
    "card",
    "whistle",
    "compass",
    "medal",
    "tag",
    "coupon",
    "pin",
    "token",
)
INTERMEDIATES = (
    "wristband",
    "clearance",
    "voucher",
    "passcode",
    "certificate",
    "bracelet",
    "license",
    "emblem",
    "docket",
    "lanyard",
    "insignia",
    "authorization",
)
TERMINALS = (
    "chamber",
    "gallery",
    "depot",
    "studio",
    "workshop",
    "courtyard",
    "observatory",
    "library",
    "greenhouse",
    "laboratory",
    "theater",
    "hangar",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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


def split_for(priority: int) -> str:
    cursor = 0
    for split, count in SPLIT_GROUP_COUNTS.items():
        if priority < cursor + count:
            return split
        cursor += count
    raise ValueError(priority)


def entity_pair(priority: int, lexical: int) -> tuple[str, str]:
    first = (17 * priority + 9 * lexical + 5) % len(NAMES)
    second = (first + 7 + 3 * lexical + priority % 5) % len(NAMES)
    if first == second:
        raise RuntimeError("Phase407 entity collision")
    return NAMES[first], NAMES[second]


def package_for(family: str, priority: int, lexical: int) -> dict[str, str]:
    index = 2 * priority + lexical
    entity_a, entity_b = entity_pair(priority, lexical)
    if family == "knowledge_binding":
        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "item": FRESH_NOUNS[index][0],
        }
    if family == "rule_reasoning":
        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "trigger": TRIGGERS[index % len(TRIGGERS)],
            "intermediate": INTERMEDIATES[(index * 5 + 1) % len(INTERMEDIATES)],
            "terminal": TERMINALS[(index * 7 + 2) % len(TERMINALS)],
        }
    singular, plural = FRESH_NOUNS[index]
    return {"noun_singular": singular, "noun_plural": plural}


def state_truth(family: str, state_id: str) -> tuple[int, ...]:
    if family == "knowledge_binding":
        return {
            "green_yellow": (0, 1),
            "yellow_green": (1, 0),
            "green_green": (0, 0),
            "yellow_yellow": (1, 1),
        }[state_id]
    if family == "rule_reasoning":
        return (0,) if state_id == "holder_a" else (1,)
    return {
        "singular_present": (0, 0),
        "plural_present": (1, 0),
        "singular_past": (0, 1),
        "plural_past": (1, 1),
    }[state_id]


def be_form(state_id: str) -> str:
    number, tense = state_truth("grammar_constraint", state_id)
    return (("is", "are"), ("was", "were"))[tense][number]


def facts_for(
    family: str,
    package: dict[str, str],
    state_id: str,
    syntax: int,
    order: int,
    prior_turn: bool,
) -> str:
    truth = state_truth(family, state_id)
    if family == "knowledge_binding":
        colors = ("green", "yellow")
        clauses = [
            f"person A ({package['entity_a']}) owns a {colors[truth[0]]} {package['item']}",
            f"person B ({package['entity_b']}) owns a {colors[truth[1]]} {package['item']}",
        ]
        if order:
            clauses.reverse()
        if prior_turn:
            return "Please retain this final ownership record: " + "; while ".join(clauses) + "."
        if syntax == 0:
            return "Final ownership record:\n- " + "\n- ".join(clauses) + "."
        return "For the final state, " + "; meanwhile, ".join(clauses) + "."
    if family == "rule_reasoning":
        holder = truth[0]
        labels = ("A", "B")
        names = (package["entity_a"], package["entity_b"])
        clauses = [
            f"person {labels[holder]} ({names[holder]}) carries the {package['trigger']}",
            f"person {labels[1 - holder]} ({names[1 - holder]}) does not carry the {package['trigger']}",
        ]
        if order:
            clauses.reverse()
        rules = [
            f"whoever carries the {package['trigger']} receives the {package['intermediate']}",
            f"whoever receives the {package['intermediate']} reaches the {package['terminal']}",
        ]
        if prior_turn:
            return "Remember this closed-world case: " + "; whereas ".join(clauses) + ". The rules say " + ", then ".join(rules) + "."
        if syntax == 0:
            return "Final facts:\n- " + "\n- ".join(clauses) + ".\nRules:\n- " + "\n- ".join(rules) + "."
        return "In the final closed world, " + "; while ".join(clauses) + ". Also, " + ", and ".join(rules) + "."

    number, tense = truth
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    quantity = "one" if number == 0 else "two"
    tense_word = "present" if tense == 0 else "past"
    clauses = [
        f"the subject is {quantity} {noun}",
        f"the required time is {tense_word}",
    ]
    if order:
        clauses.reverse()
    if prior_turn:
        return "Keep this grammar state for the next turn: " + "; and ".join(clauses) + "."
    if syntax == 0:
        return "Final grammar state:\n- " + "\n- ".join(clauses) + "."
    return "For the pending sentence, " + "; meanwhile, ".join(clauses) + "."


def primary_foil_state(family: str, state_id: str) -> str:
    if family == "knowledge_binding":
        return {
            "green_yellow": "yellow_green",
            "yellow_green": "green_yellow",
            "green_green": "yellow_yellow",
            "yellow_yellow": "green_green",
        }[state_id]
    if family == "rule_reasoning":
        return "holder_b" if state_id == "holder_a" else "holder_a"
    return {
        "singular_present": "plural_present",
        "plural_present": "singular_present",
        "singular_past": "plural_past",
        "plural_past": "singular_past",
    }[state_id]


def semantic_aliases(
    family: str, package: dict[str, str], interface: str
) -> dict[str, list[str]]:
    if family == "knowledge_binding":
        return {
            "green_yellow": [
                "green, yellow",
                "green and yellow",
                "green then yellow",
                "green / yellow",
                "A green, B yellow",
            ],
            "yellow_green": [
                "yellow, green",
                "yellow and green",
                "yellow then green",
                "yellow / green",
                "A yellow, B green",
            ],
            "green_green": ["green, green", "both green", "green and green"],
            "yellow_yellow": ["yellow, yellow", "both yellow", "yellow and yellow"],
        }
    if family == "rule_reasoning":
        if interface == "truth_condition":
            return {
                "holder_a": ["yes", "true"],
                "holder_b": ["no", "false"],
            }
        return {
            "holder_a": ["A", "person A", package["entity_a"]],
            "holder_b": ["B", "person B", package["entity_b"]],
        }
    return {
        "singular_present": ["is"],
        "plural_present": ["are"],
        "singular_past": ["was"],
        "plural_past": ["were"],
    }


def interface_contract(
    family: str,
    package: dict[str, str],
    state_id: str,
    interface: str,
    syntax: int,
) -> dict[str, str]:
    foil_state = primary_foil_state(family, state_id)
    if family == "knowledge_binding":
        colors = ("green", "yellow")
        left, right = (colors[index] for index in state_truth(family, state_id))
        foil_left, foil_right = (
            colors[index] for index in state_truth(family, foil_state)
        )
        if interface == "natural_qa":
            return {
                "query": (
                    "Report the two final colors in person A then person B order. "
                    "Give the two color words in that order, then end the sentence."
                ),
                "assistant_prefix": "Final answer:",
                "target_completion": f" {left}, {right}.",
                "foil_completion": f" {foil_left}, {foil_right}.",
            }
        return {
            "query": (
                "Complete the assistant's A-then-B color sentence with both "
                "colors, and end it at the first sentence boundary."
            ),
            "assistant_prefix": "In A-then-B order, the final colors are",
            "target_completion": f" {left}, then {right}.",
            "foil_completion": f" {foil_left}, then {foil_right}.",
        }
    if family == "rule_reasoning":
        holder = state_truth(family, state_id)[0]
        foil_holder = state_truth(family, foil_state)[0]
        if interface == "conclusion_completion":
            return {
                "query": (
                    f"Conclude which person reaches the {package['terminal']}. "
                    "Give the person label and end the sentence."
                ),
                "assistant_prefix": (
                    f"Therefore, the person reaching the {package['terminal']} is"
                ),
                "target_completion": f" {('A', 'B')[holder]}.",
                "foil_completion": f" {('A', 'B')[foil_holder]}.",
            }
        return {
            "query": (
                f"Does person A reach the {package['terminal']} under both rules? "
                "Answer yes or no, then end the sentence."
            ),
            "assistant_prefix": "Final answer:",
            "target_completion": f" {('yes', 'no')[holder]}.",
            "foil_completion": f" {('yes', 'no')[foil_holder]}.",
        }

    number, _tense = state_truth(family, state_id)
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    target_form = be_form(state_id)
    foil_form = be_form(foil_state)
    adjective = "ready" if syntax == 0 else "prepared"
    if interface == "minimal_contrast":
        return {
            "query": (
                f"Which be-form makes 'The {noun} ___ {adjective}' agree with "
                "the final grammar state? State the form and end the sentence."
            ),
            "assistant_prefix": "Required form:",
            "target_completion": f" {target_form}.",
            "foil_completion": f" {foil_form}.",
        }
    return {
        "query": (
            "Complete the assistant's sentence according to the final grammar "
            "state, then stop at its first sentence boundary."
        ),
        "assistant_prefix": f"The {noun}",
        "target_completion": f" {target_form} {adjective}.",
        "foil_completion": f" {foil_form} {adjective}.",
    }


def render_chat(
    tokenizer: Any,
    model: str,
    messages: list[dict[str, str]],
    assistant_prefix: str,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    if model == "deepseek7b" and prompt.endswith("<think>\n"):
        prompt += "</think>\n"
    return prompt + assistant_prefix


def make_prompt(
    tokenizer: Any,
    model: str,
    family: str,
    package: dict[str, str],
    state_id: str,
    surface: dict[str, Any],
    interface: str,
    history_mode: str,
) -> tuple[str, dict[str, str], str]:
    contract = interface_contract(
        family, package, state_id, interface, surface["syntax"]
    )
    prior_turn = history_mode == "prior_turn_carried_state"
    facts = facts_for(
        family,
        package,
        state_id,
        surface["syntax"],
        surface["order"],
        prior_turn,
    )
    if prior_turn:
        messages = [
            {"role": "user", "content": facts},
            {
                "role": "assistant",
                "content": "I have recorded that final state for the follow-up.",
            },
            {"role": "user", "content": contract["query"]},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": facts + "\n\nFollow-up: " + contract["query"],
            }
        ]
    return (
        render_chat(tokenizer, model, messages, contract["assistant_prefix"]),
        contract,
        facts,
    )


def previous_prompt_hashes() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    paths = (
        ROOT
        / "tests/gpt5/result/phase403_predictive_state/protocol/private/phase403_all_cases.jsonl",
        ROOT
        / "tests/gpt5/result/phase404_direct_predictive_state/protocol/private/phase404_all_cases.jsonl",
        ROOT
        / "tests/gpt5/result/phase405_natural_future_state/protocol/private/phase405_all_cases.jsonl",
        ROOT
        / "tests/gpt5/result/phase406_conditioned_sequence_state/protocol/private/phase406_all_cases.jsonl",
    )
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = row.get("private_execution_model") or row.get("model")
            prompt = row.get("prompt")
            if model and prompt:
                result.add((model, digest(prompt)))
    return result


def semantic_transition_table() -> dict[str, list[dict[str, str]]]:
    return {
        "knowledge_binding": [
            {"source": "green_yellow", "operation": "swap", "target": "yellow_green"},
            {"source": "yellow_green", "operation": "swap", "target": "green_yellow"},
            {"source": "green_yellow", "operation": "copy_a_to_b", "target": "green_green"},
            {"source": "yellow_green", "operation": "copy_a_to_b", "target": "yellow_yellow"},
        ],
        "rule_reasoning": [
            {"source": "holder_a", "operation": "swap_holder", "target": "holder_b"},
            {"source": "holder_b", "operation": "swap_holder", "target": "holder_a"},
        ],
        "grammar_constraint": [
            {"source": "singular_present", "operation": "toggle_number", "target": "plural_present"},
            {"source": "plural_present", "operation": "toggle_number", "target": "singular_present"},
            {"source": "singular_past", "operation": "toggle_number", "target": "plural_past"},
            {"source": "plural_past", "operation": "toggle_number", "target": "singular_past"},
            {"source": "singular_present", "operation": "toggle_tense", "target": "singular_past"},
            {"source": "plural_present", "operation": "toggle_tense", "target": "plural_past"},
        ],
    }


def main() -> None:
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )

    created_at = now()
    previous_hashes = previous_prompt_hashes()
    current_hashes: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    group_fingerprints: set[str] = set()
    total_groups = sum(SPLIT_GROUP_COUNTS.values())

    for family in FAMILIES:
        for priority in range(total_groups):
            split = split_for(priority)
            group_id = "p407g_" + digest(f"{family}:{priority}", 24)
            packages = [package_for(family, priority, lexical) for lexical in (0, 1)]
            fingerprint = digest(
                json.dumps([family, packages], sort_keys=True, ensure_ascii=True), 32
            )
            if fingerprint in group_fingerprints:
                raise RuntimeError(f"Phase407 repeated semantic group: {family}/{priority}")
            group_fingerprints.add(fingerprint)
            registry.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase407-EventHorizonProtocol",
                    "family_id": family,
                    "anonymous_parallel_group_id": group_id,
                    "candidate_split": split,
                    "group_priority": priority,
                    "state_count": len(STATE_IDS[family]),
                    "surface_replica_count": len(SURFACE_REPLICAS),
                    "interface_count": len(INTERFACES[family]),
                    "history_count": len(HISTORY_MODES),
                    "semantic_group_fingerprint": fingerprint,
                }
            )

            for state_index, state_id in enumerate(STATE_IDS[family]):
                for surface in SURFACE_REPLICAS:
                    package = package_for(family, priority, surface["lexical"])
                    for interface in INTERFACES[family]:
                        aliases = semantic_aliases(family, package, interface)
                        for history_mode in HISTORY_MODES:
                            for model, tokenizer in tokenizers.items():
                                prompt, contract, facts = make_prompt(
                                    tokenizer,
                                    model,
                                    family,
                                    package,
                                    state_id,
                                    surface,
                                    interface,
                                    history_mode,
                                )
                                prompt_ids = tokenizer(
                                    prompt, add_special_tokens=False
                                )["input_ids"]
                                target_ids = tokenizer(
                                    contract["target_completion"],
                                    add_special_tokens=False,
                                )["input_ids"]
                                foil_ids = tokenizer(
                                    contract["foil_completion"],
                                    add_special_tokens=False,
                                )["input_ids"]
                                if not prompt_ids or not target_ids or not foil_ids:
                                    raise RuntimeError("Phase407 empty token sequence")
                                case_key = (
                                    f"{model}:{family}:{group_id}:{state_id}:"
                                    f"{surface['surface_id']}:{interface}:{history_mode}"
                                )
                                prompt_hash = (model, digest(prompt))
                                if prompt_hash in previous_hashes:
                                    raise RuntimeError(
                                        f"Phase407 overlaps Phase403-406: {case_key}"
                                    )
                                if prompt_hash in current_hashes:
                                    raise RuntimeError(
                                        f"Phase407 duplicate: {current_hashes[prompt_hash]} == {case_key}"
                                    )
                                current_hashes[prompt_hash] = case_key
                                qualification = (
                                    split == "discovery"
                                    and priority == 0
                                    and state_index < 2
                                    and surface["surface_id"] == "r000"
                                )
                                rows.append(
                                    {
                                        "schema_version": SCHEMA_VERSION,
                                        "phase_id": "Phase407-EventHorizonProtocol",
                                        "created_at": created_at,
                                        "private_execution_model": model,
                                        "blind_case_id": "p407c_" + digest(case_key, 28),
                                        "family_id": family,
                                        "anonymous_parallel_group_id": group_id,
                                        "parallel_group_id_private": f"p407_private_{family}_{priority:02d}",
                                        "candidate_split_private": split,
                                        "group_priority": priority,
                                        "state_id_private": state_id,
                                        "abstract_state_private": list(state_truth(family, state_id)),
                                        "surface_id_private": surface["surface_id"],
                                        "surface_axes_private": surface,
                                        "interface_private": interface,
                                        "history_mode_private": history_mode,
                                        "condition_id_private": f"{interface}__{history_mode}",
                                        "prompt": prompt,
                                        "prompt_token_ids_private": [int(item) for item in prompt_ids],
                                        "prompt_token_count": len(prompt_ids),
                                        "state_facts_private": facts,
                                        "assistant_prefix_private": contract["assistant_prefix"],
                                        "target_semantic_state_private": state_id,
                                        "foil_semantic_state_private": primary_foil_state(family, state_id),
                                        "semantic_state_ids_private": list(STATE_IDS[family]),
                                        "semantic_aliases_by_state_private": aliases,
                                        "target_completion_private": contract["target_completion"],
                                        "target_completion_token_ids_private": [int(item) for item in target_ids],
                                        "foil_completion_private": contract["foil_completion"],
                                        "foil_completion_token_ids_private": [int(item) for item in foil_ids],
                                        "registered_candidate_first_token_ids_private": sorted(
                                            {
                                                int(
                                                    tokenizer(
                                                        interface_contract(
                                                            family,
                                                            package,
                                                            candidate_state,
                                                            interface,
                                                            surface["syntax"],
                                                        )["target_completion"],
                                                        add_special_tokens=False,
                                                    )["input_ids"][0]
                                                )
                                                for candidate_state in STATE_IDS[family]
                                            }
                                        ),
                                        "max_new_tokens": MAX_NEW_TOKENS,
                                        "top_k_ledger_size": TOP_K,
                                        "execution_qualification_case": qualification,
                                        "formal_denominator": True,
                                    }
                                )

    expected_rows = sum(
        total_groups
        * len(STATE_IDS[family])
        * len(SURFACE_REPLICAS)
        * len(INTERFACES[family])
        * len(HISTORY_MODES)
        * len(MODELS)
        for family in FAMILIES
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Phase407 row count {len(rows)} != {expected_rows}")
    for model in MODELS:
        for family in FAMILIES:
            count = sum(
                row["execution_qualification_case"]
                and row["private_execution_model"] == model
                and row["family_id"] == family
                for row in rows
            )
            if count != QUALIFICATION_CASES_PER_FAMILY_MODEL:
                raise RuntimeError(
                    f"Phase407 qualification count {model}/{family}: {count}"
                )

    write_jsonl(OUT / "protocol/private/phase407_all_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase407_blind_group_registry.jsonl", registry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase407-EventHorizonProtocol",
        "created_at": created_at,
        "objective": "separate_event_times_surface_interface_history_and_stop_before_any_physical_mapping",
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "device": "cuda",
            "runtime_dtype_by_model": FROZEN_DTYPES,
            "attention_implementation": "eager",
            "generation": "deterministic_greedy",
            "batch_size": 1,
            "padding": "none",
            "max_new_tokens": MAX_NEW_TOKENS,
            "termination": "model_eos_or_H48_only",
            "event_times_are_observed_without_early_semantic_truncation": True,
            "right_censor_missing_semantic_boundary_or_stop_at_H48": True,
            "full_vocab_tensor_persisted": False,
            "compressed_probability_ledger": [
                "raw_top_k_logits_and_logprobs",
                "full_vocab_logsumexp",
                "full_vocab_entropy",
                "top_k_tail_probability_mass",
                "registered_candidate_first_token_mass",
                "generated_token_logprob",
                "canonical_target_sequence_logprob",
                "canonical_foil_sequence_logprob",
            ],
        },
        "denominator": {
            "families": list(FAMILIES),
            "groups_per_family": total_groups,
            "split_group_counts": SPLIT_GROUP_COUNTS,
            "states_per_family": {
                family: len(STATE_IDS[family]) for family in FAMILIES
            },
            "surface_replicas": len(SURFACE_REPLICAS),
            "interfaces_per_family": 2,
            "history_modes": list(HISTORY_MODES),
            "case_count_all_models_all_splits": len(rows),
            "discovery_case_count_per_model": 1920,
            "discovery_case_count_all_models": 5760,
            "execution_qualification_case_count_per_model": 24,
        },
        "family_specific_interfaces": INTERFACES,
        "event_contract": {
            "tau_semantic": "first_complete_normalized_semantic_state",
            "tau_boundary": "first_legal_sentence_boundary_after_semantic_completion",
            "tau_stop": "first_model_eos",
            "events_recorded_separately": True,
            "tau_star_is_not_minimum_and_does_not_truncate_generation": True,
        },
        "independent_integer_gates": {
            "surface_unit": "3_of_4_surfaces_per_state_interface_history",
            "interface_transfer_unit": "3_of_4_paired_surfaces_per_state_history",
            "history_transfer_unit": "3_of_4_paired_surfaces_per_state_interface",
            "sequence_group": {
                "knowledge_binding": "56_of_64",
                "rule_reasoning": "28_of_32",
                "grammar_constraint": "56_of_64",
            },
            "stop_group": {
                "knowledge_binding": "48_of_64",
                "rule_reasoning": "24_of_32",
                "grammar_constraint": "48_of_64",
            },
            "model_family_gate": "9_of_12_discovery_groups",
            "crossmodel_gate": "all_three_models",
        },
        "direct_endpoint_transition_graph": semantic_transition_table(),
        "authorization": {
            "run_execution_qualification": True,
            "run_discovery_after_each_model_qualifies": True,
            "run_calibration_only_for_strict_crossmodel_semantic_candidates": True,
            "run_behavioral_holdout_only_after_calibration": True,
            "infer_direct_endpoint_operator_only_after_state_gate": True,
            "run_physical_mapping_only_after_all_functional_gates": True,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "finite_condition_panel_is_complete_language_state": False,
            "normalized_parser_is_internal_semantic_state": False,
            "greedy_sequence_is_full_stochastic_kernel": False,
            "direct_endpoint_transition_is_model_executed_operator": False,
            "history_pair_is_all_possible_generation_history": False,
        },
    }
    write_json(OUT / "phase407_event_horizon_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
