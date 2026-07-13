#!/usr/bin/env python3
"""Freeze Phase408 exclusive response partitions and interface coordinates.

The protocol separates four axes that Phase407 showed must not be collapsed:
runtime numerical validity, observed event times, semantic classification, and
the finite response mapping over states.  The interface coordinate maps below
are task definitions.  Agreement with them is functional covariance, not an
internal operator claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase386_multitime_protocol import NAMES  # noqa: E402
from phase404_direct_state_protocol import FROZEN_DTYPES, MODELS  # noqa: E402
from phase407_event_horizon_protocol import (  # noqa: E402
    FRESH_NOUNS,
    INTERMEDIATES,
    TERMINALS,
    TRIGGERS,
)


OUT = ROOT / "tests/gpt5/result/phase408_partition_interface"
SCHEMA_VERSION = "82.0.0"
FAMILIES = ("knowledge_binding", "rule_reasoning", "grammar_constraint")
SPLIT_GROUP_COUNTS = {
    "qualification": 2,
    "discovery": 12,
    "calibration": 6,
    "behavioral_holdout": 6,
    "physical_holdout": 6,
}
FORMAL_SPLITS = (
    "discovery",
    "calibration",
    "behavioral_holdout",
    "physical_holdout",
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
STATE_IDS = {
    "knowledge_binding": tuple("p" + "".join(map(str, item)) for item in PERMUTATIONS),
    "rule_reasoning": ("holder_0", "holder_1", "holder_2"),
    "grammar_constraint": (
        "singular_present",
        "plural_present",
        "singular_past",
        "plural_past",
    ),
}
INTERFACES = {
    "knowledge_binding": (
        "entity_value_order",
        "reverse_entity_value_order",
        "value_owner_order",
    ),
    "rule_reasoning": ("holder_ordinal", "holder_name", "reach_truth_vector"),
    "grammar_constraint": ("be_form", "feature_pair", "sentence_completion"),
}
STRUCTURAL_SURFACES = (
    {"surface_id": "r000", "syntax": 0, "fact_order": (0, 1, 2)},
    {"surface_id": "r001", "syntax": 1, "fact_order": (2, 0, 1)},
    {"surface_id": "r002", "syntax": 2, "fact_order": (1, 2, 0)},
    {"surface_id": "r003", "syntax": 3, "fact_order": (2, 1, 0)},
)
LEXICAL_REPLICAS = (0, 1)
MAX_NEW_TOKENS = 48
TOP_K = 8
QUALIFICATION_CASE_COUNT = 32

VALUE_TRIPLES = (
    ("amber", "indigo", "silver"),
    ("coral", "navy", "ivory"),
    ("copper", "teal", "violet"),
    ("crimson", "olive", "pearl"),
    ("gold", "cyan", "maroon"),
    ("scarlet", "azure", "bronze"),
    ("magenta", "emerald", "gray"),
    ("ochre", "turquoise", "plum"),
)
MODIFIERS = (
    "quiet",
    "bright",
    "plain",
    "small",
    "tall",
    "calm",
    "round",
    "new",
    "old",
    "clean",
    "solid",
    "light",
)
ORDINAL_WORDS = ("first", "second", "third")
AMBIGUOUS_ALIASES = (
    "cannot determine",
    "not enough information",
    "ambiguous",
    "either answer",
    "it depends",
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


def token_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def split_for(priority: int) -> str:
    cursor = 0
    for split, count in SPLIT_GROUP_COUNTS.items():
        if priority < cursor + count:
            return split
        cursor += count
    raise ValueError(priority)


def state_tuple(family: str, state_id: str) -> tuple[int, ...]:
    if family == "knowledge_binding":
        return tuple(int(item) for item in state_id[1:])
    if family == "rule_reasoning":
        return (int(state_id[-1]),)
    return {
        "singular_present": (0, 0),
        "plural_present": (1, 0),
        "singular_past": (0, 1),
        "plural_past": (1, 1),
    }[state_id]


def state_id_from_tuple(family: str, value: tuple[int, ...]) -> str:
    for state_id in STATE_IDS[family]:
        if state_tuple(family, state_id) == value:
            return state_id
    raise ValueError((family, value))


def inverse_permutation(value: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(value)
    for source, target in enumerate(value):
        result[target] = source
    return tuple(result)


def entity_triple(priority: int, lexical: int) -> tuple[str, str, str]:
    size = len(NAMES)
    first = (19 * priority + 11 * lexical + 7) % size
    offsets = (0, 17 + priority % 7, 41 + lexical * 5 + priority % 11)
    indices = tuple((first + offset) % size for offset in offsets)
    if len(set(indices)) != 3:
        indices = (first, (first + 23) % size, (first + 47) % size)
    if len(set(indices)) != 3:
        raise RuntimeError("Phase408 entity collision")
    return tuple(NAMES[index] for index in indices)


def package_for(family: str, priority: int, lexical: int) -> dict[str, Any]:
    index = 2 * priority + lexical
    entities = entity_triple(priority, lexical)
    if family == "knowledge_binding":
        values = VALUE_TRIPLES[(index * 3 + priority) % len(VALUE_TRIPLES)]
        item = FRESH_NOUNS[(index * 5 + priority) % len(FRESH_NOUNS)][0]
        return {"entities": entities, "values": values, "item": item}
    if family == "rule_reasoning":
        return {
            "entities": entities,
            "trigger": TRIGGERS[index % len(TRIGGERS)],
            "intermediate": INTERMEDIATES[(index * 5 + 1) % len(INTERMEDIATES)],
            "terminal": TERMINALS[(index * 7 + 2) % len(TERMINALS)],
        }
    singular, plural = FRESH_NOUNS[index % len(FRESH_NOUNS)]
    modifier = MODIFIERS[(index // len(FRESH_NOUNS)) % len(MODIFIERS)]
    return {"noun_singular": singular, "noun_plural": plural, "modifier": modifier}


def grammar_form(state_id: str) -> str:
    number, tense = state_tuple("grammar_constraint", state_id)
    return (("is", "are"), ("was", "were"))[tense][number]


def encode_raw_class(family: str, interface: str, state_id: str) -> str:
    truth = state_tuple(family, state_id)
    if family == "knowledge_binding":
        if interface == "entity_value_order":
            encoded = truth
            prefix = "value"
        elif interface == "reverse_entity_value_order":
            encoded = tuple(reversed(truth))
            prefix = "value"
        else:
            encoded = inverse_permutation(truth)
            prefix = "owner"
        return prefix + "_" + "".join(map(str, encoded))
    if family == "rule_reasoning":
        holder = truth[0]
        if interface == "holder_ordinal":
            return f"ordinal_{holder}"
        if interface == "holder_name":
            return f"name_slot_{holder}"
        vector = [0, 0, 0]
        vector[holder] = 1
        return "truth_" + "".join(map(str, vector))
    number, tense = truth
    if interface == "be_form":
        return "form_" + grammar_form(state_id)
    if interface == "feature_pair":
        return f"feature_{('singular', 'plural')[number]}_{('present', 'past')[tense]}"
    return "sentence_" + grammar_form(state_id)


def raw_class_to_state(family: str, interface: str) -> dict[str, str]:
    result = {
        encode_raw_class(family, interface, state_id): state_id
        for state_id in STATE_IDS[family]
    }
    if len(result) != len(STATE_IDS[family]):
        raise RuntimeError(f"Phase408 non-injective interface: {family}/{interface}")
    return result


def interface_coordinate_map(
    family: str, source_interface: str, target_interface: str
) -> dict[str, str]:
    return {
        encode_raw_class(family, source_interface, state_id): encode_raw_class(
            family, target_interface, state_id
        )
        for state_id in STATE_IDS[family]
    }


def facts_for(
    family: str,
    package: dict[str, Any],
    state_id: str,
    surface: dict[str, Any],
) -> str:
    syntax = int(surface["syntax"])
    order = tuple(int(item) for item in surface["fact_order"])
    if family == "knowledge_binding":
        truth = state_tuple(family, state_id)
        clauses = [
            (
                f"the {ORDINAL_WORDS[index]} person ({package['entities'][index]}) "
                f"has the {package['values'][truth[index]]} {package['item']}"
            )
            for index in range(3)
        ]
    elif family == "rule_reasoning":
        holder = state_tuple(family, state_id)[0]
        clauses = [
            (
                f"the {ORDINAL_WORDS[index]} person ({package['entities'][index]}) "
                + (
                    f"carries the {package['trigger']}"
                    if index == holder
                    else f"does not carry the {package['trigger']}"
                )
            )
            for index in range(3)
        ]
    else:
        number, tense = state_tuple(family, state_id)
        base_noun = package["noun_singular"] if number == 0 else package["noun_plural"]
        noun = f"{package['modifier']} {base_noun}"
        clauses = [
            f"the sentence subject is '{noun}'",
            f"its grammatical number is {('singular', 'plural')[number]}",
            f"its required tense is {('present', 'past')[tense]}",
        ]
    clauses = [clauses[index] for index in order]
    if syntax == 0:
        body = "Final record:\n- " + "\n- ".join(clauses) + "."
    elif syntax == 1:
        body = "In the final record, " + "; while ".join(clauses) + "."
    elif syntax == 2:
        body = "Use only these final facts: " + "; ".join(clauses) + "."
    else:
        body = "The closed record states that " + ", and ".join(clauses) + "."
    if family == "rule_reasoning":
        body += (
            f" Whoever carries the {package['trigger']} receives the "
            f"{package['intermediate']}; whoever receives the "
            f"{package['intermediate']} reaches the {package['terminal']}."
        )
    return body


def raw_class_aliases(
    family: str, package: dict[str, Any], interface: str
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for state_id in STATE_IDS[family]:
        raw_class = encode_raw_class(family, interface, state_id)
        truth = state_tuple(family, state_id)
        if family == "knowledge_binding":
            if interface == "entity_value_order":
                sequence = tuple(package["values"][item] for item in truth)
            elif interface == "reverse_entity_value_order":
                sequence = tuple(package["values"][item] for item in reversed(truth))
            else:
                sequence = tuple(
                    ORDINAL_WORDS[item] for item in inverse_permutation(truth)
                )
            result[raw_class] = [
                ", ".join(sequence),
                " / ".join(sequence),
                " then ".join(sequence),
            ]
        elif family == "rule_reasoning":
            holder = truth[0]
            if interface == "holder_ordinal":
                result[raw_class] = [
                    f"{ORDINAL_WORDS[holder]} person",
                    f"the {ORDINAL_WORDS[holder]} person",
                ]
            elif interface == "holder_name":
                result[raw_class] = [package["entities"][holder]]
            else:
                vector = tuple("yes" if index == holder else "no" for index in range(3))
                result[raw_class] = [
                    ", ".join(vector),
                    " / ".join(vector),
                    " then ".join(vector),
                ]
        else:
            number, tense = truth
            form = grammar_form(state_id)
            if interface == "be_form":
                result[raw_class] = [form]
            elif interface == "feature_pair":
                number_word = ("singular", "plural")[number]
                tense_word = ("present", "past")[tense]
                result[raw_class] = [
                    f"{number_word}, {tense_word}",
                    f"{number_word} {tense_word}",
                ]
            else:
                result[raw_class] = [
                    f"{form} {package['modifier']}",
                    form,
                ]
    return result


def interface_contract(
    family: str,
    package: dict[str, Any],
    state_id: str,
    interface: str,
) -> dict[str, str]:
    aliases = raw_class_aliases(family, package, interface)
    target_class = encode_raw_class(family, interface, state_id)
    target_alias = aliases[target_class][0]
    states = STATE_IDS[family]
    foil_state = states[(states.index(state_id) + 1) % len(states)]
    foil_class = encode_raw_class(family, interface, foil_state)
    foil_alias = aliases[foil_class][0]
    if family == "knowledge_binding":
        if interface == "entity_value_order":
            query = (
                "Give exactly three value words for the first, second, and third "
                "people in that order. Use commas and then end the sentence."
            )
        elif interface == "reverse_entity_value_order":
            query = (
                "Give exactly three value words for the third, second, and first "
                "people in that order. Use commas and then end the sentence."
            )
        else:
            values = ", ".join(package["values"])
            query = (
                f"For the values {values} in that order, give exactly the owner "
                "ordinals first, second, or third. Use commas and end the sentence."
            )
        prefix = "Response:"
    elif family == "rule_reasoning":
        if interface == "holder_ordinal":
            query = (
                f"Which person reaches the {package['terminal']}? Answer exactly "
                "first person, second person, or third person, then end."
            )
            prefix = "Holder:"
        elif interface == "holder_name":
            query = (
                f"Which named person reaches the {package['terminal']}? Give exactly "
                "one recorded name, then end."
            )
            prefix = "Name:"
        else:
            query = (
                f"For the first, second, and third people, state whether each reaches "
                f"the {package['terminal']}. Give exactly yes/no for all three in order."
            )
            prefix = "Reach vector:"
    else:
        number, _tense = state_tuple(family, state_id)
        base_noun = package["noun_singular"] if number == 0 else package["noun_plural"]
        noun = f"{package['modifier']} {base_noun}"
        if interface == "be_form":
            query = (
                "Give exactly the one be-form required by the recorded number and "
                "tense, then end the sentence."
            )
            prefix = "Required form:"
        elif interface == "feature_pair":
            query = (
                "Give exactly the grammatical number and tense as two words in that "
                "order, then end the sentence."
            )
            prefix = "Features:"
        else:
            query = (
                "Complete the prepared sentence with the required be-form and the "
                "given final adjective, then end it."
            )
            prefix = f"The {noun}"
    return {
        "query": query,
        "assistant_prefix": prefix,
        "target_completion": " " + target_alias + ".",
        "foil_completion": " " + foil_alias + ".",
    }


def render_chat(
    tokenizer: Any,
    model: str,
    messages: list[dict[str, str]],
    assistant_prefix: str,
) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
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
    package: dict[str, Any],
    state_id: str,
    surface: dict[str, Any],
    interface: str,
) -> tuple[str, dict[str, str], str]:
    facts = facts_for(family, package, state_id, surface)
    contract = interface_contract(family, package, state_id, interface)
    messages = [{"role": "user", "content": facts + "\n\nTask: " + contract["query"]}]
    return (
        render_chat(tokenizer, model, messages, contract["assistant_prefix"]),
        contract,
        facts,
    )


def previous_prompt_hashes() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    base = ROOT / "tests/gpt5/result"
    paths = (
        base / "phase403_predictive_state/protocol/private/phase403_all_cases.jsonl",
        base / "phase404_direct_predictive_state/protocol/private/phase404_all_cases.jsonl",
        base / "phase405_natural_future_state/protocol/private/phase405_all_cases.jsonl",
        base / "phase406_conditioned_sequence_state/protocol/private/phase406_all_cases.jsonl",
        base / "phase407_event_horizon_kernel/protocol/private/phase407_all_cases.jsonl",
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


def validate_coordinate_maps() -> dict[str, Any]:
    checked = 0
    for family in FAMILIES:
        interfaces = INTERFACES[family]
        for source in interfaces:
            identity = interface_coordinate_map(family, source, source)
            if any(key != value for key, value in identity.items()):
                raise RuntimeError(f"Phase408 identity failure: {family}/{source}")
            for middle in interfaces:
                first = interface_coordinate_map(family, source, middle)
                for target in interfaces:
                    direct = interface_coordinate_map(family, source, target)
                    second = interface_coordinate_map(family, middle, target)
                    composed = {key: second[value] for key, value in first.items()}
                    if direct != composed:
                        raise RuntimeError(
                            f"Phase408 coordinate composition failure: "
                            f"{family}/{source}/{middle}/{target}"
                        )
                    checked += 1
    return {"valid": True, "composition_checks": checked}


def validate_case_registry(row: dict[str, Any]) -> None:
    aliases = row["raw_response_aliases_private"]
    reverse: dict[tuple[str, ...], str] = {}
    for raw_class, values in aliases.items():
        if not values:
            raise RuntimeError(f"Phase408 empty aliases: {row['blind_case_id']}")
        for value in values:
            words = token_words(value)
            if not words:
                raise RuntimeError(f"Phase408 empty alias tokens: {value!r}")
            prior = reverse.get(words)
            if prior is not None and prior != raw_class:
                raise RuntimeError(
                    f"Phase408 alias collision: {prior}/{raw_class}/{value}"
                )
            reverse[words] = raw_class
    class_to_state = row["raw_class_to_semantic_state_private"]
    if set(aliases) != set(class_to_state):
        raise RuntimeError(f"Phase408 registry domain mismatch: {row['blind_case_id']}")
    if set(class_to_state.values()) != set(STATE_IDS[row["family_id"]]):
        raise RuntimeError(f"Phase408 registry codomain mismatch: {row['blind_case_id']}")
    target_class = row["target_raw_response_class_private"]
    if class_to_state[target_class] != row["target_semantic_state_private"]:
        raise RuntimeError(f"Phase408 target decode mismatch: {row['blind_case_id']}")
    rejected = set(row["explicit_rejected_raw_classes_private"])
    if rejected != set(aliases) - {target_class}:
        raise RuntimeError(f"Phase408 rejected class mismatch: {row['blind_case_id']}")


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
    group_registry: list[dict[str, Any]] = []
    total_groups = sum(SPLIT_GROUP_COUNTS.values())

    for family in FAMILIES:
        for priority in range(total_groups):
            split = split_for(priority)
            group_id = "p408g_" + digest(f"{family}:{priority}", 24)
            packages = [package_for(family, priority, lexical) for lexical in LEXICAL_REPLICAS]
            fingerprint = digest(
                json.dumps([family, priority, packages], sort_keys=True, ensure_ascii=True),
                32,
            )
            group_registry.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase408-PartitionInterfaceProtocol",
                    "family_id": family,
                    "anonymous_parallel_group_id": group_id,
                    "candidate_split": split,
                    "group_priority": priority,
                    "state_count": len(STATE_IDS[family]),
                    "structural_surface_count": len(STRUCTURAL_SURFACES),
                    "lexical_replica_count": len(LEXICAL_REPLICAS),
                    "interface_count": len(INTERFACES[family]),
                    "history_mode": "fixed_empty",
                    "semantic_group_fingerprint": fingerprint,
                }
            )
            for state_id in STATE_IDS[family]:
                for lexical in LEXICAL_REPLICAS:
                    package = package_for(family, priority, lexical)
                    for surface in STRUCTURAL_SURFACES:
                        for interface in INTERFACES[family]:
                            aliases = raw_class_aliases(family, package, interface)
                            class_to_state = raw_class_to_state(family, interface)
                            target_class = encode_raw_class(family, interface, state_id)
                            for model, tokenizer in tokenizers.items():
                                prompt, contract, facts = make_prompt(
                                    tokenizer,
                                    model,
                                    family,
                                    package,
                                    state_id,
                                    surface,
                                    interface,
                                )
                                prompt_ids = tokenizer(prompt, add_special_tokens=False)[
                                    "input_ids"
                                ]
                                target_ids = tokenizer(
                                    contract["target_completion"], add_special_tokens=False
                                )["input_ids"]
                                foil_ids = tokenizer(
                                    contract["foil_completion"], add_special_tokens=False
                                )["input_ids"]
                                case_key = (
                                    f"{model}:{family}:{group_id}:{state_id}:lex{lexical}:"
                                    f"{surface['surface_id']}:{interface}"
                                )
                                prompt_hash = (model, digest(prompt))
                                if prompt_hash in previous_hashes:
                                    raise RuntimeError(
                                        f"Phase408 overlaps Phase403-407: {case_key}"
                                    )
                                if prompt_hash in current_hashes:
                                    raise RuntimeError(
                                        f"Phase408 duplicate: {current_hashes[prompt_hash]} == {case_key}"
                                    )
                                current_hashes[prompt_hash] = case_key
                                candidate_first_ids = {
                                    int(
                                        tokenizer(
                                            " " + values[0] + ".",
                                            add_special_tokens=False,
                                        )["input_ids"][0]
                                    )
                                    for values in aliases.values()
                                }
                                row = {
                                    "schema_version": SCHEMA_VERSION,
                                    "phase_id": "Phase408-PartitionInterfaceProtocol",
                                    "created_at": created_at,
                                    "private_execution_model": model,
                                    "blind_case_id": "p408c_" + digest(case_key, 28),
                                    "family_id": family,
                                    "anonymous_parallel_group_id": group_id,
                                    "parallel_group_id_private": f"p408_private_{family}_{priority:02d}",
                                    "candidate_split_private": split,
                                    "group_priority": priority,
                                    "state_id_private": state_id,
                                    "abstract_state_private": list(state_tuple(family, state_id)),
                                    "lexical_replica_private": lexical,
                                    "surface_id_private": surface["surface_id"],
                                    "surface_axes_private": surface,
                                    "interface_private": interface,
                                    "history_mode_private": "fixed_empty",
                                    "condition_id_private": (
                                        f"lex{lexical}__{surface['surface_id']}__{interface}"
                                    ),
                                    "prompt": prompt,
                                    "prompt_token_ids_private": [int(item) for item in prompt_ids],
                                    "prompt_token_count": len(prompt_ids),
                                    "state_facts_private": facts,
                                    "assistant_prefix_private": contract["assistant_prefix"],
                                    "target_semantic_state_private": state_id,
                                    "target_raw_response_class_private": target_class,
                                    "raw_response_aliases_private": aliases,
                                    "raw_class_to_semantic_state_private": class_to_state,
                                    "semantic_state_ids_private": list(STATE_IDS[family]),
                                    "explicit_rejected_raw_classes_private": sorted(
                                        set(aliases) - {target_class}
                                    ),
                                    "ambiguous_aliases_private": list(AMBIGUOUS_ALIASES),
                                    "target_completion_private": contract["target_completion"],
                                    "target_completion_token_ids_private": [
                                        int(item) for item in target_ids
                                    ],
                                    "foil_completion_private": contract["foil_completion"],
                                    "foil_completion_token_ids_private": [int(item) for item in foil_ids],
                                    "registered_candidate_first_token_ids_private": sorted(
                                        candidate_first_ids
                                    ),
                                    "max_new_tokens": MAX_NEW_TOKENS,
                                    "top_k_ledger_size": TOP_K,
                                    "execution_qualification_case": False,
                                    "formal_denominator": split in FORMAL_SPLITS,
                                    "axis_contract": {
                                        "runtime_numeric_status": "orthogonal",
                                        "event_observation_status": "orthogonal",
                                        "semantic_class": (
                                            "allowed_rejected_ambiguous_unparsed"
                                        ),
                                        "response_mapping_class": "separate",
                                    },
                                }
                                validate_case_registry(row)
                                rows.append(row)

    qualification_targets = {
        "knowledge_binding": {
            "entity_value_order": 4,
            "reverse_entity_value_order": 4,
            "value_owner_order": 4,
        },
        "rule_reasoning": {
            "holder_ordinal": 3,
            "holder_name": 3,
            "reach_truth_vector": 2,
        },
        "grammar_constraint": {
            "be_form": 4,
            "feature_pair": 4,
            "sentence_completion": 4,
        },
    }
    for model in MODELS:
        for family, interface_counts in qualification_targets.items():
            for interface, count in interface_counts.items():
                candidates = [
                    row
                    for row in rows
                    if row["private_execution_model"] == model
                    and row["family_id"] == family
                    and row["interface_private"] == interface
                    and row["candidate_split_private"] == "qualification"
                ]
                candidates.sort(key=lambda row: digest(row["blind_case_id"] + ":qual"))
                if len(candidates) < count:
                    raise RuntimeError(f"Phase408 qualification shortage: {model}/{interface}")
                for row in candidates[:count]:
                    row["execution_qualification_case"] = True

    coordinate_audit = validate_coordinate_maps()
    expected_rows = sum(
        total_groups
        * len(STATE_IDS[family])
        * len(LEXICAL_REPLICAS)
        * len(STRUCTURAL_SURFACES)
        * len(INTERFACES[family])
        * len(MODELS)
        for family in FAMILIES
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Phase408 row count {len(rows)} != {expected_rows}")
    for model in MODELS:
        count = sum(
            row["execution_qualification_case"]
            and row["private_execution_model"] == model
            for row in rows
        )
        if count != QUALIFICATION_CASE_COUNT:
            raise RuntimeError(f"Phase408 qualification count {model}: {count}")

    maps = {
        family: {
            f"{source}__to__{target}": interface_coordinate_map(
                family, source, target
            )
            for source in INTERFACES[family]
            for target in INTERFACES[family]
        }
        for family in FAMILIES
    }
    query_registry = [
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase408-ExclusiveQueryRegistry",
            "family_id": family,
            "interface_id": interface,
            "state_domain": list(STATE_IDS[family]),
            "raw_class_to_semantic_state": raw_class_to_state(family, interface),
            "semantic_axis": ["allowed", "rejected", "ambiguous", "unparsed"],
            "runtime_numeric_axis": ["valid", "invalid"],
            "event_axes": ["semantic_observed", "boundary_observed", "stop_observed"],
            "axes_are_not_a_single_six_class_partition": True,
        }
        for family in FAMILIES
        for interface in INTERFACES[family]
    ]
    write_jsonl(OUT / "protocol/private/phase408_all_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase408_blind_group_registry.jsonl", group_registry)
    write_jsonl(OUT / "phase408_query_contract_registry.jsonl", query_registry)
    registry_payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase408-RegistryQualification",
        "created_at": created_at,
        "valid": True,
        "case_registry_count": len(rows),
        "abstract_query_contract_count": len(query_registry),
        "alias_collision_count": 0,
        "coordinate_map_audit": coordinate_audit,
        "semantic_runtime_event_axes_separated": True,
    }
    write_json(OUT / "phase408_registry_qualification.json", registry_payload)

    discovery_cases_per_model = sum(
        SPLIT_GROUP_COUNTS["discovery"]
        * len(STATE_IDS[family])
        * len(LEXICAL_REPLICAS)
        * len(STRUCTURAL_SURFACES)
        * len(INTERFACES[family])
        for family in FAMILIES
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase408-PartitionInterfaceProtocol",
        "created_at": created_at,
        "objective": "exclusive_registry_response_partition_and_interface_covariance_before_physical_mapping",
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "device": "cuda",
            "default_runtime_dtype_by_model": FROZEN_DTYPES,
            "glm_dtype_selection_rule": (
                "prefer_exact_repeat_and_all_finite_bfloat16_over_float16; "
                "otherwise require an all-finite exact option; stop if neither qualifies"
            ),
            "attention_implementation": "eager",
            "generation": "stepwise_deterministic_greedy_not_global_sequence_map",
            "batch_size": 1,
            "padding": "none",
            "max_new_tokens": MAX_NEW_TOKENS,
            "termination": "model_eos_or_H48_only",
        },
        "denominator": {
            "families": list(FAMILIES),
            "registered_groups_per_family": total_groups,
            "split_group_counts": SPLIT_GROUP_COUNTS,
            "states_per_family": {
                family: len(STATE_IDS[family]) for family in FAMILIES
            },
            "structural_surfaces": len(STRUCTURAL_SURFACES),
            "lexical_replicas": len(LEXICAL_REPLICAS),
            "interfaces_per_family": 3,
            "history_mode": "fixed_empty",
            "case_count_all_models_all_registered_splits": len(rows),
            "discovery_case_count_per_model": discovery_cases_per_model,
            "discovery_case_count_all_models": discovery_cases_per_model * len(MODELS),
            "execution_qualification_case_count_per_model": QUALIFICATION_CASE_COUNT,
        },
        "state_domains": {family: list(states) for family, states in STATE_IDS.items()},
        "family_interfaces": {family: list(values) for family, values in INTERFACES.items()},
        "frozen_task_coordinate_maps": maps,
        "independent_gates": {
            "registry": "100_percent_machine_contract_checks_before_model_execution",
            "condition_separation": "all_states_map_to_distinct_registered_classes",
            "surface_lexical_stability": "same_state_to_raw_class_map_across_4x2_conditions",
            "interface_covariance": "all_three_pairwise_maps_stable_and_cycle_consistent",
            "model_family_gate": "9_of_12_discovery_groups",
            "crossmodel_gate": "all_three_models",
            "glm_pair_diagnostic": "at_least_two_models_including_glm4",
        },
        "authorization": {
            "run_execution_qualification": True,
            "run_discovery_after_each_model_qualifies": True,
            "run_calibration_only_for_strict_crossmodel_partition_candidates": True,
            "run_behavioral_holdout_only_after_calibration": True,
            "run_direct_state_operator_only_after_unseen_combination_holdout": True,
            "run_physical_mapping_only_after_all_functional_gates": True,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "greedy_trace_is_global_sequence_map": False,
            "functional_response_separation_is_internal_information_measure": False,
            "task_coordinate_map_is_model_internal_operator": False,
            "cycle_consistency_on_task_coordinates_is_interface_algebra_discovery": False,
            "crossmodel_behavior_partition_is_physical_language_invariant": False,
            "single_global_progress_percentage_valid": False,
        },
    }
    write_json(OUT / "phase408_partition_interface_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
