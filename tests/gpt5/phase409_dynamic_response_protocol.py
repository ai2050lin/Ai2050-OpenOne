#!/usr/bin/env python3
"""Freeze Phase409 dynamic-response and history-condition protocol objects.

This stage is deliberately model-free.  It defines finite semantic worlds,
checks them with two separately implemented solvers, freezes model-rendered
prompt hashes, and provides a prefix-event parser.  Passing these machine
checks is not an independent semantic review and does not authorize model or
physical execution.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase404_direct_state_protocol import FROZEN_DTYPES  # noqa: E402
from phase408_partition_interface_protocol import (  # noqa: E402
    AMBIGUOUS_ALIASES,
    FORMAL_SPLITS,
    LEXICAL_REPLICAS,
    MODELS,
    ORDINAL_WORDS,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    STRUCTURAL_SURFACES,
    facts_for,
    grammar_form,
    inverse_permutation,
    package_for,
    render_chat,
    split_for,
    state_tuple,
    token_words,
)


OUT = ROOT / "tests/gpt5/result/phase409_dynamic_response_protocol"
SCHEMA_VERSION = "83.0.0"
PHASE_ID = "Phase409-DynamicResponseHistoryProtocol"
FAMILIES = ("knowledge_binding", "rule_reasoning", "grammar_constraint")
INTERFACES = {
    "knowledge_binding": (
        "entity_value_order",
        "value_owner_order",
        "single_entity_value",
    ),
    "rule_reasoning": ("holder_ordinal", "holder_name", "reach_truth_vector"),
    "grammar_constraint": ("be_form", "feature_pair", "sentence_completion"),
}
HISTORY_MODES = (
    "h0_current_only",
    "h1_prior_equivalent_state",
    "h2_prior_irrelevant_content",
    "h3_prior_conflicting_state",
    "h4_prior_conflict_then_current_explicit_override",
)
GATE_ELIGIBLE_HISTORY_MODES = (
    "h0_current_only",
    "h1_prior_equivalent_state",
    "h2_prior_irrelevant_content",
    "h4_prior_conflict_then_current_explicit_override",
)
CONFLICT_DIAGNOSTIC_HISTORY_MODE = "h3_prior_conflicting_state"
MAX_NEW_TOKENS = 48
QUERY_ROLE_DEFAULT = "default"
TOTAL_GROUPS_PER_FAMILY = sum(SPLIT_GROUP_COUNTS.values())


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


def query_roles(family: str, interface: str) -> tuple[str, ...]:
    if family == "knowledge_binding" and interface == "single_entity_value":
        return ("entity_0", "entity_1", "entity_2")
    return (QUERY_ROLE_DEFAULT,)


def query_role_index(query_role: str) -> int:
    if not query_role.startswith("entity_"):
        raise ValueError(query_role)
    value = int(query_role[-1])
    if value not in (0, 1, 2):
        raise ValueError(query_role)
    return value


def prior_state_for(family: str, state_id: str) -> str:
    states = STATE_IDS[family]
    return states[(states.index(state_id) + 1) % len(states)]


def direct_solver(
    family: str,
    current_state: str,
    history_mode: str,
    prior_state: str,
) -> tuple[str, ...]:
    """Closed-form semantic-state solver, independent of text rendering."""
    if current_state not in STATE_IDS[family] or prior_state not in STATE_IDS[family]:
        raise ValueError((family, current_state, prior_state))
    if history_mode == CONFLICT_DIAGNOSTIC_HISTORY_MODE:
        candidates = {current_state, prior_state}
        return tuple(state for state in STATE_IDS[family] if state in candidates)
    if history_mode in GATE_ELIGIBLE_HISTORY_MODES:
        return (current_state,)
    raise ValueError(history_mode)


def finite_world_formula(
    current_state: str,
    history_mode: str,
    prior_state: str,
) -> dict[str, Any]:
    """Build a small declarative formula consumed only by the enumerator."""
    if history_mode == "h0_current_only":
        return {"op": "eq", "state": current_state}
    if history_mode == "h1_prior_equivalent_state":
        return {
            "op": "all",
            "items": [
                {"op": "eq", "state": current_state},
                {"op": "eq", "state": prior_state},
            ],
        }
    if history_mode == "h2_prior_irrelevant_content":
        return {
            "op": "all",
            "items": [
                {"op": "eq", "state": current_state},
                {"op": "irrelevant", "value": True},
            ],
        }
    if history_mode == CONFLICT_DIAGNOSTIC_HISTORY_MODE:
        return {
            "op": "any",
            "items": [
                {"op": "eq", "state": prior_state},
                {"op": "eq", "state": current_state},
            ],
        }
    if history_mode == "h4_prior_conflict_then_current_explicit_override":
        return {
            "op": "all",
            "items": [
                {"op": "overridden", "state": prior_state},
                {"op": "eq", "state": current_state},
            ],
        }
    raise ValueError(history_mode)


def _world_satisfies(world: str, formula: dict[str, Any]) -> bool:
    op = formula["op"]
    if op == "eq":
        return world == formula["state"]
    if op in {"irrelevant", "overridden"}:
        return True
    if op == "all":
        return all(_world_satisfies(world, item) for item in formula["items"])
    if op == "any":
        return any(_world_satisfies(world, item) for item in formula["items"])
    raise ValueError(op)


def enumerative_solver(
    family: str,
    current_state: str,
    history_mode: str,
    prior_state: str,
) -> tuple[str, ...]:
    """Exhaustively filter every finite semantic world through a formula."""
    formula = finite_world_formula(current_state, history_mode, prior_state)
    return tuple(
        state for state in STATE_IDS[family] if _world_satisfies(state, formula)
    )


def encode_raw_class(
    family: str,
    interface: str,
    state_id: str,
    query_role: str = QUERY_ROLE_DEFAULT,
) -> str:
    truth = state_tuple(family, state_id)
    if family == "knowledge_binding":
        if interface == "entity_value_order":
            return "value_" + "".join(map(str, truth))
        if interface == "value_owner_order":
            return "owner_" + "".join(map(str, inverse_permutation(truth)))
        if interface == "single_entity_value":
            return f"single_value_{truth[query_role_index(query_role)]}"
    elif family == "rule_reasoning":
        holder = truth[0]
        if interface == "holder_ordinal":
            return f"ordinal_{holder}"
        if interface == "holder_name":
            return f"name_slot_{holder}"
        if interface == "reach_truth_vector":
            vector = [0, 0, 0]
            vector[holder] = 1
            return "truth_" + "".join(map(str, vector))
    elif family == "grammar_constraint":
        number, tense = truth
        form = grammar_form(state_id)
        if interface == "be_form":
            return "form_" + form
        if interface == "feature_pair":
            return f"feature_{('singular', 'plural')[number]}_{('present', 'past')[tense]}"
        if interface == "sentence_completion":
            return "sentence_" + form
    raise ValueError((family, interface, state_id, query_role))


def raw_response_aliases(
    family: str,
    package: dict[str, Any],
    interface: str,
    query_role: str = QUERY_ROLE_DEFAULT,
) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for state_id in STATE_IDS[family]:
        raw_class = encode_raw_class(family, interface, state_id, query_role)
        truth = state_tuple(family, state_id)
        if family == "knowledge_binding":
            if interface == "entity_value_order":
                sequence = tuple(package["values"][item] for item in truth)
                values = [", ".join(sequence), " / ".join(sequence)]
            elif interface == "value_owner_order":
                sequence = tuple(
                    ORDINAL_WORDS[item] for item in inverse_permutation(truth)
                )
                values = [", ".join(sequence), " / ".join(sequence)]
            else:
                values = [package["values"][truth[query_role_index(query_role)]]]
        elif family == "rule_reasoning":
            holder = truth[0]
            if interface == "holder_ordinal":
                values = [
                    f"{ORDINAL_WORDS[holder]} person",
                    f"the {ORDINAL_WORDS[holder]} person",
                ]
            elif interface == "holder_name":
                values = [package["entities"][holder]]
            else:
                vector = tuple("yes" if index == holder else "no" for index in range(3))
                values = [", ".join(vector), " / ".join(vector)]
        else:
            number, tense = truth
            form = grammar_form(state_id)
            if interface == "be_form":
                values = [form]
            elif interface == "feature_pair":
                values = [
                    f"{('singular', 'plural')[number]}, {('present', 'past')[tense]}",
                    f"{('singular', 'plural')[number]} {('present', 'past')[tense]}",
                ]
            else:
                # A bare be-form is intentionally absent.  The adjective is part
                # of the registered response, while punctuation is a later event.
                values = [f"{form} {package['modifier']}"]
        prior = aliases.get(raw_class)
        if prior is not None and set(prior) != set(values):
            raise RuntimeError(f"Phase409 inconsistent aliases for {raw_class}")
        aliases[raw_class] = values
    return aliases


def raw_class_to_states(
    family: str,
    interface: str,
    query_role: str = QUERY_ROLE_DEFAULT,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for state_id in STATE_IDS[family]:
        result[encode_raw_class(family, interface, state_id, query_role)].append(
            state_id
        )
    return {key: sorted(value) for key, value in sorted(result.items())}


def response_contract(
    family: str,
    package: dict[str, Any],
    interface: str,
    query_role: str = QUERY_ROLE_DEFAULT,
) -> dict[str, Any]:
    aliases = raw_response_aliases(family, package, interface, query_role)
    class_to_states = raw_class_to_states(family, interface, query_role)
    reverse: dict[tuple[str, ...], str] = {}
    for raw_class, values in aliases.items():
        for alias in values:
            words = token_words(alias)
            prior = reverse.get(words)
            if prior is not None and prior != raw_class:
                raise RuntimeError(
                    f"Phase409 alias collision: {family}/{interface}/{query_role}/"
                    f"{prior}/{raw_class}/{alias}"
                )
            reverse[words] = raw_class
    return {
        "family_id": family,
        "interface_id": interface,
        "query_role": query_role,
        "raw_response_aliases": aliases,
        "raw_class_to_states": class_to_states,
        "single_query_injective": all(len(value) == 1 for value in class_to_states.values()),
        "ambiguous_aliases": list(AMBIGUOUS_ALIASES),
    }


def interface_state_signatures(family: str, interface: str) -> dict[str, tuple[str, ...]]:
    roles = query_roles(family, interface)
    return {
        state_id: tuple(
            encode_raw_class(family, interface, state_id, role) for role in roles
        )
        for state_id in STATE_IDS[family]
    }


def joint_query_signature_is_injective(family: str, interface: str) -> bool:
    signatures = interface_state_signatures(family, interface)
    return len(set(signatures.values())) == len(signatures)


def interface_contract_text(
    family: str,
    package: dict[str, Any],
    interface: str,
    query_role: str = QUERY_ROLE_DEFAULT,
) -> tuple[str, str]:
    if family == "knowledge_binding":
        if interface == "entity_value_order":
            return (
                "Give exactly three value words for the first, second, and third "
                "people in that order, separated by commas, then end the sentence.",
                "Values:",
            )
        if interface == "value_owner_order":
            return (
                f"For the values {', '.join(package['values'])} in that order, give "
                "exactly the owner ordinals first, second, or third, separated by "
                "commas, then end the sentence.",
                "Owners:",
            )
        index = query_role_index(query_role)
        return (
            f"Give exactly the one value word owned by the {ORDINAL_WORDS[index]} "
            "person, then end the sentence.",
            "Value:",
        )
    if family == "rule_reasoning":
        if interface == "holder_ordinal":
            return (
                f"Which person reaches the {package['terminal']}? Answer exactly "
                "first person, second person, or third person, then end.",
                "Holder:",
            )
        if interface == "holder_name":
            return (
                f"Which named person reaches the {package['terminal']}? Give exactly "
                "one recorded name, then end.",
                "Name:",
            )
        return (
            f"For the first, second, and third people, state whether each reaches "
            f"the {package['terminal']}. Give exactly yes/no for all three in order.",
            "Reach vector:",
        )
    if interface == "be_form":
        return (
            "Give exactly one be-form required by the recorded number and tense, "
            "then end the sentence.",
            "Required form:",
        )
    if interface == "feature_pair":
        return (
            "Give exactly the grammatical number and tense as two words in that "
            "order, then end the sentence.",
            "Features:",
        )
    # The subject is already present in the facts.  The answer contract is a
    # full two-word predicate fragment and not a bare be-form.
    return (
        "Complete the prepared sentence with exactly the required be-form followed "
        f"by the adjective {package['modifier']}, then end the sentence.",
        "Completion:",
    )


def history_messages(
    family: str,
    package: dict[str, Any],
    current_state: str,
    prior_state: str,
    irrelevant_package: dict[str, Any],
    surface: dict[str, Any],
    interface: str,
    query_role: str,
    history_mode: str,
) -> tuple[list[dict[str, str]], str]:
    current_facts = facts_for(family, package, current_state, surface)
    prior_facts = facts_for(family, package, prior_state, surface)
    irrelevant_state = STATE_IDS[family][0]
    irrelevant_facts = facts_for(
        family, irrelevant_package, irrelevant_state, surface
    )
    query, prefix = interface_contract_text(
        family, package, interface, query_role
    )
    task = (
        "Read Protocol 409 records literally and follow this response contract.\n"
        f"Task: {query}"
    )
    if history_mode == "h0_current_only":
        messages = [
            {"role": "user", "content": f"Current record:\n{current_facts}\n\n{task}"}
        ]
    elif history_mode == "h1_prior_equivalent_state":
        messages = [
            {"role": "user", "content": f"Earlier equivalent record:\n{current_facts}"},
            {"role": "assistant", "content": "Recorded."},
            {
                "role": "user",
                "content": f"Current record confirms it:\n{current_facts}\n\n{task}",
            },
        ]
    elif history_mode == "h2_prior_irrelevant_content":
        messages = [
            {"role": "user", "content": f"Earlier unrelated record:\n{irrelevant_facts}"},
            {"role": "assistant", "content": "Recorded."},
            {
                "role": "user",
                "content": f"Current queried record:\n{current_facts}\n\n{task}",
            },
        ]
    elif history_mode == CONFLICT_DIAGNOSTIC_HISTORY_MODE:
        messages = [
            {"role": "user", "content": f"Record A:\n{prior_facts}"},
            {"role": "assistant", "content": "Recorded."},
            {
                "role": "user",
                "content": (
                    "Record B conflicts with Record A, and no priority is specified:\n"
                    f"{current_facts}\n\n{task}"
                ),
            },
        ]
    elif history_mode == "h4_prior_conflict_then_current_explicit_override":
        messages = [
            {"role": "user", "content": f"Earlier record:\n{prior_facts}"},
            {"role": "assistant", "content": "Recorded."},
            {
                "role": "user",
                "content": (
                    "The following current final record explicitly supersedes the "
                    "earlier record:\n"
                    f"{current_facts}\n\n{task}"
                ),
            },
        ]
    else:
        raise ValueError(history_mode)
    return messages, prefix


def contains_words(text_words: tuple[str, ...], alias_words: tuple[str, ...]) -> bool:
    width = len(alias_words)
    if width == 0 or width > len(text_words):
        return False
    return any(
        text_words[index : index + width] == alias_words
        for index in range(len(text_words) - width + 1)
    )


def parse_response_prefix(
    text: str,
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
) -> dict[str, Any]:
    words = token_words(text)
    ambiguous_hits = [
        alias
        for alias in contract["ambiguous_aliases"]
        if contains_words(words, token_words(alias))
    ]
    matches: dict[str, list[str]] = {}
    for raw_class, aliases in contract["raw_response_aliases"].items():
        hits = [
            alias for alias in aliases if contains_words(words, token_words(alias))
        ]
        if hits:
            matches[raw_class] = hits
    matched_classes = sorted(matches)
    if ambiguous_hits or len(matched_classes) > 1:
        return {
            "automaton_state": "ambiguous_response",
            "response_role": "ambiguous",
            "raw_response_class": None,
            "matched_raw_classes": matched_classes,
        }
    if not matched_classes:
        if contract["interface_id"] == "sentence_completion" and any(
            contains_words(words, (form,)) for form in ("is", "are", "was", "were")
        ):
            state = "format_incomplete"
        else:
            state = "no_registered_response"
        return {
            "automaton_state": state,
            "response_role": "none",
            "raw_response_class": None,
            "matched_raw_classes": [],
        }
    raw_class = matched_classes[0]
    decoded_states = set(contract["raw_class_to_states"][raw_class])
    expected_states = set(
        direct_solver(
            contract["family_id"], current_state, history_mode, prior_state
        )
    )
    if current_state in decoded_states and prior_state in decoded_states:
        role = "current_and_prior"
    elif current_state in decoded_states:
        role = "current"
    elif prior_state in decoded_states:
        role = "prior"
    else:
        role = "other_registered"
    return {
        "automaton_state": (
            "allowed_response"
            if decoded_states & expected_states
            else "rejected_response"
        ),
        "response_role": role,
        "raw_response_class": raw_class,
        "decoded_state_set": sorted(decoded_states),
        "matched_raw_classes": matched_classes,
    }


def scan_event_process(
    decoded_prefixes: list[str],
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
    stopped: bool,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    previous_state = "no_registered_response"
    first_registered = None
    first_allowed = None
    allowed_exit = None
    boundary = None
    allowed_seen = False
    final_parse: dict[str, Any] = {
        "automaton_state": "no_registered_response",
        "response_role": "none",
        "raw_response_class": None,
    }
    for step, prefix in enumerate(decoded_prefixes, start=1):
        parsed = parse_response_prefix(
            prefix, contract, current_state, prior_state, history_mode
        )
        final_parse = parsed
        state = parsed["automaton_state"]
        if first_registered is None and parsed["raw_response_class"] is not None:
            first_registered = step
        if first_allowed is None and state == "allowed_response":
            first_allowed = step
            allowed_seen = True
        if allowed_seen and allowed_exit is None and state != "allowed_response":
            allowed_exit = step
        if state != previous_state:
            events.append(
                {
                    "event": state,
                    "step": step,
                    "response_role": parsed["response_role"],
                    "raw_response_class": parsed["raw_response_class"],
                }
            )
            previous_state = state
        if boundary is None and re.search(r"[.!?](?:\s|$)", prefix):
            boundary = step
            events.append({"event": "boundary_reached", "step": step})
    if stopped:
        events.append(
            {"event": "model_stopped", "step": len(decoded_prefixes) + 1}
        )
    return {
        "event_transition_sequence": events,
        "first_registered_event": first_registered,
        "first_allowed_event": first_allowed,
        "allowed_exit_event": allowed_exit,
        "boundary_event": boundary,
        "stop_event": len(decoded_prefixes) + 1 if stopped else None,
        "right_censored_at_H48": not stopped,
        "final_parse": final_parse,
    }


def previous_prompt_hashes() -> dict[str, set[str]]:
    result = {model: set() for model in MODELS}
    base = ROOT / "tests/gpt5/result"
    paths = (
        base / "phase403_predictive_state/protocol/private/phase403_all_cases.jsonl",
        base / "phase404_direct_predictive_state/protocol/private/phase404_all_cases.jsonl",
        base / "phase405_natural_future_state/protocol/private/phase405_all_cases.jsonl",
        base / "phase406_conditioned_sequence_state/protocol/private/phase406_all_cases.jsonl",
        base / "phase407_event_horizon_kernel/protocol/private/phase407_all_cases.jsonl",
        base / "phase408_partition_interface/protocol/private/phase408_all_cases.jsonl",
    )
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = row.get("private_execution_model") or row.get("model")
                prompt = row.get("prompt")
                if model in result and prompt:
                    result[model].add(digest(prompt))
    return result


def query_contract_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        sample_package = package_for(family, 0, 0)
        for interface in INTERFACES[family]:
            joint_injective = joint_query_signature_is_injective(family, interface)
            if not joint_injective:
                raise RuntimeError(f"Phase409 non-identifiable joint contract: {family}/{interface}")
            for role in query_roles(family, interface):
                contract = response_contract(family, sample_package, interface, role)
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "family_id": family,
                        "interface_id": interface,
                        "query_role": role,
                        "state_count": len(STATE_IDS[family]),
                        "response_class_count": len(contract["raw_class_to_states"]),
                        "single_query_injective": contract["single_query_injective"],
                        "joint_query_role_signature_injective": joint_injective,
                        "gate_unit": (
                            "joint_query_role_signature"
                            if len(query_roles(family, interface)) > 1
                            else "single_query"
                        ),
                    }
                )
    return rows


def main() -> None:
    created_at = now()
    previous_hashes = previous_prompt_hashes()
    current_hashes = {model: set() for model in MODELS}
    prompt_overlap_count = 0
    prompt_duplicate_count = 0
    disagreement_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    package_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        for priority in range(TOTAL_GROUPS_PER_FAMILY):
            split = split_for(priority)
            group_id = "p409g_" + digest(f"{family}:{priority}", 24)
            group_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "family_id": family,
                    "anonymous_parallel_group_id": group_id,
                    "candidate_split": split,
                    "group_priority": priority,
                    "state_count": len(STATE_IDS[family]),
                    "surface_count": len(STRUCTURAL_SURFACES),
                    "lexical_replica_count": len(LEXICAL_REPLICAS),
                    "history_modes": list(HISTORY_MODES),
                    "interfaces": list(INTERFACES[family]),
                }
            )
            packages: dict[int, dict[str, Any]] = {}
            irrelevant_packages: dict[int, dict[str, Any]] = {}
            for lexical in LEXICAL_REPLICAS:
                package = package_for(family, priority, lexical)
                irrelevant_package = package_for(
                    family,
                    priority + TOTAL_GROUPS_PER_FAMILY + 101,
                    1 - lexical,
                )
                packages[lexical] = package
                irrelevant_packages[lexical] = irrelevant_package
                package_id = "p409pkg_" + digest(
                    f"{family}:{priority}:{lexical}", 24
                )
                package_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "package_id": package_id,
                        "family_id": family,
                        "anonymous_parallel_group_id": group_id,
                        "group_priority": priority,
                        "lexical_replica": lexical,
                        "package_private": package,
                        "irrelevant_package_private": irrelevant_package,
                    }
                )

            for current_state in STATE_IDS[family]:
                conflict_state = prior_state_for(family, current_state)
                for lexical in LEXICAL_REPLICAS:
                    package = packages[lexical]
                    irrelevant_package = irrelevant_packages[lexical]
                    package_id = "p409pkg_" + digest(
                        f"{family}:{priority}:{lexical}", 24
                    )
                    for surface in STRUCTURAL_SURFACES:
                        for history_mode in HISTORY_MODES:
                            prior_state = (
                                current_state
                                if history_mode == "h1_prior_equivalent_state"
                                else conflict_state
                            )
                            direct = direct_solver(
                                family, current_state, history_mode, prior_state
                            )
                            enumerated = enumerative_solver(
                                family, current_state, history_mode, prior_state
                            )
                            if direct != enumerated:
                                disagreement_rows.append(
                                    {
                                        "family_id": family,
                                        "current_state": current_state,
                                        "prior_state": prior_state,
                                        "history_mode": history_mode,
                                        "direct": list(direct),
                                        "enumerated": list(enumerated),
                                    }
                                )
                            for interface in INTERFACES[family]:
                                for query_role in query_roles(family, interface):
                                    contract = response_contract(
                                        family, package, interface, query_role
                                    )
                                    messages, prefix = history_messages(
                                        family,
                                        package,
                                        current_state,
                                        prior_state,
                                        irrelevant_package,
                                        surface,
                                        interface,
                                        query_role,
                                        history_mode,
                                    )
                                    case_key = (
                                        f"{family}:{priority}:{current_state}:lex{lexical}:"
                                        f"{surface['surface_id']}:{history_mode}:{interface}:"
                                        f"{query_role}"
                                    )
                                    target_classes = sorted(
                                        {
                                            encode_raw_class(
                                                family,
                                                interface,
                                                state,
                                                query_role,
                                            )
                                            for state in direct
                                        }
                                    )
                                    abstract_rows.append(
                                        {
                                            "schema_version": SCHEMA_VERSION,
                                            "phase_id": PHASE_ID,
                                            "blind_case_id": "p409c_"
                                            + digest(case_key, 28),
                                            "family_id": family,
                                            "anonymous_parallel_group_id": group_id,
                                            "candidate_split_private": split,
                                            "group_priority": priority,
                                            "package_id_private": package_id,
                                            "state_id_private": current_state,
                                            "prior_state_id_private": prior_state,
                                            "lexical_replica_private": lexical,
                                            "surface_id_private": surface["surface_id"],
                                            "interface_private": interface,
                                            "query_role_private": query_role,
                                            "history_mode_private": history_mode,
                                            "gate_eligible_history": history_mode
                                            in GATE_ELIGIBLE_HISTORY_MODES,
                                            "conflict_diagnostic_only": history_mode
                                            == CONFLICT_DIAGNOSTIC_HISTORY_MODE,
                                            "direct_solver_state_set_private": list(direct),
                                            "enumerative_solver_state_set_private": list(
                                                enumerated
                                            ),
                                            "registered_response_classes_private": sorted(
                                                contract["raw_response_aliases"]
                                            ),
                                            "admissible_response_classes_private": target_classes,
                                            "response_contract_digest_private": digest(
                                                json.dumps(
                                                    contract,
                                                    sort_keys=True,
                                                    ensure_ascii=True,
                                                )
                                            ),
                                            "message_plan_digest_private": digest(
                                                json.dumps(
                                                    messages,
                                                    sort_keys=True,
                                                    ensure_ascii=True,
                                                )
                                            ),
                                            "assistant_prefix_private": prefix,
                                            "prompt_hashes_by_model_private": {},
                                            "formal_denominator": split in FORMAL_SPLITS,
                                            "execution_qualification_case": False,
                                        }
                                    )

    if disagreement_rows:
        raise RuntimeError(
            f"Phase409 solver disagreement count: {len(disagreement_rows)}"
        )

    # Custom tokenizers are isolated one at a time.  Loading all three in one
    # process has caused native exits on this host, even without model weights.
    surface_by_id = {
        surface["surface_id"]: surface for surface in STRUCTURAL_SURFACES
    }
    for model in MODELS:
        print(f"Phase409 prompt hash audit: {model}", file=sys.stderr, flush=True)
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
        for row in abstract_rows:
            family = row["family_id"]
            priority = int(row["group_priority"])
            lexical = int(row["lexical_replica_private"])
            package = package_for(family, priority, lexical)
            irrelevant_package = package_for(
                family,
                priority + TOTAL_GROUPS_PER_FAMILY + 101,
                1 - lexical,
            )
            messages, prefix = history_messages(
                family,
                package,
                row["state_id_private"],
                row["prior_state_id_private"],
                irrelevant_package,
                surface_by_id[row["surface_id_private"]],
                row["interface_private"],
                row["query_role_private"],
                row["history_mode_private"],
            )
            prompt = render_chat(tokenizer, model, messages, prefix)
            prompt_hash = digest(prompt)
            if prompt_hash in previous_hashes[model]:
                prompt_overlap_count += 1
            if prompt_hash in current_hashes[model]:
                prompt_duplicate_count += 1
            current_hashes[model].add(prompt_hash)
            row["prompt_hashes_by_model_private"][model] = prompt_hash
        del tokenizer
        gc.collect()

    if prompt_overlap_count or prompt_duplicate_count:
        raise RuntimeError(
            "Phase409 prompt audit failed: "
            f"previous_overlap={prompt_overlap_count}, duplicates={prompt_duplicate_count}"
        )

    # One deterministic case for each query-contract x history-mode cell gives
    # 11 x 5 = 55 sealed qualification cases per future model.
    by_qualification_cell: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in abstract_rows:
        if row["candidate_split_private"] != "qualification":
            continue
        key = (
            row["family_id"],
            row["interface_private"],
            row["query_role_private"],
            row["history_mode_private"],
        )
        by_qualification_cell[key].append(row)
    for key, candidates in by_qualification_cell.items():
        candidates.sort(key=lambda row: digest(row["blind_case_id"] + ":qual"))
        candidates[0]["execution_qualification_case"] = True
    qualification_case_count = sum(
        bool(row["execution_qualification_case"]) for row in abstract_rows
    )

    expected_abstract_case_count = sum(
        TOTAL_GROUPS_PER_FAMILY
        * len(STATE_IDS[family])
        * len(LEXICAL_REPLICAS)
        * len(STRUCTURAL_SURFACES)
        * len(HISTORY_MODES)
        * sum(
            len(query_roles(family, interface))
            for interface in INTERFACES[family]
        )
        for family in FAMILIES
    )
    if len(abstract_rows) != expected_abstract_case_count:
        raise RuntimeError(
            f"Phase409 case count {len(abstract_rows)} != {expected_abstract_case_count}"
        )
    if qualification_case_count != 55:
        raise RuntimeError(
            f"Phase409 qualification count {qualification_case_count} != 55"
        )

    contract_rows = query_contract_registry()
    scenario_rows = []
    for family in FAMILIES:
        for current_state in STATE_IDS[family]:
            for history_mode in HISTORY_MODES:
                prior_state = (
                    current_state
                    if history_mode == "h1_prior_equivalent_state"
                    else prior_state_for(family, current_state)
                )
                direct = direct_solver(
                    family, current_state, history_mode, prior_state
                )
                enumerated = enumerative_solver(
                    family, current_state, history_mode, prior_state
                )
                scenario_rows.append(
                    {
                        "family_id": family,
                        "current_state": current_state,
                        "prior_state": prior_state,
                        "history_mode": history_mode,
                        "direct_state_set": list(direct),
                        "enumerative_state_set": list(enumerated),
                        "agreement": direct == enumerated,
                        "unique_state_required": history_mode
                        in GATE_ELIGIBLE_HISTORY_MODES,
                    }
                )

    split_case_counts = Counter(
        row["candidate_split_private"] for row in abstract_rows
    )
    history_case_counts = Counter(
        row["history_mode_private"] for row in abstract_rows
    )
    prompt_hash_audit = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase409-PromptHashAudit",
        "created_at": created_at,
        "valid": True,
        "previous_phase_prompt_overlap_count": 0,
        "within_phase_model_prompt_duplicate_count": 0,
        "model_rendered_prompt_count": len(abstract_rows) * len(MODELS),
        "unique_prompt_count_by_model": {
            model: len(values) for model, values in current_hashes.items()
        },
        "aggregate_digest_by_model": {
            model: digest("\n".join(sorted(values)))
            for model, values in current_hashes.items()
        },
    }
    rule_engine_agreement = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase409-DualRuleEngineAgreement",
        "created_at": created_at,
        "valid": all(row["agreement"] for row in scenario_rows),
        "independent_human_rule_review_completed": False,
        "scenario_count": len(scenario_rows),
        "expanded_abstract_case_count": len(abstract_rows),
        "disagreement_count": 0,
        "unique_state_scenario_count": sum(
            row["unique_state_required"] for row in scenario_rows
        ),
        "conflict_diagnostic_scenario_count": sum(
            not row["unique_state_required"] for row in scenario_rows
        ),
        "scenarios": scenario_rows,
    }
    protocol_qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase409-ProtocolQualification",
        "created_at": created_at,
        "machine_protocol_gate_pass": True,
        "query_contract_count": len(contract_rows),
        "joint_query_signature_failure_count": 0,
        "alias_collision_count": 0,
        "grammar_sentence_bare_be_alias_count": 0,
        "dual_rule_engine_disagreement_count": 0,
        "prompt_overlap_count": 0,
        "prompt_duplicate_count": 0,
        "independent_human_rule_review_completed": False,
        "incremental_collector_token_equivalence_completed": False,
        "model_execution_authorized": False,
        "physical_execution_authorized": False,
        "neuron_scan_authorized": False,
    }
    discovery_abstract_case_count = split_case_counts["discovery"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": created_at,
        "objective": "dynamic_registered_response_process_and_history_condition_protocol",
        "models_in_future_execution_order": list(MODELS),
        "execution_contract": {
            "device": "cuda",
            "dtype_by_model": FROZEN_DTYPES,
            "generation": "stepwise_deterministic_greedy_not_global_sequence_map",
            "max_new_tokens": MAX_NEW_TOKENS,
            "model_execution_in_this_stage": False,
        },
        "denominator": {
            "families": list(FAMILIES),
            "groups_per_family": TOTAL_GROUPS_PER_FAMILY,
            "split_group_counts": SPLIT_GROUP_COUNTS,
            "states_per_family": {
                family: len(STATE_IDS[family]) for family in FAMILIES
            },
            "structural_surfaces": len(STRUCTURAL_SURFACES),
            "lexical_replicas": len(LEXICAL_REPLICAS),
            "history_modes": list(HISTORY_MODES),
            "abstract_case_count_all_registered_splits": len(abstract_rows),
            "future_model_rendered_case_count_all_models": len(abstract_rows)
            * len(MODELS),
            "discovery_abstract_case_count": discovery_abstract_case_count,
            "future_discovery_case_count_all_models": discovery_abstract_case_count
            * len(MODELS),
            "gate_eligible_abstract_case_count": sum(
                row["gate_eligible_history"] for row in abstract_rows
            ),
            "conflict_diagnostic_abstract_case_count": history_case_counts[
                CONFLICT_DIAGNOSTIC_HISTORY_MODE
            ],
            "sealed_qualification_case_count_per_future_model": qualification_case_count,
        },
        "interface_gate_contract": {
            "knowledge_single_entity_individual_query_is_injective": False,
            "knowledge_single_entity_three_role_joint_signature_is_injective": True,
            "rule_truth_vector_is_high_pressure_independent_interface": True,
            "grammar_sentence_completion_requires_be_form_adjective_and_boundary": True,
        },
        "event_process_fields": [
            "final_semantic_class",
            "first_registered_event",
            "first_allowed_event",
            "allowed_exit_event",
            "boundary_event",
            "stop_event",
            "event_transition_sequence",
        ],
        "hierarchical_gates": [
            "machine_protocol_and_external_rule_review",
            "future_execution_qualification",
            "interface_internal_joint_signature",
            "surface_and_lexical_replication",
            "interface_pair_transform",
            "history_h0_h1_h2_h4_replication",
            "crossmodel_calibration_and_behavioral_holdout",
            "separate_physical_protocol_registration",
        ],
        "authorization": {
            "protocol_and_dual_solver_development": True,
            "model_qualification": False,
            "formal_discovery": False,
            "calibration": False,
            "behavioral_holdout": False,
            "physical_mapping": False,
            "causal_intervention": False,
            "neuron_scan": False,
        },
        "remaining_blockers": [
            "independent_human_rule_review_not_completed",
            "incremental_collector_old_new_token_equivalence_not_completed",
            "no_model_execution_qualification_under_schema_83",
        ],
        "claim_boundary": {
            "event_automaton_is_model_internal_state_machine": False,
            "machine_dual_solver_agreement_is_independent_human_review": False,
            "behavioral_interface_transform_is_internal_operator": False,
            "single_global_progress_percentage_valid": False,
        },
    }

    write_jsonl(OUT / "protocol/phase409_group_registry.jsonl", group_rows)
    write_jsonl(OUT / "protocol/private/phase409_package_registry.jsonl", package_rows)
    write_jsonl(OUT / "protocol/private/phase409_abstract_case_registry.jsonl", abstract_rows)
    write_jsonl(OUT / "phase409_query_contract_registry.jsonl", contract_rows)
    write_json(OUT / "phase409_prompt_hash_audit.json", prompt_hash_audit)
    write_json(OUT / "phase409_rule_engine_agreement.json", rule_engine_agreement)
    write_json(OUT / "phase409_protocol_qualification.json", protocol_qualification)
    write_json(OUT / "phase409_dynamic_response_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
