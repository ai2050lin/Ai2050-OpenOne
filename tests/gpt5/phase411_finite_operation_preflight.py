#!/usr/bin/env python3
"""Freeze Phase411 finite semantic and operation contracts without model execution.

Phase410 repaired the response ledger, but exact whole-response matching is too
strict to stand in for open-language semantics and the machine answer key must
not be treated as an independent human review.  This stage therefore adds a
second, explicitly finite semantic-template channel, a complete finite state
operation table for the three registered language families, and a reviewer
workflow that compares two people before consulting the machine registry.

All outputs in this module are protocol objects.  They do not establish model
behavior, physical trajectories, causal mechanisms, or neuron-level closure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_protocol import (  # noqa: E402
    LEXICAL_REPLICAS,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    package_for,
    state_id_from_tuple,
    state_tuple,
    token_words,
)
from phase409_dynamic_response_protocol import (  # noqa: E402
    FAMILIES,
    HISTORY_MODES,
    INTERFACES,
    MODELS,
    OUT as PHASE409_OUT,
    direct_solver,
    query_roles,
    response_contract,
)
from phase410_orthogonal_preflight import (  # noqa: E402
    OUT as PHASE410_OUT,
    exact_response_parse,
)


OUT = ROOT / "tests/gpt5/result/phase411_finite_operation_preflight"
SCHEMA_VERSION = "85.0.0"
PHASE_ID = "Phase411-FiniteSemanticOperationPreflight"
REVIEW_ATTESTATION = (
    "I reviewed every item independently, without seeing the machine registry "
    "or another reviewer's response."
)
CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 5
TOTAL_GROUPS_PER_FAMILY = sum(SPLIT_GROUP_COUNTS.values())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def semantic_contract_index(contract: dict[str, Any]) -> dict[str, Any]:
    raw_by_words: dict[tuple[str, ...], str] = {}
    for raw_class, aliases in contract["raw_response_aliases"].items():
        for alias in aliases:
            words = token_words(alias)
            prior = raw_by_words.get(words)
            if prior is not None and prior != raw_class:
                raise RuntimeError((prior, raw_class, alias))
            raw_by_words[words] = raw_class
    ambiguous_words = {token_words(alias) for alias in contract["ambiguous_aliases"]}
    return {
        "raw_by_words": raw_by_words,
        "ambiguous_words": ambiguous_words,
    }


def raw_resolution(
    raw_class: str,
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
) -> dict[str, Any]:
    decoded_states = set(contract["raw_class_to_states"][raw_class])
    expected_states = set(
        direct_solver(contract["family_id"], current_state, history_mode, prior_state)
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
        "semantic_class": (
            "allowed_response"
            if decoded_states & expected_states
            else "rejected_response"
        ),
        "response_role": role,
        "raw_response_class": raw_class,
        "decoded_state_set": sorted(decoded_states),
    }


def _split_once(
    words: tuple[str, ...], marker: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    width = len(marker)
    hits = [
        index
        for index in range(len(words) - width + 1)
        if words[index : index + width] == marker
    ]
    if len(hits) != 1:
        return None
    index = hits[0]
    return words[:index], words[index + width :]


def registered_semantic_parse(
    text: str,
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
    *,
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse only a frozen finite template language, not open-language semantics."""
    lookup = index or semantic_contract_index(contract)
    raw_by_words: dict[tuple[str, ...], str] = lookup["raw_by_words"]
    ambiguous_words: set[tuple[str, ...]] = lookup["ambiguous_words"]
    words = token_words(text)

    def resolved(raw_class: str, rule: str, *, revision: bool = False) -> dict[str, Any]:
        return {
            "channel_status": "resolved_registered_response",
            "resolution_rule": rule,
            "revision_detected": revision,
            "negation_detected": False,
            "candidate_raw_classes": [raw_class],
            **raw_resolution(
                raw_class, contract, current_state, prior_state, history_mode
            ),
        }

    if words in raw_by_words:
        return resolved(raw_by_words[words], "exact_registered_alias")
    if words in ambiguous_words:
        return {
            "channel_status": "explicit_ambiguity_marker",
            "resolution_rule": "exact_ambiguity_alias",
            "revision_detected": False,
            "negation_detected": False,
            "semantic_class": "ambiguous_response",
            "response_role": "ambiguous",
            "raw_response_class": None,
            "decoded_state_set": [],
            "candidate_raw_classes": [],
        }

    unknown_prefix = ("not", "only")
    if words[: len(unknown_prefix)] == unknown_prefix:
        candidate = raw_by_words.get(words[len(unknown_prefix) :])
        if candidate is not None:
            return {
                "channel_status": "scope_unresolved",
                "resolution_rule": None,
                "revision_detected": False,
                "negation_detected": True,
                "semantic_class": None,
                "response_role": "unresolved",
                "raw_response_class": None,
                "decoded_state_set": [],
                "candidate_raw_classes": [candidate],
            }

    revision_prefix = ("i", "first", "answered")
    revision_marker = ("but", "my", "final", "answer", "is")
    if words[: len(revision_prefix)] == revision_prefix:
        split = _split_once(words[len(revision_prefix) :], revision_marker)
        if split is not None:
            first_words, final_words = split
            first_raw = raw_by_words.get(first_words)
            final_raw = raw_by_words.get(final_words)
            if first_raw is not None and final_raw is not None:
                result = resolved(final_raw, "explicit_final_revision", revision=True)
                result["candidate_raw_classes"] = [first_raw, final_raw]
                return result

    if words[:1] == ("either",):
        split = _split_once(words[1:], ("or",))
        if split is not None:
            left_words, right_words = split
            left_raw = raw_by_words.get(left_words)
            right_raw = raw_by_words.get(right_words)
            if left_raw is not None and right_raw is not None:
                decoded = set(contract["raw_class_to_states"][left_raw])
                decoded.update(contract["raw_class_to_states"][right_raw])
                return {
                    "channel_status": "multiple_registered_candidates",
                    "resolution_rule": "explicit_either_or",
                    "revision_detected": False,
                    "negation_detected": False,
                    "semantic_class": "ambiguous_response",
                    "response_role": "ambiguous",
                    "raw_response_class": None,
                    "decoded_state_set": sorted(decoded),
                    "candidate_raw_classes": [left_raw, right_raw],
                }

    negated_raw = None
    if words[:1] == ("not",):
        negated_raw = raw_by_words.get(words[1:])
    elif words[:4] == ("the", "answer", "is", "not"):
        negated_raw = raw_by_words.get(words[4:])
    elif words[-4:] == ("is", "not", "my", "answer"):
        negated_raw = raw_by_words.get(words[:-4])
    if negated_raw is not None:
        return {
            "channel_status": "explicitly_negated_candidate",
            "resolution_rule": None,
            "revision_detected": False,
            "negation_detected": True,
            "semantic_class": None,
            "response_role": "negated",
            "raw_response_class": None,
            "decoded_state_set": [],
            "candidate_raw_classes": [negated_raw],
        }

    if words[:1] == ("maybe",):
        hedged_raw = raw_by_words.get(words[1:])
        if hedged_raw is not None:
            return {
                "channel_status": "hedged_candidate",
                "resolution_rule": None,
                "revision_detected": False,
                "negation_detected": False,
                "semantic_class": None,
                "response_role": "hedged",
                "raw_response_class": None,
                "decoded_state_set": [],
                "candidate_raw_classes": [hedged_raw],
            }

    wrappers = (
        ("the", "answer", "is"),
        ("my", "answer", "is"),
        ("my", "final", "answer", "is"),
        ("final", "answer"),
        ("i", "choose"),
    )
    for prefix in wrappers:
        if words[: len(prefix)] != prefix:
            continue
        remainder = words[len(prefix) :]
        raw_class = raw_by_words.get(remainder)
        if raw_class is not None:
            return resolved(raw_class, "registered_answer_wrapper")
        if remainder in ambiguous_words:
            return {
                "channel_status": "explicit_ambiguity_marker",
                "resolution_rule": "registered_ambiguity_wrapper",
                "revision_detected": False,
                "negation_detected": False,
                "semantic_class": "ambiguous_response",
                "response_role": "ambiguous",
                "raw_response_class": None,
                "decoded_state_set": [],
                "candidate_raw_classes": [],
            }

    return {
        "channel_status": "unregistered_response",
        "resolution_rule": None,
        "revision_detected": False,
        "negation_detected": False,
        "semantic_class": "no_registered_response",
        "response_role": "none",
        "raw_response_class": None,
        "decoded_state_set": [],
        "candidate_raw_classes": [],
    }


def finite_semantic_cases(contract: dict[str, Any]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    def add(
        text: str,
        status: str,
        *,
        raw_class: str | None = None,
        strict_raw_class: str | None = None,
        strict_semantic_class: str = "no_registered_response",
        case_kind: str,
    ) -> None:
        expected = {
            "channel_status": status,
            "raw_response_class": raw_class,
            "strict_raw_response_class": strict_raw_class,
            "strict_semantic_class": strict_semantic_class,
        }
        prior = cases.get(text)
        if prior is not None and any(
            prior[key] != value for key, value in expected.items()
        ):
            raise RuntimeError(f"Phase411 semantic case collision: {text!r}")
        expected["case_kinds"] = sorted(
            set((prior or {}).get("case_kinds", [])) | {case_kind}
        )
        cases[text] = expected

    first_alias_by_raw: dict[str, str] = {}
    for raw_class, aliases in sorted(contract["raw_response_aliases"].items()):
        first_alias_by_raw[raw_class] = aliases[0]
        for alias in aliases:
            add(
                alias,
                "resolved_registered_response",
                raw_class=raw_class,
                strict_raw_class=raw_class,
                strict_semantic_class="resolved",
                case_kind="exact_alias",
            )
            add(
                alias + ".",
                "resolved_registered_response",
                raw_class=raw_class,
                strict_raw_class=raw_class,
                strict_semantic_class="resolved",
                case_kind="punctuated_alias",
            )
            for wrapper in (
                f"the answer is {alias}",
                f"my final answer is {alias}",
                f"I choose {alias}",
            ):
                add(
                    wrapper,
                    "resolved_registered_response",
                    raw_class=raw_class,
                    case_kind="registered_wrapper",
                )
            add(
                f"not {alias}",
                "explicitly_negated_candidate",
                case_kind="prefix_negation",
            )
            add(
                f"{alias} is not my answer",
                "explicitly_negated_candidate",
                case_kind="suffix_negation",
            )
            add(
                f"maybe {alias}",
                "hedged_candidate",
                case_kind="hedged_candidate",
            )
            add(
                f"not only {alias}",
                "scope_unresolved",
                case_kind="negation_scope_control",
            )

    raw_classes = sorted(first_alias_by_raw)
    if len(raw_classes) > 1:
        for index, first_raw in enumerate(raw_classes):
            second_raw = raw_classes[(index + 1) % len(raw_classes)]
            first_alias = first_alias_by_raw[first_raw]
            second_alias = first_alias_by_raw[second_raw]
            add(
                f"either {first_alias} or {second_alias}",
                "multiple_registered_candidates",
                case_kind="explicit_multiple_candidates",
            )
            add(
                f"I first answered {first_alias}, but my final answer is {second_alias}",
                "resolved_registered_response",
                raw_class=second_raw,
                case_kind="explicit_revision",
            )

    for alias in contract["ambiguous_aliases"]:
        add(
            alias,
            "explicit_ambiguity_marker",
            strict_semantic_class="ambiguous_response",
            case_kind="exact_ambiguity_marker",
        )
        add(
            f"the answer is {alias}",
            "explicit_ambiguity_marker",
            case_kind="wrapped_ambiguity_marker",
        )
    add(
        "unregistered response",
        "unregistered_response",
        case_kind="unregistered_control",
    )
    add(
        "the answer is unregistered response",
        "unregistered_response",
        case_kind="wrapped_unregistered_control",
    )
    return [{"response_text": text, **row} for text, row in sorted(cases.items())]


def semantic_dual_channel_audit(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    agreement = read_json(PHASE409_OUT / "phase409_rule_engine_agreement.json")
    context_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    case_kind_counts: Counter[str] = Counter()
    global_hasher = hashlib.sha256()

    for scenario in agreement["scenarios"]:
        family = scenario["family_id"]
        current = scenario["current_state"]
        prior = scenario["prior_state"]
        history_mode = scenario["history_mode"]
        for priority in range(TOTAL_GROUPS_PER_FAMILY):
            for lexical in LEXICAL_REPLICAS:
                package = package_for(family, priority, lexical)
                for interface in INTERFACES[family]:
                    for query_role in query_roles(family, interface):
                        contract = response_contract(
                            family, package, interface, query_role
                        )
                        contract_index = semantic_contract_index(contract)
                        context_counter: Counter[str] = Counter()
                        context_hasher = hashlib.sha256()
                        context_failure_count = 0
                        cases = finite_semantic_cases(contract)
                        for case in cases:
                            observed = registered_semantic_parse(
                                case["response_text"],
                                contract,
                                current,
                                prior,
                                history_mode,
                                index=contract_index,
                            )
                            strict = exact_response_parse(
                                case["response_text"],
                                contract,
                                current,
                                prior,
                                history_mode,
                            )
                            semantic_class_valid = True
                            if case["raw_response_class"] is not None:
                                expected_resolution = raw_resolution(
                                    case["raw_response_class"],
                                    contract,
                                    current,
                                    prior,
                                    history_mode,
                                )
                                semantic_class_valid = bool(
                                    observed["semantic_class"]
                                    == expected_resolution["semantic_class"]
                                )
                            strict_valid = bool(
                                strict["raw_response_class"]
                                == case["strict_raw_response_class"]
                            )
                            if case["strict_semantic_class"] == "ambiguous_response":
                                strict_valid = strict["semantic_class"] == "ambiguous_response"
                            elif case["strict_semantic_class"] == "resolved":
                                strict_valid = bool(
                                    strict["raw_response_class"]
                                    == case["strict_raw_response_class"]
                                )
                            else:
                                strict_valid = bool(
                                    strict["semantic_class"]
                                    == "no_registered_response"
                                    and strict["raw_response_class"] is None
                                )
                            valid = bool(
                                observed["channel_status"] == case["channel_status"]
                                and observed["raw_response_class"]
                                == case["raw_response_class"]
                                and semantic_class_valid
                                and strict_valid
                            )
                            record = {
                                "family_id": family,
                                "current_state": current,
                                "prior_state": prior,
                                "history_mode": history_mode,
                                "priority": priority,
                                "lexical_replica": lexical,
                                "interface_id": interface,
                                "query_role": query_role,
                                "response_text": case["response_text"],
                                "case_kinds": case["case_kinds"],
                                "expected_status": case["channel_status"],
                                "observed_status": observed["channel_status"],
                                "expected_raw_class": case["raw_response_class"],
                                "observed_raw_class": observed["raw_response_class"],
                                "strict_semantic_class": strict["semantic_class"],
                                "valid": valid,
                            }
                            encoded = canonical_json(record)
                            context_hasher.update(encoded.encode("utf-8"))
                            global_hasher.update(encoded.encode("utf-8"))
                            total_counts["finite_case_count"] += 1
                            context_counter[observed["channel_status"]] += 1
                            total_counts[observed["channel_status"]] += 1
                            if strict["raw_response_class"] is not None:
                                context_counter["strict_resolved"] += 1
                                total_counts["strict_resolved"] += 1
                            if observed["raw_response_class"] is not None:
                                context_counter["semantic_resolved"] += 1
                                total_counts["semantic_resolved"] += 1
                                if strict["raw_response_class"] is None:
                                    context_counter["semantic_only_resolved"] += 1
                                    total_counts["semantic_only_resolved"] += 1
                            for kind in case["case_kinds"]:
                                case_kind_counts[kind] += 1
                            if not valid:
                                context_failure_count += 1
                                failures.append(record)
                        context_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": PHASE_ID,
                                "family_id": family,
                                "current_state": current,
                                "prior_state": prior,
                                "history_mode": history_mode,
                                "priority": priority,
                                "lexical_replica": lexical,
                                "interface_id": interface,
                                "query_role": query_role,
                                "case_count": len(cases),
                                "failure_count": context_failure_count,
                                "channel_status_counts": dict(
                                    sorted(context_counter.items())
                                ),
                                "context_digest": context_hasher.hexdigest(),
                            }
                        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase411-RegisteredSemanticDualChannelAudit",
        "created_at": created_at,
        "valid": not failures,
        "strict_channel": "whole_response_exact_registered_alias_only",
        "semantic_channel": "finite_registered_templates_with_anchored_scope_rules",
        "family_count": len(FAMILIES),
        "history_rule_scenario_count": len(agreement["scenarios"]),
        "package_count_per_family": TOTAL_GROUPS_PER_FAMILY
        * len(LEXICAL_REPLICAS),
        "contract_context_count": len(context_rows),
        "finite_response_case_count": total_counts["finite_case_count"],
        "failure_count": len(failures),
        "strict_resolved_case_count": total_counts["strict_resolved"],
        "registered_semantic_resolved_case_count": total_counts[
            "semantic_resolved"
        ],
        "semantic_only_resolved_case_count": total_counts[
            "semantic_only_resolved"
        ],
        "channel_status_counts": {
            key: value
            for key, value in sorted(total_counts.items())
            if key
            not in {
                "finite_case_count",
                "strict_resolved",
                "semantic_resolved",
                "semantic_only_resolved",
            }
        },
        "case_kind_counts": dict(sorted(case_kind_counts.items())),
        "finite_universe_digest": global_hasher.hexdigest(),
        "substring_matching_allowed": False,
        "negation_scope_is_forced_into_a_state": False,
        "hedged_response_is_forced_into_a_state": False,
        "explicit_final_revision_is_recorded": True,
        "open_language_semantics_tested": False,
        "external_semantic_validity_review_completed": False,
        "claim_boundary": (
            "exhaustive_finite_template_parser_contract_not_general_semantic_understanding"
        ),
    }
    return summary, context_rows, failures


def operation_registry() -> dict[str, list[dict[str, Any]]]:
    perms = tuple(permutations(range(3)))
    return {
        "knowledge_binding": [
            {
                "operation_id": "k_e" + "".join(map(str, entity_perm))
                + "_v"
                + "".join(map(str, value_perm)),
                "entity_permutation": entity_perm,
                "value_permutation": value_perm,
            }
            for entity_perm in perms
            for value_perm in perms
        ],
        "rule_reasoning": [
            {
                "operation_id": "r_h" + "".join(map(str, holder_perm)),
                "holder_permutation": holder_perm,
            }
            for holder_perm in perms
        ],
        "grammar_constraint": [
            {
                "operation_id": f"g_number{number_flip}_tense{tense_flip}",
                "number_flip": number_flip,
                "tense_flip": tense_flip,
            }
            for number_flip in (0, 1)
            for tense_flip in (0, 1)
        ],
    }


def apply_operation(family: str, operation: dict[str, Any], state_id: str) -> str:
    value = state_tuple(family, state_id)
    if family == "knowledge_binding":
        entity_perm = operation["entity_permutation"]
        value_perm = operation["value_permutation"]
        transformed = tuple(value_perm[value[entity_perm[index]]] for index in range(3))
    elif family == "rule_reasoning":
        transformed = (operation["holder_permutation"][value[0]],)
    else:
        transformed = (
            value[0] ^ operation["number_flip"],
            value[1] ^ operation["tense_flip"],
        )
    return state_id_from_tuple(family, transformed)


def operation_signature(family: str, operation: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        apply_operation(family, operation, state_id)
        for state_id in STATE_IDS[family]
    )


def observation_partitions(
    family: str,
) -> list[tuple[str, dict[str, tuple[str, ...]]]]:
    package = package_for(family, 0, 0)
    observers: list[tuple[str, dict[str, tuple[str, ...]]]] = []
    joint: dict[str, list[str]] = {state_id: [] for state_id in STATE_IDS[family]}
    for interface in INTERFACES[family]:
        for role in query_roles(family, interface):
            contract = response_contract(family, package, interface, role)
            signature = {
                state_id: tuple(
                    raw_class
                    for raw_class, states in contract["raw_class_to_states"].items()
                    if state_id in states
                )
                for state_id in STATE_IDS[family]
            }
            observer_id = f"{interface}:{role}"
            observers.append((observer_id, signature))
            for state_id in STATE_IDS[family]:
                joint[state_id].extend(signature[state_id])
    observers.append(
        ("joint_all_registered_queries", {key: tuple(value) for key, value in joint.items()})
    )
    return observers


def finite_operation_audit(
    created_at: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    registry = operation_registry()
    agreement = read_json(PHASE409_OUT / "phase409_rule_engine_agreement.json")
    operation_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    inverse_count = 0
    identity_count = 0

    for family in FAMILIES:
        operations = registry[family]
        signatures = {
            operation["operation_id"]: operation_signature(family, operation)
            for operation in operations
        }
        if len(set(signatures.values())) != len(operations):
            failures.append({"family_id": family, "failure": "duplicate_operation_map"})
        operation_by_id = {row["operation_id"]: row for row in operations}
        id_by_signature = {value: key for key, value in signatures.items()}
        identity_signature = tuple(STATE_IDS[family])
        identity_id = id_by_signature.get(identity_signature)
        if identity_id is None:
            failures.append({"family_id": family, "failure": "identity_missing"})
        else:
            identity_count += 1

        inverse_by_id: dict[str, str] = {}
        for first in operations:
            first_id = first["operation_id"]
            for second in operations:
                second_id = second["operation_id"]
                composite_signature = tuple(
                    apply_operation(
                        family,
                        second,
                        apply_operation(family, first, state_id),
                    )
                    for state_id in STATE_IDS[family]
                )
                composite_id = id_by_signature.get(composite_signature)
                valid = composite_id is not None
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "family_id": family,
                    "first_operation_id": first_id,
                    "second_operation_id": second_id,
                    "composite_operation_id": composite_id,
                    "valid": valid,
                }
                composition_rows.append(row)
                if not valid:
                    failures.append(row)
                if composite_id == identity_id:
                    reverse_signature = tuple(
                        apply_operation(
                            family,
                            first,
                            apply_operation(family, second, state_id),
                        )
                        for state_id in STATE_IDS[family]
                    )
                    if reverse_signature == identity_signature:
                        inverse_by_id[first_id] = second_id

        for operation in operations:
            operation_id = operation["operation_id"]
            inverse_id = inverse_by_id.get(operation_id)
            if inverse_id is None:
                failures.append(
                    {
                        "family_id": family,
                        "operation_id": operation_id,
                        "failure": "two_sided_inverse_missing",
                    }
                )
            else:
                inverse_count += 1
            operation_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "family_id": family,
                    **operation,
                    "inverse_operation_id": inverse_id,
                    "is_identity": operation_id == identity_id,
                    "state_map": {
                        state_id: apply_operation(family, operation, state_id)
                        for state_id in STATE_IDS[family]
                    },
                }
            )
            for state_id in STATE_IDS[family]:
                transition_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "family_id": family,
                        "operation_id": operation_id,
                        "source_state": state_id,
                        "target_state": apply_operation(
                            family, operation, state_id
                        ),
                    }
                )

        family_scenarios = [
            scenario
            for scenario in agreement["scenarios"]
            if scenario["family_id"] == family
        ]
        for scenario in family_scenarios:
            source_result = direct_solver(
                family,
                scenario["current_state"],
                scenario["history_mode"],
                scenario["prior_state"],
            )
            for operation in operations:
                transformed_current = apply_operation(
                    family, operation, scenario["current_state"]
                )
                transformed_prior = apply_operation(
                    family, operation, scenario["prior_state"]
                )
                expected = sorted(
                    apply_operation(family, operation, state_id)
                    for state_id in source_result
                )
                observed = sorted(
                    direct_solver(
                        family,
                        transformed_current,
                        scenario["history_mode"],
                        transformed_prior,
                    )
                )
                valid = observed == expected
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "family_id": family,
                    "history_mode": scenario["history_mode"],
                    "operation_id": operation["operation_id"],
                    "source_current_state": scenario["current_state"],
                    "source_prior_state": scenario["prior_state"],
                    "expected_transformed_state_set": expected,
                    "observed_transformed_state_set": observed,
                    "valid": valid,
                }
                covariance_rows.append(row)
                if not valid:
                    failures.append(row)

        for observer_id, signatures_by_state in observation_partitions(family):
            grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for state_id, signature in signatures_by_state.items():
                grouped[signature].append(state_id)
            nontrivial_classes = [states for states in grouped.values() if len(states) > 1]
            for operation in operations:
                violations = 0
                pair_count = 0
                for states in nontrivial_classes:
                    for state_a, state_b in combinations(states, 2):
                        pair_count += 1
                        target_a = apply_operation(family, operation, state_a)
                        target_b = apply_operation(family, operation, state_b)
                        if signatures_by_state[target_a] != signatures_by_state[target_b]:
                            violations += 1
                partition_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "family_id": family,
                        "observer_id": observer_id,
                        "operation_id": operation["operation_id"],
                        "equivalent_pair_count": pair_count,
                        "stability_violation_count": violations,
                        "operation_stable": violations == 0,
                        "joint_observer": observer_id
                        == "joint_all_registered_queries",
                    }
                )

    joint_rows = [row for row in partition_rows if row["joint_observer"]]
    coarse_rows = [row for row in partition_rows if not row["joint_observer"]]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase411-FiniteOperationClosureAudit",
        "created_at": created_at,
        "valid": not failures,
        "family_count": len(FAMILIES),
        "operation_count": len(operation_rows),
        "operation_count_by_family": dict(
            sorted(Counter(row["family_id"] for row in operation_rows).items())
        ),
        "identity_operation_count": identity_count,
        "two_sided_inverse_count": inverse_count,
        "state_transition_count": len(transition_rows),
        "composition_case_count": len(composition_rows),
        "composition_failure_count": sum(not row["valid"] for row in composition_rows),
        "history_rule_covariance_case_count": len(covariance_rows),
        "history_rule_covariance_failure_count": sum(
            not row["valid"] for row in covariance_rows
        ),
        "observer_operation_cell_count": len(partition_rows),
        "coarse_observer_unstable_operation_cell_count": sum(
            not row["operation_stable"] for row in coarse_rows
        ),
        "joint_observer_unstable_operation_cell_count": sum(
            not row["operation_stable"] for row in joint_rows
        ),
        "registered_operation_closure_pass": not failures,
        "registered_joint_observation_transition_consistency_pass": all(
            row["operation_stable"] for row in joint_rows
        ),
        "model_operation_executed": False,
        "model_state_partition_observed": False,
        "model_functional_bisimulation_established": False,
        "claim_boundary": (
            "finite_external_world_operation_algebra_not_model_internal_operator_or_bisimulation"
        ),
    }
    return (
        summary,
        operation_rows,
        transition_rows,
        composition_rows,
        covariance_rows,
        partition_rows,
    )


def review_response_template(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "review_item_id": row["review_item_id"],
            "reviewer_slot": row["reviewer_slot"],
            "packet_digest": row["packet_digest"],
            "reviewer_id": None,
            "reviewed_state_set": None,
            "confidence_1_to_5": None,
            "reason": None,
            "attestation": REVIEW_ATTESTATION,
            "reviewed_at": None,
        }
        for row in packet
    ]


def validate_review_workflow(
    items: list[dict[str, Any]],
    answer_key: list[dict[str, Any]],
    packets: list[list[dict[str, Any]]],
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = {
        row["review_item_id"]: sorted(row["admissible_state_set"])
        for row in answer_key
    }
    item_by_id = {row["review_item_id"]: row for row in items}
    packet_digest_by_slot = {
        packet[0]["reviewer_slot"]: packet[0]["packet_digest"]
        for packet in packets
        if packet
    }
    completed_paths = {
        "reviewer_a": OUT / "external_review/reviewer_a_completed.jsonl",
        "reviewer_b": OUT / "external_review/reviewer_b_completed.jsonl",
    }
    valid_rows_by_slot: dict[str, dict[str, dict[str, Any]]] = {}
    reviewer_results: list[dict[str, Any]] = []
    valid_reviewer_ids: list[str] = []

    for slot, path in completed_paths.items():
        if not path.is_file():
            reviewer_results.append(
                {
                    "reviewer_slot": slot,
                    "completed_file_present": False,
                    "structurally_valid": False,
                    "reviewed_item_count": 0,
                    "structural_error_count": len(registry),
                }
            )
            continue
        rows = read_jsonl(path)
        ids = [row.get("review_item_id") for row in rows]
        reviewer_ids = {row.get("reviewer_id") for row in rows}
        structural_errors = 0
        for row in rows:
            confidence = row.get("confidence_1_to_5")
            structural_errors += int(row.get("review_item_id") not in registry)
            structural_errors += int(not isinstance(row.get("reviewed_state_set"), list))
            structural_errors += int(
                not isinstance(confidence, int)
                or not CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX
            )
            structural_errors += int(not str(row.get("reason") or "").strip())
            structural_errors += int(row.get("attestation") != REVIEW_ATTESTATION)
            structural_errors += int(
                row.get("packet_digest") != packet_digest_by_slot.get(slot)
            )
            structural_errors += int(not row.get("reviewed_at"))
        structural_errors += len(set(registry) - set(ids))
        structural_errors += len(ids) - len(set(ids))
        structural_valid = bool(
            len(rows) == len(registry)
            and len(reviewer_ids) == 1
            and None not in reviewer_ids
            and structural_errors == 0
        )
        if structural_valid:
            valid_rows_by_slot[slot] = {row["review_item_id"]: row for row in rows}
            valid_reviewer_ids.append(next(iter(reviewer_ids)))
        reviewer_results.append(
            {
                "reviewer_slot": slot,
                "completed_file_present": True,
                "structurally_valid": structural_valid,
                "reviewed_item_count": len(rows),
                "structural_error_count": structural_errors,
            }
        )

    distinct_reviewers = bool(
        len(valid_reviewer_ids) == 2 and len(set(valid_reviewer_ids)) == 2
    )
    pair_status_counts: Counter[str] = Counter()
    adjudication_rows: list[dict[str, Any]] = []
    if distinct_reviewers and len(valid_rows_by_slot) == 2:
        for item_id in sorted(registry):
            row_a = valid_rows_by_slot["reviewer_a"][item_id]
            row_b = valid_rows_by_slot["reviewer_b"][item_id]
            answer_a = sorted(row_a["reviewed_state_set"])
            answer_b = sorted(row_b["reviewed_state_set"])
            if answer_a != answer_b:
                status = "reviewer_disagreement"
            elif answer_a != registry[item_id]:
                status = "registry_conflict"
            else:
                status = "accepted_agreement"
            pair_status_counts[status] += 1
            if status != "accepted_agreement":
                adjudication_rows.append(
                    {
                        **item_by_id[item_id],
                        "adjudication_reason": status,
                        "question": (
                            "Independently list the complete admissible registered-state set; "
                            "other responses and the machine registry remain hidden."
                        ),
                    }
                )
    else:
        pair_status_counts["pending_independent_pair"] = len(registry)

    completed = bool(
        distinct_reviewers
        and pair_status_counts["accepted_agreement"] == len(registry)
        and not adjudication_rows
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase411-IndependentReviewAdjudicationStatus",
        "created_at": created_at,
        "workflow_contract_valid": True,
        "required_reviewer_count": 2,
        "scenario_count_per_reviewer": len(registry),
        "review_results": reviewer_results,
        "distinct_reviewer_identity_pass": distinct_reviewers,
        "pair_status_counts": dict(sorted(pair_status_counts.items())),
        "third_party_adjudication_required": bool(adjudication_rows),
        "adjudication_item_count": len(adjudication_rows),
        "independent_human_rule_review_completed": completed,
        "machine_registry_is_privileged_during_disagreement": False,
        "forced_reviewer_conformity_allowed": False,
        "claim_boundary": (
            "human_review_status_only_no_machine_or_model_evidence_is_created"
        ),
    }
    return status, adjudication_rows


def review_workflow(
    created_at: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_packet = read_jsonl(
        PHASE410_OUT / "external_review/reviewer_a_packet.jsonl"
    )
    answer_key = read_jsonl(
        PHASE410_OUT / "protocol/private/phase410_review_answer_key.jsonl"
    )
    items = [
        {
            key: value
            for key, value in row.items()
            if key not in {"reviewer_slot", "packet_digest"}
        }
        for row in source_packet
    ]
    packets: list[list[dict[str, Any]]] = []
    for slot, salt in (("reviewer_a", "phase411-a"), ("reviewer_b", "phase411-b")):
        ordered = sorted(items, key=lambda row: digest(row["review_item_id"] + salt))
        instruction = {
            "phase_id": PHASE_ID,
            "reviewer_slot": slot,
            "required_fields": [
                "reviewed_state_set",
                "confidence_1_to_5",
                "reason",
            ],
            "independence_rule": REVIEW_ATTESTATION,
        }
        packet_digest = digest(
            canonical_json(instruction)
            + "\n"
            + "\n".join(canonical_json(row) for row in ordered)
        )
        packets.append(
            [
                {
                    **row,
                    "reviewer_slot": slot,
                    "packet_digest": packet_digest,
                    "review_instruction": (
                        "Answer independently, give confidence from 1 to 5, and "
                        "explain the governing history rule in your own words."
                    ),
                }
                for row in ordered
            ]
        )
    status, adjudication = validate_review_workflow(
        items, answer_key, packets, created_at
    )
    return status, packets[0], packets[1], answer_key, adjudication


def main() -> None:
    created_at = now()
    phase410_stage = read_json(
        PHASE410_OUT / "phase410_preflight_stage_summary.json"
    )
    phase410_collector = read_json(
        PHASE410_OUT / "phase410_collector_reducer_equivalence.json"
    )
    semantic, semantic_contexts, semantic_failures = semantic_dual_channel_audit(
        created_at
    )
    (
        operations,
        operation_rows,
        transition_rows,
        composition_rows,
        covariance_rows,
        partition_rows,
    ) = finite_operation_audit(created_at)
    review, packet_a, packet_b, answer_key, adjudication = review_workflow(created_at)

    machine_preflight = bool(
        phase410_stage["assessment"]["machine_preflight_pass"]
        and semantic["valid"]
        and operations["valid"]
        and review["workflow_contract_valid"]
    )
    external_review = review["independent_human_rule_review_completed"]
    collector_equivalence = phase410_collector[
        "incremental_collector_model_token_equivalence_completed"
    ]
    model_qualification_authorized = bool(
        machine_preflight and external_review and collector_equivalence
    )

    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase411-Qualification",
        "created_at": created_at,
        "machine_preflight_pass": machine_preflight,
        "phase410_machine_preflight_pass": phase410_stage["assessment"][
            "machine_preflight_pass"
        ],
        "finite_semantic_dual_channel_pass": semantic["valid"],
        "finite_operation_closure_pass": operations["valid"],
        "review_adjudication_workflow_contract_pass": review[
            "workflow_contract_valid"
        ],
        "independent_human_rule_review_completed": external_review,
        "sealed_model_collector_equivalence_completed": collector_equivalence,
        "model_qualification_authorized": model_qualification_authorized,
        "formal_model_discovery_authorized": False,
        "descriptive_physical_mapping_authorized": False,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
        "authorization_rule": (
            "machine_contracts_are_necessary_but_two_external_reviewers_and_"
            "sealed_real_model_collector_equivalence_remain_mandatory"
        ),
    }

    stage = {
        "schema_version": "85.1.0",
        "phase_id": "Phase411-FiniteSemanticOperationPreflightStage",
        "created_at": created_at,
        "objective": (
            "freeze_finite_dual_channel_semantics_operation_closure_and_"
            "independent_review_adjudication_before_any_cuda_or_physical_run"
        ),
        "assessment": {
            "phase410_direction_correct": True,
            "phase410_added_model_mechanism_evidence": False,
            "six_axes_are_separately_recorded": True,
            "six_axes_are_statistically_or_physically_independent": False,
            "strict_exact_channel_preserved": True,
            "finite_registered_semantic_channel_added": True,
            "open_language_semantics_solved": False,
            "registered_operation_closure_frozen": True,
            "model_functional_bisimulation_established": False,
            "machine_preflight_pass": machine_preflight,
            "independent_external_review_completed": external_review,
            "model_weight_loaded": False,
            "cuda_execution_performed": False,
            "behavioral_case_collected": False,
            "physical_trace_collected": False,
            "causal_intervention_performed": False,
            "neuron_scan_performed": False,
            "language_encoding_closed": False,
        },
        "denominators": {
            "finite_semantic_contract_context_count": semantic[
                "contract_context_count"
            ],
            "finite_semantic_response_case_count": semantic[
                "finite_response_case_count"
            ],
            "registered_operation_count": operations["operation_count"],
            "registered_state_transition_count": operations[
                "state_transition_count"
            ],
            "operation_composition_case_count": operations[
                "composition_case_count"
            ],
            "history_rule_covariance_case_count": operations[
                "history_rule_covariance_case_count"
            ],
            "required_independent_reviewer_count": review[
                "required_reviewer_count"
            ],
            "review_scenario_count_per_reviewer": review[
                "scenario_count_per_reviewer"
            ],
            "future_sealed_model_collector_case_count": 165,
            "model_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "finite_semantic_failure_count": semantic["failure_count"],
            "strict_resolved_case_count": semantic["strict_resolved_case_count"],
            "registered_semantic_resolved_case_count": semantic[
                "registered_semantic_resolved_case_count"
            ],
            "semantic_only_resolved_case_count": semantic[
                "semantic_only_resolved_case_count"
            ],
            "operation_composition_failure_count": operations[
                "composition_failure_count"
            ],
            "history_rule_covariance_failure_count": operations[
                "history_rule_covariance_failure_count"
            ],
            "coarse_observer_unstable_operation_cell_count": operations[
                "coarse_observer_unstable_operation_cell_count"
            ],
            "joint_observer_unstable_operation_cell_count": operations[
                "joint_observer_unstable_operation_cell_count"
            ],
            "completed_external_reviewer_count": sum(
                row["structurally_valid"] for row in review["review_results"]
            ),
            "external_review_accepted_item_count": review["pair_status_counts"].get(
                "accepted_agreement", 0
            ),
            "external_review_disagreement_item_count": review[
                "pair_status_counts"
            ].get("reviewer_disagreement", 0),
            "external_review_registry_conflict_item_count": review[
                "pair_status_counts"
            ].get("registry_conflict", 0),
            "sealed_model_collector_equivalence_case_count": phase410_collector[
                "sealed_model_case_count_compared"
            ],
            "new_behavioral_result_count": 0,
            "new_physical_path_count": 0,
            "new_causal_path_count": 0,
            "new_neuron_path_count": 0,
        },
        "hard_limits": [
            "separately_recorded_response_axes_are_not_proven_statistically_or_physically_independent",
            "finite_registered_templates_do_not_solve_open_language_semantics",
            "machine_exhaustion_validates_implementation_not_human_semantic_validity",
            "registered_world_operations_are_external_test_operations_not_model_internal_operators",
            "joint_reference_partition_is_a_contract_and_not_observed_model_bisimulation",
            "coarse_query_partitions_can_break_under_registered_operations",
            "two_external_human_reviews_are_still_absent",
            "sealed_real_model_collector_equivalence_remains_zero_of_165",
            "no_model_behavior_physical_causal_or_neuron_evidence_was_added",
            "small_models_may_use_coarse_or_model_specific_internal_structures",
            "single_global_progress_percentage_is_invalid",
        ],
        "authorization": {
            "publish_protocol_preflight": True,
            "run_qwen3_model_qualification_next": model_qualification_authorized,
            "run_glm4_model_qualification_next": False,
            "run_ds7b_model_qualification_next": False,
            "run_formal_discovery_next": False,
            "run_descriptive_physical_mapping_next": False,
            "run_causal_intervention_next": False,
            "run_neuron_scan_next": False,
        },
        "next_stage": {
            "phase_id": "Phase410A-ExternalReviewAndCollectorGate",
            "same_qualification_stage": True,
            "automatic_execution_now": False,
            "blocking_requirements": [
                "two_distinct_external_reviewers_complete_all_65_items_with_confidence_and_reasons",
                "reviewer_disagreements_or_registry_conflicts_receive_independent_adjudication_or_contract_redesign",
                "reference_and_incremental_collectors_match_tokens_raw_scores_processed_scores_six_axes_events_stop_and_censoring_on_165_sealed_model_cases",
            ],
            "model_order_after_gate": list(MODELS),
        },
        "single_global_progress_percentage_valid": False,
    }

    write_json(OUT / "phase411_registered_semantic_dual_channel_audit.json", semantic)
    write_jsonl(
        OUT / "protocol/private/phase411_semantic_context_index.jsonl",
        semantic_contexts,
    )
    write_jsonl(
        OUT / "protocol/private/phase411_semantic_failures.jsonl",
        semantic_failures,
    )
    write_json(OUT / "phase411_finite_operation_closure_audit.json", operations)
    write_jsonl(
        OUT / "protocol/private/phase411_operation_registry.jsonl", operation_rows
    )
    write_jsonl(
        OUT / "protocol/private/phase411_state_transitions.jsonl", transition_rows
    )
    write_jsonl(
        OUT / "protocol/private/phase411_operation_composition.jsonl",
        composition_rows,
    )
    write_jsonl(
        OUT / "protocol/private/phase411_history_rule_covariance.jsonl",
        covariance_rows,
    )
    write_jsonl(
        OUT / "protocol/private/phase411_observation_partition_stability.jsonl",
        partition_rows,
    )
    write_jsonl(OUT / "external_review/reviewer_a_packet_v2.jsonl", packet_a)
    write_jsonl(OUT / "external_review/reviewer_b_packet_v2.jsonl", packet_b)
    write_jsonl(
        OUT / "external_review/reviewer_a_response_template_v2.jsonl",
        review_response_template(packet_a),
    )
    write_jsonl(
        OUT / "external_review/reviewer_b_response_template_v2.jsonl",
        review_response_template(packet_b),
    )
    write_jsonl(
        OUT / "external_review/adjudication_packet.jsonl", adjudication
    )
    write_jsonl(
        OUT / "protocol/private/phase411_review_registry.jsonl", answer_key
    )
    write_json(OUT / "phase411_external_review_v2_status.json", review)
    write_json(OUT / "phase411_qualification.json", qualification)
    write_json(OUT / "phase411_stage_summary.json", stage)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
