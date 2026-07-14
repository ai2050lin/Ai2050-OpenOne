#!/usr/bin/env python3
"""Build the Phase410 protocol preflight without loading model weights.

Phase409 froze a large behavioral protocol, but its prefix parser compressed
semantic class, format completeness, sentence boundary, stopping, numeric
validity, and response role into one mutually exclusive state.  This module
keeps those axes independent, audits conflict-order symmetry, exhausts a
finite grammar response universe, and prepares blinded external-review
packets.  Machine checks do not impersonate independent reviewers and do not
authorize CUDA, physical tracing, intervention, or neuron scans.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_protocol import (  # noqa: E402
    MODIFIERS,
    STATE_IDS,
    STRUCTURAL_SURFACES,
    facts_for,
    package_for,
    token_words,
)
from phase409_dynamic_response_protocol import (  # noqa: E402
    AMBIGUOUS_ALIASES,
    FAMILIES,
    HISTORY_MODES,
    INTERFACES,
    MODELS,
    OUT as PHASE409_OUT,
    direct_solver,
    interface_contract_text,
    prior_state_for,
    query_roles,
    response_contract,
)


OUT = ROOT / "tests/gpt5/result/phase410_orthogonal_preflight"
SCHEMA_VERSION = "84.0.0"
PHASE_ID = "Phase410-OrthogonalDynamicPreflight"
SEMANTIC_CLASSES = (
    "no_registered_response",
    "allowed_response",
    "rejected_response",
    "ambiguous_response",
)
FORMAT_CLASSES = ("empty", "partial", "complete", "malformed")
NUMERIC_CLASSES = ("unknown", "finite", "nonfinite", "parse_missing")
REVIEW_ATTESTATION = "I reviewed every item independently without an answer key."
FORBIDDEN_H3_PRIORITY_CUES = (
    "current",
    "earlier",
    "prior",
    "previous",
    "latest",
    "newer",
    "supersede",
    "override",
    "authoritative",
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_response_parse(
    text: str,
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
) -> dict[str, Any]:
    """Classify the whole normalized response, never an interior substring."""
    words = token_words(text)
    ambiguous_hits = [
        alias
        for alias in contract["ambiguous_aliases"]
        if words == token_words(alias)
    ]
    matches: dict[str, list[str]] = {}
    for raw_class, aliases in contract["raw_response_aliases"].items():
        hits = [alias for alias in aliases if words == token_words(alias)]
        if hits:
            matches[raw_class] = hits
    matched_classes = sorted(matches)
    if ambiguous_hits or len(matched_classes) > 1:
        return {
            "semantic_class": "ambiguous_response",
            "response_role": "ambiguous",
            "raw_response_class": None,
            "matched_raw_classes": matched_classes,
        }
    if not matched_classes:
        return {
            "semantic_class": "no_registered_response",
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
        "semantic_class": (
            "allowed_response"
            if decoded_states & expected_states
            else "rejected_response"
        ),
        "response_role": role,
        "raw_response_class": raw_class,
        "decoded_state_set": sorted(decoded_states),
        "matched_raw_classes": matched_classes,
    }


def format_class(text: str, contract: dict[str, Any]) -> str:
    words = token_words(text)
    if not words:
        return "empty"
    alias_words = [
        token_words(alias)
        for aliases in contract["raw_response_aliases"].values()
        for alias in aliases
    ]
    ambiguous_words = [token_words(alias) for alias in contract["ambiguous_aliases"]]
    if words in alias_words or words in ambiguous_words:
        return "complete"
    if any(
        len(words) < len(alias) and alias[: len(words)] == words
        for alias in alias_words + ambiguous_words
    ):
        return "partial"
    return "malformed"


def orthogonal_prefix_state(
    text: str,
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
    *,
    numeric_validity: str = "unknown",
    stopped: bool = False,
) -> dict[str, Any]:
    if numeric_validity not in NUMERIC_CLASSES:
        raise ValueError(numeric_validity)
    parsed = exact_response_parse(
        text, contract, current_state, prior_state, history_mode
    )
    state = {
        "semantic_class": parsed["semantic_class"],
        "format_class": format_class(text, contract),
        "boundary_reached": bool(re.search(r"[.!?](?:\s|$)", text)),
        "model_stopped": bool(stopped),
        "numeric_validity": numeric_validity,
        "response_role": parsed["response_role"],
        "raw_response_class": parsed["raw_response_class"],
        "matched_raw_classes": parsed["matched_raw_classes"],
    }
    if state["semantic_class"] not in SEMANTIC_CLASSES:
        raise RuntimeError(state)
    if state["format_class"] not in FORMAT_CLASSES:
        raise RuntimeError(state)
    return state


def scan_orthogonal_event_process(
    decoded_prefixes: list[str],
    contract: dict[str, Any],
    current_state: str,
    prior_state: str,
    history_mode: str,
    *,
    numeric_validity_by_step: list[str] | None = None,
    stopped: bool,
) -> dict[str, Any]:
    numeric = numeric_validity_by_step or ["unknown"] * len(decoded_prefixes)
    if len(numeric) != len(decoded_prefixes):
        raise ValueError("numeric validity length does not match prefixes")
    axes = (
        "semantic_class",
        "format_class",
        "boundary_reached",
        "numeric_validity",
        "response_role",
    )
    previous: dict[str, Any] = {
        "semantic_class": "no_registered_response",
        "format_class": "empty",
        "boundary_reached": False,
        "numeric_validity": "unknown",
        "response_role": "none",
    }
    transitions: list[dict[str, Any]] = []
    step_states: list[dict[str, Any]] = []
    first_by_semantic: dict[str, int] = {}
    for step, (prefix, validity) in enumerate(
        zip(decoded_prefixes, numeric, strict=True), start=1
    ):
        state = orthogonal_prefix_state(
            prefix,
            contract,
            current_state,
            prior_state,
            history_mode,
            numeric_validity=validity,
            stopped=False,
        )
        step_states.append({"step": step, "text": prefix, "state": state})
        first_by_semantic.setdefault(state["semantic_class"], step)
        for axis in axes:
            if state[axis] != previous[axis]:
                transitions.append(
                    {
                        "step": step,
                        "axis": axis,
                        "from": previous[axis],
                        "to": state[axis],
                    }
                )
                previous[axis] = state[axis]
    stop_step = len(decoded_prefixes) + 1 if stopped else None
    if stopped:
        transitions.append(
            {
                "step": stop_step,
                "axis": "model_stopped",
                "from": False,
                "to": True,
            }
        )
    final_state = dict(step_states[-1]["state"]) if step_states else {
        **previous,
        "model_stopped": False,
        "raw_response_class": None,
        "matched_raw_classes": [],
    }
    final_state["model_stopped"] = bool(stopped)
    coincident = Counter(row["step"] for row in transitions)
    return {
        "axis_transitions": transitions,
        "step_states": step_states,
        "first_semantic_event_by_class": first_by_semantic,
        "stop_event": stop_step,
        "right_censored_at_horizon": not stopped,
        "coincident_transition_steps": sorted(
            step for step, count in coincident.items() if count > 1
        ),
        "final_state": final_state,
    }


def neutral_facts(text: str) -> str:
    replacements = (
        ("Final record:", "Record facts:"),
        ("In the final record,", "In this record,"),
        ("Use only these final facts:", "These record facts say:"),
        ("The closed record states that", "This record states that"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def symmetric_h3_messages(
    family: str,
    package: dict[str, Any],
    state_first: str,
    state_second: str,
    surface: dict[str, Any],
    interface: str,
    query_role: str,
) -> tuple[list[dict[str, str]], str, str, str]:
    facts_first = neutral_facts(facts_for(family, package, state_first, surface))
    facts_second = neutral_facts(facts_for(family, package, state_second, surface))
    query, prefix = interface_contract_text(
        family, package, interface, query_role
    )
    messages = [
        {"role": "user", "content": f"Conflicting record 1:\n{facts_first}"},
        {"role": "assistant", "content": "Recorded."},
        {
            "role": "user",
            "content": (
                f"Conflicting record 2:\n{facts_second}\n\n"
                "The two records have equal status. Their numbering and display "
                "order assign no priority, recency, or authority. Treat both "
                "registered states as admissible.\n"
                f"Task: {query}"
            ),
        },
    ]
    return messages, prefix, facts_first, facts_second


def h3_skeleton(
    messages: list[dict[str, str]], facts_first: str, facts_second: str
) -> str:
    normalized = [
        {
            **message,
            "content": message["content"]
            .replace(facts_first, "<FACTS>")
            .replace(facts_second, "<FACTS>"),
        }
        for message in messages
    ]
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


def h3_symmetry_audit(created_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in FAMILIES:
        package = package_for(family, 0, 0)
        for state_a, state_b in combinations(STATE_IDS[family], 2):
            for interface in INTERFACES[family]:
                for role in query_roles(family, interface):
                    for surface in STRUCTURAL_SURFACES:
                        pair_id = "p410h3p_" + digest(
                            f"{family}:{state_a}:{state_b}:{interface}:{role}:"
                            f"{surface['surface_id']}",
                            24,
                        )
                        target = sorted({state_a, state_b})
                        for first, second in ((state_a, state_b), (state_b, state_a)):
                            messages, prefix, facts_first, facts_second = (
                                symmetric_h3_messages(
                                    family,
                                    package,
                                    first,
                                    second,
                                    surface,
                                    interface,
                                    role,
                                )
                            )
                            rendered_text = "\n".join(
                                message["content"] for message in messages
                            ).lower()
                            cue_hits = [
                                cue
                                for cue in FORBIDDEN_H3_PRIORITY_CUES
                                if re.search(rf"\b{re.escape(cue)}\b", rendered_text)
                            ]
                            row = {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": PHASE_ID,
                                "pair_id": pair_id,
                                "family_id": family,
                                "interface_id": interface,
                                "query_role": role,
                                "surface_id": surface["surface_id"],
                                "first_state_private": first,
                                "second_state_private": second,
                                "admissible_state_set_private": target,
                                "order_neutral_skeleton_digest": digest(
                                    h3_skeleton(messages, facts_first, facts_second)
                                ),
                                "assistant_prefix_digest": digest(prefix),
                                "forbidden_priority_cue_hits": cue_hits,
                            }
                            rows.append(row)
                            pair_rows[pair_id].append(row)
    pair_failures = []
    for pair_id, variants in pair_rows.items():
        valid = bool(
            len(variants) == 2
            and len(
                {
                    tuple(row["admissible_state_set_private"])
                    for row in variants
                }
            )
            == 1
            and len(
                {row["order_neutral_skeleton_digest"] for row in variants}
            )
            == 1
            and not any(row["forbidden_priority_cue_hits"] for row in variants)
        )
        if not valid:
            pair_failures.append(pair_id)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase410-H3OrderSymmetryAudit",
        "created_at": created_at,
        "valid": not pair_failures,
        "unordered_contract_pair_count": len(pair_rows),
        "order_variant_count": len(rows),
        "pair_failure_count": len(pair_failures),
        "forbidden_priority_cue_hit_count": sum(
            len(row["forbidden_priority_cue_hits"]) for row in rows
        ),
        "failed_pair_ids": pair_failures,
        "behavioral_order_invariance_observed": False,
        "claim_boundary": (
            "prompt_contract_is_order_mirrored_but_model_behavior_has_not_been_run"
        ),
    }
    return summary, rows


def add_candidate(
    candidates: dict[str, dict[str, Any]],
    text: str,
    expected_semantic: str,
    expected_format: str,
    case_kind: str,
) -> None:
    expected = {
        "semantic_class": expected_semantic,
        "format_class": expected_format,
    }
    prior = candidates.get(text)
    if prior is not None and (
        prior["semantic_class"], prior["format_class"]
    ) != (expected_semantic, expected_format):
        raise RuntimeError(f"Grammar candidate collision for {text!r}")
    expected["case_kinds"] = sorted(
        set((prior or {}).get("case_kinds", [])) | {case_kind}
    )
    candidates[text] = expected


def grammar_universe_audit(
    created_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for modifier in MODIFIERS:
        package = dict(package_for("grammar_constraint", 0, 0))
        package["modifier"] = modifier
        for interface in INTERFACES["grammar_constraint"]:
            for current in STATE_IDS["grammar_constraint"]:
                prior = prior_state_for("grammar_constraint", current)
                contract = response_contract(
                    "grammar_constraint", package, interface
                )
                candidates: dict[str, dict[str, Any]] = {}
                add_candidate(
                    candidates,
                    "",
                    "no_registered_response",
                    "empty",
                    "empty",
                )
                expected_states = set(
                    direct_solver(
                        "grammar_constraint",
                        current,
                        "h0_current_only",
                        prior,
                    )
                )
                for raw_class, aliases in contract["raw_response_aliases"].items():
                    decoded = set(contract["raw_class_to_states"][raw_class])
                    semantic = (
                        "allowed_response"
                        if decoded & expected_states
                        else "rejected_response"
                    )
                    for alias in aliases:
                        add_candidate(
                            candidates, alias, semantic, "complete", "exact_alias"
                        )
                        add_candidate(
                            candidates,
                            alias + ".",
                            semantic,
                            "complete",
                            "punctuated_alias",
                        )
                        add_candidate(
                            candidates,
                            alias + " extra",
                            "no_registered_response",
                            "malformed",
                            "trailing_extra",
                        )
                        add_candidate(
                            candidates,
                            "answer " + alias,
                            "no_registered_response",
                            "malformed",
                            "leading_extra",
                        )
                        words = token_words(alias)
                        for width in range(1, len(words)):
                            add_candidate(
                                candidates,
                                " ".join(words[:width]),
                                "no_registered_response",
                                "partial",
                                "proper_prefix",
                            )
                for alias in AMBIGUOUS_ALIASES:
                    add_candidate(
                        candidates,
                        alias,
                        "ambiguous_response",
                        "complete",
                        "explicit_ambiguity",
                    )
                add_candidate(
                    candidates,
                    "unregistered response",
                    "no_registered_response",
                    "malformed",
                    "unregistered_text",
                )
                if interface == "sentence_completion":
                    for form in ("is", "are", "was", "were"):
                        add_candidate(
                            candidates,
                            form,
                            "no_registered_response",
                            "partial",
                            "bare_be_form",
                        )
                        add_candidate(
                            candidates,
                            f"{form} wrongmodifier",
                            "no_registered_response",
                            "malformed",
                            "wrong_adjective",
                        )
                for text, expected in sorted(candidates.items()):
                    observed = orthogonal_prefix_state(
                        text,
                        contract,
                        current,
                        prior,
                        "h0_current_only",
                    )
                    valid = bool(
                        observed["semantic_class"] == expected["semantic_class"]
                        and observed["format_class"] == expected["format_class"]
                    )
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE_ID,
                        "modifier": modifier,
                        "interface_id": interface,
                        "current_state": current,
                        "response_text": text,
                        "case_kinds": expected["case_kinds"],
                        "expected_semantic_class": expected["semantic_class"],
                        "expected_format_class": expected["format_class"],
                        "observed_semantic_class": observed["semantic_class"],
                        "observed_format_class": observed["format_class"],
                        "boundary_reached": observed["boundary_reached"],
                        "valid": valid,
                    }
                    rows.append(row)
                    category_counts[observed["semantic_class"]] += 1
                    if not valid:
                        failures.append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase410-GrammarFiniteUniverseAudit",
        "created_at": created_at,
        "valid": not failures,
        "interface_count": len(INTERFACES["grammar_constraint"]),
        "state_count": len(STATE_IDS["grammar_constraint"]),
        "modifier_count": len(MODIFIERS),
        "finite_response_case_count": len(rows),
        "failure_count": len(failures),
        "semantic_class_counts": dict(sorted(category_counts.items())),
        "substring_acceptance_allowed": False,
        "bare_be_form_registered_for_sentence_completion": False,
        "leading_or_trailing_extra_text_registered": False,
        "claim_boundary": "finite_registered_grammar_contract_only_not_general_grammar",
    }
    return summary, rows


def review_rule_text(history_mode: str) -> str:
    return {
        "h0_current_only": "Only the current state is present and governs.",
        "h1_prior_equivalent_state": (
            "The earlier and current records encode the same state."
        ),
        "h2_prior_irrelevant_content": (
            "The unrelated record constrains another object; the current state governs."
        ),
        "h3_prior_conflicting_state": (
            "Two distinct registered states conflict and neither has priority; both are admissible."
        ),
        "h4_prior_conflict_then_current_explicit_override": (
            "The current final state explicitly overrides the prior conflicting state."
        ),
    }[history_mode]


def review_packets(created_at: str) -> tuple[dict[str, Any], list[dict], list[dict], list[dict]]:
    agreement = read_json(PHASE409_OUT / "phase409_rule_engine_agreement.json")
    items = []
    answer_key = []
    for scenario in agreement["scenarios"]:
        item_id = "p410review_" + digest(
            ":".join(
                (
                    scenario["family_id"],
                    scenario["current_state"],
                    scenario["prior_state"],
                    scenario["history_mode"],
                )
            ),
            24,
        )
        item = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "review_item_id": item_id,
            "family_id": scenario["family_id"],
            "registered_state_universe": list(STATE_IDS[scenario["family_id"]]),
            "current_state": scenario["current_state"],
            "prior_state": scenario["prior_state"],
            "history_mode": scenario["history_mode"],
            "history_rule": review_rule_text(scenario["history_mode"]),
            "question": "List the complete admissible registered-state set.",
        }
        items.append(item)
        answer_key.append(
            {
                "review_item_id": item_id,
                "admissible_state_set": sorted(scenario["direct_state_set"]),
            }
        )
    packets = []
    for reviewer_slot, salt in (("reviewer_a", "independent-a"), ("reviewer_b", "independent-b")):
        ordered = sorted(items, key=lambda row: digest(row["review_item_id"] + salt))
        packet_digest = digest(
            "\n".join(json.dumps(row, sort_keys=True) for row in ordered)
        )
        packets.append(
            [
                {
                    **row,
                    "reviewer_slot": reviewer_slot,
                    "packet_digest": packet_digest,
                }
                for row in ordered
            ]
        )
    status = validate_external_reviews(answer_key, packets, created_at)
    return status, packets[0], packets[1], answer_key


def validate_external_reviews(
    answer_key: list[dict[str, Any]],
    packets: list[list[dict[str, Any]]],
    created_at: str,
) -> dict[str, Any]:
    expected = {
        row["review_item_id"]: sorted(row["admissible_state_set"])
        for row in answer_key
    }
    completed_paths = {
        "reviewer_a": OUT / "external_review/reviewer_a_completed.jsonl",
        "reviewer_b": OUT / "external_review/reviewer_b_completed.jsonl",
    }
    packet_digest_by_slot = {
        packet[0]["reviewer_slot"]: packet[0]["packet_digest"]
        for packet in packets
        if packet
    }
    reviewer_results = []
    reviewer_ids = []
    for slot, path in completed_paths.items():
        if not path.is_file():
            reviewer_results.append(
                {
                    "reviewer_slot": slot,
                    "completed_file_present": False,
                    "valid": False,
                    "reviewed_item_count": 0,
                    "error_count": len(expected),
                }
            )
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = [row.get("review_item_id") for row in rows]
        reviewer_id_set = {row.get("reviewer_id") for row in rows}
        attestations = {row.get("attestation") for row in rows}
        packet_digests = {row.get("packet_digest") for row in rows}
        reviewed_at_present = all(bool(row.get("reviewed_at")) for row in rows)
        errors = 0
        for row in rows:
            if sorted(row.get("reviewed_state_set") or []) != expected.get(
                row.get("review_item_id")
            ):
                errors += 1
        errors += len(set(expected) - set(ids))
        errors += len(ids) - len(set(ids))
        valid = bool(
            len(rows) == len(expected)
            and len(reviewer_id_set) == 1
            and None not in reviewer_id_set
            and attestations == {REVIEW_ATTESTATION}
            and packet_digests == {packet_digest_by_slot.get(slot)}
            and reviewed_at_present
            and errors == 0
        )
        if valid:
            reviewer_ids.append(next(iter(reviewer_id_set)))
        reviewer_results.append(
            {
                "reviewer_slot": slot,
                "completed_file_present": True,
                "valid": valid,
                "reviewed_item_count": len(rows),
                "error_count": errors,
                "packet_digest_match": packet_digests
                == {packet_digest_by_slot.get(slot)},
                "reviewed_at_present": reviewed_at_present,
            }
        )
    independent = bool(
        len(reviewer_ids) == 2 and len(set(reviewer_ids)) == 2
    )
    complete = bool(
        independent and all(row["valid"] for row in reviewer_results)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase410-IndependentRuleReviewStatus",
        "created_at": created_at,
        "required_reviewer_count": 2,
        "scenario_count_per_reviewer": len(expected),
        "packet_count": len(packets),
        "review_results": reviewer_results,
        "distinct_reviewer_identity_pass": independent,
        "independent_human_rule_review_completed": complete,
        "machine_generated_review_is_acceptable": False,
    }


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[.!?]")


def reference_decode(tokens: list[str]) -> str:
    text = ""
    for token in tokens:
        if token in ".!?":
            text += token
        else:
            text += (" " if text and not text.endswith(" ") else "") + token
    return text


class IncrementalDecoder:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def append(self, token: str) -> str:
        if token in ".!?":
            self.parts.append(token)
        else:
            if self.parts:
                self.parts.append(" ")
            self.parts.append(token)
        return "".join(self.parts)


def collector_reducer_audit(
    created_at: str, grammar_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    path_count = 0
    prefix_count = 0
    mismatch_count = 0
    for row in grammar_rows:
        tokens = TOKEN_PATTERN.findall(row["response_text"])
        incremental = IncrementalDecoder()
        for index, token in enumerate(tokens, start=1):
            reference = reference_decode(tokens[:index])
            observed = incremental.append(token)
            prefix_count += 1
            if reference != observed:
                mismatch_count += 1
        path_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase410-CollectorReducerEquivalenceAudit",
        "created_at": created_at,
        "synthetic_finite_path_count": path_count,
        "synthetic_prefix_count": prefix_count,
        "synthetic_reference_incremental_text_mismatch_count": mismatch_count,
        "synthetic_reducer_equivalence_pass": mismatch_count == 0,
        "sealed_model_case_count_required": 165,
        "sealed_model_case_count_compared": 0,
        "incremental_collector_model_token_equivalence_completed": False,
        "claim_boundary": (
            "synthetic_text_reducer_agreement_does_not_establish_model_token_or_score_equivalence"
        ),
    }


def orthogonal_contract_audit(created_at: str) -> dict[str, Any]:
    family = "grammar_constraint"
    package = package_for(family, 0, 0)
    current = "singular_present"
    prior = prior_state_for(family, current)
    contract = response_contract(family, package, "sentence_completion")
    modifier = package["modifier"]
    process = scan_orthogonal_event_process(
        [
            " is",
            f" is {modifier}",
            f" is {modifier}.",
            f" is {modifier}. extra",
        ],
        contract,
        current,
        prior,
        "h0_current_only",
        numeric_validity_by_step=["finite"] * 4,
        stopped=True,
    )
    step_two = process["step_states"][1]["state"]
    step_three = process["step_states"][2]["state"]
    valid = bool(
        step_two["semantic_class"] == "allowed_response"
        and step_two["format_class"] == "complete"
        and not step_two["boundary_reached"]
        and step_three["semantic_class"] == "allowed_response"
        and step_three["format_class"] == "complete"
        and step_three["boundary_reached"]
        and process["final_state"]["model_stopped"]
        and process["coincident_transition_steps"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase410-OrthogonalStateContractAudit",
        "created_at": created_at,
        "valid": valid,
        "orthogonal_axes": [
            "semantic_class",
            "format_class",
            "boundary_reached",
            "model_stopped",
            "numeric_validity",
            "response_role",
        ],
        "mutually_exclusive_global_automaton_removed": True,
        "coincident_events_supported": True,
        "synthetic_process": process,
        "claim_boundary": "external_response_measurement_not_model_internal_state",
    }


def response_template(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "review_item_id": row["review_item_id"],
            "reviewer_slot": row["reviewer_slot"],
            "packet_digest": row["packet_digest"],
            "reviewer_id": None,
            "reviewed_state_set": None,
            "attestation": REVIEW_ATTESTATION,
            "reviewed_at": None,
        }
        for row in packet
    ]


def main() -> None:
    created_at = now()
    orthogonal = orthogonal_contract_audit(created_at)
    h3_summary, h3_rows = h3_symmetry_audit(created_at)
    grammar_summary, grammar_rows = grammar_universe_audit(created_at)
    review_status, packet_a, packet_b, answer_key = review_packets(created_at)
    collector = collector_reducer_audit(created_at, grammar_rows)

    machine_preflight = bool(
        orthogonal["valid"]
        and h3_summary["valid"]
        and grammar_summary["valid"]
        and collector["synthetic_reducer_equivalence_pass"]
    )
    external_review = review_status[
        "independent_human_rule_review_completed"
    ]
    model_collector = collector[
        "incremental_collector_model_token_equivalence_completed"
    ]
    model_qualification_authorized = bool(
        machine_preflight and external_review and model_collector
    )
    descriptive_physical_authorized = False
    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase410-PreflightQualification",
        "created_at": created_at,
        "machine_preflight_pass": machine_preflight,
        "orthogonal_state_contract_pass": orthogonal["valid"],
        "h3_order_symmetry_contract_pass": h3_summary["valid"],
        "grammar_finite_universe_pass": grammar_summary["valid"],
        "synthetic_collector_reducer_pass": collector[
            "synthetic_reducer_equivalence_pass"
        ],
        "independent_human_rule_review_completed": external_review,
        "incremental_collector_model_token_equivalence_completed": model_collector,
        "model_qualification_authorized": model_qualification_authorized,
        "formal_model_discovery_authorized": False,
        "descriptive_physical_mapping_authorized": descriptive_physical_authorized,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
        "descriptive_physical_gate_required": [
            "registry",
            "machine_preflight",
            "external_rule_review",
            "model_collector_equivalence",
            "model_qualification",
            "interface_partition",
            "surface_replication",
            "lexical_replication",
            "negative_control_specificity",
            "instrument_qualification",
        ],
        "correction_to_proposed_phase410_gate": (
            "registry_machine_partition_surface_alone_is_insufficient_because_"
            "template_lexical_and_instrument_artifacts_can_be_mapped"
        ),
    }
    stage = {
        "schema_version": "84.1.0",
        "phase_id": "Phase410-OrthogonalDynamicPreflightStage",
        "created_at": created_at,
        "objective": (
            "repair_dynamic_measurement_object_and_freeze_external_review_and_"
            "collector_gates_before_any_new_model_or_physical_execution"
        ),
        "assessment": {
            "phase408_behavioral_evidence_preserved": True,
            "phase409_protocol_registry_preserved": True,
            "phase409_single_automaton_is_sufficient": False,
            "orthogonal_dynamic_state_contract_frozen": True,
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
            "orthogonal_axis_count": 6,
            "h3_unordered_contract_pair_count": h3_summary[
                "unordered_contract_pair_count"
            ],
            "h3_order_variant_count": h3_summary["order_variant_count"],
            "grammar_finite_response_case_count": grammar_summary[
                "finite_response_case_count"
            ],
            "independent_review_scenario_count_per_reviewer": review_status[
                "scenario_count_per_reviewer"
            ],
            "required_independent_reviewer_count": review_status[
                "required_reviewer_count"
            ],
            "synthetic_collector_path_count": collector[
                "synthetic_finite_path_count"
            ],
            "future_sealed_model_qualification_case_count": 165,
            "model_case_count_consumed": 0,
            "physical_case_count_consumed": 0,
        },
        "results": {
            "orthogonal_state_contract_failure_count": 0
            if orthogonal["valid"]
            else 1,
            "h3_order_symmetry_failure_count": h3_summary[
                "pair_failure_count"
            ],
            "h3_priority_cue_hit_count": h3_summary[
                "forbidden_priority_cue_hit_count"
            ],
            "grammar_finite_universe_failure_count": grammar_summary[
                "failure_count"
            ],
            "synthetic_collector_text_mismatch_count": collector[
                "synthetic_reference_incremental_text_mismatch_count"
            ],
            "completed_external_reviewer_count": sum(
                row["valid"] for row in review_status["review_results"]
            ),
            "sealed_model_collector_equivalence_case_count": collector[
                "sealed_model_case_count_compared"
            ],
            "new_behavioral_result_count": 0,
            "new_physical_path_count": 0,
            "new_causal_path_count": 0,
            "new_neuron_path_count": 0,
        },
        "scientific_claim_audit": {
            "existing_mathematics_cannot_predict_protein_folding": False,
            "connectome_alone_is_a_complete_behavior_model": False,
            "missing_mesoscale_process_theory_is_established_fact": False,
            "missing_mesoscale_process_theory_is_a_testable_research_hypothesis": True,
            "protein_worm_and_language_model_share_one_proven_invariant": False,
            "candidate_common_question": (
                "which_conditioned_state_partition_and_transition_composition_"
                "preserve_future_function_across_scale"
            ),
        },
        "hard_limits": [
            "phase408_883_condition_cells_are_behavioral_conditions_not_internal_physical_states",
            "phase409_65280_abstract_cases_are_protocol_expansions_not_independent_semantic_worlds",
            "orthogonal_parser_is_external_measurement_not_model_internal_dynamics",
            "h3_prompt_symmetry_does_not_establish_behavioral_order_invariance",
            "grammar_exhaustion_covers_only_the_frozen_finite_response_contract",
            "synthetic_collector_agreement_does_not_establish_model_token_or_score_equivalence",
            "external_review_is_absent_and_cannot_be_self_certified",
            "no_model_physical_causal_or_neuron_evidence_was_added",
            "small_models_may_use_coarse_or_model_specific_internal_structures",
            "single_global_progress_percentage_is_invalid",
        ],
        "authorization": {
            "publish_protocol_preflight": True,
            "run_qwen3_model_qualification_next": model_qualification_authorized,
            "run_glm4_model_qualification_next": False,
            "run_ds7b_model_qualification_next": False,
            "run_formal_discovery_next": False,
            "run_descriptive_physical_mapping_next": descriptive_physical_authorized,
            "run_causal_intervention_next": False,
            "run_neuron_scan_next": False,
        },
        "next_stage": {
            "phase_id": "Phase410A",
            "objective": (
                "two_independent_human_reviews_and_exact_reference_incremental_"
                "collector_equivalence_on_sealed_model_cases"
            ),
            "automatic_execution_now": False,
            "blocking_requirements": [
                "two_distinct_external_reviewers_complete_all_65_items",
                "reference_and_incremental_collectors_match_tokens_scores_events_stop_and_censoring_on_165_sealed_cases",
            ],
            "model_order_after_gate": list(MODELS),
        },
        "single_global_progress_percentage_valid": False,
    }

    write_json(OUT / "phase410_orthogonal_state_contract.json", orthogonal)
    write_json(OUT / "phase410_h3_order_symmetry_audit.json", h3_summary)
    write_jsonl(OUT / "protocol/private/phase410_h3_order_cases.jsonl", h3_rows)
    write_json(OUT / "phase410_grammar_finite_universe_audit.json", grammar_summary)
    write_jsonl(
        OUT / "protocol/private/phase410_grammar_finite_universe.jsonl",
        grammar_rows,
    )
    write_jsonl(OUT / "external_review/reviewer_a_packet.jsonl", packet_a)
    write_jsonl(OUT / "external_review/reviewer_b_packet.jsonl", packet_b)
    write_jsonl(
        OUT / "external_review/reviewer_a_response_template.jsonl",
        response_template(packet_a),
    )
    write_jsonl(
        OUT / "external_review/reviewer_b_response_template.jsonl",
        response_template(packet_b),
    )
    write_jsonl(
        OUT / "protocol/private/phase410_review_answer_key.jsonl", answer_key
    )
    write_json(OUT / "phase410_external_review_status.json", review_status)
    write_json(OUT / "phase410_collector_reducer_equivalence.json", collector)
    write_json(OUT / "phase410_preflight_qualification.json", qualification)
    write_json(OUT / "phase410_preflight_stage_summary.json", stage)
    print(json.dumps(stage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
