#!/usr/bin/env python3
"""Executable Phase578 scorer for the frozen Phase577 behavior contract.

This module is deliberately model-free.  It turns observed generated-token
prefix decodes into case, aggregation-unit, family, and model decisions.  The
thresholds are copied from the already frozen Phase577 protocol; they are
admission rules, not discovered laws or mechanism formulas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE = "Phase578"
SOURCE_SCHEMA = "phase578_gpt5_behavior_scorer.v1"
REPEATS = ("repeat1", "repeat2")
RELATIONS = ("fruit_membership", "citrus_membership")
INTERFACES = ("direct", "selection")
PREFIX_TOKEN_BUDGET = 8
EXPECTED_CASES = 336
EXPECTED_UNITS = 36

# Phase577 says that only ASCII whitespace plus quote, bracket, and bullet
# punctuation may precede the candidate.  Phase578 makes that prose executable
# by freezing this explicit character registry.  In particular, ':' is absent,
# so "Answer: YES" is intentionally unresolved rather than silently accepted.
LEADING_IGNORED_CHARACTERS = frozenset(
    " \t\r\n\v\f\"'`()[]{}“”‘’-*+•·‣◦▪▫"
)
EXACT_OUTER_TRIM_CHARACTERS = " \t\r\n\v\f.,!?:;\"'`()[]{}"

GATE = {
    "direct_minimum_of_6": 5,
    "selection_minimum_of_16": 14,
    "selection_each_query_polarity_minimum_of_8": 7,
    "family_minimums": {
        "fruit_membership|direct": 10,
        "citrus_membership|direct": 10,
        "fruit_membership|selection": 5,
        "citrus_membership|selection": 5,
    },
    "minimum_passing_units_of_36": 30,
    "fruit_direct_nonfruit_food_required_of_2": 2,
    "fruit_selection_nonfruit_food_required_of_2": 2,
    "case_semantic_stable_micro_floor": 0.85,
}


class ScoringError(RuntimeError):
    """The evidence does not satisfy the executable input contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_normalize(text: str) -> str:
    """Apply the Phase577 semantic normalization without deleting punctuation."""

    require(isinstance(text, str), "semantic text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def exact_short_normalize(text: str) -> str:
    """Normalize and trim the frozen, finite outer-punctuation registry."""

    return semantic_normalize(text).strip(EXACT_OUTER_TRIM_CHARACTERS)


def _is_candidate_boundary(character: str) -> bool:
    return bool(character) and (
        character.isspace()
        or (
            unicodedata.category(character).startswith("P")
            and character != "_"
        )
    )


def _trim_allowed_leading(text: str) -> str:
    index = 0
    while index < len(text) and text[index] in LEADING_IGNORED_CHARACTERS:
        index += 1
    return text[index:]


def _normalized_candidate_registry(
    candidate_groups: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    require(isinstance(candidate_groups, Mapping), "candidate_groups must be a mapping")
    require(len(candidate_groups) == 2, "exactly two registered candidates are required")
    registry: dict[str, tuple[str, ...]] = {}
    alias_owners: dict[str, set[str]] = defaultdict(set)
    for raw_owner, raw_aliases in candidate_groups.items():
        require(isinstance(raw_owner, str) and raw_owner, "candidate owner is invalid")
        require(
            isinstance(raw_aliases, Sequence)
            and not isinstance(raw_aliases, (str, bytes))
            and bool(raw_aliases),
            f"candidate {raw_owner!r} has no aliases",
        )
        aliases: list[str] = []
        for raw_alias in raw_aliases:
            alias = semantic_normalize(raw_alias)
            require(bool(alias), f"candidate {raw_owner!r} has an empty alias")
            require(alias == alias.strip(), "candidate alias outer whitespace drift")
            aliases.append(alias)
            alias_owners[alias].add(raw_owner)
        require(len(aliases) == len(set(aliases)), f"duplicate alias in {raw_owner!r}")
        registry[raw_owner] = tuple(sorted(aliases, key=lambda item: (-len(item), item)))
    require(
        all(len(owners) == 1 for owners in alias_owners.values()),
        "a normalized alias belongs to multiple candidates",
    )
    return registry


def prefix_candidate_owners(
    text: str, candidate_groups: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return candidate owners whose complete alias starts the allowed prefix."""

    registry = _normalized_candidate_registry(candidate_groups)
    normalized = _trim_allowed_leading(semantic_normalize(text))
    owners: set[str] = set()
    for owner, aliases in registry.items():
        for alias in aliases:
            if not normalized.startswith(alias):
                continue
            following = normalized[len(alias):len(alias) + 1]
            if not following or _is_candidate_boundary(following):
                owners.add(owner)
                break
    return tuple(sorted(owners))


def diagnostic_mentions(
    text: str, candidate_groups: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Find complete candidate aliases anywhere in the normalized full output."""

    registry = _normalized_candidate_registry(candidate_groups)
    normalized = semantic_normalize(text)
    owners: set[str] = set()
    for owner, aliases in registry.items():
        for alias in aliases:
            start = 0
            while True:
                index = normalized.find(alias, start)
                if index < 0:
                    break
                before = normalized[index - 1:index]
                after = normalized[index + len(alias):index + len(alias) + 1]
                if (not before or _is_candidate_boundary(before)) and (
                    not after or _is_candidate_boundary(after)
                ):
                    owners.add(owner)
                    break
                start = index + 1
            if owner in owners:
                break
    return tuple(sorted(owners))


def validate_case(case: Mapping[str, Any], expected_split: str = "development") -> None:
    require(case.get("phase_id") == "Phase577", "case phase drift")
    require(case.get("schema_version") == "phase577_gpt5_natural_behavior_case.v2",
            "case schema drift")
    require(case.get("split") == expected_split, "case split is not authorized")
    require(case.get("sealed") is False, "sealed case is forbidden")
    require(case.get("relation") in RELATIONS, "unknown relation")
    require(case.get("interface") in INTERFACES, "unknown interface")
    require(case.get("output_contract") in {"semantic_label_first", "exact_short"},
            "unknown output contract")
    require(isinstance(case.get("case_id"), str) and bool(case["case_id"]),
            "case_id is invalid")
    require(isinstance(case.get("analysis_unit_id"), str)
            and bool(case["analysis_unit_id"]), "analysis_unit_id is invalid")
    registry = _normalized_candidate_registry(case.get("candidate_groups", {}))
    target, foil = case.get("target"), case.get("foil")
    require(target in registry and foil in registry and target != foil,
            "target/foil registry is invalid")
    require(case.get("query_polarity") in {"affirmative", "positive", "negative"},
            "query polarity is invalid")
    if case["interface"] == "direct":
        require(case["query_polarity"] == "affirmative", "direct polarity drift")
        require(case.get("surface_id") in range(6), "direct surface drift")
        require(case.get("order") is None, "direct order must be null")
    else:
        require(case["query_polarity"] in {"positive", "negative"},
                "selection polarity drift")
        require(case.get("surface_id") in range(4), "selection surface drift")
        require(case.get("order") in {0, 1}, "selection order drift")


def classify_observation(
    case: Mapping[str, Any],
    generated_text: str,
    prefix_text_by_generated_token: Sequence[str],
) -> dict[str, Any]:
    """Classify one repeat without imposing EOS, identity, or format on semantics."""

    validate_case(case)
    require(isinstance(generated_text, str), "generated_text must be a string")
    require(
        isinstance(prefix_text_by_generated_token, Sequence)
        and not isinstance(prefix_text_by_generated_token, (str, bytes)),
        "prefix decodes must be a sequence",
    )
    require(
        len(prefix_text_by_generated_token) <= PREFIX_TOKEN_BUDGET,
        "prefix decode registry exceeds the frozen token budget",
    )
    require(all(isinstance(value, str) for value in prefix_text_by_generated_token),
            "prefix decode registry contains a non-string")
    target, foil = case["target"], case["foil"]
    full_prefix_owners = prefix_candidate_owners(
        generated_text, case["candidate_groups"]
    )
    resolved_owner: str | None = (
        full_prefix_owners[0] if len(full_prefix_owners) == 1 else None
    )
    resolved_at: int | None = None
    resolution_owners: tuple[str, ...] = full_prefix_owners
    first_ambiguous_at: int | None = None
    for token_index, prefix_text in enumerate(prefix_text_by_generated_token, 1):
        owners = prefix_candidate_owners(prefix_text, case["candidate_groups"])
        if len(owners) > 1 and first_ambiguous_at is None:
            first_ambiguous_at = token_index
        if resolved_owner is not None and owners == (resolved_owner,):
            resolved_at = token_index
            break
    if len(full_prefix_owners) > 1:
        semantic_event = "ambiguous_prefix"
    elif resolved_owner == target and resolved_at is not None:
        semantic_event = "target_prefix"
    elif resolved_owner == foil and resolved_at is not None:
        semantic_event = "foil_prefix"
    elif resolved_owner == target:
        semantic_event = "target_after_prefix_budget"
    elif resolved_owner == foil:
        semantic_event = "foil_after_prefix_budget"
    else:
        semantic_event = "unresolved_prefix"
    mentions = diagnostic_mentions(generated_text, case["candidate_groups"])
    exact_value = exact_short_normalize(generated_text)
    registry = _normalized_candidate_registry(case["candidate_groups"])
    exact_owners = tuple(sorted(
        owner for owner, aliases in registry.items() if exact_value in aliases
    ))
    exact_owner = exact_owners[0] if len(exact_owners) == 1 else None
    return {
        "semantic_normalized_generated": semantic_normalize(generated_text),
        "exact_short_normalized_generated": exact_value,
        "semantic_prefix_owner": resolved_owner,
        "semantic_full_prefix_owners": list(full_prefix_owners),
        "semantic_prefix_resolved_at_generated_token": resolved_at,
        "semantic_prefix_resolution_owners": list(resolution_owners),
        "semantic_prefix_first_ambiguous_at_generated_token": first_ambiguous_at,
        "semantic_event": semantic_event,
        "semantic_correct": resolved_owner == target and resolved_at is not None,
        "exact_short_owner": exact_owner,
        "exact_short_correct": exact_owner == target,
        "mentioned_candidates": list(mentions),
        "foil_mentioned_anywhere": foil in mentions,
        "contradictory_later_candidate_mention": (
            resolved_owner == target and resolved_at is not None and foil in mentions
        ),
    }


def validate_case_registry(cases: Sequence[Mapping[str, Any]]) -> None:
    require(len(cases) == EXPECTED_CASES, "development case denominator drift")
    ids = [case.get("case_id") for case in cases]
    require(len(ids) == len(set(ids)), "duplicate case_id")
    for case in cases:
        validate_case(case)
    units: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        units[case["analysis_unit_id"]].append(case)
    require(len(units) == EXPECTED_UNITS, "analysis-unit denominator drift")
    family_counts: Counter[tuple[str, str]] = Counter()
    for unit_id, bank in units.items():
        signatures = {(row["relation"], row["interface"]) for row in bank}
        require(len(signatures) == 1, f"{unit_id}: mixed family")
        relation, interface = next(iter(signatures))
        family_counts[(relation, interface)] += 1
        if interface == "direct":
            require(len(bank) == 6, f"{unit_id}: direct denominator drift")
            require({row["surface_id"] for row in bank} == set(range(6)),
                    f"{unit_id}: direct surface grid drift")
        else:
            require(len(bank) == 16, f"{unit_id}: selection denominator drift")
            observed = Counter((row["query_polarity"] for row in bank))
            require(observed == {"positive": 8, "negative": 8},
                    f"{unit_id}: selection polarity grid drift")
            grid = {
                (row["surface_id"], row["order"], row["query_polarity"])
                for row in bank
            }
            require(grid == {
                (surface, order, polarity)
                for surface in range(4)
                for order in (0, 1)
                for polarity in ("positive", "negative")
            }, f"{unit_id}: selection factorial grid drift")
    require(family_counts == {
        ("fruit_membership", "direct"): 12,
        ("citrus_membership", "direct"): 12,
        ("fruit_membership", "selection"): 6,
        ("citrus_membership", "selection"): 6,
    }, "analysis-unit family denominator drift")


def _validate_repeat_observation(
    case: Mapping[str, Any], observed: Mapping[str, Any], model: str, repeat: str,
) -> dict[str, Any]:
    require(observed.get("schema_version") == "phase578_development_behavior_row.v1",
            "behavior row schema drift")
    require(observed.get("phase_id") == PHASE, "behavior row phase drift")
    require(observed.get("model") == model, "behavior row model drift")
    require(observed.get("split") == "development", "behavior row split drift")
    require(observed.get("execution_repeat") == repeat, "behavior repeat drift")
    require(observed.get("case_id") == case["case_id"], "behavior case drift")
    if "analysis_unit_id" in observed:
        require(observed.get("analysis_unit_id") == case["analysis_unit_id"],
                "behavior unit drift")
    require(observed.get("observer_only") is True, "row is not observer-only")
    require(observed.get("activation_collected") is False, "activation access forbidden")
    require(observed.get("causal_intervention") is False, "causal intervention forbidden")
    require(observed.get("sealed_model_access") is False, "sealed access forbidden")
    token_ids = observed.get("generated_token_ids_before_eos")
    prefixes = observed.get("prefix_text_by_generated_token")
    require(isinstance(token_ids, list) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in token_ids
    ), "generated token registry is invalid")
    require(isinstance(prefixes, list), "prefix text registry is invalid")
    require(len(prefixes) == min(PREFIX_TOKEN_BUDGET, len(token_ids)),
            "prefix text registry length drift")
    full_suffix = observed.get("full_generated_suffix_token_ids")
    require(isinstance(full_suffix, list) and bool(full_suffix) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in full_suffix
    ), "full generated suffix registry is invalid")
    require(isinstance(observed.get("eos_seen"), bool), "EOS flag is invalid")
    require(isinstance(observed.get("budget_truncated"), bool),
            "budget flag is invalid")
    rebuilt = classify_observation(case, observed.get("generated_text"), prefixes)
    # The GPU runner is raw-only.  If a later container carries scorer fields,
    # they must agree exactly, but their absence is the normal frozen path.
    for key, expected in rebuilt.items():
        if key in observed:
            require(observed.get(key) == expected, f"row scorer drift: {key}")
    return {**dict(observed), **rebuilt}


def _unit_subgroup(case: Mapping[str, Any]) -> str | None:
    if case["relation"] != "fruit_membership":
        return None
    if case["interface"] == "direct":
        return "nonfruit_food" if case.get("focus_object_class") == "nonfruit_food" else None
    negative = case.get("negative_object")
    if negative == case.get("focus_object"):
        negative_class = case.get("focus_object_class")
    elif negative == case.get("comparison_object"):
        negative_class = case.get("comparison_object_class")
    else:
        raise ScoringError("selection negative-object role cannot be reconstructed")
    return "nonfruit_food" if negative_class == "nonfruit_food" else None


def score_model(
    cases: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Recompute the complete Phase577 development behavior decision."""

    validate_case_registry(cases)
    require(isinstance(model, str) and bool(model), "model name is invalid")
    case_by_id = {case["case_id"]: case for case in cases}
    expected_keys = {
        (case_id, repeat) for case_id in case_by_id for repeat in REPEATS
    }
    actual_keys = [
        (row.get("case_id"), row.get("execution_repeat")) for row in observations
    ]
    require(len(actual_keys) == len(set(actual_keys)), "duplicate behavior row")
    require(set(actual_keys) == expected_keys, "case x repeat registry is not exact")
    observed_by_key = {
        (row["case_id"], row["execution_repeat"]): row for row in observations
    }
    rebuilt_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for case_id, case in case_by_id.items():
        for repeat in REPEATS:
            rebuilt_by_key[(case_id, repeat)] = _validate_repeat_observation(
                case, observed_by_key[(case_id, repeat)], model, repeat
            )

    case_reports: dict[str, Any] = {}
    for case_id, case in case_by_id.items():
        repeats = [rebuilt_by_key[(case_id, repeat)] for repeat in REPEATS]
        stable_semantic = all(row["semantic_correct"] is True for row in repeats)
        full_identity = all(
            repeats[0][field] == repeats[1][field]
            for field in (
                "semantic_normalized_generated",
                "generated_token_ids_before_eos",
                "first_eos_token_id",
                "full_generated_suffix_token_ids",
            )
        )
        exact_contract_stable = (
            all(row["exact_short_correct"] is True for row in repeats)
            if case["output_contract"] == "exact_short" else None
        )
        case_reports[case_id] = {
            "case_id": case_id,
            "analysis_unit_id": case["analysis_unit_id"],
            "relation": case["relation"],
            "interface": case["interface"],
            "output_contract": case["output_contract"],
            "query_polarity": case["query_polarity"],
            "target_truth_polarity": case["target_truth_polarity"],
            "surface_id": case["surface_id"],
            "paraphrase_id": case["paraphrase_id"],
            "order": case["order"],
            "focus_object_class": case["focus_object_class"],
            "comparison_object_class": case.get("comparison_object_class"),
            "semantic_stable_both_repeats": stable_semantic,
            "full_generated_identity": full_identity,
            "exact_short_stable_both_repeats": exact_contract_stable,
            "both_repeats_eos": all(row["eos_seen"] is True for row in repeats),
            "either_repeat_budget_truncated": any(
                row["budget_truncated"] is True for row in repeats
            ),
            "either_repeat_contradictory_later_mention": any(
                row["contradictory_later_candidate_mention"] is True
                for row in repeats
            ),
        }

    cases_by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_unit[case["analysis_unit_id"]].append(case)
    unit_reports: dict[str, Any] = {}
    family_passed: Counter[str] = Counter()
    nonfruit_food_passed = {"direct": 0, "selection": 0}
    nonfruit_food_total = {"direct": 0, "selection": 0}
    for unit_id, bank in sorted(cases_by_unit.items()):
        relation, interface = bank[0]["relation"], bank[0]["interface"]
        stable_count = sum(
            case_reports[row["case_id"]]["semantic_stable_both_repeats"]
            for row in bank
        )
        polarity_counts: dict[str, int] = {}
        if interface == "direct":
            gate_parts = {"stable_at_least_5_of_6": stable_count >= 5}
        else:
            polarity_counts = {
                polarity: sum(
                    case_reports[row["case_id"]]["semantic_stable_both_repeats"]
                    for row in bank if row["query_polarity"] == polarity
                )
                for polarity in ("positive", "negative")
            }
            gate_parts = {
                "stable_at_least_14_of_16": stable_count >= 14,
                "positive_at_least_7_of_8": polarity_counts["positive"] >= 7,
                "negative_at_least_7_of_8": polarity_counts["negative"] >= 7,
            }
        passed = all(gate_parts.values())
        family = f"{relation}|{interface}"
        family_passed[family] += int(passed)
        subgroup = _unit_subgroup(bank[0])
        if subgroup == "nonfruit_food":
            nonfruit_food_total[interface] += 1
            nonfruit_food_passed[interface] += int(passed)
        unit_reports[unit_id] = {
            "analysis_unit_id": unit_id,
            "relation": relation,
            "interface": interface,
            "case_denominator": len(bank),
            "semantic_stable_case_count": stable_count,
            "semantic_stable_case_rate": stable_count / len(bank),
            "semantic_stable_by_query_polarity": polarity_counts,
            "gate_parts": gate_parts,
            "unit_pass": passed,
            "subgroup": subgroup,
        }

    stable_case_count = sum(
        report["semantic_stable_both_repeats"] for report in case_reports.values()
    )
    passing_units = sum(report["unit_pass"] for report in unit_reports.values())
    family_gate_parts = {
        family: family_passed[family] >= minimum
        for family, minimum in GATE["family_minimums"].items()
    }
    model_gate_parts = {
        "all_four_family_minimums": all(family_gate_parts.values()),
        "passing_units_at_least_30_of_36": passing_units >= 30,
        "fruit_direct_nonfruit_food_2_of_2": (
            nonfruit_food_total["direct"] == 2
            and nonfruit_food_passed["direct"] == 2
        ),
        "fruit_selection_nonfruit_food_2_of_2": (
            nonfruit_food_total["selection"] == 2
            and nonfruit_food_passed["selection"] == 2
        ),
        "semantic_stable_case_micro_rate_at_least_0_85": (
            stable_case_count * 100 >= 85 * len(cases)
        ),
    }

    exact_cases = [
        report for report in case_reports.values()
        if report["output_contract"] == "exact_short"
    ]
    strata: dict[str, Any] = {}
    for relation in RELATIONS:
        for interface in INTERFACES:
            bank = [
                report for report in case_reports.values()
                if report["relation"] == relation and report["interface"] == interface
            ]
            strata[f"{relation}|{interface}"] = {
                "case_count": len(bank),
                "semantic_stable_count": sum(
                    report["semantic_stable_both_repeats"] for report in bank
                ),
                "full_generated_identity_count": sum(
                    report["full_generated_identity"] for report in bank
                ),
            }
    diagnostic_strata: dict[str, Any] = {}

    def add_stratum(name: str, bank: list[dict[str, Any]]) -> None:
        diagnostic_strata[name] = {
            "case_count": len(bank),
            "semantic_stable_count": sum(
                report["semantic_stable_both_repeats"] for report in bank
            ),
            "full_generated_identity_count": sum(
                report["full_generated_identity"] for report in bank
            ),
            "exact_short_eligible_count": sum(
                report["output_contract"] == "exact_short" for report in bank
            ),
            "exact_short_stable_count": sum(
                report["exact_short_stable_both_repeats"] is True for report in bank
            ),
        }

    reports_list = list(case_reports.values())
    for contract in ("semantic_label_first", "exact_short"):
        add_stratum(
            f"output_contract|{contract}",
            [row for row in reports_list if row["output_contract"] == contract],
        )
    for relation in RELATIONS:
        for interface in INTERFACES:
            surfaces = range(6) if interface == "direct" else range(4)
            for surface_id in surfaces:
                add_stratum(
                    f"surface|{relation}|{interface}|{surface_id}",
                    [row for row in reports_list if row["relation"] == relation
                     and row["interface"] == interface
                     and row["surface_id"] == surface_id],
                )
    for order in (0, 1):
        add_stratum(
            f"selection_order|{order}",
            [row for row in reports_list if row["interface"] == "selection"
             and row["order"] == order],
        )
    for polarity in ("affirmative", "positive", "negative"):
        add_stratum(
            f"query_polarity|{polarity}",
            [row for row in reports_list if row["query_polarity"] == polarity],
        )
    for object_class in sorted({
        row["focus_object_class"] for row in reports_list
    }):
        add_stratum(
            f"focus_object_class|{object_class}",
            [row for row in reports_list
             if row["focus_object_class"] == object_class],
        )
    decision = {
        "schema_version": "phase578_development_behavior_decision.v1",
        "phase_id": PHASE,
        "source_behavior_protocol_phase": "Phase577",
        "model": model,
        "split": "development",
        "primary_metric": "both_repeats_semantic_prefix_correct",
        "case_count": len(cases),
        "repeat_row_count": len(observations),
        "analysis_unit_count": len(unit_reports),
        "semantic_stable_case_count": stable_case_count,
        "semantic_stable_case_micro_rate": stable_case_count / len(cases),
        "passing_analysis_units": passing_units,
        "family_passing_units": dict(sorted(family_passed.items())),
        "family_gate_parts": family_gate_parts,
        "nonfruit_food_units": {
            "passed": nonfruit_food_passed,
            "total": nonfruit_food_total,
        },
        "model_gate_parts": model_gate_parts,
        "behavior_gate_pass": all(model_gate_parts.values()),
        "exact_short_case_count": len(exact_cases),
        "exact_short_stable_case_count": sum(
            report["exact_short_stable_both_repeats"] is True
            for report in exact_cases
        ),
        "full_generated_identity_case_count": sum(
            report["full_generated_identity"] for report in case_reports.values()
        ),
        "both_repeats_eos_case_count": sum(
            report["both_repeats_eos"] for report in case_reports.values()
        ),
        "either_repeat_budget_truncated_case_count": sum(
            report["either_repeat_budget_truncated"] for report in case_reports.values()
        ),
        "either_repeat_contradictory_later_mention_case_count": sum(
            report["either_repeat_contradictory_later_mention"]
            for report in case_reports.values()
        ),
        "strata": strata,
        "surface_order_contract_and_object_class_diagnostics": diagnostic_strata,
        "unit_reports": unit_reports,
        "case_reports": case_reports,
        "statistical_independence_claimed": False,
        "mechanism_claim_authorized": False,
        "internal_activation_accessed": False,
    }
    decision["decision_payload_sha256"] = sha256_json(decision)
    return decision


def _mock_observations(
    cases: Sequence[Mapping[str, Any]],
    incorrect_case_ids: set[str] | None = None,
    diagnostic_only_failure: bool = False,
) -> list[dict[str, Any]]:
    incorrect = incorrect_case_ids or set()
    output = []
    for case in cases:
        answer = case["foil"] if case["case_id"] in incorrect else case["target"]
        generated = answer if not diagnostic_only_failure else f"{answer}, because this is synthetic."
        for repeat_index, repeat in enumerate(REPEATS):
            token_id = 100 + repeat_index if diagnostic_only_failure else 100
            output.append({
                "schema_version": "phase578_development_behavior_row.v1",
                "phase_id": PHASE,
                "model": "fixture_model",
                "split": "development",
                "execution_repeat": repeat,
                "case_id": case["case_id"],
                "generated_text": generated,
                "prefix_text_by_generated_token": [answer],
                "generated_token_ids_before_eos": [token_id],
                "full_generated_suffix_token_ids": [token_id],
                "first_eos_token_id": None,
                "eos_seen": False if diagnostic_only_failure else True,
                "budget_truncated": True if diagnostic_only_failure else False,
                "observer_only": True,
                "activation_collected": False,
                "causal_intervention": False,
                "sealed_model_access": False,
            })
    return output


def gate_self_test(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pin the unit/family/subgroup/micro conjunction on the frozen grid."""

    validate_case_registry(cases)
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_unit[case["analysis_unit_id"]].append(case)
    direct_unit = next(
        bank for bank in by_unit.values() if bank[0]["interface"] == "direct"
        and bank[0]["relation"] == "citrus_membership"
    )
    selection_unit = next(
        bank for bank in by_unit.values() if bank[0]["interface"] == "selection"
        and bank[0]["relation"] == "citrus_membership"
    )
    nonfruit_direct = next(
        bank for bank in by_unit.values()
        if bank[0]["interface"] == "direct"
        and bank[0]["relation"] == "fruit_membership"
        and bank[0].get("focus_object_class") == "nonfruit_food"
    )
    all_correct = score_model(cases, _mock_observations(cases), "fixture_model")
    direct_one_bad = score_model(
        cases,
        _mock_observations(cases, {direct_unit[0]["case_id"]}),
        "fixture_model",
    )
    direct_two_bad = score_model(
        cases,
        _mock_observations(cases, {row["case_id"] for row in direct_unit[:2]}),
        "fixture_model",
    )
    selection_balanced_two_bad = {
        next(row["case_id"] for row in selection_unit if row["query_polarity"] == polarity)
        for polarity in ("positive", "negative")
    }
    selection_same_polarity_two_bad = {
        row["case_id"] for row in selection_unit
        if row["query_polarity"] == "positive"
    }
    selection_same_polarity_two_bad = set(sorted(selection_same_polarity_two_bad)[:2])
    selection_balanced = score_model(
        cases, _mock_observations(cases, selection_balanced_two_bad), "fixture_model"
    )
    selection_imbalanced = score_model(
        cases, _mock_observations(cases, selection_same_polarity_two_bad), "fixture_model"
    )
    subgroup_bad_ids = {row["case_id"] for row in nonfruit_direct[:2]}
    subgroup_bad = score_model(
        cases, _mock_observations(cases, subgroup_bad_ids), "fixture_model"
    )
    diagnostics_bad = score_model(
        cases, _mock_observations(cases, diagnostic_only_failure=True), "fixture_model"
    )
    direct_id = direct_unit[0]["analysis_unit_id"]
    selection_id = selection_unit[0]["analysis_unit_id"]
    tests = {
        "all_correct_passes": all_correct["behavior_gate_pass"] is True,
        "direct_5_of_6_passes_unit": direct_one_bad["unit_reports"][direct_id][
            "unit_pass"
        ] is True,
        "direct_4_of_6_fails_unit": direct_two_bad["unit_reports"][direct_id][
            "unit_pass"
        ] is False,
        "selection_14_with_7_and_7_passes_unit": selection_balanced[
            "unit_reports"
        ][selection_id]["unit_pass"] is True,
        "selection_14_with_6_and_8_fails_unit": selection_imbalanced[
            "unit_reports"
        ][selection_id]["unit_pass"] is False,
        "micro_286_of_336_passes": 286 * 100 >= 85 * 336,
        "micro_285_of_336_fails": not (285 * 100 >= 85 * 336),
        "nonfruit_subgroup_failure_blocks_total": (
            subgroup_bad["family_gate_parts"]["fruit_membership|direct"] is True
            and subgroup_bad["model_gate_parts"][
                "fruit_direct_nonfruit_food_2_of_2"
            ] is False
            and subgroup_bad["behavior_gate_pass"] is False
        ),
        "semantic_gate_ignores_exact_eos_and_identity_diagnostics": (
            diagnostics_bad["behavior_gate_pass"] is True
            and diagnostics_bad["exact_short_stable_case_count"] == 0
            and diagnostics_bad["full_generated_identity_case_count"] == 0
            and diagnostics_bad["both_repeats_eos_case_count"] == 0
        ),
    }
    require(all(tests.values()), f"Phase578 gate self-test failed: {tests}")
    return {"passed": True, "test_count": len(tests), "tests": tests}


def _fixture_case(
    *, target: str = "yes", foil: str = "no",
    groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    return {
        "phase_id": "Phase577",
        "schema_version": "phase577_gpt5_natural_behavior_case.v2",
        "split": "development", "sealed": False,
        "case_id": "fixture", "analysis_unit_id": "fixture_unit",
        "relation": "fruit_membership", "interface": "direct",
        "surface_id": 0, "order": None, "query_polarity": "affirmative",
        "output_contract": "semantic_label_first",
        "candidate_groups": groups or {"yes": ["yes"], "no": ["no"]},
        "target": target, "foil": foil,
    }


def self_test(cases: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Adversarially pin every consequential parser boundary."""

    yes_case = _fixture_case()
    multi_case = _fixture_case(
        target="green apple", foil="red pear",
        groups={"green apple": ["green apple"], "red pear": ["red pear"]},
    )
    tests: dict[str, bool] = {}

    def classify(case: Mapping[str, Any], full: str, prefixes: Sequence[str]) -> dict[str, Any]:
        return classify_observation(case, full, prefixes)

    tests["target_plain"] = classify(yes_case, "YES", ["YES"])["semantic_correct"]
    tests["target_casefold_nfkc"] = classify(
        yes_case, "ＹＥＳ", ["ＹＥＳ"]
    )["semantic_correct"]
    tests["allowed_quote_bracket_bullet_prefix"] = all(
        classify(yes_case, text, [text])["semantic_correct"]
        for text in ('"YES"', "[YES]", "- YES", "• YES")
    )
    tests["answer_colon_rejected"] = not classify(
        yes_case, "Answer: YES", ["Answer: YES"]
    )["semantic_correct"]
    tests["word_continuation_rejected"] = not classify(
        yes_case, "yesterday", ["yesterday"]
    )["semantic_correct"]
    tests["digit_and_underscore_continuations_rejected"] = all(
        not classify(yes_case, text, [text])["semantic_correct"]
        for text in ("YES2", "YES_name")
    )
    tests["symbol_is_not_a_frozen_boundary"] = not classify(
        yes_case, "YES✅", ["YES✅"]
    )["semantic_correct"]
    tests["foil_prefix_incorrect"] = classify(
        yes_case, "NO", ["NO"]
    )["semantic_event"] == "foil_prefix"
    tests["target_then_foil_keeps_primary_but_flags"] = all((
        classify(yes_case, "YES, not NO", ["YES"])["semantic_correct"],
        classify(yes_case, "YES, not NO", ["YES"])[
            "contradictory_later_candidate_mention"
        ],
    ))
    tests["eight_token_inclusive"] = classify(
        yes_case, "YES", ["", "", "", "", "", "", "", "YES"]
    )["semantic_correct"]
    after_budget = classify(yes_case, "YES", [""] * 8)
    tests["target_after_eight_token_budget_fails"] = (
        not after_budget["semantic_correct"]
        and after_budget["semantic_event"] == "target_after_prefix_budget"
    )
    tests["budget_registry_rejected_at_nine"] = False
    try:
        classify(yes_case, "YES", [""] * 8 + ["YES"])
    except ScoringError:
        tests["budget_registry_rejected_at_nine"] = True
    tests["multiword_incomplete_rejected"] = not classify(
        multi_case, "green", ["green"]
    )["semantic_correct"]
    tests["multiword_complete_accepted"] = classify(
        multi_case, "green apple.", ["green", "green apple"]
    )["semantic_correct"]
    tests["multiword_hyphen_rejected"] = not classify(
        multi_case, "green-apple", ["green-apple"]
    )["semantic_correct"]
    tests["empty_output_unresolved"] = classify(
        yes_case, "", []
    )["semantic_event"] == "unresolved_prefix"
    overlap_case = _fixture_case(
        target="red", foil="red apple",
        groups={"red": ["red"], "red apple": ["red apple"]},
    )
    overlap = classify(overlap_case, "red apple", ["red apple"])
    tests["overlapping_candidates_ambiguous"] = (
        overlap["semantic_event"] == "ambiguous_prefix"
        and not overlap["semantic_correct"]
    )
    tests["exact_terminal_punctuation_accepted"] = classify(
        yes_case, "(YES!)", ["(YES!)"]
    )["exact_short_correct"]
    tests["exact_explanation_rejected"] = not classify(
        yes_case, "YES, because it is.", ["YES"]
    )["exact_short_correct"]
    tests["duplicate_alias_owner_rejected"] = False
    try:
        _normalized_candidate_registry({"a": ["same"], "b": ["ＳＡＭＥ"]})
    except ScoringError:
        tests["duplicate_alias_owner_rejected"] = True
    tests["target_and_foil_must_differ"] = False
    try:
        validate_case(_fixture_case(target="yes", foil="yes"))
    except ScoringError:
        tests["target_and_foil_must_differ"] = True

    require(all(tests.values()), f"Phase578 scorer self-test failed: {tests}")
    result = {
        "schema_version": "phase578_scorer_self_test.v1",
        "phase_id": PHASE,
        "passed": True,
        "test_count": len(tests),
        "tests": tests,
        "prefix_token_budget": PREFIX_TOKEN_BUDGET,
        "leading_ignored_characters_codepoints": sorted(
            ord(character) for character in LEADING_IGNORED_CHARACTERS
        ),
        "gate": GATE,
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written": False,
    }
    if cases is not None:
        gate_report = gate_self_test(cases)
        result["gate_self_test"] = gate_report
        result["test_count"] += gate_report["test_count"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--case-file", type=Path)
    args = parser.parse_args()
    if not args.self_test:  # pragma: no cover - argparse enforces this
        raise ScoringError("--self-test is required")
    cases = None
    if args.case_file is not None:
        with args.case_file.open("r", encoding="utf-8") as handle:
            cases = [json.loads(line) for line in handle if line.strip()]
    print(json.dumps(self_test(cases), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
