#!/usr/bin/env python3
"""Freeze Phase403's finite natural-future predictive-state denominator.

The protocol deliberately distinguishes finite-panel predictive equivalence
from an interventional causal state.  State labels come from an enumerated
micro-world, not from clustering hidden activations or fitting a probe.
"""

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
from phase386_multitime_protocol import NAMES, NOUNS, OBJECTS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase403_predictive_state"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_binding", "rule_reasoning", "grammar_constraint")
SPLIT_GROUP_COUNTS = {
    "discovery": 8,
    "calibration": 4,
    "behavioral_holdout": 4,
}
FROZEN_DTYPES = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}
STATE_VARIANTS = (0, 1)
SURFACE_REPLICAS = (
    {"surface_id": "r000", "lexical": 0, "syntax": 0, "order": 0},
    {"surface_id": "r011", "lexical": 0, "syntax": 1, "order": 1},
    {"surface_id": "r101", "lexical": 1, "syntax": 0, "order": 1},
    {"surface_id": "r110", "lexical": 1, "syntax": 1, "order": 0},
)
QUERIES = {
    "knowledge_binding": ("value_of_a", "value_of_b", "same_value"),
    "rule_reasoning": ("one_step_holder", "two_step_holder", "a_can_enter"),
    "grammar_constraint": ("be_auxiliary", "have_auxiliary", "demonstrative"),
}
CONTEXTS = {
    "knowledge_binding": (
        ("base", "base"),
        ("swap", "single"),
        ("copy_a_to_b", "single"),
        ("swap_then_copy", "composition"),
        ("copy_then_swap", "composition"),
    ),
    "rule_reasoning": (
        ("base", "base"),
        ("swap_holder", "single"),
        ("set_a_holder", "single"),
        ("swap_then_set_a", "composition"),
        ("set_a_then_swap", "composition"),
    ),
    "grammar_constraint": (
        ("base", "base"),
        ("toggle_number", "single"),
        ("set_past", "single"),
        ("number_then_past", "composition"),
    ),
}
REASONING_TERMS = (
    ("key", "badge", "vault"),
    ("permit", "seal", "archive"),
)
SCHEMA_VERSION = "77.0.0"
PROTOCOL_AMENDMENT = "001-explicit-finite-answer-set"


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
    raise ValueError(f"Invalid Phase403 group priority: {priority}")


def pick_distinct(pool: tuple[Any, ...], seed: int, count: int, stride: int) -> tuple[Any, ...]:
    values = tuple(pool[(seed + offset * stride) % len(pool)] for offset in range(count))
    if len(set(values)) != count:
        raise ValueError("lexical_collision")
    return values


def package_for(family: str, group_priority: int, lexical: int) -> dict[str, Any]:
    seed = 403000 + 97 * group_priority + 37 * lexical + 10000 * FAMILIES.index(family)
    if family == "knowledge_binding":
        entity_a, entity_b = pick_distinct(tuple(NAMES), seed, 2, 11)
        value_0, value_1 = pick_distinct(tuple(OBJECTS), seed * 3, 2, 13)
        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "value_0": value_0,
            "value_1": value_1,
        }
    if family == "rule_reasoning":
        entity_a, entity_b = pick_distinct(tuple(NAMES), seed, 2, 17)
        trigger, intermediate, terminal = REASONING_TERMS[lexical]
        return {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "trigger": trigger,
            "intermediate": intermediate,
            "terminal": terminal,
        }
    noun_singular, noun_plural = NOUNS[2 * group_priority + lexical]
    return {"noun_singular": noun_singular, "noun_plural": noun_plural}


def knowledge_state(state_variant: int, context: str) -> tuple[int, int]:
    state = (0, 1) if state_variant == 0 else (1, 0)
    if context == "base":
        return state
    if context == "swap":
        return state[1], state[0]
    if context == "copy_a_to_b":
        return state[0], state[0]
    if context == "swap_then_copy":
        swapped = (state[1], state[0])
        return swapped[0], swapped[0]
    if context == "copy_then_swap":
        copied = (state[0], state[0])
        return copied[1], copied[0]
    raise KeyError(context)


def reasoning_state(state_variant: int, context: str) -> int:
    holder = state_variant
    if context == "base":
        return holder
    if context == "swap_holder":
        return 1 - holder
    if context == "set_a_holder":
        return 0
    if context == "swap_then_set_a":
        return 0
    if context == "set_a_then_swap":
        return 1
    raise KeyError(context)


def grammar_state(state_variant: int, context: str) -> tuple[int, int]:
    number = state_variant
    tense = 0
    if context == "base":
        return number, tense
    if context == "toggle_number":
        return 1 - number, tense
    if context == "set_past":
        return number, 1
    if context == "number_then_past":
        return 1 - number, 1
    raise KeyError(context)


def abstract_state(family: str, state_variant: int, context: str) -> tuple[int, ...]:
    if family == "knowledge_binding":
        return knowledge_state(state_variant, context)
    if family == "rule_reasoning":
        return (reasoning_state(state_variant, context),)
    return grammar_state(state_variant, context)


def updates_for(
    family: str,
    package: dict[str, Any],
    state_variant: int,
    context: str,
) -> list[str]:
    if context == "base":
        return []
    if family == "knowledge_binding":
        a, b = package["entity_a"], package["entity_b"]
        swap = f"Exchange the objects currently assigned to {a} and {b}."
        copy = f"Then assign {b} the same object currently assigned to {a}."
        if context == "swap":
            return [swap]
        if context == "copy_a_to_b":
            return [copy.removeprefix("Then ")]
        if context == "swap_then_copy":
            return [swap, copy]
        if context == "copy_then_swap":
            return [copy.removeprefix("Then "), f"Then {swap[0].lower() + swap[1:]}"]
    if family == "rule_reasoning":
        a, b = package["entity_a"], package["entity_b"]
        trigger = package["trigger"]
        swap = f"Exchange which of {a} and {b} holds the {trigger}; exactly one still holds it."
        set_a = f"Then make {a} the only holder of the {trigger}."
        if context == "swap_holder":
            return [swap]
        if context == "set_a_holder":
            return [set_a.removeprefix("Then ")]
        if context == "swap_then_set_a":
            return [swap, set_a]
        if context == "set_a_then_swap":
            return [set_a.removeprefix("Then "), f"Then {swap[0].lower() + swap[1:]}"]
    if family == "grammar_constraint":
        next_number = 1 - state_variant
        noun = package["noun_plural"] if next_number else package["noun_singular"]
        quantity = "two" if next_number else "one"
        number_update = f"Describe {quantity} {noun} instead."
        tense_update = "Then use the past tense instead of the present tense."
        if context == "toggle_number":
            return [number_update]
        if context == "set_past":
            return [tense_update.removeprefix("Then ")]
        if context == "number_then_past":
            return [number_update, tense_update]
    raise KeyError((family, context))


def base_facts(
    family: str,
    package: dict[str, Any],
    state_variant: int,
    syntax: int,
    order: int,
) -> str:
    if family == "knowledge_binding":
        a, b = package["entity_a"], package["entity_b"]
        values = (package["value_0"], package["value_1"])
        state = knowledge_state(state_variant, "base")
        clauses = [f"{a} is assigned the {values[state[0]]}", f"{b} is assigned the {values[state[1]]}"]
        if order:
            clauses.reverse()
        if syntax == 0:
            return "Object ledger:\n- " + "\n- ".join(clauses) + "."
        return "The sealed object ledger says that " + "; meanwhile, ".join(clauses) + "."
    if family == "rule_reasoning":
        a, b = package["entity_a"], package["entity_b"]
        holder = a if state_variant == 0 else b
        nonholder = b if state_variant == 0 else a
        trigger = package["trigger"]
        intermediate = package["intermediate"]
        terminal = package["terminal"]
        facts = [f"{holder} holds the {trigger}", f"{nonholder} does not hold the {trigger}"]
        if order:
            facts.reverse()
        rules = [
            f"Anyone holding the {trigger} receives the {intermediate}",
            f"anyone receiving the {intermediate} may enter the {terminal}",
        ]
        if syntax == 0:
            return "Facts:\n- " + "\n- ".join(facts) + ".\nRules:\n- " + "\n- ".join(rules) + "."
        return "In this closed world, " + "; while ".join(facts) + ". Also, " + ", and ".join(rules) + "."
    number = state_variant
    quantity = "one" if number == 0 else "two"
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    clauses = [f"the record concerns {quantity} {noun}", "the requested description uses the present tense"]
    if order:
        clauses.reverse()
    if syntax == 0:
        return "Grammar scene:\n- " + "\n- ".join(clauses) + "."
    return "For the next completion, " + "; and ".join(clauses) + "."


def expected_answer(
    family: str,
    package: dict[str, Any],
    state_variant: int,
    context: str,
    query: str,
) -> tuple[str, str, list[str]]:
    if family == "knowledge_binding":
        state = knowledge_state(state_variant, context)
        values = (package["value_0"], package["value_1"])
        if query == "value_of_a":
            target = values[state[0]]
            canonical = f"value_{state[0]}"
            return target, canonical, list(values)
        if query == "value_of_b":
            target = values[state[1]]
            canonical = f"value_{state[1]}"
            return target, canonical, list(values)
        target = "yes" if state[0] == state[1] else "no"
        return target, target, ["yes", "no"]
    if family == "rule_reasoning":
        holder = reasoning_state(state_variant, context)
        entities = (package["entity_a"], package["entity_b"])
        if query in {"one_step_holder", "two_step_holder"}:
            return entities[holder], f"entity_{holder}", list(entities)
        target = "yes" if holder == 0 else "no"
        return target, target, ["yes", "no"]
    number, tense = grammar_state(state_variant, context)
    if query == "be_auxiliary":
        target = (("is", "are"), ("was", "were"))[tense][number]
        return target, target, ["is", "are", "was", "were"]
    if query == "have_auxiliary":
        target = ("has", "have")[number] if tense == 0 else "had"
        return target, target, ["has", "have", "had"]
    target = ("this", "these")[number]
    return target, target, ["this", "these"]


def query_text(
    family: str,
    package: dict[str, Any],
    query: str,
    query_style: int,
    state_variant: int,
    context: str,
) -> str:
    if family == "knowledge_binding":
        a, b = package["entity_a"], package["entity_b"]
        choices = f"Choose {package['value_0']} or {package['value_1']}."
        if query == "value_of_a":
            question = f"Which object is finally assigned to {a}?" if query_style == 0 else f"After all updates, name {a}'s object."
            return f"{question} {choices}"
        if query == "value_of_b":
            question = f"Which object is finally assigned to {b}?" if query_style == 0 else f"After all updates, name {b}'s object."
            return f"{question} {choices}"
        question = f"Do {a} and {b} finally have the same object?" if query_style == 0 else "Are the two final object assignments identical?"
        return f"Answer yes or no: {question}"
    if family == "rule_reasoning":
        a, b = package["entity_a"], package["entity_b"]
        intermediate, terminal = package["intermediate"], package["terminal"]
        choices = f"Choose {a} or {b}."
        if query == "one_step_holder":
            question = f"Who finally receives the {intermediate}?" if query_style == 0 else f"Name the final {intermediate} recipient."
            return f"{question} {choices}"
        if query == "two_step_holder":
            question = f"Who may finally enter the {terminal}?" if query_style == 0 else f"Name the person who can enter the {terminal} after both rules."
            return f"{question} {choices}"
        question = f"May {a} finally enter the {terminal}?" if query_style == 0 else f"After both rules, can {a} enter the {terminal}?"
        return f"Answer yes or no: {question}"
    number, _tense = grammar_state(state_variant, context)
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    if query == "be_auxiliary":
        question = f"The {noun} ___ ready." if query_style == 0 else f"The {noun} ___ prepared."
        return f"Choose is, are, was, or were to fill the blank: {question}"
    if query == "have_auxiliary":
        question = f"The {noun} ___ arrived." if query_style == 0 else f"The {noun} ___ entered."
        return f"Choose has, have, or had to fill the blank: {question}"
    question = f"___ {noun} are recorded." if number else f"___ {noun} is recorded."
    return f"Choose this or these to fill the blank: {question}"


def raw_case(
    family: str,
    group_priority: int,
    state_variant: int,
    surface: dict[str, Any],
    context: str,
    context_kind: str,
    query: str,
) -> dict[str, Any]:
    package = package_for(family, group_priority, surface["lexical"])
    facts = base_facts(
        family,
        package,
        state_variant,
        surface["syntax"],
        surface["order"],
    )
    updates = updates_for(family, package, state_variant, context)
    update_text = ""
    if updates:
        update_text = "\nApply these updates in order:\n" + "\n".join(
            f"{index}. {value}" for index, value in enumerate(updates, 1)
        )
    qtext = query_text(
        family,
        package,
        query,
        surface["syntax"],
        state_variant,
        context,
    )
    target, canonical, candidates = expected_answer(
        family, package, state_variant, context, query
    )
    prefix = facts + update_text
    raw_prompt = (
        f"{prefix}\nFuture branch: {qtext}\n"
        "Use only the final state after all updates. Return exactly one answer from the choices in the future branch and nothing else.\n"
        "Answer:"
    )
    answer_map: dict[str, str] = {}
    if family == "knowledge_binding" and query != "same_value":
        answer_map = {package["value_0"]: "value_0", package["value_1"]: "value_1"}
    elif family == "rule_reasoning" and query != "a_can_enter":
        answer_map = {package["entity_a"]: "entity_0", package["entity_b"]: "entity_1"}
    else:
        answer_map = {value: value for value in candidates}
    return {
        "raw_prompt": raw_prompt,
        "state_prefix": prefix,
        "query_fragment": qtext,
        "target": target,
        "target_aliases": [target],
        "distractors": [value for value in candidates if value.casefold() != target.casefold()],
        "candidate_answers_private": candidates,
        "answer_to_canonical_private": answer_map,
        "expected_canonical_private": canonical,
        "abstract_state_private": list(abstract_state(family, state_variant, context)),
        "operation_context_private": context,
        "context_kind_private": context_kind,
    }


def prior_prompt_hashes() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    path = ROOT / "tests/gpt5/result/phase402_multiparent_graph/protocol/private/phase402_candidate_cases.jsonl"
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result.add((row["private_execution_model"], digest(row["prompt"])))
    return result


def validate_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_groups = len(FAMILIES) * sum(SPLIT_GROUP_COUNTS.values())
    groups = {(row["family_id"], row["parallel_group_id_private"]) for row in rows if row["private_execution_model"] == MODELS[0]}
    if len(groups) != expected_groups:
        raise RuntimeError(f"Phase403 group count {len(groups)} != {expected_groups}")
    duplicates = len(rows) - len({(row["private_execution_model"], digest(row["prompt"])) for row in rows})
    if duplicates:
        raise RuntimeError(f"Phase403 duplicate prompts: {duplicates}")
    for family in FAMILIES:
        for priority in range(sum(SPLIT_GROUP_COUNTS.values())):
            sample = [
                row
                for row in rows
                if row["private_execution_model"] == MODELS[0]
                and row["family_id"] == family
                and row["group_priority"] == priority
                and row["context_kind_private"] == "base"
            ]
            for surface in SURFACE_REPLICAS:
                selected = [row for row in sample if row["surface_id_private"] == surface["surface_id"]]
                if len(selected) != len(STATE_VARIANTS) * len(QUERIES[family]):
                    raise RuntimeError("Phase403 incomplete base fingerprint")
            by_state: dict[int, list[str]] = {}
            for state_variant in STATE_VARIANTS:
                by_state[state_variant] = [
                    next(
                        row["expected_canonical_private"]
                        for row in sample
                        if row["state_variant_private"] == state_variant
                        and row["surface_id_private"] == SURFACE_REPLICAS[0]["surface_id"]
                        and row["future_query_private"] == query
                    )
                    for query in QUERIES[family]
                ]
            if sum(a != b for a, b in zip(by_state[0], by_state[1], strict=True)) < 2:
                raise RuntimeError(f"Phase403 {family} lacks state separation")
    return {
        "group_count": len(groups),
        "prompt_count": len(rows),
        "duplicate_prompt_count": duplicates,
        "base_state_separation_minimum_queries": 2,
    }


def protocol_payload(created_at: str, design_audit: dict[str, Any]) -> dict[str, Any]:
    groups_per_family = sum(SPLIT_GROUP_COUNTS.values())
    discovery_cases_per_model = (
        len(FAMILIES) * SPLIT_GROUP_COUNTS["discovery"] * len(STATE_VARIANTS)
        * len(SURFACE_REPLICAS) * 3 * 3
    )
    calibration_cases_per_model = (
        len(FAMILIES) * SPLIT_GROUP_COUNTS["calibration"] * len(STATE_VARIANTS)
        * len(SURFACE_REPLICAS) * 3 * 3
    )
    holdout_cases_per_model = SPLIT_GROUP_COUNTS["behavioral_holdout"] * len(STATE_VARIANTS) * len(SURFACE_REPLICAS) * 3 * (2 + 2 + 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase403-PredictiveStateProtocol",
        "created_at": created_at,
        "protocol_amendment": PROTOCOL_AMENDMENT,
        "objective": "test_finite_panel_predictive_equivalence_and_natural_operation_tables_before_physical_mapping",
        "phase402_boundary": {
            "four_static_parent_partition_crossmodel_candidates": 0,
            "calibration_consumed": False,
            "physical_holdout_consumed": False,
            "query_local_included_receiver_self_kv": True,
            "mixed_qkv_states_may_be_naturally_unreachable": True,
        },
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "batch_size": 1,
            "padding": "none",
            "attention_implementation": "eager",
            "use_cache": True,
            "do_sample": False,
            "max_new_tokens": 6,
            "runtime_dtype_by_model": FROZEN_DTYPES,
        },
        "finite_world": {
            "families": list(FAMILIES),
            "groups_per_family": groups_per_family,
            "split_group_counts": SPLIT_GROUP_COUNTS,
            "state_variants_per_group": len(STATE_VARIANTS),
            "surface_replicas_per_state": len(SURFACE_REPLICAS),
            "future_queries_per_state": 3,
            "surface_axes": ["lexical_package", "fact_syntax", "fact_order"],
            "surface_design": "four_run_pairwise_balanced_fraction; higher_order_interactions_not_identifiable",
            "query_wording_varies_with_syntax_and_is_not_an_independent_axis": True,
            "every_future_branch_names_a_finite_exhaustive_answer_set": True,
            "all_state_truth_and_transition_tables_are_enumerated": True,
            "hidden_state_clustering_used": False,
            "learned_probe_used": False,
        },
        "sealed_case_counts": {
            "discovery_per_model": discovery_cases_per_model,
            "calibration_per_model_if_all_families_authorized": calibration_cases_per_model,
            "behavioral_holdout_per_model_if_all_families_authorized": holdout_cases_per_model,
            "all_models_all_stages_if_all_families_authorized": (discovery_cases_per_model + calibration_cases_per_model + holdout_cases_per_model) * len(MODELS),
        },
        "future_panel": {
            "anchor_queries": [0, 1],
            "pre_registered_unseen_query": 2,
            "discovery_context_kinds": ["base", "single"],
            "calibration_context_kinds": ["base", "single"],
            "behavioral_holdout_context_kinds": ["composition"],
            "knowledge_and_reasoning_composition_orders": 2,
            "grammar_composition_orders": 1,
        },
        "integer_gates": {
            "surface_fingerprint_pass_min": "3_of_4",
            "base_single_semantic_cases_min_per_group": "63_of_72",
            "discovery_groups_min_per_model_family": "6_of_8",
            "calibration_groups_min_per_model_family": "3_of_4",
            "composition_semantic_cases_min": {
                "knowledge_binding": "42_of_48",
                "rule_reasoning": "42_of_48",
                "grammar_constraint": "21_of_24",
            },
            "behavioral_holdout_groups_min_per_model_family": "3_of_4",
            "crossmodel_gate": "all_three_models_pass_same_family_and_exact_abstract_transition_table",
            "state_blind_baseline_margin_min": 0.20,
        },
        "stage_authorization": {
            "run_discovery_behavior": True,
            "run_calibration_only_for_discovery_crossmodel_candidate_families": True,
            "run_behavioral_holdout_only_for_calibration_crossmodel_candidate_families": True,
            "run_physical_mapping_only_for_behavioral_holdout_crossmodel_families": True,
            "run_limited_causal_intervention_before_physical_map": False,
            "run_head_channel_or_neuron_scan": False,
        },
        "terminology": {
            "finite_panel_equivalence_name": "finite_predictive_state_candidate",
            "finite_panel_equivalence_is_a_causal_state": False,
            "natural_prefix_transition_is_an_internal_operator": False,
            "crossmodel_exact_table_match_name": "finite_functional_table_isomorphism",
            "operator_algebra_claim_authorized": False,
        },
        "stopping_rules": {
            "discovery_failure": "do_not_open_calibration_for_that_family",
            "calibration_failure": "do_not_open_composition_holdout_for_that_family",
            "composition_failure": "register_single_natural_update_behavior_not_composable_operator",
            "state_blind_baseline_not_exceeded": "register_surface_or_answer_prior_not_predictive_state",
            "crossmodel_failure": "do_not_register_public_crossmodel_functional_table",
            "physical_map_not_open": "no_internal_trace_or_neuron_localization",
        },
        "claim_boundary": {
            "behavioral_state_is_internal_mechanism": False,
            "natural_future_observation_is_causal_evidence": False,
            "finite_queries_cover_all_future_behavior": False,
            "successful_table_proves_brain_language_structure": False,
            "negative_result_excludes_larger_models_or_other_state_definitions": False,
        },
        "design_audit": design_audit,
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
    prior_hashes = prior_prompt_hashes()
    current_hashes: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    registries: list[dict[str, Any]] = []
    total_groups = sum(SPLIT_GROUP_COUNTS.values())
    for family in FAMILIES:
        for group_priority in range(total_groups):
            split = split_for(group_priority)
            group_id = "p403g_" + digest(f"{family}:{group_priority}", 24)
            registries.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase403-PredictiveStateProtocol",
                    "family_id": family,
                    "anonymous_parallel_group_id": group_id,
                    "candidate_split": split,
                    "group_priority": group_priority,
                    "state_variant_count": len(STATE_VARIANTS),
                    "surface_replica_count": len(SURFACE_REPLICAS),
                    "future_query_count": len(QUERIES[family]),
                }
            )
            for state_variant in STATE_VARIANTS:
                for surface in SURFACE_REPLICAS:
                    for context, context_kind in CONTEXTS[family]:
                        for query in QUERIES[family]:
                            item = raw_case(
                                family,
                                group_priority,
                                state_variant,
                                surface,
                                context,
                                context_kind,
                                query,
                            )
                            for model in MODELS:
                                tokenizer = tokenizers[model]
                                prompt, add_special, answer_phase = interface_prompt(
                                    tokenizer,
                                    model,
                                    item["raw_prompt"],
                                    "answer_aligned_chat",
                                )
                                prompt_hash = digest(prompt)
                                case_key = (
                                    f"{model}:{family}:{group_id}:{state_variant}:"
                                    f"{surface['surface_id']}:{context}:{query}"
                                )
                                if (model, prompt_hash) in prior_hashes:
                                    raise RuntimeError(
                                        f"Phase403 prompt overlaps Phase402: {case_key}"
                                    )
                                if (model, prompt_hash) in current_hashes:
                                    raise RuntimeError(
                                        "Phase403 duplicate prompt: "
                                        f"{current_hashes[(model, prompt_hash)]} == {case_key}"
                                    )
                                current_hashes[(model, prompt_hash)] = case_key
                                rows.append(
                                    {
                                        "schema_version": SCHEMA_VERSION,
                                        "phase_id": "Phase403-PredictiveStateProtocol",
                                        "created_at": created_at,
                                        "private_execution_model": model,
                                        "blind_case_id": "p403c_" + digest(case_key, 28),
                                        "family_id": family,
                                        "anonymous_parallel_group_id": group_id,
                                        "parallel_group_id_private": f"p403_private_{family}_{group_priority:02d}",
                                        "candidate_split_private": split,
                                        "group_priority": group_priority,
                                        "state_variant_private": state_variant,
                                        "surface_id_private": surface["surface_id"],
                                        "surface_axes_private": surface,
                                        "operation_context_private": context,
                                        "context_kind_private": context_kind,
                                        "future_query_private": query,
                                        "future_query_role_private": "anchor" if query in QUERIES[family][:2] else "pre_registered_unseen",
                                        "prompt": prompt,
                                        "raw_prompt": item["raw_prompt"],
                                        "state_prefix": item["state_prefix"],
                                        "query_fragment": item["query_fragment"],
                                        "tokenization_add_special_tokens": add_special,
                                        "interface": "answer_aligned_chat",
                                        "answer_phase": answer_phase,
                                        "target": item["target"],
                                        "target_aliases": item["target_aliases"],
                                        "distractors": item["distractors"],
                                        "candidate_answers_private": item["candidate_answers_private"],
                                        "answer_to_canonical_private": item["answer_to_canonical_private"],
                                        "expected_canonical_private": item["expected_canonical_private"],
                                        "abstract_state_private": item["abstract_state_private"],
                                        "prompt_token_count": len(
                                            tokenizer(
                                                prompt,
                                                add_special_tokens=add_special,
                                            )["input_ids"]
                                        ),
                                        "formal_denominator": True,
                                    }
                                )
    design_audit = validate_design(rows)
    write_jsonl(OUT / "protocol/private/phase403_all_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase403_blind_group_registry.jsonl", registries)
    payload = protocol_payload(created_at, design_audit)
    write_json(OUT / "phase403_predictive_state_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
