#!/usr/bin/env python3
"""Freeze fresh direct-endpoint states for Phase404.

Phase404 removes imperative update execution from the measurement.  Every
prompt directly states the final finite-world endpoint.  The model response is
measured over an explicit one-token answer set; operation edges remain
generator semantics and are not treated as internal model operators.
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
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402
from phase386_multitime_protocol import NAMES, NOUNS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase404_direct_predictive_state"
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = ("knowledge_binding", "rule_reasoning", "grammar_constraint")
SPLIT_GROUP_COUNTS = {
    "discovery": 8,
    "calibration": 4,
    "behavioral_holdout": 4,
    "physical_holdout": 4,
}
FROZEN_DTYPES = {
    "qwen3": "float16",
    "glm4": "float16",
    "deepseek7b": "bfloat16",
}
SURFACE_REPLICAS = (
    {"surface_id": "r000", "lexical": 0, "syntax": 0, "order": 0},
    {"surface_id": "r011", "lexical": 0, "syntax": 1, "order": 1},
    {"surface_id": "r101", "lexical": 1, "syntax": 0, "order": 1},
    {"surface_id": "r110", "lexical": 1, "syntax": 1, "order": 0},
)
STATE_IDS = {
    "knowledge_binding": ("red_blue", "blue_red", "red_red", "blue_blue"),
    "rule_reasoning": ("holder_a", "holder_b"),
    "grammar_constraint": (
        "singular_present",
        "plural_present",
        "singular_past",
        "plural_past",
    ),
}
QUERIES = {
    "knowledge_binding": ("color_a", "color_b", "same_color"),
    "rule_reasoning": ("one_step_holder", "two_step_holder", "a_can_enter"),
    "grammar_constraint": ("be_auxiliary", "have_auxiliary", "demonstrative"),
}
CANDIDATE_SETS = {
    "knowledge_binding": {
        "color_a": ("red", "blue"),
        "color_b": ("red", "blue"),
        "same_color": ("yes", "no"),
    },
    "rule_reasoning": {
        "one_step_holder": ("A", "B"),
        "two_step_holder": ("A", "B"),
        "a_can_enter": ("yes", "no"),
    },
    "grammar_constraint": {
        "be_auxiliary": ("is", "are", "was", "were"),
        "have_auxiliary": ("has", "have", "had"),
        "demonstrative": ("this", "these"),
    },
}
REASONING_TERMS = (
    ("key", "badge", "vault"),
    ("permit", "seal", "archive"),
)
KNOWLEDGE_ITEMS = ("marker", "token")
SCHEMA_VERSION = "78.0.0"


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


def entity_pair(group_priority: int, lexical: int) -> tuple[str, str]:
    first = (7 * group_priority + 13 * lexical + 3) % len(NAMES)
    second = (first + 11 + 2 * lexical) % len(NAMES)
    if first == second:
        raise ValueError("entity_collision")
    return NAMES[first], NAMES[second]


def package_for(family: str, group_priority: int, lexical: int) -> dict[str, Any]:
    if family == "knowledge_binding":
        a, b = entity_pair(group_priority, lexical)
        return {"entity_a": a, "entity_b": b, "item": KNOWLEDGE_ITEMS[lexical]}
    if family == "rule_reasoning":
        a, b = entity_pair(group_priority + 23, lexical)
        trigger, intermediate, terminal = REASONING_TERMS[lexical]
        return {
            "entity_a": a,
            "entity_b": b,
            "trigger": trigger,
            "intermediate": intermediate,
            "terminal": terminal,
        }
    noun_singular, noun_plural = NOUNS[2 * group_priority + lexical]
    return {"noun_singular": noun_singular, "noun_plural": noun_plural}


def state_truth(family: str, state_id: str) -> tuple[int, ...]:
    if family == "knowledge_binding":
        return {
            "red_blue": (0, 1),
            "blue_red": (1, 0),
            "red_red": (0, 0),
            "blue_blue": (1, 1),
        }[state_id]
    if family == "rule_reasoning":
        return (0,) if state_id == "holder_a" else (1,)
    return {
        "singular_present": (0, 0),
        "plural_present": (1, 0),
        "singular_past": (0, 1),
        "plural_past": (1, 1),
    }[state_id]


def expected_answer(family: str, state_id: str, query: str) -> str:
    truth = state_truth(family, state_id)
    if family == "knowledge_binding":
        if query == "color_a":
            return ("red", "blue")[truth[0]]
        if query == "color_b":
            return ("red", "blue")[truth[1]]
        return "yes" if truth[0] == truth[1] else "no"
    if family == "rule_reasoning":
        if query in {"one_step_holder", "two_step_holder"}:
            return ("A", "B")[truth[0]]
        return "yes" if truth[0] == 0 else "no"
    number, tense = truth
    if query == "be_auxiliary":
        return (("is", "are"), ("was", "were"))[tense][number]
    if query == "have_auxiliary":
        return ("has", "have")[number] if tense == 0 else "had"
    return ("this", "these")[number]


def facts_for(
    family: str,
    package: dict[str, Any],
    state_id: str,
    syntax: int,
    order: int,
) -> str:
    truth = state_truth(family, state_id)
    if family == "knowledge_binding":
        a, b = package["entity_a"], package["entity_b"]
        item = package["item"]
        colors = ("red", "blue")
        clauses = [
            f"person A ({a}) has a {colors[truth[0]]} {item}",
            f"person B ({b}) has a {colors[truth[1]]} {item}",
        ]
        if order:
            clauses.reverse()
        if syntax == 0:
            return "Final color record:\n- " + "\n- ".join(clauses) + "."
        return "In the final record, " + "; meanwhile, ".join(clauses) + "."
    if family == "rule_reasoning":
        a, b = package["entity_a"], package["entity_b"]
        holder_label = "A" if truth[0] == 0 else "B"
        other_label = "B" if truth[0] == 0 else "A"
        holder_name = a if truth[0] == 0 else b
        other_name = b if truth[0] == 0 else a
        trigger = package["trigger"]
        intermediate = package["intermediate"]
        terminal = package["terminal"]
        clauses = [
            f"person {holder_label} ({holder_name}) holds the {trigger}",
            f"person {other_label} ({other_name}) does not hold the {trigger}",
        ]
        if order:
            clauses.reverse()
        rules = [
            f"a person holding the {trigger} receives the {intermediate}",
            f"a person receiving the {intermediate} may enter the {terminal}",
        ]
        if syntax == 0:
            return "Final facts:\n- " + "\n- ".join(clauses) + ".\nRules:\n- " + "\n- ".join(rules) + "."
        return "In this final closed world, " + "; while ".join(clauses) + ". Also, " + ", and ".join(rules) + "."
    number, tense = truth
    quantity = "one" if number == 0 else "two"
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    tense_word = "present" if tense == 0 else "past"
    clauses = [
        f"the final subject is {quantity} {noun}",
        f"the required tense is {tense_word}",
    ]
    if order:
        clauses.reverse()
    if syntax == 0:
        return "Final grammar record:\n- " + "\n- ".join(clauses) + "."
    return "For the final completion, " + "; and ".join(clauses) + "."


def query_text(
    family: str,
    package: dict[str, Any],
    state_id: str,
    query: str,
    syntax: int,
) -> str:
    if family == "knowledge_binding":
        item = package["item"]
        if query == "color_a":
            return f"Choose red or blue: What is the final color of person A's {item}?"
        if query == "color_b":
            return f"Choose red or blue: What is the final color of person B's {item}?"
        return "Choose yes or no: Do person A and person B have the same final color?"
    if family == "rule_reasoning":
        intermediate, terminal = package["intermediate"], package["terminal"]
        if query == "one_step_holder":
            return f"Choose A or B: Which person receives the {intermediate}?"
        if query == "two_step_holder":
            return f"Choose A or B: Which person may enter the {terminal} after both rules?"
        return f"Choose yes or no: May person A enter the {terminal} after both rules?"
    number, _tense = state_truth(family, state_id)
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    if query == "be_auxiliary":
        sentence = f"The {noun} ___ ready." if syntax == 0 else f"The {noun} ___ prepared."
        return f"Choose is, are, was, or were: {sentence}"
    if query == "have_auxiliary":
        sentence = f"The {noun} ___ arrived." if syntax == 0 else f"The {noun} ___ entered."
        return f"Choose has, have, or had: {sentence}"
    sentence = f"___ {noun} is recorded." if number == 0 else f"___ {noun} are recorded."
    return f"Choose this or these: {sentence}"


def raw_case(
    family: str,
    group_priority: int,
    state_id: str,
    surface: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    package = package_for(family, group_priority, surface["lexical"])
    facts = facts_for(
        family,
        package,
        state_id,
        surface["syntax"],
        surface["order"],
    )
    question = query_text(
        family, package, state_id, query, surface["syntax"]
    )
    target = expected_answer(family, state_id, query)
    candidates = CANDIDATE_SETS[family][query]
    raw_prompt = (
        f"{facts}\nFuture branch: {question}\n"
        "Use only the directly stated final record. Return exactly one listed answer and nothing else.\n"
        "Answer:"
    )
    return {
        "raw_prompt": raw_prompt,
        "state_prefix": facts,
        "query_fragment": question,
        "target": target,
        "candidate_answers_private": list(candidates),
        "abstract_state_private": list(state_truth(family, state_id)),
    }


def previous_prompt_hashes() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for path in (
        ROOT / "tests/gpt5/result/phase403_predictive_state/protocol/private/phase403_all_cases.jsonl",
        ROOT / "tests/gpt5/result/phase403_predictive_state_pre_amendment_001/protocol/private/phase403_all_cases.jsonl",
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result.add((row["private_execution_model"], digest(row["prompt"])))
    return result


def semantic_transition_table() -> dict[str, list[dict[str, Any]]]:
    return {
        "knowledge_binding": [
            {"source": "red_blue", "operation": "swap", "target": "blue_red"},
            {"source": "blue_red", "operation": "swap", "target": "red_blue"},
            {"source": "red_blue", "operation": "copy_a_to_b", "target": "red_red"},
            {"source": "blue_red", "operation": "copy_a_to_b", "target": "blue_blue"},
        ],
        "rule_reasoning": [
            {"source": "holder_a", "operation": "swap_holder", "target": "holder_b"},
            {"source": "holder_b", "operation": "swap_holder", "target": "holder_a"},
            {"source": "holder_b", "operation": "set_a_holder", "target": "holder_a"},
        ],
        "grammar_constraint": [
            {"source": "singular_present", "operation": "toggle_number", "target": "plural_present"},
            {"source": "plural_present", "operation": "toggle_number", "target": "singular_present"},
            {"source": "singular_present", "operation": "set_past", "target": "singular_past"},
            {"source": "plural_present", "operation": "set_past", "target": "plural_past"},
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
    candidate_token_ids: dict[str, dict[str, int]] = {}
    all_candidates = sorted(
        {
            candidate
            for family in CANDIDATE_SETS.values()
            for candidates in family.values()
            for candidate in candidates
        }
    )
    for model, tokenizer in tokenizers.items():
        candidate_token_ids[model] = {}
        for candidate in all_candidates:
            ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError(
                    f"Phase404 candidate is not one token: {model}/{candidate}/{ids}"
                )
            candidate_token_ids[model][candidate] = int(ids[0])

    created_at = now()
    previous_hashes = previous_prompt_hashes()
    current_hashes: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    total_groups = sum(SPLIT_GROUP_COUNTS.values())
    for family in FAMILIES:
        for priority in range(total_groups):
            split = split_for(priority)
            group_id = "p404g_" + digest(f"{family}:{priority}", 24)
            registry.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase404-DirectStateProtocol",
                    "family_id": family,
                    "anonymous_parallel_group_id": group_id,
                    "candidate_split": split,
                    "group_priority": priority,
                    "state_count": len(STATE_IDS[family]),
                    "surface_replica_count": len(SURFACE_REPLICAS),
                    "future_query_count": len(QUERIES[family]),
                }
            )
            for state_id in STATE_IDS[family]:
                for surface in SURFACE_REPLICAS:
                    for query in QUERIES[family]:
                        item = raw_case(family, priority, state_id, surface, query)
                        for model in MODELS:
                            tokenizer = tokenizers[model]
                            prompt, add_special, answer_phase = interface_prompt(
                                tokenizer,
                                model,
                                item["raw_prompt"],
                                "answer_aligned_chat",
                            )
                            case_key = (
                                f"{model}:{family}:{group_id}:{state_id}:"
                                f"{surface['surface_id']}:{query}"
                            )
                            key = (model, digest(prompt))
                            if key in previous_hashes:
                                raise RuntimeError(
                                    f"Phase404 prompt overlaps Phase403: {case_key}"
                                )
                            if key in current_hashes:
                                raise RuntimeError(
                                    f"Phase404 duplicate: {current_hashes[key]} == {case_key}"
                                )
                            current_hashes[key] = case_key
                            rows.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "phase_id": "Phase404-DirectStateProtocol",
                                    "created_at": created_at,
                                    "private_execution_model": model,
                                    "blind_case_id": "p404c_" + digest(case_key, 28),
                                    "family_id": family,
                                    "anonymous_parallel_group_id": group_id,
                                    "parallel_group_id_private": f"p404_private_{family}_{priority:02d}",
                                    "candidate_split_private": split,
                                    "group_priority": priority,
                                    "state_id_private": state_id,
                                    "abstract_state_private": item[
                                        "abstract_state_private"
                                    ],
                                    "surface_id_private": surface["surface_id"],
                                    "surface_axes_private": surface,
                                    "future_query_private": query,
                                    "prompt": prompt,
                                    "raw_prompt": item["raw_prompt"],
                                    "state_prefix": item["state_prefix"],
                                    "query_fragment": item["query_fragment"],
                                    "tokenization_add_special_tokens": add_special,
                                    "interface": "answer_aligned_chat",
                                    "answer_phase": answer_phase,
                                    "target_private": item["target"],
                                    "candidate_answers_private": item[
                                        "candidate_answers_private"
                                    ],
                                    "candidate_token_ids_private": {
                                        candidate: candidate_token_ids[model][candidate]
                                        for candidate in item[
                                            "candidate_answers_private"
                                        ]
                                    },
                                    "target_token_id_private": candidate_token_ids[
                                        model
                                    ][item["target"]],
                                    "prompt_token_count": len(
                                        tokenizer(
                                            prompt,
                                            add_special_tokens=add_special,
                                        )["input_ids"]
                                    ),
                                    "formal_denominator": True,
                                }
                            )

    expected_rows = sum(
        total_groups
        * len(STATE_IDS[family])
        * len(SURFACE_REPLICAS)
        * len(QUERIES[family])
        * len(MODELS)
        for family in FAMILIES
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Phase404 row count {len(rows)} != {expected_rows}")
    write_jsonl(OUT / "protocol/private/phase404_all_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase404_blind_group_registry.jsonl", registry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase404-DirectStateProtocol",
        "created_at": created_at,
        "objective": "separate_direct_final_state_predictability_from_imperative_update_execution",
        "phase403_boundary": {
            "formal_discovery_cases": 5184,
            "semantic_correct_cases": 3411,
            "crossmodel_candidate_families": 0,
            "calibration_consumed": False,
            "behavioral_holdout_consumed": False,
            "physical_mapping_open": False,
            "imperative_update_execution_was_a_confound": True,
        },
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "batch_size": 1,
            "padding": "none",
            "attention_implementation": "eager",
            "use_cache": True,
            "runtime_dtype_by_model": FROZEN_DTYPES,
            "measurement": "next_token_full_logits_and_finite_candidate_response",
            "generation_used_for_candidate_gate": False,
        },
        "denominator": {
            "families": list(FAMILIES),
            "groups_per_family": total_groups,
            "split_group_counts": SPLIT_GROUP_COUNTS,
            "states_per_family": {
                family: len(STATE_IDS[family]) for family in FAMILIES
            },
            "surface_replicas_per_state": len(SURFACE_REPLICAS),
            "queries_per_state": 3,
            "case_count_all_models_all_splits": len(rows),
            "discovery_case_count_per_model": 960,
            "calibration_case_count_per_model_if_all_families_pass": 480,
            "behavioral_holdout_case_count_per_model_if_all_families_pass": 480,
            "physical_holdout_case_count_per_model_if_all_families_pass": 480,
        },
        "finite_candidate_contract": {
            "all_candidates": all_candidates,
            "all_candidates_are_one_token_in_each_model": True,
            "candidate_token_ids_private": True,
            "finite_candidate_logits_are_recorded": True,
            "global_top_token_is_recorded_as_a_control": True,
            "state_classes_are_truth_enumerated_not_clustered": True,
            "learned_probe_or_mapping": False,
        },
        "integer_gates": {
            "surface_fingerprint_pass_min": "3_of_4",
            "group_candidate_correct_min": {
                "knowledge_binding": "42_of_48",
                "rule_reasoning": "21_of_24",
                "grammar_constraint": "42_of_48",
            },
            "discovery_groups_min_per_model_family": "6_of_8",
            "calibration_groups_min_per_model_family": "3_of_4",
            "behavioral_holdout_groups_min_per_model_family": "3_of_4",
            "state_blind_baseline_margin_min": 0.20,
            "crossmodel_requires_all_three_models": True,
        },
        "semantic_transition_graph": semantic_transition_table(),
        "semantic_transition_graph_is_observed_internal_operator": False,
        "authorization": {
            "run_discovery": True,
            "run_calibration_only_for_discovery_crossmodel_families": True,
            "run_behavioral_holdout_only_for_calibration_crossmodel_families": True,
            "run_physical_holdout_only_for_behavioral_holdout_crossmodel_families": True,
            "run_internal_physical_map_before_behavioral_holdout": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "finite_candidate_response_is_full_natural_language_distribution": False,
            "direct_state_predictive_equivalence_is_causal_state": False,
            "truth_enumerated_transition_is_model_operator": False,
            "crossmodel_response_table_is_brain_isomorphism": False,
            "negative_result_excludes_other_state_panels": False,
        },
    }
    write_json(OUT / "phase404_direct_state_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
