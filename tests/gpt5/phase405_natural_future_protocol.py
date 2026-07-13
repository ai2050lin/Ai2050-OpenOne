#!/usr/bin/env python3
"""Freeze natural unfinished future branches for predictive-state testing.

Unlike Phase403 and Phase404, these prompts do not ask the model to choose or
return a listed answer.  Each branch ends immediately before one natural
one-token continuation.  The finite answer set remains an audit panel, while
the full-vocabulary top token is retained as the natural-generation control.
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
from phase404_direct_state_protocol import (  # noqa: E402
    CANDIDATE_SETS,
    FAMILIES,
    FROZEN_DTYPES,
    MODELS,
    QUERIES,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    SURFACE_REPLICAS,
    expected_answer,
    facts_for,
    package_for,
    semantic_transition_table,
    split_for,
    state_truth,
)


OUT = ROOT / "tests/gpt5/result/phase405_natural_future_state"
SCHEMA_VERSION = "79.0.0"


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


def natural_branch(
    family: str,
    package: dict[str, Any],
    state_id: str,
    query: str,
    syntax: int,
) -> str:
    if family == "knowledge_binding":
        item = package["item"]
        if query == "color_a":
            return (
                f"According to this final record, person A's {item} is"
                if syntax == 0
                else f"The final {item} color assigned to person A is"
            )
        if query == "color_b":
            return (
                f"According to this final record, person B's {item} is"
                if syntax == 0
                else f"The final {item} color assigned to person B is"
            )
        return (
            "Do person A and person B have the same final color? The answer is"
            if syntax == 0
            else "Whether both final colors match can be answered with"
        )

    if family == "rule_reasoning":
        intermediate = package["intermediate"]
        terminal = package["terminal"]
        if query == "one_step_holder":
            return (
                f"After applying the first rule, the person receiving the {intermediate} is person"
                if syntax == 0
                else f"The {intermediate} therefore goes to person"
            )
        if query == "two_step_holder":
            return (
                f"After both rules, the person allowed to enter the {terminal} is person"
                if syntax == 0
                else f"The person who may consequently enter the {terminal} is person"
            )
        return (
            f"May person A enter the {terminal} after both rules? The answer is"
            if syntax == 0
            else f"Person A's permission to enter the {terminal} can be answered with"
        )

    number, _tense = state_truth(family, state_id)
    noun = package["noun_singular"] if number == 0 else package["noun_plural"]
    if query == "be_auxiliary":
        return (
            f"For 'The {noun} ... ready', the missing be-form is"
            if syntax == 0
            else f"The required be-form after 'The {noun}' is"
        )
    if query == "have_auxiliary":
        return (
            f"For 'The {noun} ... arrived', the missing have-form is"
            if syntax == 0
            else f"The required have-form after 'The {noun}' is"
        )
    return (
        f"The demonstrative that belongs before '{noun}' is"
        if syntax == 0
        else f"Before the noun '{noun}', the matching demonstrative is"
    )


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
    branch = natural_branch(
        family,
        package,
        state_id,
        query,
        surface["syntax"],
    )
    return {
        "raw_prompt": f"{facts}\n{branch}",
        "state_prefix": facts,
        "query_fragment": branch,
        "target": expected_answer(family, state_id, query),
        "candidate_answers_private": list(CANDIDATE_SETS[family][query]),
        "abstract_state_private": list(state_truth(family, state_id)),
    }


def previous_prompt_hashes() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for path in (
        ROOT / "tests/gpt5/result/phase403_predictive_state/protocol/private/phase403_all_cases.jsonl",
        ROOT / "tests/gpt5/result/phase404_direct_predictive_state/protocol/private/phase404_all_cases.jsonl",
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = row.get("private_execution_model") or row.get("model")
            result.add((model, digest(row["prompt"])))
    return result


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

    all_candidates = sorted(
        {
            answer
            for family in CANDIDATE_SETS.values()
            for candidates in family.values()
            for answer in candidates
        }
    )
    candidate_ids: dict[str, dict[str, int]] = {}
    for model, tokenizer in tokenizers.items():
        candidate_ids[model] = {}
        for candidate in all_candidates:
            ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError(
                    f"Phase405 continuation is not one token: {model}/{candidate}/{ids}"
                )
            candidate_ids[model][candidate] = int(ids[0])

    created_at = now()
    previous_hashes = previous_prompt_hashes()
    current_hashes: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    total_groups = sum(SPLIT_GROUP_COUNTS.values())
    for family in FAMILIES:
        for priority in range(total_groups):
            split = split_for(priority)
            group_id = "p405g_" + digest(f"{family}:{priority}", 24)
            registry.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase405-NaturalFutureProtocol",
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
                        for model, tokenizer in tokenizers.items():
                            prompt = item["raw_prompt"]
                            case_key = (
                                f"{model}:{family}:{group_id}:{state_id}:"
                                f"{surface['surface_id']}:{query}"
                            )
                            prompt_key = (model, digest(prompt))
                            if prompt_key in previous_hashes:
                                raise RuntimeError(f"Phase405 overlaps prior prompt: {case_key}")
                            if prompt_key in current_hashes:
                                raise RuntimeError(
                                    f"Phase405 duplicate: {current_hashes[prompt_key]} == {case_key}"
                                )
                            current_hashes[prompt_key] = case_key
                            prompt_ids = tokenizer(
                                prompt, add_special_tokens=True
                            )["input_ids"]
                            target = item["target"]
                            extended_ids = tokenizer(
                                prompt + " " + target, add_special_tokens=True
                            )["input_ids"]
                            target_id = candidate_ids[model][target]
                            if (
                                len(extended_ids) != len(prompt_ids) + 1
                                or extended_ids[:-1] != prompt_ids
                                or int(extended_ids[-1]) != target_id
                            ):
                                raise RuntimeError(
                                    f"Phase405 unstable continuation boundary: {case_key}"
                                )
                            rows.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "phase_id": "Phase405-NaturalFutureProtocol",
                                    "created_at": created_at,
                                    "private_execution_model": model,
                                    "blind_case_id": "p405c_" + digest(case_key, 28),
                                    "family_id": family,
                                    "anonymous_parallel_group_id": group_id,
                                    "parallel_group_id_private": f"p405_private_{family}_{priority:02d}",
                                    "candidate_split_private": split,
                                    "group_priority": priority,
                                    "state_id_private": state_id,
                                    "abstract_state_private": item["abstract_state_private"],
                                    "surface_id_private": surface["surface_id"],
                                    "surface_axes_private": surface,
                                    "future_query_private": query,
                                    "prompt": prompt,
                                    "state_prefix": item["state_prefix"],
                                    "query_fragment": item["query_fragment"],
                                    "tokenization_add_special_tokens": True,
                                    "interface": "raw_natural_completion",
                                    "answer_phase": "natural_next_token",
                                    "target_private": target,
                                    "candidate_answers_private": item[
                                        "candidate_answers_private"
                                    ],
                                    "candidate_token_ids_private": {
                                        candidate: candidate_ids[model][candidate]
                                        for candidate in item[
                                            "candidate_answers_private"
                                        ]
                                    },
                                    "target_token_id_private": target_id,
                                    "prompt_token_count": len(prompt_ids),
                                    "explicit_choice_instruction": False,
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
        raise RuntimeError(f"Phase405 row count {len(rows)} != {expected_rows}")
    write_jsonl(OUT / "protocol/private/phase405_all_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase405_blind_group_registry.jsonl", registry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase405-NaturalFutureProtocol",
        "created_at": created_at,
        "objective": "test_finite_predictive_states_with_natural_unfinished_future_branches",
        "models_in_execution_order": list(MODELS),
        "execution_contract": {
            "batch_size": 1,
            "padding": "none",
            "attention_implementation": "eager",
            "use_cache": True,
            "runtime_dtype_by_model": FROZEN_DTYPES,
            "interface": "raw_natural_completion",
            "explicit_choose_or_return_instruction": False,
            "measurement": "full_next_token_logits_plus_finite_branch_panel",
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
        },
        "integer_gates": {
            "surface_truth_pass_min": "3_of_4",
            "group_candidate_correct_min": {
                "knowledge_binding": "42_of_48",
                "rule_reasoning": "21_of_24",
                "grammar_constraint": "42_of_48",
            },
            "group_natural_top_correct_min": {
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
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "finite_future_panel_is_exhaustive": False,
            "raw_continuation_is_internal_state": False,
            "predictive_equivalence_is_causal_equivalence": False,
            "truth_transition_table_is_model_operator": False,
        },
    }
    write_json(OUT / "phase405_natural_future_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
