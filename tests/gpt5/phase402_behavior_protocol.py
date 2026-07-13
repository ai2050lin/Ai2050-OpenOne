#!/usr/bin/env python3
"""Freeze Phase402's fresh six-surface, four-factor behavior denominator."""

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
from phase386_multitime_protocol import NAMES, OBJECTS  # noqa: E402
from phase390_role_mapping import fragment_positions, prompt_token_ids  # noqa: E402
from phase397_multitask_protocol import ROLE_VALUES  # noqa: E402
from phase399_dynamic_binding_protocol import compatible_pool  # noqa: E402
from phase402_multiparent_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SPLIT_CANDIDATE_COUNTS,
    SURFACES,
)


AXES = ("X", "Y")
LEVELS = (0, 1)
CONDITIONS = tuple(
    f"{axis}_R{relation}_O{order}_Q{query}"
    for axis in AXES
    for relation in LEVELS
    for order in LEVELS
    for query in LEVELS
)
CANDIDATE_GROUPS_PER_SURFACE = sum(SPLIT_CANDIDATE_COUNTS.values())
FAMILY_BY_SURFACE = {
    "entity_attribute_binding": "content_knowledge",
    "role_filling": "language_action",
    "coreference_resolution": "reasoning_constraint",
    "two_step_composition": "reasoning_constraint",
    "conditional_presence": "logic_constraint",
    "number_agreement": "syntax_structure",
}
TARGET_CARDINALITY = {
    "entity_attribute_binding": 4,
    "role_filling": 4,
    "coreference_resolution": 4,
    "two_step_composition": 4,
    "conditional_presence": 2,
    "number_agreement": 2,
}
COUNT_VALUES = ("one", "two", "single", "multiple")


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


def parse_condition(condition: str) -> tuple[str, int, int, int]:
    axis, relation, order, query = condition.split("_")
    return axis, int(relation[1:]), int(order[1:]), int(query[1:])


def split_for(priority: int) -> str:
    cursor = 0
    for split, count in SPLIT_CANDIDATE_COUNTS.items():
        if priority < cursor + count:
            return split
        cursor += count
    raise ValueError(f"Invalid Phase402 priority: {priority}")


def select_distinct(
    pool: tuple[str, ...], seed: int, count: int, multiplier: int, stride: int
) -> tuple[str, ...]:
    values = tuple(
        pool[(seed * multiplier + offset * stride) % len(pool)]
        for offset in range(count)
    )
    if len(set(values)) != count:
        raise ValueError("lexical_collision")
    return values


def lexical_package(
    surface: str,
    seed: int,
    entity_pool: tuple[str, ...],
    object_pool: tuple[str, ...],
    role_pool: tuple[str, ...],
) -> dict[str, Any]:
    entity_a, entity_b = select_distinct(entity_pool, seed, 2, 13, 17)
    if surface == "role_filling":
        values = select_distinct(role_pool, seed, 4, 19, 7)
    elif surface == "number_agreement":
        values = COUNT_VALUES
    else:
        values = select_distinct(object_pool, seed, 4, 19, 7)
    result: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "axis_values": {"X": values[:2], "Y": values[2:]},
    }
    if surface == "two_step_composition":
        intermediate = select_distinct(role_pool, seed, 4, 23, 11)
        result["axis_intermediates"] = {
            "X": intermediate[:2],
            "Y": intermediate[2:],
        }
    return result


def direct_clause(surface: str, entity: str, value: str) -> str:
    if surface == "entity_attribute_binding":
        return f"- {entity}: stored attribute = {value}"
    if surface == "role_filling":
        return f"- {entity}: assigned role = {value}"
    if surface == "coreference_resolution":
        return f'- {entity} said, "I selected {value}."'
    if surface == "conditional_presence":
        return f"- {entity}: registered item = {value}"
    if surface == "number_agreement":
        return f"- {entity}: crew quantity = {value}"
    raise ValueError(surface)


def direct_query(surface: str, entity: str, asked_value: str | None = None) -> str:
    if surface == "entity_attribute_binding":
        return f"What stored attribute belongs to {entity}?"
    if surface == "role_filling":
        return f"What assigned role belongs to {entity}?"
    if surface == "coreference_resolution":
        return f"What did {entity} say that they selected?"
    if surface == "conditional_presence":
        return f"Is {entity} registered with the item {asked_value}?"
    if surface == "number_agreement":
        return f"Complete from the recorded quantity: The crew assigned to {entity} ___ ready."
    raise ValueError(surface)


def raw_case(surface: str, condition: str, package: dict[str, Any]) -> dict[str, Any]:
    axis, relation, order, query = parse_condition(condition)
    entity_a = package["entity_a"]
    entity_b = package["entity_b"]
    value_a, value_b = package["axis_values"][axis]
    slot_entities = [entity_a, entity_b]
    slot_values = [value_a, value_b]
    if relation:
        slot_entities = [entity_b, entity_a]
    binding = dict(zip(slot_entities, slot_values))
    queried = entity_b if query else entity_a

    intermediate_a = intermediate_b = None
    if surface == "two_step_composition":
        intermediate_a, intermediate_b = package["axis_intermediates"][axis]
        slot_intermediates = [intermediate_a, intermediate_b]
        chain_blocks = [
            (
                f"- start {slot_entities[index]} = {slot_intermediates[index]}\n"
                f"- finish {slot_intermediates[index]} = {slot_values[index]}"
            )
            for index in range(2)
        ]
        if order:
            chain_blocks.reverse()
        clauses = chain_blocks
        source = "\n".join(chain_blocks)
        query_text = f"Follow both maps. What final item is reached from {queried}?"
        target = binding[queried]
        distractors = [next(value for value in slot_values if value != target)]
        heading = "Two-step maps:"
        answer_label = "final item"
    else:
        clauses = [
            direct_clause(surface, slot_entities[index], slot_values[index])
            for index in range(2)
        ]
        if order:
            clauses.reverse()
        source = "\n".join(clauses)
        asked_value = value_a if surface == "conditional_presence" else None
        query_text = direct_query(surface, queried, asked_value)
        heading = "Reference facts:"
        if surface == "conditional_presence":
            target = "yes" if binding[queried] == asked_value else "no"
            distractors = ["no" if target == "yes" else "yes"]
            answer_label = "yes or no"
        elif surface == "number_agreement":
            target = "is" if binding[queried] in {"one", "single"} else "are"
            distractors = ["are" if target == "is" else "is"]
            answer_label = "agreement word"
        else:
            target = binding[queried]
            distractors = [next(value for value in slot_values if value != target)]
            answer_label = "answer word"

    raw_prompt = (
        f"{heading}\n{source}\nQuestion: {query_text}\n"
        f"Return exactly the lowercase {answer_label} and nothing else.\nAnswer:"
    )
    slots: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "value_a": value_a,
        "value_b": value_b,
        "query_entity": queried,
    }
    if intermediate_a is not None and intermediate_b is not None:
        slots["intermediate_a"] = intermediate_a
        slots["intermediate_b"] = intermediate_b
    if surface == "conditional_presence":
        slots["asked_value"] = value_a
    return {
        "raw_prompt": raw_prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": distractors,
        "source_fragment": source,
        "query_fragment": query_text,
        "clause_fragments_private": clauses,
        "semantic_role_fragments_private": {
            "entities": [entity_a, entity_b],
            "attributes_items": [value_a, value_b],
            "relations": clauses,
            "query_keywords": ["Question:"],
        },
        "semantic_slot_fragments_private": slots,
        "axis_private": axis,
        "relation_level_private": relation,
        "order_level_private": order,
        "query_level_private": query,
    }


def render(
    tokenizer: Any,
    model: str,
    surface: str,
    condition: str,
    package: dict[str, Any],
) -> dict[str, Any]:
    item = raw_case(surface, condition, package)
    prompt, add_special, answer_phase = interface_prompt(
        tokenizer, model, item["raw_prompt"], "answer_aligned_chat"
    )
    return {
        **item,
        "prompt": prompt,
        "tokenization_add_special_tokens": add_special,
        "interface": "answer_aligned_chat",
        "answer_phase": answer_phase,
        "condition": condition,
    }


def positions_in(
    tokenizer: Any, case: dict[str, Any], fragment: str, container_key: str
) -> list[int]:
    ids = prompt_token_ids(tokenizer, case)
    allowed = set(fragment_positions(tokenizer, ids, case[container_key]))
    return [
        position
        for position in fragment_positions(tokenizer, ids, fragment)
        if position in allowed
    ]


def factorial_audit(
    tokenizer: Any, cases: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    for condition, case in cases.items():
        ids = prompt_token_ids(tokenizer, case)
        if not ids:
            return {"valid": False, "reason": "empty_prompt", "condition": condition}
        slots = case["semantic_slot_fragments_private"]
        for key in ("entity_a", "entity_b", "value_a", "value_b"):
            if not positions_in(tokenizer, case, slots[key], "source_fragment"):
                return {
                    "valid": False,
                    "reason": "missing_source_role",
                    "condition": condition,
                    "role": key,
                }
        if not positions_in(tokenizer, case, slots["query_entity"], "query_fragment"):
            return {
                "valid": False,
                "reason": "missing_query_entity",
                "condition": condition,
            }

    for axis in AXES:
        for order in LEVELS:
            for query in LEVELS:
                left = cases[f"{axis}_R0_O{order}_Q{query}"]
                right = cases[f"{axis}_R1_O{order}_Q{query}"]
                left_ids = prompt_token_ids(tokenizer, left)
                right_ids = prompt_token_ids(tokenizer, right)
                if len(left_ids) != len(right_ids) or Counter(left_ids) != Counter(right_ids):
                    return {"valid": False, "reason": "relation_token_contract"}
                if left["target"] == right["target"]:
                    return {"valid": False, "reason": "relation_did_not_change_target"}
            for relation in LEVELS:
                for query in LEVELS:
                    left = cases[f"{axis}_R{relation}_O0_Q{query}"]
                    right = cases[f"{axis}_R{relation}_O1_Q{query}"]
                    if Counter(prompt_token_ids(tokenizer, left)) != Counter(
                        prompt_token_ids(tokenizer, right)
                    ):
                        return {"valid": False, "reason": "order_token_contract"}
                    if left["target"] != right["target"]:
                        return {"valid": False, "reason": "order_changed_target"}
                for order in LEVELS:
                    left = cases[f"{axis}_R{relation}_O{order}_Q0"]
                    right = cases[f"{axis}_R{relation}_O{order}_Q1"]
                    if left["source_fragment"] != right["source_fragment"]:
                        return {"valid": False, "reason": "query_changed_source"}
                    if len(prompt_token_ids(tokenizer, left)) != len(
                        prompt_token_ids(tokenizer, right)
                    ):
                        return {"valid": False, "reason": "query_length_contract"}
                    if left["target"] == right["target"]:
                        return {"valid": False, "reason": "query_did_not_change_target"}

    for relation in LEVELS:
        for order in LEVELS:
            for query in LEVELS:
                left = cases[f"X_R{relation}_O{order}_Q{query}"]
                right = cases[f"Y_R{relation}_O{order}_Q{query}"]
                if len(prompt_token_ids(tokenizer, left)) != len(
                    prompt_token_ids(tokenizer, right)
                ):
                    return {"valid": False, "reason": "lexical_axis_length_contract"}
    return {"valid": True}


def prior_prompt_hashes() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    paths = (
        ROOT
        / "tests/gpt5/result/phase401_local_edge_graph/protocol/private/phase401_candidate_cases.jsonl",
        ROOT
        / "tests/gpt5/result/phase400_partial_order/protocol/private/phase400_candidate_cases.jsonl",
    )
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result.add((row["private_execution_model"], digest(row["prompt"])))
    return result


def main() -> None:
    frozen_contract = OUT / "phase402_multiparent_protocol.json"
    if not frozen_contract.is_file():
        raise FileNotFoundError("Freeze phase402_multiparent_protocol.json first")
    contract_hash = digest(frozen_contract.read_text(encoding="utf-8"))
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    entity_pool = compatible_pool(tuple(NAMES), tokenizers)
    object_pool = compatible_pool(tuple(OBJECTS), tokenizers)
    role_pool = compatible_pool(tuple(ROLE_VALUES), tokenizers)
    count_pool = compatible_pool(COUNT_VALUES, tokenizers)
    if set(count_pool) != set(COUNT_VALUES):
        raise RuntimeError(
            "Phase402 count replicas do not share one cross-model token-width bucket"
        )
    if min(len(entity_pool), len(object_pool), len(role_pool)) < 8:
        raise RuntimeError("Insufficient Phase402 cross-model lexical pools")

    created_at = now()
    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    prompt_hashes = prior_prompt_hashes()
    package_signatures: set[str] = set()
    for surface_index, surface in enumerate(SURFACES):
        accepted = 0
        seed = 202000 + 5000 * surface_index
        rejects: Counter[str] = Counter()
        while accepted < CANDIDATE_GROUPS_PER_SURFACE and seed < 400000:
            try:
                package = lexical_package(
                    surface, seed, entity_pool, object_pool, role_pool
                )
            except ValueError as error:
                rejects[str(error)] += 1
                seed += 1
                continue
            signature = digest(
                json.dumps(
                    {"surface": surface, **package},
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            if signature in package_signatures:
                rejects["duplicate_package"] += 1
                seed += 1
                continue
            per_model: dict[str, dict[str, dict[str, Any]]] = {}
            valid = True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases = {
                    condition: render(tokenizer, model, surface, condition, package)
                    for condition in CONDITIONS
                }
                audit = factorial_audit(tokenizer, cases)
                if not audit["valid"]:
                    rejects[f"{model}:{audit['reason']}"] += 1
                    valid = False
                    break
                target_ids = {
                    tokenizer(" " + case["target"], add_special_tokens=False)[
                        "input_ids"
                    ][0]
                    for case in cases.values()
                }
                if len(target_ids) != TARGET_CARDINALITY[surface]:
                    rejects[f"{model}:target_cardinality"] += 1
                    valid = False
                    break
                per_model[model] = cases
            if not valid:
                seed += 1
                continue

            group_id = "p402g_" + digest(f"phase402:{surface}:{seed}", 24)
            split = split_for(accepted)
            selection_priority = digest(
                f"phase402-selection:{surface}:{split}:{group_id}", 24
            )
            staged: list[dict[str, Any]] = []
            duplicate = False
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    prompt_hash = digest(item["prompt"])
                    if (model, prompt_hash) in prompt_hashes:
                        duplicate = True
                        break
                    staged.append(
                        {
                            "schema_version": "76.1.0",
                            "phase_id": "Phase402-BehaviorProtocol",
                            "created_at": created_at,
                            "frozen_multiparent_protocol_sha256": contract_hash,
                            "private_execution_model": model,
                            "anonymous_model_id": "p402m_" + digest(model, 12),
                            "blind_case_id": "p402c_"
                            + digest(f"{model}:{group_id}:{condition}", 26),
                            "anonymous_parallel_group_id": group_id,
                            "anonymous_group_id": "p402s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition,
                            "candidate_split_private": split,
                            "selection_priority_private": selection_priority,
                            "group_priority": accepted,
                            "family_id": FAMILY_BY_SURFACE[surface],
                            "mechanism_id": surface,
                            "semantic_group_id": f"p402_private_{surface}_{seed:06d}",
                            "contrast_condition": condition,
                            "task_surface_private": surface,
                            "prompt": item["prompt"],
                            "raw_prompt": item["raw_prompt"],
                            "source_fragment": item["source_fragment"],
                            "query_fragment": item["query_fragment"],
                            "clause_fragments_private": item[
                                "clause_fragments_private"
                            ],
                            "tokenization_add_special_tokens": item[
                                "tokenization_add_special_tokens"
                            ],
                            "interface": item["interface"],
                            "answer_phase": item["answer_phase"],
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "axis_private": item["axis_private"],
                            "relation_level_private": item[
                                "relation_level_private"
                            ],
                            "order_level_private": item["order_level_private"],
                            "query_level_private": item["query_level_private"],
                            "semantic_role_fragments_private": item[
                                "semantic_role_fragments_private"
                            ],
                            "semantic_slot_fragments_private": item[
                                "semantic_slot_fragments_private"
                            ],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                            "formal_denominator": True,
                        }
                    )
                if duplicate:
                    break
            if duplicate:
                rejects["prior_or_current_prompt_overlap"] += 1
                seed += 1
                continue

            package_signatures.add(signature)
            for row in staged:
                prompt_hashes.add(
                    (row["private_execution_model"], digest(row["prompt"]))
                )
            rows.extend(staged)
            groups.append(
                {
                    "schema_version": "76.1.0",
                    "phase_id": "Phase402-BehaviorProtocol",
                    "anonymous_parallel_group_id": group_id,
                    "task_surface": surface,
                    "candidate_split": split,
                    "group_priority": accepted,
                    "condition_count": len(CONDITIONS),
                    "fresh_prompt_against_phase400_and_phase401": True,
                }
            )
            accepted += 1
            print(
                f"[phase402/{surface}] {accepted}/{CANDIDATE_GROUPS_PER_SURFACE} "
                f"seed={seed}",
                flush=True,
            )
            seed += 1
        if accepted != CANDIDATE_GROUPS_PER_SURFACE:
            raise RuntimeError(
                f"Could not freeze Phase402 {surface}: {accepted}; rejects={dict(rejects)}"
            )

    expected = (
        len(SURFACES)
        * CANDIDATE_GROUPS_PER_SURFACE
        * len(CONDITIONS)
        * len(MODELS)
    )
    if len(rows) != expected:
        raise RuntimeError(f"Phase402 row count {len(rows)} != {expected}")
    private = OUT / "protocol/private"
    write_jsonl(private / "phase402_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase402_blind_group_registry.jsonl", groups)
    payload = {
        "schema_version": "76.1.0",
        "phase_id": "Phase402-BehaviorProtocol",
        "created_at": created_at,
        "frozen_multiparent_protocol_sha256": contract_hash,
        "denominator": {
            "surfaces": list(SURFACES),
            "candidate_groups_per_surface": CANDIDATE_GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_group_count": len(groups),
            "candidate_case_count": len(rows),
            "split_group_counts_per_surface": SPLIT_CANDIDATE_COUNTS,
        },
        "lexical_contract": {
            "entity_pool_size": len(entity_pool),
            "object_pool_size": len(object_pool),
            "role_pool_size": len(role_pool),
            "factorial_token_contract_checked_in_each_model": True,
            "fresh_prompt_hash_against_phase400_and_phase401": True,
        },
        "authorization": {
            "run_formal_behavior_batch1": True,
            "run_internal_before_behavior_freeze": False,
        },
        "claim_boundary": {
            "six_surfaces_are_six_closed_language_mechanisms": False,
            "behavior_success_is_internal_mechanism_evidence": False,
        },
    }
    write_json(OUT / "phase402_behavior_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
