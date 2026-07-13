#!/usr/bin/env python3
"""Freeze the fresh Phase400 four-surface behavior denominator."""

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
from phase390_role_mapping import prompt_token_ids  # noqa: E402
from phase397_multitask_protocol import ROLE_VALUES  # noqa: E402
from phase398_joint_factorial_protocol import (  # noqa: E402
    AXES,
    LEVELS,
    lexical_audit,
    order_audit,
    parse_condition,
    query_audit,
    relation_audit,
)
from phase399_dynamic_binding_protocol import compatible_pool, lexical_items  # noqa: E402
from phase400_partial_order_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SPLIT_CANDIDATE_COUNTS,
    SPLIT_SELECTED_COUNTS,
    SURFACES,
)


CONDITIONS = tuple(
    f"{axis}_R{relation}_O{order}_Q{query}"
    for axis in AXES
    for relation in LEVELS
    for order in LEVELS
    for query in LEVELS
)
CANDIDATE_GROUPS_PER_SURFACE = sum(SPLIT_CANDIDATE_COUNTS.values())
FAMILY_BY_SURFACE = {
    "possession_relation": "content_knowledge",
    "role_filling": "language_action",
    "coreference_resolution": "reasoning_constraint",
    "field_extraction": "content_knowledge",
}


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


def values_for(surface: str) -> tuple[str, ...]:
    return tuple(ROLE_VALUES if surface == "role_filling" else OBJECTS)


def split_for(priority: int) -> str:
    cursor = 0
    for split, count in SPLIT_CANDIDATE_COUNTS.items():
        if priority < cursor + count:
            return split
        cursor += count
    raise ValueError(f"Invalid Phase400 group priority: {priority}")


def clause(surface: str, entity: str, value: str) -> str:
    if surface == "possession_relation":
        return f"- {entity}: owned item = {value}"
    if surface == "role_filling":
        return f"- {entity}: assigned role = {value}"
    if surface == "coreference_resolution":
        return f"- {entity} said they selected {value}"
    return f"- record {entity}: item = {value}"


def query_fragment(surface: str, entity: str) -> str:
    if surface == "possession_relation":
        return f"What owned item is recorded for {entity}?"
    if surface == "role_filling":
        return f"What assigned role is recorded for {entity}?"
    if surface == "coreference_resolution":
        return f"What did {entity} say they selected?"
    return f"What item value is stored for record {entity}?"


def raw_case(
    surface: str,
    condition: str,
    entity_a: str,
    entity_b: str,
    value_a: str,
    value_b: str,
) -> dict[str, Any]:
    axis, relation, order, query = parse_condition(condition)
    slot_entities = [entity_a, entity_b]
    slot_values = [value_a, value_b]
    if relation:
        slot_entities = [entity_b, entity_a]
    if order:
        slot_entities = list(reversed(slot_entities))
        slot_values = list(reversed(slot_values))
    clauses = [clause(surface, slot_entities[index], slot_values[index]) for index in range(2)]
    source = "\n".join(clauses)
    queried = entity_b if query else entity_a
    binding = dict(zip(slot_entities, slot_values))
    target = binding[queried]
    distractor = next(value for value in slot_values if value != target)
    query_text = query_fragment(surface, queried)
    heading = "Reference table:" if surface == "field_extraction" else "Reference facts:"
    raw_prompt = (
        f"{heading}\n{source}\nQuestion: {query_text}\n"
        f"Return exactly the lowercase {'item value' if surface == 'field_extraction' else 'answer word'} and nothing else.\n"
        "Answer:"
    )
    return {
        "raw_prompt": raw_prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": [distractor],
        "source_fragment": source,
        "query_fragment": query_text,
        "clause_fragments_private": clauses,
        "semantic_role_fragments_private": {
            "entities": [entity_a, entity_b],
            "attributes_items": [value_a, value_b],
            "relations": clauses,
            "query_keywords": ["Question:"],
        },
        "semantic_slot_fragments_private": {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "value_a": value_a,
            "value_b": value_b,
            "query_entity": queried,
        },
        "axis_private": axis,
        "relation_level_private": relation,
        "order_level_private": order,
        "query_level_private": query,
    }


def render(tokenizer: Any, model: str, *args: Any) -> dict[str, Any]:
    item = raw_case(*args)
    prompt, add_special, answer_phase = interface_prompt(
        tokenizer, model, item["raw_prompt"], "answer_aligned_chat"
    )
    return {
        **item,
        "prompt": prompt,
        "tokenization_add_special_tokens": add_special,
        "interface": "answer_aligned_chat",
        "answer_phase": answer_phase,
        "condition": args[1],
    }


def previous_signatures() -> set[tuple[str, str, str, str, str]]:
    paths = (
        ROOT / "tests/gpt5/result/phase398_joint_binding/protocol/private/phase398_candidate_cases.jsonl",
        ROOT / "tests/gpt5/result/phase399_dynamic_binding/protocol/private/phase399_candidate_cases.jsonl",
    )
    signatures: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["private_execution_model"] != "qwen3":
                continue
            if row["anonymous_condition_slot"] not in {"X_R0_O0_Q0", "Y_R0_O0_Q0"}:
                continue
            slots = row["semantic_slot_fragments_private"]
            signatures.add(
                (
                    row["task_surface_private"],
                    slots["entity_a"],
                    slots["entity_b"],
                    slots["value_a"],
                    slots["value_b"],
                )
            )
    return signatures


def main() -> None:
    frozen_contract = OUT / "phase400_partial_order_protocol.json"
    if not frozen_contract.is_file():
        raise FileNotFoundError("Freeze phase400_partial_order_protocol.json first")
    contract_hash = digest(frozen_contract.read_text(encoding="utf-8"))
    created_at = now()
    tokenizers: dict[str, Any] = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    prior = previous_signatures()
    entity_pool = compatible_pool(tuple(NAMES), tokenizers)
    value_pools = {
        surface: compatible_pool(values_for(surface), tokenizers) for surface in SURFACES
    }
    if len(entity_pool) < 8 or any(len(pool) < 8 for pool in value_pools.values()):
        raise RuntimeError("Insufficient Phase400 cross-model token-width pools")

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    prompt_hashes: set[tuple[str, str]] = set()
    signatures = set(prior)
    for surface in SURFACES:
        accepted, seed = 0, 61000 + 2000 * SURFACES.index(surface)
        rejects: Counter[str] = Counter()
        while accepted < CANDIDATE_GROUPS_PER_SURFACE and seed < 140000:
            try:
                entity_a, entity_b, v1, v2, v3, v4 = lexical_items(
                    surface, seed, entity_pool, value_pools[surface]
                )
            except ValueError:
                seed += 1
                continue
            pair_signatures = {
                (surface, entity_a, entity_b, v1, v2),
                (surface, entity_a, entity_b, v3, v4),
            }
            if signatures.intersection(pair_signatures):
                rejects["prior_or_within_phase_signature_overlap"] += 1
                seed += 1
                continue
            per_model: dict[str, dict[str, Any]] = {}
            valid = True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases = {
                    condition: render(
                        tokenizer,
                        model,
                        surface,
                        condition,
                        entity_a,
                        entity_b,
                        *((v1, v2) if condition.startswith("X_") else (v3, v4)),
                    )
                    for condition in CONDITIONS
                }
                audits: list[dict[str, Any]] = []
                for axis in AXES:
                    for order in LEVELS:
                        for query in LEVELS:
                            audits.append(
                                relation_audit(
                                    tokenizer,
                                    cases[f"{axis}_R0_O{order}_Q{query}"],
                                    cases[f"{axis}_R1_O{order}_Q{query}"],
                                )
                            )
                    for relation in LEVELS:
                        for query in LEVELS:
                            audits.append(
                                order_audit(
                                    tokenizer,
                                    cases[f"{axis}_R{relation}_O0_Q{query}"],
                                    cases[f"{axis}_R{relation}_O1_Q{query}"],
                                )
                            )
                        for order in LEVELS:
                            audits.append(
                                query_audit(
                                    tokenizer,
                                    cases[f"{axis}_R{relation}_O{order}_Q0"],
                                    cases[f"{axis}_R{relation}_O{order}_Q1"],
                                )
                            )
                for relation in LEVELS:
                    for order in LEVELS:
                        for query in LEVELS:
                            audits.append(
                                lexical_audit(
                                    tokenizer,
                                    cases[f"X_R{relation}_O{order}_Q{query}"],
                                    cases[f"Y_R{relation}_O{order}_Q{query}"],
                                )
                            )
                invalid = next((item for item in audits if not item["valid"]), None)
                if invalid is not None:
                    rejects[f"{model}:{invalid['reason']}"] += 1
                    valid = False
                    break
                target_ids = {
                    tokenizer(
                        " " + cases[condition]["target"], add_special_tokens=False
                    )["input_ids"][0]
                    for condition in CONDITIONS
                }
                if len(target_ids) != 4:
                    rejects[f"{model}:target_first_token_not_four_way_distinct"] += 1
                    valid = False
                    break
                per_model[model] = cases
            if not valid:
                seed += 1
                continue

            signatures.update(pair_signatures)
            split = split_for(accepted)
            group_id = "p400g_" + digest(f"phase400:{surface}:{seed}", 24)
            selection_priority = digest(f"phase400-selection:{surface}:{split}:{group_id}", 24)
            groups.append(
                {
                    "schema_version": "74.1.0",
                    "phase_id": "Phase400-Protocol",
                    "anonymous_parallel_group_id": group_id,
                    "task_surface": surface,
                    "candidate_split": split,
                    "group_priority": accepted,
                    "selection_priority": selection_priority,
                    "condition_count": len(CONDITIONS),
                    "fresh_against_phase398_and_phase399_pair_signatures": True,
                    "field_contract_pilot_not_part_of_denominator": True,
                }
            )
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    prompt_hash = digest(item["prompt"])
                    if (model, prompt_hash) in prompt_hashes:
                        raise RuntimeError("Duplicate Phase400 prompt within model")
                    prompt_hashes.add((model, prompt_hash))
                    rows.append(
                        {
                            "schema_version": "74.1.0",
                            "phase_id": "Phase400-Protocol",
                            "created_at": created_at,
                            "frozen_partial_order_protocol_sha256": contract_hash,
                            "private_execution_model": model,
                            "anonymous_model_id": "p400m_" + digest(model, 12),
                            "blind_case_id": "p400c_" + digest(f"{model}:{group_id}:{condition}", 26),
                            "anonymous_parallel_group_id": group_id,
                            "anonymous_group_id": "p400s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition,
                            "candidate_split_private": split,
                            "selection_priority_private": selection_priority,
                            "group_priority": accepted,
                            "family_id": FAMILY_BY_SURFACE[surface],
                            "mechanism_id": surface,
                            "semantic_group_id": f"p400_private_{surface}_{seed:05d}",
                            "contrast_condition": condition,
                            "task_surface_private": surface,
                            "prompt": item["prompt"],
                            "raw_prompt": item["raw_prompt"],
                            "source_fragment": item["source_fragment"],
                            "query_fragment": item["query_fragment"],
                            "clause_fragments_private": item["clause_fragments_private"],
                            "tokenization_add_special_tokens": item["tokenization_add_special_tokens"],
                            "interface": item["interface"],
                            "answer_phase": item["answer_phase"],
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "axis_private": item["axis_private"],
                            "relation_level_private": item["relation_level_private"],
                            "order_level_private": item["order_level_private"],
                            "query_level_private": item["query_level_private"],
                            "semantic_role_fragments_private": item["semantic_role_fragments_private"],
                            "semantic_slot_fragments_private": item["semantic_slot_fragments_private"],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                        }
                    )
            accepted += 1
            print(
                f"[phase400/{surface}] accepted={accepted}/{CANDIDATE_GROUPS_PER_SURFACE} seed={seed}",
                flush=True,
            )
            seed += 1
        if accepted != CANDIDATE_GROUPS_PER_SURFACE:
            raise RuntimeError(
                f"Could not freeze Phase400 groups for {surface}: {accepted}; rejects={dict(rejects)}"
            )

    expected = len(SURFACES) * CANDIDATE_GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase400 row count {len(rows)} != {expected}")
    write_jsonl(OUT / "protocol/private/phase400_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase400_blind_group_registry.jsonl", groups)
    payload = {
        "schema_version": "74.1.0",
        "phase_id": "Phase400-Protocol",
        "created_at": created_at,
        "objective": "build_a_fresh_dynamic_partial_order_graph_denominator",
        "frozen_partial_order_protocol_sha256": contract_hash,
        "field_contract_pilot": {
            "case_count": 144,
            "strict_correct_count": 144,
            "complete_group_model_cells": 9,
            "formal_denominator_overlap": False,
        },
        "denominator": {
            "task_surfaces": list(SURFACES),
            "candidate_groups_per_surface": CANDIDATE_GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "candidate_split_group_counts": SPLIT_CANDIDATE_COUNTS,
            "selected_split_group_counts_if_eligible": SPLIT_SELECTED_COUNTS,
        },
        "authorization": {
            "run_behavior_qualification": True,
            "run_internal_before_behavior_freeze": False,
            "run_head_channel_or_neuron_scan": False,
        },
        "claim_boundary": {
            "behavior_factorial_is_a_partial_order_graph": False,
            "redesigned_field_contract_proves_a_field_mechanism": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase400_protocol.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
