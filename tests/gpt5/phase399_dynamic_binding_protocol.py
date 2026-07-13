#!/usr/bin/env python3
"""Freeze the fresh Phase399 four-surface dynamic-binding denominator."""

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
from phase398_joint_factorial_protocol import (  # noqa: E402
    AXES,
    LEVELS,
    lexical_audit,
    order_audit,
    parse_condition,
    query_audit,
    relation_audit,
)


OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
P398 = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "possession_relation",
    "role_filling",
    "coreference_resolution",
    "field_extraction",
)
CONDITIONS = tuple(
    f"{axis}_R{relation}_O{order}_Q{query}"
    for axis in AXES
    for relation in LEVELS
    for order in LEVELS
    for query in LEVELS
)
CANDIDATE_GROUPS_PER_SURFACE = 28
MINIMUM_QUALIFIED_GROUPS = 20
SPLIT_COUNTS = {"discovery": 10, "calibration": 5, "physical_holdout": 5}
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


def clause(surface: str, entity: str, value: str) -> str:
    if surface == "possession_relation":
        return f"For {entity}, the owned item is {value}."
    if surface == "role_filling":
        return f"For {entity}, the assigned role is {value}."
    if surface == "coreference_resolution":
        return f"{entity} said they selected {value}."
    return f"Record {entity} has the item field {value}."


def query_fragment(surface: str, entity: str) -> str:
    if surface == "possession_relation":
        return f"What item does {entity} own?"
    if surface == "role_filling":
        return f"What role is assigned to {entity}?"
    if surface == "coreference_resolution":
        return f"What did {entity} select?"
    return f"Extract the item field for record {entity}."


def raw_case(
    surface: str,
    group_tag: str,
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
    clauses = [
        clause(surface, slot_entities[index], slot_values[index])
        for index in range(2)
    ]
    source = " ".join(clauses)
    queried = entity_b if query else entity_a
    binding = dict(zip(slot_entities, slot_values))
    target = binding[queried]
    distractor = next(value for value in slot_values if value != target)
    query_text = query_fragment(surface, queried)
    prompt = (
        f"Phase399 dynamic record. KEY={group_tag}. {source}\n"
        f"Question: {query_text}\n"
        "Return only the lowercase answer word.\nAnswer:"
    )
    return {
        "raw_prompt": prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": [distractor],
        "source_fragment": source,
        "query_fragment": query_text,
        "clause_fragments_private": clauses,
        "semantic_role_fragments_private": {
            "entities": [entity_a, entity_b],
            "attributes_items": [value_a, value_b],
            "relations": [item.rstrip(".") for item in clauses],
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
        "condition": args[2],
    }


def previous_signatures() -> set[tuple[str, str, str, str, str]]:
    path = P398 / "protocol/private/phase398_candidate_cases.jsonl"
    signatures: set[tuple[str, str, str, str, str]] = set()
    if not path.is_file():
        return signatures
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row["private_execution_model"] != "qwen3"
            or row["anonymous_condition_slot"] not in {"X_R0_O0_Q0", "Y_R0_O0_Q0"}
        ):
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


def compatible_pool(items: tuple[str, ...], tokenizers: dict[str, Any]) -> tuple[str, ...]:
    buckets: dict[tuple[int, ...], list[str]] = {}
    for item in items:
        signature = tuple(
            len(tokenizer(" " + item, add_special_tokens=False)["input_ids"])
            for tokenizer in tokenizers.values()
        )
        buckets.setdefault(signature, []).append(item)
    return tuple(max(buckets.values(), key=len))


def lexical_items(
    surface: str,
    seed: int,
    entity_pool: tuple[str, ...],
    value_pool: tuple[str, ...],
) -> tuple[str, ...]:
    entities = [
        entity_pool[(seed * 13 + offset * 17 + len(surface)) % len(entity_pool)]
        for offset in range(2)
    ]
    values = [
        value_pool[(seed * 19 + offset * 7 + len(surface)) % len(value_pool)]
        for offset in range(4)
    ]
    if len(set(entities)) != 2 or len(set(values)) != 4:
        raise ValueError("lexical_collision")
    return (*entities, *values)


def main() -> None:
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
        surface: compatible_pool(values_for(surface), tokenizers)
        for surface in SURFACES
    }
    if len(entity_pool) < 8 or any(len(pool) < 8 for pool in value_pools.values()):
        raise RuntimeError("Insufficient cross-model token-width-compatible lexical pools")

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    prompt_hashes: set[tuple[str, str]] = set()
    for surface in SURFACES:
        accepted, seed = 0, 41000 + 1000 * SURFACES.index(surface)
        rejection_counts: Counter[str] = Counter()
        while accepted < CANDIDATE_GROUPS_PER_SURFACE and seed < 90000:
            try:
                entity_a, entity_b, v1, v2, v3, v4 = lexical_items(
                    surface, seed, entity_pool, value_pools[surface]
                )
            except ValueError:
                seed += 1
                continue
            if any(
                (surface, entity_a, entity_b, a, b) in prior
                for a, b in ((v1, v2), (v3, v4))
            ):
                rejection_counts["phase398_signature_overlap"] += 1
                seed += 1
                continue
            group_tag = f"P399{surface[:3].upper()}{accepted:02d}"
            per_model: dict[str, dict[str, Any]] = {}
            valid = True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases = {
                    condition: render(
                        tokenizer,
                        model,
                        surface,
                        group_tag,
                        condition,
                        entity_a,
                        entity_b,
                        *( (v1, v2) if condition.startswith("X_") else (v3, v4) ),
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
                    rejection_counts[f"{model}:{invalid['reason']}"] += 1
                    valid = False
                    break
                targets = [
                    tokenizer(" " + cases[condition]["target"], add_special_tokens=False)[
                        "input_ids"
                    ][0]
                    for condition in CONDITIONS
                ]
                if len(set(targets)) != 4:
                    rejection_counts[f"{model}:target_first_token_not_four_way_distinct"] += 1
                    valid = False
                    break
                per_model[model] = cases
            if not valid:
                seed += 1
                continue

            group_id = "p399g_" + digest(f"phase399:{surface}:{seed}", 24)
            groups.append(
                {
                    "schema_version": "73.0.0",
                    "phase_id": "Phase399-Protocol",
                    "anonymous_parallel_group_id": group_id,
                    "task_surface": surface,
                    "group_priority": accepted,
                    "condition_count": len(CONDITIONS),
                    "fresh_against_phase398_pair_signatures": True,
                    "relation_pairs_fix_value_identity_and_position": True,
                    "order_factor_is_order_position_geometry": True,
                    "query_pairs_fix_source_prefix": True,
                    "two_lexical_axes_complete_factorial": True,
                }
            )
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    prompt_hash = digest(item["prompt"])
                    if (model, prompt_hash) in prompt_hashes:
                        raise RuntimeError("Duplicate Phase399 prompt within model")
                    prompt_hashes.add((model, prompt_hash))
                    rows.append(
                        {
                            "schema_version": "73.0.0",
                            "phase_id": "Phase399-Protocol",
                            "created_at": created_at,
                            "private_execution_model": model,
                            "anonymous_model_id": "p399m_" + digest(model, 12),
                            "blind_case_id": "p399c_"
                            + digest(f"{model}:{group_id}:{condition}", 26),
                            "anonymous_parallel_group_id": group_id,
                            "anonymous_group_id": "p399s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition,
                            "group_priority": accepted,
                            "family_id": FAMILY_BY_SURFACE[surface],
                            "mechanism_id": surface,
                            "semantic_group_id": f"p399_private_{surface}_{seed:05d}",
                            "contrast_condition": condition,
                            "task_surface_private": surface,
                            "prompt": item["prompt"],
                            "raw_prompt": item["raw_prompt"],
                            "source_fragment": item["source_fragment"],
                            "query_fragment": item["query_fragment"],
                            "clause_fragments_private": item["clause_fragments_private"],
                            "tokenization_add_special_tokens": item[
                                "tokenization_add_special_tokens"
                            ],
                            "interface": item["interface"],
                            "answer_phase": item["answer_phase"],
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "axis_private": item["axis_private"],
                            "relation_level_private": item["relation_level_private"],
                            "order_level_private": item["order_level_private"],
                            "query_level_private": item["query_level_private"],
                            "semantic_role_fragments_private": item[
                                "semantic_role_fragments_private"
                            ],
                            "semantic_slot_fragments_private": item[
                                "semantic_slot_fragments_private"
                            ],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                        }
                    )
            accepted += 1
            print(
                f"[phase399/{surface}] accepted={accepted}/"
                f"{CANDIDATE_GROUPS_PER_SURFACE} seed={seed}",
                flush=True,
            )
            seed += 1
        if accepted != CANDIDATE_GROUPS_PER_SURFACE:
            raise RuntimeError(
                f"Could not freeze Phase399 groups for {surface}: {accepted}; "
                f"rejects={dict(rejection_counts)}"
            )

    expected = len(SURFACES) * CANDIDATE_GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase399 row count {len(rows)} != {expected}")
    write_jsonl(OUT / "protocol/private/phase399_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase399_blind_group_registry.jsonl", groups)
    protocol = {
        "schema_version": "73.0.0",
        "phase_id": "Phase399-Protocol",
        "created_at": created_at,
        "objective": "build_and_test_a_multi_position_multi_component_dynamic_binding_graph",
        "phase398_audit": {
            "main_judgment_correct": True,
            "stable_roq_is_observational": True,
            "single_query_position_sufficient_state_rejected": True,
            "nine_cells_are_not_nine_independent_mechanisms": True,
            "ninety_six_item_catalog_is_not_a_completion_denominator": True,
            "global_progress_percentage_is_not_an_experimental_statistic": True,
        },
        "denominator": {
            "task_surfaces": list(SURFACES),
            "candidate_groups_per_surface": CANDIDATE_GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "minimum_qualified_groups_per_surface": MINIMUM_QUALIFIED_GROUPS,
            "split_group_counts_per_eligible_surface": SPLIT_COUNTS,
        },
        "resolution_ladder": [
            "multi_position_parent_components",
            "source_role_to_query_attention_events",
            "discovery_frozen_dynamic_chain",
            "independent_calibration",
            "one_shot_physical_holdout",
            "joint_damage_and_layered_restoration_if_authorized",
            "head_channel_or_neuron_resolution_only_after_aggregate_causal_gate",
        ],
        "semantic_times": {
            "source_and_query_encoding": "single exact full-prompt causal forward",
            "first_answer_decision": "logits immediately before the first answer token",
            "target_completion": "teacher-forced replay immediately before the target completion token",
            "post_target": "teacher-forced replay after the complete target token sequence",
        },
        "authorization": {
            "run_behavior_qualification": True,
            "run_internal_before_behavior_freeze": False,
            "run_head_scan": False,
            "run_channel_scan": False,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "roq_is_binding_algorithm": False,
            "directed_observation_edge_is_causal_edge": False,
            "aggregate_dynamic_chain_is_complete_language_path": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase399_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
