#!/usr/bin/env python3
"""Freeze the Phase397 multitask relation-binding behavior denominator."""

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
from phase386_multitime_protocol import LABELS, NAMES, OBJECTS  # noqa: E402
from phase390_role_mapping import fragment_positions, prompt_token_ids  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "field_extraction",
    "possession_relation",
    "spatial_relation",
    "role_filling",
    "coreference_resolution",
    "event_state_update",
)
CONDITIONS = (
    "A_x_direct",
    "B_x_relation_swap",
    "C_x_order_control",
    "D_x_syntax_control",
    "E_x_query_switch",
    "F_y_direct",
    "G_y_relation_swap",
    "H_y_order_control",
    "I_y_syntax_control",
    "J_y_query_switch",
)
GROUPS_PER_SURFACE = 24
MINIMUM_QUALIFIED_GROUPS = 16
SPLIT_COUNTS = {"discovery": 8, "calibration": 4, "physical_holdout": 4}
ROLE_VALUES = (
    "pilot", "scribe", "guard", "judge", "medic", "chef", "clerk", "guide",
    "host", "scout", "miner", "baker", "nurse", "coach", "agent", "mayor",
    "actor", "driver", "farmer", "lawyer", "singer", "writer", "artist", "tailor",
    "porter", "keeper", "planner", "reader", "speaker", "worker", "captain", "doctor",
    "editor", "teacher", "vendor", "builder", "manager", "analyst", "officer", "leader",
)
FAMILY_BY_SURFACE = {
    "field_extraction": "content_knowledge",
    "possession_relation": "content_knowledge",
    "spatial_relation": "reasoning_constraint",
    "role_filling": "language_action",
    "coreference_resolution": "reasoning_constraint",
    "event_state_update": "state_drift",
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def value_pool(surface: str) -> tuple[str, ...]:
    if surface in {"possession_relation", "coreference_resolution"}:
        return tuple(OBJECTS)
    if surface == "role_filling":
        return ROLE_VALUES
    return tuple(LABELS)


def canonical_clause(surface: str, entity: str, value: str, old: str) -> str:
    if surface == "field_extraction":
        return f"For {entity}, the status is {value}."
    if surface == "possession_relation":
        return f"For {entity}, the owned item is {value}."
    if surface == "spatial_relation":
        return f"For {entity}, the location is {value}."
    if surface == "role_filling":
        return f"For {entity}, the assigned role is {value}."
    if surface == "coreference_resolution":
        return f"{entity} said they selected {value}."
    return f"For {entity}, the marker changed from {old} to {value}."


def syntax_clause(surface: str, entity: str, value: str, old: str) -> str:
    if surface == "field_extraction":
        return f"{entity}'s status is {value}."
    if surface == "possession_relation":
        return f"{entity} owns the {value}."
    if surface == "spatial_relation":
        return f"{entity} can be found at {value}."
    if surface == "role_filling":
        return f"{entity} serves as the {value}."
    if surface == "coreference_resolution":
        return f"The selection reported by {entity} was {value}."
    return f"{entity} previously had {old}, and now has marker {value}."


def query_fragment(surface: str, entity: str) -> str:
    if surface == "field_extraction":
        return f"What is the status of {entity}?"
    if surface == "possession_relation":
        return f"What item does {entity} own?"
    if surface == "spatial_relation":
        return f"Where is {entity} located?"
    if surface == "role_filling":
        return f"What role is assigned to {entity}?"
    if surface == "coreference_resolution":
        return f"What did {entity} select?"
    return f"What is the current marker for {entity}?"


def raw_case(
    surface: str,
    group_tag: str,
    condition: str,
    entity_a: str,
    entity_b: str,
    value_a: str,
    value_b: str,
    old_a: str,
    old_b: str,
) -> dict[str, Any]:
    relation_swap = condition[0] in "BG"
    order_control = condition[0] in "CH"
    syntax_control = condition[0] in "DI"
    query_switch = condition[0] in "EJ"

    slot_entity_a, slot_entity_b = (entity_b, entity_a) if relation_swap else (entity_a, entity_b)
    clause = syntax_clause if syntax_control else canonical_clause
    first = clause(surface, slot_entity_a, value_a, old_a)
    second = clause(surface, slot_entity_b, value_b, old_b)
    if order_control:
        first = clause(surface, entity_b, value_b, old_b)
        second = clause(surface, entity_a, value_a, old_a)
    source = f"{first} {second}"
    queried = entity_b if query_switch else entity_a
    query = query_fragment(surface, queried)

    binding = {slot_entity_a: value_a, slot_entity_b: value_b}
    if order_control or syntax_control or query_switch:
        binding = {entity_a: value_a, entity_b: value_b}
    target = binding[queried]
    other = value_b if target == value_a else value_a
    raw_prompt = (
        f"Phase397 natural relation record. KEY={group_tag}. {source}\n"
        f"Question: {query}\n"
        "Return only the lowercase answer word.\nAnswer:"
    )
    return {
        "raw_prompt": raw_prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": [other],
        "source_fragment": source,
        "query_fragment": query,
        "semantic_role_fragments_private": {
            "entities": [entity_a, entity_b],
            "attributes_items": [value_a, value_b],
            "relations": [first.rstrip("."), second.rstrip(".")],
            "query_keywords": ["Question:"],
        },
        "semantic_slot_fragments_private": {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "value_a": value_a,
            "value_b": value_b,
            "old_a": old_a,
            "old_b": old_b,
            "query_entity": queried,
        },
        "axis_private": "X" if condition[0] in "ABCDE" else "Y",
        "relation_swap_private": relation_swap,
        "order_control_private": order_control,
        "syntax_control_private": syntax_control,
        "query_switch_private": query_switch,
    }


def rendered_case(tokenizer: Any, model: str, *args: Any) -> dict[str, Any]:
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


def all_positions(tokenizer: Any, ids: list[int], literal: str) -> list[int]:
    return fragment_positions(tokenizer, ids, literal)


def source_positions(tokenizer: Any, case: dict[str, Any], literal: str) -> list[int]:
    ids = prompt_token_ids(tokenizer, case)
    source = all_positions(tokenizer, ids, case["source_fragment"])
    if not source:
        return []
    source_set = set(source)
    return [position for position in all_positions(tokenizer, ids, literal) if position in source_set]


def relation_pair_audit(tokenizer: Any, direct: dict[str, Any], swapped: dict[str, Any]) -> dict[str, Any]:
    left_ids = prompt_token_ids(tokenizer, direct)
    right_ids = prompt_token_ids(tokenizer, swapped)
    if len(left_ids) != len(right_ids):
        return {"valid": False, "reason": "prompt_token_count_mismatch"}
    if Counter(left_ids) != Counter(right_ids):
        return {"valid": False, "reason": "prompt_token_multiset_mismatch"}
    diffs = [index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]]
    if not diffs:
        return {"valid": False, "reason": "no_relation_difference"}
    value_positions: dict[str, list[int]] = {}
    for key in ("value_a", "value_b"):
        literal = direct["semantic_slot_fragments_private"][key]
        positions = source_positions(tokenizer, direct, literal)
        right_positions = source_positions(tokenizer, swapped, literal)
        if not positions or positions != right_positions:
            return {"valid": False, "reason": "literal_value_position_changed", "key": key}
        if any(left_ids[position] != right_ids[position] for position in positions):
            return {"valid": False, "reason": "literal_value_token_changed", "key": key}
        value_positions[key] = positions
    entity_positions: set[int] = set()
    for key in ("entity_a", "entity_b"):
        literal = direct["semantic_slot_fragments_private"][key]
        entity_positions.update(source_positions(tokenizer, direct, literal))
    if not set(diffs).issubset(entity_positions):
        return {"valid": False, "reason": "difference_outside_source_entity_slots"}
    return {
        "valid": True,
        "prompt_token_count": len(left_ids),
        "diff_positions": diffs,
        "source_entity_positions": sorted(entity_positions),
        "literal_value_positions": value_positions,
    }


def content_pair_audit(tokenizer: Any, axis_x: dict[str, Any], axis_y: dict[str, Any]) -> dict[str, Any]:
    x_ids = prompt_token_ids(tokenizer, axis_x)
    y_ids = prompt_token_ids(tokenizer, axis_y)
    if len(x_ids) != len(y_ids):
        return {"valid": False, "reason": "content_prompt_token_count_mismatch"}
    diffs = [index for index, pair in enumerate(zip(x_ids, y_ids)) if pair[0] != pair[1]]
    allowed: set[int] = set()
    positions: dict[str, list[int]] = {}
    for case, prefix in ((axis_x, "x"), (axis_y, "y")):
        for key in ("value_a", "value_b"):
            literal = case["semantic_slot_fragments_private"][key]
            mapped = source_positions(tokenizer, case, literal)
            if not mapped:
                return {"valid": False, "reason": "missing_content_value", "key": f"{prefix}_{key}"}
            positions[f"{prefix}_{key}"] = mapped
            allowed.update(mapped)
    if not diffs or not set(diffs).issubset(allowed):
        return {"valid": False, "reason": "content_difference_outside_value_slots"}
    if positions["x_value_a"] != positions["y_value_a"] or positions["x_value_b"] != positions["y_value_b"]:
        return {"valid": False, "reason": "content_value_positions_not_matched"}
    return {"valid": True, "diff_positions": diffs, "literal_value_positions": positions}


def first_target_token_distinct(tokenizer: Any, cases: dict[str, dict[str, Any]]) -> bool:
    tokens: list[int] = []
    for key in ("A_x_direct", "B_x_relation_swap", "F_y_direct", "G_y_relation_swap"):
        target = tokenizer(" " + cases[key]["target"], add_special_tokens=False)["input_ids"]
        if not target:
            return False
        tokens.append(target[0])
    return tokens[0] != tokens[1] and tokens[2] != tokens[3]


def lexical_items(surface: str, seed: int) -> tuple[str, ...]:
    pool = value_pool(surface)
    entities = [NAMES[(seed * 5 + offset * 11 + len(surface)) % len(NAMES)] for offset in range(2)]
    values = [pool[(seed * 7 + offset * 13 + len(surface)) % len(pool)] for offset in range(6)]
    if len(set(entities)) != 2 or len(set(values)) != 6:
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

    rows: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    prompt_hashes: set[tuple[str, str]] = set()
    for surface in SURFACES:
        accepted = 0
        seed = 0
        while accepted < GROUPS_PER_SURFACE and seed < 12000:
            try:
                entity_a, entity_b, v1, v2, v3, v4, old_a, old_b = lexical_items(surface, seed)
            except ValueError:
                seed += 1
                continue
            group_tag = f"{surface[:3].upper()}{accepted:02d}"
            per_model: dict[str, dict[str, dict[str, Any]]] = {}
            audits: dict[str, dict[str, Any]] = {}
            valid = True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases: dict[str, dict[str, Any]] = {}
                for condition in CONDITIONS:
                    use_y = condition[0] in "FGHIJ"
                    value_a, value_b = (v3, v4) if use_y else (v1, v2)
                    cases[condition] = rendered_case(
                        tokenizer,
                        model,
                        surface,
                        group_tag,
                        condition,
                        entity_a,
                        entity_b,
                        value_a,
                        value_b,
                        old_a,
                        old_b,
                    )
                relation_x = relation_pair_audit(tokenizer, cases["A_x_direct"], cases["B_x_relation_swap"])
                relation_y = relation_pair_audit(tokenizer, cases["F_y_direct"], cases["G_y_relation_swap"])
                content = content_pair_audit(tokenizer, cases["A_x_direct"], cases["F_y_direct"])
                if not relation_x["valid"] or not relation_y["valid"] or not content["valid"]:
                    valid = False
                    break
                if not first_target_token_distinct(tokenizer, cases):
                    valid = False
                    break
                per_model[model] = cases
                audits[model] = {"relation_x": relation_x, "relation_y": relation_y, "content": content}
            if not valid:
                seed += 1
                continue

            anonymous_group = "p397g_" + digest(f"phase397:{surface}:{seed}", 24)
            groups.append(
                {
                    "schema_version": "71.0.0",
                    "phase_id": "Phase397-Protocol",
                    "anonymous_parallel_group_id": anonymous_group,
                    "task_surface": surface,
                    "group_priority": accepted,
                    "relation_value_positions_fixed_all_models": True,
                    "relation_pair_token_multiset_fixed_all_models": True,
                    "content_value_positions_matched_all_models": True,
                    "condition_count": len(CONDITIONS),
                }
            )
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    relation_key = "relation_y" if item["axis_private"] == "Y" else "relation_x"
                    relation_audit = audits[model][relation_key]
                    prompt_hash = digest(item["prompt"])
                    if (model, prompt_hash) in prompt_hashes:
                        raise RuntimeError("Duplicate Phase397 prompt within model")
                    prompt_hashes.add((model, prompt_hash))
                    rows.append(
                        {
                            "schema_version": "71.0.0",
                            "phase_id": "Phase397-Protocol",
                            "created_at": created_at,
                            "private_execution_model": model,
                            "anonymous_model_id": "p397m_" + digest(model, 12),
                            "blind_case_id": "p397c_" + digest(f"{model}:{anonymous_group}:{condition}", 26),
                            "anonymous_parallel_group_id": anonymous_group,
                            "anonymous_group_id": "p397s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition[0],
                            "group_priority": accepted,
                            "family_id": FAMILY_BY_SURFACE[surface],
                            "mechanism_id": surface,
                            "semantic_group_id": f"p397_private_{surface}_{seed:05d}",
                            "contrast_condition": condition,
                            "task_surface_private": surface,
                            "prompt": item["prompt"],
                            "raw_prompt": item["raw_prompt"],
                            "source_fragment": item["source_fragment"],
                            "query_fragment": item["query_fragment"],
                            "tokenization_add_special_tokens": item["tokenization_add_special_tokens"],
                            "interface": item["interface"],
                            "answer_phase": item["answer_phase"],
                            "target": item["target"],
                            "target_aliases": item["target_aliases"],
                            "distractors": item["distractors"],
                            "axis_private": item["axis_private"],
                            "relation_swap_private": item["relation_swap_private"],
                            "order_control_private": item["order_control_private"],
                            "syntax_control_private": item["syntax_control_private"],
                            "query_switch_private": item["query_switch_private"],
                            "semantic_role_fragments_private": item["semantic_role_fragments_private"],
                            "semantic_slot_fragments_private": item["semantic_slot_fragments_private"],
                            "literal_value_positions_private": relation_audit["literal_value_positions"],
                            "relation_diff_positions_private": relation_audit["diff_positions"],
                            "source_entity_positions_private": relation_audit["source_entity_positions"],
                            "content_diff_positions_private": audits[model]["content"]["diff_positions"],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                        }
                    )
            accepted += 1
            seed += 1
        if accepted != GROUPS_PER_SURFACE:
            raise RuntimeError(f"Could not freeze {GROUPS_PER_SURFACE} groups for {surface}: {accepted}")

    expected = len(SURFACES) * GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase397 row count {len(rows)} != {expected}")
    write_jsonl(OUT / "protocol/private/phase397_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase397_blind_group_registry.jsonl", groups)
    protocol = {
        "schema_version": "71.0.0",
        "phase_id": "Phase397-Protocol",
        "created_at": created_at,
        "objective": "separate_relation_binding_from_content_position_order_syntax_query_and_task_factors",
        "phase396_evidence_audit": {
            "field_same_literal_context_state_sufficiency_retained": True,
            "wrong_depth_zero_proves_unique_formation_window": False,
            "query_writes_backward_into_source_value_state": False,
            "causal_direction": "source_value_context_to_query_integration_to_terminal_readout",
            "four_conditions_support_six_independent_factors": False,
        },
        "denominator": {
            "task_surfaces": list(SURFACES),
            "groups_per_surface": GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "minimum_fully_qualified_groups_per_surface": MINIMUM_QUALIFIED_GROUPS,
            "split_group_counts_per_eligible_surface": SPLIT_COUNTS,
        },
        "factor_controls": {
            "relation": "swap only source entity slots while literal values stay at identical token positions",
            "content_identity": "matched X/Y values at the same structural positions",
            "order": "reverse whole clauses while preserving entity-value binding",
            "syntax": "paraphrase binding while preserving entities, values and query",
            "query": "hold source fixed and ask for the other entity",
            "task": "repeat the same abstract contrasts over six natural task surfaces",
        },
        "split_rule": "surface eligible at >=16 fully qualified groups; hash-order 8/4/4 without replacement; reserve groups are not backfill",
        "externally_frozen_candidate_layers": {"qwen3": 20, "glm4": 22, "deepseek7b": 15},
        "externally_frozen_wrong_depth_layers": {"qwen3": 5, "glm4": 6, "deepseek7b": 4},
        "authorization": {
            "run_behavior_qualification": True,
            "run_internal_before_behavior_freeze": False,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "behavior_factorization_is_internal_mechanism": False,
            "aggregate_token_state_is_single_neuron": False,
            "six_task_replication_is_nine_family_layout": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase397_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
