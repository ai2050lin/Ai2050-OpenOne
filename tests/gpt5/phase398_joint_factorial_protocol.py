#!/usr/bin/env python3
"""Freeze fresh Phase398 2x2x2 joint-binding factorial cases."""

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


OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
P397 = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling", "coreference_resolution")
AXES = ("X", "Y")
LEVELS = (0, 1)
CONDITIONS = tuple(
    f"{axis}_R{relation}_O{order}_Q{query}"
    for axis in AXES
    for relation in LEVELS
    for order in LEVELS
    for query in LEVELS
)
GROUPS_PER_SURFACE = 24
MINIMUM_QUALIFIED_GROUPS = 16
SPLIT_COUNTS = {"discovery": 8, "calibration": 4, "physical_holdout": 4}
FAMILY_BY_SURFACE = {
    "possession_relation": "content_knowledge",
    "role_filling": "language_action",
    "coreference_resolution": "reasoning_constraint",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str, length: int = 64) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def values_for(surface: str) -> tuple[str, ...]:
    return tuple(ROLE_VALUES if surface == "role_filling" else OBJECTS)


def clause(surface: str, entity: str, value: str) -> str:
    if surface == "possession_relation":
        return f"For {entity}, the owned item is {value}."
    if surface == "role_filling":
        return f"For {entity}, the assigned role is {value}."
    return f"{entity} said they selected {value}."


def query_fragment(surface: str, entity: str) -> str:
    if surface == "possession_relation":
        return f"What item does {entity} own?"
    if surface == "role_filling":
        return f"What role is assigned to {entity}?"
    return f"What did {entity} select?"


def parse_condition(condition: str) -> tuple[str, int, int, int]:
    axis, relation, order, query = condition.split("_")
    return axis, int(relation[1]), int(order[1]), int(query[1])


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
    clauses = [clause(surface, slot_entities[index], slot_values[index]) for index in range(2)]
    source = " ".join(clauses)
    queried = entity_b if query else entity_a
    binding = dict(zip(slot_entities, slot_values))
    target = binding[queried]
    distractor = next(value for value in slot_values if value != target)
    query_text = query_fragment(surface, queried)
    prompt = (
        f"Phase398 natural joint record. KEY={group_tag}. {source}\n"
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
    prompt, add_special, answer_phase = interface_prompt(tokenizer, model, item["raw_prompt"], "answer_aligned_chat")
    return {
        **item,
        "prompt": prompt,
        "tokenization_add_special_tokens": add_special,
        "interface": "answer_aligned_chat",
        "answer_phase": answer_phase,
        "condition": args[2],
    }


def source_positions(tokenizer: Any, case: dict[str, Any], fragment: str) -> list[int]:
    ids = prompt_token_ids(tokenizer, case)
    source = set(fragment_positions(tokenizer, ids, case["source_fragment"]))
    return [position for position in fragment_positions(tokenizer, ids, fragment) if position in source]


def query_positions(tokenizer: Any, case: dict[str, Any], fragment: str) -> list[int]:
    ids = prompt_token_ids(tokenizer, case)
    query = set(fragment_positions(tokenizer, ids, case["query_fragment"]))
    return [position for position in fragment_positions(tokenizer, ids, fragment) if position in query]


def relation_audit(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids, right_ids = prompt_token_ids(tokenizer, left), prompt_token_ids(tokenizer, right)
    if len(left_ids) != len(right_ids) or Counter(left_ids) != Counter(right_ids):
        return {"valid": False, "reason": "relation_pair_token_contract"}
    diffs = [index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]]
    value_positions = {}
    for key in ("value_a", "value_b"):
        literal = left["semantic_slot_fragments_private"][key]
        mapped = source_positions(tokenizer, left, literal)
        right_mapped = source_positions(tokenizer, right, literal)
        if not mapped or mapped != right_mapped or any(left_ids[position] != right_ids[position] for position in mapped):
            return {"valid": False, "reason": "relation_value_position_or_token_changed", "key": key}
        value_positions[key] = mapped
    entities = set()
    for key in ("entity_a", "entity_b"):
        entities.update(source_positions(tokenizer, left, left["semantic_slot_fragments_private"][key]))
    if not diffs or not set(diffs).issubset(entities):
        return {"valid": False, "reason": "relation_difference_outside_entity_slots"}
    return {"valid": True, "diff_positions": diffs, "value_positions": value_positions}


def order_audit(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids, right_ids = prompt_token_ids(tokenizer, left), prompt_token_ids(tokenizer, right)
    if len(left_ids) != len(right_ids) or Counter(left_ids) != Counter(right_ids):
        return {"valid": False, "reason": "order_pair_token_contract"}
    moved = False
    for key in ("value_a", "value_b"):
        literal = left["semantic_slot_fragments_private"][key]
        a = source_positions(tokenizer, left, literal)
        b = source_positions(tokenizer, right, literal)
        if not a or not b:
            return {"valid": False, "reason": "order_value_missing", "key": key}
        moved = moved or a != b
    return {"valid": moved, "reason": None if moved else "order_did_not_move_values"}


def query_audit(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids, right_ids = prompt_token_ids(tokenizer, left), prompt_token_ids(tokenizer, right)
    if len(left_ids) != len(right_ids) or left["source_fragment"] != right["source_fragment"]:
        return {"valid": False, "reason": "query_source_or_length_changed"}
    diffs = [index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]]
    allowed = set()
    for case in (left, right):
        for key in ("entity_a", "entity_b"):
            allowed.update(query_positions(tokenizer, case, case["semantic_slot_fragments_private"][key]))
    if not diffs or not set(diffs).issubset(allowed):
        return {"valid": False, "reason": "query_difference_outside_query_entity"}
    return {"valid": True, "diff_positions": diffs}


def lexical_audit(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids, right_ids = prompt_token_ids(tokenizer, left), prompt_token_ids(tokenizer, right)
    if len(left_ids) != len(right_ids):
        return {"valid": False, "reason": "lexical_length_changed"}
    diffs = [index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]]
    allowed = set()
    positions = {}
    for prefix, case in (("x", left), ("y", right)):
        for key in ("value_a", "value_b"):
            mapped = source_positions(tokenizer, case, case["semantic_slot_fragments_private"][key])
            if not mapped:
                return {"valid": False, "reason": "lexical_value_missing"}
            positions[f"{prefix}_{key}"] = mapped
            allowed.update(mapped)
    if not diffs or not set(diffs).issubset(allowed):
        return {"valid": False, "reason": "lexical_difference_outside_values"}
    if positions["x_value_a"] != positions["y_value_a"] or positions["x_value_b"] != positions["y_value_b"]:
        return {"valid": False, "reason": "lexical_positions_changed"}
    return {"valid": True, "diff_positions": diffs}


def previous_signatures() -> set[tuple[str, str, str, str, str]]:
    path = P397 / "protocol/private/phase397_candidate_cases.jsonl"
    if not path.is_file():
        return set()
    signatures = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["private_execution_model"] != "qwen3" or row["anonymous_condition_slot"] not in {"A", "F"}:
            continue
        slots = row["semantic_slot_fragments_private"]
        signatures.add((row["task_surface_private"], slots["entity_a"], slots["entity_b"], slots["value_a"], slots["value_b"]))
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
    entities = [entity_pool[(seed * 5 + offset * 11 + len(surface)) % len(entity_pool)] for offset in range(2)]
    values = [value_pool[(seed * 7 + offset * 5 + len(surface)) % len(value_pool)] for offset in range(4)]
    if len(set(entities)) != 2 or len(set(values)) != 4:
        raise ValueError("lexical_collision")
    return (*entities, *values)


def main() -> None:
    created_at = now()
    tokenizers = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
    prior = previous_signatures()
    entity_pool = compatible_pool(tuple(NAMES), tokenizers)
    value_pools = {
        surface: compatible_pool(values_for(surface), tokenizers)
        for surface in SURFACES
    }
    if len(entity_pool) < 8 or any(len(pool) < 8 for pool in value_pools.values()):
        raise RuntimeError("Insufficient crossmodel token-width-compatible lexical pools")
    rows, groups = [], []
    prompt_hashes: set[tuple[str, str]] = set()
    for surface in SURFACES:
        accepted, seed = 0, 12000
        rejection_counts: Counter[str] = Counter()
        while accepted < GROUPS_PER_SURFACE and seed < 40000:
            try:
                entity_a, entity_b, v1, v2, v3, v4 = lexical_items(
                    surface, seed, entity_pool, value_pools[surface]
                )
            except ValueError:
                seed += 1
                continue
            if any((surface, entity_a, entity_b, a, b) in prior for a, b in ((v1, v2), (v3, v4))):
                rejection_counts["phase397_signature_overlap"] += 1
                seed += 1
                continue
            group_tag = f"P398{surface[:3].upper()}{accepted:02d}"
            per_model, model_audits, valid = {}, {}, True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases = {}
                for condition in CONDITIONS:
                    axis, _r, _o, _q = parse_condition(condition)
                    a, b = (v1, v2) if axis == "X" else (v3, v4)
                    cases[condition] = render(tokenizer, model, surface, group_tag, condition, entity_a, entity_b, a, b)
                audits = {"relation": [], "order": [], "query": [], "lexical": []}
                for axis in AXES:
                    for order in LEVELS:
                        for query in LEVELS:
                            audits["relation"].append(relation_audit(tokenizer, cases[f"{axis}_R0_O{order}_Q{query}"], cases[f"{axis}_R1_O{order}_Q{query}"]))
                    for relation in LEVELS:
                        for query in LEVELS:
                            audits["order"].append(order_audit(tokenizer, cases[f"{axis}_R{relation}_O0_Q{query}"], cases[f"{axis}_R{relation}_O1_Q{query}"]))
                        for order in LEVELS:
                            audits["query"].append(query_audit(tokenizer, cases[f"{axis}_R{relation}_O{order}_Q0"], cases[f"{axis}_R{relation}_O{order}_Q1"]))
                for relation in LEVELS:
                    for order in LEVELS:
                        for query in LEVELS:
                            audits["lexical"].append(lexical_audit(tokenizer, cases[f"X_R{relation}_O{order}_Q{query}"], cases[f"Y_R{relation}_O{order}_Q{query}"]))
                invalid_audits = [item for items in audits.values() for item in items if not item["valid"]]
                if invalid_audits:
                    rejection_counts[f"{model}:{invalid_audits[0]['reason']}"] += 1
                    valid = False
                    break
                targets = []
                for condition in CONDITIONS:
                    token_ids = tokenizer(" " + cases[condition]["target"], add_special_tokens=False)["input_ids"]
                    if not token_ids:
                        valid = False
                        break
                    targets.append(token_ids[0])
                if not valid or len(set(targets)) != 4:
                    rejection_counts[f"{model}:target_first_token_not_four_way_distinct"] += 1
                    valid = False
                    break
                per_model[model] = cases
                model_audits[model] = audits
            if not valid:
                seed += 1
                if seed % 250 == 0:
                    print(f"[phase398/{surface}] seed={seed} accepted={accepted} rejects={dict(rejection_counts.most_common(5))}", flush=True)
                continue
            group_id = "p398g_" + digest(f"phase398:{surface}:{seed}", 24)
            groups.append({
                "schema_version": "72.0.0", "phase_id": "Phase398-Protocol",
                "anonymous_parallel_group_id": group_id, "task_surface": surface,
                "group_priority": accepted, "condition_count": len(CONDITIONS),
                "fresh_against_phase397_pair_signatures": True,
                "relation_pairs_fix_value_identity_and_position": True,
                "order_pairs_move_whole_clauses_and_literal_positions": True,
                "query_pairs_fix_source_prefix": True,
                "two_lexical_axes_complete_factorial": True,
            })
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    prompt_hash = digest(item["prompt"])
                    if (model, prompt_hash) in prompt_hashes:
                        raise RuntimeError("Duplicate Phase398 prompt within model")
                    prompt_hashes.add((model, prompt_hash))
                    rows.append({
                        "schema_version": "72.0.0", "phase_id": "Phase398-Protocol", "created_at": created_at,
                        "private_execution_model": model, "anonymous_model_id": "p398m_" + digest(model, 12),
                        "blind_case_id": "p398c_" + digest(f"{model}:{group_id}:{condition}", 26),
                        "anonymous_parallel_group_id": group_id,
                        "anonymous_group_id": "p398s_" + digest(surface, 12),
                        "anonymous_condition_slot": condition,
                        "group_priority": accepted,
                        "family_id": FAMILY_BY_SURFACE[surface], "mechanism_id": surface,
                        "semantic_group_id": f"p398_private_{surface}_{seed:05d}",
                        "contrast_condition": condition, "task_surface_private": surface,
                        "prompt": item["prompt"], "raw_prompt": item["raw_prompt"],
                        "source_fragment": item["source_fragment"], "query_fragment": item["query_fragment"],
                        "tokenization_add_special_tokens": item["tokenization_add_special_tokens"],
                        "interface": item["interface"], "answer_phase": item["answer_phase"],
                        "target": item["target"], "target_aliases": item["target_aliases"], "distractors": item["distractors"],
                        "axis_private": item["axis_private"],
                        "relation_level_private": item["relation_level_private"],
                        "order_level_private": item["order_level_private"],
                        "query_level_private": item["query_level_private"],
                        "semantic_role_fragments_private": item["semantic_role_fragments_private"],
                        "semantic_slot_fragments_private": item["semantic_slot_fragments_private"],
                        "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                    })
            accepted += 1
            print(f"[phase398/{surface}] accepted={accepted}/{GROUPS_PER_SURFACE} seed={seed}", flush=True)
            seed += 1
        if accepted != GROUPS_PER_SURFACE:
            raise RuntimeError(f"Could not freeze Phase398 groups for {surface}: {accepted}; rejects={dict(rejection_counts)}")
    expected = len(SURFACES) * GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase398 row count {len(rows)} != {expected}")
    write_jsonl(OUT / "protocol/private/phase398_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase398_blind_group_registry.jsonl", groups)
    protocol = {
        "schema_version": "72.0.0", "phase_id": "Phase398-Protocol", "created_at": created_at,
        "objective": "map_relation_order_query_interactions_at_query_integration_without_assuming_a_relation_vector",
        "phase397_audit": {
            "stable_relation_signature_retained": True,
            "portable_same_position_relation_carrier_rejected": True,
            "content_or_order_globally_dominates_relation": False,
            "reason": "dominance was observed only under the frozen Phase397 intervention and margin boundary",
        },
        "phase398_design_corrections": {
            "eight_factorial_plus_two_lexical_axes_is_twelve_conditions": False,
            "complete_factorial_condition_count": 16,
            "sentence_order_can_keep_all_absolute_value_positions_fixed": False,
            "relation_pairs_fix_positions_within_each_order_level": True,
            "syntax_replication_is_a_later_independent_layer": True,
            "unqualified_phase397_surfaces_enter_internal_trace": False,
        },
        "denominator": {
            "task_surfaces": list(SURFACES), "groups_per_surface": GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS), "models": list(MODELS),
            "candidate_case_count": len(rows), "minimum_qualified_groups_per_surface": MINIMUM_QUALIFIED_GROUPS,
            "split_group_counts_per_eligible_surface": SPLIT_COUNTS,
        },
        "factor_contract": {
            "relation": "within each fixed order/query cell, only source entity slots swap and value positions remain fixed",
            "order": "whole source clauses reverse; value identity is mapped but absolute positions intentionally change",
            "query": "source prefix remains identical and only the queried entity changes",
            "lexical_replication": "the complete 2x2x2 factorial is repeated on X and Y value axes",
        },
        "authorization": {"run_behavior_qualification": True, "run_internal_before_behavior_freeze": False, "run_single_neuron_scan": False},
        "claim_boundary": {"factorial_behavior_is_joint_binding_state": False, "interaction_contrast_is_causal_operator": False, "language_encoding_closed": False},
    }
    write_json(OUT / "phase398_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
