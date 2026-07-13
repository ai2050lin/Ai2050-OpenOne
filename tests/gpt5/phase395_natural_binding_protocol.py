#!/usr/bin/env python3
"""Freeze natural same-token-multiset binding cases for Phase395."""

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


OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
TASK_SURFACES = ("field_extraction", "relation_qa", "entity_recency")
CONDITIONS = (
    "A_direct_lex_x",
    "B_swapped_lex_x",
    "C_direct_lex_y",
    "D_swapped_lex_y",
)
GROUPS_PER_SURFACE = 24
SPLIT_COUNTS = {"discovery": 12, "calibration": 6, "physical_holdout": 6}

TEMPLATES = {
    "field_extraction": (
        "Phase395 paired natural records. KEY={group_tag}; AXIS={axis}. "
        "The record for {entity_a} has status {assigned_a}. "
        "The record for {entity_b} has status {assigned_b}.\n"
        "Question: What is the status of {entity_a}?\n"
        "Return only the lowercase status word.\nAnswer:"
    ),
    "relation_qa": (
        "Phase395 paired location facts. KEY={group_tag}; AXIS={axis}. "
        "The {entity_a} is located in {assigned_a}. "
        "The {entity_b} is located in {assigned_b}.\n"
        "Question: Where is the {entity_a} located?\n"
        "Return only the lowercase location word.\nAnswer:"
    ),
    "entity_recency": (
        "Phase395 paired latest-event notes. KEY={group_tag}; AXIS={axis}. "
        "The latest marker for {entity_a} is {assigned_a}. "
        "The latest marker for {entity_b} is {assigned_b}.\n"
        "Question: What is the latest marker for {entity_a}?\n"
        "Return only the lowercase marker word.\nAnswer:"
    ),
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


def lexical_items(surface: str, seed: int) -> tuple[str, str, str, str, str, str, str, str]:
    entity_pool = OBJECTS if surface == "relation_qa" else NAMES
    entities = [
        entity_pool[(seed * 5 + offset * 11 + len(surface)) % len(entity_pool)]
        for offset in range(4)
    ]
    values = [
        LABELS[(seed * 7 + offset * 13 + len(surface)) % len(LABELS)]
        for offset in range(4)
    ]
    if len(set(entities)) != 4 or len(set(values)) != 4:
        raise ValueError("lexical collision")
    return (*entities, *values)


def query_fragment(surface: str, entity_a: str) -> str:
    if surface == "field_extraction":
        return f"What is the status of {entity_a}?"
    if surface == "relation_qa":
        return f"Where is the {entity_a} located?"
    return f"What is the latest marker for {entity_a}?"


def raw_case(
    surface: str,
    group_tag: str,
    axis: str,
    direct: bool,
    entity_a: str,
    entity_b: str,
    value_a: str,
    value_b: str,
) -> dict[str, Any]:
    assigned_a, assigned_b = (value_a, value_b) if direct else (value_b, value_a)
    prompt = TEMPLATES[surface].format(
        group_tag=group_tag,
        axis=axis,
        entity_a=entity_a,
        entity_b=entity_b,
        assigned_a=assigned_a,
        assigned_b=assigned_b,
    )
    query = query_fragment(surface, entity_a)
    relation_fragments = {
        "field_extraction": [f"{entity_a} has status {assigned_a}", f"{entity_b} has status {assigned_b}"],
        "relation_qa": [f"{entity_a} is located in {assigned_a}", f"{entity_b} is located in {assigned_b}"],
        "entity_recency": [f"latest marker for {entity_a} is {assigned_a}", f"latest marker for {entity_b} is {assigned_b}"],
    }[surface]
    source_fragment = {
        "field_extraction": (
            f"The record for {entity_a} has status {assigned_a}. "
            f"The record for {entity_b} has status {assigned_b}."
        ),
        "relation_qa": (
            f"The {entity_a} is located in {assigned_a}. "
            f"The {entity_b} is located in {assigned_b}."
        ),
        "entity_recency": (
            f"The latest marker for {entity_a} is {assigned_a}. "
            f"The latest marker for {entity_b} is {assigned_b}."
        ),
    }[surface]
    return {
        "raw_prompt": prompt,
        "target": assigned_a,
        "target_aliases": [assigned_a],
        "distractors": [assigned_b],
        "source_fragment": source_fragment,
        "query_fragment": query,
        "semantic_role_fragments_private": {
            "entities": [entity_a, entity_b],
            "attributes_items": [value_a, value_b],
            "relations": relation_fragments,
            "query_keywords": ["Question:"],
        },
        "semantic_slot_fragments_private": {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "value_a": value_a,
            "value_b": value_b,
            "assigned_to_entity_a": assigned_a,
            "assigned_to_entity_b": assigned_b,
        },
    }


def rendered_case(
    tokenizer: Any,
    model: str,
    surface: str,
    group_tag: str,
    condition: str,
    entity_a: str,
    entity_b: str,
    value_a: str,
    value_b: str,
) -> dict[str, Any]:
    axis = "X" if condition.startswith(("A_", "B_")) else "Y"
    direct = condition.startswith(("A_", "C_"))
    item = raw_case(surface, group_tag, axis, direct, entity_a, entity_b, value_a, value_b)
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
        "axis": axis,
        "direct_binding": direct,
    }


def pair_audit(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = prompt_token_ids(tokenizer, left)
    right_ids = prompt_token_ids(tokenizer, right)
    if len(left_ids) != len(right_ids):
        return {"valid": False, "reason": "prompt_token_count_mismatch"}
    if Counter(left_ids) != Counter(right_ids):
        return {"valid": False, "reason": "prompt_token_multiset_mismatch"}
    diff_positions = [
        index for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids))
        if left_id != right_id
    ]
    if not diff_positions:
        return {"valid": False, "reason": "no_binding_difference"}

    fixed_fragments = (
        left["semantic_slot_fragments_private"]["entity_a"],
        left["semantic_slot_fragments_private"]["entity_b"],
        left["query_fragment"],
    )
    fixed_positions: set[int] = set()
    for fragment in fixed_fragments:
        mapped = fragment_positions(tokenizer, left_ids, fragment)
        if not mapped:
            return {"valid": False, "reason": "missing_fixed_fragment", "fragment": fragment}
        fixed_positions.update(mapped)
    if any(left_ids[index] != right_ids[index] for index in fixed_positions):
        return {"valid": False, "reason": "entity_or_query_token_changed"}

    value_positions: dict[str, dict[str, list[int]]] = {"left": {}, "right": {}}
    for name, case, ids in (("left", left, left_ids), ("right", right, right_ids)):
        for key in ("value_a", "value_b"):
            literal = case["semantic_slot_fragments_private"][key]
            positions = fragment_positions(tokenizer, ids, literal)
            if not positions:
                return {"valid": False, "reason": "missing_value_fragment", "fragment": literal}
            value_positions[name][key] = positions
    value_union = set(
        value_positions["left"]["value_a"]
        + value_positions["left"]["value_b"]
        + value_positions["right"]["value_a"]
        + value_positions["right"]["value_b"]
    )
    if not set(diff_positions).issubset(value_union):
        return {"valid": False, "reason": "difference_outside_value_positions"}
    if value_positions["left"]["value_a"] == value_positions["right"]["value_a"]:
        return {"valid": False, "reason": "same_literal_value_did_not_move"}

    left_target = tokenizer(" " + left["target"], add_special_tokens=False)["input_ids"]
    right_target = tokenizer(" " + right["target"], add_special_tokens=False)["input_ids"]
    if not left_target or not right_target or left_target[0] == right_target[0]:
        return {"valid": False, "reason": "target_first_token_not_distinct"}
    return {
        "valid": True,
        "prompt_token_count": len(left_ids),
        "diff_positions": diff_positions,
        "fixed_entity_query_position_count": len(fixed_positions),
        "left_value_positions": value_positions["left"],
        "right_value_positions": value_positions["right"],
    }


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
    prompt_hashes: set[str] = set()
    for surface in TASK_SURFACES:
        accepted = 0
        seed = 0
        while accepted < GROUPS_PER_SURFACE and seed < 2000:
            try:
                e1, e2, e3, e4, v1, v2, v3, v4 = lexical_items(surface, seed)
            except ValueError:
                seed += 1
                continue
            group_tag = f"{surface[:3].upper()}{accepted:02d}"
            per_model: dict[str, dict[str, dict[str, Any]]] = {}
            audits: dict[str, dict[str, Any]] = {}
            valid = True
            for model in MODELS:
                tokenizer = tokenizers[model]
                cases = {
                    "A_direct_lex_x": rendered_case(tokenizer, model, surface, group_tag, "A_direct_lex_x", e1, e2, v1, v2),
                    "B_swapped_lex_x": rendered_case(tokenizer, model, surface, group_tag, "B_swapped_lex_x", e1, e2, v1, v2),
                    "C_direct_lex_y": rendered_case(tokenizer, model, surface, group_tag, "C_direct_lex_y", e3, e4, v3, v4),
                    "D_swapped_lex_y": rendered_case(tokenizer, model, surface, group_tag, "D_swapped_lex_y", e3, e4, v3, v4),
                }
                audit_x = pair_audit(tokenizer, cases["A_direct_lex_x"], cases["B_swapped_lex_x"])
                audit_y = pair_audit(tokenizer, cases["C_direct_lex_y"], cases["D_swapped_lex_y"])
                if not audit_x["valid"] or not audit_y["valid"]:
                    valid = False
                    break
                per_model[model] = cases
                audits[model] = {"lex_x": audit_x, "lex_y": audit_y}
            if not valid:
                seed += 1
                continue

            anonymous_group = "p395g_" + digest(f"phase395:{surface}:{seed}", 24)
            groups.append(
                {
                    "schema_version": "69.0.0",
                    "phase_id": "Phase395-Protocol",
                    "anonymous_parallel_group_id": anonymous_group,
                    "task_surface": surface,
                    "group_priority": accepted,
                    "same_token_multiset_all_models": True,
                    "entity_and_query_positions_fixed": True,
                    "literal_values_move_between_relations": True,
                }
            )
            for model in MODELS:
                tokenizer = tokenizers[model]
                for condition in CONDITIONS:
                    item = per_model[model][condition]
                    axis_key = "lex_x" if item["axis"] == "X" else "lex_y"
                    audit = audits[model][axis_key]
                    prompt_hash = digest(item["prompt"])
                    if prompt_hash in prompt_hashes:
                        raise RuntimeError("Duplicate Phase395 prompt")
                    prompt_hashes.add(prompt_hash)
                    case_id = "p395c_" + digest(f"{model}:{anonymous_group}:{condition}", 26)
                    own_positions = audit["left_value_positions"] if item["direct_binding"] else audit["right_value_positions"]
                    rows.append(
                        {
                            "schema_version": "69.0.0",
                            "phase_id": "Phase395-Protocol",
                            "created_at": created_at,
                            "private_execution_model": model,
                            "anonymous_model_id": "p395m_" + digest(model, 12),
                            "blind_case_id": case_id,
                            "anonymous_parallel_group_id": anonymous_group,
                            "anonymous_group_id": "p395s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition[0],
                            "group_priority": accepted,
                            "family_id": "content_knowledge",
                            "mechanism_id": surface,
                            "semantic_group_id": f"p395_private_{surface}_{seed:04d}",
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
                            "axis_private": item["axis"],
                            "direct_binding_private": item["direct_binding"],
                            "semantic_role_fragments_private": item["semantic_role_fragments_private"],
                            "semantic_slot_fragments_private": item["semantic_slot_fragments_private"],
                            "literal_value_positions_private": own_positions,
                            "binding_diff_positions_private": audit["diff_positions"],
                            "fixed_entity_query_position_count": audit["fixed_entity_query_position_count"],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                        }
                    )
            accepted += 1
            seed += 1

    expected = len(TASK_SURFACES) * GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase395 row count {len(rows)} != {expected}")
    write_jsonl(OUT / "protocol/private/phase395_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase395_blind_group_registry.jsonl", groups)
    protocol = {
        "schema_version": "69.0.0",
        "phase_id": "Phase395-Protocol",
        "created_at": created_at,
        "objective": "separate_literal_content_identity_from_natural_contextual_binding_state",
        "phase394_amendment": {
            "formal_pointer_interface_is_prerequisite_for_natural_binding": False,
            "reason": "formal pointer following is an artificial interface ability and failed crossmodel qualification",
            "phase394_data_reused": False,
        },
        "denominator": {
            "task_surfaces": list(TASK_SURFACES),
            "groups_per_surface": GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "split_group_counts_per_eligible_surface": SPLIT_COUNTS,
        },
        "natural_identity_contract": {
            "same_input_token_multiset_within_pair": True,
            "same_entities_and_query_within_pair": True,
            "same_literal_value_set_within_pair": True,
            "literal_values_swap_relation_positions": True,
            "same_literal_mapping_is_available_for_causal_control": True,
            "direct_position_mapping_is_content_swap_control": True,
            "single_sample_model_native_interface": True,
        },
        "planned_causal_contrasts": {
            "same_literal_state_mapping": "same word donor state to same word recipient position; tests contextual binding beyond token identity",
            "same_position_state_mapping": "donor position to same recipient position; swaps literal content and reproduces content transport",
            "entity_state_mapping": "fixed entity token states as structure control",
            "random_state_mapping": "same number of causal prefix positions",
            "wrong_depth_mapping": "frozen alternate relative depth",
        },
        "split_rule": "all 24 groups and all four conditions must pass on all three models; then hash-order 12/6/6 without replacement",
        "authorization": {
            "run_behavior_qualification": True,
            "run_internal_before_behavior_freeze": False,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "same_token_multiset_behavior_is_binding_mechanism": False,
            "same_literal_patch_effect_is_abstract_binding_rule": False,
            "three_surface_replication_is_nine_family_layout": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase395_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
