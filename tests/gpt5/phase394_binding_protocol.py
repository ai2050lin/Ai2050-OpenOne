#!/usr/bin/env python3
"""Freeze token-identity-controlled formal binding cases for Phase394."""

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
from phase386_multitime_protocol import LABELS, NAMES, OBJECTS  # noqa: E402
from phase390_role_mapping import fragment_positions, prompt_token_ids  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase394_binding_separation"
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

TASK_TEXT = {
    "field_extraction": {
        "task": "status field extraction",
        "entity": "record",
        "value": "status value",
    },
    "relation_qa": {
        "task": "object-location relation lookup",
        "entity": "object",
        "value": "location value",
    },
    "entity_recency": {
        "task": "entity latest-event lookup",
        "entity": "actor",
        "value": "latest event value",
    },
}

TEMPLATE = """Phase394 sealed formal pointer ledger. GROUP={group_tag}; AXIS={axis}
TASK={task}
CATALOG
ENTITY_A={entity_a}
ENTITY_B={entity_b}
VALUE_A={value_a}
VALUE_B={value_b}
BINDINGS
EDGE_1=ENTITY_A->{pointer_a}
EDGE_2=ENTITY_B->{pointer_b}
QUERY
resolve(ENTITY_A) for the {value_kind} of this {entity_kind}
CONTRACT
Follow ENTITY_A's EDGE, dereference that VALUE slot in CATALOG, and return only its lowercase catalog word.
Answer:"""


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
    entity_words = [
        entity_pool[(seed * 7 + offset * 9 + len(surface)) % len(entity_pool)]
        for offset in range(4)
    ]
    value_words = [
        LABELS[(seed * 11 + offset * 13 + 3 * len(surface)) % len(LABELS)]
        for offset in range(4)
    ]
    if len(set(entity_words)) != 4 or len(set(value_words)) != 4:
        raise ValueError("lexical collision")
    return (*entity_words, *value_words)


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
    spec = TASK_TEXT[surface]
    pointer_a = "[VALUE_A]" if direct else "[VALUE_B]"
    pointer_b = "[VALUE_B]" if direct else "[VALUE_A]"
    target = value_a if direct else value_b
    rejected = value_b if direct else value_a
    raw_prompt = TEMPLATE.format(
        group_tag=group_tag,
        axis=axis,
        task=spec["task"],
        entity_a=entity_a,
        entity_b=entity_b,
        value_a=value_a,
        value_b=value_b,
        pointer_a=pointer_a,
        pointer_b=pointer_b,
        value_kind=spec["value"],
        entity_kind=spec["entity"],
    )
    return {
        "raw_prompt": raw_prompt,
        "target": target,
        "target_aliases": [target],
        "distractors": [rejected],
        "source_fragment": (
            f"BINDINGS\nEDGE_1=ENTITY_A->{pointer_a}\n"
            f"EDGE_2=ENTITY_B->{pointer_b}"
        ),
        "query_fragment": (
            f"resolve(ENTITY_A) for the {spec['value']} of this {spec['entity']}"
        ),
        "semantic_role_fragments_private": {
            "entities": [entity_a, entity_b],
            "attributes_items": [value_a, value_b],
            "relations": ["[VALUE_A]", "[VALUE_B]"],
            "query_keywords": ["resolve(ENTITY_A)"],
        },
        "semantic_slot_fragments_private": {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "value_a": value_a,
            "value_b": value_b,
            "edge_1_pointer": pointer_a,
            "edge_2_pointer": pointer_b,
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
    item = raw_case(
        surface,
        group_tag,
        axis,
        direct,
        entity_a,
        entity_b,
        value_a,
        value_b,
    )
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
    diff_positions = [
        index for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids))
        if left_id != right_id
    ]
    if not diff_positions:
        return {"valid": False, "reason": "no_binding_pointer_difference"}

    content_fragments = (
        left["semantic_slot_fragments_private"]["entity_a"],
        left["semantic_slot_fragments_private"]["entity_b"],
        left["semantic_slot_fragments_private"]["value_a"],
        left["semantic_slot_fragments_private"]["value_b"],
        left["query_fragment"],
    )
    content_positions: set[int] = set()
    missing: list[str] = []
    for fragment in content_fragments:
        positions = fragment_positions(tokenizer, left_ids, fragment)
        if not positions:
            missing.append(fragment)
        content_positions.update(positions)
    if missing:
        return {"valid": False, "reason": "missing_content_fragments", "missing": missing}
    if any(left_ids[index] != right_ids[index] for index in content_positions):
        return {"valid": False, "reason": "content_token_identity_changed"}

    relation_positions = set()
    for fragment in ("[VALUE_A]", "[VALUE_B]"):
        relation_positions.update(fragment_positions(tokenizer, left_ids, fragment))
        relation_positions.update(fragment_positions(tokenizer, right_ids, fragment))
    if not set(diff_positions).issubset(relation_positions):
        return {
            "valid": False,
            "reason": "difference_outside_binding_pointer",
            "diff_positions": diff_positions,
            "relation_positions": sorted(relation_positions),
        }

    left_target = tokenizer(" " + left["target"], add_special_tokens=False)["input_ids"]
    right_target = tokenizer(" " + right["target"], add_special_tokens=False)["input_ids"]
    if not left_target or not right_target or left_target[0] == right_target[0]:
        return {"valid": False, "reason": "target_first_token_not_distinct"}
    return {
        "valid": True,
        "prompt_token_count": len(left_ids),
        "binding_diff_positions": diff_positions,
        "binding_relation_positions": sorted(relation_positions),
        "content_identity_position_count": len(content_positions),
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
    public_groups: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    accepted_by_surface: dict[str, int] = {surface: 0 for surface in TASK_SURFACES}
    for surface in TASK_SURFACES:
        seed = 0
        while accepted_by_surface[surface] < GROUPS_PER_SURFACE and seed < 1000:
            group_index = accepted_by_surface[surface]
            group_tag = f"{surface[:3].upper()}{group_index:02d}"
            try:
                e1, e2, e3, e4, v1, v2, v3, v4 = lexical_items(surface, seed)
            except ValueError:
                seed += 1
                continue
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

            source_group = f"p394_private_{surface}_{seed:04d}"
            anonymous_group = "p394g_" + digest(f"phase394:{surface}:{seed}", 24)
            public_groups.append(
                {
                    "schema_version": "68.0.0",
                    "phase_id": "Phase394-Protocol",
                    "anonymous_parallel_group_id": anonymous_group,
                    "task_surface": surface,
                    "group_priority": group_index,
                    "four_condition_three_model_identity_contract": True,
                    "formal_pointer_contract_only": True,
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
                        raise RuntimeError("Duplicate Phase394 prompt")
                    prompt_hashes.add(prompt_hash)
                    case_id = "p394c_" + digest(f"{model}:{anonymous_group}:{condition}", 26)
                    rows.append(
                        {
                            "schema_version": "68.0.0",
                            "phase_id": "Phase394-Protocol",
                            "created_at": created_at,
                            "private_execution_model": model,
                            "anonymous_model_id": "p394m_" + digest(model, 12),
                            "blind_case_id": case_id,
                            "anonymous_parallel_group_id": anonymous_group,
                            "anonymous_group_id": "p394s_" + digest(surface, 12),
                            "anonymous_condition_slot": condition[0],
                            "group_priority": group_index,
                            "family_id": "content_knowledge",
                            "mechanism_id": surface,
                            "semantic_group_id": source_group,
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
                            "binding_diff_positions_private": audit["binding_diff_positions"],
                            "binding_relation_positions_private": audit["binding_relation_positions"],
                            "content_identity_position_count": audit["content_identity_position_count"],
                            "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                        }
                    )
            accepted_by_surface[surface] += 1
            seed += 1

    expected = len(TASK_SURFACES) * GROUPS_PER_SURFACE * len(CONDITIONS) * len(MODELS)
    if len(rows) != expected:
        raise RuntimeError(f"Invalid Phase394 row count {len(rows)} != {expected}")
    write_jsonl(OUT / "protocol/private/phase394_candidate_cases.jsonl", rows)
    write_jsonl(OUT / "protocol/phase394_blind_group_registry.jsonl", public_groups)
    protocol = {
        "schema_version": "68.0.0",
        "phase_id": "Phase394-Protocol",
        "created_at": created_at,
        "objective": "separate_content_identity_from_formal_binding_pointer_and_query_route",
        "audit_of_proposal": {
            "all_input_tokens_can_be_identical_while_binding_changes": False,
            "content_tokens_and_positions_can_be_identical": True,
            "binding_change_requires_separate_pointer_tokens": True,
            "formal_pointer_success_is_natural_language_binding": False,
            "three_formal_surfaces_are_three_independent_language_mechanisms": False,
        },
        "denominator": {
            "task_surfaces": list(TASK_SURFACES),
            "groups_per_surface": GROUPS_PER_SURFACE,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "split_group_counts_per_eligible_surface": SPLIT_COUNTS,
        },
        "identity_contract": {
            "entity_content_tokens_fixed_within_pair": True,
            "attribute_content_tokens_fixed_within_pair": True,
            "query_tokens_fixed_within_pair": True,
            "content_token_positions_fixed_within_pair": True,
            "only_binding_pointer_token_ids_change_within_pair": True,
            "prompt_token_count_fixed_within_pair": True,
            "target_first_tokens_distinct": True,
            "single_sample_model_native_interface": True,
        },
        "split_rule": (
            "a task surface enters internal collection only if all 24 groups pass all "
            "four conditions on all three models with target and post-target events; "
            "then hash-order 12/6/6 without replacement"
        ),
        "internal_objects": [
            "entity_content",
            "attribute_content",
            "binding_pointer",
            "query_integrated_state",
            "terminal_carrier",
            "readout_competition",
        ],
        "causal_controls": [
            "identity_binding_pointer",
            "donor_binding_pointer",
            "donor_attribute_content",
            "donor_structure_positions",
            "same_count_random_positions",
            "wrong_entity_pointer",
            "wrong_depth_binding_pointer",
            "wrong_semantic_time",
        ],
        "authorization": {
            "run_behavior_qualification": True,
            "run_internal_before_behavior_freeze": False,
            "run_natural_language_transfer_before_formal_physical_gate": False,
            "run_single_neuron_scan": False,
        },
        "claim_boundary": {
            "formal_pointer_binding_is_natural_language_binding": False,
            "binding_transport_is_natural_necessity": False,
            "binding_transport_is_complete_language_path": False,
            "language_encoding_closed": False,
        },
    }
    write_json(OUT / "phase394_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
