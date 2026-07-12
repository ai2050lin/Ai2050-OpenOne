#!/usr/bin/env python3
"""Freeze the Phase390 multi-source joint-graph denominator and analysis contract."""

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
from phase386_multitime_protocol import (  # noqa: E402
    LABELS,
    NAMES,
    NOUNS,
    OBJECTS,
    PREPOSITIONS,
    VALUES,
)


OUT = ROOT / "tests/gpt5/result/phase390_joint_formation_graph"
P386 = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("content_knowledge", "relation_binding"),
    ("state_drift", "entity_recency"),
    ("language_action", "field_extraction"),
    ("readout_competition", "target_vs_wrong"),
    ("syntax_structure", "number_agreement"),
    ("reasoning_constraint", "missing_condition_control"),
)
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
CANDIDATE_GROUPS_PER_MECHANISM = 24
FROZEN_SPLIT_GROUPS = {"discovery": 12, "calibration": 6, "physical_holdout": 6}
TEMPLATES = (
    "Phase390 sealed joint-state card. {context}\nRequested value: {question}\nResponse contract: {instruction}\nAnswer:",
    "Independent Phase390 record. {context}\nTask field: {question}\n{instruction}\nValue:",
    "Phase390 isolated evidence packet. {context}\nQuery: {question}\nOutput rule: {instruction}\nResult:",
    "Fresh Phase390 computation note. {context}\nConstraint: {instruction}\nQuestion: {question}\nReply:",
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def item_at(values: tuple[Any, ...], index: int) -> Any:
    return values[index % len(values)]


def task(
    mechanism: str, group_index: int, lexical_slot: str, operation: bool
) -> dict[str, Any]:
    offset = 7 if lexical_slot == "x" else 29
    index = group_index * 3 + offset
    code = f"j390-{mechanism[:4]}-{group_index:02d}-{lexical_slot}"
    if mechanism == "relation_binding":
        entity = item_at(OBJECTS, index)
        target = item_at(VALUES, index + 1)
        wrong = item_at(VALUES, index + 10)
        site = f"bay-{code}"
        relation = "registered-object material rule"
        context = (
            f"Object {entity}-{code} is logged at {site}. Under the {relation}, every object at {site} has material {target}. A rejected material is {wrong}."
            if operation
            else f"Object {entity}-{code} has material {target}. Its archive location is {site}; rejected material {wrong} is irrelevant."
        )
        question = f"What material belongs to object {entity}-{code}?"
        instruction = "Return only the material without explanation."
        roles = {
            "entities": [f"{entity}-{code}", site],
            "attributes_items": [target, wrong],
            "relations": [relation if operation else "has material"],
            "query_keywords": ["What material"],
        }
    elif mechanism == "entity_recency":
        record = f"case-{code}"
        target = item_at(NAMES, index)
        wrong = item_at(NAMES, index + 11)
        relation = "verified custodian"
        context = (
            f"For {record}, an old unverified note names {wrong}. The later {relation} entry names {target}."
            if operation
            else f"For {record}, the {relation} entry names {target}. A separate note about {wrong} is unrelated."
        )
        question = f"Who is the {relation} for {record}?"
        instruction = "Return only the person's name without explanation."
        roles = {
            "entities": [record, target, wrong],
            "attributes_items": [target, wrong],
            "relations": [relation],
            "query_keywords": [f"Who is the {relation}"],
        }
    elif mechanism == "field_extraction":
        record = f"entry-{code}"
        target = item_at(LABELS, index)
        wrong = item_at(LABELS, index + 13)
        batch = item_at(LABELS, index + 5)
        owner = item_at(NAMES, index + 2)
        relation = "status field"
        context = (
            f"Structured {record}: owner={owner}; batch={batch}; status={target}; rejected_status={wrong}."
            if operation
            else f"Structured {record}: status={target}. The unrelated label {wrong} is not a field value."
        )
        question = f"Extract the {relation} from {record}."
        instruction = "Return only the status value without explanation."
        roles = {
            "entities": [record, owner],
            "attributes_items": [target, wrong, batch],
            "relations": [relation],
            "query_keywords": ["Extract the status field"],
        }
    elif mechanism == "target_vs_wrong":
        register = f"register-{code}"
        target = item_at(LABELS, index)
        wrong = item_at(LABELS, index + 17)
        relation = "approval decision"
        context = (
            f"The {relation} for {register} accepts {target} and rejects {wrong}."
            if operation
            else f"The accepted label for {register} is {target}; the label {wrong} belongs to another register."
        )
        question = f"Which label is accepted for {register}?"
        instruction = "Return only the accepted label without explanation."
        roles = {
            "entities": [register],
            "attributes_items": [target, wrong],
            "relations": [relation if operation else "accepted label"],
            "query_keywords": ["Which label is accepted"],
        }
    elif mechanism == "number_agreement":
        singular, plural = item_at(NOUNS, index)
        plural_head = (group_index + (lexical_slot == "y")) % 2 == 0
        head = plural if plural_head else singular
        target, wrong = ("are", "is") if plural_head else ("is", "are")
        attractor_singular, attractor_plural = item_at(NOUNS, index + 9)
        attractor = attractor_singular if plural_head else attractor_plural
        relation = item_at(PREPOSITIONS, group_index)
        subject = f"the {head} {relation} the {attractor}" if operation else f"the {head}"
        context = (
            f"Agreement record {code}: the head noun is {head}; the noun after {relation} cannot control agreement."
            if operation
            else f"Agreement record {code}: apply ordinary agreement to head noun {head}."
        )
        question = f"Fill only the blank: {subject.capitalize()} ___ ready."
        instruction = "Return exactly one word: is or are."
        roles = {
            "entities": [head, attractor] if operation else [head],
            "attributes_items": [target, wrong],
            "relations": [relation if operation else "ordinary agreement"],
            "query_keywords": ["Fill only the blank"],
        }
    elif mechanism == "missing_condition_control":
        subject = f"item-{code}"
        prop_a = item_at(LABELS, index)
        prop_b = item_at(LABELS, index + 8)
        conclusion = item_at(LABELS, index + 15)
        target, wrong = "unknown", "yes"
        relation = "two-condition rule"
        context = (
            f"The {relation} says: if something is {prop_a} and {prop_b}, it is {conclusion}. {subject} is {prop_a}; no record states whether it is {prop_b}."
            if operation
            else f"The sealed record says the status of {subject} as {conclusion} cannot be determined because a required condition is absent."
        )
        question = f"Is {subject} definitely {conclusion}?"
        instruction = "Answer yes, no, or unknown only."
        roles = {
            "entities": [subject],
            "attributes_items": [prop_a, prop_b, conclusion],
            "relations": [relation if operation else "required condition"],
            "query_keywords": ["definitely"],
        }
    else:
        raise KeyError(mechanism)
    visible_source = context + "\n" + question
    visible_roles = {
        role: [fragment for fragment in role_fragments if fragment in visible_source]
        for role, role_fragments in roles.items()
    }
    return {
        "context": context,
        "question": question,
        "instruction": instruction,
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong],
        "semantic_role_fragments": visible_roles,
    }


def main() -> None:
    created_at = now()
    execution_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    prior_hashes = {
        digest(row["prompt"])
        for row in read_jsonl(
            P386 / "protocol/private/phase386_candidate_execution_cases.jsonl"
        )
        if row.get("prompt")
    }
    tokenizers: dict[str, Any] = {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizer = AutoTokenizer.from_pretrained(
                str(spec.local_dir),
                trust_remote_code=spec.trust_remote_code,
                local_files_only=True,
                use_fast=False,
            )
            tokenizers[model] = tokenizer
            for family, mechanism in MECHANISMS:
                for group_index in range(CANDIDATE_GROUPS_PER_MECHANISM):
                    semantic_group = f"phase390_{family}_{mechanism}_{group_index:02d}"
                    parallel_group = "parallel390_" + digest(semantic_group, 20)
                    model_group = "group390_" + digest(f"{model}:{semantic_group}", 20)
                    items = {
                        "A": task(mechanism, group_index, "x", True),
                        "B": task(mechanism, group_index, "x", False),
                        "C": task(mechanism, group_index, "y", True),
                        "D": task(mechanism, group_index, "y", False),
                    }
                    for condition in CONDITIONS:
                        item = items[condition[0]]
                        raw_prompt = TEMPLATES[group_index % len(TEMPLATES)].format(**item)
                        prompt, add_special, answer_phase = interface_prompt(
                            tokenizer, model, raw_prompt, "answer_aligned_chat"
                        )
                        prompt_hash = digest(prompt)
                        if prompt_hash in prompt_hashes or prompt_hash in prior_hashes:
                            raise RuntimeError("Duplicate or reused Phase390 rendered prompt")
                        prompt_hashes.add(prompt_hash)
                        case_id = "p390c_" + digest(
                            f"{model}:{semantic_group}:{condition}", 26
                        )
                        common = {
                            "schema_version": "64.0.0",
                            "phase_id": "Phase390-Protocol",
                            "created_at": created_at,
                            "blind_case_id": case_id,
                            "anonymous_model_id": "am390_" + digest(model, 12),
                            "anonymous_parallel_group_id": parallel_group,
                            "anonymous_group_id": model_group,
                            "anonymous_condition_slot": "slot390_"
                            + digest(f"{model_group}:{condition}", 12),
                            "prompt": prompt,
                            "raw_prompt": raw_prompt,
                            "source_fragment": item["context"],
                            "query_fragment": item["question"],
                            "prompt_token_count": len(
                                tokenizer(prompt, add_special_tokens=False)["input_ids"]
                            ),
                            "tokenization_add_special_tokens": add_special,
                            "interface": "answer_aligned_chat",
                            "answer_phase": answer_phase,
                        }
                        execution_rows.append(
                            {
                                **common,
                                "private_execution_model": model,
                                "family_id": family,
                                "mechanism_id": mechanism,
                                "semantic_group_id": semantic_group,
                                "contrast_condition": condition,
                                "operation_demanded": condition[0] in {"A", "C"},
                                "target": item["target"],
                                "target_aliases": item["target_aliases"],
                                "distractors": item["distractors"],
                                "semantic_role_fragments_private": item[
                                    "semantic_role_fragments"
                                ],
                                "language": "en",
                            }
                        )
                        blind_rows.append(
                            {
                                **common,
                                "semantic_label_used_for_collection": False,
                                "target_or_distractor_exported": False,
                            }
                        )
    finally:
        tokenizers.clear()

    expected = (
        len(MODELS)
        * len(MECHANISMS)
        * CANDIDATE_GROUPS_PER_MECHANISM
        * len(CONDITIONS)
    )
    if len(execution_rows) != expected or len(prompt_hashes) != expected:
        raise RuntimeError(
            f"Invalid Phase390 bank: rows={len(execution_rows)} hashes={len(prompt_hashes)}"
        )
    private = OUT / "protocol/private"
    write_jsonl(private / "phase390_candidate_execution_cases.jsonl", execution_rows)
    write_jsonl(OUT / "protocol/phase390_blind_case_registry.jsonl", blind_rows)
    protocol = {
        "schema_version": "64.0.0",
        "phase_id": "Phase390-Protocol",
        "created_at": created_at,
        "objective": "map_graph_legal_multi_source_multi_head_cross_layer_joint_formation",
        "denominator": {
            "models": list(MODELS),
            "families": [family for family, _ in MECHANISMS],
            "mechanisms": [mechanism for _, mechanism in MECHANISMS],
            "candidate_groups_per_mechanism": CANDIDATE_GROUPS_PER_MECHANISM,
            "conditions_per_group": len(CONDITIONS),
            "candidate_parallel_group_count": len(MECHANISMS)
            * CANDIDATE_GROUPS_PER_MECHANISM,
            "candidate_case_count": expected,
            "frozen_split_groups_per_qualified_mechanism": FROZEN_SPLIT_GROUPS,
            "failed_group_replacement_allowed": False,
        },
        "runtime_contract": {
            "generation_path": "model_native_answer_aligned_single_sample",
            "internal_path": "initial_prompt_then_actual_incremental_kv_cache",
            "execution_batch_size": 1,
            "model_order": list(MODELS),
            "dtype_by_model": {
                "qwen3": "float16",
                "glm4": "float16",
                "deepseek7b": "bfloat16",
            },
        },
        "directed_parent_graph": [
            "source_layer_input_to_source_key_value",
            "query_layer_input_and_source_keys_to_attention_probabilities",
            "source_values_and_probabilities_to_head_write",
            "all_head_writes_to_attention_output",
            "layer_input_and_attention_output_to_post_attention_state",
            "post_attention_state_to_mlp_write",
            "post_attention_state_and_mlp_write_to_layer_output",
            "layer_output_to_next_layer_input",
        ],
        "forbidden_edges": [
            "source_same_layer_attention_output_to_query_same_layer_head_state",
            "independently_patched_key_value_states_without_parent_recomputation",
            "independent_multi_layer_patches_that_overwrite_natural_propagation",
        ],
        "source_roles": [
            "entities",
            "attributes_items",
            "relations",
            "query_keywords",
            "query_window",
            "other_causal_prefix",
        ],
        "role_contract": {
            "all_prompt_positions_partitioned_once": True,
            "strongest_source_selection_used": False,
            "all_heads_retained": True,
            "top_k_used": False,
        },
        "natural_window_contract": {
            "window_lengths": [1, 2, 4],
            "cross_layer_write": "sum_of_exact_residual_writes_in_one_residual_coordinate_system",
            "multi_layer_intervention": "patch_once_at_earliest_parent_boundary_then_recompute_naturally",
        },
        "frozen_analysis": {
            "learned_black_box_predictor_used": False,
            "sae_or_learned_basis_used": False,
            "joint_attention_write": "sum_over_all_registered_source_roles_and_all_heads",
            "single_source_baseline": "best_pre_registered_role_without_posthoc_role_merging",
            "single_head_baseline": "best_head_within_model_without_crossmodel_head_id_matching",
            "cross_layer_baseline": "best_single_layer_against_pre_registered_two_and_four_layer_windows",
            "terminal_target": "operation_minus_control_target_encoded_layer_output",
            "lexical_replication": ["A_minus_B", "C_minus_D"],
            "controls": [
                "wrong_parallel_group",
                "wrong_semantic_time",
                "wrong_depth",
                "source_role_permutation",
                "head_permutation",
                "same_energy_random_natural_parent_set",
            ],
        },
        "quality_gates": {
            "component_parent_child_max_relative_error": 0.01,
            "source_role_partition_max_relative_error": 0.01,
            "minimum_discovery_support_groups": 8,
            "minimum_calibration_support_groups": 4,
            "minimum_physical_support_groups": 4,
            "minimum_joint_advantage_over_best_single": 0.05,
            "minimum_correct_minus_wrong_control_advantage": 0.05,
            "all_three_models_required_for_crossmodel_claim": True,
        },
        "causal_replay_contract": {
            "authorized_before_physical_gate": False,
            "patch_boundary": "earliest_source_layer_input_parent_boundary",
            "patch_all_matching_source_positions_together": True,
            "natural_recomputation_after_patch": True,
            "controls": [
                "identity",
                "best_single_source_role",
                "wrong_source_role_set",
                "wrong_depth",
                "wrong_semantic_time",
                "same_count_random_parent_positions",
            ],
        },
        "claim_boundary": {
            "predictive_joint_graph_is_causal_path": False,
            "attention_additivity_is_multi_source_synergy": False,
            "fixed_head_identity_required_across_models": False,
            "single_neuron_scan_authorized": False,
            "language_encoding_closed": False,
        },
        "prior_prompt_overlap_count": 0,
    }
    write_json(OUT / "phase390_protocol.json", protocol)
    print(json.dumps(protocol["denominator"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
