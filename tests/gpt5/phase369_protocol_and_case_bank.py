#!/usr/bin/env python3
"""Freeze Phase369 raw-vector topology protocol and 576 fresh parallel cases."""

from __future__ import annotations

import hashlib
import json
import shutil
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


PHASE = "Phase369"
SCHEMA_VERSION = "46.0.0"
ROUND = "raw_topology_preregister"
OUT = ROOT / "tests/gpt5/result/phase369_raw_topology_flow" / ROUND
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    ("content_knowledge", "relation_binding"),
    ("readout_competition", "target_vs_wrong"),
    ("state_drift", "entity_recency"),
    ("syntax_structure", "number_agreement"),
)
CONDITIONS = ("A_target_lex_x", "B_control_lex_x", "C_target_lex_y", "D_control_lex_y")
INTERFACE = "answer_aligned_chat"
GROUPS_PER_MECHANISM = 12
TEMPLATES = (
    "Evidence: {context}\nQuestion: {question}\nConstraint: {instruction}\nAnswer:",
    "Reference record: {context}\nRequest: {question}\nOutput rule: {instruction}\nFinal answer:",
    "Use only this context: {context}\n{question}\n{instruction}\nResponse:",
    "Facts: {context}\nTask: {question}\nRule: {instruction}\nAnswer only:",
)
MATERIALS = ("ceramic", "copper", "linen", "granite", "rubber", "silver", "plastic", "glass", "bamboo", "leather", "paper", "steel")
LABELS = ("amber", "violet", "teal", "scarlet", "indigo", "silver", "coral", "olive", "gold", "azure", "maroon", "ivory")
NAMES = ("Mira", "Jonah", "Leona", "Darius", "Nadia", "Caleb", "Selene", "Tobias", "Iris", "Felix", "Amina", "Hector")
OBJECTS = ("casket", "goblet", "satchel", "lantern", "tablet", "compass", "vessel", "tripod", "parcel", "flask", "cabinet", "medallion")
PRIOR_CASE_FILES = (
    ROOT / "tests/gpt5/result/phase353_family_contracts/family_specific_contract_compiler/phase353_registered_cases.jsonl",
    ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time/phase354_registered_cases.jsonl",
    ROOT / "tests/gpt5/result/phase361_contract_repair/seven_contract_repair/phase361_registered_cases.jsonl",
    ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time/private/phase362_execution_cases.jsonl",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def split_for(group_index: int) -> str:
    if group_index < 6:
        return "fresh_discovery"
    if group_index < 9:
        return "fresh_calibration"
    return "physical_holdout_sealed"


def render(context: str, question: str, instruction: str, group_index: int) -> str:
    return TEMPLATES[group_index % len(TEMPLATES)].format(
        context=context, question=question, instruction=instruction,
    )


def task_pair(mechanism: str, group_index: int, lexical_slot: str) -> tuple[dict[str, Any], dict[str, Any]]:
    offset = 0 if lexical_slot == "x" else 5
    index = (group_index + offset) % 12
    code = f"p369-{mechanism[:3]}-{group_index:02d}-{lexical_slot}"
    exact = "Return exactly the requested answer and nothing else."
    if mechanism == "relation_binding":
        obj = f"{OBJECTS[index]}-{code}"
        batch = f"registry-{code}"
        target = MATERIALS[index]
        wrong = MATERIALS[(index + 3) % 12]
        demanded = {
            "context": f"The {obj} is assigned to {batch}. Every object assigned to {batch} is made from {target}.",
            "question": f"What material is the {obj} made from?",
        }
        control = {
            "context": f"The {obj} is made from {target}. Its registry code is {batch}.",
            "question": f"What material is the {obj} made from?",
        }
    elif mechanism == "target_vs_wrong":
        target = LABELS[index]
        wrong = LABELS[(index + 4) % 12]
        demanded = {
            "context": f"For dossier {code}, the valid code word is {target}. The invalid competing code word is {wrong}.",
            "question": "Return the valid code word.",
        }
        control = {
            "context": f"For dossier {code}, the valid code word is {target}. No competing code word is supplied.",
            "question": "Return the valid code word.",
        }
    elif mechanism == "entity_recency":
        target = NAMES[index]
        wrong = NAMES[(index + 5) % 12]
        demanded = {
            "context": f"The designated custodian for file {code} is {target}. A later unrelated memo mentions {wrong}.",
            "question": f"Who is the designated custodian for file {code}?",
        }
        control = {
            "context": f"An unrelated memo mentions {wrong}. The designated custodian for file {code} is {target}.",
            "question": f"Who is the designated custodian for file {code}?",
        }
    elif mechanism == "number_agreement":
        plural = (group_index + offset) % 2 == 0
        root = f"{OBJECTS[index]}-{code}"
        noun = f"{root}s" if plural else root
        target, wrong = ("are", "is") if plural else ("is", "are")
        demanded = {
            "context": f"The grammatical subject is 'the {noun}'.",
            "question": f"Complete the sentence: The {noun} ___ ready.",
        }
        control = {
            "context": f"The number feature belongs to the subject 'the {noun}'.",
            "question": "Which English verb agrees with this subject: is or are?",
        }
    else:
        raise KeyError(mechanism)
    shared = {
        "target": target,
        "target_aliases": [target],
        "distractors": [wrong, "unknown"],
        "instruction": exact,
        "language": "en",
    }
    return ({**demanded, **shared}, {**control, **shared})


def prior_prompt_hashes() -> set[str]:
    hashes = set()
    for path in PRIOR_CASE_FILES:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            hashes.add(digest(str(row.get("raw_prompt", row.get("prompt", "")))))
            hashes.add(digest(str(row.get("prompt", ""))))
    return hashes


def protocol_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "objective": "test_whether_raw_vector_relations_and_topology_add_cross_model_future_information_beyond_phase367_descriptors",
        "evidence_denominators": {
            "registered_language_family_count": 9,
            "registered_representative_mechanism_count": 18,
            "phase369_admitted_mechanism_count": 4,
            "strict_closure_cell_count": 72,
            "strictly_closed_cell_count": 0,
            "single_global_progress_percentage_valid": False,
        },
        "dataset_contract": {
            "model_count": 3,
            "mechanism_count": 4,
            "fresh_group_count_per_model_mechanism": 12,
            "condition_count_per_group": 4,
            "case_count": 576,
            "fresh_discovery_groups_per_model_mechanism": 6,
            "fresh_calibration_groups_per_model_mechanism": 3,
            "physical_holdout_groups_per_model_mechanism": 3,
            "minimum_cross_model_qualified_discovery_groups_per_mechanism": 4,
            "minimum_cross_model_qualified_calibration_groups_per_mechanism": 2,
            "qualification_requires_all_four_conditions_and_all_three_models": True,
            "phase368_calibration_reused_for_mapping_or_thresholds": False,
            "old_physical_confirmation_opened": False,
        },
        "blind_event_contract": {
            "allowed": [
                "event_type", "source_role_alias", "receiver_role_alias", "relative_depth",
                "generation_time", "raw_vector_reference", "parent_event_reference",
                "branch_count", "merge_count", "conservation_ratio", "label_free_full_vocab_state_reference",
            ],
            "forbidden": [
                "family_id", "mechanism_id", "condition_semantics", "correct_answer",
                "target_token_id", "target_rank", "target_margin", "distractor_rank",
            ],
        },
        "raw_relation_signature": {
            "event_vectors_retained_by_hash_reference": True,
            "normalized_gram": "K_ij=dot(e_i,e_j)/(norm(e_i)*norm(e_j))",
            "norm_share": "r_i=norm(e_i)/sum_j(norm(e_j))",
            "unrestricted_learned_cross_model_rotation_allowed": False,
            "coordinate_invariant_relations_are_primary": True,
        },
        "topology_gate_vector": {
            "weighted_scalar_distance_used": False,
            "components": [
                "event_type", "partial_order", "source_receiver_role", "branch_merge",
                "relative_depth_drift", "generation_delay", "conservation", "future_prediction",
            ],
            "all_components_must_pass_their_frozen_gate": True,
            "calibration_may_retune_component_gate": False,
        },
        "head_neuron_contract": {
            "attention_heads_merged_before_discovery": False,
            "mlp_single_neuron_writes_offline_recoverable": True,
            "task_score_top_k_allowed": False,
            "fixed_hash_shard_counts": [8, 32, 128],
            "fixed_hash_seed_count": 3,
            "multiple_hash_seeds_count_as_independent_replication": False,
            "single_units_enter_only_after_conserved_shard_localization": True,
        },
        "cross_model_evidence_levels": {
            "level_1": "single_model_calibrated_topology",
            "level_2": "glm4_plus_qwen3_or_deepseek7b_calibrated_topology",
            "level_2_architecture_family_only": "qwen3_plus_deepseek7b_not_heterogeneous_replication",
            "level_3": "all_three_models_calibrated_topology",
            "physical_holdout_entry_minimum": "level_2_with_glm4",
            "unified_theory_entry_minimum": "level_3",
        },
        "controls": [
            "phase367_ten_descriptor_baseline", "matched_size_random_flow", "time_order_shuffle",
            "source_receiver_role_permutation", "equal_energy_wrong_flow", "public_architecture_backbone",
            "same_model_different_independent_group",
        ],
        "stage_order": [
            "fresh_case_contract", "discovery_and_calibration_behavior_qualification",
            "fresh_discovery_raw_collection", "candidate_and_gate_freeze",
            "fresh_calibration_once", "physical_holdout_only_if_authorized",
        ],
        "stop_rules": [
            "if_raw_relations_do_not_beat_phase367_descriptor_baseline_stop_before_calibration",
            "if_calibration_fails_do_not_reuse_calibration_to_redesign_mapping",
            "if_only_qwen3_and_deepseek7b_match_do_not_call_it_heterogeneous_cross_model",
            "if_no_glm4_cross_model_candidate_keep_physical_holdout_sealed",
            "never_open_old_or_new_physical_holdout_to_repair_discovery",
        ],
        "authorization": {
            "fresh_case_bank_creation": True,
            "discovery_and_calibration_behavior_qualification": True,
            "fresh_discovery_raw_collection_before_behavior_gate": False,
            "physical_holdout_execution": False,
            "causal_intervention": False,
        },
    }


def main() -> None:
    protocol = protocol_payload()
    prior_hashes = prior_prompt_hashes()
    execution_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    prompt_hashes = set()
    raw_hashes = set()
    tokenizers = {}
    try:
        for model in MODELS:
            spec = get_model_spec(model)
            tokenizer = AutoTokenizer.from_pretrained(
                str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
                local_files_only=True, use_fast=False,
            )
            tokenizers[model] = tokenizer
            for family, mechanism in MECHANISMS:
                for group_index in range(GROUPS_PER_MECHANISM):
                    split = split_for(group_index)
                    semantic_group = f"phase369_{family}_{mechanism}_{group_index:02d}"
                    parallel_group = "parallel_" + digest(semantic_group)[:20]
                    model_group = "group_" + digest(f"{model}:{semantic_group}")[:20]
                    condition_tasks = {}
                    for lexical_slot, letters in (("x", ("A", "B")), ("y", ("C", "D"))):
                        demanded, control = task_pair(mechanism, group_index, lexical_slot)
                        condition_tasks[letters[0]] = demanded
                        condition_tasks[letters[1]] = control
                    for condition in CONDITIONS:
                        letter = condition[0]
                        task = condition_tasks[letter]
                        raw_prompt = render(task["context"], task["question"], task["instruction"], group_index)
                        prompt, add_special, answer_phase = interface_prompt(
                            tokenizer, model, raw_prompt, INTERFACE,
                        )
                        raw_hash = digest(raw_prompt)
                        prompt_hash = digest(prompt)
                        if raw_hash in prior_hashes or prompt_hash in prior_hashes:
                            raise RuntimeError(f"Prior prompt overlap for {model}/{semantic_group}/{condition}")
                        if prompt_hash in prompt_hashes:
                            raise RuntimeError(f"Duplicate rendered prompt: {model}/{semantic_group}/{condition}")
                        prompt_hashes.add(prompt_hash)
                        raw_hashes.add(raw_hash)
                        blind_case_id = "p369_" + digest(f"{model}:{semantic_group}:{condition}")[:24]
                        condition_slot = "slot_" + digest(f"{model_group}:{condition}")[:12]
                        prompt_token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                        common = {
                            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                            "blind_case_id": blind_case_id,
                            "anonymous_model_id": "am369_" + digest(model)[:12],
                            "anonymous_parallel_group_id": parallel_group,
                            "anonymous_group_id": model_group,
                            "anonymous_condition_slot": condition_slot,
                            "phase369_split": split,
                            "prompt": prompt, "raw_prompt": raw_prompt,
                            "source_fragment": task["context"], "query_fragment": task["question"],
                            "tokenization_add_special_tokens": add_special,
                            "prompt_token_count": prompt_token_count,
                            "interface": INTERFACE, "answer_phase": answer_phase,
                        }
                        execution_rows.append({
                            **common,
                            "private_execution_model": model,
                            "family_id": family, "mechanism_id": mechanism,
                            "semantic_group_id": semantic_group,
                            "contrast_condition": condition,
                            "operation_demanded": letter in {"A", "C"},
                            "target": task["target"], "target_aliases": task["target_aliases"],
                            "distractors": task["distractors"], "language": task["language"],
                            "instruction": task["instruction"], "question": task["question"],
                            "semantic_labels_available_to_collector": False,
                            "target_specific_competition_available_to_collector": False,
                        })
                        blind_rows.append({
                            **common,
                            "semantic_label_used_for_selection": False,
                            "target_or_distractor_exported": False,
                        })
                        label_rows.append({
                            "blind_case_id": blind_case_id, "model": model,
                            "family_id": family, "mechanism_id": mechanism,
                            "semantic_group_id": semantic_group,
                            "contrast_condition": condition, "phase369_split": split,
                            "target": task["target"], "target_aliases": task["target_aliases"],
                            "distractors": task["distractors"],
                        })
        if len(execution_rows) != 576 or len(blind_rows) != 576:
            raise RuntimeError(f"Invalid Phase369 case count: {len(execution_rows)}")
        split_counts = Counter(row["phase369_split"] for row in execution_rows)
        group_condition_counts = Counter(
            (row["anonymous_model_id"], row["anonymous_group_id"])
            for row in execution_rows
        )
        if set(group_condition_counts.values()) != {4}:
            raise RuntimeError("Every model group must contain four conditions")
        physical_ids = {
            row["blind_case_id"] for row in execution_rows
            if row["phase369_split"] == "physical_holdout_sealed"
        }
        nonphysical_ids = {row["blind_case_id"] for row in execution_rows} - physical_ids
        summary = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "objective": "freeze_fresh_parallel_denominator_before_any_phase369_model_result",
            "denominator": {
                "model_count": 3, "mechanism_count": 4,
                "parallel_group_count": 4 * 12,
                "model_group_count": 3 * 4 * 12,
                "condition_count_per_group": 4,
                "case_count": len(execution_rows),
                "fresh_discovery_case_count": split_counts["fresh_discovery"],
                "fresh_calibration_case_count": split_counts["fresh_calibration"],
                "physical_holdout_case_count": split_counts["physical_holdout_sealed"],
            },
            "quality": {
                "unique_rendered_prompt_count": len(prompt_hashes),
                "unique_raw_prompt_count": len(raw_hashes),
                "prior_prompt_overlap_count": 0,
                "every_group_has_four_conditions": True,
                "parallel_group_ids_shared_across_models": True,
                "semantic_labels_exported_to_blind_registry": False,
                "physical_holdout_case_overlap_with_nonphysical": len(physical_ids & nonphysical_ids),
                "phase368_calibration_reused": False,
            },
            "storage": {
                "free_disk_bytes_at_freeze": shutil.disk_usage(ROOT).free,
                "phase365_observed_bytes_per_case": 18848575296 / 288,
                "estimated_discovery_raw_bytes": round(18848575296 / 288 * 288),
                "estimated_calibration_raw_bytes": round(18848575296 / 288 * 144),
                "estimated_physical_holdout_raw_bytes": round(18848575296 / 288 * 144),
            },
            "authorization": {
                "run_discovery_and_calibration_behavior_qualification": True,
                "run_fresh_discovery_raw_collection_before_behavior_gate": False,
                "run_physical_holdout": False,
            },
            "next_decision": "run_behavior_qualification_sequentially_qwen3_glm4_deepseek7b_without_physical_holdout",
        }
        write_json(OUT / "phase369_protocol.json", protocol)
        write_json(OUT / "phase369_case_bank_summary.json", summary)
        write_jsonl(OUT / "phase369_blind_case_registry.jsonl", blind_rows)
        write_jsonl(OUT / "private" / "phase369_execution_cases.jsonl", execution_rows)
        write_jsonl(OUT / "private" / "phase369_label_key.jsonl", label_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        tokenizers.clear()


if __name__ == "__main__":
    main()
