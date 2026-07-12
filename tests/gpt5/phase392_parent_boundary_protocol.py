#!/usr/bin/env python3
"""Freeze fresh token-aligned paired field cases for Phase392 parent-boundary replay."""

from __future__ import annotations

import hashlib
import itertools
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
from phase386_multitime_protocol import LABELS, NAMES  # noqa: E402
from phase390_role_mapping import (  # noqa: E402
    REGISTERED_ROLES,
    fragment_positions,
    prompt_token_ids,
    semantic_role_indices,
)


OUT = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = ("mapping_x", "mapping_y")
GROUP_COUNT = 40
TEMPLATE = (
    "Phase392 sealed paired field record. {context}\n"
    "Question: {question}\nConstraint: {instruction}\nAnswer:"
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


def raw_case(seed: int, target: str, rejected: str) -> dict[str, Any]:
    record = f"paired-field-{seed:03d}"
    owner = NAMES[(seed * 3 + 5) % len(NAMES)]
    batch = LABELS[(seed * 5 + 9) % len(LABELS)]
    if batch in {target, rejected}:
        batch = LABELS[(seed * 5 + 12) % len(LABELS)]
    context = (
        f"record={record}; owner={owner}; batch={batch}; "
        f"status={target}; rejected_status={rejected}."
    )
    question = f"Extract the status field from record {record}."
    return {
        "context": context,
        "question": question,
        "instruction": "Return only the status value without explanation.",
        "target": target,
        "target_aliases": [target],
        "distractors": [rejected],
        "semantic_role_fragments_private": {
            "entities": [record, owner],
            "attributes_items": [target, rejected, batch],
            "relations": ["status field"],
            "query_keywords": ["Extract the status field"],
        },
        "semantic_slot_fragments_private": {
            "status_value": target,
            "rejected_value": rejected,
        },
    }


def rendered_case(
    tokenizer: Any,
    model: str,
    seed: int,
    condition: str,
    target: str,
    rejected: str,
) -> dict[str, Any]:
    item = raw_case(seed, target, rejected)
    raw_prompt = TEMPLATE.format(**item)
    prompt, add_special, answer_phase = interface_prompt(
        tokenizer, model, raw_prompt, "answer_aligned_chat"
    )
    return {
        **item,
        "prompt": prompt,
        "raw_prompt": raw_prompt,
        "source_fragment": item["context"],
        "query_fragment": item["question"],
        "tokenization_add_special_tokens": add_special,
        "interface": "answer_aligned_chat",
        "answer_phase": answer_phase,
        "condition": condition,
    }


def aligned_pair(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ids = prompt_token_ids(tokenizer, left)
    right_ids = prompt_token_ids(tokenizer, right)
    left_roles, left_audit = semantic_role_indices(tokenizer, left, len(left_ids) - 1)
    right_roles, right_audit = semantic_role_indices(tokenizer, right, len(right_ids) - 1)
    if left_audit["missing_fragments"] or right_audit["missing_fragments"]:
        return False
    if any(
        len(left_roles[role]) != len(right_roles[role])
        for role in REGISTERED_ROLES[:-1]
    ):
        return False
    left_status = fragment_positions(
        tokenizer, left_ids, left["semantic_slot_fragments_private"]["status_value"]
    )
    right_status = fragment_positions(
        tokenizer, right_ids, right["semantic_slot_fragments_private"]["status_value"]
    )
    left_rejected = fragment_positions(
        tokenizer, left_ids, left["semantic_slot_fragments_private"]["rejected_value"]
    )
    right_rejected = fragment_positions(
        tokenizer, right_ids, right["semantic_slot_fragments_private"]["rejected_value"]
    )
    if len(left_status) != len(right_status) or len(left_rejected) != len(right_rejected):
        return False
    left_target_ids = tokenizer(
        " " + left["target"], add_special_tokens=False
    )["input_ids"]
    right_target_ids = tokenizer(
        " " + right["target"], add_special_tokens=False
    )["input_ids"]
    return bool(left_target_ids and right_target_ids and left_target_ids[0] != right_target_ids[0])


def main() -> None:
    created_at = now()
    tokenizers = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    accepted: list[tuple[int, str, str, dict[str, dict[str, dict[str, Any]]]]] = []
    seed = 0
    label_pairs = list(itertools.permutations(LABELS, 2))
    while len(accepted) < GROUP_COUNT and seed < len(label_pairs):
        target_x, target_y = label_pairs[(seed * 37 + 11) % len(label_pairs)]
        model_cases: dict[str, dict[str, dict[str, Any]]] = {}
        valid = True
        for model in MODELS:
            tokenizer = tokenizers[model]
            left = rendered_case(
                tokenizer, model, seed, "mapping_x", target_x, target_y
            )
            right = rendered_case(
                tokenizer, model, seed, "mapping_y", target_y, target_x
            )
            if not aligned_pair(tokenizer, left, right):
                valid = False
                break
            model_cases[model] = {"mapping_x": left, "mapping_y": right}
        if valid:
            accepted.append((seed, target_x, target_y, model_cases))
        seed += 1
    if len(accepted) != GROUP_COUNT:
        raise RuntimeError(f"Only {len(accepted)} token-aligned Phase392 groups found")

    rows: list[dict[str, Any]] = []
    public_groups: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    for priority, (source_seed, target_x, target_y, model_cases) in enumerate(accepted):
        group_id = "p392g_" + digest(f"{source_seed}:{target_x}:{target_y}", 24)
        public_groups.append(
            {
                "parallel_group_id": group_id,
                "group_priority": priority,
                "source_seed": source_seed,
                "conditions": list(CONDITIONS),
                "token_aligned_all_models": True,
            }
        )
        for model in MODELS:
            tokenizer = tokenizers[model]
            for condition in CONDITIONS:
                item = model_cases[model][condition]
                prompt_hash = digest(item["prompt"])
                if prompt_hash in prompt_hashes:
                    raise RuntimeError("Duplicate Phase392 prompt")
                prompt_hashes.add(prompt_hash)
                case_id = "p392c_" + digest(f"{model}:{group_id}:{condition}", 26)
                rows.append(
                    {
                        "schema_version": "66.0.0",
                        "phase_id": "Phase392-Protocol",
                        "created_at": created_at,
                        "private_execution_model": model,
                        "blind_case_id": case_id,
                        "parallel_group_id": group_id,
                        "group_priority": priority,
                        "condition": condition,
                        "prompt": item["prompt"],
                        "raw_prompt": item["raw_prompt"],
                        "source_fragment": item["context"],
                        "query_fragment": item["question"],
                        "tokenization_add_special_tokens": item[
                            "tokenization_add_special_tokens"
                        ],
                        "interface": item["interface"],
                        "answer_phase": item["answer_phase"],
                        "target": item["target"],
                        "target_aliases": item["target_aliases"],
                        "distractors": item["distractors"],
                        "semantic_role_fragments_private": item[
                            "semantic_role_fragments_private"
                        ],
                        "semantic_slot_fragments_private": item[
                            "semantic_slot_fragments_private"
                        ],
                        "prompt_token_count": len(prompt_token_ids(tokenizer, item)),
                    }
                )
    if len(rows) != GROUP_COUNT * len(CONDITIONS) * len(MODELS):
        raise RuntimeError("Invalid Phase392 row count")
    write_jsonl(OUT / "protocol/private/phase392_candidate_cases.jsonl", rows)
    protocol = {
        "schema_version": "66.0.0",
        "phase_id": "Phase392-Protocol",
        "created_at": created_at,
        "objective": "graph_consistent_parent_boundary_joint_role_replay",
        "denominator": {
            "candidate_group_count": GROUP_COUNT,
            "conditions_per_group": len(CONDITIONS),
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "instrument_groups_required": 2,
            "causal_groups_required": 24,
        },
        "case_contract": {
            "same_record_owner_batch_query_within_pair": True,
            "status_and_rejected_values_swapped": True,
            "semantic_role_token_counts_match_within_pair_all_models": True,
            "status_slot_token_counts_match_within_pair_all_models": True,
            "target_first_token_candidates_distinct_all_models": True,
            "single_sample_model_native_interface": True,
        },
        "frozen_model_layers": {"qwen3": 20, "glm4": 22, "deepseek7b": 15},
        "frozen_interventions": [
            "no_intervention",
            "identity_semantic_joint",
            "donor_semantic_joint",
            "donor_attributes_only",
            "donor_fixed_best_role",
            "donor_frozen_structure_roles",
            "donor_same_count_random_parent_positions",
            "donor_semantic_joint_wrong_depth",
        ],
        "frozen_outcomes": [
            "query_layer_output_shift_toward_donor",
            "donor_vs_recipient_target_logit_margin_shift",
            "joint_generation_switch_to_donor_target",
            "full_vocabulary_top_change",
        ],
        "frozen_gates": {
            "identity_max_abs_error": 0.01,
            "median_joint_normalized_margin_mediation": 0.10,
            "median_joint_advantage_over_fixed_role": 0.05,
            "median_joint_advantage_over_attributes_only": 0.05,
            "median_joint_advantage_over_random_positions": 0.05,
            "median_joint_advantage_over_wrong_depth": 0.05,
            "minimum_positive_direction_rate": 0.75,
            "minimum_strict_answer_switch_rate_for_function_path": 0.50,
            "all_three_models_required_for_shared_causal_path": True,
        },
        "selection_rule": (
            "first two behavior-and-position qualified groups are engineering instruments; "
            "next 24 are causal test; no replacement after intervention starts"
        ),
        "authorization": {
            "intervention_before_behavior_freeze": False,
            "single_neuron_scan": False,
        },
        "groups": public_groups,
    }
    write_json(OUT / "phase392_protocol.json", protocol)
    print(json.dumps(protocol["denominator"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
