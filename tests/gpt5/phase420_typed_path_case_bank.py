#!/usr/bin/env python3
"""Freeze the Phase420 fully crossed history/current-identity denominator.

Phase419 held the current task fixed, so compatible/conflicting history also
changed the literal history token.  Phase420 uses two independently valid
current tasks from the same mechanism and crosses their answer identities with
both history identities.  Candidate pairs are admitted only when all four
cells have the same full-prompt token count for both interfaces and all three
models.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


PHASE_ID = "Phase420-TypedNaturalPathDenominator"
SCHEMA_VERSION = "93.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("chat", "completion")
CURRENT_IDENTITIES = ("a", "b")
HISTORY_IDENTITIES = ("a", "b")
SHARED_SUFFIX = "\nFinal answer:"
HISTORY_QUESTION = "Earlier draft answer for this same task:"

SOURCE_345 = ROOT / "tests/gpt5/result/phase345_three_core_protocol/three_core_protocol_qualification/phase345_registered_cases.jsonl"
SOURCE_346 = ROOT / "tests/gpt5/result/phase346_protocol_repair/three_core_protocol_repair/phase346_registered_cases.jsonl"
OUT = ROOT / "tests/gpt5/result/phase420_typed_path_atlas"

MECHANISMS = (
    "context_relation_binding",
    "parameter_knowledge_retrieval",
    "explicit_copy_control",
    "two_hop_entailment",
    "direct_fact_control",
    "sentence_past_tense",
    "no_morphology_control",
    "answer_only_protocol",
    "contiguous_multi_token_answer",
    "simple_no_source_answer",
)
SOURCE_SPLIT_TO_PHASE420 = {
    "discovery": "discovery",
    "calibration": "calibration",
    "heldout": "behavior_holdout",
    "private_heldout": "physical_holdout",
}
SPLIT_QUOTAS = {
    "discovery": {
        "knowledge_network": 5,
        "reasoning": 3,
        "grammar": 3,
        "protocol_control": 4,
    },
    "calibration": {
        "knowledge_network": 2,
        "reasoning": 1,
        "grammar": 1,
        "protocol_control": 2,
    },
    "behavior_holdout": {
        "knowledge_network": 2,
        "reasoning": 1,
        "grammar": 1,
        "protocol_control": 2,
    },
    "physical_holdout": {
        "knowledge_network": 2,
        "reasoning": 1,
        "grammar": 1,
        "protocol_control": 2,
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    return AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )


def serialize_crossed_prompt(
    tokenizer: Any,
    raw_prompt: str,
    interface: str,
    history_answer: str,
) -> tuple[str, list[dict[str, str]]]:
    messages = [
        {"role": "user", "content": HISTORY_QUESTION},
        {"role": "assistant", "content": history_answer},
        {"role": "user", "content": raw_prompt},
    ]
    if interface == "chat":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif interface == "completion":
        prompt = "\n".join(
            [
                "Previous exchange:",
                f"Question: {HISTORY_QUESTION}",
                f"Answer: {history_answer}",
                "Current task:",
                raw_prompt,
            ]
        )
    else:
        raise KeyError(interface)
    return prompt.rstrip() + SHARED_SUFFIX, messages


def source_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(SOURCE_345) + read_jsonl(SOURCE_346)
    rows = [row for row in rows if row["mechanism_id"] in MECHANISMS]
    augmented = list(rows)
    for row in rows:
        if row["mechanism_id"] not in {"direct_fact_control", "two_hop_entailment"}:
            continue
        query = row["query_fragment"]
        marker = query.rfind("set")
        if marker < 0:
            raise RuntimeError(f"Cannot build reasoning counterfactual: {query}")
        end = marker + 3
        while end < len(query) and query[end].isdigit():
            end += 1
        source_number = int(query[marker + 3 : end])
        negative_query = query[: marker + 3] + str(source_number + 100) + query[end:]
        negative_raw = row["raw_prompt"].replace(query, negative_query, 1)
        if negative_raw == row["raw_prompt"]:
            raise RuntimeError(f"Counterfactual query replacement failed: {row['case_id']}")
        negative = {
            **row,
            "case_id": row["case_id"] + "_registered_negative",
            "semantic_case_id": row["semantic_case_id"] + "_registered_negative",
            "item_index": int(row["item_index"]) + 100,
            "raw_prompt": negative_raw,
            "query_fragment": negative_query,
            "target": "no",
            "target_aliases": ["no"],
            "distractors": ["yes", "unknown"],
            "phase420_counterfactual_current_state": True,
            "phase420_counterpart_semantic_case_id": row["semantic_case_id"],
        }
        augmented.append(negative)
        row["phase420_counterfactual_current_state"] = False
        row["phase420_counterpart_semantic_case_id"] = negative["semantic_case_id"]
    return augmented


def candidate_pairs(
    rows: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_model_semantic = {(row["model"], row["semantic_case_id"]): row for row in rows}
    qwen = [row for row in rows if row["model"] == "qwen3"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in qwen:
        groups[(row["mechanism_id"], row["template_id"], row["split"])].append(row)

    audit = Counter()
    candidates: list[dict[str, Any]] = []
    for (mechanism, template, source_split), values in sorted(groups.items()):
        phase_split = SOURCE_SPLIT_TO_PHASE420[source_split]
        for left, right in combinations(sorted(values, key=lambda row: int(row["item_index"])), 2):
            audit["candidate_pair_count"] += 1
            if left["target"] == right["target"]:
                audit["same_target_rejection_count"] += 1
                continue
            semantic_a = left["semantic_case_id"]
            semantic_b = right["semantic_case_id"]
            prompt_counts: dict[str, dict[str, int]] = {}
            target_token_counts: dict[str, dict[str, int]] = {}
            valid = True
            for model in MODELS:
                model_a = by_model_semantic[(model, semantic_a)]
                model_b = by_model_semantic[(model, semantic_b)]
                tokenizer = tokenizers[model]
                target_token_counts[model] = {
                    "a": len(tokenizer(model_a["target"], add_special_tokens=False)["input_ids"]),
                    "b": len(tokenizer(model_b["target"], add_special_tokens=False)["input_ids"]),
                }
                prompt_counts[model] = {}
                for interface in INTERFACES:
                    counts_by_current: dict[str, list[int]] = defaultdict(list)
                    for current in CURRENT_IDENTITIES:
                        current_row = model_a if current == "a" else model_b
                        for history in HISTORY_IDENTITIES:
                            history_answer = model_a["target"] if history == "a" else model_b["target"]
                            prompt, _ = serialize_crossed_prompt(
                                tokenizer,
                                current_row["raw_prompt"],
                                interface,
                                history_answer,
                            )
                            counts_by_current[current].append(
                                len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
                            )
                    prompt_counts[model][interface] = {
                        current: counts[0] for current, counts in counts_by_current.items()
                    }
                    valid = valid and all(len(set(counts)) == 1 for counts in counts_by_current.values())
            if not valid:
                audit["full_prompt_shape_rejection_count"] += 1
                continue
            audit["qualified_candidate_pair_count"] += 1
            candidates.append(
                {
                    "mechanism_id": mechanism,
                    "family_id": left["family_id"],
                    "template_id": template,
                    "source_split": source_split,
                    "split": phase_split,
                    "semantic_case_a": semantic_a,
                    "semantic_case_b": semantic_b,
                    "item_index_a": int(left["item_index"]),
                    "item_index_b": int(right["item_index"]),
                    "target_a": left["target"],
                    "target_b": right["target"],
                    "prompt_token_counts": prompt_counts,
                    "target_token_counts": target_token_counts,
                    "current_prompt_token_count_exact_across_identities": all(
                        prompt_counts[model][interface]["a"]
                        == prompt_counts[model][interface]["b"]
                        for model in MODELS
                        for interface in INTERFACES
                    ),
                }
            )
    return candidates, dict(audit)


def select_pairs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split, family_quotas in SPLIT_QUOTAS.items():
        for family, quota in family_quotas.items():
            pool = [row for row in candidates if row["split"] == split and row["family_id"] == family]
            mechanism_use = Counter()
            template_use = Counter()
            used_items: set[tuple[str, int]] = set()
            for _ in range(quota):
                eligible = [
                    row
                    for row in pool
                    if (row["mechanism_id"], row["item_index_a"]) not in used_items
                    and (row["mechanism_id"], row["item_index_b"]) not in used_items
                    and row not in selected
                ]
                if not eligible:
                    raise RuntimeError(f"No disjoint Phase420 pair left for {split}/{family}")
                choice = min(
                    eligible,
                    key=lambda row: (
                        mechanism_use[row["mechanism_id"]],
                        template_use[row["template_id"]],
                        row["item_index_a"],
                        row["item_index_b"],
                        row["mechanism_id"],
                        row["template_id"],
                    ),
                )
                selected.append(choice)
                mechanism_use[choice["mechanism_id"]] += 1
                template_use[choice["template_id"]] += 1
                used_items.add((choice["mechanism_id"], choice["item_index_a"]))
                used_items.add((choice["mechanism_id"], choice["item_index_b"]))
    return selected


def freeze() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = source_rows()
    index = {(row["model"], row["semantic_case_id"]): row for row in rows}
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    candidates, candidate_audit = candidate_pairs(rows, tokenizers)
    selected = select_pairs(candidates)
    created_at = now()
    frozen_groups = []
    conditions = []
    for group_index, pair in enumerate(selected):
        group_id = (
            f"phase420_group_{group_index:02d}_{pair['mechanism_id']}_"
            f"{pair['item_index_a']:02d}_{pair['item_index_b']:02d}_{pair['template_id']}"
        )
        frozen_groups.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": created_at,
                "group_id": group_id,
                **pair,
                "fully_crossed": True,
                "full_prompt_history_pair_token_count_exact_within_current_state": True,
                "causal": False,
            }
        )
        for model in MODELS:
            source_a = index[(model, pair["semantic_case_a"])]
            source_b = index[(model, pair["semantic_case_b"])]
            for interface in INTERFACES:
                for current_identity in CURRENT_IDENTITIES:
                    current_source = source_a if current_identity == "a" else source_b
                    current_target = source_a["target"] if current_identity == "a" else source_b["target"]
                    other_target = source_b["target"] if current_identity == "a" else source_a["target"]
                    for history_identity in HISTORY_IDENTITIES:
                        history_answer = source_a["target"] if history_identity == "a" else source_b["target"]
                        compatible = current_identity == history_identity
                        condition_id = (
                            f"phase420_{model}_{group_index:02d}_{interface}_"
                            f"current_{current_identity}_history_{history_identity}"
                        )
                        conditions.append(
                            {
                                **current_source,
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": PHASE_ID,
                                "created_at": created_at,
                                "phase420_condition_id": condition_id,
                                "group_id": group_id,
                                "group_index": group_index,
                                "split": pair["split"],
                                "source_split": pair["source_split"],
                                "interface": interface,
                                "current_identity": current_identity,
                                "history_identity": history_identity,
                                "history_compatible": compatible,
                                "history_condition": "compatible" if compatible else "conflict",
                                "history_answer": history_answer,
                                "target": current_target,
                                "target_aliases": [current_target],
                                "opposite_identity_target": other_target,
                                "identity_target_a": source_a["target"],
                                "identity_target_b": source_b["target"],
                                "semantic_case_a": pair["semantic_case_a"],
                                "semantic_case_b": pair["semantic_case_b"],
                                "registered_prompt_token_count": pair["prompt_token_counts"][model][interface][current_identity],
                                "registered_target_token_count": pair["target_token_counts"][model][current_identity],
                                "full_prompt_history_pair_token_count_exact": True,
                                "full_prompt_four_cell_token_count_exact": pair[
                                    "current_prompt_token_count_exact_across_identities"
                                ],
                                "behavior_collection_authorized": True,
                                "development_physical_collection_authorized": pair["split"] != "physical_holdout",
                                "physical_holdout_sealed": pair["split"] == "physical_holdout",
                                "causal_intervention_authorized": False,
                                "single_neuron_scan_authorized": False,
                            }
                        )
    conditions.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            row["group_index"],
            INTERFACES.index(row["interface"]),
            CURRENT_IDENTITIES.index(row["current_identity"]),
            HISTORY_IDENTITIES.index(row["history_identity"]),
        )
    )
    group_splits = Counter(row["split"] for row in frozen_groups)
    family_splits = Counter((row["split"], row["family_id"]) for row in frozen_groups)
    model_counts = Counter(row["model"] for row in conditions)
    valid = bool(
        len(frozen_groups) == 33
        and len(conditions) == 792
        and all(model_counts[model] == 264 for model in MODELS)
        and group_splits == Counter({"discovery": 15, "calibration": 6, "behavior_holdout": 6, "physical_holdout": 6})
        and len({row["phase420_condition_id"] for row in conditions}) == len(conditions)
        and all(row["full_prompt_history_pair_token_count_exact"] for row in conditions)
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": created_at,
        "objective": "trace_typed_history_and_current_source_writes_after_full_current_by_history_identity_cross",
        "model_order": list(MODELS),
        "interfaces": list(INTERFACES),
        "current_identities": list(CURRENT_IDENTITIES),
        "history_identities": list(HISTORY_IDENTITIES),
        "group_count": len(frozen_groups),
        "condition_count": len(conditions),
        "group_split_count": dict(group_splits),
        "family_split_count": {f"{split}:{family}": count for (split, family), count in sorted(family_splits.items())},
        "denominator_correction": {
            "phase419_cases_are_not_intrinsically_two_current_state_groups": True,
            "phase420_rebuilds_pairs_from_full_phase345_phase346_qualified_case_banks": True,
            "current_a_and_b_are_independently_valid_same_mechanism_tasks": True,
            "history_pair_full_prompt_token_count_is_exact_within_each_current_identity": True,
            "current_a_b_prompt_lengths_are_registered_nuisance_not_assumed_equal": True,
            "reasoning_b_states_are_registered_same_mechanism_counterfactual_queries": True,
        },
        "registered_contrasts": {
            "compatibility_effect": "0.5*((X_a_b-X_a_a)+(X_b_a-X_b_b))",
            "history_identity_main_effect": "0.5*((X_a_b-X_a_a)+(X_b_b-X_b_a))",
            "current_identity_main_effect": "0.5*((X_b_a-X_a_a)+(X_b_b-X_a_b))",
        },
        "execution_contract": {
            "behavior_max_new_tokens": 12,
            "behavior_horizon_separate_from_prefill_physical_horizon": True,
            "development_physical_splits": ["discovery", "calibration", "behavior_holdout"],
            "physical_holdout_requires_all_prediction_gates": True,
            "output_attentions_required": True,
            "actual_qkv_capture_required": True,
            "attention_and_mlp_reconstruction_error_max": 0.01,
        },
        "claim_boundary": "typed_natural_path_observation_and_holdout_prediction_only_no_causal_or_neuron_claim",
    }
    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase420-DenominatorQualification",
        "created_at": created_at,
        "valid": valid,
        "candidate_audit": candidate_audit,
        "qualified_candidate_pair_count": len(candidates),
        "selected_group_count": len(frozen_groups),
        "condition_count": len(conditions),
        "model_condition_count": dict(model_counts),
        "group_split_count": dict(group_splits),
        "behavior_collection_authorized": valid,
        "development_physical_collection_authorized": valid,
        "physical_holdout_collection_authorized": False,
        "causal_intervention_authorized": False,
        "single_neuron_scan_authorized": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    return frozen_groups, conditions, protocol, qualification


def main() -> None:
    groups, conditions, protocol, qualification = freeze()
    write_jsonl(OUT / "phase420_frozen_groups.jsonl", groups)
    write_jsonl(OUT / "phase420_registered_conditions.jsonl", conditions)
    write_json(OUT / "phase420_protocol.json", protocol)
    write_json(OUT / "phase420_denominator_qualification.json", qualification)
    print(json.dumps(qualification, ensure_ascii=False, indent=2))
    if not qualification["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
