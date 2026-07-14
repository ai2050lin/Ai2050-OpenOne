#!/usr/bin/env python3
"""Freeze exact-token matched compatible/conflicting history pairs."""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase416_dual_track_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402
from phase418_interface_history_case_bank import MODELS, SCHEMA_VERSION  # noqa: E402
from phase418_interface_history_trace import serialize_prompt  # noqa: E402


SOURCE = ROOT / "tests/gpt5/result/phase418_interface_history_atlas/phase418_registered_conditions.jsonl"
OUT = ROOT / "tests/gpt5/result/phase419_token_matched_history_atlas"
PHASE_ID = "Phase419-TokenMatchedHistoryDenominator"
INTERFACES = ("chat", "completion")
HISTORIES = ("compatible", "conflict")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    return AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )


def matching_distractors(tokenizer: Any, source: dict[str, Any]) -> list[dict[str, Any]]:
    matches = []
    for distractor in source["distractors"]:
        counts = {}
        valid = True
        for interface in INTERFACES:
            compatible = copy.deepcopy(source)
            compatible["interface"] = interface
            compatible["history_condition"] = "compatible"
            conflict = copy.deepcopy(source)
            conflict["interface"] = interface
            conflict["history_condition"] = "conflict"
            conflict["distractors"] = [distractor]
            compatible_prompt, _ = serialize_prompt(tokenizer, compatible)
            conflict_prompt, _ = serialize_prompt(tokenizer, conflict)
            left = len(tokenizer(compatible_prompt, add_special_tokens=True)["input_ids"])
            right = len(tokenizer(conflict_prompt, add_special_tokens=True)["input_ids"])
            counts[interface] = {"compatible": left, "conflict": right}
            valid = valid and left == right
        if valid:
            matches.append({"distractor": distractor, "prompt_token_counts": counts})
    return matches


def freeze() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = read_jsonl(SOURCE)
    base = {
        (row["model"], row["semantic_case_id"]): row
        for row in rows
        if row["interface"] == "chat" and row["history_condition"] == "none"
    }
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    matches = {
        key: matching_distractors(tokenizers[key[0]], source)
        for key, source in base.items()
    }
    semantic_ids = sorted({semantic_id for _model, semantic_id in base})
    common = [
        semantic_id
        for semantic_id in semantic_ids
        if all(matches[(model, semantic_id)] for model in MODELS)
    ]
    created_at = now()
    conditions = []
    for model in MODELS:
        for semantic_id in common:
            source = base[(model, semantic_id)]
            choice = matches[(model, semantic_id)][0]
            for interface in INTERFACES:
                for history in HISTORIES:
                    conditions.append(
                        {
                            **source,
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE_ID,
                            "created_at": created_at,
                            "phase419_condition_id": f"phase419_{model}_{semantic_id}_{interface}_{history}",
                            "phase419_source_condition_id": source["phase418_condition_id"],
                            "interface": interface,
                            "history_condition": history,
                            "distractors": [choice["distractor"]],
                            "matched_conflict_answer": choice["distractor"],
                            "registered_prompt_token_count": choice["prompt_token_counts"][interface][history],
                            "compatible_conflict_prompt_token_count_exact": True,
                            "same_history_structure": True,
                            "same_current_task_text": True,
                            "same_terminal_literal": True,
                            "answer_identity_is_intended_manipulation": True,
                            "causal_intervention_authorized": False,
                            "single_neuron_scan_authorized": False,
                        }
                    )
    conditions.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            row["family_id"],
            row["semantic_case_id"],
            INTERFACES.index(row["interface"]),
            HISTORIES.index(row["history_condition"]),
        )
    )
    family_semantic = Counter(base[("qwen3", semantic_id)]["family_id"] for semantic_id in common)
    per_model = Counter(row["model"] for row in conditions)
    valid = bool(
        len(common) == 33
        and len(conditions) == 396
        and all(per_model[model] == 132 for model in MODELS)
        and all(row["compatible_conflict_prompt_token_count_exact"] for row in conditions)
        and len({row["phase419_condition_id"] for row in conditions}) == len(conditions)
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": created_at,
        "objective": "isolate_prior_answer_identity_with_exact_prompt_token_count_pairs",
        "model_order": list(MODELS),
        "interfaces": list(INTERFACES),
        "histories": list(HISTORIES),
        "cross_model_semantic_case_count": len(common),
        "family_semantic_case_count": dict(sorted(family_semantic.items())),
        "condition_count_per_model": 132,
        "condition_count": len(conditions),
        "pair_contract": {
            "same_prior_question": True,
            "same_current_task": True,
            "same_interface_serialization": True,
            "same_full_prompt_token_count_within_interface": True,
            "compatible_answer": "registered target",
            "conflict_answer": "registered distractor with exact prompt-token-count match",
            "answer_token_identity_differs_by_design": True,
        },
        "registered_contrasts": {
            "chat_history_identity": "chat/conflict-chat/compatible",
            "completion_history_identity": "completion/conflict-completion/compatible",
            "interface_interaction": "(completion/conflict-completion/compatible)-(chat/conflict-chat/compatible)",
        },
        "analysis_contract": {
            "exact_vector_differences": True,
            "component_relative_delta": True,
            "direction_consistency": True,
            "discovery_non_discovery_separation": True,
            "cross_model_region_only_not_hidden_vector_alignment": True,
        },
        "stop_rules": [
            "any_pair_token_count_mismatch_blocks_that_semantic_case",
            "nonfinite_or_component_ledger_failure_blocks_model_atlas",
            "direction_or_region_replication_does_not_authorize_causality",
            "language_family_names_remain_external_task_labels",
        ],
        "claim_boundary": "exact_length_matched_prior_answer_identity_physical_difference_only",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase419-TokenMatchedDenominatorQualification",
        "created_at": created_at,
        "valid": valid,
        "source_semantic_case_count": len(semantic_ids),
        "qualified_cross_model_semantic_case_count": len(common),
        "excluded_semantic_case_count": len(semantic_ids) - len(common),
        "family_semantic_case_count": dict(sorted(family_semantic.items())),
        "condition_count": len(conditions),
        "model_condition_count": dict(per_model),
        "exact_prompt_token_count_pair_count": len(common) * len(MODELS) * len(INTERFACES),
        "model_execution_authorized": valid,
        "causal_intervention_authorized": False,
        "single_neuron_scan_authorized": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    return conditions, protocol, summary


def main() -> None:
    conditions, protocol, summary = freeze()
    write_jsonl(OUT / "phase419_registered_conditions.jsonl", conditions)
    write_json(OUT / "phase419_protocol.json", protocol)
    write_json(OUT / "phase419_denominator_qualification.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
