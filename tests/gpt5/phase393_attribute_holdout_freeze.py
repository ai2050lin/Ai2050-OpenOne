#!/usr/bin/env python3
"""Freeze unused Phase392 groups for an independent Phase393 attribute holdout."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase379_global_layout_protocol import first_target_step  # noqa: E402
from phase390_role_mapping import REGISTERED_ROLES, prompt_token_ids, semantic_role_indices  # noqa: E402


P392 = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
OUT = ROOT / "tests/gpt5/result/phase393_attribute_content_holdout"
MODELS = ("qwen3", "glm4", "deepseek7b")
LAYERS = {"qwen3": 20, "glm4": 22, "deepseek7b": 15}
LAYER_COUNTS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
STRUCTURE_ROLES = ("entities", "relations", "query_keywords", "query_window")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    phase392 = read_json(P392 / "phase392_causal_summary.json")
    if phase392["results"]["crossmodel_language_function_path_established"]:
        raise RuntimeError("Phase393 fallback is unnecessary after a Phase392 shared path")
    used = set(read_json(P392 / "phase392_intervention_freeze.json")["selection"]["selected_groups"])
    cases = {
        row["blind_case_id"]: row
        for row in read_jsonl(P392 / "protocol/private/phase392_candidate_cases.jsonl")
    }
    tokenizers = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=True, local_files_only=True, use_fast=False
        )
    behavior = [
        row
        for model in MODELS
        for row in read_jsonl(P392 / "behavior/private" / model / "rows.jsonl")
    ]
    enriched = []
    for row in behavior:
        case = cases[row["blind_case_id"]]
        tokenizer = tokenizers[row["model"]]
        step = first_target_step(tokenizer, row["generated_token_ids"], case["target_aliases"])
        ids = prompt_token_ids(tokenizer, case)
        partition, audit = semantic_role_indices(tokenizer, case, len(ids) - 1)
        enriched.append(
            {
                **row,
                "target_step": step,
                "has_post_target_token": step is not None and step + 1 < len(row["generated_token_ids"]),
                "prompt_token_ids_private": ids,
                "role_positions_private": partition,
                "role_mapping_valid": not audit["missing_fragments"] and audit["partition_conserved"],
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["parallel_group_id"]].append(row)
    eligible = []
    for group_id, rows in grouped.items():
        if group_id in used or len(rows) != 6:
            continue
        valid = all(
            row["strict_behavior_correct"]
            and row["has_post_target_token"]
            and row["role_mapping_valid"]
            for row in rows
        )
        if valid:
            for model in MODELS:
                pair = {row["condition"]: row for row in rows if row["model"] == model}
                left, right = pair["mapping_x"], pair["mapping_y"]
                valid = valid and left["generated_token_ids"][left["target_step"]] != right["generated_token_ids"][right["target_step"]]
                valid = valid and all(
                    len(left["role_positions_private"][role]) == len(right["role_positions_private"][role])
                    for role in REGISTERED_ROLES[:-1]
                )
        if valid:
            eligible.append(group_id)
    priorities = {row["parallel_group_id"]: row["group_priority"] for row in cases.values()}
    selected = sorted(eligible, key=lambda value: priorities[value])[:12]
    if len(selected) != 12:
        raise RuntimeError(f"Only {len(selected)} unused Phase393 holdout groups available")
    frozen = []
    for row in enriched:
        if row["parallel_group_id"] not in selected:
            continue
        case = cases[row["blind_case_id"]]
        step = int(row["target_step"])
        model = row["model"]
        frozen.append(
            {
                **case,
                "schema_version": "67.0.0",
                "phase_id": "Phase393-FrozenAttributeHoldout",
                "generated_token_ids_private": row["generated_token_ids"],
                "target_decision_prefix_token_ids_private": row["generated_token_ids"][:step],
                "target_first_token_id_private": int(row["generated_token_ids"][step]),
                "prompt_token_ids_private": row["prompt_token_ids_private"],
                "role_positions_private": row["role_positions_private"],
                "candidate_layer": LAYERS[model],
                "wrong_depth_layer": LAYER_COUNTS[model] - 1 - LAYERS[model],
                "frozen_structure_roles": list(STRUCTURE_ROLES),
            }
        )
    write_jsonl(OUT / "protocol/private/phase393_holdout_cases.jsonl", frozen)
    protocol = {
        "schema_version": "67.0.0",
        "phase_id": "Phase393-AttributeHoldoutProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis_source": "Phase392 joint patch explained by attributes-only mediation",
        "denominator": {
            "models": list(MODELS),
            "unused_qualified_group_count_before_selection": len(eligible),
            "frozen_group_count": len(selected),
            "case_count": len(frozen),
            "direction_count": len(selected) * 2 * len(MODELS),
        },
        "independence": {
            "group_overlap_with_phase392_intervention": 0,
            "groups_selected_before_phase393_intervention": True,
            "failed_groups_replaceable_after_intervention": False,
        },
        "scenarios": [
            "no_intervention",
            "identity_attributes",
            "donor_attributes_candidate_depth",
            "donor_structure_candidate_depth",
            "donor_random_candidate_depth",
            "donor_attributes_wrong_depth",
        ],
        "frozen_gates": {
            "median_attribute_normalized_margin_mediation": 0.10,
            "median_attribute_advantage_over_structure": 0.05,
            "median_attribute_advantage_over_random": 0.05,
            "minimum_positive_attribute_direction_rate": 0.75,
            "minimum_attribute_answer_switch_rate": 0.75,
            "minimum_candidate_depth_advantage_for_depth_specificity": 0.05,
            "all_three_models_required_for_shared_attribute_transport": True,
        },
        "claim_boundary": {
            "attribute_transport_is_multi_source_joint_path": False,
            "attribute_transport_is_field_extraction_algorithm_closure": False,
            "depth_nonspecific_transport_is_specialized_layer_path": False,
            "single_neuron_scan_authorized": False,
        },
        "selected_groups": selected,
    }
    write_json(OUT / "phase393_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
