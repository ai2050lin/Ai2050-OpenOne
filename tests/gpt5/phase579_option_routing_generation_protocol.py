#!/usr/bin/env python3
"""Freeze full-generation validation for confirmed Phase579 branches."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase578_choice_world_protocol as source  # noqa: E402
import phase579_option_routing_causal_protocol as causal  # noqa: E402


CONFIRMATION_DECISION_PATH = (
    source.OUT_DIR / "phase579_option_routing_causal_confirmation_decision.json"
)
PROTOCOL_PATH = source.OUT_DIR / "phase579_option_routing_generation_protocol.json"
CONDITIONS = (
    "natural_baseline",
    "option_score_swap",
    "object_relation_score_swap_control",
    "option_score_swap_restore",
    "option_weight_swap",
    "option_value_swap_positive_control",
)
REPEATS = ("repeat1", "repeat2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def world_relation_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    with gzip.open(source.SOURCE_CASES_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            lookup.setdefault(row["world_id"], row["relation"])
    return lookup


def freeze() -> Path:
    causal_protocol = read_json(causal.PROTOCOL_PATH)
    confirmation = read_json(CONFIRMATION_DECISION_PATH)
    if not confirmation["any_branch_passed"]:
        raise RuntimeError("Phase579 confirmation authorized no generation branch")
    if confirmation["sealed_split_read"]:
        raise RuntimeError("Phase579 confirmation crossed the sealed boundary")
    branches = {
        model: relations
        for model, relations in confirmation[
            "confirmed_generation_relations_by_model"
        ].items()
        if relations
    }
    if not branches:
        raise RuntimeError("Phase579 has no confirmed model-relation branch")
    relation_by_world = world_relation_lookup()
    world_ids = {}
    for model, relations in branches.items():
        allowed = set(relations)
        contract_path = (
            source.OUT_DIR
            / f"phase579_{model}_causal_confirmation_option_routing_causal_contract.json"
        )
        contract = read_json(contract_path)
        if set(contract["world_ids"]) != set(
            causal_protocol["causal_holdout_world_ids_by_model_and_split"][model][
                "causal_confirmation"
            ]
        ):
            # The confirmation executor is allowed to filter failed relations.
            if contract["world_count"] != len(contract["world_ids"]):
                raise RuntimeError(f"Phase579 confirmation contract drift: {model}")
        world_ids[model] = [
            world_id
            for world_id in contract["world_ids"]
            if relation_by_world[world_id] in allowed
        ]
        if not world_ids[model]:
            raise RuntimeError(f"Phase579 generation selected no worlds: {model}")
        selected_relations = {
            causal_protocol["selected_coordinates_by_model_and_relation"][model][
                relation
            ]["relation"]
            for relation in allowed
        }
        if selected_relations != allowed:
            raise RuntimeError(f"Phase579 confirmed relation drift: {model}")

    payload = {
        "schema_version": "phase579_option_routing_generation_protocol.v1",
        "phase_id": causal_protocol["phase_id"],
        "created_at": now(),
        "confirmed_relations_by_model": branches,
        "world_ids_by_model": world_ids,
        "selected_coordinates_by_model_and_relation": {
            model: {
                relation: causal_protocol[
                    "selected_coordinates_by_model_and_relation"
                ][model][relation]
                for relation in relations
            }
            for model, relations in branches.items()
        },
        "conditions": list(CONDITIONS),
        "repeats": list(REPEATS),
        "execution": {
            "device": "cuda",
            "torch_dtype": "torch.bfloat16",
            "batch_size": 8,
            "max_new_tokens": 8,
            "do_sample": False,
            "left_padding_and_explicit_position_ids": True,
            "patch_only_first_prompt_forward": True,
            "patch_all_heads": True,
        },
        "generation_gate": {
            "minimum_natural_target_rate_each_order": 0.90,
            "minimum_score_swap_target_rate_drop_each_order": 0.05,
            "minimum_score_swap_foil_rate_gain_each_order": 0.05,
            "minimum_score_swap_vs_nonoption_target_drop_gap_each_order": 0.03,
            "minimum_repeat_exact_match_rate": 1.0,
            "minimum_restore_exact_match_to_natural_rate": 1.0,
            "both_option_orders_must_pass": True,
        },
        "claim_boundary": {
            "full_generation_pass_is_local_candidate_selection_only": True,
            "does_not_locate_parametric_knowledge": True,
            "does_not_close_object_attribute_binding": True,
        },
        "sealed_split_read": False,
        "causal_protocol_sha256": sha256_file(causal.PROTOCOL_PATH),
        "confirmation_decision_sha256": sha256_file(CONFIRMATION_DECISION_PATH),
        "source_cases_sha256": sha256_file(source.SOURCE_CASES_PATH),
    }
    if PROTOCOL_PATH.exists():
        existing = read_json(PROTOCOL_PATH)
        ignored = {"created_at"}
        if {key: value for key, value in existing.items() if key not in ignored} != {
            key: value for key, value in payload.items() if key not in ignored
        }:
            raise RuntimeError("Phase579 generation protocol drift")
    else:
        write_json(PROTOCOL_PATH, payload)
    print(
        json.dumps(
            {
                "confirmed_relations_by_model": branches,
                "world_count_by_model": {
                    model: len(ids) for model, ids in world_ids.items()
                },
                "condition_count": len(CONDITIONS),
                "repeat_count": len(REPEATS),
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return PROTOCOL_PATH


if __name__ == "__main__":
    freeze()
