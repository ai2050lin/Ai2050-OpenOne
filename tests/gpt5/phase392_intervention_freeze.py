#!/usr/bin/env python3
"""Freeze disjoint Phase392 engineering and 24-group causal denominators."""

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


OUT = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
MODELS = ("qwen3", "glm4", "deepseek7b")
LAYERS = {"qwen3": 20, "glm4": 22, "deepseek7b": 15}
LAYER_COUNTS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
BEST_ROLES = {"qwen3": "relations", "glm4": "query_window", "deepseek7b": "relations"}
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
    cases = {row["blind_case_id"]: row for row in read_jsonl(OUT / "protocol/private/phase392_candidate_cases.jsonl")}
    tokenizers = {}
    for model in MODELS:
        complete = read_json(OUT / "behavior" / model / "complete.json")
        if not complete["valid"] or complete["batch_size"] != 1:
            raise RuntimeError(f"Invalid Phase392 behavior for {model}")
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=True, local_files_only=True, use_fast=False
        )
    behavior = [
        row
        for model in MODELS
        for row in read_jsonl(OUT / "behavior/private" / model / "rows.jsonl")
    ]
    enriched: list[dict[str, Any]] = []
    for row in behavior:
        case = cases[row["blind_case_id"]]
        tokenizer = tokenizers[row["model"]]
        step = first_target_step(tokenizer, row["generated_token_ids"], case["target_aliases"])
        ids = prompt_token_ids(tokenizer, case)
        partition, role_audit = semantic_role_indices(tokenizer, case, len(ids) - 1)
        valid_step = step is not None and step + 1 < len(row["generated_token_ids"])
        enriched.append(
            {
                **row,
                "target_step": step,
                "has_post_target_token": valid_step,
                "prompt_token_ids_private": ids,
                "role_positions_private": partition,
                "role_mapping_valid": not role_audit["missing_fragments"] and role_audit["partition_conserved"],
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[row["parallel_group_id"]].append(row)
    eligible = []
    for group_id, rows in grouped.items():
        if (
            len(rows) == 6
            and {row["model"] for row in rows} == set(MODELS)
            and all(row["strict_behavior_correct"] and row["has_post_target_token"] and row["role_mapping_valid"] for row in rows)
        ):
            distinct = True
            for model in MODELS:
                model_rows = {row["condition"]: row for row in rows if row["model"] == model}
                left = model_rows["mapping_x"]
                right = model_rows["mapping_y"]
                if left["generated_token_ids"][left["target_step"]] == right["generated_token_ids"][right["target_step"]]:
                    distinct = False
                if any(
                    len(left["role_positions_private"][role]) != len(right["role_positions_private"][role])
                    for role in REGISTERED_ROLES[:-1]
                ):
                    distinct = False
            if distinct:
                eligible.append(group_id)
    priority = {
        row["parallel_group_id"]: int(row["group_priority"])
        for row in cases.values()
    }
    ordered = sorted(eligible, key=lambda group_id: priority[group_id])
    if len(ordered) < 26:
        raise RuntimeError(f"Only {len(ordered)} Phase392 groups qualified; 26 required")
    selected = ordered[:26]
    assignment = {group_id: ("instrument_audit" if index < 2 else "causal_test") for index, group_id in enumerate(selected)}
    frozen: list[dict[str, Any]] = []
    for row in enriched:
        if row["parallel_group_id"] not in assignment:
            continue
        case = cases[row["blind_case_id"]]
        step = int(row["target_step"])
        model = row["model"]
        frozen.append(
            {
                **case,
                "schema_version": "66.2.0",
                "phase_id": "Phase392-FrozenInterventionDenominator",
                "phase392_split": assignment[row["parallel_group_id"]],
                "generated_token_ids_private": row["generated_token_ids"],
                "target_decision_prefix_token_ids_private": row["generated_token_ids"][:step],
                "target_first_token_id_private": int(row["generated_token_ids"][step]),
                "post_target_token_id_private": int(row["generated_token_ids"][step + 1]),
                "prompt_token_ids_private": row["prompt_token_ids_private"],
                "role_positions_private": row["role_positions_private"],
                "candidate_layer": LAYERS[model],
                "wrong_depth_layer": LAYER_COUNTS[model] - 1 - LAYERS[model],
                "fixed_best_role": BEST_ROLES[model],
                "frozen_structure_roles": list(STRUCTURE_ROLES),
            }
        )
    write_jsonl(OUT / "protocol/private/phase392_frozen_intervention_cases.jsonl", frozen)
    summary = {
        "schema_version": "66.2.0",
        "phase_id": "Phase392-InterventionFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "candidate_group_count": len(grouped),
            "qualified_group_count": len(eligible),
            "instrument_group_count": 2,
            "causal_test_group_count": 24,
            "selected_case_count": len(frozen),
            "causal_direction_count": 24 * 2 * 3,
        },
        "selection": {
            "selected_groups": selected,
            "failed_groups_replaced_after_intervention": False,
            "instrument_and_causal_groups_disjoint": True,
        },
        "authorization": {
            "instrument_audit": True,
            "causal_test": False,
            "single_neuron_scan": False,
        },
    }
    write_json(OUT / "phase392_intervention_freeze.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
