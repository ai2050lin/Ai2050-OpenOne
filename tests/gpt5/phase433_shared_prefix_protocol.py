#!/usr/bin/env python3
"""Freeze the Phase433 shared-prefix multi-token event protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase433_shared_prefix"
PHASE_ID = "Phase433-SharedPrefixProtocol"
SCHEMA_VERSION = "phase433_shared_prefix.v1"
TRACE_SCHEMA_VERSION = "phase433_shared_prefix_trace.v1"

MODELS = ("qwen3", "glm4", "deepseek7b")
DTYPES = {"qwen3": "float16", "glm4": "bfloat16", "deepseek7b": "bfloat16"}
LANGUAGE_MODEL = "qwen3"
ROLES = ("a", "b")
MAIN_ROUTES = ("source_only", "query_only", "consistent")
STRESS_ROUTES = ("none", "conflict")
OPEN_SPLITS = ("observer_calibration", "physical_calibration", "behavior_holdout")
SEALED_SPLIT = "sealed_physical_holdout"
STRESS_SPLIT = "conflict_stress"
GROUPS_PER_BLOCK_SPLIT = 128

ACTIVE_TAG = {"a": "selector-alpha", "b": "selector-beta"}
NEUTRAL_TAG = "selector-neutral"

BLOCKS = (
    {
        "block_id": "language_action_shared_prefix_candidate",
        "family_id": "language_action",
        "mechanism_id": "shared_prefix_dual_route_lookup",
        "candidate": True,
        "matched_control_block_id": "language_action_shared_prefix_control",
    },
    {
        "block_id": "language_action_shared_prefix_control",
        "family_id": "language_action",
        "mechanism_id": "shared_prefix_stable_result_control",
        "candidate": False,
        "matched_control_block_id": "language_action_shared_prefix_candidate",
    },
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def route_tags(route_mode: str, role: str) -> tuple[str, str, str, str]:
    active = ACTIVE_TAG[role]
    opposite_role = "b" if role == "a" else "a"
    opposite = ACTIVE_TAG[opposite_role]
    if route_mode == "none":
        return NEUTRAL_TAG, NEUTRAL_TAG, "none", "none"
    if route_mode == "source_only":
        return active, NEUTRAL_TAG, role, "none"
    if route_mode == "query_only":
        return NEUTRAL_TAG, active, "none", role
    if route_mode == "consistent":
        return active, active, role, role
    if route_mode == "conflict":
        return active, opposite, role, opposite_role
    raise ValueError(route_mode)


def build_group(
    block: dict[str, Any], split: str, index: int, *, stress: bool = False
) -> dict[str, Any]:
    split_order = (*OPEN_SPLITS, SEALED_SPLIT, STRESS_SPLIT)
    split_index = split_order.index(split)
    serial = 433000 + split_index * 1000 + index
    shared_stem = f"unit-{serial:06d}-shared"
    first = f"{shared_stem}-A"
    second = f"{shared_stem}-B"
    role_swap = index % 2 == 1
    stable_swap = (index // 2) % 2 == 1
    source = f"record-{serial:06d}"
    if block["candidate"]:
        role_targets = {
            "a": second if role_swap else first,
            "b": first if role_swap else second,
        }
        stable_target = None
        decoy = None
        record = (
            f"Record: source {source}; role-A result {role_targets['a']}; "
            f"role-B result {role_targets['b']}."
        )
        task = (
            "Selector-alpha selects role-A result; selector-beta selects role-B "
            "result; selector-neutral carries no selector signal."
        )
    else:
        stable_target = second if stable_swap else first
        decoy = first if stable_target == second else second
        role_targets = {"a": stable_target, "b": stable_target}
        record = (
            f"Record: source {source}; stable result {stable_target}; decoy {decoy}."
        )
        task = (
            "Both selector-alpha and selector-beta select stable result; "
            "selector-neutral also leaves stable result unchanged. Never select decoy."
        )
    pair_id = f"phase433__{split}__pair_{index:03d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "pipeline_sealed": split == SEALED_SPLIT,
        "stress_only": stress,
        "contract_variant": "no_examples_shared_prefix",
        "group_index": index,
        "paired_group_id": pair_id,
        "semantic_group_id": f"{pair_id}__{block['block_id']}",
        "source": source,
        "shared_stem": shared_stem,
        "first_item": first,
        "second_item": second,
        "source_1": first,
        "source_2": second,
        "record": record,
        "task": task,
        "role_targets": role_targets,
        "stable_target": stable_target,
        "decoy": decoy,
        "role_mapping_variant": "swapped" if role_swap else "direct",
        "control_target_variant": "second" if stable_swap else "first",
        "history_text": "",
        "demonstration_cells": [],
    }


def build_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        for split in OPEN_SPLITS:
            open_rows.extend(
                build_group(block, split, index)
                for index in range(GROUPS_PER_BLOCK_SPLIT)
            )
        sealed_rows.extend(
            build_group(block, SEALED_SPLIT, index)
            for index in range(GROUPS_PER_BLOCK_SPLIT)
        )
        stress_rows.extend(
            build_group(block, STRESS_SPLIT, index, stress=True)
            for index in range(GROUPS_PER_BLOCK_SPLIT)
        )
    return open_rows, sealed_rows, stress_rows


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase433_shared_prefix_protocol.py",
        "phase433_shared_prefix_collect.py",
        "phase433_shared_prefix_analysis.py",
        "test_phase433_shared_prefix.py",
    )
    return {
        name: (
            hashlib.sha256((ROOT / "tests/gpt5" / name).read_bytes()).hexdigest()
            if (ROOT / "tests/gpt5" / name).exists()
            else None
        )
        for name in names
    }


def denominator_audit(
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = [*open_rows, *sealed_rows, *stress_rows]
    counts = Counter((row["block_id"], row["split"]) for row in all_rows)
    expected_open = len(BLOCKS) * len(OPEN_SPLITS) * GROUPS_PER_BLOCK_SPLIT
    expected_single = len(BLOCKS) * GROUPS_PER_BLOCK_SPLIT
    unique_vocab = len({row["shared_stem"] for row in all_rows}) == len(all_rows) // 2
    valid = bool(
        len(open_rows) == expected_open
        and len(sealed_rows) == expected_single
        and len(stress_rows) == expected_single
        and all(value == GROUPS_PER_BLOCK_SPLIT for value in counts.values())
        and len({row["semantic_group_id"] for row in all_rows}) == len(all_rows)
        and all(not row["pipeline_sealed"] for row in [*open_rows, *stress_rows])
        and all(row["pipeline_sealed"] for row in sealed_rows)
        and all(len(row["source_1"]) == len(row["source_2"]) for row in all_rows)
        and all(row["source_1"][:-1] == row["source_2"][:-1] for row in all_rows)
        and unique_vocab
    )
    main_per_split = len(BLOCKS) * GROUPS_PER_BLOCK_SPLIT * len(ROLES) * len(MAIN_ROUTES)
    stress_count = len(BLOCKS) * GROUPS_PER_BLOCK_SPLIT * len(ROLES) * len(STRESS_ROUTES)
    return {
        "valid": valid,
        "groups_per_block_split": GROUPS_PER_BLOCK_SPLIT,
        "open_group_count": len(open_rows),
        "sealed_group_count": len(sealed_rows),
        "stress_group_count": len(stress_rows),
        "conditions_per_open_split_per_model": main_per_split,
        "main_open_conditions_per_model": main_per_split * len(OPEN_SPLITS),
        "stress_open_conditions_per_model": stress_count,
        "total_open_conditions_per_model": main_per_split * len(OPEN_SPLITS) + stress_count,
        "sealed_conditions_qwen": main_per_split,
        "three_model_open_conditions": (
            main_per_split * len(OPEN_SPLITS) + stress_count
        ) * len(MODELS),
        "vocabulary_disjoint_across_splits": unique_vocab,
        "counts": {"::".join(key): value for key, value in sorted(counts.items())},
    }


def freeze() -> dict[str, Any]:
    open_rows, sealed_rows, stress_rows = build_groups()
    audit = denominator_audit(open_rows, sealed_rows, stress_rows)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "phase433_groups_open.jsonl", open_rows)
    write_jsonl(OUT / "phase433_groups_stress_open.jsonl", stress_rows)
    sealed_path = OUT / "sealed/phase433_groups_sealed.jsonl"
    write_jsonl(sealed_path, sealed_rows)
    commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": SEALED_SPLIT,
        "sealed_group_count": len(sealed_rows),
        "sealed_condition_count": audit["sealed_conditions_qwen"],
        "sealed_group_rows_sha256": digest_rows(sealed_rows),
        "read_requires_open_gate": True,
        "open_analysis_must_not_import_sealed_rows": True,
    }
    write_json(OUT / "phase433_sealed_commitment.json", commitment)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models_in_execution_order": list(MODELS),
        "execution_dtypes": DTYPES,
        "language_interpretation_model": LANGUAGE_MODEL,
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "stress_split": STRESS_SPLIT,
        "main_routes": list(MAIN_ROUTES),
        "stress_routes": list(STRESS_ROUTES),
        "denominator_audit": audit,
        "open_rows_sha256": digest_rows(open_rows),
        "stress_rows_sha256": digest_rows(stress_rows),
        "sealed_commitment": commitment,
        "event_contract": {
            "minimum_common_prefix_tokens_each_model": 2,
            "same_first_token_required": True,
            "same_surface_length_required": True,
            "first_divergence_token_observer_is_observer_only": True,
            "complete_sequence_behavior_is_primary": True,
            "prompt_terminal_is_pre_generation": True,
            "teacher_branch_boundary_is_not_prompt_prechoice": True,
            "natural_branch_boundary_requires_exact_generated_common_prefix": True,
        },
        "window_contract": {
            "calibration_split": "observer_calibration",
            "candidate_layers_qwen": [24, 25, 26, 27, 28, 29],
            "position_role": "prompt_terminal",
            "maximum_selected_windows": 3,
            "selection": "highest balanced accuracy; deterministic lower-layer tie break",
            "holdout_reselection_forbidden": True,
            "sealed_reselection_forbidden": True,
        },
        "physical_contract": {
            "positions": [
                "source_1_end",
                "source_2_end",
                "question_end",
                "instruction_start",
                "instruction_mid",
                "instruction_end",
                "assistant_boundary",
                "prompt_terminal",
                "teacher_branch_boundary",
            ],
            "component_layers_qwen": [24, 25, 26, 27, 28, 29],
            "component_split": "physical_calibration",
            "component_routes": ["consistent"],
            "component_receivers": ["prompt_terminal", "teacher_branch_boundary"],
            "no_head_channel_neuron_scan": True,
        },
        "numeric_gates": {
            "behavior_group_all_lcb_min": 0.95,
            "behavior_role_contract_lcb_min": 0.95,
            "token_contract_valid_fraction_min": 1.0,
            "registered_source_coverage_lcb_min": 0.95,
            "observer_per_class_lcb_min": 0.90,
            "candidate_role_flip_lcb_min": 0.95,
            "control_role_invariance_lcb_min": 0.95,
            "control_role_flip_ucb_max": 0.05,
            "natural_common_prefix_lcb_min": 0.95,
            "hidden_state_hook_max_abs_error": 0.001,
            "attention_replay_median_max": 0.002,
        },
        "gate_order": [
            "P0_main_behavior",
            "P1_coordinate_and_token_identity",
            "P2_fixed_window_holdout_event_prediction",
            "P3_candidate_specificity",
            "P4_component_conservation",
            "P5_sealed_replication",
            "P6_optional_aggregate_causal",
        ],
        "stress_does_not_block_main_observer": True,
        "causal_unlock_requires_sealed_pass": True,
        "implementation_hashes": implementation_hashes(),
    }
    write_json(OUT / "phase433_protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
