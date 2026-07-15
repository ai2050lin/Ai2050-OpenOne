#!/usr/bin/env python3
"""Freeze the Phase431 position-time atlas protocol and sealed denominator.

The protocol keeps language behaviour qualification separate from physical
prediction.  Open stages never load the sealed group file.  The selected-source
prediction target is balanced by construction and is not compared against an
oracle prompt-metadata baseline that already contains the selector answer.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase431_position_time"
PHASE_ID = "Phase431-PositionTimeProtocol"
SCHEMA_VERSION = "phase431_position_time.v1"
TRACE_SCHEMA_VERSION = "position_time_trace.v1"

MODELS = ("qwen3", "glm4", "deepseek7b")
LANGUAGE_MODEL = "qwen3"
INTERFACE = "direct_item"
CONTRACT_VARIANT = "no_examples"
ROLES = ("a", "b")
ROUTE_MODES = ("none", "source_only", "query_only", "consistent", "conflict")
SCORABLE_ROUTES = ("source_only", "query_only", "consistent")
GROUPS_PER_BLOCK_SPLIT = 96

OPEN_SPLITS = ("coordinate_calibration", "blind_discovery", "behavior_holdout")
SEALED_SPLIT = "sealed_physical_holdout"
ALL_SPLITS = (*OPEN_SPLITS, SEALED_SPLIT)

ACTIVE_TAG = {"a": "selector-alpha", "b": "selector-beta"}
NEUTRAL_TAG = "selector-neutral"

BLOCKS = (
    {
        "block_id": "language_action_dual_route_candidate",
        "family_id": "language_action",
        "mechanism_id": "dual_route_lookup",
        "candidate": True,
        "matched_control_block_id": "language_action_stable_result_control",
    },
    {
        "block_id": "language_action_stable_result_control",
        "family_id": "language_action",
        "mechanism_id": "stable_result_control",
        "candidate": False,
        "matched_control_block_id": "language_action_dual_route_candidate",
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


def build_group(block: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    split_index = ALL_SPLITS.index(split)
    serial = 431000 + split_index * 1000 + index
    suffix = f"{serial:06d}"
    first, second = f"X{suffix}", f"Y{suffix}"

    # Role-to-item mapping flips every group.  The control target follows a
    # separate two-group cycle, preventing a single parity field from serving
    # as the physical prediction target.
    role_swap = index % 2 == 1
    stable_swap = (index // 2) % 2 == 1
    source = f"source-Z{suffix}"
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

    paired_group_id = f"phase431__{split}__pair_{index:03d}"
    semantic_group_id = f"{paired_group_id}__{block['block_id']}"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "pipeline_sealed": split == SEALED_SPLIT,
        "contract_variant": CONTRACT_VARIANT,
        "group_index": index,
        "paired_group_id": paired_group_id,
        "semantic_group_id": semantic_group_id,
        "source": source,
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


def build_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        for split in ALL_SPLITS:
            target = sealed_rows if split == SEALED_SPLIT else open_rows
            target.extend(
                build_group(block, split, index)
                for index in range(GROUPS_PER_BLOCK_SPLIT)
            )
    return open_rows, sealed_rows


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase431_position_time_protocol.py",
        "phase431_position_time_collect.py",
        "phase431_position_time_analysis.py",
    )
    output: dict[str, str | None] = {}
    for name in names:
        path = ROOT / "tests/gpt5" / name
        output[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return output


def validate_groups(
    open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    all_rows = [*open_rows, *sealed_rows]
    counts = Counter((row["block_id"], row["split"]) for row in all_rows)
    expected_open = len(BLOCKS) * len(OPEN_SPLITS) * GROUPS_PER_BLOCK_SPLIT
    expected_sealed = len(BLOCKS) * GROUPS_PER_BLOCK_SPLIT
    valid = bool(
        len(open_rows) == expected_open
        and len(sealed_rows) == expected_sealed
        and len({row["semantic_group_id"] for row in all_rows}) == len(all_rows)
        and all(count == GROUPS_PER_BLOCK_SPLIT for count in counts.values())
        and all(not row["pipeline_sealed"] for row in open_rows)
        and all(row["pipeline_sealed"] for row in sealed_rows)
        and all(
            set(row["role_targets"].values()).issubset(
                {row["first_item"], row["second_item"]}
            )
            for row in all_rows
        )
    )
    return {
        "valid": valid,
        "open_group_count": len(open_rows),
        "sealed_group_count": len(sealed_rows),
        "open_condition_count": len(open_rows) * len(ROLES) * len(ROUTE_MODES),
        "sealed_condition_count": len(sealed_rows) * len(ROLES) * len(ROUTE_MODES),
        "groups_per_block_split": GROUPS_PER_BLOCK_SPLIT,
        "counts": {"::".join(key): value for key, value in sorted(counts.items())},
    }


def freeze() -> dict[str, Any]:
    open_rows, sealed_rows = build_groups()
    audit = validate_groups(open_rows, sealed_rows)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))

    write_jsonl(OUT / "phase431_groups_open.jsonl", open_rows)
    sealed_path = OUT / "sealed" / "phase431_groups_sealed.jsonl"
    write_jsonl(sealed_path, sealed_rows)
    sealed_commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": SEALED_SPLIT,
        "sealed_group_count": len(sealed_rows),
        "sealed_condition_count": len(sealed_rows) * len(ROLES) * len(ROUTE_MODES),
        "sealed_group_rows_sha256": digest_rows(sealed_rows),
        "open_pipeline_must_not_import_sealed_file": True,
        "read_requires_open_gates": ["G0", "G1", "G2", "G3", "G4", "G5"],
    }
    write_json(OUT / "phase431_sealed_commitment.json", sealed_commitment)

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models": list(MODELS),
        "execution_dtypes": {
            "qwen3": "float16",
            "glm4": "bfloat16",
            "deepseek7b": "bfloat16",
        },
        "language_model": LANGUAGE_MODEL,
        "language_candidate": {
            "block_id": BLOCKS[0]["block_id"],
            "contract_variant": CONTRACT_VARIANT,
            "matched_control_block_id": BLOCKS[0]["matched_control_block_id"],
            "model_specific": True,
            "cross_model": False,
        },
        "instrument_only_models": [model for model in MODELS if model != LANGUAGE_MODEL],
        "interface": INTERFACE,
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "roles": list(ROLES),
        "routes": list(ROUTE_MODES),
        "scorable_routes": list(SCORABLE_ROUTES),
        "denominator_audit": audit,
        "open_group_rows_sha256": digest_rows(open_rows),
        "sealed_commitment": sealed_commitment,
        "prediction_target": {
            "coverage_axis": "registered_source versus other",
            "choice_axis": "source_1 versus source_2 conditional on registered source",
            "other_is_never_deleted": True,
            "source_1_source_2_balanced_on_scorable_candidate_conditions": True,
            "behavior_qualification_is_not_prediction_target": True,
        },
        "baseline_contract": {
            "primary": "balanced 50/50 source-choice prior on eligible conditions",
            "nuisance_only": ["prompt_length", "source_token_length", "absolute_positions"],
            "forbidden_oracle_fields": [
                "role",
                "active_selector_identity",
                "role_mapping_variant",
                "normative_target",
                "actual_choice",
            ],
            "reason": "selector-bearing metadata deterministically contains the answer and would recreate a perfect, non-improvable baseline",
        },
        "upstream_predictor_contract": {
            "window_selection_reads_behavior_labels": False,
            "per_window_prediction": "sign of source_1 minus source_2 residual logit margin",
            "aggregate": "majority vote across all blind-frozen windows",
            "tie_break": "prediction from the latest-layer frozen window",
            "holdout_window_reselection": False,
        },
        "position_contract": {
            "prompt_roles": [
                "source_1_end",
                "source_2_end",
                "before_selector_end",
                "after_selector_end",
                "question_end",
                "instruction_end",
                "assistant_boundary",
                "prompt_terminal",
            ],
            "generation_steps": [0, 1, 2, 3, 4],
            "generation_step_zero": "prompt terminal state predicting first natural token",
            "overlapping_roles_share_one_physical_token": True,
        },
        "record_contract": {
            "all_layers_compact_ledger": True,
            "receivers": ["question_end", "instruction_end", "prompt_terminal", "g1", "g2", "g3", "g4"],
            "disjoint_source_partition": [
                "source_1",
                "source_2",
                "before_selector",
                "after_selector",
                "question",
                "instruction",
                "other_positions",
            ],
            "registered_sources_require_other_remainder": True,
            "random_projection": {
                "seeds": [43101, 43102, 43103],
                "dimensions_per_seed": 8,
                "position": "prompt_terminal",
                "observer_only": True,
            },
            "full_vector_windows_max": 4,
            "head_channel_neuron_scan": False,
            "intervention": False,
        },
        "open_gates": {
            "G0": "fresh Qwen3 candidate behavior and matched-control specificity",
            "G1": "native terminal, cache/no-cache and batch/single identity",
            "G2": "residual and full attention-source partition reconstruction",
            "G3": "registered-source coverage and conditional source-choice terminal identity",
            "G4": "blind-frozen upstream event predicts heldout source choice above balanced prior",
            "G5": "candidate role-selective event exceeds matched control and swaps",
        },
        "numeric_gates": {
            "identity_top1_exact": True,
            "behavior_group_all_lcb_min": 0.70,
            "reconstruction_relative_error_median_max": 0.01,
            "registered_source_coverage_lcb_min": 0.90,
            "terminal_choice_accuracy_lcb_min": 0.90,
            "upstream_balanced_accuracy_lcb_min": 0.55,
            "candidate_role_flip_lcb_min": 0.60,
            "control_role_flip_ucb_max": 0.40,
        },
        "evidence_contract": {
            "physical": True,
            "observer": True,
            "predictive_only_after_independent_holdout": True,
            "causal": False,
            "single_neuron": False,
            "strict_double_blind": False,
        },
        "stop_rules": [
            "behavior failure closes candidate before large physical collection",
            "language-model identity failure blocks its physical collection",
            "instrument-only model identity failure blocks the strict three-model gate and sealed unlock but does not rewrite Qwen3 observations",
            "ledger failure blocks prediction interpretation",
            "holdout failure cannot change layers positions features or thresholds",
            "open gate failure keeps sealed file unread",
            "no head channel neuron or causal test in Phase431",
        ],
        "implementation_sha256": implementation_hashes(),
    }
    write_json(OUT / "phase431_protocol.json", protocol)
    return protocol


def main() -> None:
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
