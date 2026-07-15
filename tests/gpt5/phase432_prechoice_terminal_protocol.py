#!/usr/bin/env python3
"""Freeze the Phase432 independent pre-choice terminal observer protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase432_prechoice_terminal"
PHASE_ID = "Phase432-PrechoiceTerminalProtocol"
SCHEMA_VERSION = "phase432_prechoice_terminal.v1"
TRACE_SCHEMA_VERSION = "phase432_prechoice_terminal_trace.v1"

MODELS = ("qwen3", "glm4", "deepseek7b")
DTYPES = {"qwen3": "float16", "glm4": "bfloat16", "deepseek7b": "bfloat16"}
LANGUAGE_MODEL = "qwen3"
OPEN_SPLIT = "independent_confirmation"
SEALED_SPLIT = "sealed_replication"
GROUPS_PER_BLOCK_SPLIT = 128
ROLES = ("a", "b")
ROUTE_MODES = ("none", "source_only", "query_only", "consistent", "conflict")
SCORABLE_ROUTES = ("source_only", "query_only", "consistent")
PRIMARY_WINDOW = {
    "model": "qwen3",
    "layer": 26,
    "position_role": "prompt_terminal",
    "relative_depth": 26 / 35,
    "selected_from": "Phase431 posthoc descriptive failure map",
    "labels_used_before_phase432_freeze": True,
}

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


def build_group(block: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    split_offset = 0 if split == OPEN_SPLIT else 1000
    serial = 432000 + split_offset + index
    suffix = f"{serial:06d}"
    first, second = f"X{suffix}", f"Y{suffix}"
    role_swap = index % 2 == 1
    stable_swap = (index // 2) % 2 == 1
    source = f"source-W{suffix}"
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
    paired_group_id = f"phase432__{split}__pair_{index:03d}"
    semantic_group_id = f"{paired_group_id}__{block['block_id']}"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        **block,
        "split": split,
        "pipeline_sealed": split == SEALED_SPLIT,
        "contract_variant": "no_examples",
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
        for split, target in ((OPEN_SPLIT, open_rows), (SEALED_SPLIT, sealed_rows)):
            target.extend(
                build_group(block, split, index)
                for index in range(GROUPS_PER_BLOCK_SPLIT)
            )
    return open_rows, sealed_rows


def implementation_hashes() -> dict[str, str | None]:
    names = (
        "phase432_prechoice_terminal_protocol.py",
        "phase432_prechoice_terminal_collect.py",
        "phase432_prechoice_terminal_analysis.py",
        "test_phase432_prechoice_terminal.py",
    )
    return {
        name: hashlib.sha256((ROOT / "tests/gpt5" / name).read_bytes()).hexdigest()
        if (ROOT / "tests/gpt5" / name).exists()
        else None
        for name in names
    }


def denominator_audit(
    open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = [*open_rows, *sealed_rows]
    counts = Counter((row["block_id"], row["split"]) for row in rows)
    valid = bool(
        len(open_rows) == 2 * GROUPS_PER_BLOCK_SPLIT
        and len(sealed_rows) == 2 * GROUPS_PER_BLOCK_SPLIT
        and len({row["semantic_group_id"] for row in rows}) == len(rows)
        and all(value == GROUPS_PER_BLOCK_SPLIT for value in counts.values())
        and set(open_rows[0]) == set(sealed_rows[0])
        and all(not row["pipeline_sealed"] for row in open_rows)
        and all(row["pipeline_sealed"] for row in sealed_rows)
    )
    conditions_per_split = len(BLOCKS) * GROUPS_PER_BLOCK_SPLIT * len(ROLES) * len(ROUTE_MODES)
    return {
        "valid": valid,
        "groups_per_block_split": GROUPS_PER_BLOCK_SPLIT,
        "open_group_count": len(open_rows),
        "sealed_group_count": len(sealed_rows),
        "conditions_per_model_open": conditions_per_split,
        "conditions_per_model_sealed": conditions_per_split,
        "three_model_open_condition_count": conditions_per_split * len(MODELS),
        "counts": {"::".join(key): value for key, value in sorted(counts.items())},
    }


def freeze() -> dict[str, Any]:
    open_rows, sealed_rows = build_groups()
    audit = denominator_audit(open_rows, sealed_rows)
    if not audit["valid"]:
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    write_jsonl(OUT / "phase432_groups_open.jsonl", open_rows)
    sealed_path = OUT / "sealed/phase432_groups_sealed.jsonl"
    write_jsonl(sealed_path, sealed_rows)
    sealed_commitment = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_split": SEALED_SPLIT,
        "sealed_group_rows_sha256": digest_rows(sealed_rows),
        "sealed_group_count": len(sealed_rows),
        "sealed_condition_count": audit["conditions_per_model_sealed"],
        "read_requires_open_confirmation_pass": True,
    }
    write_json(OUT / "phase432_sealed_commitment.json", sealed_commitment)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "models": list(MODELS),
        "execution_dtypes": DTYPES,
        "language_model": LANGUAGE_MODEL,
        "open_split": OPEN_SPLIT,
        "sealed_split": SEALED_SPLIT,
        "denominator_audit": audit,
        "open_group_rows_sha256": digest_rows(open_rows),
        "sealed_commitment": sealed_commitment,
        "primary_window": PRIMARY_WINDOW,
        "observer_contract": {
            "prediction": "sign(final_norm(residual_post) dot (unembed(source_1)-unembed(source_2)))",
            "target": "first registered source selected by natural generation",
            "positions_before_first_answer_only": True,
            "generation_positions_forbidden": True,
            "behavior_labels_used_for_phase432_window_selection": False,
            "window_reselection": False,
            "qwen_primary_only": True,
            "glm4_deepseek7b_scope": "same-protocol diagnostic, not a language mechanism claim",
        },
        "shape_map_contract": {
            "positions": ["question_end", "instruction_end", "prompt_terminal"],
            "layers": "all model layers",
            "labels_used": True,
            "evidence": "descriptive observer",
            "can_select_new_primary_window": False,
        },
        "numeric_gates": {
            "behavior_group_all_lcb_min": 0.90,
            "registered_source_coverage_lcb_min": 0.95,
            "primary_choice_per_class_lcb_min": 0.95,
            "candidate_role_flip_lcb_min": 0.95,
            "control_role_invariance_lcb_min": 0.95,
            "control_role_flip_ucb_max": 0.05,
            "terminal_native_top1_identity_required": True,
            "hidden_state_hook_max_abs_error": 1e-4,
        },
        "open_gate": [
            "Qwen3 behavior qualification",
            "native terminal and hidden-state coordinate identity",
            "fixed L26 prompt-terminal choice prediction",
            "candidate role flip with matched-control invariance",
        ],
        "sealed_policy": {
            "only_qwen3": True,
            "no_read_before_open_gate": True,
            "same_fixed_window_and_thresholds": True,
            "failure_cannot_reselect": True,
        },
        "evidence_contract": {
            "physical": True,
            "observer": True,
            "predictive_after_open_and_sealed": True,
            "causal": False,
            "single_neuron": False,
            "mechanism_closure": False,
        },
        "stop_rules": [
            "open failure keeps sealed data unread",
            "instrument model behavior failure limits only that model's interpretation",
            "no window search or threshold change after open results",
            "no causal, head, channel, or neuron intervention in Phase432",
        ],
        "implementation_sha256": implementation_hashes(),
    }
    write_json(OUT / "phase432_protocol.json", protocol)
    return protocol


def main() -> None:
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
