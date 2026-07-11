#!/usr/bin/env python3
"""Freeze Phase343-qualified heldout cases for copy-block causal auditing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase344"
SCHEMA_VERSION = "20.0.0"
ROUND_DEFAULT = "copy_block_heldout_boundary"
OUT = ROOT / "tests/gpt5/result/phase344_copy_block_boundary"
PHASE343 = ROOT / "tests/gpt5/result/phase343_copy_boundary_protocol/copy_boundary_protocol_qualification"
SELECTED_TASKS = (
    "random_label_copy", "digit_copy", "arbitrary_symbol_relay",
    "cross_sentence_pointer", "multi_token_phrase_copy", "delayed_copy",
    "key_value_read", "object_name_relay", "field_extraction",
    "material_relation_binding", "singular_agreement", "direct_entailment",
    "token_transformation",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    source = read_jsonl(PHASE343 / "phase343_registered_cases.jsonl")
    rows = []
    for row in source:
        if row["mechanism_id"] not in SELECTED_TASKS or row["split"] not in {"heldout", "private_heldout"}:
            continue
        rows.append({
            **row, "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "created_at": now(), "phase343_case_id": row["case_id"],
            "case_id": row["case_id"].replace("phase343_", "phase344_", 1),
            "baseline_only": False, "internal_intervention_allowed": True,
            "block_reselection_allowed": False, "layer_shrink_allowed": False,
            "single_unit_intervention_allowed": False,
        })
    if len(rows) != 585 or len({row["case_id"] for row in rows}) != 585:
        raise RuntimeError(f"Invalid Phase344 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Test the Phase338 frozen blocks across qualified copy and noncopy heldouts.",
        "registered_case_count": len(rows), "selected_tasks": list(SELECTED_TASKS),
        "splits": ["heldout", "private_heldout"], "execution_mode": "b1_left_cache0",
        "phrase_conditions": [
            "baseline", "correct_zero", "correct_half", "correct_permutation",
            "wrong_depth_zero", "wrong_position_zero",
        ],
        "rollout_conditions": ["baseline", "correct_zero", "wrong_depth_zero", "wrong_position_zero"],
        "thresholds": {
            "baseline_capability_rate_min": 0.8,
            "required_phrase_valid_rate_min": 1.0,
            "correct_behavior_loss_rate_min": 0.5,
            "wrong_control_behavior_loss_rate_max": 0.1,
            "phrase_control_superiority_min": 0.05,
            "glm4_explicit_copy_task_pass_min": 4,
            "glm4_noncopy_task_pass_max": 0,
        },
        "lexical_generalization_required_tasks": [
            "random_label_copy", "digit_copy", "arbitrary_symbol_relay", "multi_token_phrase_copy"
        ],
        "claim_boundaries": [
            "Phase338 blocks remain frozen and cannot be reselected.",
            "All phrase scoring and behavior generation use batch size one.",
            "Copy-neighbor effects do not count as noncopy side effects.",
            "No natural-state replacement, layer shrink, or neuron scan is allowed in Phase344.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "task_count": len(SELECTED_TASKS),
        "model_case_count": {
            model: sum(row["model"] == model for row in rows)
            for model in ("qwen3", "glm4", "deepseek7b")
        },
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("heldout", "private_heldout")
        },
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase344_registered_cases.jsonl", rows)
    write_json(root / "phase344_registered_protocol.json", protocol)
    write_json(root / "phase344_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
