#!/usr/bin/env python3
"""Freeze a finite parent-boundary causal test for Phase398 ROQ paths."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
SOURCE = OUT / "query_trace/protocol/private/phase398_physical_holdout_query_trace_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SCENARIOS = (
    "no_intervention",
    "identity_candidate_parent",
    "same_order_joint_donor_candidate",
    "wrong_order_joint_donor_candidate",
    "same_order_joint_donor_wrong_depth",
    "same_order_donor_answer_anchor_control",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    physical = read_json(OUT / "phase398_order_conditioned_physical_validation.json")
    if physical["results"]["passing_model_surface_cell_count"] != 9:
        raise RuntimeError("Phase398 physical ROQ validation did not pass")
    freeze = read_json(OUT / "phase398_order_conditioned_candidate_freeze.json")
    layer_by_cell = {(row["model"], row["task_surface"]): row["candidate_layer"] for row in freeze["frozen_cells"]}
    rows = read_jsonl(SOURCE)
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row["private_execution_model"], row["task_surface_private"])].add(row["anonymous_parallel_group_id"])
    instrument_groups = {
        cell: min(groups, key=lambda group: hashlib.sha256(f"phase398-causal-instrument:{group}".encode()).hexdigest())
        for cell, groups in grouped.items()
    }
    frozen = []
    for row in rows:
        model = row["private_execution_model"]
        surface = row["task_surface_private"]
        candidate = int(layer_by_cell[(model, surface)])
        parent = candidate + 1
        wrong = max(1, parent // 2)
        if wrong == parent:
            wrong = max(1, parent - 2)
        frozen.append({
            **row,
            "schema_version": "72.10.0",
            "phase_id": "Phase398-OrderConditionedCausalFrozen",
            "causal_split": "instrument" if row["anonymous_parallel_group_id"] == instrument_groups[(model, surface)] else "causal_test",
            "candidate_output_layer_private": candidate,
            "candidate_parent_layer_private": parent,
            "wrong_depth_parent_layer_private": wrong,
        })
    private = OUT / "causal/protocol/private"
    write_jsonl(private / "phase398_causal_cases.jsonl", frozen)
    for split in ("instrument", "causal_test"):
        write_jsonl(private / f"phase398_{split}_causal_cases.jsonl", [row for row in frozen if row["causal_split"] == split])
    protocol = {
        "schema_version": "72.10.0",
        "phase_id": "Phase398-OrderConditionedCausalProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_the_frozen_order_conditioned_query_end_state_is_causally_sufficient_and_order_specific_at_the_next_layer_parent_boundary",
        "denominator": {
            "instrument_group_count": 9,
            "causal_test_group_count": 27,
            "conditions_per_group": 16,
            "directions_per_group": 16,
            "scenario_count_per_direction": len(SCENARIOS),
            "instrument_generation_count": 9 * 16 * len(SCENARIOS),
            "causal_test_generation_count": 27 * 16 * len(SCENARIOS),
        },
        "direction_contract": "for_each_axis_relation_order_and_both_query_directions_copy_the_opposite_query_state_within_the_same_order;_wrong_order_donor_keeps_the_donor_target_fixed",
        "patch_contract": {
            "candidate_state": "candidate_layer_output_equivalently_next_layer_input",
            "recipient_position": "query_end_only",
            "same_order_donor": "same_axis_relation_order_opposite_query",
            "wrong_order_donor": "same_axis_relation_opposite_order_opposite_query_with_same_target",
            "wrong_depth": "same_order_donor_at_frozen_shallower_parent_layer",
            "answer_anchor_control": "same_order_donor_answer_anchor_to_recipient_answer_anchor_at_candidate_parent",
        },
        "scenarios": list(SCENARIOS),
        "frozen_gates": {
            "baseline_recipient_correct_rate": 1.0,
            "identity_recipient_correct_rate": 1.0,
            "minimum_same_order_donor_answer_switch_rate": 0.5,
            "minimum_same_order_advantage_over_wrong_order_switch_rate": 0.1,
            "minimum_same_order_advantage_over_wrong_depth_switch_rate": 0.1,
            "all_nine_model_surface_cells_required": True,
        },
        "claim_boundary": {
            "state_swap_sufficiency_is_natural_necessity": False,
            "query_end_state_is_single_neuron": False,
            "failure_proves_no_order_conditioned_computation": False,
        },
        "authorization": {
            "run_instrument": True,
            "run_causal_test": False,
            "single_neuron_scan": False,
        },
    }
    (OUT / "phase398_order_conditioned_causal_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
