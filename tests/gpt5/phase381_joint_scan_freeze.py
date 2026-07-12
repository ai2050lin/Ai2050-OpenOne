#!/usr/bin/env python3
"""Freeze replay-qualified groups and the Phase381 joint-state scan grid."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase381_joint_state_case_bank import read_jsonl, write_json


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
P380 = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = read_json(OUT / "phase381_protocol.json")
    selected = read_jsonl(OUT / "phase381_selected_blind_groups.jsonl")
    mechanism_by_group = {
        row["anonymous_parallel_group_id"]: row["mechanism_id"] for row in selected
    }
    trace_rows: list[dict[str, Any]] = []
    complete = []
    for model in MODELS:
        trace_rows.extend(
            read_jsonl(OUT / "trace/models" / model / "phase381_trace_rows.jsonl")
        )
        complete.append(read_json(OUT / "trace/models" / model / "complete.json"))
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_group[row["anonymous_parallel_group_id"]].append(row)
    replay_qualified = {
        group
        for group, rows in by_group.items()
        if len(rows) == 12
        and all(row["baseline_replay_matches_observed_target_token"] for row in rows)
    }
    groups_by_mechanism = {
        mechanism: sorted(
            group
            for group in replay_qualified
            if mechanism_by_group[group] == mechanism
        )
        for mechanism in ("relation_binding", "entity_recency", "target_vs_wrong")
    }
    if any(len(groups) < 6 for groups in groups_by_mechanism.values()):
        raise RuntimeError(f"Insufficient replay-qualified groups: {groups_by_mechanism}")
    stable_objects = [
        row
        for row in read_json(P380 / "phase380_residual_validation_summary.json")[
            "results"
        ]["stable_objects"]
        if row["mechanism_id"] in groups_by_mechanism
    ]
    object_group_count = sum(
        len(groups_by_mechanism[row["mechanism_id"]]) for row in stable_objects
    )
    transfer_task_count = object_group_count * 4
    role_sets = {
        "source": ["source"],
        "query": ["query"],
        "current": ["current"],
        "source_query_current": ["source", "query", "current"],
    }
    rows_per_task = 5 * 4 * len(role_sets) * 2
    condition_rows_per_model = transfer_task_count * rows_per_task
    freeze = {
        "schema_version": "54.4.0",
        "phase_id": "Phase381-JointScanFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stable_objects": stable_objects,
        "selected_replay_qualified_groups": groups_by_mechanism,
        "replay_mismatch_groups": sorted(set(by_group) - replay_qualified),
        "transfer_pairs_by_axis": {
            "content_change_same_operation": ["A_to_C", "C_to_A", "B_to_D", "D_to_B"],
            "joint_content_operation_change": ["A_to_D", "D_to_A", "B_to_C", "C_to_B"],
        },
        "scan_grid": {
            "relative_depths": ["early", "middle_early", "middle", "middle_late", "late"],
            "relative_depth_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
            "component_boundaries": ["layer_input", "attention_output", "mlp_output", "layer_output"],
            "role_sets": role_sets,
            "conditions": ["natural_swap", "equal_energy_permutation"],
            "top_k_selection": False,
        },
        "frozen_joint_gates": protocol["frozen_joint_gates"],
        "denominator": {
            "model_count": 3,
            "trace_case_count": len(trace_rows),
            "replay_match_case_count": sum(
                row["baseline_replay_matches_observed_target_token"] for row in trace_rows
            ),
            "replay_qualified_group_count": len(replay_qualified),
            "replay_groups_by_mechanism": {
                mechanism: len(groups) for mechanism, groups in groups_by_mechanism.items()
            },
            "stable_object_count": len(stable_objects),
            "selected_group_object_count": object_group_count,
            "transfer_task_count_per_model": transfer_task_count,
            "condition_rows_per_model": condition_rows_per_model,
            "condition_rows_total": condition_rows_per_model * 3,
        },
        "quality": {
            "all_trace_files_valid": all(row["valid"] for row in complete),
            "mismatch_groups_retained_in_quality_ledger": True,
            "mismatch_groups_used_for_causal_claims": False,
            "threshold_retuned": False,
            "single_neuron_scan": False,
        },
        "authorization": {
            "run_joint_scan_sequentially": all(row["valid"] for row in complete),
            "open_single_neuron_scan": False,
            "claim_complete_path_before_analysis": False,
        },
    }
    write_json(OUT / "phase381_joint_scan_freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
