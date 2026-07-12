#!/usr/bin/env python3
"""Freeze the finite Phase380 causal layout scan before model execution."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("early", "middle_early", "middle", "middle_late", "late")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLES = ("source", "query", "current")
CONDITIONS = ("natural_swap", "equal_energy_permutation")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    authorization = read_json(OUT / "phase380_causal_authorization.json")
    if not authorization["authorization"]["run_registered_natural_boundary_causal_scan"]:
        raise RuntimeError("Residual validation did not authorize a causal scan")
    stable = authorization["stable_objects"]
    cases = read_jsonl(OUT / "private/phase380_qualified_trace_cases.jsonl")
    metadata = {}
    for model in MODELS:
        metadata.update(
            {
                row["blind_case_id"]: row
                for row in read_jsonl(OUT / "trace/models" / model / "phase380_trace_rows.jsonl")
            }
        )
    parallel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        parallel[case["anonymous_parallel_group_id"]].append(case)
    eligible_by_mechanism: dict[str, list[str]] = defaultdict(list)
    for group, rows in sorted(parallel.items()):
        if len(rows) == 12 and all(
            metadata[row["blind_case_id"]]["baseline_replay_matches_observed_target_token"]
            for row in rows
        ):
            eligible_by_mechanism[rows[0]["mechanism_id"]].append(group)
    selected = {
        mechanism: sorted(groups)[:8]
        for mechanism, groups in eligible_by_mechanism.items()
        if any(row["mechanism_id"] == mechanism for row in stable)
    }
    if any(len(groups) != 8 for groups in selected.values()):
        raise RuntimeError("Every selected mechanism must contribute exactly eight groups")
    transfer_pairs = {
        "content_change_same_operation": ["A_to_C", "C_to_A", "B_to_D", "D_to_B"],
        "joint_content_operation_change": ["A_to_D", "D_to_A", "B_to_C", "C_to_B"],
    }
    object_count = len(stable)
    group_object_count = sum(len(selected[row["mechanism_id"]]) for row in stable)
    rows_per_model = (
        group_object_count
        * 4
        * len(DEPTHS)
        * len(COMPONENTS)
        * len(ROLES)
        * len(CONDITIONS)
    )
    freeze = {
        "schema_version": "53.8.0",
        "phase_id": "Phase380-CausalScanFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stable_objects": stable,
        "selected_parallel_groups": selected,
        "transfer_pairs_by_axis": transfer_pairs,
        "scan_grid": {
            "relative_depths": list(DEPTHS),
            "relative_depth_fractions": [0.10, 0.30, 0.50, 0.70, 0.90],
            "component_boundaries": list(COMPONENTS),
            "position_roles": list(ROLES),
            "conditions": list(CONDITIONS),
            "top_k_selection": False,
        },
        "frozen_gates": {
            "minimum_natural_transfer_gain": 0.10,
            "minimum_gain_over_equal_energy": 0.05,
            "minimum_terminal_transfer_share": 0.02,
            "minimum_share_over_equal_energy": 0.01,
            "minimum_gain_over_cyclic_wrong_depth": 0.05,
            "minimum_gain_over_cyclic_wrong_role": 0.05,
            "minimum_groups_all_four_directions": 6,
        },
        "denominator": {
            "model_count": 3,
            "stable_object_count": object_count,
            "selected_group_object_count": group_object_count,
            "selected_unique_group_count": len(
                {group for groups in selected.values() for group in groups}
            ),
            "condition_rows_per_model": rows_per_model,
            "condition_rows_total": rows_per_model * len(MODELS),
        },
        "claim_boundary": {
            "same_normalized_cell_across_models_is_same_neuron": False,
            "single_passing_cell_is_a_complete_language_path": False,
            "late_current_layer_output_is_upstream_rule": False,
            "physical_holdout_opened": False,
            "single_neuron_scan": False,
        },
        "execution_order": list(MODELS),
    }
    write_json(OUT / "phase380_causal_scan_freeze.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
