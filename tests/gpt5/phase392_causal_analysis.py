#!/usr/bin/env python3
"""Analyze the frozen 24-group Phase392 parent-boundary causal denominator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    model_rows = []
    all_directions = []
    for model in MODELS:
        root = OUT / "collection/causal_test" / model
        complete = read_json(root / "complete.json")
        rows = read_jsonl(root / "direction_rows.jsonl")
        if not complete["valid"] or len(rows) != 48:
            raise RuntimeError(f"Invalid Phase392 causal denominator for {model}")
        all_directions.extend(rows)
        scenarios = [
            {item["scenario"]: item for item in row["scenario_rows"]}
            for row in rows
        ]
        joint = [items["donor_semantic_joint"]["normalized_margin_mediation"] for items in scenarios]
        fixed = [items["donor_fixed_best_role"]["normalized_margin_mediation"] for items in scenarios]
        attributes = [items["donor_attributes_only"]["normalized_margin_mediation"] for items in scenarios]
        structure = [items["donor_frozen_structure_roles"]["normalized_margin_mediation"] for items in scenarios]
        random = [items["donor_same_count_random_parent_positions"]["normalized_margin_mediation"] for items in scenarios]
        wrong = [items["donor_semantic_joint_wrong_depth"]["normalized_margin_mediation"] for items in scenarios]
        metrics = {
            "median_joint_normalized_margin_mediation": median(joint),
            "median_fixed_role_mediation": median(fixed),
            "median_attributes_only_mediation": median(attributes),
            "median_structure_only_mediation": median(structure),
            "median_random_position_mediation": median(random),
            "median_wrong_depth_mediation": median(wrong),
            "median_joint_advantage_over_fixed_role": median([a - b for a, b in zip(joint, fixed)]),
            "median_joint_advantage_over_attributes_only": median([a - b for a, b in zip(joint, attributes)]),
            "median_joint_advantage_over_random_positions": median([a - b for a, b in zip(joint, random)]),
            "median_joint_advantage_over_wrong_depth": median([a - b for a, b in zip(joint, wrong)]),
            "positive_joint_direction_rate": rate([value > 0 for value in joint]),
            "strict_answer_switch_rate": rate([row["joint_generation"]["strict_donor_target_switch"] for row in rows]),
            "median_query_projection_toward_donor": median(
                items["donor_semantic_joint"]["query_projection_toward_donor"]
                for items in scenarios
            ),
        }
        participation_gate = (
            metrics["median_joint_normalized_margin_mediation"] >= 0.10
            and metrics["median_joint_advantage_over_fixed_role"] >= 0.05
            and metrics["median_joint_advantage_over_attributes_only"] >= 0.05
            and metrics["median_joint_advantage_over_random_positions"] >= 0.05
            and metrics["median_joint_advantage_over_wrong_depth"] >= 0.05
            and metrics["positive_joint_direction_rate"] >= 0.75
        )
        function_gate = participation_gate and metrics["strict_answer_switch_rate"] >= 0.50
        model_rows.append(
            {
                "schema_version": "66.5.0",
                "phase_id": "Phase392-CausalAnalysis",
                "model": model,
                "direction_count": len(rows),
                **metrics,
                "parent_boundary_participation_gate_pass": participation_gate,
                "language_function_path_gate_pass": function_gate,
            }
        )
    shared_participation = all(row["parent_boundary_participation_gate_pass"] for row in model_rows)
    shared_function = all(row["language_function_path_gate_pass"] for row in model_rows)
    summary = {
        "schema_version": "66.5.0",
        "phase_id": "Phase392-CausalAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "group_count": 24,
            "direction_count": len(all_directions),
            "scenario_count": len(all_directions) * 8,
            "joint_generation_count": len(all_directions),
            "failed_group_replacement_count": 0,
        },
        "models": model_rows,
        "results": {
            "models_passing_parent_boundary_participation": sum(
                row["parent_boundary_participation_gate_pass"] for row in model_rows
            ),
            "models_passing_language_function_path": sum(
                row["language_function_path_gate_pass"] for row in model_rows
            ),
            "crossmodel_parent_boundary_participation_established": shared_participation,
            "crossmodel_language_function_path_established": shared_function,
            "strict_answer_switch_count": sum(
                row["joint_generation"]["strict_donor_target_switch"]
                for row in all_directions
            ),
            "single_neuron_causal_path_count": 0,
            "language_encoding_closed": False,
        },
        "authorization": {
            "promote_parent_boundary_causal_edge": shared_participation,
            "promote_complete_language_path": shared_function,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "internal_state_shift_without_specific_controls_is_path": False,
            "one_model_pass_is_crossmodel_path": False,
            "language_encoding_closed": False,
        },
    }
    (OUT / "phase392_causal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
