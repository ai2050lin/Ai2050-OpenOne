#!/usr/bin/env python3
"""Analyze frozen Phase398 order-conditioned causal generations."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
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


def rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 9) if values else 0.0


def main() -> None:
    protocol = read_json(OUT / "phase398_order_conditioned_causal_protocol.json")
    instrument = read_json(OUT / "phase398_order_conditioned_causal_instrument_audit.json")
    if not instrument["authorization"]["run_causal_test"]:
        raise RuntimeError("Phase398 causal test was not authorized")
    rows = []
    completes = []
    for model in MODELS:
        root = OUT / f"causal/causal_test/private/models/{model}"
        complete = read_json(root / "complete.json")
        if not complete["valid"] or complete["direction_count"] != 144:
            raise RuntimeError(f"Invalid Phase398 causal result for {model}")
        completes.append(complete)
        rows.extend(read_jsonl(root / "direction_rows.jsonl"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["task_surface"])].append(row)
    gates = protocol["frozen_gates"]
    cells = []
    for (model, surface), items in sorted(grouped.items()):
        if len(items) != 48:
            raise RuntimeError(f"Phase398 causal cell {model}/{surface} has {len(items)} directions")
        by_scenario = {
            scenario: [next(value for value in item["scenario_rows"] if value["scenario"] == scenario) for item in items]
            for scenario in SCENARIOS
        }
        baseline_rate = rate([row["recipient_target_present"] for row in by_scenario["no_intervention"]])
        identity_rate = rate([row["recipient_target_present"] for row in by_scenario["identity_candidate_parent"]])
        switch_rates = {
            scenario: rate([row["strict_donor_answer_switch"] for row in values])
            for scenario, values in by_scenario.items()
            if scenario not in {"no_intervention", "identity_candidate_parent"}
        }
        same = switch_rates["same_order_joint_donor_candidate"]
        wrong_order = switch_rates["wrong_order_joint_donor_candidate"]
        wrong_depth = switch_rates["same_order_joint_donor_wrong_depth"]
        baseline_pass = baseline_rate == gates["baseline_recipient_correct_rate"]
        identity_pass = identity_rate == gates["identity_recipient_correct_rate"]
        sufficiency_pass = same >= gates["minimum_same_order_donor_answer_switch_rate"]
        order_pass = same - wrong_order >= gates["minimum_same_order_advantage_over_wrong_order_switch_rate"]
        depth_pass = same - wrong_depth >= gates["minimum_same_order_advantage_over_wrong_depth_switch_rate"]
        cells.append({
            "model": model,
            "task_surface": surface,
            "direction_count": len(items),
            "baseline_recipient_correct_rate": baseline_rate,
            "identity_recipient_correct_rate": identity_rate,
            "same_order_donor_answer_switch_rate": same,
            "wrong_order_donor_answer_switch_rate": wrong_order,
            "wrong_depth_donor_answer_switch_rate": wrong_depth,
            "answer_anchor_control_answer_switch_rate": switch_rates["same_order_donor_answer_anchor_control"],
            "same_order_advantage_over_wrong_order": round(same - wrong_order, 9),
            "same_order_advantage_over_wrong_depth": round(same - wrong_depth, 9),
            "baseline_gate_pass": baseline_pass,
            "identity_gate_pass": identity_pass,
            "sufficiency_gate_pass": sufficiency_pass,
            "order_specificity_gate_pass": order_pass,
            "depth_specificity_gate_pass": depth_pass,
            "causal_cell_gate_pass": baseline_pass and identity_pass and sufficiency_pass and order_pass and depth_pass,
        })
    pass_count = sum(cell["causal_cell_gate_pass"] for cell in cells)
    result = {
        "schema_version": "72.13.0",
        "phase_id": "Phase398-OrderConditionedCausalAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": protocol["objective"],
        "frozen_gates": gates,
        "denominator": {
            "model_count": 3,
            "task_surface_count": 3,
            "group_count": sum(row["group_count"] for row in completes),
            "direction_count": len(rows),
            "scenario_count": sum(row["scenario_count"] for row in completes),
            "model_surface_cell_count": len(cells),
        },
        "results": {
            "passing_causal_cell_count": pass_count,
            "same_order_total_answer_switch_count": sum(round(cell["same_order_donor_answer_switch_rate"] * cell["direction_count"]) for cell in cells),
            "direction_count": len(rows),
            "crossmodel_crosssurface_causal_gate_pass": pass_count == 9,
        },
        "cells": cells,
        "claim_boundary": {
            "candidate_state_causal_sufficiency_established": pass_count == 9,
            "candidate_state_natural_necessity_established": False,
            "candidate_is_portable_order_invariant_vector": False,
            "complete_binding_algorithm_established": False,
            "complete_language_path_established": False,
            "single_neuron_mechanism_established": False,
        },
        "authorization": {
            "promote_order_conditioned_query_state_to_aggregate_causal_path": pass_count == 9,
            "run_single_neuron_localization": False,
            "claim_language_encoding_closure": False,
        },
    }
    path = OUT / "phase398_order_conditioned_causal_analysis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
