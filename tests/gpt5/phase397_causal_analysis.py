#!/usr/bin/env python3
"""Evaluate frozen Phase397 causal factor-separation gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling", "coreference_resolution")
CONTROL_SCENARIOS = (
    "donor_content_candidate",
    "donor_order_candidate",
    "donor_syntax_candidate",
    "donor_query_source_candidate",
    "donor_entities_candidate",
    "donor_random_candidate",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in row["scenario_rows"] if item["scenario"] == name)


def summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [scenario(row, name)["normalized_relation_margin_mediation"] for row in rows]
    return {
        "median_normalized_relation_margin_mediation": median(values),
        "positive_direction_count": sum(value > 0 for value in values),
        "positive_direction_rate": sum(value > 0 for value in values) / len(values),
    }


def main() -> None:
    protocol = read_json(OUT / "phase397_causal_protocol.json")
    gates = protocol["frozen_gates_inherited_and_extended_from_phase395"]
    all_rows: list[dict[str, Any]] = []
    for model in MODELS:
        root = OUT / "causal/discovery" / model
        complete = read_json(root / "complete.json")
        rows = read_jsonl(root / "direction_rows.jsonl")
        if not complete["valid"] or len(rows) != 48:
            raise RuntimeError(f"Invalid Phase397 causal output for {model}")
        all_rows.extend(rows)
    if len(all_rows) != 144:
        raise RuntimeError(f"Expected 144 causal directions, got {len(all_rows)}")

    cells = []
    for model in MODELS:
        for surface in SURFACES:
            rows = [row for row in all_rows if row["model"] == model and row["task_surface"] == surface]
            if len(rows) != 16:
                raise RuntimeError(f"Expected 16 directions for {model}/{surface}, got {len(rows)}")
            names = [item["scenario"] for item in rows[0]["scenario_rows"]]
            summaries = {name: summarize(rows, name) for name in names}
            relation = summaries["donor_relation_candidate"]
            control_advantages = {
                name: relation["median_normalized_relation_margin_mediation"]
                - summaries[name]["median_normalized_relation_margin_mediation"]
                for name in CONTROL_SCENARIOS
            }
            depth_advantage = (
                relation["median_normalized_relation_margin_mediation"]
                - summaries["donor_relation_wrong_depth"]["median_normalized_relation_margin_mediation"]
            )
            switch_count = sum(row["relation_generation_row"]["strict_donor_target_switch"] for row in rows)
            switch_rate = switch_count / len(rows)
            gate = (
                relation["median_normalized_relation_margin_mediation"]
                >= gates["minimum_median_relation_normalized_margin_mediation"]
                and all(value >= gates["minimum_relation_advantage_over_each_local_control"] for value in control_advantages.values())
                and relation["positive_direction_rate"] >= gates["minimum_positive_relation_direction_rate"]
                and switch_rate >= gates["minimum_relation_answer_switch_rate"]
                and depth_advantage >= gates["minimum_candidate_advantage_over_wrong_depth"]
            )
            cells.append(
                {
                    "model": model,
                    "task_surface": surface,
                    "direction_count": len(rows),
                    "scenario_summaries": summaries,
                    "relation_advantage_over_controls": control_advantages,
                    "minimum_relation_advantage_over_local_controls": min(control_advantages.values()),
                    "candidate_advantage_over_wrong_depth": depth_advantage,
                    "relation_answer_switch_count": switch_count,
                    "relation_answer_switch_rate": switch_rate,
                    "causal_relation_context_specificity_gate_pass": gate,
                }
            )
    shared_gate = len(cells) == 9 and all(cell["causal_relation_context_specificity_gate_pass"] for cell in cells)
    maximum_identity_effect = max(
        abs(scenario(row, "identity_relation_candidate")["normalized_relation_margin_mediation"])
        for row in all_rows
    )
    maximum_query_control_effect = max(
        abs(scenario(row, "donor_query_source_candidate")["normalized_relation_margin_mediation"])
        for row in all_rows
    )
    maximum_patch_error = max(
        max(item["patch_audit"]["max_patch_error"], item["patch_audit"]["max_outside_error"])
        for row in all_rows for item in row["scenario_rows"]
    )
    payload = {
        "schema_version": "71.11.0",
        "phase_id": "Phase397-CausalAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_cell": 8,
            "directions_per_cell": 16,
            "direction_count": len(all_rows),
            "scenario_count": len(all_rows) * len(all_rows[0]["scenario_rows"]),
            "generation_count": len(all_rows),
        },
        "frozen_gates": gates,
        "cells": cells,
        "results": {
            "passing_model_surface_cell_count": sum(cell["causal_relation_context_specificity_gate_pass"] for cell in cells),
            "crossmodel_crosssurface_causal_relation_context_gate_pass": shared_gate,
            "maximum_identity_effect": maximum_identity_effect,
            "maximum_query_source_control_effect": maximum_query_control_effect,
            "maximum_patch_locality_error": maximum_patch_error,
            "calibrated_causal_relation_binding_count": 0,
            "natural_necessity_count": 0,
            "single_neuron_mechanism_count": 0,
        },
        "authorization": {
            "calibration_causal_intervention": shared_gate,
            "physical_causal_holdout": False,
            "record_crosssurface_causal_rule": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "discovery_causal_pass_is_calibrated_rule": False,
            "failed_shared_gate_erases_local_positive_cells": False,
            "aggregate_sufficiency_is_natural_necessity": False,
        },
    }
    path = OUT / "phase397_causal_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
