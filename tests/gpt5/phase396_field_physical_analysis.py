#!/usr/bin/env python3
"""Evaluate the frozen Phase396 field-extraction physical gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase396_field_binding_physical"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in row["scenario_rows"] if item["scenario"] == name)


def generation(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in row["generation_rows"] if item["scenario"] == name)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [scenario(row, name)["normalized_margin_mediation"] for row in rows]
    generated = [
        generation(row, name) for row in rows
        if any(item["scenario"] == name for item in row["generation_rows"])
    ]
    return {
        "median_normalized_margin_mediation": median(values),
        "positive_direction_count": sum(value > 0 for value in values),
        "positive_direction_rate": sum(value > 0 for value in values) / len(values),
        "answer_switch_count": sum(item["strict_donor_target_switch"] for item in generated),
        "answer_switch_rate": (
            sum(item["strict_donor_target_switch"] for item in generated) / len(generated)
            if generated else None
        ),
    }


def main() -> None:
    protocol = read_json(OUT / "phase396_protocol.json")
    gates = protocol["frozen_gates_inherited_without_change"]
    cells = []
    all_rows = []
    for model in MODELS:
        complete = read_json(OUT / "collection" / model / "complete.json")
        rows = read_jsonl(OUT / "collection" / model / "direction_rows.jsonl")
        if not complete["valid"] or len(rows) != 24:
            raise RuntimeError(f"Invalid Phase396 result for {model}")
        all_rows.extend(rows)
        summaries = {
            name: summarize(rows, name)
            for name in (item["scenario"] for item in rows[0]["scenario_rows"])
        }
        same = summaries["donor_same_literal_candidate"]
        entity = summaries["donor_source_entities_candidate"]
        random = summaries["donor_same_count_random_candidate"]
        wrong = summaries["donor_same_literal_wrong_depth"]
        advantage_entity = (
            same["median_normalized_margin_mediation"]
            - entity["median_normalized_margin_mediation"]
        )
        advantage_random = (
            same["median_normalized_margin_mediation"]
            - random["median_normalized_margin_mediation"]
        )
        depth_advantage = (
            same["median_normalized_margin_mediation"]
            - wrong["median_normalized_margin_mediation"]
        )
        gate = (
            same["median_normalized_margin_mediation"]
            >= gates["minimum_median_same_literal_normalized_margin_mediation"]
            and advantage_entity >= gates["minimum_same_literal_advantage_over_entity"]
            and advantage_random >= gates["minimum_same_literal_advantage_over_random"]
            and same["positive_direction_rate"]
            >= gates["minimum_positive_same_literal_direction_rate"]
            and same["answer_switch_rate"]
            >= gates["minimum_same_literal_answer_switch_rate"]
        )
        cells.append({
            "model": model,
            "task_surface": "field_extraction",
            "direction_count": len(rows),
            "scenario_summaries": summaries,
            "same_literal_advantage_over_entity": advantage_entity,
            "same_literal_advantage_over_random": advantage_random,
            "candidate_advantage_over_wrong_depth": depth_advantage,
            "candidate_depth_specific": depth_advantage
            >= gates["minimum_candidate_advantage_over_wrong_depth_for_depth_specificity"],
            "physical_static_same_literal_context_transport_gate_pass": gate,
        })
    shared_gate = all(cell["physical_static_same_literal_context_transport_gate_pass"] for cell in cells)
    maximum_patch_error = max(
        max(item["patch_audit"]["max_patch_error"], item["patch_audit"]["max_outside_error"])
        for row in all_rows for item in row["scenario_rows"]
    )
    maximum_identity_effect = max(
        abs(scenario(row, "identity_same_literal_candidate")["normalized_margin_mediation"])
        for row in all_rows
    )
    payload = {
        "schema_version": "70.2.0",
        "phase_id": "Phase396-FieldPhysicalAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surface": "field_extraction",
            "unseen_group_count_per_model": 6,
            "direction_count": len(all_rows),
            "scenario_count": len(all_rows) * 9,
            "generation_count": len(all_rows) * 7,
        },
        "frozen_gates": gates,
        "cells": cells,
        "results": {
            "physical_model_cell_pass_count": sum(
                cell["physical_static_same_literal_context_transport_gate_pass"] for cell in cells
            ),
            "crossmodel_field_specific_physical_replication_gate_pass": shared_gate,
            "maximum_identity_effect": maximum_identity_effect,
            "maximum_patch_locality_error": maximum_patch_error,
            "crosssurface_shared_binding_rule_count": 0,
            "abstract_binding_algorithm_count": 0,
            "natural_necessity_count": 0,
            "single_neuron_mechanism_count": 0,
        },
        "authorization": {
            "record_field_specific_context_carrier": shared_gate,
            "crosssurface_binding_claim": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "field_context_carrier_is_binding_algorithm": False,
            "field_replication_generalizes_to_entity_recency": False,
            "same_literal_transport_proves_natural_necessity": False,
            "same_position_transport_is_abstract_binding": False,
        },
    }
    path = OUT / "phase396_physical_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
