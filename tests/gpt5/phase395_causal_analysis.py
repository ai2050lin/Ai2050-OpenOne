#!/usr/bin/env python3
"""Apply the frozen Phase395 causal gates without post-result threshold changes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("field_extraction", "entity_recency")
SAME_LITERAL = "donor_same_literal_candidate"
ENTITY = "donor_source_entities_candidate"
RANDOM = "donor_same_count_random_candidate"
WRONG = "donor_same_literal_wrong_depth"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in row["scenario_rows"] if item["scenario"] == name)


def generation(row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in row["generation_rows"] if item["scenario"] == name)


def summarize_scenario(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [scenario(row, name)["normalized_margin_mediation"] for row in rows]
    shifts = [scenario(row, name)["donor_margin_shift"] for row in rows]
    projections = [scenario(row, name)["query_projection_toward_donor"] for row in rows]
    generation_rows = [
        generation(row, name)
        for row in rows
        if any(item["scenario"] == name for item in row["generation_rows"])
    ]
    return {
        "median_normalized_margin_mediation": median(values),
        "median_donor_margin_shift": median(shifts),
        "median_query_projection_toward_donor": median(projections),
        "positive_direction_count": sum(value > 0 for value in values),
        "positive_direction_rate": sum(value > 0 for value in values) / len(values),
        "answer_switch_count": sum(item["strict_donor_target_switch"] for item in generation_rows),
        "answer_switch_rate": (
            sum(item["strict_donor_target_switch"] for item in generation_rows)
            / len(generation_rows)
            if generation_rows else None
        ),
    }


def main() -> None:
    protocol = read_json(OUT / "phase395_causal_calibration_protocol.json")
    gates = protocol["frozen_gates"]
    rows_by_model = {}
    for model in MODELS:
        complete = read_json(OUT / "causal/calibration" / model / "complete.json")
        rows = read_jsonl(OUT / "causal/calibration" / model / "direction_rows.jsonl")
        if not complete["valid"] or len(rows) != 48:
            raise RuntimeError(f"Invalid Phase395 calibration intervention for {model}")
        rows_by_model[model] = rows

    cells = []
    for model in MODELS:
        for surface in SURFACES:
            rows = [row for row in rows_by_model[model] if row["task_surface"] == surface]
            if len(rows) != 24:
                raise RuntimeError(f"Invalid cell denominator for {model}/{surface}")
            scenario_summaries = {
                name: summarize_scenario(rows, name)
                for name in (item["scenario"] for item in rows[0]["scenario_rows"])
            }
            same = scenario_summaries[SAME_LITERAL]
            entity = scenario_summaries[ENTITY]
            random = scenario_summaries[RANDOM]
            wrong = scenario_summaries[WRONG]
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
                "task_surface": surface,
                "direction_count": len(rows),
                "scenario_summaries": scenario_summaries,
                "same_literal_advantage_over_entity": advantage_entity,
                "same_literal_advantage_over_random": advantage_random,
                "candidate_advantage_over_wrong_depth": depth_advantage,
                "candidate_depth_specific": depth_advantage
                >= gates["minimum_candidate_advantage_over_wrong_depth_for_depth_specificity"],
                "static_same_literal_context_transport_gate_pass": gate,
            })

    shared_gate = all(cell["static_same_literal_context_transport_gate_pass"] for cell in cells)
    surface_gates = {
        surface: all(
            cell["static_same_literal_context_transport_gate_pass"]
            for cell in cells if cell["task_surface"] == surface
        )
        for surface in SURFACES
    }
    identity_max = max(
        abs(scenario(row, "identity_same_literal_candidate")["normalized_margin_mediation"])
        for rows in rows_by_model.values()
        for row in rows
    )
    patch_max = max(
        max(item["patch_audit"]["max_patch_error"], item["patch_audit"]["max_outside_error"])
        for rows in rows_by_model.values()
        for row in rows
        for item in row["scenario_rows"]
    )
    payload = {
        "schema_version": "69.10.0",
        "phase_id": "Phase395-CausalCalibrationAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "direction_count": sum(len(rows) for rows in rows_by_model.values()),
            "scenario_count": sum(len(rows) * 9 for rows in rows_by_model.values()),
            "generation_count": sum(len(rows) * 7 for rows in rows_by_model.values()),
            "cell_count": len(cells),
        },
        "frozen_gates": gates,
        "cells": cells,
        "surface_gates": surface_gates,
        "results": {
            "local_static_same_literal_context_transport_cell_count": sum(
                cell["static_same_literal_context_transport_gate_pass"] for cell in cells
            ),
            "crossmodel_field_extraction_gate_pass": surface_gates["field_extraction"],
            "crossmodel_entity_recency_gate_pass": surface_gates["entity_recency"],
            "crossmodel_crosssurface_shared_state_gate_pass": shared_gate,
            "maximum_identity_effect": identity_max,
            "maximum_patch_locality_error": patch_max,
            "abstract_binding_rule_count": 0,
            "natural_necessity_count": 0,
            "single_neuron_mechanism_count": 0,
        },
        "authorization": {
            "phase395_physical_holdout": shared_gate,
            "phase396_field_specific_physical_protocol": (
                not shared_gate and surface_gates["field_extraction"]
            ),
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "same_literal_context_transport_is_abstract_binding_algorithm": False,
            "surface_local_success_is_crosssurface_state": False,
            "same_position_content_transport_is_binding": False,
            "calibration_result_is_physical_holdout": False,
        },
    }
    path = OUT / "phase395_causal_calibration_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
