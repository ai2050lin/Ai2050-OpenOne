#!/usr/bin/env python3
"""Analyze Phase397 discovery factor traces with fixed elementary gates."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling", "coreference_resolution")
MIN_RELATION_DELTA = 0.02
MIN_DEPTH_INCREMENT = 0.005
MIN_SUPPORT = 6
MAX_QUERY_SOURCE_DELTA = 1e-6


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
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        root = OUT / "factor_trace/discovery" / model
        complete = read_json(root / "complete.json")
        if not complete["valid"] or complete["group_count"] != 24 or complete["natural_case_count"] != 240:
            raise RuntimeError(f"Invalid Phase397 discovery trace for {model}")
        rows.extend(read_jsonl(root / "factor_rows.jsonl"))
    if len(rows) != 648:
        raise RuntimeError(f"Expected 648 Phase397 discovery factor rows, got {len(rows)}")

    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["task_surface"], row["parallel_group_id"])][row["factor"]] = row
    if len(grouped) != 72 or any(len(factors) != 9 for factors in grouped.values()):
        raise RuntimeError("Invalid Phase397 discovery group factor coverage")

    group_rows = []
    for (model, surface, group_id), factors in sorted(grouped.items()):
        relation_candidate = min(
            factors["relation_x"]["candidate_relative_delta"],
            factors["relation_y"]["candidate_relative_delta"],
        )
        relation_wrong = min(
            factors["relation_x"]["wrong_depth_relative_delta"],
            factors["relation_y"]["wrong_depth_relative_delta"],
        )
        group_rows.append(
            {
                "schema_version": "71.6.0",
                "phase_id": "Phase397-FactorDiscoveryAnalysis",
                "model": model,
                "task_surface": surface,
                "parallel_group_id": group_id,
                "minimum_two_axis_relation_candidate_delta": relation_candidate,
                "minimum_two_axis_relation_wrong_depth_delta": relation_wrong,
                "relation_candidate_minus_wrong_delta": relation_candidate - relation_wrong,
                "content_candidate_delta": factors["content"]["candidate_relative_delta"],
                "minimum_order_candidate_delta": min(factors["order_x"]["candidate_relative_delta"], factors["order_y"]["candidate_relative_delta"]),
                "minimum_syntax_candidate_delta": min(factors["syntax_x"]["candidate_relative_delta"], factors["syntax_y"]["candidate_relative_delta"]),
                "maximum_query_source_candidate_delta": max(factors["query_x"]["candidate_relative_delta"], factors["query_y"]["candidate_relative_delta"]),
                "relation_threshold_pass": relation_candidate >= MIN_RELATION_DELTA,
                "depth_increment_threshold_pass": relation_candidate - relation_wrong >= MIN_DEPTH_INCREMENT,
            }
        )

    cells = []
    for model in MODELS:
        for surface in SURFACES:
            selected = [row for row in group_rows if row["model"] == model and row["task_surface"] == surface]
            if len(selected) != 8:
                raise RuntimeError(f"Expected 8 groups for {model}/{surface}, got {len(selected)}")
            relation_support = sum(row["relation_threshold_pass"] for row in selected)
            depth_support = sum(row["depth_increment_threshold_pass"] for row in selected)
            relation_median = median(row["minimum_two_axis_relation_candidate_delta"] for row in selected)
            wrong_median = median(row["minimum_two_axis_relation_wrong_depth_delta"] for row in selected)
            increment_median = median(row["relation_candidate_minus_wrong_delta"] for row in selected)
            query_maximum = max(row["maximum_query_source_candidate_delta"] for row in selected)
            gate = (
                relation_median >= MIN_RELATION_DELTA
                and increment_median >= MIN_DEPTH_INCREMENT
                and relation_support >= MIN_SUPPORT
                and depth_support >= MIN_SUPPORT
                and query_maximum <= MAX_QUERY_SOURCE_DELTA
            )
            cells.append(
                {
                    "model": model,
                    "task_surface": surface,
                    "group_count": len(selected),
                    "median_minimum_two_axis_relation_candidate_delta": relation_median,
                    "median_minimum_two_axis_relation_wrong_depth_delta": wrong_median,
                    "median_relation_candidate_minus_wrong_delta": increment_median,
                    "median_content_candidate_delta": median(row["content_candidate_delta"] for row in selected),
                    "median_minimum_order_candidate_delta": median(row["minimum_order_candidate_delta"] for row in selected),
                    "median_minimum_syntax_candidate_delta": median(row["minimum_syntax_candidate_delta"] for row in selected),
                    "maximum_query_source_candidate_delta": query_maximum,
                    "relation_support_count": relation_support,
                    "depth_increment_support_count": depth_support,
                    "observational_relation_context_candidate_gate_pass": gate,
                }
            )
    shared_gate = len(cells) == 9 and all(cell["observational_relation_context_candidate_gate_pass"] for cell in cells)
    payload = {
        "schema_version": "71.6.0",
        "phase_id": "Phase397-FactorDiscoveryAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_cell": 8,
            "cell_count": len(cells),
            "group_factor_summary_count": len(group_rows),
            "raw_factor_pair_count": len(rows),
        },
        "frozen_gates": {
            "minimum_median_two_axis_relation_candidate_delta": MIN_RELATION_DELTA,
            "minimum_median_candidate_minus_wrong_delta": MIN_DEPTH_INCREMENT,
            "minimum_group_support": MIN_SUPPORT,
            "maximum_query_source_delta": MAX_QUERY_SOURCE_DELTA,
            "all_nine_model_surface_cells_required": True,
        },
        "cells": cells,
        "results": {
            "passing_model_surface_cell_count": sum(cell["observational_relation_context_candidate_gate_pass"] for cell in cells),
            "crossmodel_crosssurface_observational_candidate_gate_pass": shared_gate,
            "causal_relation_binding_count": 0,
            "natural_necessity_count": 0,
            "single_neuron_mechanism_count": 0,
        },
        "authorization": {
            "calibration_trace": shared_gate,
            "causal_discovery_intervention": shared_gate,
            "physical_holdout_trace": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "observational_relation_delta_is_causal_binding": False,
            "candidate_minus_one_wrong_depth_maps_full_formation_curve": False,
            "relation_delta_exceeding_zero_is_abstract_rule": False,
        },
    }
    write_jsonl(OUT / "factor_trace/analysis/phase397_discovery_group_rows.jsonl", group_rows)
    path = OUT / "phase397_factor_discovery_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
