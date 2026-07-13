#!/usr/bin/env python3
"""Replicate the Phase397 observational factor candidate on calibration groups."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling", "coreference_resolution")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    discovery = read_json(OUT / "phase397_factor_discovery_analysis.json")
    gates = discovery["frozen_gates"]
    rows = []
    for model in MODELS:
        root = OUT / "factor_trace/calibration" / model
        complete = read_json(root / "complete.json")
        if not complete["valid"] or complete["group_count"] != 12:
            raise RuntimeError(f"Invalid Phase397 calibration trace for {model}")
        rows.extend(read_jsonl(root / "factor_rows.jsonl"))
    if len(rows) != 324:
        raise RuntimeError(f"Expected 324 Phase397 calibration factor rows, got {len(rows)}")
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["task_surface"], row["parallel_group_id"])][row["factor"]] = row
    cells = []
    for model in MODELS:
        for surface in SURFACES:
            groups = [factors for (item_model, item_surface, _group), factors in grouped.items() if item_model == model and item_surface == surface]
            if len(groups) != 4 or any(len(factors) != 9 for factors in groups):
                raise RuntimeError(f"Invalid calibration groups for {model}/{surface}")
            items = []
            for factors in groups:
                relation_candidate = min(factors["relation_x"]["candidate_relative_delta"], factors["relation_y"]["candidate_relative_delta"])
                relation_wrong = min(factors["relation_x"]["wrong_depth_relative_delta"], factors["relation_y"]["wrong_depth_relative_delta"])
                items.append(
                    {
                        "relation": relation_candidate,
                        "wrong": relation_wrong,
                        "increment": relation_candidate - relation_wrong,
                        "query": max(factors["query_x"]["candidate_relative_delta"], factors["query_y"]["candidate_relative_delta"]),
                        "content": factors["content"]["candidate_relative_delta"],
                        "order": min(factors["order_x"]["candidate_relative_delta"], factors["order_y"]["candidate_relative_delta"]),
                        "syntax": min(factors["syntax_x"]["candidate_relative_delta"], factors["syntax_y"]["candidate_relative_delta"]),
                    }
                )
            relation_support = sum(item["relation"] >= gates["minimum_median_two_axis_relation_candidate_delta"] for item in items)
            depth_support = sum(item["increment"] >= gates["minimum_median_candidate_minus_wrong_delta"] for item in items)
            gate = (
                median(item["relation"] for item in items) >= gates["minimum_median_two_axis_relation_candidate_delta"]
                and median(item["increment"] for item in items) >= gates["minimum_median_candidate_minus_wrong_delta"]
                and relation_support >= 3
                and depth_support >= 3
                and max(item["query"] for item in items) <= gates["maximum_query_source_delta"]
            )
            cells.append(
                {
                    "model": model,
                    "task_surface": surface,
                    "group_count": len(items),
                    "median_minimum_two_axis_relation_candidate_delta": median(item["relation"] for item in items),
                    "median_minimum_two_axis_relation_wrong_depth_delta": median(item["wrong"] for item in items),
                    "median_relation_candidate_minus_wrong_delta": median(item["increment"] for item in items),
                    "median_content_candidate_delta": median(item["content"] for item in items),
                    "median_minimum_order_candidate_delta": median(item["order"] for item in items),
                    "median_minimum_syntax_candidate_delta": median(item["syntax"] for item in items),
                    "maximum_query_source_candidate_delta": max(item["query"] for item in items),
                    "relation_support_count": relation_support,
                    "depth_increment_support_count": depth_support,
                    "calibration_observational_candidate_gate_pass": gate,
                }
            )
    shared_gate = len(cells) == 9 and all(cell["calibration_observational_candidate_gate_pass"] for cell in cells)
    payload = {
        "schema_version": "71.7.0",
        "phase_id": "Phase397-FactorCalibrationAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_cell": 4,
            "cell_count": len(cells),
            "raw_factor_pair_count": len(rows),
            "group_overlap_with_discovery": 0,
        },
        "frozen_gates_inherited_from_discovery": gates,
        "calibration_support_required_of_four": 3,
        "cells": cells,
        "results": {
            "passing_model_surface_cell_count": sum(cell["calibration_observational_candidate_gate_pass"] for cell in cells),
            "crossmodel_crosssurface_calibration_gate_pass": shared_gate,
            "causal_relation_binding_count": 0,
            "natural_necessity_count": 0,
        },
        "authorization": {
            "physical_holdout_trace": shared_gate,
            "causal_calibration_intervention": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "calibrated_observational_factor_is_causal_binding": False,
            "single_wrong_depth_is_complete_physical_path": False,
        },
    }
    path = OUT / "phase397_factor_calibration_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
