#!/usr/bin/env python3
"""Evaluate Phase397 observational factor replication on untouched physical groups."""

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
    gates = read_json(OUT / "phase397_factor_discovery_analysis.json")["frozen_gates"]
    rows = []
    for model in MODELS:
        root = OUT / "factor_trace/physical_holdout" / model
        complete = read_json(root / "complete.json")
        if not complete["valid"] or complete["group_count"] != 12:
            raise RuntimeError(f"Invalid Phase397 physical trace for {model}")
        rows.extend(read_jsonl(root / "factor_rows.jsonl"))
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["task_surface"], row["parallel_group_id"])][row["factor"]] = row
    cells = []
    for model in MODELS:
        for surface in SURFACES:
            groups = [factors for (item_model, item_surface, _group), factors in grouped.items() if item_model == model and item_surface == surface]
            if len(groups) != 4 or any(len(factors) != 9 for factors in groups):
                raise RuntimeError(f"Invalid physical groups for {model}/{surface}")
            items = []
            for factors in groups:
                candidate = min(factors["relation_x"]["candidate_relative_delta"], factors["relation_y"]["candidate_relative_delta"])
                wrong = min(factors["relation_x"]["wrong_depth_relative_delta"], factors["relation_y"]["wrong_depth_relative_delta"])
                items.append({"candidate": candidate, "wrong": wrong, "increment": candidate - wrong, "query": max(factors["query_x"]["candidate_relative_delta"], factors["query_y"]["candidate_relative_delta"])})
            relation_support = sum(item["candidate"] >= gates["minimum_median_two_axis_relation_candidate_delta"] for item in items)
            depth_support = sum(item["increment"] >= gates["minimum_median_candidate_minus_wrong_delta"] for item in items)
            gate = (
                median(item["candidate"] for item in items) >= gates["minimum_median_two_axis_relation_candidate_delta"]
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
                    "median_minimum_two_axis_relation_candidate_delta": median(item["candidate"] for item in items),
                    "median_minimum_two_axis_relation_wrong_depth_delta": median(item["wrong"] for item in items),
                    "median_relation_candidate_minus_wrong_delta": median(item["increment"] for item in items),
                    "maximum_query_source_candidate_delta": max(item["query"] for item in items),
                    "relation_support_count": relation_support,
                    "depth_increment_support_count": depth_support,
                    "physical_observational_candidate_gate_pass": gate,
                }
            )
    shared_gate = len(cells) == 9 and all(cell["physical_observational_candidate_gate_pass"] for cell in cells)
    payload = {
        "schema_version": "71.8.0",
        "phase_id": "Phase397-FactorPhysicalAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": list(SURFACES),
            "groups_per_cell": 4,
            "cell_count": len(cells),
            "raw_factor_pair_count": len(rows),
            "group_overlap_with_discovery_or_calibration": 0,
        },
        "frozen_gates_inherited_from_discovery": gates,
        "physical_support_required_of_four": 3,
        "cells": cells,
        "results": {
            "passing_model_surface_cell_count": sum(cell["physical_observational_candidate_gate_pass"] for cell in cells),
            "crossmodel_crosssurface_physical_observational_gate_pass": shared_gate,
            "causal_relation_binding_count": 0,
            "natural_necessity_count": 0,
            "complete_physical_path_count": 0,
        },
        "authorization": {
            "record_observational_relation_context_distribution": shared_gate,
            "causal_relation_binding_claim": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "three_split_observational_replication_is_causality": False,
            "two_depth_samples_are_complete_depth_path": False,
            "aggregate_value_state_is_single_neuron": False,
        },
    }
    path = OUT / "phase397_factor_physical_analysis.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
