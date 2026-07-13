#!/usr/bin/env python3
"""Validate frozen Phase399 dynamic chains on calibration or physical holdout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = ("calibration", "physical_holdout")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def aggregate(rows: list[dict[str, Any]], layer: int, protocol: dict[str, Any]) -> dict[str, Any]:
    group_gate = protocol["per_group_layer_gate"]
    validation_gate = protocol["calibration_and_physical_gate"]
    norms = [
        row["interaction_trajectories"]["ROQ"]["min_axis_normalized_norm"][layer]
        for row in rows
    ]
    cosines = [
        row["interaction_trajectories"]["ROQ"]["cross_axis_cosine"][layer]
        for row in rows
    ]
    ratios = [row["roq_to_strongest_competing_interaction"][layer] for row in rows]
    group_passes = [
        norm >= group_gate["roq_min_axis_normalized_norm_min"]
        and cosine >= group_gate["roq_cross_axis_cosine_min"]
        and ratio >= group_gate["roq_to_competing_interaction_min"]
        for norm, cosine, ratio in zip(norms, cosines, ratios, strict=True)
    ]
    metrics = {
        "group_count": len(rows),
        "group_pass_count": sum(group_passes),
        "group_pass_rate": sum(group_passes) / len(group_passes),
        "median_roq_min_axis_normalized_norm": median(norms),
        "median_roq_cross_axis_cosine": median(cosines),
        "median_roq_to_competing_interaction": median(ratios),
    }
    metrics["gate_pass"] = (
        metrics["group_pass_rate"] >= validation_gate["group_pass_rate_min"]
        and metrics["median_roq_min_axis_normalized_norm"]
        >= validation_gate["median_roq_min_axis_normalized_norm_min"]
        and metrics["median_roq_cross_axis_cosine"]
        >= validation_gate["median_roq_cross_axis_cosine_min"]
        and metrics["median_roq_to_competing_interaction"]
        >= validation_gate["median_roq_to_competing_interaction_min"]
    )
    return metrics


def main(stage: str) -> None:
    protocol = read_json(OUT / "phase399_dynamic_candidate_protocol.json")
    candidate = read_json(OUT / "phase399_dynamic_candidate_freeze.json")
    if stage == "calibration" and not candidate["authorization"]["run_calibration_trace"]:
        raise RuntimeError("Phase399 calibration is not authorized")
    if stage == "physical_holdout":
        calibration = read_json(OUT / "phase399_dynamic_calibration_validation.json")
        if not calibration["authorization"]["open_physical_holdout"]:
            raise RuntimeError("Phase399 physical holdout is not authorized")
    all_rows: list[dict[str, Any]] = []
    quality_group_count = 0
    group_count = 0
    case_count = 0
    for model in MODELS:
        root = OUT / "dynamic_trace" / stage / "private/models" / model
        complete = read_json(root / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase399 {stage} collection for {model}")
        all_rows.extend(read_jsonl(root / "event_trajectory_rows.jsonl"))
        quality_group_count += complete["quality_group_count"]
        group_count += complete["group_count"]
        case_count += complete["case_count"]
    indexed: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        indexed[(row["model"], row["surface_private"], row["event_id"])].append(row)
    cells: list[dict[str, Any]] = []
    for frozen_cell in candidate["cells"]:
        class_results: dict[str, Any] = {}
        for class_name, frozen_class in frozen_cell["event_classes"].items():
            rows = indexed[
                (
                    frozen_cell["model"],
                    frozen_cell["surface"],
                    frozen_class["event_id"],
                )
            ]
            if not rows:
                raise RuntimeError(
                    f"Missing Phase399 {stage} event {frozen_class['event_id']}"
                )
            metrics = aggregate(rows, frozen_class["layer_index"], protocol)
            class_results[class_name] = {
                "event_id": frozen_class["event_id"],
                "layer_index": frozen_class["layer_index"],
                "relative_depth": frozen_class["relative_depth"],
                "required_for_chain": frozen_class["required_for_chain"],
                "metrics": metrics,
                "validation_pass": metrics["gate_pass"],
            }
        required = protocol["chain_gate"]["required_classes"]
        required_pass = all(
            class_results[name]["validation_pass"] for name in required
        )
        cells.append(
            {
                "model": frozen_cell["model"],
                "surface": frozen_cell["surface"],
                "event_classes": class_results,
                "discovery_chain_pass": frozen_cell[
                    "dynamic_chain_discovery_pass"
                ],
                "ordered_peak_layer_gate_pass": frozen_cell[
                    "ordered_peak_layer_gate_pass"
                ],
                "dynamic_chain_validation_pass": bool(
                    frozen_cell["dynamic_chain_discovery_pass"] and required_pass
                ),
            }
        )
    surfaces = sorted({row["surface"] for row in cells})
    crossmodel = []
    for surface in surfaces:
        surface_cells = [row for row in cells if row["surface"] == surface]
        crossmodel.append(
            {
                "surface": surface,
                "model_cell_count": len(surface_cells),
                "passing_model_cell_count": sum(
                    row["dynamic_chain_validation_pass"] for row in surface_cells
                ),
                "crossmodel_validation_pass": len(surface_cells) == len(MODELS)
                and all(
                    row["dynamic_chain_validation_pass"] for row in surface_cells
                ),
            }
        )
    valid_quality = quality_group_count == group_count
    passing_cells = sum(row["dynamic_chain_validation_pass"] for row in cells)
    passing_surfaces = sum(
        row["crossmodel_validation_pass"] for row in crossmodel
    )
    result = {
        "schema_version": "73.8.0",
        "phase_id": f"Phase399-Dynamic-{stage}-Validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "denominator": {
            "case_count": case_count,
            "group_model_cell_count": group_count,
            "quality_group_model_cell_count": quality_group_count,
            "candidate_model_surface_cell_count": len(cells),
        },
        "cells": cells,
        "crossmodel_surfaces": crossmodel,
        "results": {
            "dynamic_chain_validation_cell_count": passing_cells,
            "candidate_model_surface_cell_count": len(cells),
            "crossmodel_surface_count": passing_surfaces,
            "all_collection_quality_gates_pass": valid_quality,
        },
        "authorization": {
            "open_physical_holdout": stage == "calibration"
            and valid_quality
            and passing_cells > 0,
            "run_joint_causal_intervention": stage == "physical_holdout"
            and valid_quality
            and passing_surfaces > 0,
            "head_channel_or_neuron_scan": False,
        },
        "claim_boundary": {
            "validated_chain_is_causal": False,
            "crossmodel_role_chain_uses_identical_neurons": False,
            "failed_cell_has_no_dynamic_process": False,
        },
    }
    output_name = (
        "phase399_dynamic_calibration_validation.json"
        if stage == "calibration"
        else "phase399_dynamic_physical_validation.json"
    )
    write_json(OUT / output_name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    main(args.stage)
