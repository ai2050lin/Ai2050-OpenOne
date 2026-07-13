#!/usr/bin/env python3
"""Select Phase399 aggregate dynamic event chains on discovery only."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def event_allowed(event_id: str, config: dict[str, Any]) -> bool:
    return event_id in config["prefixes"]


def aggregate_event_layer(
    rows: list[dict[str, Any]], layer: int, group_gate: dict[str, float]
) -> dict[str, Any]:
    norms = [
        row["interaction_trajectories"]["ROQ"]["min_axis_normalized_norm"][layer]
        for row in rows
    ]
    cosines = [
        row["interaction_trajectories"]["ROQ"]["cross_axis_cosine"][layer]
        for row in rows
    ]
    ratios = [row["roq_to_strongest_competing_interaction"][layer] for row in rows]
    passes = [
        norm >= group_gate["roq_min_axis_normalized_norm_min"]
        and cosine >= group_gate["roq_cross_axis_cosine_min"]
        and ratio >= group_gate["roq_to_competing_interaction_min"]
        for norm, cosine, ratio in zip(norms, cosines, ratios, strict=True)
    ]
    return {
        "group_count": len(rows),
        "group_pass_count": sum(passes),
        "group_pass_rate": sum(passes) / len(passes),
        "median_roq_min_axis_normalized_norm": median(norms),
        "median_roq_cross_axis_cosine": median(cosines),
        "median_roq_to_competing_interaction": median(ratios),
    }


def cell_pass(metrics: dict[str, Any], gate: dict[str, float]) -> bool:
    return (
        metrics["group_pass_rate"] >= gate["group_pass_rate_min"]
        and metrics["median_roq_min_axis_normalized_norm"]
        >= gate["median_roq_min_axis_normalized_norm_min"]
        and metrics["median_roq_cross_axis_cosine"]
        >= gate["median_roq_cross_axis_cosine_min"]
        and metrics["median_roq_to_competing_interaction"]
        >= gate["median_roq_to_competing_interaction_min"]
    )


def score(metrics: dict[str, Any]) -> float:
    return (
        metrics["group_pass_rate"]
        * metrics["median_roq_min_axis_normalized_norm"]
        * max(metrics["median_roq_cross_axis_cosine"], 0.0)
        * min(metrics["median_roq_to_competing_interaction"], 8.0)
    )


def main() -> None:
    instrument = read_json(OUT / "phase399_instrument_audit.json")
    if not instrument["authorization"]["run_discovery_trace"]:
        raise RuntimeError("Phase399 discovery analysis is not authorized")
    protocol = read_json(OUT / "phase399_dynamic_candidate_protocol.json")
    freeze = read_json(OUT / "phase399_behavior_freeze_summary.json")
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        root = OUT / "dynamic_trace/discovery/private/models" / model
        complete = read_json(root / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase399 discovery collection for {model}")
        rows.extend(read_jsonl(root / "event_trajectory_rows.jsonl"))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["surface_private"], row["event_id"])].append(row)
    candidates: list[dict[str, Any]] = []
    search_count = 0
    for model in MODELS:
        for surface in freeze["eligible_surfaces"]:
            cell_candidates: dict[str, dict[str, Any]] = {}
            for class_name, class_config in protocol["event_classes"].items():
                options: list[dict[str, Any]] = []
                for (row_model, row_surface, event_id), event_rows in grouped.items():
                    if row_model != model or row_surface != surface:
                        continue
                    if not event_allowed(event_id, class_config):
                        continue
                    layer_count = event_rows[0]["layer_count"]
                    for layer in range(layer_count):
                        search_count += 1
                        metrics = aggregate_event_layer(
                            event_rows,
                            layer,
                            protocol["per_group_layer_gate"],
                        )
                        options.append(
                            {
                                "event_id": event_id,
                                "layer_index": layer,
                                "relative_depth": layer / max(layer_count - 1, 1),
                                "metrics": metrics,
                                "gate_pass": cell_pass(
                                    metrics, protocol["discovery_cell_gate"]
                                ),
                                "score": score(metrics),
                            }
                        )
                if not options:
                    raise RuntimeError(
                        f"No Phase399 discovery options for {model}/{surface}/{class_name}"
                    )
                passing = [option for option in options if option["gate_pass"]]
                chosen = max(passing or options, key=lambda option: option["score"])
                cell_candidates[class_name] = {
                    "class_name": class_name,
                    "required_for_chain": class_config["required_for_chain"],
                    **chosen,
                    "searched_event_layer_count": len(options),
                    "selected_from_passing_set": bool(passing),
                }
            required = protocol["chain_gate"]["required_classes"]
            class_pass = all(cell_candidates[name]["gate_pass"] for name in required)
            layers = [cell_candidates[name]["layer_index"] for name in required]
            ordered = layers == sorted(layers)
            candidates.append(
                {
                    "model": model,
                    "surface": surface,
                    "event_classes": cell_candidates,
                    "required_class_gate_pass": class_pass,
                    "ordered_peak_layer_gate_pass": ordered,
                    "dynamic_chain_discovery_pass": class_pass and ordered,
                }
            )
    crossmodel = []
    for surface in freeze["eligible_surfaces"]:
        cells = [row for row in candidates if row["surface"] == surface]
        crossmodel.append(
            {
                "surface": surface,
                "model_cell_count": len(cells),
                "passing_model_cell_count": sum(
                    row["dynamic_chain_discovery_pass"] for row in cells
                ),
                "crossmodel_discovery_pass": len(cells) == len(MODELS)
                and all(row["dynamic_chain_discovery_pass"] for row in cells),
            }
        )
    result = {
        "schema_version": "73.7.0",
        "phase_id": "Phase399-DynamicDiscoveryAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "search_denominator": {
            "event_layer_candidate_count": search_count,
            "model_surface_cell_count": len(candidates),
            "event_class_count_per_cell": len(protocol["event_classes"]),
            "discovery_group_count_per_cell": 10,
        },
        "cells": candidates,
        "crossmodel_surfaces": crossmodel,
        "results": {
            "dynamic_chain_discovery_cell_count": sum(
                row["dynamic_chain_discovery_pass"] for row in candidates
            ),
            "model_surface_cell_count": len(candidates),
            "crossmodel_surface_count": sum(
                row["crossmodel_discovery_pass"] for row in crossmodel
            ),
        },
        "authorization": {
            "run_calibration_trace": any(
                row["dynamic_chain_discovery_pass"] for row in candidates
            ),
            "open_physical_holdout": False,
            "run_joint_causal_intervention": False,
            "head_channel_or_neuron_scan": False,
        },
        "claim_boundary": {
            "discovery_selected_chain_is_confirmed": False,
            "discovery_selected_chain_is_causal": False,
            "nonpassing_cell_has_no_dynamic_binding": False,
        },
    }
    write_json(OUT / "phase399_dynamic_discovery_analysis.json", result)
    write_json(OUT / "phase399_dynamic_candidate_freeze.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
