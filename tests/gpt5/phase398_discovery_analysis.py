#!/usr/bin/env python3
"""Analyze frozen Phase398 discovery traces without changing the gates."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling", "coreference_resolution")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def rounded(value: float) -> float:
    return round(float(value), 9)


def main() -> None:
    protocol = read_json(OUT / "phase398_discovery_analysis_protocol.json")
    instrument = read_json(OUT / "phase398_instrument_audit.json")
    if not instrument["authorization"]["run_discovery_trace"]:
        raise RuntimeError("Phase398 discovery was not authorized")
    gates = protocol["frozen_gates"]
    rows, audits = [], []
    completes = []
    for model in MODELS:
        root = OUT / f"query_trace/discovery/private/models/{model}"
        complete = read_json(root / "complete.json")
        if not complete["valid"] or complete["case_count"] != 384:
            raise RuntimeError(f"Invalid Phase398 discovery collection for {model}")
        completes.append(complete)
        rows.extend(read_jsonl(root / "factorial_effect_rows.jsonl"))
        audits.extend(read_jsonl(root / "group_audit_rows.jsonl"))
    surface_by_group = {
        (row["model"], row["public_parallel_group_id"]): row["surface_private"]
        for row in audits
    }
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        surface = surface_by_group[(row["model"], row["public_parallel_group_id"])]
        grouped[(row["model"], surface, row["layer_index"], row["coordinate"], row["component"])].append(row)
    profile = []
    for (model, surface, layer, coordinate, component), items in sorted(grouped.items()):
        if len(items) != 8:
            raise RuntimeError(f"Phase398 discovery profile cell has {len(items)} groups")
        effect_summary = {}
        for effect in ("R", "O", "Q", "RO", "RQ", "OQ", "ROQ"):
            metrics = [item["factorial_effect_metrics"][effect] for item in items]
            effect_summary[effect] = {
                "median_min_axis_normalized_norm": rounded(median(metric["min_axis_normalized_norm"] for metric in metrics)),
                "support_at_0_005": sum(metric["min_axis_normalized_norm"] >= 0.005 for metric in metrics),
                "support_at_0_02": sum(metric["min_axis_normalized_norm"] >= 0.02 for metric in metrics),
                "median_cross_axis_cosine": rounded(median(metric["cross_axis_cosine"] for metric in metrics)),
                "median_x_normalized_norm": rounded(median(metric["x_normalized_norm"] for metric in metrics)),
                "median_y_normalized_norm": rounded(median(metric["y_normalized_norm"] for metric in metrics)),
            }
        nuisance_ratios = []
        for item in items:
            metrics = item["factorial_effect_metrics"]
            nuisance = max(metrics[name]["min_axis_normalized_norm"] for name in ("RO", "OQ", "ROQ"))
            nuisance_ratios.append(metrics["RQ"]["min_axis_normalized_norm"] / max(nuisance, 1e-12))
        profile.append({
            "schema_version": "72.7.0",
            "phase_id": "Phase398-DiscoveryPhysicalMap",
            "model": model,
            "task_surface": surface,
            "layer_index": layer,
            "relative_depth": items[0]["relative_depth"],
            "coordinate": coordinate,
            "component": component,
            "group_count": len(items),
            "effect_summary": effect_summary,
            "median_rq_to_max_nuisance_interaction_ratio": rounded(median(nuisance_ratios)),
            "causal": False,
            "single_neuron_claim": False,
        })

    primary = [
        row for row in profile
        if row["coordinate"] == "query_end" and row["component"] == "layer_output"
    ]
    cells = []
    for model in MODELS:
        for surface in SURFACES:
            candidates = [row for row in primary if row["model"] == model and row["task_surface"] == surface]
            assessed = []
            for row in candidates:
                rq = row["effect_summary"]["RQ"]
                magnitude = bool(
                    rq["median_min_axis_normalized_norm"] >= gates["minimum_median_min_axis_normalized_rq_norm"]
                    and rq["support_at_0_005"] >= gates["minimum_support_groups_at_0_005"]
                )
                specificity = row["median_rq_to_max_nuisance_interaction_ratio"] >= gates["minimum_median_rq_to_max_nuisance_interaction_ratio"]
                direction = rq["median_cross_axis_cosine"] >= gates["minimum_shared_operation_cross_axis_cosine"]
                assessed.append((row, magnitude, specificity, direction))
            qualified = [item for item in assessed if item[1] and item[2]]
            pool = qualified or assessed
            selected = max(
                pool,
                key=lambda item: item[0]["effect_summary"]["RQ"]["median_min_axis_normalized_norm"]
                * item[0]["median_rq_to_max_nuisance_interaction_ratio"],
            )
            row, magnitude, specificity, direction = selected
            if magnitude and specificity and direction:
                classification = "shared_operation_candidate"
            elif magnitude and specificity:
                classification = "content_conditioned_interaction_candidate"
            else:
                classification = "not_qualified"
            cells.append({
                "model": model,
                "task_surface": surface,
                "selected_layer": row["layer_index"],
                "selected_relative_depth": row["relative_depth"],
                "median_min_axis_normalized_rq_norm": row["effect_summary"]["RQ"]["median_min_axis_normalized_norm"],
                "rq_support_group_count": row["effect_summary"]["RQ"]["support_at_0_005"],
                "median_rq_to_max_nuisance_interaction_ratio": row["median_rq_to_max_nuisance_interaction_ratio"],
                "median_rq_cross_axis_cosine": row["effect_summary"]["RQ"]["median_cross_axis_cosine"],
                "magnitude_gate_pass": magnitude,
                "specificity_gate_pass": specificity,
                "direction_gate_pass": direction,
                "classification": classification,
                "candidate_layer_count": sum(a and b for _, a, b, _ in assessed),
            })
    qualified_count = sum(cell["magnitude_gate_pass"] and cell["specificity_gate_pass"] for cell in cells)
    shared_count = sum(cell["classification"] == "shared_operation_candidate" for cell in cells)
    calibration = qualified_count == 9
    result = {
        "schema_version": "72.7.0",
        "phase_id": "Phase398-DiscoveryAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": protocol["primary_candidate_space"],
        "frozen_gates": gates,
        "denominator": {
            "behavior_case_count": 3456,
            "frozen_trace_case_count": 2304,
            "instrument_case_count": instrument["instrument_case_count"],
            "discovery_case_count": sum(item["case_count"] for item in completes),
            "discovery_parallel_group_count": sum(item["group_count"] for item in completes),
            "model_surface_cell_count": len(cells),
            "groups_per_model_surface": 8,
            "physical_profile_row_count": len(profile),
        },
        "results": {
            "qualified_model_surface_cell_count": qualified_count,
            "shared_operation_candidate_cell_count": shared_count,
            "content_conditioned_candidate_cell_count": sum(cell["classification"] == "content_conditioned_interaction_candidate" for cell in cells),
            "not_qualified_cell_count": sum(cell["classification"] == "not_qualified" for cell in cells),
            "all_discovery_replays_exact": all(item["target_completion_argmax_match_count"] == item["case_count"] for item in completes),
            "all_component_conservation_pass": all(item["all_block_conservation_pass"] for item in completes),
        },
        "cells": cells,
        "interpretation_boundary": {
            "factorial_rq_interaction_is_observational": True,
            "factorial_rq_interaction_proves_binding_algorithm": False,
            "cross_axis_direction_failure_proves_no_joint_process": False,
            "target_completion_margin_encodes_answer_identity": False,
            "complete_language_path_established": False,
            "single_neuron_mechanism_established": False,
        },
        "authorization": {
            "run_calibration_trace": calibration,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
            "single_neuron_scan": False,
        },
    }
    write_jsonl(OUT / "phase398_discovery_physical_map.jsonl", profile)
    write_json(OUT / "phase398_discovery_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
