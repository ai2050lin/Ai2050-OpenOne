#!/usr/bin/env python3
"""Confirm Phase552 semantic-route events against surface and answer controls."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from phase551_full_layer_route_analysis import depth_band
from phase551_full_layer_route_observer import COMPONENTS, ROLES
from phase552_factorial_behavior_analysis import QUALIFICATION_PATH
from phase552_full_layer_factorial_observer import output_path, summary_path
from phase552_surface_route_answer_protocol import MODELS, OUT_DIR, now


EVENTS_PATH = OUT_DIR / "phase552_confirmed_semantic_route_events.jsonl"
WINDOWS_PATH = OUT_DIR / "phase552_confirmed_semantic_route_windows.json"
SUMMARY_PATH = OUT_DIR / "phase552_full_layer_factorial_summary.json"
PHASE551_SUMMARY_PATH = (
    OUT_DIR.parent / "phase551_model_specific_route/phase551_full_layer_route_summary.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def event_report(rows: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    metrics = [row["features"][feature_key] for row in rows]
    margins = [float(value["route_minus_max_control"]) for value in metrics]
    report = {
        "n": len(rows),
        "semantic_route_effect_median": median([
            float(value["semantic_route_effect"]) for value in metrics
        ]) if metrics else 0.0,
        "surface_form_effect_median": median([
            float(value["surface_form_effect"]) for value in metrics
        ]) if metrics else 0.0,
        "answer_identity_effect_median": median([
            float(value["answer_identity_effect"]) for value in metrics
        ]) if metrics else 0.0,
        "route_minus_max_control_median": median(margins) if margins else 0.0,
        "route_to_max_control_ratio_median": median([
            float(value["route_to_max_control_ratio"]) for value in metrics
        ]) if metrics else 0.0,
        "route_dominance_fraction": sum(value > 0.0 for value in margins) / max(len(margins), 1),
    }
    report["semantic_route_gate_pass"] = bool(
        len(rows) == 73
        and report["semantic_route_effect_median"] >= 0.02
        and report["route_to_max_control_ratio_median"] >= 1.10
        and report["route_dominance_fraction"] >= 0.70
    )
    return report


def contiguous_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(event["model"], event["mechanism_id"], event["feature_key"])].append(event)
    windows = []
    for (model, mechanism, feature_key), values in grouped.items():
        ordered = sorted(values, key=lambda row: row["layer"])
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in ordered:
            if current and row["layer"] != current[-1]["layer"] + 1:
                runs.append(current)
                current = []
            current.append(row)
        if current:
            runs.append(current)
        for run in runs:
            if len(run) < 2:
                continue
            windows.append({
                "schema_version": "phase552_confirmed_semantic_route_window.v1",
                "phase_id": "Phase552",
                "created_at": now(),
                "model": model,
                "mechanism_id": mechanism,
                "feature_key": feature_key,
                "component": feature_key.split("__", 1)[0],
                "role": feature_key.split("__", 1)[1],
                "start_layer": run[0]["layer"],
                "end_layer": run[-1]["layer"],
                "layer_count": len(run),
                "start_relative_depth": run[0]["relative_depth"],
                "end_relative_depth": run[-1]["relative_depth"],
                "depth_bands": sorted({row["depth_band"] for row in run}),
                "minimum_confirmation_route_fraction": min(
                    row["independent_confirmation"]["route_dominance_fraction"] for row in run
                ),
                "median_confirmation_route_to_max_control_ratio": median([
                    row["independent_confirmation"]["route_to_max_control_ratio_median"] for row in run
                ]),
                "observer_only": True,
                "compute_edge": False,
                "causal": False,
                "single_neuron": False,
                "sealed": False,
            })
    return sorted(
        windows,
        key=lambda row: (row["model"], row["mechanism_id"], -row["layer_count"], row["feature_key"]),
    )


def gate_progress(report: dict[str, Any]) -> float:
    """Return the weakest normalized gate margin for near-miss ranking."""
    return min(
        report["semantic_route_effect_median"] / 0.02,
        report["route_to_max_control_ratio_median"] / 1.10,
        report["route_dominance_fraction"] / 0.70,
    )


def analyze() -> dict[str, Any]:
    qualification = read_jsonl(QUALIFICATION_PATH)
    authorized = {
        (row["model"], row["mechanism_id"])
        for row in qualification if row["observer_collection_authorized"]
    }
    execution = {model: read_json(summary_path(model)) for model in MODELS}
    rows_by_model = {
        model: read_jsonl(output_path(model)) if output_path(model).exists() else []
        for model in MODELS
    }
    events = []
    evaluated_coordinates = []
    tested_count = 0
    discovery_count = 0
    for model, mechanism in sorted(authorized):
        selected = [row for row in rows_by_model[model] if row["mechanism_id"] == mechanism]
        for layer in sorted({row["layer"] for row in selected}):
            for component in COMPONENTS:
                for role in ROLES:
                    tested_count += 1
                    feature_key = f"{component}__{role}"
                    discovery_rows = [
                        row for row in selected if row["split"] == "discovery" and row["layer"] == layer
                    ]
                    discovery = event_report(discovery_rows, feature_key)
                    confirmation_rows = [
                        row for row in selected
                        if row["split"] == "independent_confirmation" and row["layer"] == layer
                    ]
                    confirmation = event_report(confirmation_rows, feature_key)
                    relative_depth = confirmation_rows[0]["relative_depth"]
                    evaluated_coordinates.append({
                        "model": model,
                        "mechanism_id": mechanism,
                        "layer": layer,
                        "relative_depth": relative_depth,
                        "depth_band": depth_band(relative_depth),
                        "feature_key": feature_key,
                        "component": component,
                        "role": role,
                        "discovery": discovery,
                        "independent_confirmation": confirmation,
                        "minimum_gate_progress": min(
                            gate_progress(discovery), gate_progress(confirmation)
                        ),
                    })
                    if not discovery["semantic_route_gate_pass"]:
                        continue
                    discovery_count += 1
                    if not confirmation["semantic_route_gate_pass"]:
                        continue
                    events.append({
                        "schema_version": "phase552_confirmed_semantic_route_event.v1",
                        "phase_id": "Phase552",
                        "created_at": now(),
                        "model": model,
                        "family_id": confirmation_rows[0]["family_id"],
                        "mechanism_id": mechanism,
                        "surface0_scaffold_id": confirmation_rows[0]["surface0_scaffold_id"],
                        "surface1_scaffold_id": confirmation_rows[0]["surface1_scaffold_id"],
                        "stage": "prompt_end",
                        "layer": layer,
                        "layer_count": confirmation_rows[0]["layer_count"],
                        "relative_depth": relative_depth,
                        "depth_band": depth_band(relative_depth),
                        "feature_key": feature_key,
                        "component": component,
                        "role": role,
                        "discovery": discovery,
                        "independent_confirmation": confirmation,
                        "physical": True,
                        "predictive": True,
                        "observer_only": True,
                        "compute_edge": False,
                        "causal": False,
                        "single_neuron": False,
                        "sealed": False,
                    })
    windows = contiguous_windows(events)
    shared: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for event in events:
        shared[(event["mechanism_id"], event["feature_key"], event["depth_band"])].add(event["model"])
    shared_topologies = [
        {
            "mechanism_id": mechanism,
            "feature_key": feature_key,
            "depth_band": band,
            "models": sorted(models),
            "model_count": len(models),
            "observer_only": True,
            "compute_edge": False,
            "causal": False,
        }
        for (mechanism, feature_key, band), models in sorted(shared.items()) if len(models) >= 2
    ]
    write_jsonl(EVENTS_PATH, events)
    write_json(WINDOWS_PATH, {
        "schema_version": "phase552_confirmed_semantic_route_windows.v1",
        "phase_id": "Phase552",
        "created_at": now(),
        "windows": windows,
        "shared_cross_model_topologies": shared_topologies,
    })
    near_misses = sorted(
        evaluated_coordinates,
        key=lambda row: (
            row["minimum_gate_progress"],
            min(
                row["discovery"]["route_minus_max_control_median"],
                row["independent_confirmation"]["route_minus_max_control_median"],
            ),
        ),
        reverse=True,
    )[:24]
    contract_overview = []
    for model, mechanism in sorted(authorized):
        coordinates = [
            row for row in evaluated_coordinates
            if row["model"] == model and row["mechanism_id"] == mechanism
        ]
        contract_overview.append({
            "model": model,
            "mechanism_id": mechanism,
            "tested_coordinate_count": len(coordinates),
            "best_minimum_gate_progress": max(
                row["minimum_gate_progress"] for row in coordinates
            ),
            "maximum_confirmation_route_to_control_ratio": max(
                row["independent_confirmation"]["route_to_max_control_ratio_median"]
                for row in coordinates
            ),
            "maximum_confirmation_route_dominance_fraction": max(
                row["independent_confirmation"]["route_dominance_fraction"]
                for row in coordinates
            ),
            "confirmed_event_count": sum(
                row["discovery"]["semantic_route_gate_pass"]
                and row["independent_confirmation"]["semantic_route_gate_pass"]
                for row in coordinates
            ),
        })
    phase551 = read_json(PHASE551_SUMMARY_PATH)
    if events:
        interpretation = (
            "Confirmed semantic-route coordinates exceeded independently varied surface-form and "
            "answer-identity controls in both splits. They remain observational and the route factor "
            "still includes relation assignment plus query-role changes."
        )
        stop_reason = "confirmed_observer_coordinates_remain_noncausal"
    else:
        interpretation = (
            "No layer-component-role coordinate exceeded independently varied surface-form and "
            "answer-identity controls in both splits. The 1053 Phase551 events were therefore "
            "contrast-confounded observations, not an intervention search space."
        )
        stop_reason = "zero_controlled_semantic_route_candidates"
    summary = {
        "schema_version": "phase552_full_layer_factorial_summary.v1",
        "phase_id": "Phase552",
        "created_at": now(),
        "status": "surface_route_answer_factorial_observation_complete",
        "execution": execution,
        "authorized_contract_count": len(authorized),
        "tested_layer_feature_event_count": tested_count,
        "discovery_semantic_route_candidate_count": discovery_count,
        "independently_confirmed_semantic_route_event_count": len(events),
        "confirmed_contiguous_window_count": len(windows),
        "shared_cross_model_topology_count": len(shared_topologies),
        "contract_overview": contract_overview,
        "top_near_miss_coordinates": near_misses,
        "phase551_uncontrolled_confirmed_event_count": phase551["independently_confirmed_route_event_count"],
        "event_count_reduction_from_phase551": (
            phase551["independently_confirmed_route_event_count"] - len(events)
        ),
        "max_component_ledger_relative_error": max(
            (value.get("max_component_ledger_relative_error", 0.0) for value in execution.values()),
            default=0.0,
        ),
        "full_hidden_vectors_persisted": False,
        "head_channel_neuron_scan_executed": False,
        "compute_edges": 0,
        "causal_edges": 0,
        "single_neuron_mechanisms": 0,
        "intervention_authorized": False,
        "new_sealed_split_read": False,
        "stop_reason": stop_reason,
        "interpretation": interpretation,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
