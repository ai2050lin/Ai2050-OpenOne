#!/usr/bin/env python3
"""Confirm full-layer Phase551 route-dominant observer events without intervention."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from phase551_full_layer_route_observer import COMPONENTS, ROLES, output_path, summary_path
from phase551_model_specific_behavior_analysis import QUALIFICATION_PATH
from phase551_model_specific_route_protocol import MECHANISMS, MODELS, OUT_DIR, SPLITS, now


EVENTS_PATH = OUT_DIR / "phase551_confirmed_route_observer_events.jsonl"
WINDOWS_PATH = OUT_DIR / "phase551_confirmed_route_windows.json"
SUMMARY_PATH = OUT_DIR / "phase551_full_layer_route_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def event_report(rows: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    metrics = [row["features"][feature_key] for row in rows]
    margins = [float(value["route_minus_answer_effect"]) for value in metrics]
    ratios = [float(value["route_to_answer_ratio"]) for value in metrics]
    route_effects = [float(value["route_effect"]) for value in metrics]
    answer_effects = [float(value["answer_identity_effect"]) for value in metrics]
    route_fraction = sum(value > 0.0 for value in margins) / max(len(margins), 1)
    report = {
        "n": len(rows),
        "route_effect_median": median(route_effects) if metrics else 0.0,
        "answer_identity_effect_median": median(answer_effects) if metrics else 0.0,
        "route_minus_answer_median": median(margins) if margins else 0.0,
        "route_to_answer_ratio_median": median(ratios) if ratios else 0.0,
        "route_dominance_fraction": route_fraction,
        "route_direction_alignment_median": median([
            float(value["route_direction_alignment"]) for value in metrics
        ]) if metrics else 0.0,
    }
    report["route_dominant_gate_pass"] = bool(
        len(rows) == 73
        and report["route_effect_median"] >= 0.02
        and report["route_to_answer_ratio_median"] >= 1.10
        and route_fraction >= 0.70
    )
    return report


def depth_band(relative_depth: float) -> str:
    if relative_depth < 1.0 / 3.0:
        return "early"
    if relative_depth < 2.0 / 3.0:
        return "middle"
    return "late"


def contiguous_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(event["model"], event["mechanism_id"], event["feature_key"])].append(event)
    windows = []
    for (model, mechanism, feature_key), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row["layer"])
        runs = []
        current = []
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
                "schema_version": "phase551_confirmed_route_window.v1",
                "phase_id": "Phase551",
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
                "median_confirmation_route_to_answer_ratio": median([
                    row["independent_confirmation"]["route_to_answer_ratio_median"] for row in run
                ]),
                "observer_only": True,
                "compute_edge": False,
                "causal": False,
                "single_neuron": False,
                "sealed": False,
            })
    return sorted(
        windows,
        key=lambda row: (
            row["model"], row["mechanism_id"], -row["layer_count"], row["feature_key"], row["start_layer"]
        ),
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
    tested_event_count = 0
    discovery_candidate_count = 0
    for model, mechanism in sorted(authorized):
        model_rows = [row for row in rows_by_model[model] if row["mechanism_id"] == mechanism]
        layers = sorted({row["layer"] for row in model_rows})
        for layer in layers:
            for component in COMPONENTS:
                for role in ROLES:
                    feature_key = f"{component}__{role}"
                    tested_event_count += 1
                    discovery_rows = [
                        row for row in model_rows
                        if row["split"] == "discovery" and row["layer"] == layer
                    ]
                    discovery = event_report(discovery_rows, feature_key)
                    if not discovery["route_dominant_gate_pass"]:
                        continue
                    discovery_candidate_count += 1
                    confirmation_rows = [
                        row for row in model_rows
                        if row["split"] == "independent_confirmation" and row["layer"] == layer
                    ]
                    confirmation = event_report(confirmation_rows, feature_key)
                    if not confirmation["route_dominant_gate_pass"]:
                        continue
                    relative_depth = confirmation_rows[0]["relative_depth"]
                    events.append({
                        "schema_version": "phase551_confirmed_route_observer_event.v1",
                        "phase_id": "Phase551",
                        "created_at": now(),
                        "model": model,
                        "mechanism_id": mechanism,
                        "family_id": confirmation_rows[0]["family_id"],
                        "selected_scaffold_id": confirmation_rows[0]["selected_scaffold_id"],
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
    shared_keys: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for event in events:
        shared_keys[(event["mechanism_id"], event["feature_key"], event["depth_band"])].add(event["model"])
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
        for (mechanism, feature_key, band), models in sorted(shared_keys.items())
        if len(models) >= 2
    ]
    write_jsonl(EVENTS_PATH, events)
    write_json(WINDOWS_PATH, {
        "schema_version": "phase551_confirmed_route_windows.v1",
        "phase_id": "Phase551",
        "created_at": now(),
        "windows": windows,
        "shared_cross_model_topologies": shared_topologies,
    })
    summary = {
        "schema_version": "phase551_full_layer_route_summary.v1",
        "phase_id": "Phase551",
        "created_at": now(),
        "status": "full_layer_route_observation_complete",
        "execution": execution,
        "authorized_contract_count": len(authorized),
        "tested_layer_feature_event_count": tested_event_count,
        "discovery_route_candidate_count": discovery_candidate_count,
        "independently_confirmed_route_event_count": len(events),
        "confirmed_contiguous_window_count": len(windows),
        "shared_cross_model_topology_count": len(shared_topologies),
        "max_component_ledger_relative_error": max(
            (value.get("max_component_ledger_relative_error", 0.0) for value in execution.values()),
            default=0.0,
        ),
        "full_hidden_vectors_persisted": False,
        "head_channel_neuron_scan_executed": False,
        "compute_edges": 0,
        "causal_edges": 0,
        "single_neuron_mechanisms": 0,
        "new_sealed_split_read": False,
        "interpretation": (
            "Confirmed events are route-dominant observer coordinates under model-specific contracts. "
            "Route still mixes wording, relation assignment, query polarity, and structural complexity."
        ),
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
