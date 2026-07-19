#!/usr/bin/env python3
"""Analyze Phase557 natural-color ledgers and freeze source-layer candidates."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
MODELS = ("qwen3", "glm4")
ZONES = ("early", "middle", "late")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def zone(relative_depth: float) -> str:
    if relative_depth < 1.0 / 3.0:
        return "early"
    if relative_depth < 2.0 / 3.0:
        return "middle"
    return "late"


def discovery_color_score(row: dict[str, Any]) -> float:
    discovery = row["split_geometry"]["behavior_discovery"]
    margin = finite(discovery.get("same_minus_different_color_cosine"))
    stability = finite(discovery.get("mean_object_surface_direction_stability"))
    cross_category = finite(discovery.get("cross_category_same_color_cosine_mean"))
    # Discovery-only ranking. Negative margins remain negative rather than being
    # converted into evidence; a candidate is merely a coordinate to test.
    return margin + 0.25 * cross_category + 0.10 * stability


def top_row(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    return max(rows, key=lambda row: (key(row), -int(row["layer"])))


def analyze_model(model: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = OUT_DIR / "natural_color_events" / model / "phase557_natural_color_event_rows.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"Missing Phase557 natural-color events for {model}")
    source_rows = [
        row for row in rows
        if row["component"] == "layer_output" and row["semantic_position"] == "object_source_end"
    ]
    candidates: list[dict[str, Any]] = []
    for zone_name in ZONES:
        available = [row for row in source_rows if zone(float(row["relative_depth"])) == zone_name]
        selected = top_row(available, discovery_color_score)
        layer_count = int(selected["layer_count"])
        selected_layer = int(selected["layer"])
        wrong_layer = (selected_layer + max(2, layer_count // 2)) % layer_count
        if wrong_layer == selected_layer:
            wrong_layer = (selected_layer + 1) % layer_count
        discovery = selected["split_geometry"]["behavior_discovery"]
        confirmation = selected["split_geometry"]["behavior_confirmation"]
        candidates.append({
            "candidate_id": f"{model}__natural_color_object_source__{zone_name}__L{selected_layer}",
            "model": model,
            "relation": "color",
            "source_position": "object_source_end",
            "component": "layer_output",
            "zone": zone_name,
            "layer": selected_layer,
            "wrong_depth_control_layer": wrong_layer,
            "layer_count": layer_count,
            "selection_split": "behavior_discovery",
            "selection_score": discovery_color_score(selected),
            "discovery_same_minus_different_color_cosine": discovery.get(
                "same_minus_different_color_cosine"
            ),
            "confirmation_same_minus_different_color_cosine": confirmation.get(
                "same_minus_different_color_cosine"
            ),
            "discovery_surface_direction_stability": discovery.get(
                "mean_object_surface_direction_stability"
            ),
            "confirmation_surface_direction_stability": confirmation.get(
                "mean_object_surface_direction_stability"
            ),
            "cross_split_category_direction_cosine": selected[
                "cross_split_category_direction_cosine"
            ],
            "mean_cross_split_shared_color_direction_cosine": selected[
                "mean_cross_split_shared_color_direction_cosine"
            ],
            "candidate_is_mechanism_evidence": False,
            "confirmation_used_for_selection": False,
            "sealed_used_for_selection": False,
        })

    position_reports = {}
    for position in ("object_source_end", "relation_request_end", "answer_boundary"):
        position_rows = [row for row in rows if row["semantic_position"] == position]
        best_category = top_row(position_rows, lambda row: finite(row["cross_split_category_direction_cosine"]))
        best_color = top_row(
            position_rows,
            lambda row: finite(row["mean_cross_split_shared_color_direction_cosine"], -1.0),
        )
        position_reports[position] = {
            "best_cross_split_category_coordinate": {
                "layer": best_category["layer"],
                "component": best_category["component"],
                "cosine": best_category["cross_split_category_direction_cosine"],
            },
            "best_cross_split_color_coordinate": {
                "layer": best_color["layer"],
                "component": best_color["component"],
                "mean_shared_color_cosine": best_color[
                    "mean_cross_split_shared_color_direction_cosine"
                ],
            },
        }
    return ({
        "model": model,
        "event_row_count": len(rows),
        "layer_count": int(rows[0]["layer_count"]),
        "position_reports": position_reports,
        "candidate_count": len(candidates),
        "candidate_layers": [row["layer"] for row in candidates],
        "observer_only": True,
        "compute_edge_confirmed": False,
    }, candidates)


def main() -> None:
    reports = []
    candidates = []
    for model in MODELS:
        report, model_candidates = analyze_model(model)
        reports.append(report)
        candidates.extend(model_candidates)
    registry = {
        "schema_version": "phase557_natural_color_source_candidate_registry.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "selection_policy": (
            "one discovery-ranked layer_output object-source coordinate per fixed depth zone; "
            "confirmation and sealed data excluded from selection"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_is_mechanism_evidence": False,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    summary = {
        "schema_version": "phase557_natural_color_event_analysis.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "model_reports": reports,
        "candidate_registry_path": (
            "tests/gpt5/result/phase557_fruit_composite/"
            "phase557_natural_color_source_candidate_registry.json"
        ),
        "causal_claim": False,
        "sealed_split_read": False,
    }
    write_json(OUT_DIR / "phase557_natural_color_source_candidate_registry.json", registry)
    write_json(OUT_DIR / "phase557_natural_color_event_analysis.json", summary)
    print(json.dumps({
        "models": [row["model"] for row in reports],
        "candidate_layers": {
            model: [row["layer"] for row in candidates if row["model"] == model]
            for model in MODELS
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
