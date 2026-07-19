#!/usr/bin/env python3
"""Summarize Phase559 event geometry and freeze coarse source/query candidates."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
EVENT_ROWS = OUT_DIR / "phase559_binding_event_rows.jsonl"
EVENT_SUMMARY = OUT_DIR / "phase559_binding_event_summary.json"
ANALYSIS_PATH = OUT_DIR / "phase559_binding_event_analysis.json"
CANDIDATE_REGISTRY = OUT_DIR / "phase559_binding_candidate_registry.json"
ZONES = ("early", "middle", "late")
BOUNDARIES = {
    "source": "source_fact_end",
    "query": "query_object_end",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
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


def discovery_score(row: dict[str, Any]) -> float:
    metrics = row["split_metrics"]["path_discovery"]
    relative = max(0.0, finite(metrics["mean_relative_binding_delta_norm"]))
    stability = max(0.0, finite(metrics["mean_surface_order_direction_stability"]))
    role = max(0.0, min(1.0, (finite(metrics["mean_query_role_direction_cosine"]) + 1.0) / 2.0))
    return relative * stability * role


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": row["layer"],
        "component": row["component"],
        "semantic_position": row["semantic_position"],
        "discovery": row["split_metrics"]["path_discovery"],
        "confirmation": row["split_metrics"]["path_confirmation"],
        "mean_cross_split_matched_color_pair_direction_cosine": row[
            "mean_cross_split_matched_color_pair_direction_cosine"
        ],
        "discovery_selection_score": discovery_score(row),
    }


def main() -> None:
    event_summary = read_json(EVENT_SUMMARY)
    rows = read_jsonl(EVENT_ROWS)
    if event_summary["status"] != "complete" or len(rows) != 1008:
        raise RuntimeError("Phase559 event ledger is incomplete")
    layer_count = int(event_summary["layer_count"])
    candidates: list[dict[str, Any]] = []
    boundary_reports: dict[str, Any] = {}
    for boundary, position in BOUNDARIES.items():
        available = [
            row for row in rows
            if row["component"] == "layer_output" and row["semantic_position"] == position
        ]
        trajectory = sorted((compact(row) for row in available), key=lambda row: int(row["layer"]))
        boundary_candidates = []
        for zone_name in ZONES:
            zone_rows = [row for row in available if zone(float(row["relative_depth"])) == zone_name]
            selected = max(zone_rows, key=lambda row: (discovery_score(row), -int(row["layer"])))
            selected_layer = int(selected["layer"])
            wrong_layer = (selected_layer + max(2, layer_count // 2)) % layer_count
            wrong_position = "nontarget_fact_end" if boundary == "source" else "query_relation_end"
            candidate = {
                "candidate_id": f"qwen3__fixed_identity_binding__{boundary}__{zone_name}__L{selected_layer}",
                "model": "qwen3",
                "boundary": boundary,
                "semantic_position": position,
                "wrong_position_control": wrong_position,
                "component": "layer_output",
                "zone": zone_name,
                "layer": selected_layer,
                "wrong_depth_control_layer": wrong_layer,
                "layer_count": layer_count,
                "selection_split": "path_discovery",
                "selection_score": discovery_score(selected),
                "discovery_metrics": selected["split_metrics"]["path_discovery"],
                "confirmation_metrics_not_used_for_selection": selected[
                    "split_metrics"
                ]["path_confirmation"],
                "mean_cross_split_matched_color_pair_direction_cosine": selected[
                    "mean_cross_split_matched_color_pair_direction_cosine"
                ],
                "candidate_is_mechanism_evidence": False,
                "confirmation_used_for_selection": False,
                "sealed_used_for_selection": False,
            }
            candidates.append(candidate)
            boundary_candidates.append(candidate)
        peak = max(available, key=discovery_score)
        earliest_stable = next((
            row for row in sorted(available, key=lambda row: int(row["layer"]))
            if finite(row["split_metrics"]["path_discovery"]["mean_relative_binding_delta_norm"]) >= 0.05
            and finite(row["split_metrics"]["path_discovery"]["mean_query_role_direction_cosine"]) >= 0.40
            and finite(row["mean_cross_split_matched_color_pair_direction_cosine"]) >= 0.80
        ), None)
        boundary_reports[boundary] = {
            "semantic_position": position,
            "peak_discovery_coordinate": compact(peak),
            "earliest_stable_coordinate": compact(earliest_stable) if earliest_stable else None,
            "zone_candidates": boundary_candidates,
            "layer_trajectory": trajectory,
        }

    component_reports = {}
    for position in ("source_fact_end", "query_object_end", "answer_boundary"):
        position_rows = [row for row in rows if row["semantic_position"] == position]
        component_reports[position] = {}
        for component in ("layer_input", "attention_output", "mlp_output", "layer_output"):
            component_rows = [row for row in position_rows if row["component"] == component]
            component_reports[position][component] = compact(max(component_rows, key=discovery_score))

    registry = {
        "schema_version": "phase559_binding_candidate_registry.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "selection_policy": (
            "one discovery-only layer_output coordinate per fixed depth zone at the full source-fact "
            "boundary and query-object boundary; confirmation and sealed data excluded from ranking"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_is_mechanism_evidence": False,
        "confirmation_intervention_authorized": True,
        "unseen_intervention_authorized_only_after_confirmation": True,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    analysis = {
        "schema_version": "phase559_binding_event_analysis.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "model": "qwen3",
        "event_row_count": len(rows),
        "layer_count": layer_count,
        "boundary_reports": boundary_reports,
        "component_peak_reports": component_reports,
        "candidate_registry_path": str(CANDIDATE_REGISTRY.relative_to(ROOT)),
        "physical_observations": {
            "source_fact_binding_difference_present_from_early_layers": True,
            "query_binding_difference_is_gradually_formed": True,
            "query_peak_is_mid_depth": True,
            "observation_is_not_causal_evidence": True,
        },
        "causal_claim": False,
        "sealed_split_read": False,
    }
    write_json(CANDIDATE_REGISTRY, registry)
    write_json(ANALYSIS_PATH, analysis)
    print(json.dumps({
        "candidate_count": len(candidates),
        "candidate_layers": {
            boundary: [
                row["layer"] for row in candidates if row["boundary"] == boundary
            ] for boundary in BOUNDARIES
        },
        "source_peak": boundary_reports["source"]["peak_discovery_coordinate"]["layer"],
        "query_earliest_stable": (
            boundary_reports["query"]["earliest_stable_coordinate"] or {}
        ).get("layer"),
        "query_peak": boundary_reports["query"]["peak_discovery_coordinate"]["layer"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
