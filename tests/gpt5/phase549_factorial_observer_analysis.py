#!/usr/bin/env python3
"""Classify Phase549 frozen-window geometry as route- or answer-dominated."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from phase548_matched_observer_analysis import sign_flip_p
from phase549_route_answer_factorial_protocol import MECHANISMS, MODELS, OUT_DIR, SPLITS


RESULT_PATH = OUT_DIR / "phase549_factorial_observer_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase549_factorial_observer_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def report(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    margins = [float(row["route_minus_answer_effect"]) for row in rows]
    route_fraction = sum(value > 0.0 for value in margins) / max(len(margins), 1)
    answer_fraction = sum(value < 0.0 for value in margins) / max(len(margins), 1)
    route_p = sign_flip_p(margins, f"phase549:route:{key}") if margins else 1.0
    answer_p = sign_flip_p([-value for value in margins], f"phase549:answer:{key}") if margins else 1.0
    route_dominant = (
        len(rows) == 73 and median(margins) > 0.0 and route_fraction >= 0.70 and route_p <= 0.01
    ) if margins else False
    answer_dominant = (
        len(rows) == 73 and median(margins) < 0.0 and answer_fraction >= 0.70 and answer_p <= 0.01
    ) if margins else False
    return {
        "n": len(rows),
        "route_effect_median": median([float(row["route_effect"]) for row in rows]) if rows else 0.0,
        "answer_identity_effect_median": median([float(row["answer_identity_effect"]) for row in rows]) if rows else 0.0,
        "route_minus_answer_median": median(margins) if margins else 0.0,
        "route_dominance_fraction": route_fraction,
        "answer_dominance_fraction": answer_fraction,
        "route_one_sided_p": route_p, "answer_one_sided_p": answer_p,
        "classification": "route_dominant" if route_dominant else "answer_identity_dominant" if answer_dominant else "inseparable",
    }


def analyze() -> dict[str, Any]:
    behavior = read_jsonl(OUT_DIR / "phase549_behavior_qualification.jsonl")
    behavior_by_cell = {(row["model"], row["mechanism_id"]): row for row in behavior}
    execution = {
        model: read_json(OUT_DIR / f"phase549_{model}_factorial_observer_execution.json")
        for model in MODELS
    }
    all_rows = []
    for model in MODELS:
        path = OUT_DIR / f"phase549_{model}_factorial_observer_rows.jsonl"
        if path.exists():
            all_rows.extend(read_jsonl(path))
    result_rows = []
    for model in MODELS:
        for mechanism in MECHANISMS:
            behavior_row = behavior_by_cell[(model, mechanism)]
            platform = [
                row for row in all_rows if row["model"] == model
                and row["mechanism_id"] == mechanism
                and row["aggregation"] == "frozen_three_layer_platform"
            ]
            reports = {
                split: report(
                    [row for row in platform if row["split"] == split],
                    f"{model}:{mechanism}:{split}",
                ) for split in SPLITS
            } if platform else {}
            stable_classification = None
            if reports:
                classes = {value["classification"] for value in reports.values()}
                if len(classes) == 1:
                    stable_classification = next(iter(classes))
            result_rows.append({
                "schema_version": "phase549_factorial_observer_qualification.v1",
                "phase_id": "Phase549", "created_at": now(), "model": model,
                "family_id": "content_knowledge", "mechanism_id": mechanism,
                "behavior_gate_pass": behavior_row["behavior_gate_pass"],
                "observer_collection_authorized": behavior_row["observer_collection_authorized"],
                "split_reports": reports, "stable_classification": stable_classification,
                "route_dominant": stable_classification == "route_dominant",
                "answer_identity_dominant": stable_classification == "answer_identity_dominant",
                "compute_intervention_authorized": False, "observer_only": True,
                "compute_edge": False, "causal": False, "single_neuron": False, "sealed": False,
            })
    write_jsonl(RESULT_PATH, result_rows)
    summary = {
        "schema_version": "phase549_factorial_observer_summary.v1", "phase_id": "Phase549",
        "created_at": now(), "status": "route_answer_factorial_complete_observation_only",
        "execution": execution, "cell_count": len(result_rows),
        "observed_cell_count": sum(bool(row["split_reports"]) for row in result_rows),
        "stable_route_dominant_cell_count": sum(row["route_dominant"] for row in result_rows),
        "stable_answer_identity_dominant_cell_count": sum(row["answer_identity_dominant"] for row in result_rows),
        "stable_inseparable_cell_count": sum(row["stable_classification"] == "inseparable" for row in result_rows),
        "compute_intervention_authorized_cell_count": 0,
        "compute_edges": 0, "causal_edges": 0, "new_sealed_split_read": False,
        "head_channel_neuron_scan_executed": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
