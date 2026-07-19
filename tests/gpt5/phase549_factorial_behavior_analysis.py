#!/usr/bin/env python3
"""Analyze Phase549 factorial behavior by independent world anchor."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase549_route_answer_factorial_protocol import (
    CELLS, MECHANISMS, MODELS, OUT_DIR, PAIR_UNITS_PER_SPLIT, SPLITS, WINDOWS, wilson,
)


RESULT_PATH = OUT_DIR / "phase549_behavior_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase549_behavior_summary.json"


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


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        anchors[row["anchor_id"]].append(row)
    exact = sum(
        len(group) == len(CELLS)
        and {row["factorial_cell"] for row in group} == set(CELLS)
        and all(row["semantic_correct"] for row in group)
        for group in anchors.values()
    )
    unrecoverable = sum(any(not row["semantic_event_recoverable"] for row in group) for group in anchors.values())
    lcb, ucb = wilson(exact, len(anchors))
    _bad_lcb, bad_ucb = wilson(unrecoverable, len(anchors))
    by_cell = {}
    for cell in CELLS:
        selected = [row for row in rows if row["factorial_cell"] == cell]
        count = sum(row["semantic_correct"] for row in selected)
        cell_lcb, cell_ucb = wilson(count, len(selected))
        by_cell[cell] = {
            "count": count, "n": len(selected), "rate": count / max(len(selected), 1),
            "lcb95": cell_lcb, "ucb95": cell_ucb,
        }
    return {
        "anchor_count": len(anchors),
        "all_cells_exact": {"count": exact, "n": len(anchors), "rate": exact / max(len(anchors), 1), "lcb95": lcb, "ucb95": ucb},
        "unrecoverable_anchor": {"count": unrecoverable, "n": len(anchors), "rate": unrecoverable / max(len(anchors), 1), "ucb95": bad_ucb},
        "by_cell": by_cell,
        "gate_pass": (
            len(anchors) == PAIR_UNITS_PER_SPLIT and lcb >= 0.90 and bad_ucb <= 0.05
            and all(value["lcb95"] >= 0.90 for value in by_cell.values())
        ),
    }


def analyze() -> dict[str, Any]:
    rows_out = []
    execution = {}
    for model in MODELS:
        execution[model] = read_json(OUT_DIR / f"phase549_{model}_behavior_execution.json")
        model_rows = read_jsonl(OUT_DIR / f"phase549_{model}_behavior_rows.jsonl")
        for mechanism in MECHANISMS:
            reports = {
                split: report([
                    row for row in model_rows
                    if row["mechanism_id"] == mechanism and row["split"] == split
                ]) for split in SPLITS
            }
            passed = all(value["gate_pass"] for value in reports.values())
            registered_window = bool(WINDOWS[model]["target_layers"])
            rows_out.append({
                "schema_version": "phase549_behavior_qualification.v1", "phase_id": "Phase549",
                "created_at": now(), "model": model, "family_id": "content_knowledge",
                "mechanism_id": mechanism, "split_reports": reports,
                "behavior_gate_pass": passed, "phase546_window_registered": registered_window,
                "observer_collection_authorized": passed and registered_window,
                "compute_intervention_authorized": False, "observer_only": True,
                "compute_edge": False, "causal": False, "single_neuron": False, "sealed": False,
            })
    write_jsonl(RESULT_PATH, rows_out)
    summary = {
        "schema_version": "phase549_behavior_summary.v1", "phase_id": "Phase549",
        "created_at": now(), "status": "factorial_behavior_complete",
        "models_in_required_execution_order": list(MODELS), "execution": execution,
        "cell_count": len(rows_out), "behavior_pass_cell_count": sum(row["behavior_gate_pass"] for row in rows_out),
        "observer_authorized_cell_count": sum(row["observer_collection_authorized"] for row in rows_out),
        "compute_edges": 0, "causal_edges": 0, "new_sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
