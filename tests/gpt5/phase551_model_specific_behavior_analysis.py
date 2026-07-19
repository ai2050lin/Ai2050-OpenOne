#!/usr/bin/env python3
"""Qualify independently validated Phase551 model-specific route contracts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase548_shared_attention_compute_protocol import wilson
from phase551_model_specific_route_protocol import (
    CELLS,
    FROZEN_SCAFFOLDS_PATH,
    MECHANISMS,
    MODELS,
    OUT_DIR,
    PHASE,
    SPLITS,
    VALIDATION_WORLDS,
    now,
    read_jsonl,
    write_json,
    write_jsonl,
)


QUALIFICATION_PATH = OUT_DIR / "phase551_validation_behavior_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase551_validation_behavior_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    unrecoverable = sum(
        any(not row["semantic_event_recoverable"] for row in group)
        for group in anchors.values()
    )
    exact_lcb, exact_ucb = wilson(exact, len(anchors))
    _bad_lcb, bad_ucb = wilson(unrecoverable, len(anchors))
    by_cell = {}
    for cell in CELLS:
        selected = [row for row in rows if row["factorial_cell"] == cell]
        count = sum(row["semantic_correct"] for row in selected)
        lcb, ucb = wilson(count, len(selected))
        by_cell[cell] = {
            "count": count,
            "n": len(selected),
            "rate": count / max(len(selected), 1),
            "lcb95": lcb,
            "ucb95": ucb,
        }
    gate = bool(
        len(anchors) == VALIDATION_WORLDS
        and exact_lcb >= 0.90
        and bad_ucb <= 0.05
        and all(value["lcb95"] >= 0.90 for value in by_cell.values())
    )
    return {
        "anchor_count": len(anchors),
        "all_cells_exact": {
            "count": exact,
            "n": len(anchors),
            "rate": exact / max(len(anchors), 1),
            "lcb95": exact_lcb,
            "ucb95": exact_ucb,
        },
        "unrecoverable_anchor": {
            "count": unrecoverable,
            "n": len(anchors),
            "rate": unrecoverable / max(len(anchors), 1),
            "ucb95": bad_ucb,
        },
        "by_cell": by_cell,
        "gate_pass": gate,
    }


def analyze() -> dict[str, Any]:
    frozen = read_json(FROZEN_SCAFFOLDS_PATH)
    selection_map = {
        (row["model"], row["mechanism_id"]): row for row in frozen["selections"]
    }
    execution = {
        model: read_json(OUT_DIR / f"phase551_validation_{model}_behavior_execution.json")
        for model in MODELS
    }
    model_rows = {}
    for model in MODELS:
        path = OUT_DIR / f"phase551_validation_{model}_behavior_rows.jsonl"
        model_rows[model] = read_jsonl(path) if path.exists() else []
    result_rows = []
    for model in MODELS:
        for mechanism in MECHANISMS:
            selection = selection_map[(model, mechanism)]
            reports = {}
            if selection["validation_authorized"]:
                reports = {
                    split: report([
                        row for row in model_rows[model]
                        if row["mechanism_id"] == mechanism and row["split"] == split
                    ])
                    for split in SPLITS
                }
            passed = bool(reports) and all(value["gate_pass"] for value in reports.values())
            result_rows.append({
                "schema_version": "phase551_validation_behavior_qualification.v1",
                "phase_id": PHASE,
                "created_at": now(),
                "model": model,
                "family_id": selection["family_id"],
                "mechanism_id": mechanism,
                "selected_scaffold_id": selection["selected_scaffold_id"],
                "calibration_gate_pass": selection["calibration_gate_pass"],
                "validation_executed": bool(reports),
                "split_reports": reports,
                "behavior_gate_pass": passed,
                "observer_collection_authorized": passed,
                "compute_intervention_authorized": False,
                "observer_only": True,
                "compute_edge": False,
                "causal": False,
                "single_neuron": False,
                "sealed": False,
            })
    write_jsonl(QUALIFICATION_PATH, result_rows)
    summary = {
        "schema_version": "phase551_validation_behavior_summary.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "status": "model_specific_validation_complete",
        "models_in_required_execution_order": list(MODELS),
        "execution": execution,
        "registered_contract_count": len(result_rows),
        "calibration_pass_contract_count": sum(row["calibration_gate_pass"] for row in result_rows),
        "validation_pass_contract_count": sum(row["behavior_gate_pass"] for row in result_rows),
        "observer_authorized_contract_count": sum(row["observer_collection_authorized"] for row in result_rows),
        "compute_edges": 0,
        "causal_edges": 0,
        "single_neuron_mechanisms": 0,
        "new_sealed_split_read": False,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
