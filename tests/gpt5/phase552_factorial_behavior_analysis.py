#!/usr/bin/env python3
"""Qualify Phase552 eight-cell behavior before hidden-state observation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase548_shared_attention_compute_protocol import wilson
from phase552_factorial_behavior import rows_path, summary_path
from phase552_surface_route_answer_protocol import (
    CELLS, MODELS, OUT_DIR, PHASE, PROTOCOL_PATH, SPLITS, WORLDS_PER_SPLIT, now,
)


QUALIFICATION_PATH = OUT_DIR / "phase552_behavior_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase552_behavior_summary.json"


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
        len(anchors) == WORLDS_PER_SPLIT
        and exact_lcb >= 0.90
        and bad_ucb <= 0.05
        and all(value["lcb95"] >= 0.90 for value in by_cell.values())
    )
    return {
        "anchor_count": len(anchors),
        "all_cells_exact": {
            "count": exact, "n": len(anchors), "rate": exact / max(len(anchors), 1),
            "lcb95": exact_lcb, "ucb95": exact_ucb,
        },
        "unrecoverable_anchor": {
            "count": unrecoverable, "n": len(anchors),
            "rate": unrecoverable / max(len(anchors), 1), "ucb95": bad_ucb,
        },
        "by_cell": by_cell,
        "gate_pass": gate,
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    contracts = protocol["surface_contracts"]
    execution = {model: read_json(summary_path(model)) for model in MODELS}
    model_rows = {
        model: read_jsonl(rows_path(model)) if rows_path(model).exists() else []
        for model in MODELS
    }
    result_rows = []
    contract_keys = {(row["model"], row["mechanism_id"]): row for row in contracts}
    for model in MODELS:
        for (contract_model, mechanism), contract in sorted(contract_keys.items()):
            if contract_model != model:
                continue
            reports = {
                split: report([
                    row for row in model_rows[model]
                    if row["mechanism_id"] == mechanism and row["split"] == split
                ]) for split in SPLITS
            }
            passed = all(value["gate_pass"] for value in reports.values())
            result_rows.append({
                "schema_version": "phase552_behavior_qualification.v1",
                "phase_id": PHASE,
                "created_at": now(),
                "model": model,
                "family_id": contract["family_id"],
                "mechanism_id": mechanism,
                "surface0_scaffold_id": contract["surface0_scaffold_id"],
                "surface1_scaffold_id": contract["surface1_scaffold_id"],
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
        "schema_version": "phase552_behavior_summary.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "status": "surface_route_answer_behavior_complete",
        "models_in_required_execution_order": list(MODELS),
        "execution": execution,
        "registered_contract_count": len(result_rows),
        "behavior_pass_contract_count": sum(row["behavior_gate_pass"] for row in result_rows),
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
