#!/usr/bin/env python3
"""Analyze the Phase548 behavior gate without opening the physical stage."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase548_shared_attention_compute_protocol import (
    MECHANISMS,
    MODELS,
    OUT_DIR,
    PAIR_UNITS_PER_SPLIT,
    SPLITS,
    VARIANTS,
    wilson,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = OUT_DIR / "phase548_behavior_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase548_behavior_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def split_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        anchors[row["anchor_id"]].append(row)
    exact = sum(
        len(group) == len(VARIANTS)
        and {row["variant"] for row in group} == set(VARIANTS)
        and all(row["semantic_correct"] for row in group)
        for group in anchors.values()
    )
    unrecoverable = sum(any(not row["semantic_event_recoverable"] for row in group) for group in anchors.values())
    lcb, exact_ucb = wilson(exact, len(anchors))
    unrecoverable_lcb, unrecoverable_ucb = wilson(unrecoverable, len(anchors))
    by_variant = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        count = sum(row["semantic_correct"] for row in selected)
        variant_lcb, variant_ucb = wilson(count, len(selected))
        by_variant[variant] = {
            "count": count, "n": len(selected), "rate": count / max(len(selected), 1),
            "lcb95": variant_lcb, "ucb95": variant_ucb,
        }
    return {
        "anchor_count": len(anchors),
        "all_variants_exact": {
            "count": exact, "n": len(anchors), "rate": exact / max(len(anchors), 1),
            "lcb95": lcb, "ucb95": exact_ucb,
        },
        "unrecoverable_anchor": {
            "count": unrecoverable, "n": len(anchors),
            "rate": unrecoverable / max(len(anchors), 1),
            "lcb95": unrecoverable_lcb, "ucb95": unrecoverable_ucb,
        },
        "by_variant": by_variant,
        "gate_pass": (
            len(anchors) == PAIR_UNITS_PER_SPLIT
            and lcb >= 0.90
            and unrecoverable_ucb <= 0.05
            and all(report["lcb95"] >= 0.90 for report in by_variant.values())
        ),
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(OUT_DIR / "phase548_frozen_protocol.json")
    rows_out = []
    execution = {}
    for model in MODELS:
        execution_path = OUT_DIR / f"phase548_{model}_behavior_execution.json"
        rows_path = OUT_DIR / f"phase548_{model}_behavior_rows.jsonl"
        if not execution_path.exists() or not rows_path.exists():
            raise RuntimeError(f"Missing completed Phase548 behavior run for {model}")
        execution[model] = read_json(execution_path)
        model_rows = read_jsonl(rows_path)
        for mechanism in MECHANISMS:
            reports = {
                split: split_report([
                    row for row in model_rows
                    if row["mechanism_id"] == mechanism and row["split"] == split
                ])
                for split in SPLITS
            }
            eligible = all(report["gate_pass"] for report in reports.values())
            phase546_window_registered = bool(protocol["frozen_windows"][model]["target_layers"])
            rows_out.append({
                "schema_version": "phase548_behavior_qualification.v1",
                "phase_id": "Phase548",
                "created_at": now(),
                "model": model,
                "family_id": "content_knowledge",
                "mechanism_id": mechanism,
                "behavior_gate_pass": eligible,
                "split_reports": reports,
                "phase546_window_registered": phase546_window_registered,
                "observer_collection_authorized": eligible and phase546_window_registered,
                "observer_stop_reason": (
                    "authorized by behavior and Phase546 frozen-window gates"
                    if eligible and phase546_window_registered
                    else "no Phase546 registered window" if eligible
                    else "matched natural behavior gate failed"
                ),
                "compute_intervention_authorized": False,
                "compute_edge": False,
                "causal": False,
                "single_neuron": False,
                "sealed": False,
            })
    write_jsonl(RESULT_PATH, rows_out)
    summary = {
        "schema_version": "phase548_behavior_summary.v1",
        "phase_id": "Phase548",
        "created_at": now(),
        "status": "behavior_gate_complete_observer_not_yet_run",
        "models_in_required_execution_order": list(MODELS),
        "execution": execution,
        "model_mechanism_cell_count": len(rows_out),
        "behavior_pass_cell_count": sum(row["behavior_gate_pass"] for row in rows_out),
        "observer_authorized_cell_count": sum(row["observer_collection_authorized"] for row in rows_out),
        "deepseek_observer_authorized": any(
            row["observer_collection_authorized"] for row in rows_out if row["model"] == "deepseek7b"
        ),
        "new_sealed_split_read": False,
        "compute_edges": 0,
        "causal_edges": 0,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
