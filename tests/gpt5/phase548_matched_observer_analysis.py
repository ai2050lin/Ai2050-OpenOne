#!/usr/bin/env python3
"""Apply the frozen matched-control gate to Phase548 observer geometry."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from phase548_shared_attention_compute_protocol import MECHANISMS, MODELS, OUT_DIR, SPLITS


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = OUT_DIR / "phase548_matched_observer_qualification.jsonl"
SUMMARY_PATH = OUT_DIR / "phase548_matched_observer_summary.json"
CONTROLS = ("identity_delta", "answer_token_delta", "template_delta")
PERMUTATIONS = 1024


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


def sign_flip_p(margins: list[float], key: str) -> float:
    observed = sum(margins) / max(len(margins), 1)
    if observed <= 0.0:
        return 1.0
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(PERMUTATIONS):
        value = sum(margin if rng.getrandbits(1) else -margin for margin in margins) / len(margins)
        exceed += value >= observed
    return (exceed + 1) / (PERMUTATIONS + 1)


def split_report(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    functional = [float(row["functional_delta"]) for row in rows]
    reports = {}
    for control in CONTROLS:
        values = [float(row[control]) for row in rows]
        margins = [left - right for left, right in zip(functional, values)]
        dominance = sum(margin > 0.0 for margin in margins) / max(len(margins), 1)
        p_value = sign_flip_p(margins, f"{key}:{control}")
        reports[control] = {
            "control_median": median(values) if values else 0.0,
            "functional_minus_control_median": median(margins) if margins else 0.0,
            "functional_dominance_fraction": dominance,
            "one_sided_sign_flip_p": p_value,
            "gate_pass": (
                bool(functional)
                and median(functional) > median(values)
                and dominance >= 0.70
                and p_value <= 0.01
            ),
        }
    return {
        "n": len(rows),
        "functional_delta_median": median(functional) if functional else 0.0,
        "controls": reports,
        "gate_pass": len(rows) == 73 and all(report["gate_pass"] for report in reports.values()),
    }


def analyze() -> dict[str, Any]:
    behavior = read_jsonl(OUT_DIR / "phase548_behavior_qualification.jsonl")
    behavior_by_cell = {(row["model"], row["mechanism_id"]): row for row in behavior}
    execution = {
        model: read_json(OUT_DIR / f"phase548_{model}_matched_observer_execution.json")
        for model in MODELS
    }
    result_rows = []
    all_observer_rows = []
    for model in MODELS:
        path = OUT_DIR / f"phase548_{model}_matched_observer_rows.jsonl"
        if path.exists():
            all_observer_rows.extend(read_jsonl(path))
    for model in MODELS:
        for mechanism in MECHANISMS:
            behavior_row = behavior_by_cell[(model, mechanism)]
            platform = [
                row for row in all_observer_rows
                if row["model"] == model and row["mechanism_id"] == mechanism
                and row["aggregation"] == "frozen_three_layer_platform"
            ]
            reports = {
                split: split_report(
                    [row for row in platform if row["split"] == split],
                    f"{model}:{mechanism}:{split}",
                )
                for split in SPLITS
            } if platform else {}
            gate_pass = (
                behavior_row["observer_collection_authorized"]
                and bool(reports)
                and all(report["gate_pass"] for report in reports.values())
            )
            failed_controls = sorted({
                control
                for report in reports.values()
                for control, control_report in report["controls"].items()
                if not control_report["gate_pass"]
            }) if reports else []
            result_rows.append({
                "schema_version": "phase548_matched_observer_qualification.v1",
                "phase_id": "Phase548", "created_at": now(), "model": model,
                "family_id": "content_knowledge", "mechanism_id": mechanism,
                "behavior_gate_pass": behavior_row["behavior_gate_pass"],
                "observer_collection_authorized": behavior_row["observer_collection_authorized"],
                "split_reports": reports,
                "matched_observer_gate_pass": gate_pass,
                "failed_control_axes": failed_controls,
                "compute_intervention_authorized": gate_pass,
                "compute_edge": False, "causal": False, "single_neuron": False,
                "status": (
                    "matched_observer_pass_intervention_authorized" if gate_pass
                    else "matched_observer_stop_control_or_reconfirmation_failed"
                    if behavior_row["observer_collection_authorized"]
                    else "stopped_before_observer"
                ),
                "sealed": False,
            })
    write_jsonl(RESULT_PATH, result_rows)
    shared_observer = {
        mechanism: all(
            next(row for row in result_rows if row["model"] == model and row["mechanism_id"] == mechanism)["matched_observer_gate_pass"]
            for model in ("qwen3", "glm4")
        )
        for mechanism in MECHANISMS
    }
    summary = {
        "schema_version": "phase548_matched_observer_summary.v1",
        "phase_id": "Phase548", "created_at": now(),
        "status": (
            "observer_gate_complete_intervention_partially_authorized"
            if any(row["compute_intervention_authorized"] for row in result_rows)
            else "observer_gate_complete_current_shared_compute_route_closed_before_intervention"
        ),
        "execution": execution,
        "cell_count": len(result_rows),
        "matched_observer_pass_cell_count": sum(row["matched_observer_gate_pass"] for row in result_rows),
        "intervention_authorized_cell_count": sum(row["compute_intervention_authorized"] for row in result_rows),
        "shared_matched_observer_mechanisms": [key for key, passed in shared_observer.items() if passed],
        "permutation_count_per_control": PERMUTATIONS,
        "full_hidden_vectors_persisted": False,
        "head_channel_neuron_scan_executed": False,
        "new_sealed_split_read": False,
        "compute_edges": 0, "causal_edges": 0,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
