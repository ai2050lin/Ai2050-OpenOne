#!/usr/bin/env python3
"""Analyze Phase565 distributed residual-state sufficiency."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator"
CONTRACT_PATH = OUT_DIR / "phase565_residual_operator_frozen_contract.json"
ROWS_PATH = OUT_DIR / "phase565_residual_operator_rows.jsonl"
EXECUTION_PATH = OUT_DIR / "phase565_residual_operator_execution_summary.json"
ANALYSIS_PATH = OUT_DIR / "phase565_residual_operator_analysis.json"
QUALIFIED_PATH = OUT_DIR / "phase565_qualified_residual_operators.jsonl"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def condition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [float(row["donor_switch_effect"]) for row in rows]
    cell_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    regime_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_rows[row["factorial_cell_without_binding"]].append(row)
        regime_rows[row["color_regime"]].append(row)
    cell_rates = {
        cell: sum(item["intervention_donor_wins"] for item in members) / len(members)
        for cell, members in sorted(cell_rows.items())
    }
    regime_rates = {
        regime: sum(item["intervention_donor_wins"] for item in members) / len(members)
        for regime, members in sorted(regime_rows.items())
    }
    return {
        "row_count": len(rows),
        "donor_win_rate": sum(row["intervention_donor_wins"] for row in rows) / len(rows),
        "recipient_retention_rate": sum(row["intervention_recipient_retained"] for row in rows) / len(rows),
        "mean_donor_switch_effect": mean(effects),
        "minimum_donor_switch_effect": min(effects),
        "maximum_donor_switch_effect": max(effects),
        "maximum_absolute_donor_switch_effect": max(abs(value) for value in effects),
        "factorial_cell_donor_win_rates": cell_rates,
        "minimum_factorial_cell_donor_win_rate": min(cell_rates.values()),
        "color_regime_donor_win_rates": regime_rates,
        "minimum_color_regime_donor_win_rate": min(regime_rates.values()),
    }


def analyze() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    execution = read_json(EXECUTION_PATH)
    rows = read_jsonl(ROWS_PATH)
    if execution["status"] != "complete" or len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase565 residual operator execution is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    gate = contract["operator_gate"]
    reports = []
    qualified = []
    control_names = (
        "paired_contrast_neutralize", "wrong_depth_donor_replace",
        "wrong_position_donor_replace", "channel_roll_donor_replace",
    )
    for candidate_id, members in sorted(grouped.items()):
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            by_condition[row["condition"]].append(row)
        conditions = {
            condition: condition_report(condition_rows)
            for condition, condition_rows in sorted(by_condition.items())
        }
        same = conditions["same_case_restore"]
        donor = conditions["paired_donor_residual_replace"]
        failures = []
        if same["maximum_absolute_donor_switch_effect"] > gate["same_restore_max_abs_effect"]:
            failures.append("same_restore")
        if donor["donor_win_rate"] < gate["paired_donor_win_rate_min"]:
            failures.append("donor_win_rate")
        if donor["minimum_factorial_cell_donor_win_rate"] < gate["minimum_factorial_cell_donor_win_rate"]:
            failures.append("minimum_factorial_cell")
        if donor["mean_donor_switch_effect"] < gate["paired_donor_mean_effect_min"]:
            failures.append("donor_effect")
        for name in control_names:
            if donor["mean_donor_switch_effect"] <= conditions[name]["mean_donor_switch_effect"]:
                failures.append(f"specificity:{name}")
        report = {
            "candidate_id": candidate_id,
            "layer": members[0]["layer"],
            "position_block": members[0]["position_block"],
            "row_count": len(members),
            "conditions": conditions,
            "operator_gate_pass": not failures,
            "operator_gate_failures": failures,
            "evidence_grade": "distributed_state_sufficiency" if not failures else "causal_negative",
            "compute_edge": False,
        }
        reports.append(report)
        if not failures:
            qualified.append({
                "schema_version": "phase565_qualified_residual_operator.v1",
                "phase_id": "Phase565",
                "candidate_id": candidate_id,
                "layer": report["layer"],
                "position_block": report["position_block"],
                "donor_win_rate": donor["donor_win_rate"],
                "minimum_factorial_cell_donor_win_rate": donor["minimum_factorial_cell_donor_win_rate"],
                "minimum_color_regime_donor_win_rate": donor["minimum_color_regime_donor_win_rate"],
                "mean_donor_switch_effect": donor["mean_donor_switch_effect"],
                "evidence_grade": "distributed_state_sufficiency",
                "natural_necessity_tested": False,
                "compute_edge": False,
                "sealed": False,
            })
    write_jsonl(QUALIFIED_PATH, qualified)
    summary = {
        "schema_version": "phase565_residual_operator_analysis.v1",
        "phase_id": "Phase565",
        "created_at": now(),
        "row_count": len(rows),
        "candidate_count": len(reports),
        "qualified_operator_count": len(qualified),
        "qualified_operator_ids": [row["candidate_id"] for row in qualified],
        "candidate_reports": reports,
        "compute_edge_count": 0,
        "natural_necessity_tested": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, summary)
    print(json.dumps({
        "qualified_operator_ids": summary["qualified_operator_ids"],
        "candidates": [{
            "candidate_id": report["candidate_id"],
            "donor_win_rate": report["conditions"]["paired_donor_residual_replace"]["donor_win_rate"],
            "minimum_cell": report["conditions"]["paired_donor_residual_replace"]["minimum_factorial_cell_donor_win_rate"],
            "minimum_regime": report["conditions"]["paired_donor_residual_replace"]["minimum_color_regime_donor_win_rate"],
            "donor_effect": report["conditions"]["paired_donor_residual_replace"]["mean_donor_switch_effect"],
            "gate_failures": report["operator_gate_failures"],
        } for report in reports],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
