#!/usr/bin/env python3
"""Apply the frozen Phase559 sufficiency screen and diagnose surface failures."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CONTRACT_PATH = OUT_DIR / "phase559_causal_screen_frozen_contract.json"
ROWS_PATH = OUT_DIR / "phase559_causal_screen_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase559_causal_screen_analysis.json"
QUALIFIED_PATH = OUT_DIR / "phase559_screen_qualified_candidates.json"


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


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(ROWS_PATH)
    if len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase559 causal screen denominator is incomplete")
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidates[row["candidate_id"]].append(row)
    reports = []
    qualified = []
    gate = contract["screen_gate"]
    for candidate_id, candidate_rows in sorted(candidates.items()):
        conditions = {
            condition: [row for row in candidate_rows if row["condition"] == condition]
            for condition in contract["conditions"]
        }
        correct = conditions["correct_paired_donor_replace"]
        rolled = conditions["channel_roll_donor_replace"]
        same = conditions["same_case_restore"]
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        surfaces: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in correct:
            cells[row["factorial_cell_without_binding"]].append(row)
            surfaces[int(row["surface_id"])].append(row)
        cell_win_rates = {cell: rate(group, "intervention_donor_wins") for cell, group in cells.items()}
        surface_win_rates = {str(surface): rate(group, "intervention_donor_wins") for surface, group in surfaces.items()}
        correct_effect = mean(correct, "donor_switch_effect")
        rolled_effect = mean(rolled, "donor_switch_effect")
        same_max = max((abs(float(row["donor_switch_effect"])) for row in same), default=0.0)
        donor_win_rate = rate(correct, "intervention_donor_wins")
        minimum_cell = min(cell_win_rates.values(), default=0.0)
        passed = bool(
            same_max <= gate["same_case_max_absolute_switch_effect"]
            and donor_win_rate >= gate["correct_donor_win_rate_min"]
            and minimum_cell >= gate["minimum_factorial_cell_donor_win_rate"]
            and correct_effect >= gate["correct_donor_mean_switch_effect_min"]
            and correct_effect - rolled_effect >= gate[
                "correct_minus_channel_roll_mean_switch_effect_min"
            ]
        )
        report = {
            "candidate_id": candidate_id,
            "boundary": candidate_rows[0]["boundary"],
            "zone": candidate_rows[0]["zone"],
            "layer": candidate_rows[0]["layer"],
            "row_count": len(candidate_rows),
            "same_case_max_absolute_switch_effect": same_max,
            "correct_donor_mean_switch_effect": correct_effect,
            "channel_roll_mean_switch_effect": rolled_effect,
            "correct_minus_channel_roll_mean_switch_effect": correct_effect - rolled_effect,
            "correct_donor_win_rate": donor_win_rate,
            "minimum_factorial_cell_donor_win_rate": minimum_cell,
            "factorial_cell_donor_win_rates": cell_win_rates,
            "surface_donor_win_rates": surface_win_rates,
            "screen_gate_pass": passed,
            "screen_pass_would_be_sufficiency_only": True,
            "compute_edge": False,
        }
        reports.append(report)
        if passed:
            qualified.append({
                "candidate_id": candidate_id,
                "boundary": report["boundary"],
                "zone": report["zone"],
                "layer": report["layer"],
                "status": "confirmation_sufficiency_candidate",
                "compute_edge": False,
            })
    source_reports = [row for row in reports if row["boundary"] == "source"]
    query_reports = [row for row in reports if row["boundary"] == "query"]
    source_surface_pattern = {
        surface: sum(row["surface_donor_win_rates"][surface] for row in source_reports) / len(source_reports)
        for surface in ("0", "1", "2", "3")
    }
    registry = {
        "schema_version": "phase559_screen_qualified_candidates.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "qualified_candidate_count": len(qualified),
        "qualified_candidates": qualified,
        "unseen_intervention_authorized": bool(qualified),
        "compute_edge_confirmed": False,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    summary = {
        "schema_version": "phase559_causal_screen_analysis.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "candidate_reports": reports,
        "qualified_candidate_count": len(qualified),
        "source_surface_mean_donor_win_rates": source_surface_pattern,
        "query_boundary_max_donor_win_rate": max(
            (row["correct_donor_win_rate"] for row in query_reports), default=0.0
        ),
        "diagnosis": {
            "query_single_position_complete_state_sufficiency_supported": False,
            "source_fact_terminal_is_surface_role_mixed": source_surface_pattern["3"] < 0.5,
            "source_fact_terminal_result_is_not_binding_mechanism": True,
            "next_corrective_coordinate": "source_color_end",
        },
        "compute_edge_confirmed": False,
        "sealed_split_read": False,
    }
    write_json(QUALIFIED_PATH, registry)
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({
        "qualified_candidate_count": len(qualified),
        "source_surface_mean_donor_win_rates": source_surface_pattern,
        "query_boundary_max_donor_win_rate": summary["query_boundary_max_donor_win_rate"],
        "candidate_reports": [{
            "candidate_id": row["candidate_id"],
            "donor_win_rate": row["correct_donor_win_rate"],
            "min_cell": row["minimum_factorial_cell_donor_win_rate"],
            "correct_effect": row["correct_donor_mean_switch_effect"],
            "roll_effect": row["channel_roll_mean_switch_effect"],
            "pass": row["screen_gate_pass"],
        } for row in reports],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
