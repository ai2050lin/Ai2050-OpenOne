#!/usr/bin/env python3
"""Apply the frozen Phase562 reader gate and close failed static readers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
CONTRACT_PATH = OUT_DIR / "phase562_reader_validation_frozen_contract.json"
REGISTRY_PATH = OUT_DIR / "phase562_reader_candidate_registry.json"
ROWS_PATH = OUT_DIR / "phase562_reader_validation_rows.jsonl"
EXECUTION_PATH = OUT_DIR / "phase562_reader_validation_execution_summary.json"
ANALYSIS_PATH = OUT_DIR / "phase562_reader_validation_analysis.json"
QUALIFIED_PATH = OUT_DIR / "phase562_reader_qualified_edges.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    registry = read_json(REGISTRY_PATH)
    execution = read_json(EXECUTION_PATH)
    rows = read_jsonl(ROWS_PATH)
    if execution["status"] != "complete":
        raise RuntimeError("Phase562 reader validation did not complete")
    if len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase562 reader denominator is incomplete")
    if sha256(ROWS_PATH) != execution["rows_sha256"]:
        raise RuntimeError("Phase562 reader rows changed after execution")

    candidates = {row["candidate_id"]: row for row in registry["candidates"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    if set(grouped) != set(candidates):
        raise RuntimeError("Phase562 candidate registry and rows disagree")

    gate = contract["validation_gate"]
    reports: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        candidate_rows = grouped[candidate_id]
        conditions = {
            condition: [row for row in candidate_rows if row["condition"] == condition]
            for condition in contract["conditions"]
        }
        if any(len(condition_rows) != contract["recipient_case_count"] for condition_rows in conditions.values()):
            raise RuntimeError(f"Phase562 condition denominator drift for {candidate_id}")

        same = conditions["same_case_restore"]
        neutral = conditions["paired_contrast_neutralize"]
        correct = conditions["correct_paired_donor_replace"]
        wrong_depth = conditions["wrong_depth_donor_replace"]
        wrong_position = conditions["wrong_position_donor_replace"]
        rolled = conditions["channel_roll_donor_replace"]
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        surfaces: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in correct:
            cells[row["factorial_cell_without_binding"]].append(row)
            surfaces[str(row["surface_id"])].append(row)

        cell_rates = {cell: rate(cell_rows, "intervention_donor_wins") for cell, cell_rows in cells.items()}
        surface_rates = {
            surface: rate(surface_rows, "intervention_donor_wins")
            for surface, surface_rows in surfaces.items()
        }
        same_max = max(abs(float(row["donor_switch_effect"])) for row in same)
        correct_rate = rate(correct, "intervention_donor_wins")
        minimum_cell = min(cell_rates.values())
        correct_effect = mean(correct, "donor_switch_effect")
        neutral_effect = mean(neutral, "donor_switch_effect")
        roll_effect = mean(rolled, "donor_switch_effect")
        wrong_depth_effect = mean(wrong_depth, "donor_switch_effect")
        wrong_position_effect = mean(wrong_position, "donor_switch_effect")
        checks = {
            "same_case_noop": same_max <= gate["same_case_max_absolute_switch_effect"],
            "correct_donor_win_rate": correct_rate >= gate["correct_donor_win_rate_min"],
            "minimum_factorial_cell": minimum_cell >= gate["minimum_factorial_cell_donor_win_rate"],
            "correct_donor_effect": correct_effect >= gate["correct_donor_mean_switch_effect_min"],
            "paired_neutralize_effect": neutral_effect >= gate["paired_neutralize_mean_switch_effect_min"],
            "channel_roll_specificity": correct_effect - roll_effect >= gate[
                "correct_minus_channel_roll_mean_switch_effect_min"
            ],
            "wrong_position_specificity": correct_effect - wrong_position_effect >= gate[
                "correct_minus_wrong_position_mean_switch_effect_min"
            ],
        }
        passed = all(checks.values())
        report = {
            "candidate_id": candidate_id,
            "boundary": candidate["boundary"],
            "semantic_position": candidate["semantic_position"],
            "component": candidate["component"],
            "layer": candidate["layer"],
            "row_count": len(candidate_rows),
            "same_case_max_absolute_switch_effect": same_max,
            "correct_donor_win_rate": correct_rate,
            "minimum_factorial_cell_donor_win_rate": minimum_cell,
            "correct_donor_mean_switch_effect": correct_effect,
            "paired_neutralize_mean_switch_effect": neutral_effect,
            "channel_roll_mean_switch_effect": roll_effect,
            "wrong_depth_mean_switch_effect_diagnostic": wrong_depth_effect,
            "wrong_position_mean_switch_effect": wrong_position_effect,
            "correct_minus_channel_roll_mean_switch_effect": correct_effect - roll_effect,
            "correct_minus_wrong_position_mean_switch_effect": correct_effect - wrong_position_effect,
            "factorial_cell_donor_win_rates": cell_rates,
            "surface_donor_win_rates": surface_rates,
            "gate_checks": checks,
            "validation_gate_pass": passed,
            "static_single_position_reader_supported": passed,
            "compute_edge_confirmed": False,
        }
        reports.append(report)
        if passed:
            qualified.append({
                "schema_version": "phase562_reader_edge.v1",
                "phase_id": "Phase562",
                "created_at": now(),
                "model": "qwen3",
                "candidate_id": candidate_id,
                "source_layer": candidate["layer"],
                "source_component": candidate["component"],
                "source_position": candidate["semantic_position"],
                "edge_type": "static_single_position_reader_sufficiency",
                "compute_edge": False,
                "single_neuron": False,
                "sealed": False,
            })

    write_jsonl(QUALIFIED_PATH, qualified)
    analysis = {
        "schema_version": "phase562_reader_validation_analysis.v1",
        "phase_id": "Phase562",
        "created_at": now(),
        "model": "qwen3",
        "candidate_reports": reports,
        "qualified_reader_edge_count": len(qualified),
        "qualified_reader_edges_path": str(QUALIFIED_PATH.relative_to(ROOT)),
        "static_single_position_reader_route_closed": len(qualified) == 0,
        "causal_propagation_signature_remains_observational": True,
        "distributed_multi_position_reader_operator_required": len(qualified) == 0,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
        "conclusion": (
            "The L4 and L10 causal propagation signatures are not transportable static reader states. "
            "The source-color intervention changes a distributed downstream computation that cannot "
            "be recovered by replacing one query or answer-position attention output."
        ),
    }
    write_json(ANALYSIS_PATH, analysis)
    print(json.dumps({
        "qualified_reader_edge_count": len(qualified),
        "static_single_position_reader_route_closed": analysis[
            "static_single_position_reader_route_closed"
        ],
        "candidate_reports": [
            {
                "candidate_id": report["candidate_id"],
                "correct_donor_win_rate": report["correct_donor_win_rate"],
                "correct_donor_mean_switch_effect": report["correct_donor_mean_switch_effect"],
                "channel_roll_mean_switch_effect": report["channel_roll_mean_switch_effect"],
                "wrong_position_mean_switch_effect": report["wrong_position_mean_switch_effect"],
                "pass": report["validation_gate_pass"],
            }
            for report in reports
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
