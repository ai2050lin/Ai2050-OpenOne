#!/usr/bin/env python3
"""Analyze Phase560 screen and freeze unseen delete/restore/exclusion validation."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
SCREEN_CONTRACT = OUT_DIR / "phase560_semantic_color_screen_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase560_semantic_color_candidate_registry.json"
SCREEN_ROWS = OUT_DIR / "phase560_semantic_color_screen_rows.jsonl"
SCREEN_ANALYSIS = OUT_DIR / "phase560_semantic_color_screen_analysis.json"
QUALIFIED_PATH = OUT_DIR / "phase560_semantic_color_qualified_candidates.json"
VALIDATION_CONTRACT = OUT_DIR / "phase560_semantic_color_unseen_frozen_contract.json"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHORS_PATH = PARENT_DIR / "phase559_path_anchor_registry.json"
VALIDATION_CONDITIONS = (
    "same_case_restore",
    "paired_contrast_neutralize",
    "correct_paired_donor_replace",
    "wrong_depth_donor_replace",
    "wrong_position_donor_replace",
    "channel_roll_donor_replace",
)


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
    contract = read_json(SCREEN_CONTRACT)
    registry = read_json(CANDIDATES_PATH)
    rows = read_jsonl(SCREEN_ROWS)
    if len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase560 screen denominator is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    reports = []
    qualified_ids = []
    gate = contract["screen_gate"]
    for candidate_id, candidate_rows in sorted(grouped.items()):
        by_condition = {
            condition: [row for row in candidate_rows if row["condition"] == condition]
            for condition in contract["conditions"]
        }
        correct = by_condition["correct_paired_donor_replace"]
        rolled = by_condition["channel_roll_donor_replace"]
        same = by_condition["same_case_restore"]
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        surfaces: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in correct:
            cells[row["factorial_cell_without_binding"]].append(row)
            surfaces[int(row["surface_id"])].append(row)
        cell_rates = {cell: rate(group, "intervention_donor_wins") for cell, group in cells.items()}
        correct_effect = mean(correct, "donor_switch_effect")
        roll_effect = mean(rolled, "donor_switch_effect")
        same_max = max(abs(float(row["donor_switch_effect"])) for row in same)
        donor_rate = rate(correct, "intervention_donor_wins")
        min_cell = min(cell_rates.values())
        passed = bool(
            same_max <= gate["same_case_max_absolute_switch_effect"]
            and donor_rate >= gate["correct_donor_win_rate_min"]
            and min_cell >= gate["minimum_factorial_cell_donor_win_rate"]
            and correct_effect >= gate["correct_donor_mean_switch_effect_min"]
            and correct_effect - roll_effect >= gate[
                "correct_minus_channel_roll_mean_switch_effect_min"
            ]
        )
        reports.append({
            "candidate_id": candidate_id,
            "layer": candidate_rows[0]["layer"],
            "zone": candidate_rows[0]["zone"],
            "same_case_max_absolute_switch_effect": same_max,
            "correct_donor_win_rate": donor_rate,
            "minimum_factorial_cell_donor_win_rate": min_cell,
            "correct_donor_mean_switch_effect": correct_effect,
            "channel_roll_mean_switch_effect": roll_effect,
            "correct_minus_channel_roll_mean_switch_effect": correct_effect - roll_effect,
            "surface_donor_win_rates": {
                str(surface): rate(group, "intervention_donor_wins")
                for surface, group in surfaces.items()
            },
            "screen_gate_pass": passed,
            "source_content_sufficiency_only": True,
            "binding_compute_edge": False,
        })
        if passed:
            qualified_ids.append(candidate_id)

    candidates = [row for row in registry["candidates"] if row["candidate_id"] in qualified_ids]
    qualified = {
        "schema_version": "phase560_semantic_color_qualified_candidates.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "qualified_candidate_count": len(candidates),
        "qualified_candidates": candidates,
        "status": "confirmation_source_content_sufficiency_only",
        "compute_edge_confirmed": False,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }

    anchor_registry = read_json(ANCHORS_PATH)
    valid_unseen = {
        row["anchor_id"] for row in anchor_registry["anchors"]
        if row["split"] == "unseen_recombination" and row["reserved_for_unseen_validation"]
    }
    unseen_rows = [
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == "unseen_recombination" and row["anchor_id"] in valid_unseen
    ]
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unseen_rows:
        worlds[row["anchor_id"]].append(row)
    groups: dict[str, list[str]] = defaultdict(list)
    for anchor_id, world_rows in worlds.items():
        group = f"{world_rows[0]['color_regime']}::{world_rows[0]['color_a']}|{world_rows[0]['color_b']}"
        groups[group].append(anchor_id)
    selected_unseen = sorted(
        anchor_id
        for group in sorted(groups)
        for anchor_id in sorted(groups[group])[:2]
    )
    if len(groups) != 20 or len(selected_unseen) != 40:
        raise RuntimeError("Phase560 unseen color-regime stratification drift")
    validation = {
        "schema_version": "phase560_semantic_color_unseen_frozen_contract.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "model": "qwen3",
        "split": "unseen_recombination",
        "selected_anchor_ids": selected_unseen,
        "selected_anchor_count": len(selected_unseen),
        "selected_color_regime_pair_count": len(groups),
        "recipient_case_count": len(selected_unseen) * 32,
        "candidate_count": len(candidates),
        "conditions": list(VALIDATION_CONDITIONS),
        "expected_intervention_rows": (
            len(selected_unseen) * 32 * len(candidates) * len(VALIDATION_CONDITIONS)
        ),
        "validation_gate": {
            "same_case_max_absolute_switch_effect": 0.0001,
            "correct_donor_win_rate_min": 0.90,
            "minimum_factorial_cell_donor_win_rate": 0.80,
            "correct_donor_mean_switch_effect_min": 5.0,
            "paired_neutralize_mean_switch_effect_min": 2.0,
            "correct_minus_channel_roll_mean_switch_effect_min": 5.0,
            "correct_minus_wrong_position_mean_switch_effect_min": 5.0,
        },
        "evidence_policy": {
            "successful_result_is_coarse_source_color_route_only": True,
            "object_color_binding_operation_not_identified": True,
            "wrong_depth_is_diagnostic_not_a_gate": True,
            "parent_decomposition_requires_unseen_pass": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    analysis = {
        "schema_version": "phase560_semantic_color_screen_analysis.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "candidate_reports": reports,
        "qualified_candidate_count": len(candidates),
        "all_four_surfaces_recovered": all(
            min(row["surface_donor_win_rates"].values()) == 1.0 for row in reports
        ),
        "claim": "source_color_complete_state_sufficiency_candidate",
        "compute_edge_confirmed": False,
        "sealed_split_read": False,
    }
    write_json(QUALIFIED_PATH, qualified)
    write_json(VALIDATION_CONTRACT, validation)
    write_json(SCREEN_ANALYSIS, analysis)
    print(json.dumps({
        "qualified_candidate_count": len(candidates),
        "candidate_reports": reports,
        "unseen_anchor_count": len(selected_unseen),
        "unseen_color_regime_pair_count": len(groups),
        "expected_unseen_rows": validation["expected_intervention_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
