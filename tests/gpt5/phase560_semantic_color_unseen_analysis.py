#!/usr/bin/env python3
"""Qualify Phase560 coarse source-color routes and freeze parent decomposition."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CONTRACT_PATH = OUT_DIR / "phase560_semantic_color_unseen_frozen_contract.json"
ROWS_PATH = OUT_DIR / "phase560_semantic_color_unseen_rows.jsonl"
QUALIFIED_CANDIDATES = OUT_DIR / "phase560_semantic_color_qualified_candidates.json"
ANALYSIS_PATH = OUT_DIR / "phase560_semantic_color_unseen_analysis.json"
EDGES_PATH = OUT_DIR / "phase560_coarse_source_color_edges.jsonl"
PARENT_CONTRACT = OUT_DIR / "phase560_parent_decomposition_frozen_contract.json"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHORS_PATH = PARENT_DIR / "phase559_path_anchor_registry.json"
PARENT_CONDITIONS = (
    "same_case_restore",
    "layer_input_donor_replace",
    "attention_output_donor_replace",
    "mlp_output_donor_replace",
    "layer_output_donor_replace",
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(ROWS_PATH)
    candidates = read_json(QUALIFIED_CANDIDATES)["qualified_candidates"]
    if len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase560 unseen denominator is incomplete")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    gate = contract["validation_gate"]
    reports = []
    edges = []
    for candidate in candidates:
        candidate_rows = grouped[candidate["candidate_id"]]
        by_condition = {
            condition: [row for row in candidate_rows if row["condition"] == condition]
            for condition in contract["conditions"]
        }
        correct = by_condition["correct_paired_donor_replace"]
        same = by_condition["same_case_restore"]
        neutral = by_condition["paired_contrast_neutralize"]
        rolled = by_condition["channel_roll_donor_replace"]
        wrong_position = by_condition["wrong_position_donor_replace"]
        wrong_depth = by_condition["wrong_depth_donor_replace"]
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        regimes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in correct:
            cells[row["factorial_cell_without_binding"]].append(row)
            anchor = row["anchor_id"]
            regime = "heldout" if int(anchor.rsplit("_", 1)[-1]) >= 48 else "core"
            regimes[regime].append(row)
        cell_rates = {cell: rate(group, "intervention_donor_wins") for cell, group in cells.items()}
        same_max = max(abs(float(row["donor_switch_effect"])) for row in same)
        correct_rate = rate(correct, "intervention_donor_wins")
        correct_effect = mean(correct, "donor_switch_effect")
        neutral_effect = mean(neutral, "donor_switch_effect")
        roll_effect = mean(rolled, "donor_switch_effect")
        wrong_position_effect = mean(wrong_position, "donor_switch_effect")
        wrong_depth_effect = mean(wrong_depth, "donor_switch_effect")
        passed = bool(
            same_max <= gate["same_case_max_absolute_switch_effect"]
            and correct_rate >= gate["correct_donor_win_rate_min"]
            and min(cell_rates.values()) >= gate["minimum_factorial_cell_donor_win_rate"]
            and correct_effect >= gate["correct_donor_mean_switch_effect_min"]
            and neutral_effect >= gate["paired_neutralize_mean_switch_effect_min"]
            and correct_effect - roll_effect >= gate[
                "correct_minus_channel_roll_mean_switch_effect_min"
            ]
            and correct_effect - wrong_position_effect >= gate[
                "correct_minus_wrong_position_mean_switch_effect_min"
            ]
        )
        report = {
            "candidate_id": candidate["candidate_id"],
            "layer": candidate["layer"],
            "zone": candidate["zone"],
            "same_case_max_absolute_switch_effect": same_max,
            "correct_donor_win_rate": correct_rate,
            "minimum_factorial_cell_donor_win_rate": min(cell_rates.values()),
            "correct_donor_mean_switch_effect": correct_effect,
            "paired_neutralize_mean_switch_effect": neutral_effect,
            "channel_roll_mean_switch_effect": roll_effect,
            "wrong_position_mean_switch_effect": wrong_position_effect,
            "wrong_depth_mean_switch_effect_diagnostic": wrong_depth_effect,
            "correct_minus_channel_roll_mean_switch_effect": correct_effect - roll_effect,
            "correct_minus_wrong_position_mean_switch_effect": correct_effect - wrong_position_effect,
            "core_donor_win_rate": rate(regimes["core"], "intervention_donor_wins"),
            "heldout_donor_win_rate": rate(regimes["heldout"], "intervention_donor_wins"),
            "validation_gate_pass": passed,
            "coarse_source_color_route": passed,
            "object_color_binding_operation_identified": False,
        }
        reports.append(report)
        if passed:
            edges.append({
                "schema_version": "phase560_coarse_source_color_edge.v1",
                "phase_id": "Phase560",
                "created_at": now(),
                "model": "qwen3",
                "edge_id": f"qwen3__source_color_L{candidate['layer']}__to_answer",
                "source_position": "source_color_end",
                "source_component": "layer_output",
                "source_layer": candidate["layer"],
                "destination": "restricted_next_color_readout_after_natural_downstream_recompute",
                "confirmation_passed": True,
                "unseen_core_passed": report["core_donor_win_rate"] >= 0.90,
                "unseen_heldout_passed": report["heldout_donor_win_rate"] >= 0.90,
                "delete_restore_exclusion_passed": True,
                "wrong_depth_effect": wrong_depth_effect,
                "depth_coordinate_unique": False,
                "coarse_compute_edge": True,
                "binding_operation": False,
                "single_neuron": False,
                "sealed": False,
            })

    write_jsonl(EDGES_PATH, edges)
    used = set(contract["selected_anchor_ids"])
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
    parent_anchors = sorted(
        next(anchor for anchor in sorted(groups[group]) if anchor not in used)
        for group in sorted(groups)
    )
    if len(parent_anchors) != 20 or set(parent_anchors) & used:
        raise RuntimeError("Phase560 parent anchors are not independent")
    parent_contract = {
        "schema_version": "phase560_parent_decomposition_frozen_contract.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "model": "qwen3",
        "split": "unseen_recombination",
        "selected_anchor_ids": parent_anchors,
        "selected_anchor_count": len(parent_anchors),
        "prior_unseen_anchor_overlap_count": 0,
        "recipient_case_count": len(parent_anchors) * 32,
        "candidate_count": len(edges),
        "candidate_ids": [row["candidate_id"] for row in candidates if any(
            edge["source_layer"] == row["layer"] for edge in edges
        )],
        "conditions": list(PARENT_CONDITIONS),
        "expected_intervention_rows": len(parent_anchors) * 32 * len(edges) * len(PARENT_CONDITIONS),
        "evidence_policy": {
            "component_result_is_parent_diagnostic": True,
            "full_layer_output_edge_already_independently_qualified": bool(edges),
            "single_component_does_not_define_full_binding_operation": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    summary = {
        "schema_version": "phase560_semantic_color_unseen_analysis.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "candidate_reports": reports,
        "qualified_coarse_edge_count": len(edges),
        "coarse_edges_path": str(EDGES_PATH.relative_to(ROOT)),
        "binding_operation_identified": False,
        "depth_coordinate_unique": False,
        "parent_decomposition_authorized": bool(edges),
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    write_json(ANALYSIS_PATH, summary)
    write_json(PARENT_CONTRACT, parent_contract)
    print(json.dumps({
        "qualified_coarse_edge_count": len(edges),
        "candidate_reports": reports,
        "parent_anchor_count": len(parent_anchors),
        "expected_parent_rows": parent_contract["expected_intervention_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
