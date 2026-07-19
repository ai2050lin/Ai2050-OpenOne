#!/usr/bin/env python3
"""Analyze Phase564 aggregate source-edge discovery and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
PROTOCOL_PATH = OUT_DIR / "phase564_frozen_protocol.json"
ANCHOR_REGISTRY_PATH = OUT_DIR / "phase564_edge_anchor_registry.json"
DISCOVERY_CONTRACT_PATH = OUT_DIR / "phase564_source_edge_discovery_frozen_contract.json"
DISCOVERY_CANDIDATES_PATH = OUT_DIR / "phase564_source_edge_discovery_candidates.json"
DISCOVERY_ROWS_PATH = OUT_DIR / "phase564_source_edge_discovery_rows.jsonl"
DISCOVERY_EXECUTION_PATH = OUT_DIR / "phase564_source_edge_discovery_execution_summary.json"
DISCOVERY_ANALYSIS_PATH = OUT_DIR / "phase564_source_edge_discovery_analysis.json"
CONFIRMATION_CANDIDATES_PATH = OUT_DIR / "phase564_source_edge_confirmation_candidates.json"
CONFIRMATION_CONTRACT_PATH = OUT_DIR / "phase564_source_edge_confirmation_frozen_contract.json"
CONFIRMATION_ROWS_PATH = OUT_DIR / "phase564_source_edge_confirmation_rows.jsonl"
CONFIRMATION_EXECUTION_PATH = OUT_DIR / "phase564_source_edge_confirmation_execution_summary.json"
CONFIRMATION_ANALYSIS_PATH = OUT_DIR / "phase564_source_edge_confirmation_analysis.json"
UNSEEN_CONTRACT_PATH = OUT_DIR / "phase564_source_edge_unseen_frozen_contract.json"
UNSEEN_CANDIDATES_PATH = OUT_DIR / "phase564_source_edge_unseen_candidates.json"
UNSEEN_ROWS_PATH = OUT_DIR / "phase564_source_edge_unseen_rows.jsonl"
UNSEEN_EXECUTION_PATH = OUT_DIR / "phase564_source_edge_unseen_execution_summary.json"
UNSEEN_ANALYSIS_PATH = OUT_DIR / "phase564_source_edge_unseen_analysis.json"
QUALIFIED_EDGES_PATH = OUT_DIR / "phase564_qualified_source_compute_edges.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [float(row["donor_switch_effect"]) for row in rows]
    cell_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell_rows[row["factorial_cell_without_binding"]].append(row)
    cell_win_rates = {
        cell: sum(item["intervention_donor_wins"] for item in members) / len(members)
        for cell, members in sorted(cell_rows.items())
    }
    return {
        "row_count": len(rows),
        "donor_win_rate": sum(row["intervention_donor_wins"] for row in rows) / len(rows),
        "recipient_retention_rate": sum(row["intervention_recipient_retained"] for row in rows) / len(rows),
        "mean_donor_switch_effect": mean(effects),
        "minimum_donor_switch_effect": min(effects),
        "maximum_donor_switch_effect": max(effects),
        "maximum_absolute_donor_switch_effect": max(abs(value) for value in effects),
        "factorial_cell_donor_win_rates": cell_win_rates,
        "minimum_factorial_cell_donor_win_rate": min(cell_win_rates.values()),
    }


def candidate_reports(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"]].append(row)
    reports = []
    for candidate_id, members in sorted(grouped.items()):
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            by_condition[row["condition"]].append(row)
        conditions = {
            condition: condition_report(condition_rows)
            for condition, condition_rows in sorted(by_condition.items())
        }
        donor = conditions.get("paired_donor_edge_replace", {})
        removal = conditions.get("source_edge_remove", {})
        same = conditions.get("same_case_restore", {})
        reports.append({
            "candidate_id": candidate_id,
            "layer": members[0]["candidate_layer"],
            "target_role": members[0]["target_role"],
            "source_role": members[0]["source_role"],
            "row_count": len(members),
            "conditions": conditions,
            "selection_tuple": [
                donor.get("donor_win_rate", 0.0),
                donor.get("mean_donor_switch_effect", 0.0),
                removal.get("mean_donor_switch_effect", 0.0),
            ],
            "same_restore_max_abs_effect": same.get("maximum_absolute_donor_switch_effect"),
        })
    return reports


def discovery() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    contract = read_json(DISCOVERY_CONTRACT_PATH)
    registry = read_json(DISCOVERY_CANDIDATES_PATH)
    execution = read_json(DISCOVERY_EXECUTION_PATH)
    rows = read_jsonl(DISCOVERY_ROWS_PATH)
    if execution["status"] != "complete" or len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase564 source-edge discovery is incomplete")
    if execution["rows_sha256"] != sha256_file(DISCOVERY_ROWS_PATH):
        raise RuntimeError("Phase564 source-edge discovery row hash drift")
    reports = candidate_reports(rows)
    if len(reports) != contract["candidate_count"]:
        raise RuntimeError("Phase564 source-edge discovery candidate drift")
    ranked = sorted(
        reports,
        key=lambda report: tuple(report["selection_tuple"]),
        reverse=True,
    )
    maximum = int(contract["selection_policy"]["maximum_confirmation_candidates"])
    selected_ids = {report["candidate_id"] for report in ranked[:maximum]}
    candidate_lookup = {row["candidate_id"]: row for row in registry["candidates"]}
    selected_candidates = []
    for report in ranked[:maximum]:
        candidate = dict(candidate_lookup[report["candidate_id"]])
        layer = int(candidate["layer"])
        candidate["wrong_depth_control_layer"] = layer + 3 if layer <= 7 else layer - 3
        candidate["discovery_selection_tuple"] = report["selection_tuple"]
        candidate["selected_for_independent_confirmation"] = True
        selected_candidates.append(candidate)
    confirmation_registry = {
        "schema_version": "phase564_source_edge_confirmation_candidates.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "candidate_family_frozen_before_intervention": True,
        "selection_source": "edge_discovery_only",
        "confirmation_or_unseen_rows_read_for_selection": False,
        "candidate_count": len(selected_candidates),
        "candidates": selected_candidates,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed_split_read": False,
    }
    write_json(CONFIRMATION_CANDIDATES_PATH, confirmation_registry)

    anchors = read_json(ANCHOR_REGISTRY_PATH)
    selected_anchors = sorted(
        row["anchor_id"] for row in anchors["anchors"]
        if row["model"] == "qwen3"
        and row["split"] == "edge_confirmation"
        and row["authorized_for_internal_collection"]
    )
    expected_worlds = int(protocol["split_world_counts"]["edge_confirmation"])
    if len(selected_anchors) != expected_worlds:
        raise RuntimeError("Phase564 edge-confirmation all-correct denominator drift")
    conditions = list(protocol["edge_design"]["confirmation_conditions"])
    confirmation_contract = {
        "schema_version": "phase564_source_edge_confirmation_frozen_contract.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "model": "qwen3",
        "split": "edge_confirmation",
        "selected_anchor_ids": selected_anchors,
        "world_count": len(selected_anchors),
        "recipient_case_count": len(selected_anchors) * 32,
        "candidate_count": len(selected_candidates),
        "conditions": conditions,
        "expected_intervention_rows": len(selected_anchors) * 32 * len(selected_candidates) * len(conditions),
        "parent_discovery_rows_sha256": sha256_file(DISCOVERY_ROWS_PATH),
        "parent_discovery_analysis_pending_at_freeze": False,
        "candidate_registry_sha256": sha256_file(CONFIRMATION_CANDIDATES_PATH),
        "reconstruction_relative_error_max": 0.01,
        "effect_baseline": "same_case_restore_from_same_fused_batch_shape",
        "evidence_policy": {
            "post_softmax_value_contribution_only": True,
            "key_or_attention_weight_mechanism_claimed": False,
            "same_case_restore_required": True,
            "sealed_split_read": False,
            "fine_scan_executed": False,
        },
    }
    write_json(CONFIRMATION_CONTRACT_PATH, confirmation_contract)
    summary = {
        "schema_version": "phase564_source_edge_discovery_analysis.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "row_count": len(rows),
        "candidate_count": len(reports),
        "condition_count": len(contract["conditions"]),
        "candidate_reports": reports,
        "ranked_candidate_ids": [report["candidate_id"] for report in ranked],
        "selected_confirmation_candidate_ids": sorted(selected_ids),
        "selected_confirmation_candidate_count": len(selected_candidates),
        "confirmation_expected_rows": confirmation_contract["expected_intervention_rows"],
        "maximum_reconstruction_relative_error": execution["maximum_reconstruction_relative_error"],
        "numeric_calibration_applied_before_any_intervention_row": True,
        "compute_edge_count": 0,
        "sealed_split_read": False,
    }
    write_json(DISCOVERY_ANALYSIS_PATH, summary)
    print(json.dumps({
        "top_candidates": [{
            "candidate_id": report["candidate_id"],
            "selection_tuple": report["selection_tuple"],
            "same_restore_max_abs_effect": report["same_restore_max_abs_effect"],
        } for report in ranked[:maximum]],
        "confirmation_world_count": len(selected_anchors),
        "confirmation_expected_rows": confirmation_contract["expected_intervention_rows"],
        "maximum_reconstruction_relative_error": execution["maximum_reconstruction_relative_error"],
    }, ensure_ascii=False, indent=2))
    return summary


def gate_report(report: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, list[str]]:
    conditions = report["conditions"]
    same = conditions["same_case_restore"]
    removal = conditions["source_edge_remove"]
    donor = conditions["paired_donor_edge_replace"]
    wrong_names = (
        "nontarget_source_edge_replace", "wrong_target_donor_replace",
        "wrong_depth_donor_replace", "channel_roll_donor_replace",
    )
    failures = []
    if same["maximum_absolute_donor_switch_effect"] > gate["same_restore_max_abs_effect"]:
        failures.append("same_restore")
    if donor["donor_win_rate"] < gate["donor_win_rate_min"]:
        failures.append("donor_win_rate")
    if donor["minimum_factorial_cell_donor_win_rate"] < gate["minimum_factorial_cell_donor_win_rate"]:
        failures.append("minimum_factorial_cell")
    if removal["mean_donor_switch_effect"] < gate["removal_mean_damage_min"]:
        failures.append("necessity_removal")
    if donor["mean_donor_switch_effect"] < gate["donor_mean_effect_min"]:
        failures.append("donor_effect")
    for name in wrong_names:
        if donor["mean_donor_switch_effect"] <= conditions[name]["mean_donor_switch_effect"]:
            failures.append(f"specificity:{name}")
    return not failures, failures


def confirmation() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    contract = read_json(CONFIRMATION_CONTRACT_PATH)
    execution = read_json(CONFIRMATION_EXECUTION_PATH)
    rows = read_jsonl(CONFIRMATION_ROWS_PATH)
    if execution["status"] != "complete" or len(rows) != contract["expected_intervention_rows"]:
        raise RuntimeError("Phase564 source-edge confirmation is incomplete")
    reports = candidate_reports(rows)
    gate = protocol["edge_gate"]
    passing = []
    for report in reports:
        report["confirmation_gate_pass"], report["confirmation_gate_failures"] = gate_report(report, gate)
        if report["confirmation_gate_pass"]:
            passing.append(report["candidate_id"])
    summary = {
        "schema_version": "phase564_source_edge_confirmation_analysis.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "row_count": len(rows),
        "candidate_count": len(reports),
        "candidate_reports": reports,
        "confirmation_passing_candidate_ids": passing,
        "confirmation_passing_candidate_count": len(passing),
        "unseen_required": bool(passing),
        "compute_edge_count": 0,
        "maximum_reconstruction_relative_error": execution["maximum_reconstruction_relative_error"],
        "sealed_split_read": False,
    }
    write_json(CONFIRMATION_ANALYSIS_PATH, summary)
    if passing:
        anchors = read_json(ANCHOR_REGISTRY_PATH)
        selected_anchors = sorted(
            row["anchor_id"] for row in anchors["anchors"]
            if row["model"] == "qwen3"
            and row["split"] == "edge_unseen"
            and row["authorized_for_internal_collection"]
        )
        candidate_registry = read_json(CONFIRMATION_CANDIDATES_PATH)
        candidates = [
            row for row in candidate_registry["candidates"] if row["candidate_id"] in set(passing)
        ]
        conditions = list(protocol["edge_design"]["confirmation_conditions"])
        unseen_contract = {
            "schema_version": "phase564_source_edge_unseen_frozen_contract.v1",
            "phase_id": "Phase564",
            "created_at": now(),
            "model": "qwen3",
            "split": "edge_unseen",
            "selected_anchor_ids": selected_anchors,
            "world_count": len(selected_anchors),
            "recipient_case_count": len(selected_anchors) * 32,
            "candidate_count": len(candidates),
            "conditions": conditions,
            "expected_intervention_rows": len(selected_anchors) * 32 * len(candidates) * len(conditions),
            "candidate_registry_sha256": sha256_file(CONFIRMATION_CANDIDATES_PATH),
            "reconstruction_relative_error_max": 0.01,
            "effect_baseline": "same_case_restore_from_same_fused_batch_shape",
            "evidence_policy": {
                "post_softmax_value_contribution_only": True,
                "key_or_attention_weight_mechanism_claimed": False,
                "same_case_restore_required": True,
                "sealed_split_read": False,
                "fine_scan_executed": False,
            },
        }
        write_json(UNSEEN_CONTRACT_PATH, unseen_contract)
        reduced_registry = dict(candidate_registry)
        reduced_registry["candidate_count"] = len(candidates)
        reduced_registry["candidates"] = candidates
        write_json(UNSEEN_CANDIDATES_PATH, reduced_registry)
        summary["unseen_expected_rows"] = unseen_contract["expected_intervention_rows"]
        write_json(CONFIRMATION_ANALYSIS_PATH, summary)
    else:
        QUALIFIED_EDGES_PATH.write_text("", encoding="utf-8")
    print(json.dumps({
        "confirmation_passing_candidate_ids": passing,
        "unseen_required": bool(passing),
        "candidates": [{
            "candidate_id": report["candidate_id"],
            "donor_win_rate": report["conditions"]["paired_donor_edge_replace"]["donor_win_rate"],
            "donor_effect": report["conditions"]["paired_donor_edge_replace"]["mean_donor_switch_effect"],
            "removal_damage": report["conditions"]["source_edge_remove"]["mean_donor_switch_effect"],
            "gate_failures": report["confirmation_gate_failures"],
        } for report in reports],
    }, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("discovery", "confirmation"))
    args = parser.parse_args()
    if args.mode == "discovery":
        discovery()
    else:
        confirmation()
