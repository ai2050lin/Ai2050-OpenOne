#!/usr/bin/env python3
"""Discover repeated physical trend motifs without reading any semantic label key."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase356_blind_neural_path_cartography/coarse_trace_feasibility"
PHASE = "Phase356"
SCHEMA_VERSION = "32.0.0"
MIN_DISCOVERY_CASES = 24
MIN_CALIBRATION_CASES = 8
RELATIVE_CHANGE = 0.10
DISCOVERY_CASES = 432
CALIBRATION_CASES = 144
DEPTHS = ("early", "middle", "late")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def trend(before: float, after: float) -> str:
    scale = max(abs(before), 1e-8)
    change = (after - before) / scale
    if change > RELATIVE_CHANGE:
        return "increase"
    if change < -RELATIVE_CHANGE:
        return "decrease"
    return "stable"


def motif_id(signature: str) -> str:
    return "motif_" + hashlib.sha256(signature.encode()).hexdigest()[:20]


def discover() -> dict[str, Any]:
    rows = read_jsonl(BASE / "phase356_blind_skeleton_rows.jsonl")
    values = {
        (
            row["blind_case_id"], row["blind_split"], row["generation_step"],
            row["component"], row["position_role"], row["relative_depth"],
        ): row["mean_component_l2_norm"]
        for row in rows
    }
    case_meta = {}
    for row in rows:
        case_meta[row["blind_case_id"]] = row["blind_split"]
    steps: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for case_id, _split, step, component, role, _depth in values:
        steps[(case_id, component, role)].add(step)

    assignments = []
    static_by_slot_step: dict[tuple[str, str, str, int], str] = {}
    for (case_id, component, role), case_steps in sorted(steps.items()):
        split = case_meta[case_id]
        ordered = sorted(case_steps)
        for step in ordered:
            depth_values = [values[(case_id, split, step, component, role, depth)] for depth in DEPTHS]
            depth_shape = (
                trend(depth_values[0], depth_values[1]),
                trend(depth_values[1], depth_values[2]),
            )
            step_role = "start" if step == 0 else "continuation"
            static_signature = "|".join(("static", step_role, component, role, *depth_shape))
            static_id = motif_id(static_signature)
            static_by_slot_step[(case_id, component, role, step)] = static_id
            assignments.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "blind_case_id": case_id, "blind_split": split,
                "generation_step": step, "motif_id": static_id,
                "motif_kind": "depth_shape", "motif_signature": static_signature,
            })
            if step - 1 not in case_steps:
                continue
            previous = [values[(case_id, split, step - 1, component, role, depth)] for depth in DEPTHS]
            time_shape = tuple(trend(before, after) for before, after in zip(previous, depth_values, strict=True))
            dynamic_signature = "|".join(("dynamic", component, role, *depth_shape, *time_shape))
            assignments.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "blind_case_id": case_id, "blind_split": split,
                "generation_step": step, "motif_id": motif_id(dynamic_signature),
                "motif_kind": "depth_time_transition", "motif_signature": dynamic_signature,
            })

    support: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    signatures = {}
    kinds = {}
    for row in assignments:
        support[row["motif_id"]][row["blind_split"]].add(row["blind_case_id"])
        signatures[row["motif_id"]] = row["motif_signature"]
        kinds[row["motif_id"]] = row["motif_kind"]
    discovery_frozen = {
        motif for motif, splits in support.items()
        if len(splits["blind_discovery"]) >= MIN_DISCOVERY_CASES
    }
    registry = []
    for motif in sorted(support):
        discovery_count = len(support[motif]["blind_discovery"])
        calibration_count = len(support[motif]["blind_calibration"])
        discovery_rate = discovery_count / DISCOVERY_CASES
        calibration_rate = calibration_count / CALIBRATION_CASES
        frozen = motif in discovery_frozen
        calibration_stable = bool(
            frozen and calibration_count >= MIN_CALIBRATION_CASES
            and calibration_rate >= 0.5 * discovery_rate
        )
        registry.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "motif_id": motif, "motif_kind": kinds[motif],
            "motif_signature": signatures[motif],
            "discovery_case_support": discovery_count,
            "discovery_case_rate": round(discovery_rate, 7),
            "discovery_candidate_frozen": frozen,
            "calibration_case_support": calibration_count,
            "calibration_case_rate": round(calibration_rate, 7),
            "blind_calibration_stable": calibration_stable,
            "semantic_labels_used_for_discovery": False,
        })

    edge_support: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (case_id, component, role), case_steps in steps.items():
        for step in sorted(case_steps):
            if step - 1 not in case_steps:
                continue
            source = static_by_slot_step[(case_id, component, role, step - 1)]
            target = static_by_slot_step[(case_id, component, role, step)]
            edge_support[(source, target, case_meta[case_id])].add(case_id)
    edges = []
    candidate_ids = {row["motif_id"] for row in registry if row["blind_calibration_stable"]}
    pairs = {(source, target) for source, target, _split in edge_support}
    for source, target in sorted(pairs):
        if source not in candidate_ids or target not in candidate_ids:
            continue
        edges.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "edge_id": f"{source}->{target}", "source_node_id": source,
            "target_node_id": target, "edge_type": "blind_generation_transition",
            "discovery_case_support": len(edge_support[(source, target, "blind_discovery")]),
            "calibration_case_support": len(edge_support[(source, target, "blind_calibration")]),
            "semantic_labels_used_for_discovery": False, "causal_status": "not_tested",
        })

    write_jsonl(BASE / "phase356_blind_motif_assignments.jsonl", assignments)
    write_jsonl(BASE / "phase356_blind_motif_registry.jsonl", registry)
    write_jsonl(BASE / "phase356_blind_graph_edges.jsonl", edges)
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "blind_case_count": len(case_meta), "blind_skeleton_row_count": len(rows),
            "motif_assignment_count": len(assignments), "unique_motif_count": len(registry),
            "blind_graph_edge_count": len(edges),
        },
        "results": {
            "discovery_frozen_motif_count": len(discovery_frozen),
            "blind_calibration_stable_motif_count": sum(row["blind_calibration_stable"] for row in registry),
            "semantic_labels_used_for_discovery": False,
            "top_k_selection_used": False,
            "target_direction_used": False,
            "historical_candidate_used": False,
        },
        "fixed_parameters": {
            "relative_change": RELATIVE_CHANGE,
            "minimum_discovery_cases": MIN_DISCOVERY_CASES,
            "minimum_calibration_cases": MIN_CALIBRATION_CASES,
            "minimum_calibration_to_discovery_prevalence_ratio": 0.5,
        },
        "next_decision": "reveal_labels_for_validation_only",
        "physical_heldout_revealed": False,
        "causal_intervention_executed": False,
        "language_encoding_mechanism_closed": False,
    }
    write_json(BASE / "phase356_blind_discovery_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(discover(), ensure_ascii=False, indent=2))
