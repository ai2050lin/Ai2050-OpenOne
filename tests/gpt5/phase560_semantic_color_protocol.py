#!/usr/bin/env python3
"""Freeze semantic color-value source candidates after Phase559's role-mixing failure."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
EVENT_ROWS = PARENT_DIR / "phase559_binding_event_rows.jsonl"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHORS_PATH = PARENT_DIR / "phase559_path_anchor_registry.json"
PRIOR_CONTRACT = PARENT_DIR / "phase559_causal_screen_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase560_semantic_color_candidate_registry.json"
SCREEN_CONTRACT_PATH = OUT_DIR / "phase560_semantic_color_screen_frozen_contract.json"
ZONES = ("early", "middle", "late")
CONDITIONS = (
    "same_case_restore",
    "correct_paired_donor_replace",
    "channel_roll_donor_replace",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def zone(relative_depth: float) -> str:
    if relative_depth < 1.0 / 3.0:
        return "early"
    if relative_depth < 2.0 / 3.0:
        return "middle"
    return "late"


def score(row: dict[str, Any]) -> float:
    metrics = row["split_metrics"]["path_discovery"]
    relative = max(0.0, finite(metrics["mean_relative_binding_delta_norm"]))
    stability = max(0.0, finite(metrics["mean_surface_order_direction_stability"]))
    role = max(0.0, min(1.0, (finite(metrics["mean_query_role_direction_cosine"]) + 1.0) / 2.0))
    return relative * stability * role


def freeze() -> dict[str, Any]:
    event_rows = [
        row for row in read_jsonl(EVENT_ROWS)
        if row["component"] == "layer_output" and row["semantic_position"] == "source_color_end"
    ]
    if len(event_rows) != 36:
        raise RuntimeError("Phase560 semantic color event denominator drift")
    layer_count = int(event_rows[0]["layer_count"])
    candidates = []
    for zone_name in ZONES:
        available = [row for row in event_rows if zone(float(row["relative_depth"])) == zone_name]
        selected = max(available, key=lambda row: (score(row), -int(row["layer"])))
        layer = int(selected["layer"])
        candidates.append({
            "candidate_id": f"qwen3__semantic_source_color__{zone_name}__L{layer}",
            "model": "qwen3",
            "boundary": "semantic_source_color",
            "semantic_position": "source_color_end",
            "wrong_position_control": "nontarget_fact_end",
            "component": "layer_output",
            "zone": zone_name,
            "layer": layer,
            "wrong_depth_control_layer": (layer + layer_count // 2) % layer_count,
            "layer_count": layer_count,
            "selection_split": "path_discovery",
            "selection_score": score(selected),
            "discovery_metrics": selected["split_metrics"]["path_discovery"],
            "confirmation_metrics_not_used_for_selection": selected["split_metrics"]["path_confirmation"],
            "candidate_is_mechanism_evidence": False,
            "confirmation_used_for_selection": False,
            "sealed_used_for_selection": False,
        })

    anchor_registry = read_json(ANCHORS_PATH)
    eligible = {
        row["anchor_id"] for row in anchor_registry["anchors"]
        if row["split"] == "path_confirmation" and row["authorized_for_internal_collection"]
    }
    prior = set(read_json(PRIOR_CONTRACT)["selected_anchor_ids"])
    path_rows = [
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == "path_confirmation" and row["anchor_id"] in eligible
    ]
    worlds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        worlds[row["anchor_id"]].append(row)
    color_groups: dict[str, list[str]] = defaultdict(list)
    for anchor_id, rows in worlds.items():
        color_groups[f"{rows[0]['color_a']}|{rows[0]['color_b']}"] .append(anchor_id)
    selected_anchors = sorted(
        anchor_id
        for color_key in sorted(color_groups)
        for anchor_id in [value for value in sorted(color_groups[color_key]) if value not in prior][:2]
    )
    if len(selected_anchors) != 24 or set(selected_anchors) & prior:
        raise RuntimeError("Phase560 confirmation anchors overlap Phase559 screen")

    registry = {
        "schema_version": "phase560_semantic_color_candidate_registry.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "selection_policy": "discovery-only source_color_end layer_output candidate per depth zone",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_is_mechanism_evidence": False,
        "head_channel_parameter_neuron_scan_authorized": False,
        "sealed_split_read": False,
    }
    contract = {
        "schema_version": "phase560_semantic_color_screen_frozen_contract.v1",
        "phase_id": "Phase560",
        "created_at": now(),
        "model": "qwen3",
        "split": "path_confirmation",
        "candidate_registry_sha256": "pending",
        "selected_anchor_ids": selected_anchors,
        "selected_anchor_count": len(selected_anchors),
        "prior_phase559_anchor_overlap_count": 0,
        "recipient_case_count": len(selected_anchors) * 32,
        "candidate_count": len(candidates),
        "conditions": list(CONDITIONS),
        "expected_intervention_rows": len(selected_anchors) * 32 * len(candidates) * len(CONDITIONS),
        "screen_gate": {
            "same_case_max_absolute_switch_effect": 0.0001,
            "correct_donor_win_rate_min": 0.70,
            "minimum_factorial_cell_donor_win_rate": 0.70,
            "correct_donor_mean_switch_effect_min": 1.0,
            "correct_minus_channel_roll_mean_switch_effect_min": 0.50,
        },
        "evidence_policy": {
            "screen_pass_is_source_content_sufficiency_only": True,
            "binding_compute_edge_not_claimed_by_screen": True,
            "unseen_delete_restore_exclusion_required": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    write_json(CANDIDATES_PATH, registry)
    contract["candidate_registry_sha256"] = sha256_file(CANDIDATES_PATH)
    write_json(SCREEN_CONTRACT_PATH, contract)
    print(json.dumps({
        "candidate_layers": [row["layer"] for row in candidates],
        "selected_anchor_count": len(selected_anchors),
        "prior_overlap": 0,
        "expected_intervention_rows": contract["expected_intervention_rows"],
    }, ensure_ascii=False, indent=2))
    return contract


if __name__ == "__main__":
    freeze()
