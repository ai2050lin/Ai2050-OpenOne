#!/usr/bin/env python3
"""Freeze the Phase559 confirmation-only coarse causal screen."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CANDIDATES_PATH = OUT_DIR / "phase559_binding_candidate_registry.json"
PATH_ROWS = OUT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ANCHORS_PATH = OUT_DIR / "phase559_path_anchor_registry.json"
CONTRACT_PATH = OUT_DIR / "phase559_causal_screen_frozen_contract.json"
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


def freeze() -> dict[str, Any]:
    registry = read_json(CANDIDATES_PATH)
    anchor_registry = read_json(ANCHORS_PATH)
    if registry["candidate_count"] != 6 or anchor_registry["authorized_models"] != ["qwen3"]:
        raise RuntimeError("Phase559 causal screen prerequisites drifted")
    eligible = {
        row["anchor_id"] for row in anchor_registry["anchors"]
        if row["split"] == "path_confirmation" and row["authorized_for_internal_collection"]
    }
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
    selected = sorted(
        anchor_id
        for color_key in sorted(color_groups)
        for anchor_id in sorted(color_groups[color_key])[:2]
    )
    if len(color_groups) != 12 or len(selected) != 24:
        raise RuntimeError("Phase559 confirmation color-pair stratification drift")
    contract = {
        "schema_version": "phase559_causal_screen_frozen_contract.v1",
        "phase_id": "Phase559",
        "created_at": now(),
        "model": "qwen3",
        "split": "path_confirmation",
        "candidate_registry_sha256": sha256_file(CANDIDATES_PATH),
        "path_anchor_registry_sha256": sha256_file(ANCHORS_PATH),
        "selected_anchor_ids": selected,
        "selected_anchor_count": len(selected),
        "selected_color_pair_count": len(color_groups),
        "world_rows": 32,
        "recipient_case_count": len(selected) * 32,
        "candidate_count": registry["candidate_count"],
        "conditions": list(CONDITIONS),
        "expected_intervention_rows": len(selected) * 32 * registry["candidate_count"] * len(CONDITIONS),
        "readout": "restricted_first_non_whitespace_token_over_the_two_registered_colors",
        "screen_gate": {
            "same_case_max_absolute_switch_effect": 0.0001,
            "correct_donor_win_rate_min": 0.70,
            "minimum_factorial_cell_donor_win_rate": 0.50,
            "correct_donor_mean_switch_effect_min": 1.0,
            "correct_minus_channel_roll_mean_switch_effect_min": 0.50,
        },
        "evidence_policy": {
            "screen_pass_is_sufficiency_candidate_only": True,
            "compute_edge_not_claimed_by_screen": True,
            "deletion_and_exclusion_required_after_screen": True,
            "unseen_required_after_screen": True,
            "head_channel_parameter_neuron_scan_authorized": False,
            "sealed_split_read": False,
        },
    }
    write_json(CONTRACT_PATH, contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return contract


if __name__ == "__main__":
    freeze()
