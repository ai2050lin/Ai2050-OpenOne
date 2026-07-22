#!/usr/bin/env python3
"""Freeze the object-end coordinate control for Phase603 residual candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase602_three_track_protocol as phase602  # noqa: E402
import phase603_fruit_residual_protocol as phase603  # noqa: E402


PHASE = "Phase604"
SCHEMA_VERSION = "phase604_object_coordinate_control.v1"
FROZEN_AT = "2026-07-22T13:25:00+00:00"
OUT_DIR = ROOT / "tests/gpt5/result/phase604_object_coordinate_control"
PROTOCOL_PATH = OUT_DIR / "phase604_frozen_protocol.json"
ANALYSIS_PATH = OUT_DIR / "phase604_coordinate_analysis.json"
QUALIFIED_BRANCHES = phase603.QUALIFIED_BRANCHES
FIXED_BATCH_SIZE = phase603.FIXED_BATCH_SIZE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        "phase602_cases_sha256": sha256_file(phase602.CASES_PATH),
        "phase603_protocol_sha256": sha256_file(phase603.PROTOCOL_PATH),
        "phase603_analysis_sha256": sha256_file(phase603.ANALYSIS_PATH),
        "qualified_branches": {model: list(tracks) for model, tracks in QUALIFIED_BRANCHES.items()},
        "coordinate": "last exact token of the last object-label occurrence in the rendered prompt",
        "occurrence_rule": "Search exact tokenizations of label and space+label; deduplicate positions; take the last.",
        "candidate_rule_reused_without_change": {
            "top_units_per_layer": phase603.TOP_UNITS_PER_LAYER,
            "discovery_min_normalized_effect": phase603.DISCOVERY_MIN_NORMALIZED_EFFECT,
            "confirmation_min_normalized_effect": phase603.CONFIRMATION_MIN_NORMALIZED_EFFECT,
            "heldout_min_normalized_effect": phase603.HELDOUT_MIN_NORMALIZED_EFFECT,
            "all_role_surface_directions_required": True,
        },
        "constraints": {
            "same_cases_and_branches_as_phase603": True,
            "future_option_tokens_excluded_at_observation_coordinate": True,
            "causal_intervention": False,
            "mechanism_claim": False,
            "theory_or_formula_update": False,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
