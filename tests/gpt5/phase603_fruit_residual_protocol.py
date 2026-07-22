#!/usr/bin/env python3
"""Freeze model-specific observational rules for qualified Phase602 branches."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase602_three_track_protocol as phase602  # noqa: E402


PHASE = "Phase603"
SCHEMA_VERSION = "phase603_fruit_residual_observer.v1"
FROZEN_AT = "2026-07-22T13:10:00+00:00"
OUT_DIR = ROOT / "tests/gpt5/result/phase603_fruit_residual_observer"
PROTOCOL_PATH = OUT_DIR / "phase603_frozen_protocol.json"
ANALYSIS_PATH = OUT_DIR / "phase603_residual_analysis.json"
QUALIFIED_BRANCHES = {
    "qwen3": ("technical", "daily", "explicit_evidence"),
    "glm4": ("daily",),
}
NONFRUIT_ROLES = ("seed_vegetable", "meat", "dairy", "seafood")
SURFACES = tuple(f"surface_{index}" for index in range(4))
SPLITS = phase602.SPLITS
FIXED_BATCH_SIZE = 8
TOP_UNITS_PER_LAYER = 4
DISCOVERY_MIN_NORMALIZED_EFFECT = 0.05
CONFIRMATION_MIN_NORMALIZED_EFFECT = 0.02
HELDOUT_MIN_NORMALIZED_EFFECT = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze() -> dict:
    analysis = json.loads((phase602.OUT_DIR / "phase602_cross_model_analysis.json").read_text())
    authorized = sorted(analysis["model_specific_internal_observation_authorized"])
    expected = sorted(f"{model}/{track}" for model, tracks in QUALIFIED_BRANCHES.items() for track in tracks)
    if authorized != expected:
        raise RuntimeError(f"Phase603 branch authorization drift: {authorized} != {expected}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": FROZEN_AT,
        "phase602_protocol_sha256": sha256_file(phase602.PROTOCOL_PATH),
        "phase602_analysis_sha256": sha256_file(phase602.OUT_DIR / "phase602_cross_model_analysis.json"),
        "qualified_branches": {model: list(tracks) for model, tracks in QUALIFIED_BRANCHES.items()},
        "splits": list(SPLITS),
        "surfaces": list(SURFACES),
        "nonfruit_roles": list(NONFRUIT_ROLES),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "candidate_rule": {
            "discovery_only_selection": True,
            "top_units_per_layer": TOP_UNITS_PER_LAYER,
            "discovery_min_normalized_effect": DISCOVERY_MIN_NORMALIZED_EFFECT,
            "required_discovery_role_surface_direction_checks": len(NONFRUIT_ROLES) * len(SURFACES),
            "confirmation_min_normalized_effect": CONFIRMATION_MIN_NORMALIZED_EFFECT,
            "heldout_min_normalized_effect": HELDOUT_MIN_NORMALIZED_EFFECT,
            "confirmation_and_heldout_require_all_role_surface_directions": True,
        },
        "observation_coordinate": "last prompt token residual stream at embedding output and every block output",
        "constraints": {
            "raw_attention_collected": False,
            "raw_mlp_collected": False,
            "parameters_collected": False,
            "unqualified_branches_collected": False,
            "causal_intervention": False,
            "single_neuron_mechanism_claim": False,
            "theory_or_formula_update": False,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
