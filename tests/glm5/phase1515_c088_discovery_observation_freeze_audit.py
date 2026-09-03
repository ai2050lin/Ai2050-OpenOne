#!/usr/bin/env python3
"""Independent audit for Phase1515 discovery isolation and frozen predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1514_c088_factorial_field_atlas"
OUT = RESULT / "phase1515_c088_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    freeze = core.load(OUT / "protocol/frozen_factorial_predictions.json")
    observations = core.rows(OUT / "analysis/discovery_full_layer_role_observations.jsonl")
    group = np.load(ATLAS / "atlas/group_factorial_effect.float16.npy", mmap_mode="r")
    index = core.rows(ATLAS / "atlas/group_factorial_effect_index.jsonl")
    selected = [row["group_index"] for row in index if row["partition"] == "response_discovery"]
    panel = np.asarray(group[selected, :, 0, 35, 3], dtype=np.float64)
    surface_centroids = panel.mean(axis=0)
    recomputed = cosine(surface_centroids[0], surface_centroids[1])
    recorded = final["discovery"]["target"]["effects"]["semantic"]["surface_centroid_cosine"]
    checks = {
        "authorization": final["authorization"] == "run_phase1516_c088_holdout_and_fresh_reveal",
        "discovery_exact": selected == list(range(72)),
        "row_count": len(observations) == 148,
        "metric_recompute": abs(recomputed - recorded) < 1e-12,
        "freeze_hash": freeze["freeze_sha256"] == core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}),
        "holdouts_untouched": freeze["untouched_partitions"] == ["confirmation", "lockbox", "fresh_external"],
        "gate_separation": set(("structure_presence_gates", "paired_effect_size_tolerances")) <= set(freeze),
        "claim_boundary": "no universal semantic vector" in freeze["claim_boundary"],
    }
    audit = {
        "phase": 1515,
        "campaign": "C088",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
