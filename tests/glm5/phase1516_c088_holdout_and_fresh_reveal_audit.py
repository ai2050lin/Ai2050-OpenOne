#!/usr/bin/env python3
"""Independent audit for Phase1516 holdout reveal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1514_c088_factorial_field_atlas"
FREEZE = RESULT / "phase1515_c088_discovery_observation_freeze"
OUT = RESULT / "phase1516_c088_holdout_and_fresh_reveal"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1515_c088_discovery_observation_freeze import cosine


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/partition_reveal_metrics.jsonl")
    freeze = core.load(FREEZE / "protocol/frozen_factorial_predictions.json")
    group = np.load(ATLAS / "atlas/group_factorial_effect.float16.npy", mmap_mode="r")
    index = core.rows(ATLAS / "atlas/group_factorial_effect_index.jsonl")
    fresh_ids = [row["group_index"] for row in index if row["partition"] == "fresh_external"]
    fresh_panel = np.asarray(group[fresh_ids, :, 0, 35, 3], dtype=np.float64)
    recomputed = cosine(fresh_panel.mean(axis=0)[0], fresh_panel.mean(axis=0)[1])
    recorded = next(row for row in rows if row["partition"] == "fresh_external")["effects"]["semantic"]["surface_centroid_cosine"]
    checks = {
        "authorization": final["authorization"] == "run_phase1517_c088_full_dimensional_diagnostics",
        "partitions": [row["partition"] for row in rows] == ["confirmation", "lockbox", "fresh_external"],
        "counts": [row["groups"] for row in rows] == [72, 72, 32],
        "fresh_metric_recompute": abs(recomputed - recorded) < 1e-12,
        "fresh_no_equality": next(row for row in rows if row["partition"] == "fresh_external")["effect_size_equality_passed"] is None,
        "freeze_hash": freeze["freeze_sha256"] == core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}),
        "verdict_recompute": final["verdict"]["structure_presence_paired_holdouts"] == all(row["structure_presence_passed"] for row in rows[:2]),
    }
    audit = {
        "phase": 1516,
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
