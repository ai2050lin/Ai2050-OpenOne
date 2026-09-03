#!/usr/bin/env python3
"""Independent audit for Phase1524."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1524_c089_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    freeze = core.load(OUT / "protocol/frozen_descriptive_predictions.json")
    summary = core.load(OUT / "analysis/discovery_observation_summary.json")
    rows = core.rows(OUT / "analysis/discovery_state_role_observations.jsonl")
    vectors = np.load(OUT / "protocol/frozen_discovery_centroids.float32.npy")
    py_compile.compile(str(TESTS / "phase1524_c089_discovery_observation_freeze.py"), doraise=True)
    checks = {
        "status": final["status"] == "discovery_observation_frozen_without_semantic_qualification",
        "rows": len(rows) == 444 and {row["family"] for row in rows} == {"synonym", "kind_of", "part_of"},
        "states_roles": all(sum(row["state"] == state and row["role"] == role for row in rows) == 3 for state in range(37) for role in ("source_word", "target_word", "relation_anchor", "boundary")),
        "vectors": list(vectors.shape) == [4, 2560] and bool(np.isfinite(vectors).all()),
        "hashes": core.sha(OUT / "protocol/frozen_discovery_centroids.float32.npy") == freeze["centroid_sha256"] and core.sha(OUT / "analysis/discovery_state_role_observations.jsonl") == freeze["observation_sha256"],
        "freeze_hash": core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}) == freeze["freeze_sha256"] == final["freeze_sha256"],
        "unqualified": freeze["behavior_qualified_families"] == [] and not freeze["semantic_validation_authorized"],
        "blinding": not freeze["holdout_hidden_accessed"],
        "summary": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1525_c089_descriptive_holdout_reveal",
    }
    result = {"phase": 1524, "campaign": "C089", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
