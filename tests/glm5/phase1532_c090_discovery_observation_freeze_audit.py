#!/usr/bin/env python3
"""Independent audit for Phase1532."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1532_c090_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    freeze = core.load(OUT / "protocol/frozen_canonical_descriptive_predictions.json")
    summary = core.load(OUT / "analysis/canonical_discovery_observation_summary.json")
    rows = core.rows(OUT / "analysis/canonical_discovery_state_role_observations.jsonl")
    vectors = np.load(OUT / "protocol/frozen_canonical_discovery_centroids.float32.npy")
    py_compile.compile(str(TESTS / "phase1532_c090_discovery_observation_freeze.py"), doraise=True)
    checks = {
        "status": final["status"] == "canonical_discovery_frozen_without_semantic_qualification",
        "rows": len(rows) == 444, "source": all(row["mean_norm"] == 0.0 for row in rows if row["role"] == "source_word"),
        "vectors": list(vectors.shape) == [4, 2560] and bool(np.isfinite(vectors).all()),
        "hashes": core.sha(OUT / "protocol/frozen_canonical_discovery_centroids.float32.npy") == freeze["centroid_sha256"] and core.sha(OUT / "analysis/canonical_discovery_state_role_observations.jsonl") == freeze["observation_sha256"],
        "freeze": core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}) == freeze["freeze_sha256"] == final["freeze_sha256"],
        "unqualified": freeze["behavior_qualified_families"] == [] and not freeze["semantic_validation_authorized"],
        "blinding": not freeze["holdout_hidden_accessed"], "checks": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1533_c090_holdout_and_artifact_adjudication",
    }
    result = {"phase": 1532, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
