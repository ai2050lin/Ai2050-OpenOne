#!/usr/bin/env python3
"""Independent audit for Phase1508."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1508_c087_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    freeze = core.load(OUT / "protocol/frozen_holdout_predictions.json")
    rows = core.rows(OUT / "analysis/discovery_layer_role_observations.jsonl")
    py_compile.compile(str(TESTS / "phase1508_c087_discovery_observation_freeze.py"), doraise=True)
    checks = {
        "partition": freeze["source_partition"] == "response_discovery" and freeze["untouched_partitions"] == ["confirmation", "lockbox"],
        "stratum": freeze["stratum"] == "all",
        "rows": len(rows) == 111 and {row["role"] for row in rows} == {"source_relation", "candidate_relation", "boundary"},
        "freeze": freeze["freeze_sha256"] == core.digest({key: value for key, value in freeze.items() if key != "freeze_sha256"}) == final["freeze_sha256"],
        "source_causal_mask": next(row for row in rows if row["state"] == 35 and row["role"] == "source_relation")["centroid_norm"] == 0.0,
        "scope": "universal comparator" in freeze["claim_boundary"],
        "summary": all(final["checks"].values()),
    }
    result = {"phase": 1508, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
