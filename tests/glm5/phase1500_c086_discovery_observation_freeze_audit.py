#!/usr/bin/env python3
"""Independent audit for Phase1500 discovery-only freeze."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1500_c086_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    final = core.load(OUT / "analysis/final.json")
    rows = core.rows(OUT / "analysis/discovery_layer_role_observations.jsonl")
    frozen = core.load(OUT / "protocol/frozen_holdout_predictions.json")
    py_compile.compile(str(TESTS / "phase1500_c086_discovery_observation_freeze.py"), doraise=True)
    payload = {k: v for k, v in frozen.items() if k != "freeze_sha256"}
    checks = {
        "rows": len(rows) == 259 and {(r["state"], r["role"]) for r in rows} == {
            (state, role) for state in range(37) for role in (
                "record_target", "record_relation", "record_object", "query_target",
                "query_relation", "query_object", "boundary"
            )
        },
        "discovery_only": frozen["source_partition"] == "response_discovery"
        and frozen["untouched_partitions"] == ["confirmation", "lockbox"],
        "hash": frozen["freeze_sha256"] == core.digest(payload) == final["freeze_sha256"],
        "prediction_count": len(frozen["predictions"]) == 6,
        "checks": all(final["checks"].values()),
    }
    result = {
        "phase": 1500,
        "campaign": "C086",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
