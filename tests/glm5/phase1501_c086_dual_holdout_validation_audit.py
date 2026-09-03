#!/usr/bin/env python3
"""Independent audit for Phase1501."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1501_c086_dual_holdout_validation"
DISCOVERY = TESTS / "result/phase1500_c086_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1501_c086_dual_holdout_validation import adjudicate


def main():
    final = core.load(OUT / "analysis/final.json")
    frozen = core.load(DISCOVERY / "protocol/frozen_holdout_predictions.json")
    py_compile.compile(str(TESTS / "phase1501_c086_dual_holdout_validation.py"), doraise=True)
    recomputed = {
        split: adjudicate(values["observed"], frozen)
        for split, values in final["validation"].items()
    }
    checks = {
        "freeze": frozen["freeze_sha256"]
        == core.load(DISCOVERY / "analysis/final.json")["freeze_sha256"],
        "recompute": all(
            recomputed[split] == final["validation"][split]["prediction_checks"]
            for split in recomputed
        ),
        "counts": all(
            len(core.rows(OUT / f"analysis/{split}_layer_role_observations.jsonl")) == 259
            for split in ("confirmation", "lockbox")
        ),
        "status": final["status"]
        == (
            "dual_holdout_confirmed"
            if all(final["validation"][split]["all_predictions_passed"] for split in final["validation"])
            else "dual_holdout_boundary_failure"
        ),
        "checks": all(final["checks"].values()),
    }
    result = {
        "phase": 1501,
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
