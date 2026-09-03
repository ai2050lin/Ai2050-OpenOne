#!/usr/bin/env python3
"""Independent audit for Phase1509."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1509_c087_dual_holdout_validation"
FREEZE = TESTS / "result/phase1508_c087_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    result = core.load(OUT / "analysis/dual_holdout_validation.json")
    frozen = core.load(FREEZE / "protocol/frozen_holdout_predictions.json")
    py_compile.compile(str(TESTS / "phase1509_c087_dual_holdout_validation.py"), doraise=True)
    holdouts = result["holdouts"]
    checks = {
        "freeze": result["freeze_sha256"] == frozen["freeze_sha256"],
        "holdouts": set(holdouts) == {"confirmation", "lockbox"},
        "rows": all(len(core.rows(OUT / f"analysis/{partition}_layer_role_observations.jsonl")) == 111 for partition in holdouts),
        "check_counts": all(len(row["primary_checks"]) == 7 for row in holdouts.values()),
        "pass_identity": result["dual_holdout_primary_pass"] == all(all(row["primary_checks"].values()) for row in holdouts.values()),
        "scope": "execution identity failure" in result["evidence_scope"],
    }
    audited = {"phase": 1509, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not audited["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", audited)
    print(json.dumps(audited, indent=2))


if __name__ == "__main__":
    main()
