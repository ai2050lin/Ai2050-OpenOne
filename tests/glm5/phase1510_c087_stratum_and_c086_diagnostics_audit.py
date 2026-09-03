#!/usr/bin/env python3
"""Independent audit for Phase1510."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1510_c087_stratum_and_c086_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    data = core.load(OUT / "analysis/stratum_and_c086_diagnostics.json")
    py_compile.compile(str(TESTS / "phase1510_c087_stratum_and_c086_diagnostics.py"), doraise=True)
    checks = {
        "strata": all(set(rows) == {"all", "success", "mixed"} for rows in data["state35_boundary_by_behavior_stratum"].values()),
        "behavior": data["behavior_correct_margin"]["global"]["count"] == 864,
        "disagreements": data["capture_disagreement_count"] == len(data["capture_execution_disagreements"]) == 4,
        "cross_partition": len(data["cross_partition_state35_boundary_cosines"]) == 3,
        "trajectory": len(data["c086_alignment_trajectory"]) == 37,
        "failure_preserved": data["lockbox_failure_anatomy"]["gate_remains_failed"] and data["lockbox_failure_anatomy"]["only_failed_check"],
        "summary": all(data["checks"].values()),
    }
    result = {"phase": 1510, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
