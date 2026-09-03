#!/usr/bin/env python3
"""Independent audit for Phase1502."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1502_c086_stratum_and_c085_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    summary = core.load(OUT / "analysis/diagnostic_summary.json")
    trajectory = core.rows(OUT / "analysis/three_split_boundary_trajectory.jsonl")
    matrix = core.rows(OUT / "analysis/behavior_truth_code_matrix.jsonl")
    logits = core.rows(OUT / "analysis/logit_margin_four_factor_summary.jsonl")
    py_compile.compile(str(TESTS / "phase1502_c086_stratum_and_c085_diagnostics.py"), doraise=True)
    checks = {
        "trajectory": len(trajectory) == 37 and trajectory[35] == summary["field_formation"]["state35"],
        "matrix": len(matrix) == 8 and sum(r["count"] for r in matrix) == 6912,
        "logits": len(logits) == 36,
        "typed_missingness": summary["behavior"]["strata"] == {"mixed": 216},
        "checks": all(summary["checks"].values()),
    }
    result = {
        "phase": 1502,
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
