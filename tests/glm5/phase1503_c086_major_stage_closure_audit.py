#!/usr/bin/env python3
"""Independent audit for Phase1503."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1503_c086_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    final = core.load(OUT / "analysis/final.json")
    py_compile.compile(str(TESTS / "phase1503_c086_major_stage_closure.py"), doraise=True)
    checks = {
        "phase_checks": all(final["checks"].values()),
        "audit_count": len(final["audits"]) == 7,
        "script_count": len(final["compiled_scripts"]) == 14,
        "puzzle_scope": final["core_puzzle"]["id"] == "K263"
        and "DIAGNOSTIC" in final["core_puzzle"]["evidence"],
        "boundaries": len(final["hard_boundaries"]) == 6,
        "no_new_math": final["theory"]["new_foundational_mathematics"] is False,
        "authorization": final["authorization"]
        == "preregister_c087_cross_root_paraphrase_layered_observation",
    }
    result = {
        "phase": 1503,
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
