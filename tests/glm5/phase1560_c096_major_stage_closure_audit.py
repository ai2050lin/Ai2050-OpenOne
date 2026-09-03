#!/usr/bin/env python3
"""Independent closure audit for Phase1560."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1560_c096_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1560_c096_major_stage_closure.py"), doraise=True)
    closure = core.load(OUT / "analysis/c096_major_stage_closure.json")
    final = core.load(OUT / "analysis/final.json")
    requirements = core.load(OUT / "protocol/c097_requirements.json")
    k268 = closure["puzzle_updates"]["K268"]
    k269 = closure["puzzle_updates"]["K269"]
    checks = {
        "phase": closure["phase"] == 1560 and closure["campaign"] == "C096",
        "inputs_audited": closure["checks"]["all_audited"],
        "positive_and_negative": "supported" in closure["major_answer"] and "refuted" in closure["major_answer"],
        "k268_partial": "not fully confirmed" in k268["status_after_c096"],
        "k269_scoped": k269["grade"] == "E3-OBS-prospective-scoped" and "task-scoped" in k269["scope"],
        "failed_prediction_preserved": k269["negative_boundary"]["observed"] < k269["negative_boundary"]["threshold"],
        "no_causal_upgrade": any("no coordinate intervention" in item for item in closure["hard_limits"]),
        "math_boundary": closure["unified_theory"]["math_status"].startswith("Not closed."),
        "next_three_routes": set(requirements) >= {"route_A", "route_B", "route_C_after_A_or_B"},
        "no_auto_causal": requirements["forbidden"].startswith("Do not launch"),
        "authorization": final["authorization"] == "preregister_C097_targeted_postquery_residual_and_independent_material_stage",
    }
    result = {"phase": 1560, "campaign": "C096", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
