#!/usr/bin/env python3
"""Independent audit for Phase1488."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1488_c084_layered_observation_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    manifest = core.load(RESULT / "phase1487_c084_joint_synthesis_and_prediction_freeze/frozen/future_prediction_manifest.json")
    policy = core.load(RESULT / "phase1482_layered_observation_policy/protocol/layered_observation_policy.json")
    py_compile.compile(str(TESTS / "phase1488_c084_layered_observation_major_stage_closure.py"), doraise=True)
    checks = {
        "status": final["status"] == "major_stage_closed_with_layered_policy_and_refined_c079_factorial_coordinate_candidates",
        "all_checks": all(final["checks"].values()) and final["passed"] == final["total"] == 8,
        "policy": policy["schema"] == "glm5.layered_observation_predefined_missingness_batch_validation.v1",
        "prediction": manifest["freeze_sha256"] == core.digest({key: value for key, value in manifest.items() if key != "freeze_sha256"}),
        "scope": final["claim_scope"].startswith("strong retrospective") and "new mathematics" in final["claim_scope"],
        "coordinates": "threshold slice" in final["major_stage_answer"]["coordinates"],
        "factorial": "0.9806-0.9942" in final["major_stage_answer"]["factorial"],
        "next": final["next_stage"]["automatic_continuation_recommended"] and final["authorization"].startswith("preregister_c085"),
    }
    result = {"phase": 1488, "campaign": "C084", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
