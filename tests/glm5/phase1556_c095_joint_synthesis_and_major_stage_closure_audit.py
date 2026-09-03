#!/usr/bin/env python3
"""Independent audit for Phase1556."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1556_c095_joint_synthesis_and_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1556_c095_joint_synthesis_and_major_stage_closure.py"), doraise=True)
    synthesis = core.load(OUT / "analysis/c095_major_stage_synthesis.json")
    final = core.load(OUT / "analysis/final.json")
    requirements = core.load(OUT / "protocol/c096_requirements.json")
    k268 = synthesis["adjudication"]["K268"]
    checks = {
        "phase": synthesis["phase"] == 1556 and synthesis["campaign"] == "C095",
        "audited_inputs": synthesis["checks"]["all_audited"],
        "evidence_grade": k268["grade"] == "E2-OBS-retrospective-candidate",
        "five_missing_layers": len(k268["missing"]) == 5,
        "coordinate_boundary": "not a fixed top-k neuron set" in synthesis["adjudication"]["coordinate_identity"],
        "no_new_math": synthesis["theory"]["math_status"].endswith("No invariant composition law or conservation object licenses new mathematics."),
        "fresh_material": "no lexical overlap" in requirements["object"],
        "sample_target": requirements["sample_target"].startswith("90 pairs"),
        "five_predictions": len(requirements["predictions_to_freeze_before_capture"]) == 5,
        "route_branches": len(requirements["routes_after_reveal"]) == 4,
        "layered_policy": "M_BEHAVIOR" in requirements["policy"],
        "authorization": final["authorization"] == "run_phase1557_c096_fresh_human_relation_field_contract",
    }
    result = {"phase": 1556, "campaign": "C095", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
