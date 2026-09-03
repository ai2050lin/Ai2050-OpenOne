#!/usr/bin/env python3
"""Independent audit for Phase1481."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1481_c080_c083_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    c080 = core.load(RESULT / "phase1470_c080_explicit_behavior/analysis/behavior_summary.json")
    c081 = core.load(RESULT / "phase1473_c081_behavior/analysis/behavior_summary.json")
    c083 = core.load(RESULT / "phase1480_c083_behavior/analysis/behavior_summary.json")
    manifest = core.load(RESULT / "phase1477_c082_atlas_synthesis/frozen/future_prediction_manifest.json")
    py_compile.compile(str(TESTS / "phase1481_c080_c083_major_stage_closure.py"), doraise=True)
    checks = {
        "status": final["status"] == "major_stage_closed_with_one_retrospective_structure_candidate_and_unresolved_fresh_validation",
        "all_checks": all(final["checks"].values()),
        "behavior": not c080["behavior_qualified"] and not c081["behavior_qualified"] and not c083["behavior_qualified"],
        "hidden": not c080["hidden_state_accessed"] and not c081["hidden_state_accessed"] and not c083["hidden_state_accessed"],
        "predictions_unconfirmed": manifest["not_confirmed_here"] and final["answers"]["fresh_validation"].startswith("not tested"),
        "candidate_scope": final["theory_boundary"].startswith("consistent with RDC"),
        "no_auto": not final["next_legal_stage"]["automatic_model_run"],
        "authorization": final["authorization"] == "no_automatic_continuation_until_a_new_behavior_qualified_object_or_project_level_gate_policy_is_preregistered",
    }
    result = {"phase": 1481, "campaign": "C080-C083", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
