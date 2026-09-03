#!/usr/bin/env python3
"""Independent audit for Phase1474."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1474_c081_route_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    c080 = core.load(TESTS / "result/phase1470_c080_explicit_behavior/analysis/behavior_summary.json")
    c081 = core.load(TESTS / "result/phase1473_c081_behavior/analysis/behavior_summary.json")
    py_compile.compile(str(TESTS / "phase1474_c081_route_closure.py"), doraise=True)
    checks = {
        "status": final["status"] == "explicit_label_balanced_interaction_route_closed_before_hidden_access",
        "failures": not c080["behavior_qualified"] and not c081["behavior_qualified"],
        "errors": c080["error_counts"]["truth"] == {"true": 607, "false": 0} and c081["error_counts"]["truth"] == {"true": 121, "false": 0},
        "hidden": not c080["hidden_state_accessed"] and not c081["hidden_state_accessed"],
        "route_scope": len(final["route_closed"]) == 3,
        "atlas_exploratory": final["campaign_continues_with"]["scope"].startswith("exploratory observation only"),
        "authorization": final["authorization"] == "preregister_c082_c079_coordinate_resolved_exploratory_atlas",
    }
    result = {"phase": 1474, "campaign": "C081", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
