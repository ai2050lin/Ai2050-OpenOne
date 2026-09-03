#!/usr/bin/env python3
"""Independent audit for Phase1471."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1471_c080_behavior_gate_closure"
BEHAVIOR = TESTS / "result/phase1470_c080_explicit_behavior"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    behavior = core.load(BEHAVIOR / "analysis/behavior_summary.json")
    rows = core.rows(BEHAVIOR / "raw/explicit_behavior.jsonl")
    py_compile.compile(str(TESTS / "phase1471_c080_behavior_gate_closure.py"), doraise=True)
    errors = [row for row in rows if not row["correct"]]
    checks = {
        "status": final["status"] == "closed_at_explicit_behavior_gate_with_surface_specific_positive_failure",
        "counts": len(rows) == 10368 and len(errors) == 607,
        "positive_errors": all(row["truth"] for row in errors),
        "surface_a": abs(behavior["surface"]["a_explicit"]["balanced_accuracy"] - 0.9994212962962963) < 1e-12,
        "surface_b": abs(behavior["surface"]["b_explicit"]["balanced_accuracy"] - 0.6493055555555556) < 1e-12,
        "hidden_not_tested": "any C080 Hidden State" in final["not_tested"],
        "rescue_bounded": final["rescue_limit"].startswith("one fresh-material rescue"),
        "authorization": final["authorization"] == "preregister_c081_historically_validated_interface_rescue_on_fresh_material",
    }
    result = {"phase": 1471, "campaign": "C080", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
