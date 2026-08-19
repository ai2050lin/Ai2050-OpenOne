#!/usr/bin/env python3
"""Independent audit for Phase1419."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1419_c068_behavior_gate_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_at_behavior_gate_before_hidden_state",
        "two_families": final["retained"]["qualified_families"] == ["organ", "month"],
        "catalog": min(final["retained"]["catalog_accuracy_by_family"].values()) >= 0.99,
        "untested": len(final["untested"]) == 4 and all("failed" not in item for item in final["untested"]),
        "scope_diagnosis": "scope mismatch" in final["design_diagnosis"],
        "unchanged_state": "state16" in final["next_question"]["unchanged"],
        "authorization": final["authorization"] == "preregister_c069_catalog_scoped_four_role_composition",
        "checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {"phase": 1419, "campaign": "C068", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
