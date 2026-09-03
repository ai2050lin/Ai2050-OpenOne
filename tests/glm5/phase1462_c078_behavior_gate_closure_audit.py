#!/usr/bin/env python3
"""Independent audit for Phase1462 C078 closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1462_c078_behavior_gate_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_at_behavior_gate_after_sparse_subcell_conjunction_failure",
        "main": all(final["checks"].values()),
        "failing_count": final["failing_family_relation_surface_count"] == len(final["failing_family_relation_surface"]) == 7,
        "granular": all(not row["checks"]["cell"] for row in final["failing_family_relation_surface"]),
        "hidden": "C078 hidden-state access" in final["rejected"],
        "authorization": final["authorization"] == "preregister_c079_aggregate_eligible_observation_campaign_on_fresh_material",
    }
    result = {"phase": 1462, "campaign": "C078", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
