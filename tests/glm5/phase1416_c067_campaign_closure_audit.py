#!/usr/bin/env python3
"""Independent audit for Phase1416."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1416_c067_campaign_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_after_failed_discrete_pair_composition_with_graded_ordering",
        "retained_not_closed": all(final["retained"]["graded_effects"].values()),
        "strong_rejected": "sufficient discrete family-equality comparator" in final["rejected"]["strong_hypothesis"],
        "zero_families": final["rejected"]["qualified_families"] == [],
        "unseen_not_negative": len(final["untested"]) == 4,
        "no_layer_search": "no layer search" in final["next_question"]["constraints"],
        "four_roles": all(role in final["next_question"]["object"] for role in ("record_target", "record_family", "query_target", "query_family")),
        "authorization": final["authorization"] == "preregister_c068_distributed_four_role_composition",
        "checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {
        "phase": 1416,
        "campaign": "C067",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
