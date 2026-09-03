#!/usr/bin/env python3
"""Independent audit for Phase1452 C075 behavior-gate closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1452, "C075"
OUT = TESTS / "result/phase1452_c075_behavior_gate_closure"
P1451 = TESTS / "result/phase1451_c075_behavior"


def main() -> None:
    result = core.load(OUT / "analysis/final.json")
    behavior = core.load(P1451 / "analysis/behavior_summary.json")
    errors = [row for row in core.rows(P1451 / "raw/active_behavior.jsonl") if not row["correct"]]
    checks = {
        "closure": result["all_checks_passed"] and all(result["checks"].values()),
        "status": result["status"] == "closed_at_behavior_gate_before_hiddenstate_capture",
        "behavior": not behavior["behavior_qualified"] and behavior["qualified_relations"] == ["supported"],
        "errors": len(errors) == 96 and {row["cell"] for row in errors} == {"110"},
        "hidden": behavior["hidden_state_accessed"] is False,
        "boundary": "no Hidden State was read" in result["rejected"]["hidden_inference"],
        "authorization": result["authorization"] == "preregister_c076_explicit_relation_discrimination_atlas",
    }
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
