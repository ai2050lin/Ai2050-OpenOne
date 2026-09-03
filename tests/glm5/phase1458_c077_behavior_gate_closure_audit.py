#!/usr/bin/env python3
"""Independent audit for Phase1458 C077 closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1458, "C077"
BEHAVIOR = TESTS / "result/phase1457_c077_behavior"
OUT = TESTS / "result/phase1458_c077_behavior_gate_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(BEHAVIOR / "analysis/behavior_summary.json")
    rows = core.rows(BEHAVIOR / "raw/active_behavior.jsonl")
    b_true = [row for row in rows if row["surface"] == "b_labeled" and row["truth"]]
    b_false = [row for row in rows if row["surface"] == "b_labeled" and not row["truth"]]
    checks = {
        "status": final["status"] == "closed_at_behavior_gate_after_surface_specific_true_label_failure",
        "main_checks": all(final["checks"].values()),
        "behavior_failed": not summary["behavior_qualified"],
        "errors": final["error_count"] == sum(not row["correct"] for row in rows) == 547,
        "b_true": sum(row["correct"] for row in b_true) == 320,
        "b_false": sum(row["correct"] for row in b_false) == 864,
        "hidden": summary["hidden_state_accessed"] is False,
        "authorization": final["authorization"] == "preregister_c078_colon_label_observation_campaign",
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
