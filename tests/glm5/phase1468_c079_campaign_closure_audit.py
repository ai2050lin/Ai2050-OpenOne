#!/usr/bin/env python3
"""Independent audit for Phase1468 C079 closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1468_c079_campaign_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_with_cross_split_explicit_label_trajectory_regularities",
        "main": all(final["checks"].values()),
        "retained": len(final["retained"]) == 5,
        "boundaries": "six cross-split late-boundary full-vector regularities at states 32-33" in final["retained"],
        "limits": len(final["not_established"]) == 5 and "unlabeled natural relation semantics" in final["not_established"],
        "interaction": final["next_object"]["formula"] == "I_AB = 0.5 * (H_AA + H_BB - H_AB - H_BA)",
        "sequence": final["next_object"]["sequence"][-1] == "only then weak causal tests",
        "authorization": final["authorization"] == "preregister_c080_balanced_equality_interaction_and_label_withdrawal_campaign",
    }
    result = {"phase": 1468, "campaign": "C079", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
