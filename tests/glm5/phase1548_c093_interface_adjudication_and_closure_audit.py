#!/usr/bin/env python3
"""Independent audit for Phase1548."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1548_c093_interface_adjudication_and_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/c093_closure.json")
    next_campaign = core.load(OUT / "protocol/next_campaign_authorization.json")
    checks = {
        "none_passed": report["passing_interfaces"] == [],
        "four_interfaces": set(report["reversed_postquery_balanced_accuracy"]) == {"han", "latin", "digits", "xy"},
        "all_postquery_failed": all(value < 0.75 for value in report["reversed_postquery_balanced_accuracy"].values()),
        "hidden_not_accessed": report["hidden_states_accessed"] is False,
        "no_puzzle": report["core_puzzle_update"] == "none",
        "claim_boundary": "arbitrary mapping impossible" in report["not_concluded"] and "K267 false" in report["not_concluded"],
        "next_identical": next_campaign == report["next_campaign"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1549_c094_demonstrated_codebook_contract",
    }
    result = {"phase": 1548, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
