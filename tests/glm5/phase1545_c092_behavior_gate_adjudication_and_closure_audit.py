#!/usr/bin/env python3
"""Independent audit for Phase1545."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1545_c092_behavior_gate_adjudication_and_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/c092_closure.json")
    next_campaign = core.load(OUT / "protocol/next_campaign_authorization.json")
    checks = {
        "native_pass": report["adjudication"]["native"]["qualified"],
        "reversed_fail": not report["adjudication"]["reversed"]["qualified"],
        "aggregate_fail": not report["both_codebooks_qualified"],
        "hidden_not_accessed": report["hidden_states_accessed"] is False,
        "no_puzzle": report["core_puzzle_update"] == "none",
        "claim_boundary": "K267 false" in report["not_concluded"] and "inseparable in principle" in report["not_concluded"][-1],
        "next_identical": next_campaign == report["next_campaign"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] == "run_phase1546_c093_symmetric_code_interface_breadth_contract",
    }
    result = {"phase": 1545, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
