#!/usr/bin/env python3
"""Independent audit for Phase1389."""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
OUT = TESTS / "result/phase1389_c061_behavior_gate_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "closed": final["status"] == "closed_at_behavior_gate",
        "all_internal_checks": final["all_checks_passed"] and final["passed"] == final["total"],
        "no_mechanism_claim": "no hidden-state" in final["claim_boundary"],
        "next_is_new_contract": final["authorization"] == "preregister_c062_route_factorized_behavior_and_hidden_campaign",
    }
    result = {"phase": 1389, "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
