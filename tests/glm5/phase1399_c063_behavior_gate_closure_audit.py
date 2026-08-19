#!/usr/bin/env python3
"""Independent closure audit for Phase1399."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1399_c063_behavior_gate_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "closed": final["status"] == "closed_at_behavior_gate",
        "all_internal_checks": final["all_checks_passed"] and final["passed"] == final["total"],
        "zero_qualified": final["formal_results"]["qualified_families"] == [],
        "no_hidden_overclaim": "absence of family identity state" in final["claim_boundary"]["not_supported"],
        "next_is_new_contract": final["authorization"] == "preregister_c064_fixed_natural_answer_factorial_campaign",
    }
    result = {"phase": 1399, "campaign": "C063", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError({k: v for k, v in checks.items() if not v})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
