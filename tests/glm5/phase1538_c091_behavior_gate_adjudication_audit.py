#!/usr/bin/env python3
"""Independent audit for Phase1538."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1538_c091_behavior_gate_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    frozen = core.load(OUT / "protocol/frozen_behavior_routes_and_hidden_scope.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "qualified": frozen["qualified_families"] == ["whole_part"],
        "retired": sorted(frozen["retired_behavior_routes"]) == ["class_inclusion", "similarity"],
        "whole_pass": all(frozen["adjudication"]["whole_part"]["checks"].values()),
        "similar_fail": not frozen["adjudication"]["similarity"]["behavior_qualified"],
        "class_fail": not frozen["adjudication"]["class_inclusion"]["behavior_qualified"],
        "scope": frozen["hidden_capture_scope"]["semantic_interpretation"].startswith("only rows queried for whole_part"),
        "lexical_boundary": "not exactly canceled" in frozen["frozen_behavior_grounded_contrast"]["interpretation"],
        "authorization": final["authorization"] == "run_phase1539_c091_canonical_all_state_capture",
    }
    result = {
        "phase": 1538,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
