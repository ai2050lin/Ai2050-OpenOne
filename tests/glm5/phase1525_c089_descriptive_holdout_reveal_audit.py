#!/usr/bin/env python3
"""Independent audit for Phase1525."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1525_c089_descriptive_holdout_reveal"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/descriptive_holdout_reveal_summary.json")
    py_compile.compile(str(TESTS / "phase1525_c089_descriptive_holdout_reveal.py"), doraise=True)
    checks = {
        "status": final["status"] == "descriptive_holdouts_revealed_semantic_claim_blocked",
        "families": set(summary["family_results"]) == {"synonym", "kind_of", "part_of"},
        "family_components": all(len(value["components"]) == 7 for value in summary["family_results"].values()),
        "shared_components": len(summary["shared_result"]["components"]) == 4,
        "scope": not summary["semantic_validation_authorized"] and "cannot establish" in summary["strict_conclusion"],
        "checks": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1526_c089_full_dimensional_diagnostics",
    }
    result = {"phase": 1525, "campaign": "C089", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
