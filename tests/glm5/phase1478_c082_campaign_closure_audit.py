#!/usr/bin/env python3
"""Independent audit for Phase1478."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1478_c082_campaign_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    manifest = core.load(RESULT / "phase1477_c082_atlas_synthesis/frozen/future_prediction_manifest.json")
    py_compile.compile(str(TESTS / "phase1478_c082_campaign_closure.py"), doraise=True)
    checks = {
        "status": final["status"] == "closed_with_retrospective_lexical_to_common_boundary_convergence_candidate",
        "checks": all(final["checks"].values()),
        "scope": final["evidence_level"] == "exploratory retrospective candidate only",
        "predictions": [row["id"] for row in manifest["future_fresh_material_predictions"]] == [f"P082-{index}" for index in range(1, 6)],
        "not_established": len(final["not_established"]) == 6,
        "authorization": final["authorization"] == "preregister_c083_fresh_material_validation_of_lexical_to_common_boundary_convergence",
    }
    result = {"phase": 1478, "campaign": "C082", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
