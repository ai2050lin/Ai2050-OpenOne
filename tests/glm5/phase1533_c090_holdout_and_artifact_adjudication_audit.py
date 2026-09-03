#!/usr/bin/env python3
"""Independent audit for Phase1533."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1533_c090_holdout_and_artifact_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/holdout_and_artifact_adjudication.json")
    py_compile.compile(str(TESTS / "phase1533_c090_holdout_and_artifact_adjudication.py"), doraise=True)
    checks = {
        "status": final["status"] == "canonical_descriptive_replication_complete_semantic_block_retained",
        "families": set(summary["family_results"]) == {"synonym", "kind_of", "part_of"},
        "components": all(len(row["components"]) == 7 for row in summary["family_results"].values()),
        "source_repair": summary["execution_adjudication"]["source_truth_contrast_old_max_abs"] > 1e-2 and summary["execution_adjudication"]["source_truth_contrast_canonical_max_abs"] == 0.0,
        "locations": all(row["old_candidate_same_location"] for row in summary["execution_adjudication"]["family_candidate_comparison"].values()),
        "scope": not summary["semantic_validation_authorized"], "checks": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1534_c089_c090_major_stage_closure",
    }
    result = {"phase": 1533, "campaign": "C090", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
