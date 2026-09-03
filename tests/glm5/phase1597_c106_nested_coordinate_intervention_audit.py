#!/usr/bin/env python3
"""Independent audit for the C106 nested-coordinate intervention."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1596_c106_minimal_coordinate_coalition"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1597_c106_nested_coordinate_intervention.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/final.json")
    results = core.rows(OUT / "analysis/nested_coordinate_intervention_results.jsonl")
    summaries = core.rows(OUT / "analysis/nested_coordinate_intervention_summary.jsonl")
    families = core.rows(OUT / "analysis/minimal_coordinate_coalition_by_family.jsonl")
    checks = {
        "producer": py_compile.compile(str(producer), doraise=True) is not None,
        "source": all(report["checks"].values()),
        "rows": len(results) == protocol["pairs"] == 96,
        "nested": all(sorted(int(k) for k in row["nested"]) == protocol["nested_k"] for row in results),
        "summary": len(summaries) == 80 and len(families) == 2,
        "hashes": core.sha(OUT / "analysis/nested_coordinate_intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/nested_coordinate_intervention_summary.jsonl") == report["summary_sha256"] and core.sha(OUT / "analysis/minimal_coordinate_coalition_by_family.jsonl") == report["family_sha256"],
        "positive_control": report["whole_state_positive_control_max_abs"] == 0.0,
        "candidate_order": protocol["candidate_order"] == ["yes", "no"],
        "scope": "not necessity" in report["interpretation"] and "weight localization" in report["interpretation"],
        "authorization": report["authorization"] == "audit_export_and_close_c106",
    }
    result = {"phase": 1597, "campaign": "C106", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
