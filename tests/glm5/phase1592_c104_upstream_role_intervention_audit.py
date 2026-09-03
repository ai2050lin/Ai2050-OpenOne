#!/usr/bin/env python3
"""Independent audit for Phase1592 / C104 upstream role intervention."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1592_c104_upstream_role_intervention.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/upstream_intervention_protocol.json")
    report = core.load(OUT / "analysis/upstream_role_intervention_final.json")
    results = core.rows(OUT / "analysis/upstream_role_intervention_results.jsonl")
    summary = core.rows(OUT / "analysis/upstream_role_intervention_summary.jsonl")
    rollup = core.rows(OUT / "analysis/upstream_role_intervention_family_rollup.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "parent": core.sha(OUT / "analysis/frozen_candidate_validation_final.json") == protocol["validation_final_sha256"],
        "pairs": len(results) == protocol["pairs"] == report["pairs"] == 192,
        "summary": len(summary) == 16 and len(rollup) == 4,
        "hashes": core.sha(OUT / "analysis/upstream_role_intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/upstream_role_intervention_summary.jsonl") == report["summary_sha256"] and core.sha(OUT / "analysis/upstream_role_intervention_family_rollup.jsonl") == report["rollup_sha256"],
        "modes": all(set(row["modes"]) == set(protocol["modes"]) for row in results),
        "strata": {(row["partition"], row["code"]) for row in summary} == {("confirmation", 1), ("confirmation", -1), ("lockbox", 1), ("lockbox", -1)},
        "source_checks": all(report["checks"].values()),
        "scope": "causal sufficiency" in report["interpretation"] and "predictive barcode" in report["interpretation"],
        "authorization": report["authorization"] == "export_and_close_c104_major_stage",
    }
    result = {"phase": 1592, "campaign": "C104", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_upstream_role_intervention_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
