#!/usr/bin/env python3
"""Independent audit for Phase1585 / C102 intervention."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1585_c102_coordinate_coalition_intervention.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/intervention_protocol.json")
    report = core.load(OUT / "analysis/coordinate_coalition_intervention_final.json")
    results = core.rows(OUT / "analysis/coordinate_coalition_intervention_results.jsonl")
    summary = core.rows(OUT / "analysis/coordinate_coalition_intervention_summary.jsonl")
    checks = {
        "producer": core.sha(producer) == protocol["producer_sha256"],
        "parents": core.sha(OUT / "analysis/staged_barcode_final.json") == protocol["staged_final_sha256"] and core.sha(OUT / "protocol/response_discovery_selection.json") == protocol["selection_sha256"],
        "pairs": len(results) == protocol["pairs"] == report["pairs"] == 384,
        "summary": len(summary) == 16,
        "hashes": core.sha(OUT / "analysis/coordinate_coalition_intervention_results.jsonl") == report["results_sha256"] and core.sha(OUT / "analysis/coordinate_coalition_intervention_summary.jsonl") == report["summary_sha256"],
        "modes": all(set(row["modes"]) == set(protocol["modes"]) for row in results),
        "typed_missing": all(row["typed_missing"]["wrong_family_support"] == (row["k"] == 2560) for row in results),
        "source_checks": all(report["checks"].values()),
        "scope": "not sparse semantic-neuron" in report["interpretation"],
        "authorization": report["authorization"] == "export_c102_coordinate_and_token_heatmap",
    }
    result = {"phase": 1585, "campaign": "C102", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_coordinate_intervention_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
