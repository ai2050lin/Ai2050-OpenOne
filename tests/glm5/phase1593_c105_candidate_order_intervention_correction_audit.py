#!/usr/bin/env python3
"""Independent audit for the C105 intervention readout correction."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
C102 = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1593_c105_candidate_order_intervention_correction.py"
    py_compile.compile(str(producer), doraise=True)
    final = core.load(OUT / "analysis/final.json")
    c102_results = core.rows(OUT / "analysis/c102_corrected_intervention_results.jsonl")
    c104_results = core.rows(OUT / "analysis/c104_corrected_intervention_results.jsonl")
    c104_old = core.rows(C104 / "analysis/upstream_role_intervention_results.jsonl")
    checks = {
        "producer": py_compile.compile(str(producer), doraise=True) is not None,
        "candidate_order": final["candidate_order"]["normalized"] == ["yes", "no"],
        "parents": core.sha(C102 / "analysis/coordinate_coalition_intervention_results.jsonl") == final["c102"]["source_sha256"] and core.sha(C104 / "analysis/upstream_role_intervention_results.jsonl") == final["c104"]["source_sha256"],
        "rows": len(c102_results) == 384 and len(c104_results) == len(c104_old) == 192,
        "exact_negation": all(new["modes"][mode]["true_direction_gain_corrected"] == -old["modes"][mode]["true_direction_gain"] for new, old in zip(c104_results, c104_old, strict=True) for mode in old["modes"]),
        "hashes": core.sha(OUT / "analysis/c102_corrected_intervention_results.jsonl") == final["c102"]["results_sha256"] and core.sha(OUT / "analysis/c104_corrected_intervention_results.jsonl") == final["c104"]["results_sha256"],
        "c102_result": sorted(final["c102"]["fully_controlled_families"]) == sorted(["agent_patient", "attribute_binding", "containment", "negation_scope", "whole_part_exception"]),
        "c104_result": sorted(final["c104"]["fully_controlled_families"]) == ["agent_patient", "attribute_binding"],
        "checks": all(final["checks"].values()),
        "scope": "no new model run" in final["claim_boundary"],
        "authorization": final["authorization"] == "export_corrected_c104_heatmap_and_close_c102_c104_c105_stage",
    }
    result = {"phase": 1593, "campaign": "C105", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
