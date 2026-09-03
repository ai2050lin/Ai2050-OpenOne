#!/usr/bin/env python3
"""Independent closure audit for C104-C105."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C102 = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
C105 = TESTS / "result/phase1593_c105_candidate_order_intervention_correction"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1595_c104_c105_major_stage_closure.py"
    py_compile.compile(str(producer), doraise=True)
    final = core.load(C104 / "analysis/final.json")
    c102_asset = core.load(C102 / "visualization/c102_coordinate_barcode_heatmap.json")
    checks = {
        "producer": py_compile.compile(str(producer), doraise=True) is not None,
        "source_checks": all(final["checks"].values()),
        "fresh": final["fresh_barcode"] == {"passed": 4, "total": 4},
        "c104": sorted(final["c104_corrected_causal"]["fully_controlled_families"]) == ["agent_patient", "attribute_binding"],
        "c102": final["c102_corrected_causal"]["passed"] == 5 and c102_asset["headline"]["controlled_intervention_passed"] == 5,
        "puzzles": set(final["new_puzzles"]) == {"K276", "K277", "K278"},
        "scope": "not sparse coordinates" in final["claim_boundary"] and "single Qwen3" in final["claim_boundary"],
        "authorization": final["next_authorization"].startswith("observation-first C106"),
    }
    result = {"phase": 1595, "campaign": "C104-C105", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(C104 / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
