#!/usr/bin/env python3
"""Independent final audit for C102."""
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
    producer = TESTS / "phase1587_c102_major_stage_closure.py"
    py_compile.compile(str(producer), doraise=True)
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "source": final["all_checks_passed"] and final["passed"] == final["total"] == 6,
        "field": final["field"]["shape"] == [37, 179416, 2560] and final["field"]["activation_coordinates_not_parameters"],
        "barcode": final["barcode"]["three_stage_passed"] == final["barcode"]["total"] == 8,
        "intervention": final["intervention"]["families_passing_both_partitions"] == 0,
        "scope": any("no manifold" in row for row in final["adjudication"]["corrected"]),
        "next": final["next_authorization"]["no_new_model_run"] and final["authorization"] == "append_phase1587_c102_memo_then_run_c103_existing_data_observation",
    }
    result = {"phase": 1587, "campaign": "C102", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
