#!/usr/bin/env python3
"""Independent audit for the C104 upstream role-state heatmap asset."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C104 = TESTS / "result/phase1589_c104_upstream_candidate_validation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c104_upstream_role_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1594_c104_upstream_role_heatmap_export.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(C104 / "analysis/upstream_heatmap_export.json")
    asset = ROOT / report["asset"]
    payload = core.load(asset)
    checks = {
        "producer": py_compile.compile(str(producer), doraise=True) is not None,
        "source": report["all_checks_passed"] and report["passed"] == report["total"] == 10,
        "schema": payload["schema"] == "c104_upstream_role_barcode_heatmap.v1" and payload["result_type"] == "upstream_role_barcode_heatmap",
        "coordinates": len(payload["dimensions"]) == 2560 and len(payload["default_coordinates"]) == 64,
        "effects": len(payload["effect_rows"]) == 40 and all(len(row["values"]) == 2560 for row in payload["effect_rows"]),
        "raw": len(payload["raw_rows"]) > 400 and all(len(row["values"]) == 2560 for row in payload["raw_rows"]),
        "correction": payload["headline"]["fully_controlled_intervention_passed"] == 2,
        "identity": core.sha(asset) == core.sha(PUBLIC) == report["sha256"],
        "authorization": report["authorization"] == "integrate_build_and_close_c104_c105_major_stage",
    }
    result = {"phase": 1594, "campaign": "C104-C105", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(C104 / "audit/independent_upstream_heatmap_export_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
