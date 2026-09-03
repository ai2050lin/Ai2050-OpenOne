#!/usr/bin/env python3
"""Independent audit for the C102 heatmap export."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c102_coordinate_barcode_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1586_c102_coordinate_barcode_heatmap_export.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "analysis/heatmap_export.json")
    payload = core.load(OUT / "visualization/c102_coordinate_barcode_heatmap.json")
    checks = {
        "source": report["all_checks_passed"] and report["authorization"] == "integrate_and_build_c102_heatmap_client",
        "schema": payload["schema"] == "c102_coordinate_barcode_heatmap.v1" and payload["result_type"] == "coordinate_barcode_heatmap",
        "coordinates": payload["dimensions"] == list(range(2560)),
        "states": {row["state"] for row in payload["raw_rows"] if row["scope"] == "all_states_boundary"} == set(range(37)),
        "headline": payload["headline"]["barcode_three_stage_passed"] == 8 and payload["headline"]["controlled_intervention_passed"] == 0,
        "identity": core.sha(OUT / "visualization/c102_coordinate_barcode_heatmap.json") == core.sha(PUBLIC) == report["sha256"],
        "scope": "not a semantic code" in payload["claim_boundary"],
    }
    result = {"phase": 1586, "campaign": "C102", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_heatmap_export_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
