#!/usr/bin/env python3
"""Independent audit for the C101 2560-coordinate heatmap export."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1575_c101_dual_arm"
CLIENT = ROOT / "frontend/public/vis_data/research_kernel/c101_activation_coordinate_heatmap.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1579_c101_activation_coordinate_heatmap_export.py"), doraise=True)
    report = core.load(OUT / "analysis/visualization_export.json")
    canonical = ROOT / report["asset"]
    payload = core.load(canonical)
    checks = {
        "producer": report["all_checks_passed"] and report["passed"] == 8,
        "schema": payload["schema"] == "c101_activation_coordinate_heatmap.v1" and payload["result_type"] == "activation_coordinate_heatmap",
        "dimensions": payload["dimensions"] == list(range(2560)),
        "rows": len(payload["walsh_rows"]) == 144 and len(payload["raw_rows"]) > 100,
        "raw_types": {row["state_kind"] for row in payload["raw_rows"]} == {"embedding", "hidden_state"},
        "identity": core.sha(canonical) == report["sha256"] == core.sha(CLIENT),
        "scope": "not weight parameters" in payload["coordinate_semantics"],
        "authorization": report["authorization"] == "integrate_c101_activation_coordinate_heatmap_client",
    }
    result = {"phase": 1579, "campaign": "C101", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_visualization_export_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
