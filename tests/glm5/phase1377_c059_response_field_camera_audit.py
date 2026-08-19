#!/usr/bin/env python3
"""Independent audit for Phase1377."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1377_c059_response_field_camera"
SCRIPT = TESTS / "phase1377_c059_response_field_camera.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/calibration_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_response_systems.jsonl")
    camera = core.rows(OUT / "raw/qwen_exact_shape_identity.jsonl")
    gate = manifest["camera_gate"]
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    checks = {
        "known_records": len(known) == gate["known_truth_systems"] * 12,
        "known_balance": sum(r["topology"] == "serial" for r in known) ==
                         sum(r["topology"] == "parallel" for r in known),
        "dose_exact": summary["known_truth"]["correct_alpha_exact"] and
                      summary["known_truth"]["correct_eta_exact"],
        "distance_exact": max(r["decomposition_relative_error"] for r in known) <=
                          manifest["distance_gate"]["direction_decomposition_relative_error_max"],
        "topology_exact": summary["known_truth"]["topology_exact"],
        "coordinate_routes": all(summary["known_truth"]["coordinate_routes_recovered"].values()),
        "camera_records": len(camera) == gate["qwen_cases"] * len(manifest["paths"]) * len(manifest["target_layout"]),
        "camera_output": max(r["output_max_abs_diff"] for r in camera) <= gate["same_shape_output_max_abs_diff"],
        "camera_checkpoint": max(r["checkpoint_relative_l2_max"] for r in camera) <= gate["same_shape_checkpoint_relative_l2_max"],
        "checks_consistent": summary["camera_qualified"] == all(summary["checks"].values()),
        "final_consistent": final["camera_qualified"] == summary["camera_qualified"],
        "authorization": final["authorization"] == "run_phase1378_c059_dose_distance_observation",
        "scripts_compile": True,
    }
    audit = {"phase": 1377, "campaign": "C059", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks),
             "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
