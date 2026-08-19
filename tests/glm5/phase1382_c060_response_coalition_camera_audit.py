#!/usr/bin/env python3
"""Independent audit for Phase1382."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1382_c060_response_coalition_camera"
SCRIPT = TESTS / "phase1382_c060_response_coalition_camera.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/calibration_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_systems.jsonl")
    camera = core.rows(OUT / "raw/qwen_exact_shape_identity.jsonl")
    gate = manifest["camera_gate"]
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    expected_camera = gate["qwen_cases"] * len(manifest["paths"]) * sum(
        len(v) for v in manifest["mode_layouts"].values()
    )
    checks = {
        "known_records": len(known) == gate["known_truth_systems"],
        "known_balance": abs(sum(r["topology"] == "serial" for r in known) - sum(r["topology"] == "parallel" for r in known)) <= 1,
        "curve_exact": summary["known_truth"]["curve_classification_exact"],
        "coalition_exact": summary["known_truth"]["coalition_union_complement_exact"],
        "dynamic_exact": summary["known_truth"]["dynamic_mask_exact"],
        "topology_exact": summary["known_truth"]["serial_parallel_exact"],
        "camera_records": len(camera) == expected_camera,
        "camera_output": max(r["output_max_abs_diff"] for r in camera) <= gate["same_shape_output_max_abs_diff"],
        "camera_checkpoint": max(r["checkpoint_relative_l2_max"] for r in camera) <= gate["same_shape_checkpoint_relative_l2_max"],
        "checks_consistent": summary["camera_qualified"] == all(summary["checks"].values()),
        "final_consistent": final["camera_qualified"] == summary["camera_qualified"],
        "authorization": final["authorization"] == "run_phase1383_c060_refined_dose_observation",
        "scripts_compile": True,
    }
    audit = {
        "phase": 1382,
        "campaign": "C060",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
