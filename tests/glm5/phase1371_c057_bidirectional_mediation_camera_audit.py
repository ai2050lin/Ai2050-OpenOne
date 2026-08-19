#!/usr/bin/env python3
"""Independent artifact audit for Phase1371."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1371_c057_bidirectional_mediation_camera"
SCRIPT = TESTS / "phase1371_c057_bidirectional_mediation_camera.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/calibration_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_systems.jsonl")
    camera = core.rows(OUT / "raw/qwen_exact_shape_identity.jsonl")
    gate = manifest["camera_gate"]
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    recomputed = {
        "known_count": len(known) == gate["known_truth_systems"],
        "known_balance": sum(r["topology"] == "serial" for r in known) ==
                         sum(r["topology"] == "parallel" for r in known) == 128,
        "topology_exact": all(r["topology_correct"] for r in known),
        "bidirectional_exact": all(r["suff_gain"] == 2.0 and r["necessity_damage"] == 2.0 for r in known),
        "coordinate_routes": all(summary["known_truth"]["coordinate_routes_recovered"].values()),
        "camera_records": len(camera) == gate["calibration_cases"] * len(manifest["paths"]) * 8,
        "camera_output": max(r["output_max_abs_diff"] for r in camera) <= gate["same_shape_output_max_abs_diff"],
        "camera_checkpoint": max(r["checkpoint_relative_l2_max"] for r in camera) <= gate["same_shape_checkpoint_relative_l2_max"],
        "checks_consistent": all(summary["checks"].values()) == summary["camera_qualified"],
        "final_consistent": final["camera_qualified"] == summary["camera_qualified"],
        "authorization": final["authorization"] == "run_phase1372_c057_whole_state_bidirectional",
        "scripts_compile": True,
    }
    audit = {
        "phase": 1371, "campaign": "C057", "checks": recomputed,
        "passed": sum(recomputed.values()), "total": len(recomputed),
        "all_checks_passed": all(recomputed.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
