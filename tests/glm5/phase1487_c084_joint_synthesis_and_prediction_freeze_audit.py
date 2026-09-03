#!/usr/bin/env python3
"""Independent audit for Phase1487."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1487_c084_joint_synthesis_and_prediction_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    synthesis = core.load(OUT / "analysis/synthesis.json")
    manifest = core.load(OUT / "frozen/future_prediction_manifest.json")
    coord = core.load(RESULT / "phase1485_c084_coordinate_stability_atlas/analysis/coordinate_atlas_summary.json")
    factor = core.load(RESULT / "phase1486_c084_factorial_surface_atlas/analysis/factorial_atlas_summary.json")
    py_compile.compile(str(TESTS / "phase1487_c084_joint_synthesis_and_prediction_freeze.py"), doraise=True)
    unsigned = {key: value for key, value in manifest.items() if key != "freeze_sha256"}
    checks = {
        "status": final["status"] == "joint_synthesis_complete_with_refined_retrospective_candidates",
        "digest": manifest["freeze_sha256"] == core.digest(unsigned) == synthesis["prediction_freeze_sha256"],
        "prediction_count": len(manifest["predictions"]) == 6,
        "early_correction": synthesis["findings"]["F084_1_early_geometry_correction"]["centroid_norm_ratio"] == coord["state0_cyclic_geometry"]["global|query_label"]["centroid_norm_over_mean_vector_norm"],
        "coordinate_band": synthesis["findings"]["F084_3_thresholded_coordinate_band"]["threshold_intersection_counts"] == {str(row["fraction"]): row["intersection_count"] for row in coord["boundary_state35"]["thresholds"]},
        "factorial": synthesis["findings"]["F084_4_factorial_specificity"]["relation_coefficient_energy_fraction"]["minimum"] >= 0.98,
        "reproduction": factor["relation_reproduction_max_abs"] == 0.0,
        "claim_boundary": "new mathematics" in synthesis["adjudication"]["reject"] and synthesis["theory_update"]["status"].startswith("descriptive decomposition"),
        "no_model": not final["model_run"],
    }
    result = {"phase": 1487, "campaign": "C084", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
