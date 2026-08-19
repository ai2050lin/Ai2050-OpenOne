#!/usr/bin/env python3
"""Independent audit for Phase1378."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1378_c059_dose_distance_observation"
SCRIPT = TESTS / "phase1378_c059_dose_distance_observation.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_dose_distance_summary.json")
    final = core.load(OUT / "analysis/final.json")
    records = core.rows(OUT / "raw/qwen3_dose_response.jsonl")
    distances = core.rows(OUT / "raw/qwen3_pairwise_distance.jsonl")
    fields = core.rows(OUT / "raw/response_discovery_field_index.jsonl")
    geometry = core.load(OUT / "analysis/full_geometry_summary.json")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    layout_n = len(manifest["layout"])
    expected_records = manifest["case_count"] * len(manifest["paths"]) * layout_n
    expected_distances = manifest["case_count"] * len(manifest["paths"]) * 2 * 6 * 2
    field_files_ok = all((OUT / r["file"]).is_file() and core.sha(OUT / r["file"]) == r["sha256"] for r in fields)
    checks = {
        "record_count": len(records) == expected_records == summary["record_count"],
        "distance_count": len(distances) == expected_distances == summary["distance_record_count"],
        "field_count": len(fields) == 72 == summary["discovery_field_count"],
        "field_files": field_files_ok,
        "partitions": {r["partition"] for r in records} == {"response_discovery", "confirmation", "lockbox"},
        "all_arms": {(r["mode"], r["direction"], r["dose"]) for r in records} ==
                    {(m, d, float(l)) for m in ("sufficiency", "reverse")
                     for d in manifest["dose_gate"]["directions"] for l in manifest["dose_gate"]["values"]},
        "finite": all(r["all_finite"] and math.isfinite(r["output_effect"]) for r in records),
        "norm_match": max(r["norm_ratio_abs_error"] for r in records) <= manifest["dose_gate"]["norm_ratio_abs_error_max"],
        "lambda0_identity": max(abs(r["output_effect"]) for r in records if r["dose"] == 0.0) <=
                            manifest["dose_gate"]["self_output_max_abs_diff"],
        "distance_decomposition": max(r["decomposition_relative_error_max"] for r in records) <=
                                  manifest["distance_gate"]["direction_decomposition_relative_error_max"],
        "summary_metrics": all(summary["path_metrics"][p]["count"] == manifest["case_count"] for p in manifest["paths"]),
        "geometry_postanalysis": geometry["postprocessing_only"] and
                                 not geometry["thresholds_or_eligibility_changed"] and
                                 geometry["record_count"] == len(records) and
                                 geometry["distance_record_count"] == len(distances),
        "eligibility_consistent": final["path_eligibility"] == summary["path_eligibility"],
        "mediation_consistent": final["mediation_eligible"] == summary["mediation_eligible"],
        "authorization": final["authorization"] == "run_phase1379_c059_coordinate_group_evaluation",
        "scripts_compile": True,
    }
    audit = {"phase": 1378, "campaign": "C059", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks),
             "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
