#!/usr/bin/env python3
"""Independent audit for Phase1379."""
from __future__ import annotations

import json
import math
import py_compile
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1379_c059_coordinate_group_evaluation"
SCRIPT = TESTS / "phase1379_c059_coordinate_group_evaluation.py"


def main() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    groups = core.load(OUT / "protocol/candidate_groups.json")
    summary = core.load(OUT / "analysis/qwen3_coordinate_group_summary.json")
    final = core.load(OUT / "analysis/final.json")
    split = core.load(OUT / "analysis/split_stability_postanalysis.json")
    records = core.rows(OUT / "raw/qwen3_coordinate_groups.jsonl")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    expected = manifest["case_count"] * len(manifest["routes"]) * len(manifest["sizes"])
    combo_counts = Counter((r["route"], r["size"]) for r in records)
    group_shapes = all(len(groups["groups"][route][str(size)]) == size
                       for route in manifest["routes"] for size in manifest["sizes"])
    expected_auth = ("run_phase1380_c059_early_mediation" if manifest["mediation_was_eligible"]
                     else "close_c059_after_all_frozen_eligible_branches")
    checks = {
        "candidate_hash": core.sha(OUT / "protocol/candidate_groups.json") == manifest["candidate_groups_sha256"],
        "candidate_source_hash": groups["candidate_source_sha256"] == manifest["candidate_source_sha256"],
        "group_shapes": group_shapes,
        "record_count": len(records) == expected == summary["record_count"],
        "combos_balanced": set(combo_counts.values()) == {manifest["case_count"]},
        "partitions": Counter(r["partition"] for r in records) ==
                      {"confirmation": expected // 2, "lockbox": expected // 2},
        "finite": summary["runtime"]["all_finite"] and all(
            math.isfinite(v) for r in records for v in list(r["suff_gain"].values()) +
            list(r["reverse_damage"].values())),
        "self": max(r["self_max_abs_diff"] for r in records) <= manifest["gate"]["self_max_abs_diff"],
        "norm": max(r["norm_ratio_abs_error_max"] for r in records) <= 1e-5,
        "qualification_shape": all("minimal_sufficiency_size" in value and "minimal_reverse_size" in value
                                   for value in summary["qualifications"].values()),
        "split_postanalysis": split["postprocessing_only"] and
                              not split["thresholds_or_eligibility_changed"] and
                              split["record_count"] == len(records),
        "final_consistent": final["any_sufficiency_route"] == summary["any_sufficiency_route"] and
                            final["any_reverse_route"] == summary["any_reverse_route"],
        "authorization": final["authorization"] == expected_auth,
        "scripts_compile": True,
    }
    audit = {"phase": 1379, "campaign": "C059", "checks": checks,
             "passed": sum(checks.values()), "total": len(checks),
             "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
