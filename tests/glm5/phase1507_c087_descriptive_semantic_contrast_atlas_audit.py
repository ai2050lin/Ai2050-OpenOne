#!/usr/bin/env python3
"""Independent audit for Phase1507."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
CAPTURE = RESULT / "phase1506_c087_all_case_field_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    summary = core.load(OUT / "analysis/semantic_contrast_atlas_summary.json")
    group = np.load(OUT / "atlas/group_semantic_contrast.float32.npy", mmap_mode="r")
    aggregate = np.load(OUT / "atlas/partition_stratum_semantic_mean.float32.npy", mmap_mode="r")
    index = core.rows(OUT / "atlas/group_semantic_contrast_index.jsonl")
    source = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    source_index = {row["case_id"]: row for row in core.rows(CAPTURE / "raw/all_role_field_index.jsonl")}
    behavior_groups = core.rows(RESULT / "phase1505_c087_behavior_stratification/material/stratified_composition_sets.jsonl")
    surfaces = ("a_natural", "b_natural")
    exact = True
    for gi in (0, 37, 108, 215):
        row = behavior_groups[gi]
        for ui, surface_name in enumerate(surfaces):
            a = source_index[row[f"{surface_name}_same"]]["row_index"]
            b = source_index[row[f"{surface_name}_different"]]["row_index"]
            expected = np.asarray(source[a], dtype=np.float32) - np.asarray(source[b], dtype=np.float32)
            exact = exact and bool(np.array_equal(np.asarray(group[gi, ui]), expected))
    py_compile.compile(str(TESTS / "phase1507_c087_descriptive_semantic_contrast_atlas.py"), doraise=True)
    checks = {
        "hashes": core.sha(OUT / "atlas/group_semantic_contrast.float32.npy") == summary["files"]["group"]["sha256"] and core.sha(OUT / "atlas/partition_stratum_semantic_mean.float32.npy") == summary["files"]["aggregate"]["sha256"],
        "shapes": list(group.shape) == [216, 2, 37, 3, 2560] and list(aggregate.shape) == [3, 3, 2, 37, 3, 2560],
        "index": len(index) == 216 and all(row["group_index"] == i for i, row in enumerate(index)),
        "exact_recompute": exact,
        "finite": bool(np.isfinite(np.asarray(group)).all()) and bool(np.isfinite(np.asarray(aggregate)).all()),
        "summary": all(summary["checks"].values()),
    }
    result = {"phase": 1507, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
