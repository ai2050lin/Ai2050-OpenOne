#!/usr/bin/env python3
"""Independent audit for Phase1513."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1513_c088_unified_forward_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def finite_chunks(arr, chunk=32):
    return all(bool(np.isfinite(np.asarray(arr[start:start + chunk])).all()) for start in range(0, len(arr), chunk))


def main() -> None:
    summary = core.load(OUT / "analysis/unified_behavior_and_capture_summary.json")
    rows = core.rows(OUT / "raw/all_role_field_index.jsonl")
    groups = core.rows(OUT / "material/stratified_composition_sets.jsonl")
    arr = np.load(OUT / "raw/all_role_field.float16.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1513_c088_unified_forward_capture.py"), doraise=True)
    checks = {
        "shape": list(arr.shape) == [1984, 37, 4, 2560],
        "index": len(rows) == 1984 and all(row["row_index"] == i for i, row in enumerate(rows)),
        "groups": len(groups) == 248 and sum(row["case_count"] for row in groups) == 1984,
        "hashes": core.sha(OUT / "raw/all_role_field.float16.npy") == summary["files"]["field"]["sha256"] and core.sha(OUT / "raw/all_role_field_index.jsonl") == summary["files"]["index"]["sha256"],
        "finite": finite_chunks(arr),
        "behavior": sum(row["correct"] for row in rows) / len(rows) == summary["global_accuracy"],
        "repeat": summary["runtime"]["numeric_repeat_max_abs_diff"] <= 1e-6,
        "single_forward": summary["single_authoritative_forward"],
        "summary": all(summary["checks"].values()),
    }
    result = {"phase": 1513, "campaign": "C088", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
