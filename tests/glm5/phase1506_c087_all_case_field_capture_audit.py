#!/usr/bin/env python3
"""Independent audit for Phase1506."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1506_c087_all_case_field_capture"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    meta = core.load(OUT / "analysis/capture_metadata.json")
    index = core.rows(OUT / "raw/all_role_field_index.jsonl")
    arr = np.load(OUT / "raw/all_role_field.float16.npy", mmap_mode="r")
    py_compile.compile(str(TESTS / "phase1506_c087_all_case_field_capture.py"), doraise=True)
    checks = {
        "shape": list(arr.shape) == meta["shape"] == [864, 37, 3, 2560],
        "index": len(index) == 864 and all(row["row_index"] == i for i, row in enumerate(index)),
        "hashes": core.sha(OUT / "raw/all_role_field.float16.npy") == meta["raw_sha256"] and core.sha(OUT / "raw/all_role_field_index.jsonl") == meta["index_sha256"],
        "finite": bool(np.isfinite(np.asarray(arr)).all()),
        "semantic": sum(row["semantic_match"] for row in index) == 432,
        "acquisition": meta["acquisition_complete"] and all(
            value for key, value in meta["checks"].items() if key != "stratum_identity"
        ),
        "gate_failure_recorded": (
            not meta["execution_identity_gate_passed"]
            and not meta["checks"]["stratum_identity"]
            and meta["evidence_scope"] == "descriptive_only_due_to_cross_execution_stratum_mismatch"
        ),
    }
    result = {"phase": 1506, "campaign": "C087", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
