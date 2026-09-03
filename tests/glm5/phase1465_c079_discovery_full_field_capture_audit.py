#!/usr/bin/env python3
"""Independent audit for Phase1465 C079 raw capture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

BEHAVIOR = TESTS / "result/phase1464_c079_behavior"
OUT = TESTS / "result/phase1465_c079_discovery_full_field_capture"


def main() -> None:
    metadata = core.load(OUT / "analysis/capture_metadata.json")
    final = core.load(OUT / "analysis/final.json")
    index = core.rows(OUT / "raw/discovery_role_field_index.jsonl")
    field = np.load(OUT / "raw/discovery_role_field.float16.npy", mmap_mode="r")
    behavior = {row["case_id"]: row for row in core.rows(BEHAVIOR / "raw/active_behavior.jsonl")}
    finite = True
    for start in range(0, len(field), 64):
        finite = finite and bool(np.isfinite(field[start:start + 64]).all())
    checks = {
        "main": all(metadata["checks"].values()),
        "shape": list(field.shape) == metadata["shape"] == [1104, 37, 9, 2560],
        "dtype": field.dtype == np.float16,
        "index": len(index) == 1104 and [row["row_index"] for row in index] == list(range(1104)),
        "discovery": all(row["partition"] == "response_discovery" for row in index),
        "behavior": all(behavior[row["case_id"]]["correct"] for row in index),
        "finite": finite,
        "raw_hash": core.sha(OUT / "raw/discovery_role_field.float16.npy") == metadata["raw_sha256"] == final["raw_sha256"],
        "index_hash": core.sha(OUT / "raw/discovery_role_field_index.jsonl") == metadata["index_sha256"],
        "authorization": final["authorization"] == "run_phase1466_c079_discovery_basic_observation_and_freeze",
    }
    result = {"phase": 1465, "campaign": "C079", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
