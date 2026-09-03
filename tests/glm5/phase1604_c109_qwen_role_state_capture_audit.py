#!/usr/bin/env python3
"""Independent audit for the C109 exact-BF16 role-state archive."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1604_c109_qwen_role_state_capture.py"
    finalizer = TESTS / "phase1604_c109_capture_finalize.py"
    py_compile.compile(str(producer), doraise=True)
    py_compile.compile(str(finalizer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/capture_summary.json")
    field_path = OUT / protocol["archive"]["path"]
    logits_path = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index_path = OUT / "raw/qwen3_behavior_index.jsonl"
    field = np.load(field_path, mmap_mode="r")
    logits = np.load(logits_path, mmap_mode="r")
    index = core.rows(index_path)
    checks = {
        "producer": core.sha(producer) == report["capture_producer_sha256"],
        "finalizer": core.sha(finalizer) == report["finalizer_sha256"],
        "source_checks": all(report["checks"].values()),
        "shape": list(field.shape) == protocol["archive"]["shape"],
        "dtype": field.dtype == np.uint16,
        "bytes": field.nbytes == protocol["archive"]["expected_data_bytes"],
        "logits": bool(logits.shape == (384, 2) and logits.dtype == np.float32 and np.isfinite(logits).all()),
        "index": len(index) == 384 and all(row["row_index"] == i for i, row in enumerate(index)),
        "hashes": core.sha(field_path) == report["raw_sha256"] and core.sha(logits_path) == report["logits_sha256"] and core.sha(index_path) == report["index_sha256"],
        "causal": report["numeric"]["causal_prefix_max_abs"] == 0.0 and report["numeric"]["code_previsible_max_abs"] == 0.0,
        "authorization": report["authorization"] == "run_phase1605_c109_basic_coordinate_observation",
    }
    result = {"phase": 1604, "campaign": "C109", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_capture_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
