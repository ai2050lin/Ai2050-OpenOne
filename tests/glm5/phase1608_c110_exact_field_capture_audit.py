#!/usr/bin/env python3
"""Independent audit of the C110 exact field capture."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    producer = TESTS / "phase1608_c110_exact_field_capture.py"
    py_compile.compile(str(producer), doraise=True)
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/capture_summary.json")
    raw = OUT / protocol["archive"]["path"]
    logits = OUT / "raw/qwen3_candidate_logits.float32.npy"
    index = OUT / "raw/qwen3_behavior_index.jsonl"
    field = np.load(raw, mmap_mode="r")
    scores = np.load(logits, mmap_mode="r")
    rows = core.rows(index)
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"], "source": all(report["checks"].values()),
        "shape": list(field.shape) == protocol["archive"]["shape"], "dtype": field.dtype == np.uint16,
        "logits": bool(scores.shape == (384, 2) and scores.dtype == np.float32 and np.isfinite(scores).all()),
        "index": len(rows) == 384 and all(row["row_index"] == i for i, row in enumerate(rows)),
        "hashes": core.sha(raw) == report["raw_sha256"] and core.sha(logits) == report["logits_sha256"] and core.sha(index) == report["index_sha256"],
        "causal": report["numeric"]["causal_prefix_max_abs"] == 0.0 and report["numeric"]["code_previsible_max_abs"] == 0.0,
        "authorization": report["authorization"] == "run_phase1609_c110_field_prediction_and_transport_tests",
    }
    result = {"phase": 1608, "campaign": "C110", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_capture_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
