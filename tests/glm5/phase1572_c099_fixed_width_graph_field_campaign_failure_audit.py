#!/usr/bin/env python3
"""Independent audit of C099 capture success and adapter-only closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1572_c099_fixed_width_graph_field_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    py_compile.compile(str(TESTS / "phase1572_c099_fixed_width_graph_field_campaign.py"), doraise=True)
    pre = core.load(OUT / "audit/pre_model_correction_audit.json")
    capture = core.load(OUT / "analysis/capture_summary.json")
    adapter = core.load(OUT / "analysis/analysis_adapter_failure.json")
    final = core.load(OUT / "analysis/final.json")
    raw_path = OUT / "raw/all_token_all_state_field.float16.npy"
    index_path = OUT / "raw/all_token_field_index.jsonl"
    raw = np.load(raw_path, mmap_mode="r")
    index_rows = sum(1 for line in index_path.open("r", encoding="utf-8") if line.strip())
    sample = np.asarray(raw[:, [0, raw.shape[1] // 2, raw.shape[1] - 1], :32], dtype=np.float32)
    checks = {
        "correction_audit": pre["all_checks_passed"] and pre["passed"] == 9,
        "capture_checks": all(capture["checks"].values()) and len(capture["checks"]) == 9,
        "numeric_exact": all(value == 0.0 for value in capture["numeric"].values()),
        "raw_shape": tuple(raw.shape) == tuple(capture["shape"]) == (37, 191600, 2560),
        "raw_dtype": raw.dtype == np.float16,
        "raw_bytes": raw_path.stat().st_size == capture["bytes"] == 36296704128,
        "raw_sample_finite": bool(np.isfinite(sample).all()),
        "reported_hashes": len(capture["raw_sha256"]) == 64 and len(capture["index_sha256"]) == 64,
        "index_coverage": index_rows == 1152,
        "behavior_typed": capture["behavior"]["stratum"] == "descriptive; behavior does not stop C099",
        "adapter_only_failure": adapter["status"] == "analysis_not_started_authorization_literal_mismatch" and adapter["hidden_field_loaded_by_failed_call"] is False,
        "formal_closure": final["status"] == "numeric_capture_passed_analysis_adapter_closed" and final["hidden_structure_analyzed"] is False,
        "authorized_adapter": final["authorization"] == "run_C100_immutable_field_analysis_adapter",
    }
    result = {
        "phase": 1572,
        "campaign": "C099",
        "audit_type": "capture_success_adapter_failure_path",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_failure_path_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
