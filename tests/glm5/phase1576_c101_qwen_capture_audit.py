#!/usr/bin/env python3
"""Independent pre/post audit for Phase1576 C101 capture."""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1575_c101_dual_arm"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def pre() -> None:
    producer = TESTS / "phase1576_c101_qwen_capture.py"
    py_compile.compile(str(producer), doraise=True)
    adapter = core.load(OUT / "protocol/capture_adapter.json")
    parent = core.load(OUT / "audit/independent_pre_model_audit.json")
    checks = {
        "producer": adapter["producer_sha256"] == core.sha(producer),
        "parent": parent["all_checks_passed"],
        "width": 150 <= adapter["fixed_global_sequence_length"] < 320,
        "real_boundary": "real token boundary" in adapter["behavior_boundary"],
        "authorization": adapter["authorization"] == "execute_qwen_capture",
    }
    result = {"phase": 1576, "campaign": "C101", "stage": "pre_capture", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_capture_audit.json", result)
    print(json.dumps(result, indent=2))


def post() -> None:
    report = core.load(OUT / "analysis/qwen_capture_summary.json")
    raw_path = OUT / "raw/qwen3_registered_role_field.float16.npy"
    index = core.rows(OUT / "raw/qwen3_registered_role_index.jsonl")
    raw = np.load(raw_path, mmap_mode="r")
    sample = np.asarray(raw[:, [0, raw.shape[1] // 2, raw.shape[1] - 1], :64], dtype=np.float32)
    calibration = core.load(OUT / "analysis/c099_behavior_boundary_recalibration.json")
    checks = {
        "producer_checks": all(report["checks"].values()),
        "shape": list(raw.shape) == report["shape"] and raw.shape[0] == 37 and raw.shape[2] == 2560,
        "bytes": raw_path.stat().st_size == report["bytes"],
        "hash": core.sha(raw_path) == report["raw_sha256"],
        "finite": bool(np.isfinite(sample).all()),
        "index": len(index) == 1920 and core.sha(OUT / "raw/qwen3_registered_role_index.jsonl") == report["index_sha256"],
        "numeric": all(value == 0.0 for value in report["numeric"].values()),
        "calibration": calibration["archived_final_hidden_max_abs"] == 0.0 and calibration["old_vs_corrected_score_max_abs"] > 0.0,
        "authorization": report["authorization"] == "run_phase1577_c101_analysis",
    }
    result = {"phase": 1576, "campaign": "C101", "stage": "post_capture", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_capture_audit.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre", "post"))
    args = parser.parse_args()
    pre() if args.stage == "pre" else post()


if __name__ == "__main__":
    main()
