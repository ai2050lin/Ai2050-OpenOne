#!/usr/bin/env python3
"""Independent audit for Phase1577 C101 analysis."""
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
    producer = TESTS / "phase1577_c101_dual_arm_analysis.py"
    py_compile.compile(str(producer), doraise=True)
    adapter = core.load(OUT / "protocol/analysis_adapter.json")
    capture = core.load(OUT / "audit/independent_capture_audit.json")
    checks = {
        "producer": adapter["producer_sha256"] == core.sha(producer),
        "capture": capture["all_checks_passed"],
        "primary": adapter["primary"]["state"] == 24 and adapter["primary"]["threshold"] == 0.5,
        "null": adapter["null"]["draws"] == 1000 and adapter["null"]["quantile"] == 0.99,
        "breadth": adapter["breadth"]["status"] == "exploratory" and adapter["breadth"]["no_universal_gate"],
        "authorization": adapter["authorization"] == "execute_c101_analysis",
    }
    result = {"phase": 1577, "campaign": "C101", "stage": "pre_analysis", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_analysis_audit.json", result)
    print(json.dumps(result, indent=2))


def post() -> None:
    final = core.load(OUT / "analysis/final.json")
    result = final["result"]
    conf_path = OUT / "raw/qwen3_confirmation_walsh_coefficients.float32.npy"
    breadth_path = OUT / "raw/qwen3_breadth_walsh_coefficients.float32.npy"
    conf = np.load(conf_path, mmap_mode="r")
    breadth = np.load(breadth_path, mmap_mode="r")
    conf_sample = np.asarray(conf[[0, -1], :, [0, 24, 36], :, :32], dtype=np.float32)
    breadth_sample = np.asarray(breadth[[0, -1], :, [0, 24, 36], :, :32], dtype=np.float32)
    null = core.rows(OUT / "analysis/c101_design_preserving_null.jsonl")
    checks = {
        "producer": final["all_checks_passed"] and final["passed"] == final["total"],
        "conf_shape": list(conf.shape) == result["coefficients"]["confirmation"]["shape"],
        "breadth_shape": list(breadth.shape) == result["coefficients"]["breadth"]["shape"],
        "hashes": core.sha(conf_path) == result["coefficients"]["confirmation"]["sha256"] and core.sha(breadth_path) == result["coefficients"]["breadth"]["sha256"],
        "finite": bool(np.isfinite(conf_sample).all() and np.isfinite(breadth_sample).all()),
        "primary_count": result["confirmation"]["primary"]["required"] == 24,
        "null": len(null) == 32 and all(row["draws"] == 1000 for row in null),
        "scope": "weight parameters" in result["claim_boundary"]["forbidden"] and result["flags"]["behavior_missingness"] == "M_BEHAVIOR",
    }
    audited = {"phase": 1577, "campaign": "C101", "stage": "post_analysis", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not audited["all_checks_passed"]:
        raise RuntimeError(audited)
    core.save(OUT / "audit/independent_analysis_audit.json", audited)
    print(json.dumps(audited, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre", "post"))
    args = parser.parse_args()
    pre() if args.stage == "pre" else post()


if __name__ == "__main__":
    main()
