#!/usr/bin/env python3
"""Independent audit for the Phase1578 C101 index adapter."""
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
    producer = TESTS / "phase1578_c101_analysis_index_adapter.py"
    py_compile.compile(str(producer), doraise=True)
    adapter = core.load(OUT / "protocol/analysis_index_adapter.json")
    audit = core.load(OUT / "audit/pre_analysis_index_adapter_audit.json")
    failure = core.load(OUT / "analysis/phase1577_adapter_failure.json")
    checks = {
        "producer": adapter["producer_sha256"] == core.sha(producer),
        "audit": audit["all_checks_passed"],
        "failure_typed": failure["scientific_result"] == "none" and not failure["walsh_coefficients_completed"],
        "single_change": adapter["single_change"].startswith("derive unit identity"),
        "authorization": adapter["authorization"] == "execute_phase1578_analysis_adapter",
    }
    result = {"phase": 1578, "campaign": "C101", "stage": "pre", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_pre_index_adapter_audit.json", result)
    print(json.dumps(result, indent=2))


def post() -> None:
    adapter_final = core.load(OUT / "analysis/phase1578_adapter_final.json")
    final = core.load(OUT / "analysis/final.json")
    conf_path = OUT / "raw/qwen3_confirmation_walsh_coefficients_v2.float32.npy"
    breadth_path = OUT / "raw/qwen3_breadth_walsh_coefficients_v2.float32.npy"
    conf = np.load(conf_path, mmap_mode="r")
    breadth = np.load(breadth_path, mmap_mode="r")
    null = core.rows(OUT / "analysis/c101_design_preserving_null.jsonl")
    checks = {
        "adapter": adapter_final["scientific_checks_passed"] and adapter_final["scientific_final_sha256"] == core.sha(OUT / "analysis/final.json"),
        "producer": final["all_checks_passed"],
        "shapes": list(conf.shape) == [72, 15, 37, 6, 2560] and list(breadth.shape) == [48, 15, 37, 7, 2560],
        "hashes": core.sha(conf_path) == final["result"]["coefficients"]["confirmation"]["sha256"] and core.sha(breadth_path) == final["result"]["coefficients"]["breadth"]["sha256"],
        "null": len(null) == 32 and all(row["draws"] == 1000 for row in null),
        "scope": final["result"]["flags"]["behavior_missingness"] == "M_BEHAVIOR",
    }
    result = {"phase": 1578, "campaign": "C101", "stage": "post", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pre", "post"))
    args = parser.parse_args()
    pre() if args.stage == "pre" else post()


if __name__ == "__main__":
    main()
