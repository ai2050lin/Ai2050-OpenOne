#!/usr/bin/env python3
"""Independent audit for the C110 fresh-field prediction adjudication."""
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
    producer = TESTS / "phase1609_c110_field_prediction_adjudication.py"
    py_compile.compile(str(producer), doraise=True)
    report = core.load(OUT / "analysis/field_prediction_adjudication.json")
    internal = core.load(OUT / "audit/internal_field_adjudication_audit.json")
    results = core.rows(OUT / "analysis/fresh_field_prediction_results.jsonl")
    unit = np.load(OUT / "analysis/unit_truth_role_state.float32.npy", mmap_mode="r")
    mean = np.load(OUT / "analysis/mean_truth_role_state.float32.npy", mmap_mode="r")
    checks = {
        "producer": core.sha(producer) == report["producer_sha256"] == internal["producer_sha256"], "internal": internal["all_checks_passed"],
        "unit": unit.shape == (24, 7, 37, 2560) and core.sha(OUT / "analysis/unit_truth_role_state.float32.npy") == report["unit_sha256"],
        "mean": mean.shape == (2, 2, 7, 37, 2560) and core.sha(OUT / "analysis/mean_truth_role_state.float32.npy") == report["mean_sha256"],
        "results": len(results) == 2 and all(set(row["gates"]) == {"cross_partition", "reference", "support_overlap"} for row in results),
        "boundary": "readout stability only" in report["interpretation"],
        "authorization": report["authorization"] == "execute_phase1610_c110_frozen_transport_comparison_regardless_of_field_gate",
    }
    result = {"phase": 1609, "campaign": "C110", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/independent_field_adjudication_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
