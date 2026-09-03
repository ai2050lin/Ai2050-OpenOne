#!/usr/bin/env python3
"""Independent audit for Phase1526 causal-prefix failure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1526_c089_full_dimensional_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    summary = core.load(OUT / "analysis/causal_prefix_identity_failure.json")
    protocol = core.load(OUT / "protocol/singleton_calibration_protocol.json")
    cases = core.rows(OUT / "protocol/singleton_calibration_cases.jsonl")
    pairs = core.rows(OUT / "analysis/causal_prefix_pair_diagnostics.jsonl")
    py_compile.compile(str(TESTS / "phase1526_c089_full_dimensional_diagnostics.py"), doraise=True)
    checks = {
        "status": final["status"] == "left_padding_camera_failed_causal_prefix_identity",
        "violation": summary["source_truth_contrast_all_state_max_abs"] > 1e-2,
        "state0": summary["source_truth_contrast_max_abs_by_state"][0] == 0.0,
        "pairs": len(pairs) == 180 and any(row["all_state_max_abs"] > 1e-2 for row in pairs),
        "partitions": all(value > 0 for value in summary["pair_violation_counts"]["by_partition"].values()),
        "calibration": len(cases) == protocol["case_count"] == 72 and core.sha(OUT / "protocol/singleton_calibration_cases.jsonl") == protocol["case_sha256"],
        "scope": summary["adjudication"]["semantic_result"] == "not tested by a qualified camera",
        "no_model": not summary["model_run"],
        "checks": all(summary["checks"].values()),
        "authorization": final["authorization"] == "run_phase1527_c090_singleton_numeric_calibration",
    }
    result = {"phase": 1526, "campaign": "C089", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
