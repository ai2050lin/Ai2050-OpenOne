#!/usr/bin/env python3
"""Independent audit of the C120 behavior-only closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1647_c120_controlled_comparison_observation_campaign"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


if __name__ == "__main__":
    capture = core.load(OUT / "analysis/capture_summary.json")
    diagnostic = core.load(OUT / "analysis/behavior_boundary_diagnostic.json")
    closure = core.load(OUT / "analysis/closure.json")
    internal = core.load(OUT / "audit/internal_closure_audit.json")
    rows = core.rows(OUT / "raw/qwen3_behavior_index.jsonl")
    truth_positive = sum(row["correct"] for row in rows if row["truth_factor"] == 1) / 576
    truth_negative = sum(row["correct"] for row in rows if row["truth_factor"] == -1) / 576
    near = sum(row["correct"] for row in rows if row["gap_factor"] == 1) / 576
    far = sum(row["correct"] for row in rows if row["gap_factor"] == -1) / 576
    checks = {
        "internal": internal["all_checks_passed"],
        "capture": closure["headline"] == capture["behavior"] and not capture["behavior_gate_passed"],
        "cells": len(diagnostic["factor_cells"]) == 144 and all(row["n"] == 8 for row in diagnostic["factor_cells"]),
        "truth": abs(diagnostic["marginal_accuracy"]["truth_factor"]["1"]["accuracy"] - truth_positive) < 1e-12 and abs(diagnostic["marginal_accuracy"]["truth_factor"]["-1"]["accuracy"] - truth_negative) < 1e-12,
        "gap": abs(diagnostic["marginal_accuracy"]["gap_factor"]["1"]["accuracy"] - near) < 1e-12 and abs(diagnostic["marginal_accuracy"]["gap_factor"]["-1"]["accuracy"] - far) < 1e-12,
        "hashes": diagnostic["input_hashes"]["behavior_index"] == core.sha(OUT / "raw/qwen3_behavior_index.jsonl") and diagnostic["input_hashes"]["sealed_raw"] == capture["raw_sha256"],
        "sealed": "sealed" in closure["raw_archive_status"] and not (OUT / "visualization").exists(),
        "forbidden": len(diagnostic["strict_adjudication"]["forbidden_claims"]) == 4,
        "authorization": closure["next_authorization"].endswith("C121_fresh_structured_comparison_qualification"),
    }
    report = {
        "phase": 1649,
        "campaign": "C120",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": closure["next_authorization"],
    }
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))
