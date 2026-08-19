#!/usr/bin/env python3
"""Independent audit for Phase1414."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1412_c067_paired_state_composition_contract"
OUT = TESTS / "result/phase1414_c067_dual_write_camera"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/camera_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_systems.jsonl")
    qwen = core.rows(OUT / "raw/qwen_dual_write_identity.jsonl")
    expected = "run_phase1415_c067_paired_composition" if summary["camera_qualified"] else "close_c067_at_camera_gate"
    checks = {
        "known": len(known) == 256 and all(all(row[key] for key in ("dual_write_exact", "self_dual_exact", "unwritten_exact", "shape_exact")) for row in known),
        "qwen_count": len(qwen) == protocol["camera"]["qwen_discovery_sets"],
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "state16": {row["state_index"] for row in qwen} == {16},
        "roles": all(len(row["record_points"]) == len(row["query_points"]) == 1 for row in qwen),
        "identity": max(row["output_max_abs_diff"] for row in qwen) <= protocol["camera"]["self_dual_max_abs_diff"],
        "finite": all(math.isfinite(row["output_max_abs_diff"]) for row in qwen),
        "decision": final["authorization"] == expected,
    }
    result = {
        "phase": 1414,
        "campaign": "C067",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
