#!/usr/bin/env python3
"""Independent audit for Phase1422."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1420_c069_catalog_four_role_contract"
OUT = TESTS / "result/phase1422_c069_quartet_camera"
ROLES = ("record_target", "record_family", "query_target", "query_family")


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/camera_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_systems.jsonl")
    qwen = core.rows(OUT / "raw/qwen_quartet_identity.jsonl")
    expected = "run_phase1423_c069_bidirectional_composition" if summary["camera_qualified"] else "close_c069_at_camera_gate"
    checks = {
        "known": len(known) == 256 and all(
            all(row[key] for key in ("quartet_write_exact", "self_quartet_exact", "unwritten_exact", "shape_exact"))
            for row in known
        ),
        "qwen_count": len(qwen) == protocol["camera"]["qwen_discovery_sets"] * 2,
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "directions": {row["direction"] for row in qwen} == {"true_recipient", "false_recipient"},
        "state16": {row["state_index"] for row in qwen} == {16},
        "roles": all(all(len(row["role_points"][role]) == 1 for role in ROLES) for row in qwen),
        "identity": max(row["output_max_abs_diff"] for row in qwen) <= protocol["camera"]["self_quartet_max_abs_diff"],
        "finite": all(math.isfinite(row["output_max_abs_diff"]) for row in qwen),
        "decision": final["authorization"] == expected,
    }
    result = {
        "phase": 1422,
        "campaign": "C069",
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
