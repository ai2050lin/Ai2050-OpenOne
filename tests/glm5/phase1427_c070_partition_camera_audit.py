#!/usr/bin/env python3
"""Independent audit for Phase1427."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1425_c070_quartet_complement_contract"
OUT = TESTS / "result/phase1427_c070_partition_camera"
ROLES = ("record_target", "record_family", "query_target", "query_family")


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/camera_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_systems.jsonl")
    qwen = core.rows(OUT / "raw/qwen_full_state_transport.jsonl")
    expected = "run_phase1428_c070_support_partition" if summary["camera_qualified"] else "close_c070_at_camera_gate"
    checks = {
        "known": len(known) == 256 and all(
            all(row[key] for key in ("partition_disjoint", "partition_complete", "quartet_exact", "complement_exact", "full_exact", "self_exact"))
            for row in known
        ),
        "qwen_count": len(qwen) == protocol["camera"]["qwen_discovery_sets"] * 2,
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "directions": {row["direction"] for row in qwen} == {"true_to_false", "false_to_true"},
        "state16": {row["state_index"] for row in qwen} == {16},
        "roles": all(all(len(row["role_points"][role]) == 1 for role in ROLES) for row in qwen),
        "same_shape": {row["sequence_length"] for row in qwen} == {protocol["material"]["prompt_token_length"]},
        "self": max(row["self_full_max_abs_diff"] for row in qwen) <= protocol["camera"]["self_full_max_abs_diff"],
        "donor": max(row["donor_full_max_abs_diff"] for row in qwen) <= protocol["camera"]["donor_full_transport_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in qwen for key in ("self_full_max_abs_diff", "donor_full_max_abs_diff")),
        "decision": final["authorization"] == expected,
    }
    result = {
        "phase": 1427,
        "campaign": "C070",
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
