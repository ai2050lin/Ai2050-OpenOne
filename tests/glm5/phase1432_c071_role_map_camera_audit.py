#!/usr/bin/env python3
"""Independent audit for Phase1432."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1430_c071_cross_surface_role_contract"
OUT = TESTS / "result/phase1432_c071_role_map_camera"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/camera_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_systems.jsonl")
    qwen = core.rows(OUT / "raw/qwen_role_map_camera.jsonl")
    gate = protocol["camera"]
    expected = "run_phase1433_c071_cross_surface_mechanism" if summary["camera_qualified"] else "close_c071_at_camera_gate"
    checks = {
        "known": len(known) == 256 and all(all(row[key] for key in ("source_roles_distinct", "target_roles_distinct", "permutation_deranged", "self_exact", "mapped_roles_exact", "permuted_roles_exact", "mapped_complement_exact", "permuted_complement_exact")) for row in known),
        "qwen_count": len(qwen) == 96,
        "discovery_only": {row["partition"] for row in qwen} == {"response_discovery"},
        "balance": all(sum(row["surface_transfer"] == transfer and row["direction"] == direction for row in qwen) == 24 for transfer in gate["surface_transfers"] for direction in gate["directions"]),
        "state16": {row["state_index"] for row in qwen} == {16},
        "different_shapes": all(row["source_length"] != row["target_length"] for row in qwen),
        "role_points": all(len(set(row["source_role_points"].values())) == len(set(row["target_role_points"].values())) == 4 for row in qwen),
        "writes": max(row["self_role_max_abs_diff"] for row in qwen) <= 1e-4 and max(row["mapped_role_max_abs_diff"] for row in qwen) <= 1e-4 and max(row["permuted_role_max_abs_diff"] for row in qwen) <= 1e-4,
        "complements": max(row["mapped_complement_max_abs_diff"] for row in qwen) <= 1e-4 and max(row["permuted_complement_max_abs_diff"] for row in qwen) <= 1e-4,
        "self_output": max(row["self_output_max_abs_diff"] for row in qwen) <= 1e-4,
        "finite": all(math.isfinite(value) for row in qwen for key, value in row.items() if key.endswith("max_abs_diff")),
        "decision": final["authorization"] == expected,
    }
    result = {
        "phase": 1432, "campaign": "C071", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
