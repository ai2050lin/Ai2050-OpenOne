#!/usr/bin/env python3
"""Independent audit for Phase1437 C072 permutation camera."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
OUT = TESTS / "result/phase1437_c072_permutation_camera"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    registry = core.rows(CONTRACT / "material/permutation_registry.jsonl")
    known = core.rows(OUT / "raw/known_truth_permutations.jsonl")
    qwen = core.rows(OUT / "raw/qwen_permutation_camera.jsonl")
    summary = core.load(OUT / "analysis/camera_summary.json")
    final = core.load(OUT / "analysis/final.json")
    gate = protocol["camera"]
    balance = Counter((row["surface_transfer"], row["direction"], row["permutation_id"]) for row in qwen)
    checks = {
        "known": len(known) == 128 * 24 and all(row["roles_exact"] and row["complement_exact"] for row in known),
        "known_balance": Counter(row["permutation_id"] for row in known) == {row["permutation_id"]: 128 for row in registry},
        "qwen": len(qwen) == 1152 and len({row["set_id"] for row in qwen}) == 12,
        "balance": len(balance) == 2 * 2 * 24 and set(balance.values()) == {12},
        "discovery": {row["partition"] for row in qwen} == {"response_discovery"},
        "state16": {row["state_index"] for row in qwen} == {16},
        "shapes": all(row["source_length"] != row["target_length"] for row in qwen),
        "quartet": all(row["quartet_size"] == 4 for row in qwen),
        "writes": max(row["write_max_abs_diff"] for row in qwen) <= gate["write_max_abs_diff"],
        "complement": max(row["complement_max_abs_diff"] for row in qwen) <= gate["untouched_complement_max_abs_diff"],
        "self": max(max(row["self_role_max_abs_diff"], row["self_output_max_abs_diff"]) for row in qwen) <= gate["self_output_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in qwen for key in ("write_max_abs_diff", "complement_max_abs_diff", "self_output_max_abs_diff")),
        "decision": summary["camera_qualified"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == ("run_phase1438_c072_permutation_spectrum" if summary["camera_qualified"] else "close_c072_at_camera_gate"),
    }
    result = {"phase": 1437, "campaign": "C072", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
