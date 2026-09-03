#!/usr/bin/env python3
"""Independent audit for Phase1447 C074 identity camera."""
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

PHASE, CAMPAIGN = 1447, "C074"
CONTRACT = TESTS / "result/phase1445_c074_directional_domain_contract"
OUT = TESTS / "result/phase1447_c074_identity_camera"


def main() -> None:
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    summary = core.load(OUT / "analysis/camera_summary.json")
    final = core.load(OUT / "analysis/final.json")
    known = core.rows(OUT / "raw/known_truth_identity.jsonl")
    rows = core.rows(OUT / "raw/qwen_identity_camera.jsonl")
    gate = protocol["camera"]
    expected = "run_phase1448_c074_directional_domain_map" if summary["camera_qualified"] else "close_c074_at_camera_gate"
    checks = {
        "known": len(known) == gate["known_truth_systems"] * len(gate["arms"]) and all(row["roles_exact"] and row["complement_exact"] for row in known),
        "count": len(rows) == gate["qwen_discovery_sets"] * len(gate["routes"]) * len(gate["directions"]) * len(gate["arms"]),
        "balance": Counter((row["route"], row["direction"], row["arm"]) for row in rows) == {(route, direction, arm): gate["qwen_discovery_sets"] for route in gate["routes"] for direction in gate["directions"] for arm in gate["arms"]},
        "sets": len({row["set_id"] for row in rows}) == gate["qwen_discovery_sets"],
        "families": len({row["family"] for row in rows}) == 6,
        "discovery": {row["partition"] for row in rows} == {"response_discovery"},
        "metadata": all(protocol["routes"][row["route"]][key] == row[key] for row in rows for key in ("same_surface", "same_frame", "same_order")),
        "exact": max(row["write_max_abs_diff"] for row in rows) <= gate["write_max_abs_diff"] and max(row["complement_max_abs_diff"] for row in rows) <= gate["untouched_complement_max_abs_diff"],
        "self": max(row["self_output_max_abs_diff"] for row in rows if row["arm"] == "self") <= gate["self_output_max_abs_diff"],
        "finite": all(math.isfinite(row[key]) for row in rows for key in ("write_max_abs_diff", "complement_max_abs_diff", "self_output_max_abs_diff")),
        "decision": summary["camera_qualified"] == all(summary["checks"].values()),
        "authorization": final["authorization"] == expected,
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
