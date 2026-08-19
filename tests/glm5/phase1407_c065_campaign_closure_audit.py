#!/usr/bin/env python3
"""Independent audit for the C065 closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1407_c065_campaign_closure"
SWAPS = TESTS / "result/phase1406_c065_holdout_factorial_swaps"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    swaps = core.load(SWAPS / "analysis/factorial_swap_summary.json")
    checks = {
        "status": final["status"] == "closed_after_confirmed_selective_whole_state_routes",
        "family": final["confirmed_routes"]["family_identity"] == swaps["route_status"]["family_identity"],
        "polarity": final["confirmed_routes"]["joint_polarity"] == swaps["route_status"]["joint_polarity"],
        "state16": final["next_prediction"]["state_index"] == 16,
        "no_search": final["next_prediction"]["no_new_candidate_search"],
        "authorization": final["authorization"] == "preregister_c066_midstate_breadth_confirmation",
    }
    result = {"phase": 1407, "campaign": "C065", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
