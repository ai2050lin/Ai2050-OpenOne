#!/usr/bin/env python3
"""Independent audit for C172."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1706_c172_typed_response_graph_master_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "seven_arms": len(protocol["campaign_arms"]) == 7,
        "signed_primary_metrics": "signed_nrmse" in protocol["primary_metrics"],
        "route_level_branching": "failed arm" in protocol["branch_policy"],
        "no_component_attribution": all(x in protocol["forbidden"] for x in ("attention attribution", "MLP attribution", "weights")),
        "hashes": all(len(v) == 64 for v in protocol["source_hashes"].values()),
    }
    result = {"phase": 1706, "campaign": "C172", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
