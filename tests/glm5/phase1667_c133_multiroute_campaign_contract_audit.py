#!/usr/bin/env python3
"""Independent audit of the C133 A-E campaign preregistration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1667_c133_multiroute_campaign_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    checks = {
        "internal": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "routes": list(protocol["routes"]) == list("ABCDE"),
        "continue_after_failure": "continue with all other" in protocol["route_policy"],
        "observation_first": protocol["priority"][:3] == ["observe", "find repeatable structure", "predict unseen trajectories"],
        "causal_last": protocol["priority"][-1] == "causal adjudication last",
        "all_coordinates": protocol["shared_measurement"]["coordinates"] == "all physical activation coordinates",
        "no_reduction_or_modules": set(("PCA", "SVD", "attention inspection", "MLP inspection")).issubset(protocol["shared_measurement"]["forbidden"]),
        "anchor_scope": protocol["routes"]["B"]["anchor_limit"] == 12 and protocol["resource_limits"]["large_sample_full_token_not_claimed"],
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "boundary": protocol["claim_boundary"] == "a campaign plan, not model evidence or a discovered operator",
    }
    report = {"phase": 1667, "campaign": "C133", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "start_route_A_C134" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
