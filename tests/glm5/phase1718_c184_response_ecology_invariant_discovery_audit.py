#!/usr/bin/env python3
"""Independent audit for C184 response ecology invariants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1718_c184_response_ecology_invariant_discovery"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    atlas = core.load(OUT / "analysis/invariant_atlas.json")
    final = core.load(OUT / "analysis/final.json")
    producer = Path(__file__).with_name("phase1718_c184_response_ecology_invariant_discovery.py")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "three_roles": set(atlas["role_summaries"]) == {"primary", "query", "relation"},
        "two_holdouts": len(atlas["rows"]) == 6,
        "no_cosine": "cosine" not in json.dumps(protocol["metrics"]).lower(),
        "support_recorded": all(summary["discovery_target_support"] for summary in atlas["role_summaries"].values()),
        "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {
        "phase": 1718,
        "campaign": "C184",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "authorization": final["next_authorization"],
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
