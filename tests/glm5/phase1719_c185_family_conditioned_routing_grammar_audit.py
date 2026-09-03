#!/usr/bin/env python3
"""Independent audit for C185 family-conditioned routing profiles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1719_c185_family_conditioned_routing_grammar"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    atlas = core.load(OUT / "analysis/family_routing_atlas.json")
    final = core.load(OUT / "analysis/final.json")
    producer = Path(__file__).with_name("phase1719_c185_family_conditioned_routing_grammar.py")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "126_rows": len(atlas["rows"]) == 126,
        "seven_families": len(atlas["families"]) == 7,
        "profiles": set(protocol["profiles"]) == {"routing", "source", "target"},
        "no_cosine": "cosine" not in json.dumps(protocol["profiles"]).lower(),
        "hash": core.sha(producer) == protocol["producer_sha256"],
    }
    result = {
        "phase": 1719,
        "campaign": "C185",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "authorization": final["next_authorization"],
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
