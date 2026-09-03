#!/usr/bin/env python3
"""Independent audit for C177."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1711_c177_missing_aware_broad_family_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = core.load(OUT / "analysis/missing_aware_atlas.json")
    checks = {
        "closed": final["status"] == "closed" and final["all_checks_passed"],
        "missing_rule": "undefined" in protocol["missing_rule"],
        "query_missing": report["role_support"]["query"]["supported_fraction"] == 0.0,
        "primary_supported": report["role_support"]["primary"]["supported_fraction"] == 1.0,
        "accounted": report["valid_transfer_rows"] + report["missing_transfer_rows"] == 5544,
        "hash": core.sha(Path(__file__).with_name("phase1711_c177_missing_aware_broad_family_atlas.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1711, "campaign": "C177", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
