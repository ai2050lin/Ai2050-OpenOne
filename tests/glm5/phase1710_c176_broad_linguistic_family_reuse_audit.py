#!/usr/bin/env python3
"""Independent audit for C176."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1710_c176_broad_linguistic_family_reuse"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    final = core.load(OUT / "analysis/final.json")
    report = core.load(OUT / "analysis/broad_family_atlas.json")
    validity = core.load(OUT / "analysis/measurement_validity.json")
    checks = {
        "closed_invalid": final["status"] == "closed_invalid_measurement" and final["all_checks_passed"] and not final["scientific_result_valid"],
        "terms": len(report["formation"]) == 21,
        "sets": set(report["summary_q24"]) == {"primary", "query", "relation"},
        "distinct_objects": "distinct mathematical objects" in protocol["claim_boundary"],
        "no_reduction": "PCA" in protocol["forbidden"],
        "zero_support_found": validity["zero_fraction_by_role"]["query"] == 1.0,
        "hash": core.sha(Path(__file__).with_name("phase1710_c176_broad_linguistic_family_reuse.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1710, "campaign": "C176", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
