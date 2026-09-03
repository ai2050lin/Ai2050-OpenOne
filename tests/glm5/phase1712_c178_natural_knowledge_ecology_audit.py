#!/usr/bin/env python3
"""Independent audit for C178."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1712_c178_natural_knowledge_ecology"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    eligibility = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "closed_invalid": final["status"] == "closed_behavior_interface_invalid" and final["all_checks_passed"] and not final["scientific_result_valid"],
        "behavior_before_hidden": protocol["hidden_policy"].startswith("no HiddenState"),
        "eight_families": len(protocol["families"]) == 8,
        "ineligible": len(eligibility["eligible_families"]) == 0,
        "hidden_absent": not (OUT / "raw/anchor_role_response.float16.npy").exists(),
        "hash": core.sha(Path(__file__).with_name("phase1712_c178_natural_knowledge_ecology.py")) == protocol["producer_sha256"],
    }
    result = {"phase": 1712, "campaign": "C178", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
