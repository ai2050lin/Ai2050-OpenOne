#!/usr/bin/env python3
"""Independent audit for C186 prospective relation-target response prediction."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1720_c186_new_material_response_ecology_prediction"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    protocol = core.load(OUT / "protocol/preregistration.json")
    lock = core.load(OUT / "protocol/behavior_eligibility_lock.json")
    final = core.load(OUT / "analysis/final.json")
    producer = Path(__file__).with_name("phase1720_c186_new_material_response_ecology_prediction.py")
    hidden_expected = bool(lock["eligible_families"])
    checks = {"closed": final["status"] == "closed" and final["all_checks_passed"], "168_cases": protocol["cases"] == 168, "behavior_before_hidden": (OUT / "protocol/behavior_eligibility_lock.json").stat().st_mtime <= ((OUT / "raw/new_relation_role_response.float16.npy").stat().st_mtime if hidden_expected else float("inf")), "typed_hidden": hidden_expected == (OUT / "raw/new_relation_role_response.float16.npy").exists(), "supported_object_only": protocol["hidden_policy"].startswith("relation q24 source only"), "hash": core.sha(producer) == protocol["producer_sha256"]}
    result = {"phase": 1720, "campaign": "C186", "checks": checks, "all_checks_passed": all(checks.values()), "authorization": final["next_authorization"]}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
