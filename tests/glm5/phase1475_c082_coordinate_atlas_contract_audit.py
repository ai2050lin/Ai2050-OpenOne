#!/usr/bin/env python3
"""Independent audit for Phase1475."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1475_c082_coordinate_atlas_contract"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    preaudit = core.load(OUT / "audit/pre_run_source_and_scope_audit.json")
    final = core.load(OUT / "analysis/final.json")
    py_compile.compile(str(TESTS / "phase1475_c082_coordinate_atlas_contract.py"), doraise=True)
    checks = {
        "preaudit": preaudit["all_checks_passed"] and preaudit["model_run"] is False,
        "hash": protocol["contract_sha256"] == core.digest({key: value for key, value in protocol.items() if key not in ("contract_sha256", "authorization")}),
        "authorization": final["authorization"] == protocol["authorization"] == "run_phase1476_c082_coordinate_atlas",
        "axes": len(protocol["axes"]["relations"]) == 6 and len(protocol["axes"]["splits"]) == 3 and len(protocol["axes"]["surfaces"]) == 2 and len(protocol["axes"]["states"]) == 37 and len(protocol["axes"]["roles"]) == 9 and protocol["axes"]["coordinates"] == 2560,
        "raw_hashes": protocol["source"]["discovery_raw_sha256"] == core.sha(RESULT / "phase1465_c079_discovery_full_field_capture/raw/discovery_role_field.float16.npy") and protocol["source"]["holdout_raw_sha256"] == core.sha(RESULT / "phase1467_c079_holdout_capture_and_validation/raw/holdout_role_field.float16.npy"),
        "retrospective": protocol["evidence_scope"].startswith("retrospective exploratory") and "calling any panel a holdout confirmation" in protocol["forbidden"],
        "full_coordinates": protocol["integrity"]["process_all_2560_coordinates"] and protocol["integrity"]["no_coordinate_filter_before_outputs"],
        "no_model": "new model run" in protocol["forbidden"],
    }
    result = {"phase": 1475, "campaign": "C082", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    if not result["all_checks_passed"]:
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
