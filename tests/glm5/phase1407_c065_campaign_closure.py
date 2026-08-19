#!/usr/bin/env python3
"""Phase1407: close C065 and authorize its independent breadth replication."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1407_c065_campaign_closure"
P1403 = TESTS / "result/phase1403_c065_active_only_natural_state_contract"
P1404 = TESTS / "result/phase1404_c065_state_swap_camera"
P1405 = TESTS / "result/phase1405_c065_natural_discovery_field"
P1406 = TESTS / "result/phase1406_c065_holdout_factorial_swaps"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1407 exists")
    contract = core.load(P1403 / "protocol/preregistration.json")
    field = core.load(P1405 / "analysis/field_summary.json")
    swaps = core.load(P1406 / "analysis/factorial_swap_summary.json")
    audit_paths = [
        P1403 / "audit/independent_final_audit.json",
        P1404 / "audit/independent_final_audit.json",
        P1405 / "audit/independent_final_audit.json",
        P1406 / "audit/independent_final_audit.json",
    ]
    audits = [core.load(path) for path in audit_paths]
    scripts = [
        TESTS / f"phase{phase}_c065_{name}.py"
        for phase, name in (
            (1403, "active_only_natural_state_contract"),
            (1404, "state_swap_camera"),
            (1405, "natural_discovery_field"),
            (1406, "holdout_factorial_swaps"),
        )
    ]
    scripts += [Path(str(path).replace(".py", "_audit.py")) for path in scripts]
    compiled = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            compiled = False
    family = swaps["route_status"]["family_identity"]
    polarity = swaps["route_status"]["joint_polarity"]
    checks = {
        "all_audits": all(a["all_checks_passed"] for a in audits),
        "contract_hash": isinstance(contract["contract_sha256"], str) and len(contract["contract_sha256"]) == 64,
        "discovery_cases": field["case_count"] == 18,
        "holdout_cases": swaps["holdout_count"] == 36,
        "frozen_candidates": field["candidate_count"] == 18,
        "family_confirmed": family["confirmed"] and len(family["qualified_candidates"]) == 5,
        "polarity_confirmed": polarity["confirmed"] and len(polarity["qualified_candidates"]) == 6,
        "late_candidates_rejected": not any(":w2" in cid for cid in family["qualified_candidates"] + polarity["qualified_candidates"]),
        "scripts_compile": compiled,
    }
    result = {
        "phase": 1407,
        "campaign": "C065",
        "status": "closed_after_confirmed_selective_whole_state_routes",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "confirmed_routes": swaps["route_status"],
        "claim_boundary": contract["claim_boundary"],
        "next_prediction": {
            "state_index": 16,
            "family_role": "record_family",
            "polarity_role": "query_family",
            "surfaces": ["ordinary", "catalog", "statement"],
            "no_new_candidate_search": True,
        },
        "authorization": "preregister_c066_midstate_breadth_confirmation",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
