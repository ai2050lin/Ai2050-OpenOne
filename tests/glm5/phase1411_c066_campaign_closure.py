#!/usr/bin/env python3
"""Phase1411: close C066 and freeze the next relational-composition question."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1411_c066_campaign_closure"
P1408 = TESTS / "result/phase1408_c066_midstate_breadth_contract"
P1409 = TESTS / "result/phase1409_c066_behavior"
P1410 = TESTS / "result/phase1410_c066_state16_factorial_replication"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1411 exists")
    protocol = core.load(P1408 / "protocol/preregistration.json")
    behavior = core.load(P1409 / "analysis/behavior_summary.json")
    replication = core.load(P1410 / "analysis/state16_replication_summary.json")
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1408, P1409, P1410)]
    scripts = []
    for phase, stem in (
        (1408, "c066_midstate_breadth_contract"),
        (1409, "c066_behavior"),
        (1410, "c066_state16_factorial_replication"),
    ):
        scripts.extend([TESTS / f"phase{phase}_{stem}.py", TESTS / f"phase{phase}_{stem}_audit.py"])
    compiled = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            compiled = False
    expected_family = ["catalog:family_identity:s16"]
    expected_polarity = ["ordinary:joint_polarity:s16", "catalog:joint_polarity:s16", "statement:joint_polarity:s16"]
    checks = {
        "audits": all(a["all_checks_passed"] for a in audits),
        "behavior_five_families": len(behavior["qualified_families"]) == 5,
        "holdout_120": replication["holdout_set_count"] == 120,
        "state16_only": protocol["mechanism"]["state_index"] == 16,
        "family_exact": replication["route_status"]["family_identity"]["qualified_candidates"] == expected_family,
        "query_exact": replication["route_status"]["joint_polarity"]["qualified_candidates"] == expected_polarity,
        "no_discovery_search": "new candidate search" in protocol["forbidden"],
        "scripts_compile": compiled,
    }
    result = {
        "phase": 1411,
        "campaign": "C066",
        "status": "closed_after_partial_state16_breadth_replication",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "confirmed": {
            "record_family_state16": expected_family,
            "query_family_state16": expected_polarity,
        },
        "rejected": {
            "record_family_state16": ["ordinary:family_identity:s16", "statement:family_identity:s16"],
            "other_states": "not tested in C066 and not authorized as fallback",
        },
        "claim_boundary": protocol["claim_boundary"],
        "next_question": {
            "campaign": "C067",
            "object": "catalog state-16 record/query paired relational composition",
            "prediction": "single-side mismatch damages yes margin; matched dual-side replacement restores it",
            "no_layer_search": True,
        },
        "authorization": "preregister_c067_paired_state_relational_composition",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
