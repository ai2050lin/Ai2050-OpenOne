#!/usr/bin/env python3
"""Independent audit for Phase1386 C060 closure."""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1386_c060_campaign_closure"
SCRIPT = TESTS / "phase1386_c060_campaign_closure.py"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(__file__, doraise=True)
    expected_audits = {
        "1380": (14, 14),
        "1381": (11, 11),
        "1382": (13, 13),
        "1383": (14, 14),
        "1384": (15, 15),
        "1385": (10, 10),
    }
    checks = {
        "status": final["status"] == "closed_after_all_frozen_eligible_routes",
        "closure_checks": final["all_checks_passed"] and final["passed"] == final["total"],
        "audit_counts": all(
            (final["phase_audits"][p]["passed"], final["phase_audits"][p]["total"]) == counts
            for p, counts in expected_audits.items()
        ),
        "early": final["formal_results"]["early_sufficiency_replicated"],
        "mid_reverse": final["formal_results"]["mid_reverse_replicated"],
        "no_threshold": not final["formal_results"]["strong_threshold_gate"],
        "no_cancellation": final["formal_results"]["cancellation_candidate_fraction"] == 0.0,
        "dynamic_candidate": final["formal_results"]["discovery_family_512_dynamic_qualified"],
        "mediation_boundary": final["formal_results"]["boundary_block_fraction_median"] >= 0.5,
        "mediation_query_failed": final["formal_results"]["query_block_fraction_median"] < 0.5 and not final["formal_results"]["full_serial_mediation_qualified"],
        "forbidden": final["forbidden_hits"] == [],
        "no_automatic_extension": not final["automatic_next_phase"],
        "scripts_compile": True,
    }
    audit = {
        "phase": 1386,
        "campaign": "C060",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
