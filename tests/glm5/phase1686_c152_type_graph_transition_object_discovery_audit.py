#!/usr/bin/env python3
"""Independent audit for Phase1686/C152."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1686_c152_type_graph_transition_object_discovery"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

r = core.load(OUT / "analysis/discovery.json")
checks = {
    "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
    "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
    "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
    "six_candidates": len(r["ranking"]) == 6,
    "two_panels": set(r["panel_reports"]) == {"c141_confirmation", "c151_fresh"},
    "winner_recomputed": r["selected_candidate"] == sorted(
        r["ranking"],
        key=lambda c: (
            min(r["panel_reports"][p][c]["median_cosine"] for p in r["panel_reports"]),
            -max(r["panel_reports"][p][c]["median_relative_error"] for p in r["panel_reports"]),
        ),
        reverse=True,
    )[0],
    "retrospective": r["claim_boundary"].startswith("retrospective"),
}
audit = {
    "phase": 1686,
    "campaign": "C152",
    "checks": checks,
    "passed": sum(checks.values()),
    "total": len(checks),
    "all_checks_passed": all(checks.values()),
    "scientific_candidate_stable": r["stable_candidate"],
    "authorization": "memo_and_C153_if_stable_else_campaign_close",
}
core.save(OUT / "audit/independent_closure_audit.json", audit)
print(json.dumps(audit, indent=2))
