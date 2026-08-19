#!/usr/bin/env python3
"""Independent audit for Phase1429."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1429_c070_campaign_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_after_bidirectional_quartet_dominance_under_controlled_roster_contract",
        "audits_reran": final["checks"]["audits_reran"] and all(value["returncode"] == 0 for value in final["audit_rerun_outputs"].values()),
        "quartet_retained": final["retained"]["classification"] == "quartet_dominant" and all(value == 1.0 for splits in final["retained"]["quartet_desired_sign_fractions"].values() for value in splits.values()),
        "complement_rejected": all(value == 0.0 for splits in final["rejected"]["complement_desired_sign_fractions"].values() for value in splits.values()),
        "conditional_claim": "conditionally sufficient" in final["claim_boundary"]["allowed"],
        "no_necessity_claim": any("necessary" in item for item in final["claim_boundary"]["forbidden"]),
        "untested_preserved": len(final["untested"]) == 5,
        "cross_surface_next": "cross-surface" in final["next_question"]["object"],
        "fixed_state16": any("fixed Qwen state16" in item for item in final["next_question"]["constraints"]),
        "no_forbidden_route": all(term in final["next_question"]["constraints"][-1] for term in ("attention", "MLP", "gradients", "PCA", "probes")),
        "authorization": final["authorization"] == "preregister_c071_cross_surface_role_isomorphic_quartet_transport",
        "checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {
        "phase": 1429,
        "campaign": "C070",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
