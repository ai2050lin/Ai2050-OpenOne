#!/usr/bin/env python3
"""Independent audit for Phase1444 C073 campaign closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1444_c073_campaign_closure"
MECHANISM = TESTS / "result/phase1443_c073_side_phase_competition"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    mechanism = core.load(MECHANISM / "analysis/side_phase_summary.json")
    cells = [cell for routes in mechanism["cell_results"].values() for directions in routes.values() for cell in directions.values()]
    checks = {
        "status": final["status"] == "closed_at_executor_gate_after_matched_side_phase_reveal",
        "audit_reruns": final["checks"]["audits_reran"] and all(value["returncode"] == 0 for value in final["audit_rerun_outputs"].values()),
        "execution": mechanism["all_execution_checks_passed"] and mechanism["record_count"] == 2688,
        "classification": mechanism["overall_classification"] == final["retained"]["classification"] == "executor_failed",
        "executor_half": sum(cell["executor_pass"] for cell in cells) == 8 and "eight of sixteen" in final["retained"]["partial_transport"],
        "wrong_control": all(cell["controls"]["wrong_identity_expected_sign_fraction"] == 1.0 for cell in cells),
        "no_reversed_claim": "no reversed-order cell" in final["rejected"]["semantic_side_confirmed"] and "no reversed-order cell" in final["rejected"]["physical_phase_confirmed"],
        "candidate_narrow": "evidence_same" in final["retained"]["same_surface_candidate"] and "subset" in final["rejected"]["c072_candidate_as_law"],
        "five_untested": len(final["untested"]) == 5,
        "claim_boundary": "preventing" in final["claim_boundary"]["allowed"] and any("failed reversed" in value for value in final["claim_boundary"]["forbidden"]),
        "partial_operator": "partial operator" in final["theory_update"]["statement"] and "E_identity" in final["theory_update"]["formula"],
        "next_campaign": final["next_question"]["campaign"] == "C074" and "identity-only" in final["next_question"]["requirements"][1],
        "observables": "full-dimensional input embeddings, Hidden State, and logits only" in final["next_question"]["requirements"],
        "forbidden": all(term in final["next_question"]["requirements"][6] for term in ("attention", "MLP", "parameters", "gradients", "dimensionality reduction", "probes")),
        "authorization": final["authorization"] == "preregister_c074_directional_transport_domain_test",
        "all_checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {"phase": 1444, "campaign": "C073", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
