#!/usr/bin/env python3
"""Independent audit for Phase1434 C071 campaign closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1434, "C071"
OUT = TESTS / "result/phase1434_c071_campaign_closure"
MECHANISM = TESTS / "result/phase1433_c071_cross_surface_mechanism"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    mechanism = core.load(MECHANISM / "analysis/mechanism_summary.json")
    cells = [
        value
        for directions in mechanism["cell_results"].values()
        for value in directions.values()
    ]
    checks = {
        "status": final["status"] == "closed_after_cross_surface_nonspecific_quartet_transport",
        "audit_reruns": final["checks"]["audits_reran"] and all(value["returncode"] == 0 for value in final["audit_rerun_outputs"].values()),
        "nonspecific": mechanism["overall_classification"] == "cross_surface_nonspecific" == final["retained"]["classification"],
        "four_cells": len(cells) == 4 and all(value["classification"] == "cross_surface_nonspecific" for value in cells),
        "mapped_retained": all(value["cross_surface_mapped_pass"] for value in cells),
        "selectivity_rejected": all(not value["selective_pass"] for value in cells) and "did not beat" in final["rejected"]["role_isomorphic_selectivity"],
        "claim_narrow": "one frozen role derangement" in final["claim_boundary"]["allowed"],
        "no_isomorphism_claim": any("isomorphism" in value for value in final["claim_boundary"]["forbidden"]),
        "untested": len(final["untested"]) == 5 and any("unordered multiset" in value for value in final["untested"]),
        "next_campaign": final["next_question"]["campaign"] == "C072" and "24 role permutations" in final["next_question"]["constraints"][1],
        "observables": "full-dimensional input embeddings, Hidden State, and logits only" in final["next_question"]["constraints"],
        "forbidden": all(term in final["next_question"]["constraints"][4] for term in ("attention", "MLP", "parameters", "gradients", "PCA", "probes")),
        "authorization": final["authorization"] == "preregister_c072_exhaustive_quartet_permutation_response_spectrum",
        "all_checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
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
