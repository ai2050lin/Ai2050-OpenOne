#!/usr/bin/env python3
"""Independent audit for Phase1439 C072 campaign closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1439, "C072"
OUT = TESTS / "result/phase1439_c072_campaign_closure"
MECHANISM = TESTS / "result/phase1438_c072_permutation_spectrum"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    mechanism = core.load(MECHANISM / "analysis/permutation_spectrum_summary.json")
    cells = [cell for directions in mechanism["cell_results"].values() for cell in directions.values()]
    qualified = [set(cell["qualified_permutations"]) for cell in cells]
    intersection = set.intersection(*qualified)
    checks = {
        "status": final["status"] == "closed_after_heterogeneous_exhaustive_permutation_spectrum",
        "audit_reruns": final["checks"]["audits_reran"]
        and all(value["returncode"] == 0 for value in final["audit_rerun_outputs"].values()),
        "execution": mechanism["all_execution_checks_passed"] and all(cell["executor_pass"] for cell in cells),
        "classification": mechanism["overall_classification"] == final["retained"]["classification"] == "heterogeneous_or_executor_failed",
        "cell_counts": [len(values) for values in qualified] == [14, 13, 12, 3],
        "intersection": intersection == {"p00", "p01", "p06"},
        "not_subgroup": "p07" not in intersection and "not composition-closed" in final["rejected"]["subgroup_structured"],
        "not_multiset": "not all 24" in final["rejected"]["permutation_symmetric_multiset"],
        "not_role_order": "identity never" in final["rejected"]["role_order_selective"],
        "candidate_narrow": "descriptive" in final["retained"]["axis_candidate"]
        and any("independently confirmed" in value for value in final["claim_boundary"]["forbidden"]),
        "five_untested": len(final["untested"]) == 5,
        "next_campaign": final["next_question"]["campaign"] == "C073"
        and "matched axis-preserving" in final["next_question"]["requirements"][2],
        "observables": "full-dimensional input embeddings, Hidden State, and logits only" in final["next_question"]["requirements"],
        "forbidden": all(
            term in final["next_question"]["requirements"][6]
            for term in ("attention", "MLP", "parameters", "gradients", "dimensionality reduction", "probes")
        ),
        "authorization": final["authorization"] == "preregister_c073_independent_record_query_side_preservation_test",
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
