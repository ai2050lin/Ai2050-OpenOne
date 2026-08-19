#!/usr/bin/env python3
"""Independent audit for Phase1424."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1424_c069_campaign_closure"


def main() -> None:
    final = core.load(OUT / "analysis/final.json")
    checks = {
        "status": final["status"] == "closed_after_graded_quartet_confirmation_and_failed_discrete_sufficiency",
        "graded_retained": all(
            final["retained"]["interaction_medians"][split][direction] > 0.0
            for split in ("confirmation", "lockbox")
            for direction in ("true_recipient", "false_recipient")
        ),
        "strong_rejected": "sufficient discrete relation state" in final["rejected"]["strong_hypothesis"],
        "zero_discrete_families": final["rejected"]["discrete_qualified_families"] == [],
        "unseen_not_negative": len(final["untested"]) == 4,
        "fixed_state16": "fixed state16" in final["next_question"]["constraints"],
        "complement_object": "quartet-versus-complement" in final["next_question"]["object"],
        "no_forbidden_route": all(term in final["next_question"]["constraints"][-1] for term in ("attention", "MLP", "PCA", "probes")),
        "authorization": final["authorization"] == "preregister_c070_quartet_complement_support_partition",
        "checks": final["all_checks_passed"] and all(final["checks"].values()),
    }
    result = {"phase": 1424, "campaign": "C069", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    core.save(OUT / "audit/independent_final_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
