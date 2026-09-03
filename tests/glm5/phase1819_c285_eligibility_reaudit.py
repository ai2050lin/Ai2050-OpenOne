#!/usr/bin/env python3
"""Independent C285 re-audit: distinguish no eligible edge from causal failure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C285"]
PRODUCER = Path(__file__).with_name("phase1819_c285_prospective_hyperedge_causal.py")


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "raw/sample_results.jsonl")
    original = core.load(OUT / "analysis/summary.json")
    eligible_total = sum(row["eligible_coordinates"] for row in rows)
    checks = {
        "samples_complete": len(rows) == 12,
        "six_families": len({row["family"] for row in rows}) == 6,
        "conditions_complete": all(set(row["conditions"]) == {"natural", "delete", "correct_rescue", "coordinate_roll_rescue", "wrong_role_rescue"} for row in rows),
        "finite": bool(np.isfinite([row[key] for row in rows for key in ("deletion_drop", "correct_minus_best_wrong")]).all()),
        "eligibility_failure_registered": eligible_total <= 1,
        "original_producer_hash": core.sha(PRODUCER) == protocol["producer_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": True})
    report = {
        **original,
        "status": "causal_no_test_local_coordinate_eligibility_failure",
        "eligible_coordinates_total": eligible_total,
        "families_passing": 0,
        "broad_gate_passed": False,
        "strict_interpretation": "The fixed q16-to-q17 boundary panel contained only one eligible coordinate across 12 samples. Deletion and rescue therefore did not test the C280 joint-word rule causally. This is an eligibility failure caused by aggregating prediction qualification over all transitions and destination roles, not evidence for or against local causal use.",
        "next_authorization": "C286_generation_and_side_effects; future_causal_contract_must_freeze_training_supported_transition_role_strata_before_holdout",
    }
    core.save(OUT / "analysis/summary.json", report)
    final_checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": True, "producer_hash": checks["original_producer_hash"]}
    final = {"phase": 1819, "campaign": "C285", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_reaudit.json", {"checks": checks, "all_checks_passed": True, "adjudication": report["status"]})
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

