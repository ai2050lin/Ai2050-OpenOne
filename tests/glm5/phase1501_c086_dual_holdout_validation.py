#!/usr/bin/env python3
"""Phase1501: reveal C086 confirmation and lockbox against frozen predictions."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
ATLAS = RESULT / "phase1499_c086_four_factor_atlas"
DISCOVERY = RESULT / "phase1500_c086_discovery_observation_freeze"
C085_CONTRACT = RESULT / "phase1489_c085_prospective_layered_contract"
C085_ATLAS = RESULT / "phase1492_c085_stratified_factorial_atlas"
OUT = RESULT / "phase1501_c086_dual_holdout_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1500_c086_discovery_observation_freeze import metrics_for_split


def within(observed, reference, tolerance):
    return abs(float(observed) - float(reference)) <= float(tolerance)


def adjudicate(observed, frozen):
    reference = frozen["reference"]
    tolerance = frozen["tolerances"]
    checks = {
        "P086-1": observed["field_class"] == reference["field_class"],
        "P086-2": within(
            observed["rho_content_median"],
            reference["rho_content_median"],
            tolerance["rho_content_median"],
        ),
        "P086-3": within(
            observed["conditional_cosine_mean"],
            reference["conditional_cosine_mean"],
            tolerance["conditional_cosine_mean"],
        ),
        "P086-4": within(
            observed["beta_relation_pairwise_mean"],
            reference["beta_relation_pairwise_mean"],
            tolerance["beta_relation_pairwise_mean"],
        )
        and within(
            observed["beta_relation_code_pairwise_mean"],
            reference["beta_relation_code_pairwise_mean"],
            tolerance["beta_relation_code_pairwise_mean"],
        ),
        "P086-5": within(
            observed["c085_standard_alignment_mean"],
            reference["c085_standard_alignment_mean"],
            tolerance["c085_standard_alignment_mean"],
        ),
        "P086-6": within(
            observed["beta_relation_top1pct_energy"],
            reference["beta_relation_top1pct_energy"],
            tolerance["beta_relation_top1pct_energy"],
        )
        and within(
            observed["beta_relation_code_top1pct_energy"],
            reference["beta_relation_code_top1pct_energy"],
            tolerance["beta_relation_code_top1pct_energy"],
        ),
    }
    return checks


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1501 exists")
    discovery_final = core.load(DISCOVERY / "analysis/final.json")
    discovery_audit = core.load(DISCOVERY / "audit/independent_final_audit.json")
    frozen = core.load(DISCOVERY / "protocol/frozen_holdout_predictions.json")
    atlas_summary = core.load(ATLAS / "analysis/four_factor_atlas_summary.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    c085_protocol = core.load(C085_CONTRACT / "protocol/preregistration.json")
    if (
        discovery_final["authorization"] != "run_phase1501_c086_dual_holdout_validation"
        or not discovery_audit["all_checks_passed"]
        or frozen["freeze_sha256"] != discovery_final["freeze_sha256"]
    ):
        raise RuntimeError("Phase1500 freeze missing")
    full = np.load(ATLAS / "atlas/all_four_factor_contrast_mean.float32.npy", mmap_mode="r")
    c085 = np.load(C085_ATLAS / "atlas/success_factorial_contrast_mean.float32.npy", mmap_mode="r")
    effect_index = {name: i for i, name in enumerate(atlas_summary["effects"])}
    results = {}
    for split_index, split in ((1, "confirmation"), (2, "lockbox")):
        rows, observed = metrics_for_split(
            full,
            c085,
            effect_index,
            split_index,
            protocol["roles"],
            c085_protocol["roles"],
        )
        checks = adjudicate(observed, frozen)
        core.write_rows(OUT / f"analysis/{split}_layer_role_observations.jsonl", rows)
        results[split] = {
            "observed": observed,
            "prediction_checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
            "all_predictions_passed": all(checks.values()),
        }
    checks = {
        "freeze_hash": frozen["freeze_sha256"] == discovery_final["freeze_sha256"],
        "confirmation_complete": results["confirmation"]["total"] == 6,
        "lockbox_complete": results["lockbox"]["total"] == 6,
        "finite": all(
            np.isfinite(value)
            for split in results.values()
            for value in split["observed"].values()
            if isinstance(value, (int, float))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    status = (
        "dual_holdout_confirmed"
        if results["confirmation"]["all_predictions_passed"]
        and results["lockbox"]["all_predictions_passed"]
        else "dual_holdout_boundary_failure"
    )
    result = {
        "phase": 1501,
        "campaign": "C086",
        "status": status,
        "frozen_reference": frozen["reference"],
        "validation": results,
        "checks": checks,
        "claim_boundary": "a pass confirms repeated coefficients in the controlled mixed-behavior field only",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1502_c086_stratum_and_c085_diagnostics",
    }
    core.save(OUT / "analysis/dual_holdout_validation.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
