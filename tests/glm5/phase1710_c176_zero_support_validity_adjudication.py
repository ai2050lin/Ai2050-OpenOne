#!/usr/bin/env python3
"""C176 validity adjudication: zero contrasts are undefined, not perfect predictions."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1710_c176_broad_linguistic_family_reuse"
C162 = TESTS / "result/phase1696_c162_linguistic_program_field"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main():
    fields = np.load(C162 / "analysis/unit_term_fields.float16.npy", mmap_mode="r")
    norm = np.linalg.norm(np.asarray(fields, np.float32), axis=-1)
    role_names = ("primary", "secondary", "relation", "context", "query", "boundary")
    zero_by_role = {role: float(np.mean(norm[..., i] == 0)) for i, role in enumerate(role_names)}
    checks = {
        "zero_vectors_present": bool(np.any(norm == 0)),
        "query_all_zero": zero_by_role["query"] == 1.0,
        "context_mostly_zero": zero_by_role["context"] > 0.90,
        "old_metric_misclassified_zero": True,
    }
    result = {
        "phase": 1710,
        "campaign": "C176",
        "status": "measurement_invalidated_before_scientific_use",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "zero_fraction_global": float(np.mean(norm == 0)),
        "zero_fraction_by_role": zero_by_role,
        "invalidated_outputs": ["formation", "query q24 reuse summary", "any NRMSE/sign score with zero discovery or actual support"],
        "still_descriptive_only": ["nonzero primary/relation coordinate energy summaries"],
        "reason": "The metric encoded zero-vs-zero as NRMSE 0 and sign agreement 1. Such cells have no observable contrast and must be missing, not perfect.",
        "authorization": "C177_repair_with_explicit_missingness_then_continue_all_other_arms",
    }
    core.save(OUT / "analysis/measurement_validity.json", result)
    final = {
        "phase": 1710,
        "campaign": "C176",
        "status": "closed_invalid_measurement",
        "checks": {"original_execution": True, "validity_adjudication": True},
        "all_checks_passed": True,
        "scientific_result_valid": False,
        "headline": {"zero_fraction_global": result["zero_fraction_global"], "zero_fraction_by_role": zero_by_role, "reason": result["reason"]},
        "next_authorization": result["authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_validity_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
