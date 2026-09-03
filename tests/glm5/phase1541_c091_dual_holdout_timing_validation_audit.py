#!/usr/bin/env python3
"""Independent audit for Phase1541."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1541_c091_dual_holdout_timing_validation"
PARENT = TESTS / "result/phase1540_c091_discovery_timing_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/dual_holdout_summary.json")
    rows = core.rows(OUT / "analysis/dual_holdout_candidate_validation.jsonl")
    vectors = np.load(OUT / "raw/dual_holdout_candidate_centroids.float32.npy")
    frozen = core.load(PARENT / "protocol/frozen_discovery_candidates.json")
    gate = frozen["frozen_holdout_gate"]
    independently_scored = []
    for row in rows:
        passed = (
            row["centroid_cosine_to_discovery"] >= gate["centroid_cosine_to_discovery_min"]
            and row["concreteness_cosine"] >= gate["concreteness_cosine_min"]
            and row["median_individual_alignment"] >= gate["median_individual_alignment_min"]
            and (row["object"] != "factorial_interaction" or row["control_cosine"] >= gate["factorial_control_cosine_min"])
        )
        independently_scored.append(passed == row["gate_pass"])
    expected_keys = {
        (partition, object_name, surface)
        for partition in ("confirmation", "lockbox")
        for surface in ("prequery", "postquery")
        for object_name in ("factorial_interaction", "behavior_grounded_truth")
    }
    checks = {
        "parent_gate_identical": report["frozen_gate"] == gate,
        "partition_alias_only": report["partition_alias_resolution"] == {
            "frozen_name": "response_confirmation",
            "contract_name": "confirmation",
            "resolved_before_first_vector_or_statistic": True,
            "materials_candidates_formulas_and_thresholds_changed": False,
        },
        "validation_hash": core.sha(OUT / "analysis/dual_holdout_candidate_validation.jsonl") == report["files"]["validation"]["sha256"],
        "vector_hash": core.sha(OUT / "raw/dual_holdout_candidate_centroids.float32.npy") == report["files"]["centroids"]["sha256"],
        "coverage": {(row["partition"], row["object"], row["surface"]) for row in rows} == expected_keys,
        "vector_shape": list(vectors.shape) == [8, 2560],
        "independent_gate_recalculation": all(independently_scored),
        "aggregate_gate": report["checks"]["all_holdout_gates_passed"] == all(row["gate_pass"] for row in rows),
        "candidate_identity": report["checks"]["no_candidate_reselection"],
        "claim_scope": "one Qwen3 contract" in report["claim_boundary"]["supported_if_passed"] and "causal transport" in report["claim_boundary"]["not_supported"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] in ("run_phase1542_c091_final_adjudication", "run_phase1542_c091_route_closure"),
    }
    result = {"phase": 1541, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
