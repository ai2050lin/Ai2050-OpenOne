#!/usr/bin/env python3
"""Independent audit for Phase1540."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1540_c091_discovery_timing_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    report = core.load(OUT / "analysis/discovery_timing_atlas_summary.json")
    interaction = core.rows(OUT / "analysis/discovery_factorial_interaction_atlas.jsonl")
    truth = core.rows(OUT / "analysis/discovery_behavior_grounded_truth_atlas.jsonl")
    vectors = np.load(OUT / "raw/discovery_candidate_centroids.float32.npy")
    checks = {
        "interaction_hash": core.sha(OUT / "analysis/discovery_factorial_interaction_atlas.jsonl") == report["files"]["interaction_atlas"]["sha256"],
        "truth_hash": core.sha(OUT / "analysis/discovery_behavior_grounded_truth_atlas.jsonl") == report["files"]["truth_atlas"]["sha256"],
        "vector_hash": core.sha(OUT / "raw/discovery_candidate_centroids.float32.npy") == report["files"]["candidate_centroids"]["sha256"],
        "coverage": len(interaction) == 296 and len(truth) == 296 and list(vectors.shape) == [4, 2560],
        "causal_nulls": all(value == 0.0 for value in report["causal_nulls"].values()),
        "interaction_candidates": all(value["discovery_candidate_pass"] for value in report["factorial_interaction_candidates"].values()),
        "truth_candidates": all(value["discovery_candidate_pass"] for value in report["behavior_grounded_truth_candidates"].values()),
        "holdout_gate_frozen": report["frozen_holdout_gate"] == {
            "partitions": ["response_confirmation", "lockbox"],
            "same_formula_state_and_role": True,
            "centroid_cosine_to_discovery_min": 0.5,
            "concreteness_cosine_min": 0.3,
            "median_individual_alignment_min": 0.0,
            "factorial_control_cosine_min": 0.3,
            "all_candidates_all_partitions_required": True,
            "failure_action": "close_C091_without_hidden_mechanism_claim",
        },
        "candidate_vector_order_frozen": report["candidate_vector_order"] == [
            "prequery_factorial_interaction",
            "prequery_behavior_grounded_truth",
            "postquery_factorial_interaction",
            "postquery_behavior_grounded_truth",
        ],
        "scope": "lack behavior qualification" in report["claim_boundary"]["factorial"] and "not exactly canceled" in report["claim_boundary"]["truth"],
        "authorization": core.load(OUT / "analysis/final.json")["authorization"] in ("run_phase1541_c091_dual_holdout_timing_validation", "run_phase1542_c091_route_closure"),
    }
    result = {"phase": 1540, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "checks": checks}
    core.save(OUT / "audit/independent_final_audit.json", result)
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
