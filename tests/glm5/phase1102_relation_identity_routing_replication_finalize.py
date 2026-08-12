#!/usr/bin/env python3
"""Finalize the Phase1101-1102 relation-routing behavior qualification chain."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import phase1102_relation_identity_routing_replication_protocol as protocol


def failed_pairs(model_row: dict) -> set[str]:
    return {
        pair for pair, row in model_row["pair_results"].items()
        if not row["passed"]
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    phase1101_revision2 = protocol.read_json(
        protocol.SOURCE_PHASE1101_AUTHORIZATION
    )
    phase1101_revision1 = protocol.read_json(
        protocol.base.REVISION1_AUTHORIZATION
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1102 protocol audit failed")

    model_results = {}
    pair_failure_counts = Counter()
    for model in protocol.MODELS:
        current = authorization["models"][model]
        prior = phase1101_revision2["models"][model]
        current_failed = failed_pairs(current)
        prior_failed = failed_pairs(prior)
        for pair in current_failed:
            pair_failure_counts[pair] += 1
        union = current_failed | prior_failed
        intersection = current_failed & prior_failed
        noncoverage_gates = {
            key: value for key, value in current["gates"].items()
            if key != "pair_coverage"
        }
        model_results[model] = {
            "candidate_finite_fraction": current["candidate_finite_fraction"],
            "candidate_accuracy": current["candidate_accuracy"],
            "semantic_conflict_accuracy": current["semantic_conflict_accuracy"],
            "ordinal_conflict_accuracy": current["ordinal_conflict_accuracy"],
            "congruent_accuracy": current["congruent_accuracy"],
            "generation_accuracy": current["generation_accuracy"],
            "passing_pairs_phase1101_revision2": prior["passing_pairs"],
            "passing_pairs_phase1102": current["passing_pairs"],
            "passing_pair_change": current["passing_pairs"] - prior["passing_pairs"],
            "failed_pairs_phase1101_revision2": sorted(prior_failed),
            "failed_pairs_phase1102": sorted(current_failed),
            "stable_failed_pairs": sorted(intersection),
            "failed_pair_jaccard": len(intersection) / max(len(union), 1),
            "all_noncoverage_behavior_gates_passed": all(noncoverage_gates.values()),
            "pair_coverage_passed": current["gates"]["pair_coverage"],
            "model_behavior_passed": current["model_behavior_passed"],
        }

    all_noncoverage = all(
        row["all_noncoverage_behavior_gates_passed"]
        for row in model_results.values()
    )
    pair_coverage_models = sum(
        row["pair_coverage_passed"] for row in model_results.values()
    )
    gates = {
        "P1_protocol_integrity": bool(audit["all_checks_passed"]),
        "P2_independent_large_sample_completed": all(
            (protocol.OUT_ROOT / "behavior" / model / "summary.json").exists()
            for model in protocol.MODELS
        ),
        "P3_global_behavior_quality": all_noncoverage,
        "P4_two_model_pair_coverage": pair_coverage_models >= 2,
        "P5_hidden_scan_authorized": bool(authorization["hidden_scan_authorized"]),
        "P6_no_unauthorized_hidden_access": not any(
            (protocol.OUT_ROOT / "atlas" / model).exists()
            for model in protocol.MODELS
        ),
    }
    result = {
        "schema_version": "phase1102_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "behavior_authorization_digest": authorization["authorization_digest"],
        "source_phase1101_revision1_authorization_digest": phase1101_revision1[
            "authorization_digest"
        ],
        "source_phase1101_revision2_authorization_digest": phase1101_revision2[
            "authorization_digest"
        ],
        "models": model_results,
        "pair_failure_counts_phase1102": dict(pair_failure_counts),
        "pairs_failed_in_two_or_more_models": sorted(
            pair for pair, count in pair_failure_counts.items() if count >= 2
        ),
        "pairs_failed_in_all_models": sorted(
            pair for pair, count in pair_failure_counts.items()
            if count == len(protocol.MODELS)
        ),
        "gates": gates,
        "automatic_next_required": False,
        "decision": "stop_before_hidden_scan_and_change_observational_object",
        "frozen_interpretation": (
            "All three models show high aggregate relation-address routing behavior, but no model clears the preregistered 13-of-15 worst-cell coverage gate in the independent larger name world. The failure is model- and token-world-sensitive rather than a common missing relation pair. Hidden-state scanning, lexical inheritance claims, component selection, and causal interface tests remain unauthorized."
        ),
        "scientific_update": (
            "Phase1100 did not contain a behavior-geometry contradiction because its single-relation max/min task did not require relation identity. Phase1101-1102 repair that logical gap and show that relation identity can guide behavior in most cells, but the artificial two-record conflict interface is not uniformly stable enough across these small FP16 models for a 15-pair physical atlas."
        ),
        "mathematics_status": (
            "No new mathematics is needed at this qualification stage. Counts, exact factorial balance, per-cell accuracy, finite-rate audits, independent name-world replication, and frozen thresholds are sufficient to reject hidden-state authorization."
        ),
        "next_research_direction": (
            "Do not revise this prompt again or select the passing relation subset post hoc. Preserve the behavior map and design a natural-context relation-routing task whose correct continuation intrinsically depends on relation identity without explicit contradictory ranking records; preregister it on new lexical worlds before any hidden-state scan."
        ),
    }
    result["final_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_pairs": {
            model: row["passing_pairs_phase1102"]
            for model, row in model_results.items()
        },
        "pairs_failed_in_two_or_more_models": result[
            "pairs_failed_in_two_or_more_models"
        ],
        "automatic_next_required": False,
        "final_digest": result["final_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
