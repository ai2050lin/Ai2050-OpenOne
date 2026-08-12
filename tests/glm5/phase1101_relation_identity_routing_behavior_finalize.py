#!/usr/bin/env python3
"""Freeze Phase1101 behavior authorization before hidden-state access."""

from __future__ import annotations

import json

import phase1101_relation_identity_routing_protocol as protocol


def cell(summary: dict, pair: str, surface: str, split: str, route: str, congruence: str) -> dict:
    return summary["per_cell"]["|".join((pair, surface, split, route, congruence))]


def weighted_accuracy(records: list[dict]) -> float:
    count = sum(int(row["candidate_count"]) for row in records)
    return (
        sum(float(row["candidate_accuracy"]) * int(row["candidate_count"]) for row in records)
        / max(count, 1)
    )


def evaluate_model(model: str) -> dict:
    summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")
    thresholds = protocol.THRESHOLDS
    semantic_conflict = []
    ordinal_conflict = []
    congruent = []
    pair_results = {}
    for pair in protocol.RELATION_PAIRS:
        pair_cells = []
        for surface in protocol.SURFACES:
            for split in protocol.SPLITS:
                semantic = cell(summary, pair, surface, split, "semantic", "conflict")
                ordinal = cell(summary, pair, surface, split, "ordinal", "conflict")
                semantic_conflict.append(semantic)
                ordinal_conflict.append(ordinal)
                pair_cells.extend((semantic, ordinal))
                congruent.extend((
                    cell(summary, pair, surface, split, "semantic", "congruent"),
                    cell(summary, pair, surface, split, "ordinal", "congruent"),
                ))
        pair_passed = all(
            row["candidate_finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"]
            and row["candidate_accuracy"] >= thresholds["minimum_pair_cell_accuracy"]
            for row in pair_cells
        )
        pair_results[pair] = {
            "minimum_conflict_cell_accuracy": min(row["candidate_accuracy"] for row in pair_cells),
            "minimum_conflict_cell_finite_fraction": min(row["candidate_finite_fraction"] for row in pair_cells),
            "passed": pair_passed,
        }
    semantic_accuracy = weighted_accuracy(semantic_conflict)
    ordinal_accuracy = weighted_accuracy(ordinal_conflict)
    congruent_accuracy = weighted_accuracy(congruent)
    passing_pairs = sum(row["passed"] for row in pair_results.values())
    gates = {
        "candidate_finite": summary["candidate_finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"],
        "candidate_accuracy": summary["candidate_accuracy"] >= thresholds["minimum_candidate_accuracy"],
        "semantic_conflict": semantic_accuracy >= thresholds["minimum_conflict_accuracy"],
        "ordinal_conflict": ordinal_accuracy >= thresholds["minimum_conflict_accuracy"],
        "congruent": congruent_accuracy >= thresholds["minimum_congruent_accuracy"],
        "pair_coverage": passing_pairs >= thresholds["minimum_passing_pairs"],
        "generation": summary["generation_target_before_distractor_accuracy"] >= thresholds["minimum_generation_accuracy"],
        "precision": (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        ),
    }
    return {
        "model": model,
        "summary_digest": summary["summary_digest"],
        "candidate_finite_fraction": summary["candidate_finite_fraction"],
        "candidate_accuracy": summary["candidate_accuracy"],
        "semantic_conflict_accuracy": semantic_accuracy,
        "ordinal_conflict_accuracy": ordinal_accuracy,
        "congruent_accuracy": congruent_accuracy,
        "generation_accuracy": summary["generation_target_before_distractor_accuracy"],
        "passing_pairs": passing_pairs,
        "pair_results": pair_results,
        "gates": gates,
        "model_behavior_passed": all(gates.values()),
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1101 protocol audit failed")
    models = {model: evaluate_model(model) for model in protocol.MODELS}
    passing = [model for model, row in models.items() if row["model_behavior_passed"]]
    hidden_authorized = len(passing) >= protocol.THRESHOLDS["minimum_behavior_models"]
    result = {
        "schema_version": "phase1101_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "behavior_authorized_models": passing,
        "formal_models_passing": [model for model in protocol.FORMAL_MODELS if models[model]["model_behavior_passed"]],
        "hidden_scan_authorized": hidden_authorized,
        "decision": (
            "authorize_all_three_sequential_hidden_scans"
            if hidden_authorized else "stop_at_behavior_gate"
        ),
        "interpretation": (
            "At least two models can solve a relation-identity-necessary late-routing task; hidden-state measurement may proceed without claiming a semantic interface."
            if hidden_authorized
            else "The task interface did not establish a shared behavior base in two models; hidden-state differences would mix task failure with relation routing."
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_models": passing,
        "hidden_scan_authorized": hidden_authorized,
        "authorization_digest": result["authorization_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
