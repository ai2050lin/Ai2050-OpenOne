#!/usr/bin/env python3
"""Freeze pair-specific Phase1103 behavior authorization."""

from __future__ import annotations

import json

import phase1103_natural_relation_route_protocol as protocol


def cell(
    summary: dict,
    pair: str,
    surface: str,
    split: str,
    route: str,
    congruence: str,
) -> dict:
    return summary["per_cell"]["|".join((
        pair, surface, split, route, congruence,
    ))]


def generation_cell(
    summary: dict,
    pair: str,
    surface: str,
    split: str,
    route: str,
) -> dict:
    return summary["per_generation_cell"]["|".join((
        pair, surface, split, route,
    ))]


def weighted_accuracy(records: list[dict]) -> float:
    count = sum(int(row["candidate_count"]) for row in records)
    return sum(
        float(row["candidate_accuracy"]) * int(row["candidate_count"])
        for row in records
    ) / max(count, 1)


def evaluate_pair_split(summary: dict, pair: str, split: str) -> dict:
    thresholds = protocol.THRESHOLDS
    route_records = {route: [] for route in protocol.ROUTE_TYPES}
    congruent_records = []
    generation_records = []
    cell_results = {}
    for surface in protocol.SURFACES:
        for route in protocol.ROUTE_TYPES:
            conflict = cell(
                summary, pair, surface, split, route, "conflict"
            )
            route_records[route].append(conflict)
            congruent = cell(
                summary, pair, surface, split, route, "congruent"
            )
            congruent_records.append(congruent)
            passed = (
                conflict["candidate_finite_fraction"]
                >= thresholds["minimum_candidate_finite_fraction"]
                and conflict["candidate_accuracy"]
                >= thresholds["minimum_pair_cell_accuracy"]
            )
            cell_results["|".join((surface, route, "conflict"))] = {
                **conflict,
                "passed": passed,
            }
        for route in ("exact", "paraphrase"):
            generation = generation_cell(
                summary, pair, surface, split, route
            )
            generation_records.append(generation)
            cell_results["|".join((surface, route, "generation"))] = {
                **generation,
                "passed": (
                    generation["target_before_distractor_accuracy"]
                    >= thresholds["minimum_generation_accuracy"]
                ),
            }
    route_accuracy = {
        route: weighted_accuracy(records)
        for route, records in route_records.items()
    }
    congruent_accuracy = weighted_accuracy(congruent_records)
    generation_accuracy = sum(
        float(row["target_before_distractor_accuracy"])
        * int(row["generation_count"])
        for row in generation_records
    ) / max(sum(int(row["generation_count"]) for row in generation_records), 1)
    gates = {
        "all_conflict_cells": all(
            row["passed"] for key, row in cell_results.items()
            if key.endswith("|conflict")
        ),
        "exact_route": (
            route_accuracy["exact"]
            >= thresholds["minimum_route_accuracy"]
        ),
        "paraphrase_route": (
            route_accuracy["paraphrase"]
            >= thresholds["minimum_route_accuracy"]
        ),
        "ordinal_route": (
            route_accuracy["ordinal"]
            >= thresholds["minimum_route_accuracy"]
        ),
        "congruent": (
            congruent_accuracy
            >= thresholds["minimum_congruent_accuracy"]
        ),
        "all_generation_cells": all(
            row["passed"] for key, row in cell_results.items()
            if key.endswith("|generation")
        ),
        "generation": (
            generation_accuracy
            >= thresholds["minimum_generation_accuracy"]
        ),
    }
    return {
        "split": split,
        "route_accuracy": route_accuracy,
        "congruent_accuracy": congruent_accuracy,
        "generation_accuracy": generation_accuracy,
        "minimum_conflict_cell_accuracy": min(
            row["candidate_accuracy"]
            for records in route_records.values() for row in records
        ),
        "minimum_conflict_cell_finite_fraction": min(
            row["candidate_finite_fraction"]
            for records in route_records.values() for row in records
        ),
        "cell_results": cell_results,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_model(model: str) -> dict:
    summary = protocol.read_json(
        protocol.OUT_ROOT / "behavior" / model / "summary.json"
    )
    pair_results = {}
    for pair in protocol.RELATION_PAIRS:
        splits = {
            split: evaluate_pair_split(summary, pair, split)
            for split in protocol.SPLITS
        }
        pair_results[pair] = {
            "splits": splits,
            "passed": all(row["passed"] for row in splits.values()),
        }
    passing_pairs = [
        pair for pair, row in pair_results.items() if row["passed"]
    ]
    precision_gate = (
        summary["precision"]["has_fp16_parameters"]
        and not summary["precision"]["has_bf16_parameters"]
        and not summary["precision"]["has_quantized_modules"]
    )
    return {
        "model": model,
        "summary_digest": summary["summary_digest"],
        "candidate_finite_fraction": summary[
            "candidate_finite_fraction"
        ],
        "candidate_accuracy": summary["candidate_accuracy"],
        "generation_accuracy": summary[
            "generation_target_before_distractor_accuracy"
        ],
        "precision_gate": precision_gate,
        "passing_pairs": passing_pairs,
        "passing_pair_count": len(passing_pairs),
        "pair_results": pair_results,
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1103 protocol audit failed")
    models = {
        model: evaluate_model(model) for model in protocol.MODELS
    }
    pair_models = {
        pair: [
            model for model in protocol.MODELS
            if models[model]["precision_gate"]
            and models[model]["pair_results"][pair]["passed"]
        ]
        for pair in protocol.RELATION_PAIRS
    }
    shared_pairs = [
        pair for pair, passing_models in pair_models.items()
        if len(passing_models)
        >= protocol.THRESHOLDS["minimum_models_per_shared_pair"]
    ]
    wrong_pair_controls: dict[str, dict[str, str | None]] = {}
    for model in protocol.MODELS:
        passing = list(models[model]["passing_pairs"])
        wrong_pair_controls[model] = {}
        for pair in shared_pairs:
            candidates = [value for value in passing if value != pair]
            wrong_pair_controls[model][pair] = (
                candidates[0] if candidates else None
            )
    causally_eligible_pairs = []
    causal_models_by_pair = {}
    for pair in shared_pairs:
        eligible_models = [
            model for model in pair_models[pair]
            if wrong_pair_controls[model][pair] is not None
        ]
        causal_models_by_pair[pair] = eligible_models
        if (
            len(eligible_models)
            >= protocol.THRESHOLDS["minimum_models_per_shared_pair"]
        ):
            causally_eligible_pairs.append(pair)
    causal_authorized = bool(causally_eligible_pairs)
    result = {
        "schema_version": "phase1103_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "pair_models": pair_models,
        "shared_behavior_authorized_pairs": shared_pairs,
        "shared_behavior_pair_count": len(shared_pairs),
        "wrong_pair_controls": wrong_pair_controls,
        "causal_models_by_pair": causal_models_by_pair,
        "causally_eligible_pairs": causally_eligible_pairs,
        "causal_scan_authorized": causal_authorized,
        "claim_scope": (
            "Only listed model-pair cells are behavior-authorized. No result "
            "is a 15-pair family-wide authorization."
        ),
        "selection_independence": (
            "Pair eligibility uses only preregistered behavior metrics in both "
            "splits and is frozen before any Phase1103 hidden-state result."
        ),
        "decision": (
            "run_pair_specific_signed_residual_transport"
            if causal_authorized else "stop_at_behavior_gate"
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json",
        result,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_pair_counts": {
            model: models[model]["passing_pair_count"]
            for model in protocol.MODELS
        },
        "shared_pairs": shared_pairs,
        "causally_eligible_pairs": causally_eligible_pairs,
        "causal_scan_authorized": causal_authorized,
        "authorization_digest": result["authorization_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
