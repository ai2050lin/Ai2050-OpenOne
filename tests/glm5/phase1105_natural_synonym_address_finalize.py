#!/usr/bin/env python3
"""Freeze Phase1105 relation-specific semantic behavior authorization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1105_natural_synonym_address_protocol as protocol


def cell(summary: dict[str, Any], pair: str, split: str, template: int, route: str, congruence: str) -> dict[str, Any]:
    return summary["per_cell"][f"{pair}|{split}|{template}|{route}|{congruence}"]


def aggregate(summary: dict[str, Any], pair: str, split: str, route: str, congruence: str) -> dict[str, float]:
    rows = [cell(summary, pair, split, template, route, congruence) for template in protocol.TEMPLATES_BY_SPLIT[split]]
    count = sum(int(row["candidate_count"]) for row in rows)
    finite = sum(int(row["candidate_count"]) * float(row["candidate_finite_fraction"]) for row in rows)
    hits = sum(int(row["candidate_count"]) * float(row["candidate_finite_fraction"]) * float(row["candidate_accuracy"]) for row in rows)
    return {
        "candidate_count": count,
        "candidate_finite_fraction": finite / max(count, 1),
        "candidate_accuracy": hits / max(finite, 1),
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    thresholds = protocol.THRESHOLDS
    summaries = {
        model: protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")
        for model in protocol.MODELS
    }
    models = {}
    pair_models: dict[str, list[str]] = {pair: [] for pair in protocol.RELATION_PAIRS}
    for model, summary in summaries.items():
        pair_results = {}
        passing_pairs = []
        for pair in protocol.RELATION_PAIRS:
            split_results = {}
            for split in protocol.TEMPLATES_BY_SPLIT:
                route_metrics = {
                    route: aggregate(summary, pair, split, route, "conflict")
                    for route in protocol.ROUTE_TYPES
                }
                congruent_metrics = {
                    route: aggregate(summary, pair, split, route, "congruent")
                    for route in protocol.ROUTE_TYPES
                }
                template_gates = {}
                for route in protocol.ROUTE_TYPES:
                    minimum = (
                        thresholds["minimum_semantic_template_accuracy"]
                        if route in {"close_synonym", "natural_definition"}
                        else thresholds["minimum_exact_template_accuracy"]
                        if route == "exact"
                        else thresholds["minimum_ordinal_template_accuracy"]
                    )
                    template_gates[route] = all(
                        cell(summary, pair, split, template, route, "conflict")["candidate_accuracy"] >= minimum
                        and cell(summary, pair, split, template, route, "conflict")["candidate_finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"]
                        for template in protocol.TEMPLATES_BY_SPLIT[split]
                    )
                aggregate_gates = {
                    "finite": all(row["candidate_finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"] for row in route_metrics.values()),
                    "exact": route_metrics["exact"]["candidate_accuracy"] >= thresholds["minimum_exact_aggregate_accuracy"],
                    "close_synonym": route_metrics["close_synonym"]["candidate_accuracy"] >= thresholds["minimum_semantic_aggregate_accuracy"],
                    "natural_definition": route_metrics["natural_definition"]["candidate_accuracy"] >= thresholds["minimum_semantic_aggregate_accuracy"],
                    "ordinal": route_metrics["ordinal"]["candidate_accuracy"] >= thresholds["minimum_ordinal_aggregate_accuracy"],
                    "congruent": all(row["candidate_accuracy"] >= thresholds["minimum_congruent_accuracy"] for row in congruent_metrics.values()),
                }
                gates = {**{f"template_{key}": value for key, value in template_gates.items()}, **aggregate_gates}
                split_results[split] = {
                    "route_metrics": route_metrics,
                    "congruent_metrics": congruent_metrics,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            passed = all(row["passed"] for row in split_results.values())
            pair_results[pair] = {"splits": split_results, "passed_both_splits": passed}
            if passed:
                passing_pairs.append(pair)
                pair_models[pair].append(model)
        models[model] = {
            "summary_digest": summary["summary_digest"],
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "candidate_accuracy": summary["candidate_accuracy"],
            "pair_results": pair_results,
            "passing_pairs": passing_pairs,
        }
    cross_model_pairs = [
        pair for pair, passing_models in pair_models.items()
        if len(passing_models) >= thresholds["minimum_models_for_shared_semantic_pair"]
    ]
    result = {
        "schema_version": "phase1105_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "pair_passing_models": pair_models,
        "cross_model_semantic_pairs": cross_model_pairs,
        "semantic_event_mapping_authorized": bool(cross_model_pairs),
        "hidden_state_accessed_in_phase1105": False,
        "automatic_next_required": True,
        "automatic_next_task": (
            "Phase1106: preregister a signed query-triggered semantic routing event map on the shared natural-synonym pairs; do not localize components or claim causality."
            if cross_model_pairs else
            "Phase1106: repair or independently validate natural paraphrase/interface behavior before any semantic hidden-state access."
        ),
        "claim_boundary": (
            "A shared pass establishes only behavior-level equivalence for this task and pair. "
            "It does not establish a shared neural direction or causal semantic transport."
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_pairs_by_model": {model: row["passing_pairs"] for model, row in models.items()},
        "cross_model_semantic_pairs": cross_model_pairs,
        "semantic_event_mapping_authorized": result["semantic_event_mapping_authorized"],
        "authorization_digest": result["authorization_digest"],
    }))


if __name__ == "__main__":
    main()
