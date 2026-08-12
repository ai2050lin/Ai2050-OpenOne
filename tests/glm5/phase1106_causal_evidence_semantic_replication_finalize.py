#!/usr/bin/env python3
"""Freeze Phase1106 claim-aligned semantic replication decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1106_causal_evidence_semantic_replication_protocol as protocol


def rows_for(summary: dict[str, Any], split: str, route: str, congruence: str, target: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for template in protocol.TEMPLATES_BY_SPLIT[split]:
        targets = protocol.TARGET_RELATIONS if target is None else (target,)
        for target_value in targets:
            rows.append(summary["per_cell"][f"{split}|{template}|{route}|{congruence}|{target_value}"])
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
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
    models = {}
    passing_models = []
    for model in protocol.MODELS:
        summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")
        split_results = {}
        for split in protocol.TEMPLATES_BY_SPLIT:
            semantic_routes = ("close_synonym", "natural_definition")
            route_metrics = {
                route: aggregate(rows_for(summary, split, route, "conflict"))
                for route in semantic_routes
            }
            target_metrics = {
                f"{route}|q{target}": aggregate(rows_for(summary, split, route, "conflict", target))
                for route in semantic_routes for target in protocol.TARGET_RELATIONS
            }
            template_gates = {
                route: all(
                    aggregate([
                        summary["per_cell"][f"{split}|{template}|{route}|conflict|0"],
                        summary["per_cell"][f"{split}|{template}|{route}|conflict|1"],
                    ])["candidate_accuracy"] >= thresholds["minimum_semantic_template_accuracy"]
                    for template in protocol.TEMPLATES_BY_SPLIT[split]
                )
                for route in semantic_routes
            }
            gates = {
                "finite": all(row["candidate_finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"] for row in route_metrics.values()),
                "close_synonym_templates": template_gates["close_synonym"],
                "natural_definition_templates": template_gates["natural_definition"],
                "close_synonym_aggregate": route_metrics["close_synonym"]["candidate_accuracy"] >= thresholds["minimum_semantic_aggregate_accuracy"],
                "natural_definition_aggregate": route_metrics["natural_definition"]["candidate_accuracy"] >= thresholds["minimum_semantic_aggregate_accuracy"],
                "both_target_relations": all(row["candidate_accuracy"] >= thresholds["minimum_semantic_target_accuracy"] for row in target_metrics.values()),
            }
            diagnostics = {
                route: {
                    congruence: aggregate(rows_for(summary, split, route, congruence))
                    for congruence in protocol.CONGRUENCES
                }
                for route in protocol.ROUTE_TYPES
            }
            split_results[split] = {
                "semantic_route_metrics": route_metrics,
                "semantic_target_metrics": target_metrics,
                "diagnostics": diagnostics,
                "gates": gates,
                "passed": all(gates.values()),
            }
        passed = all(row["passed"] for row in split_results.values())
        if passed:
            passing_models.append(model)
        models[model] = {
            "summary_digest": summary["summary_digest"],
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "candidate_accuracy": summary["candidate_accuracy"],
            "splits": split_results,
            "passed_both_splits": passed,
        }
    shared = len(passing_models) >= thresholds["minimum_models_for_shared_replication"]
    result = {
        "schema_version": "phase1106_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "relation_pair": protocol.RELATION_PAIR,
        "models": models,
        "passing_models": passing_models,
        "cross_model_semantic_replication": shared,
        "signed_event_mapping_authorized": shared,
        "hidden_state_accessed_in_phase1106": False,
        "automatic_next_required": True,
        "automatic_next_task": (
            "Phase1107: map the signed query-triggered semantic routing event with matched exact, ordinal, and congruent controls; no causal or component claim."
            if shared else
            "Stop semantic hidden-state access and repair the natural behavior protocol or add a larger-model behavior arm."
        ),
        "frozen_conclusion": (
            "Natural synonym routing for causal influence versus evidence strength replicated across independent words, prompts, names, and at least two models."
            if shared else
            "Natural synonym routing did not meet the cross-model independent replication gate."
        ),
        "claim_boundary": "This is behavior-level relation selection, not a shared vector, physical coordinate, causal transporter, or full semantic code.",
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_models": passing_models,
        "cross_model_semantic_replication": shared,
        "signed_event_mapping_authorized": shared,
        "authorization_digest": result["authorization_digest"],
    }))


if __name__ == "__main__":
    main()
