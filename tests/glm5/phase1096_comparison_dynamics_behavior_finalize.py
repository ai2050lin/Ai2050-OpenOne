#!/usr/bin/env python3
"""Finalize Phase1096 behavior authorization before hidden-state access."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1096_comparison_dynamics_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("protocol audit failed")
    thresholds = prereg["evidence_thresholds"]
    model_records = {}
    passing_models = []
    for model_name in protocol.MODELS:
        summary = protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
        )
        relation_records = {}
        passing_relations = []
        for relation in protocol.RELATIONS:
            candidate_cells = [
                summary["per_cell"]["|".join((relation, surface, split, panel))]
                for surface in protocol.SURFACES
                for split in protocol.SPLITS
                for panel in protocol.PANELS
            ]
            generation_cells = [
                summary["per_generation_cell"]["|".join((relation, surface, split))]
                for surface in protocol.SURFACES
                for split in protocol.SPLITS
            ]
            finite_min = min(row["candidate_finite_fraction"] for row in candidate_cells)
            candidate_min = min(row["candidate_accuracy"] for row in candidate_cells)
            generation_min = min(row["target_before_distractor_accuracy"] for row in generation_cells)
            passed = (
                finite_min >= thresholds["minimum_candidate_finite_fraction"]
                and candidate_min >= thresholds["minimum_candidate_accuracy"]
                and generation_min >= thresholds["minimum_generation_accuracy"]
            )
            if passed:
                passing_relations.append(relation)
            relation_records[relation] = {
                "minimum_candidate_finite_fraction": finite_min,
                "minimum_candidate_accuracy": candidate_min,
                "minimum_generation_accuracy": generation_min,
                "passed": passed,
            }
        model_passed = len(passing_relations) >= thresholds["minimum_relations_per_model"]
        if model_passed:
            passing_models.append(model_name)
        model_records[model_name] = {
            "summary_digest": summary["summary_digest"],
            "relations": relation_records,
            "passing_relations": passing_relations,
            "model_behavior_passed": model_passed,
            "overall_candidate_finite_fraction": summary["candidate_finite_fraction"],
            "overall_candidate_accuracy": summary["candidate_accuracy"],
            "overall_generation_accuracy": summary["generation_target_before_distractor_accuracy"],
        }
    authorized = len(passing_models) >= thresholds["minimum_behavior_models"]
    result = {
        "schema_version": "phase1096_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": model_records,
        "passing_models": passing_models,
        "hidden_scan_authorized": authorized,
        "decision": (
            "run_three_ledger_hidden_scan"
            if authorized else "stop_before_hidden_scan_and_repair_behavior_base"
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result)
    print({
        "phase": protocol.PHASE,
        "passing_models": passing_models,
        "hidden_scan_authorized": authorized,
        "authorization_digest": result["authorization_digest"],
    })


if __name__ == "__main__":
    main()
