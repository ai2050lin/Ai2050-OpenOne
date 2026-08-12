#!/usr/bin/env python3
"""Freeze Phase1094 behavior authorization before hidden-state collection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1094_semantic_topology_protocol as protocol


def condition_key(attribute: str, topology: str, coherence: str, surface: str) -> str:
    return f"{attribute}__{topology}__{coherence}__{surface}"


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    threshold = prereg["evidence_thresholds"]
    minimum_accuracy = float(threshold["minimum_candidate_accuracy"])
    minimum_finite = float(threshold["minimum_candidate_finite_fraction"])
    minimum_generation = float(threshold["minimum_generation_accuracy"])
    minimum_worlds = int(threshold["minimum_behavior_worlds_per_edge"])
    minimum_edges = int(threshold["minimum_behavior_edges_per_condition"])
    minimum_models = int(threshold["minimum_behavior_models"])

    models: dict[str, Any] = {}
    authorized_models = []
    for model_name in protocol.MODELS:
        raw = protocol.read_json(protocol.OUT_ROOT / "pilot" / f"{model_name}.json")
        conditions: dict[str, Any] = {}
        for attribute in protocol.ATTRIBUTES:
            for topology in protocol.TOPOLOGIES:
                for coherence in protocol.COHERENCES:
                    operations = protocol.operation_names(attribute, topology, coherence)
                    for surface in protocol.SURFACES:
                        edges = {}
                        passing_edges = []
                        for operation in operations:
                            panel_worlds = {}
                            for panel in protocol.PANELS:
                                passing_worlds = []
                                world_rows = {}
                                for world in protocol.BASE_WORLDS:
                                    cell = raw["per_cell"][
                                        f"{attribute}__{operation}__{surface}__{world}__{panel}"
                                    ]
                                    passed = (
                                        float(cell["accuracy"]) >= minimum_accuracy
                                        and float(cell["finite_fraction"]) >= minimum_finite
                                    )
                                    if passed:
                                        passing_worlds.append(world)
                                    world_rows[world] = {**cell, "passed": passed}
                                panel_worlds[panel] = {
                                    "passing_worlds": passing_worlds,
                                    "worlds": world_rows,
                                    "passed": len(passing_worlds) >= minimum_worlds,
                                }
                            edge_passed = all(
                                panel_worlds[panel]["passed"] for panel in protocol.PANELS
                            )
                            if edge_passed:
                                passing_edges.append(operation)
                            edges[operation] = {
                                "panels": panel_worlds,
                                "passed": edge_passed,
                            }
                        generation = raw["generation_by_attribute_surface"][
                            f"{attribute}__{surface}"
                        ]
                        generation_passed = all(
                            float(generation[panel]["target_before_distractor_accuracy"])
                            >= minimum_generation
                            for panel in protocol.PANELS
                        )
                        passed = len(passing_edges) >= minimum_edges and generation_passed
                        conditions[condition_key(attribute, topology, coherence, surface)] = {
                            "attribute": attribute,
                            "topology": topology,
                            "coherence": coherence,
                            "surface": surface,
                            "passing_edges": passing_edges,
                            "passing_edge_count": len(passing_edges),
                            "minimum_edges": minimum_edges,
                            "edges": edges,
                            "generation": generation,
                            "generation_passed": generation_passed,
                            "passed": passed,
                        }

        primary_keys = [
            condition_key(protocol.PRIMARY_ATTRIBUTE, topology, coherence, surface)
            for topology in protocol.TOPOLOGIES
            for coherence in protocol.COHERENCES
            for surface in protocol.SURFACES
        ]
        secondary_keys = [
            condition_key(protocol.SECONDARY_ATTRIBUTE, topology, "coherent", surface)
            for topology in protocol.TOPOLOGIES
            for surface in protocol.SURFACES
        ]
        precision_ok = (
            raw["precision"]["has_fp16_parameters"]
            and not raw["precision"]["has_bf16_parameters"]
            and not raw["precision"]["has_quantized_modules"]
        )
        numerical_ok = float(raw["candidate_finite_fraction"]) >= minimum_finite
        primary_passed = all(conditions[key]["passed"] for key in primary_keys)
        secondary_passed = all(conditions[key]["passed"] for key in secondary_keys)
        model_authorized = precision_ok and numerical_ok and primary_passed and secondary_passed
        if model_authorized:
            authorized_models.append(model_name)
        models[model_name] = {
            "precision_ok": precision_ok,
            "numerical_ok": numerical_ok,
            "candidate_finite_fraction": raw["candidate_finite_fraction"],
            "primary_required_conditions": primary_keys,
            "secondary_required_conditions": secondary_keys,
            "primary_passed": primary_passed,
            "secondary_passed": secondary_passed,
            "conditions": conditions,
            "model_authorized": model_authorized,
            "elapsed_seconds": raw["elapsed_seconds"],
            "result_digest": raw["result_digest"],
        }

    predictions = {
        "P1": {
            "passed": bool(static["all_checks_passed"]),
            "criterion": "all static semantic/topology orthogonalization checks pass",
        },
        "P2_precision": {
            "passed": all(row["precision_ok"] for row in models.values()),
            "criterion": "all models load as FP16 without quantization",
        },
        "P2_behavior": {
            "passed": len(authorized_models) >= minimum_models,
            "authorized_models": authorized_models,
            "minimum_models": minimum_models,
        },
    }
    hidden_scan_authorized = all(row["passed"] for row in predictions.values())
    result = {
        "schema_version": "phase1094_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "predictions": predictions,
        "models": models,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_scan_authorized,
        "decision": (
            "run_phase1094_signed_hidden_scan"
            if hidden_scan_authorized else "stop_before_hidden_state_collection"
        ),
        "causal_authorized": False,
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result)
    print({
        "phase": protocol.PHASE,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_scan_authorized,
        "decision": result["decision"],
        "summary_digest": result["summary_digest"],
    })


if __name__ == "__main__":
    main()
