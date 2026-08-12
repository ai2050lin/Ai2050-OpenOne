#!/usr/bin/env python3
"""Finalize the preregistered Phase1088 answer-balanced binding analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1088_answer_balanced_color_binding_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_finalize as engine


def heldout_pair_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cosine_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_split_cosine"]
    )
    advantage_min = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_shared_content_over_null_advantage"
        ]
    )
    minimum_cells = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_heldout_pair_cells"]
    )
    by_model = {}
    for model_name, data in models.items():
        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            cells = []
            for heldout in protocol.OPERATIONS:
                others = tuple(
                    operation for operation in protocol.OPERATIONS
                    if operation != heldout
                )
                for world in protocol.WORLDS:
                    def centroid(field: str) -> np.ndarray:
                        return engine.unit_vector(np.mean(np.stack([
                            engine.unit_vector(engine.profile(
                                data, operation, (world,), "discovery",
                                field, replicate,
                            ))
                            for operation in others
                        ]), axis=0))

                    content_source = centroid("content")
                    content_target = engine.unit_vector(engine.profile(
                        data, heldout, (world,), "confirmation", "content",
                        replicate,
                    ))
                    null_source = centroid("field_null")
                    null_target = engine.unit_vector(engine.profile(
                        data, heldout, (world,), "confirmation", "field_null",
                        replicate,
                    ))
                    content_cosine = engine.cosine(
                        content_source, content_target
                    )
                    null_cosine = engine.cosine(null_source, null_target)
                    advantage = content_cosine - null_cosine
                    cells.append({
                        "heldout_color_pair": heldout,
                        "world": world,
                        "content_cosine": content_cosine,
                        "field_null_cosine": null_cosine,
                        "content_over_null_advantage": advantage,
                        "passed": (
                            content_cosine >= cosine_min
                            and advantage >= advantage_min
                        ),
                    })
            passing = sum(int(row["passed"]) for row in cells)
            replicate_rows.append({
                "replicate": replicate,
                "passing_cells": passing,
                "total_cells": len(cells),
                "passed": passing >= minimum_cells,
                "cells": cells,
            })
        by_model[model_name] = {
            "replicates": replicate_rows,
            "passed": all(row["passed"] for row in replicate_rows),
        }
    return {"by_model": by_model}


def write_output(
    root: Path,
    filename: str,
    schema: str,
    payload: dict[str, Any],
    digest_key: str,
    protocol_digest: str,
) -> None:
    row = {
        "schema_version": schema,
        "phase": protocol.PHASE,
        "protocol_digest": protocol_digest,
        **payload,
    }
    engine.write_output(root / filename, row, digest_key)


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {name: engine.load_model(name) for name in protocol.MODELS}
    root = protocol.OUT_ROOT / "analysis"
    root.mkdir(parents=True, exist_ok=True)

    shared = engine.shared_field_analysis(models)
    controls = engine.control_analysis(models)
    residuals = engine.attribute_analysis(models)
    cross_model = engine.cross_model_analysis(models)
    decomposition = engine.decomposition_analysis(models)
    physical = engine.physical_map(models)
    projection = engine.projection_gate(models)
    numeric = engine.numeric_gate(models, authorization)
    heldout = heldout_pair_analysis(models)

    minimum_models = int(
        prereg["evidence_thresholds"]["minimum_behavior_models"]
    )
    minimum_numeric = int(
        prereg["evidence_thresholds"]["minimum_numeric_models"]
    )
    minimum_cross_pairs = int(
        prereg["evidence_thresholds"]["minimum_cross_model_geometry_pairs"]
    )
    healthy = set(numeric["healthy_models"])

    def passing(source: dict[str, Any], key: str) -> list[str]:
        return [
            name for name, row in source["by_model"].items()
            if name in healthy and row[key]
        ]

    p2_models = [
        name for name, row in projection["by_model"].items()
        if name in healthy and row["passed"]
    ]
    p3_models = passing(shared, "split_gate_passed")
    p4_models = passing(shared, "cross_world_gate_passed")
    p5_models = passing(controls, "combined_gate_passed")
    p6_models = passing(residuals, "split_gate_passed")
    p7_models = passing(residuals, "heldout_gate_passed")
    p10_models = passing(heldout, "passed")
    cross_specific = [
        row for row in cross_model["rows"]
        if row["source_model"] in healthy
        and row["target_model"] in healthy
        and row["posthoc_content_specific_advantage_passed"]
    ]

    predictions = {
        "P1": authorization["predictions"]["P1"],
        "P2": {
            "passed": len(p2_models) >= minimum_models,
            "passing_models": p2_models,
            "by_model": projection["by_model"],
        },
        "P3": {
            "passed": len(p3_models) >= minimum_models,
            "passing_models": p3_models,
        },
        "P4": {
            "passed": len(p4_models) >= minimum_models,
            "passing_models": p4_models,
        },
        "P5": {
            "passed": len(p5_models) >= minimum_models,
            "passing_models": p5_models,
        },
        "P6": {
            "passed": len(p6_models) >= minimum_models,
            "passing_models": p6_models,
        },
        "P7": {
            "passed": len(p7_models) >= minimum_models,
            "passing_models": p7_models,
        },
        "P8": {
            "passed": len(cross_specific) >= minimum_cross_pairs,
            "content_specific_passing_directed_pairs": len(cross_specific),
            "rows": cross_specific,
        },
        "P9": {
            "passed": len(healthy) >= minimum_numeric,
            "healthy_models": sorted(healthy),
            "all_models_passed": numeric["passed"],
            "by_model": numeric["by_model"],
        },
        "P10": {
            "passed": len(p10_models) >= minimum_models,
            "passing_models": p10_models,
        },
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]
    full = all(
        predictions[f"P{index}"]["passed"] for index in range(1, 11)
    )
    if full:
        decision = "authorize_full_answer_balanced_binding_atlas_and_causal_confirmation"
    elif predictions["P3"]["passed"] and predictions["P4"]["passed"]:
        decision = "retain_answer_balanced_binding_map_and_repair_failed_controls"
    else:
        decision = "truth_route_explains_phase1087_shared_field; stop_escalation"
    automatic = {
        "schema_version": "phase1088_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "decision": decision,
        "full_atlas_authorized": full,
        "local_causal_authorized": full,
        "offline_diagnostics_authorized": True,
        "failed_predictions": failed,
        "allowed_next": (
            "No component or neuron intervention unless every answer-balanced "
            "gate passes. Preserve any repeated field as descriptive evidence."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)

    outputs = (
        ("shared_binding_field_audit.json", "phase1088_shared_binding_field_audit.v1", shared, "shared_binding_field_digest"),
        ("surface_control_audit.json", "phase1088_surface_control_audit.v1", controls, "surface_control_digest"),
        ("color_pair_residual_audit.json", "phase1088_color_pair_residual_audit.v1", residuals, "color_pair_residual_digest"),
        ("heldout_color_pair_audit.json", "phase1088_heldout_color_pair_audit.v1", heldout, "heldout_color_pair_digest"),
        ("cross_model_geometry.json", "phase1088_cross_model_geometry.v1", cross_model, "cross_model_geometry_digest"),
        ("signed_decomposition.json", "phase1088_signed_decomposition.v1", decomposition, "signed_decomposition_digest"),
        ("physical_map.json", "phase1088_physical_map.v1", physical, "physical_map_digest"),
        ("projection_audit.json", "phase1088_projection_audit.v1", projection, "projection_gate_digest"),
        ("numeric_audit.json", "phase1088_numeric_audit.v1", numeric, "numeric_audit_digest"),
        (
            "prediction_audit.json", "phase1088_prediction_audit.v1",
            {
                "predictions": predictions,
                "passed_predictions": passed,
                "failed_predictions": failed,
            },
            "prediction_audit_digest",
        ),
    )
    for filename, schema, payload, key in outputs:
        write_output(
            root, filename, schema, payload, key, prereg["protocol_digest"]
        )
    protocol.write_json(root / "automatic_next.json", automatic)

    final = {
        "schema_version": "phase1088_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1087_protocol_digest": prereg[
            "source_phase1087_protocol_digest"
        ],
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "models": {
            name: {
                "summary_digest": data["summary"]["summary_digest"],
                "npz_sha256": data["npz_sha256"],
                "candidate_accuracy": data["summary"]["candidate_accuracy"],
                "candidate_finite_fraction": data["summary"]["candidate_finite_fraction"],
                "hidden_finite_fraction": data["summary"]["hidden_finite_fraction_lower_bound"],
                "event_count": data["summary"]["event_count"],
            }
            for name, data in models.items()
        },
        "predictions": predictions,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "shared_binding_summary": {
            name: {
                "split_gate_passed": row["split_gate_passed"],
                "cross_world_gate_passed": row["cross_world_gate_passed"],
                "median_signed_shared_fraction": row[
                    "median_signed_shared_fraction"
                ],
            }
            for name, row in shared["by_model"].items()
        },
        "control_summary": controls["by_model"],
        "color_pair_residual_summary": {
            name: {
                "split_gate_passed": row["split_gate_passed"],
                "heldout_world_gate_passed": row["heldout_gate_passed"],
                "replicate_split_top1": [
                    value["split_assignment"]["top1_correct"]
                    for value in row["replicates"]
                ],
                "replicate_heldout_world_pass_counts": [
                    value["passing_heldout_worlds"]
                    for value in row["replicates"]
                ],
            }
            for name, row in residuals["by_model"].items()
        },
        "heldout_color_pair_summary": {
            name: {
                "passed": row["passed"],
                "replicate_pass_counts": [
                    value["passing_cells"] for value in row["replicates"]
                ],
            }
            for name, row in heldout["by_model"].items()
        },
        "cross_model_geometry": cross_model,
        "numeric_integrity": numeric,
        "automatic_next": automatic,
        "evidence_boundary": {
            "supports": (
                "A descriptive answer-balanced binding map when its content "
                "field beats the same-word anchor null."
            ),
            "does_not_support": [
                "color semantics if the answer-balanced field collapses",
                "a fixed neuron or context-free relation vector",
                "causal transport without all prospective gates",
                "brain homology, optimality, or new mathematics",
            ],
        },
    }
    engine.write_output(root / "final_summary.json", final, "summary_digest")
    print({
        "phase": protocol.PHASE,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "healthy_models": sorted(healthy),
        "decision": decision,
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
