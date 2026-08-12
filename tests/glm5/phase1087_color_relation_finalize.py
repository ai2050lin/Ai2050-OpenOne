#!/usr/bin/env python3
"""Finalize the preregistered Phase1087 color-relation analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1087_color_relation_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_finalize as engine


def heldout_pair_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    threshold_cosine = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_split_cosine"]
    )
    threshold_advantage = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_shared_content_over_null_advantage"
        ]
    )
    minimum_cells = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_heldout_pair_cells"]
    )
    by_model: dict[str, Any] = {}
    for model_name, data in models.items():
        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            cells = []
            for heldout in protocol.OPERATIONS:
                source_operations = tuple(
                    operation for operation in protocol.OPERATIONS
                    if operation != heldout
                )
                for world in protocol.WORLDS:
                    content_source = engine.unit_vector(np.mean(np.stack([
                        engine.unit_vector(engine.profile(
                            data, operation, (world,), "discovery",
                            "content", replicate,
                        ))
                        for operation in source_operations
                    ]), axis=0))
                    content_target = engine.unit_vector(engine.profile(
                        data, heldout, (world,), "confirmation",
                        "content", replicate,
                    ))
                    null_source = engine.unit_vector(np.mean(np.stack([
                        engine.unit_vector(engine.profile(
                            data, operation, (world,), "discovery",
                            "field_null", replicate,
                        ))
                        for operation in source_operations
                    ]), axis=0))
                    null_target = engine.unit_vector(engine.profile(
                        data, heldout, (world,), "confirmation",
                        "field_null", replicate,
                    ))
                    content_cosine = engine.cosine(
                        content_source, content_target
                    )
                    null_cosine = engine.cosine(null_source, null_target)
                    advantage = content_cosine - null_cosine
                    passed = (
                        content_cosine >= threshold_cosine
                        and advantage >= threshold_advantage
                    )
                    cells.append({
                        "heldout_color_pair": heldout,
                        "world": world,
                        "content_cosine": content_cosine,
                        "field_null_cosine": null_cosine,
                        "content_over_null_advantage": advantage,
                        "passed": passed,
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
    return {
        "by_model": by_model,
        "definition": (
            "For each held-out color pair and entity world, average the seven "
            "other discovery-pair directions, then compare with the held-out "
            "confirmation direction. The same calculation is repeated for the "
            "query-irrelevant binding null."
        ),
    }


def write_output(
    analysis_root: Path,
    filename: str,
    schema: str,
    payload: dict[str, Any],
    digest_key: str,
) -> None:
    row = {
        "schema_version": schema,
        "phase": protocol.PHASE,
        "protocol_digest": protocol.read_json(
            protocol.OUT_ROOT / "protocol" / "preregistration.json"
        )["protocol_digest"],
        **payload,
    }
    engine.write_output(analysis_root / filename, row, digest_key)


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior_authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {name: engine.load_model(name) for name in protocol.MODELS}
    analysis_root = protocol.OUT_ROOT / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    shared = engine.shared_field_analysis(models)
    controls = engine.control_analysis(models)
    pair_residuals = engine.attribute_analysis(models)
    cross_model = engine.cross_model_analysis(models)
    decomposition = engine.decomposition_analysis(models)
    physical = engine.physical_map(models)
    projection = engine.projection_gate(models)
    numeric = engine.numeric_gate(models, behavior_authorization)
    heldout_pairs = heldout_pair_analysis(models)

    minimum_models = int(
        prereg["evidence_thresholds"]["minimum_behavior_models"]
    )
    minimum_numeric_models = int(
        prereg["evidence_thresholds"]["minimum_numeric_models"]
    )
    healthy = set(numeric["healthy_models"])

    def passing_models(source: dict[str, Any], key: str) -> list[str]:
        return [
            name for name, row in source["by_model"].items()
            if name in healthy and bool(row[key])
        ]

    p3_models = [
        name for name, row in projection["by_model"].items()
        if name in healthy and row["passed"]
    ]
    p4_models = passing_models(shared, "split_gate_passed")
    p5_models = passing_models(shared, "cross_world_gate_passed")
    p6_models = passing_models(controls, "combined_gate_passed")
    p7_models = passing_models(pair_residuals, "split_gate_passed")
    p8_models = passing_models(pair_residuals, "heldout_gate_passed")
    p11_models = passing_models(heldout_pairs, "passed")

    content_specific_cross_rows = [
        row for row in cross_model["rows"]
        if row["source_model"] in healthy
        and row["target_model"] in healthy
        and row["posthoc_content_specific_advantage_passed"]
    ]
    minimum_cross_pairs = int(
        prereg["evidence_thresholds"]["minimum_cross_model_geometry_pairs"]
    )

    predictions = {
        "P1": behavior_authorization["predictions"]["P1"],
        "P2": behavior_authorization["predictions"]["P2"],
        "P3": {
            "passed": len(p3_models) >= minimum_models,
            "passing_models": p3_models,
            "by_model": projection["by_model"],
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
            "passed": len(p8_models) >= minimum_models,
            "passing_models": p8_models,
        },
        "P9": {
            "passed": len(content_specific_cross_rows) >= minimum_cross_pairs,
            "content_specific_passing_directed_pairs": len(
                content_specific_cross_rows
            ),
            "rows": content_specific_cross_rows,
        },
        "P10": {
            "passed": len(healthy) >= minimum_numeric_models,
            "healthy_models": sorted(healthy),
            "all_models_passed": numeric["passed"],
            "by_model": numeric["by_model"],
        },
        "P11": {
            "passed": len(p11_models) >= minimum_models,
            "passing_models": p11_models,
        },
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]
    full_atlas_authorized = all(
        predictions[f"P{index}"]["passed"] for index in range(1, 12)
    )
    local_causal_authorized = full_atlas_authorized
    if full_atlas_authorized:
        decision = "continue_to_full_color_relation_atlas_and_minimum_component_alliance"
    elif (
        predictions["P4"]["passed"]
        and predictions["P5"]["passed"]
        and predictions["P11"]["passed"]
    ):
        decision = "retain_repeated_relation_field_and_purify_failed_controls"
    else:
        decision = "retain_descriptive_map_and_redesign_relation_or_null_protocol"
    automatic = {
        "schema_version": "phase1087_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "decision": decision,
        "offline_diagnostics_authorized": True,
        "full_atlas_authorized": full_atlas_authorized,
        "local_causal_authorized": local_causal_authorized,
        "failed_predictions": failed,
        "allowed_next": (
            "Preserve the frozen physical map. Component or neuron selection "
            "is forbidden unless every prospective gate passes. A failed gate "
            "may motivate a new independent protocol, never deletion of a "
            "repeated descriptive structure."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)

    write_output(
        analysis_root, "shared_relation_field_audit.json",
        "phase1087_shared_relation_field_audit.v1", shared,
        "shared_relation_field_digest",
    )
    write_output(
        analysis_root, "surface_control_audit.json",
        "phase1087_surface_control_audit.v1", controls,
        "surface_control_digest",
    )
    write_output(
        analysis_root, "color_pair_residual_audit.json",
        "phase1087_color_pair_residual_audit.v1", pair_residuals,
        "color_pair_residual_digest",
    )
    write_output(
        analysis_root, "heldout_color_pair_audit.json",
        "phase1087_heldout_color_pair_audit.v1", heldout_pairs,
        "heldout_color_pair_digest",
    )
    write_output(
        analysis_root, "cross_model_geometry.json",
        "phase1087_cross_model_geometry.v1", cross_model,
        "cross_model_geometry_digest",
    )
    write_output(
        analysis_root, "signed_decomposition.json",
        "phase1087_signed_decomposition.v1", decomposition,
        "signed_decomposition_digest",
    )
    write_output(
        analysis_root, "physical_map.json",
        "phase1087_physical_map.v1", physical,
        "physical_map_digest",
    )
    write_output(
        analysis_root, "projection_audit.json",
        "phase1087_projection_audit.v1", projection,
        "projection_gate_digest",
    )
    write_output(
        analysis_root, "numeric_audit.json",
        "phase1087_numeric_audit.v1", numeric,
        "numeric_audit_digest",
    )
    write_output(
        analysis_root, "prediction_audit.json",
        "phase1087_prediction_audit.v1",
        {
            "predictions": predictions,
            "passed_predictions": passed,
            "failed_predictions": failed,
        },
        "prediction_audit_digest",
    )
    protocol.write_json(analysis_root / "automatic_next.json", automatic)

    final = {
        "schema_version": "phase1087_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
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
        "shared_relation_summary": {
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
            for name, row in pair_residuals["by_model"].items()
        },
        "heldout_color_pair_summary": {
            name: {
                "passed": row["passed"],
                "replicate_pass_counts": [
                    value["passing_cells"] for value in row["replicates"]
                ],
            }
            for name, row in heldout_pairs["by_model"].items()
        },
        "cross_model_geometry": cross_model,
        "numeric_integrity": numeric,
        "automatic_next": automatic,
        "evidence_boundary": {
            "supports": (
                "A descriptive, signed middle-band map of one controlled "
                "color-binding relation, with same-word query-irrelevant nulls."
            ),
            "does_not_support": [
                "a complete color concept or language mechanism",
                "semantic reuse when content does not beat the matched null",
                "a fixed neuron or context-free color vector",
                "direct physical correspondence across models",
                "brain homology, biological optimality, or new mathematics",
            ],
        },
    }
    engine.write_output(
        analysis_root / "final_summary.json", final, "summary_digest"
    )
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
