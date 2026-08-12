#!/usr/bin/env python3
"""Finalize the preregistered Phase1089 truth-matched analysis."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1089_truth_matched_color_binding_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_finalize as engine


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_phase1088_model(model_name: str) -> dict[str, Any]:
    root = protocol.SOURCE_PHASE1088 / "atlas" / model_name
    summary = protocol.read_json(root / "summary.json")
    with np.load(root / "signed_fields.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    count = arrays["direction_count"]
    mean = np.divide(
        arrays["direction_sum"],
        count[..., None],
        out=np.zeros_like(arrays["direction_sum"], dtype=np.float32),
        where=count[..., None] > 0,
    )
    return {
        "summary": summary,
        "arrays": arrays,
        "direction_mean": mean,
        "npz_sha256": file_sha256(root / "signed_fields.npz"),
    }


def cross_phase_pair_analysis(
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cosine_min = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_cross_phase_pair_gram_cosine"
        ]
    )
    advantage_min = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_cross_phase_content_over_null_advantage"
        ]
    )
    top1_min = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_attribute_top1"]
    )
    p_max = float(protocol.EVIDENCE_THRESHOLDS["permutation_p_max"])
    by_model = {}
    for model_name, current in models.items():
        previous = load_phase1088_model(model_name)
        current_projection = protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model_name / "projection_audit.json"
        )
        previous_projection = protocol.read_json(
            protocol.SOURCE_PHASE1088 / "atlas" / model_name
            / "projection_audit.json"
        )
        basis_equal = all(
            current_projection["replicates"][replicate]["matrix_digest"]
            == previous_projection["replicates"][replicate]["matrix_digest"]
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES)
        )
        rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            for direction in ("1088_discovery_to_1089_confirmation",
                              "1089_discovery_to_1088_confirmation"):
                if direction.startswith("1088"):
                    source, target = previous, current
                else:
                    source, target = current, previous
                content_source = engine.operation_bank(
                    source, protocol.WORLDS, "discovery", "content",
                    replicate, centered=True,
                )
                content_target = engine.operation_bank(
                    target, protocol.WORLDS, "confirmation", "content",
                    replicate, centered=True,
                )
                null_source = engine.operation_bank(
                    source, protocol.WORLDS, "discovery", "field_null",
                    replicate, centered=True,
                )
                null_target = engine.operation_bank(
                    target, protocol.WORLDS, "confirmation", "field_null",
                    replicate, centered=True,
                )
                content_geometry = engine.cosine(
                    engine.relation_vector(content_source),
                    engine.relation_vector(content_target),
                )
                null_geometry = engine.cosine(
                    engine.relation_vector(null_source),
                    engine.relation_vector(null_target),
                )
                assignment = engine.assignment_record(
                    content_source,
                    content_target,
                    comparison="cross_phase_pair_identity",
                    model=model_name,
                    replicate=replicate,
                    direction=direction,
                    field="content",
                )
                null_assignment = engine.assignment_record(
                    null_source,
                    null_target,
                    comparison="cross_phase_pair_identity",
                    model=model_name,
                    replicate=replicate,
                    direction=direction,
                    field="field_null",
                )
                advantage = content_geometry - null_geometry
                passed = (
                    basis_equal
                    and content_geometry >= cosine_min
                    and advantage >= advantage_min
                    and assignment["top1_correct"] >= top1_min
                    and assignment["exact_upper_tail_p"] <= p_max
                    and assignment["top1_correct"]
                    > null_assignment["top1_correct"]
                )
                rows.append({
                    "replicate": replicate,
                    "direction": direction,
                    "content_pair_gram_cosine": content_geometry,
                    "field_null_pair_gram_cosine": null_geometry,
                    "content_over_null_geometry_advantage": advantage,
                    "content_assignment": assignment,
                    "field_null_assignment": null_assignment,
                    "passed": passed,
                })
        by_model[model_name] = {
            "projection_basis_equal": basis_equal,
            "rows": rows,
            "passed": all(row["passed"] for row in rows),
            "phase1088_npz_sha256": previous["npz_sha256"],
            "phase1089_npz_sha256": current["npz_sha256"],
        }
    return {
        "by_model": by_model,
        "definition": (
            "Compare centered eight-color-pair Gram geometry and exact pair "
            "assignment in both cross-phase directions. The same within-model "
            "projection basis is required. A stable null is not credited as "
            "content-specific preservation."
        ),
    }


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
    cross_phase = cross_phase_pair_analysis(models)

    minimum_models = int(
        prereg["evidence_thresholds"]["minimum_behavior_models"]
    )
    minimum_numeric = int(
        prereg["evidence_thresholds"]["minimum_numeric_models"]
    )
    minimum_cross_pairs = int(
        prereg["evidence_thresholds"]
        ["minimum_cross_model_geometry_pairs"]
    )
    minimum_cross_phase = int(
        prereg["evidence_thresholds"]["minimum_cross_phase_models"]
    )
    healthy = set(numeric["healthy_models"])

    def passing(source: dict[str, Any], key: str) -> list[str]:
        return [
            name for name, row in source["by_model"].items()
            if name in healthy and bool(row[key])
        ]

    p3_models = [
        name for name, row in projection["by_model"].items()
        if name in healthy and row["passed"]
    ]
    p4_models = [
        name for name in healthy
        if shared["by_model"][name]["split_gate_passed"]
        and shared["by_model"][name]["cross_world_gate_passed"]
    ]
    p5_models = passing(residuals, "split_gate_passed")
    p6_models = passing(residuals, "heldout_gate_passed")
    p8_models = passing(controls, "combined_gate_passed")
    p10_models = passing(cross_phase, "passed")
    cross_specific = [
        row for row in cross_model["rows"]
        if row["source_model"] in healthy
        and row["target_model"] in healthy
        and row["posthoc_content_specific_advantage_passed"]
    ]

    predictions = {
        "P1": authorization["predictions"]["P1"],
        "P2": authorization["predictions"]["P2"],
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
            "passed": len(cross_specific) >= minimum_cross_pairs,
            "content_specific_passing_directed_pairs": len(cross_specific),
            "rows": cross_specific,
        },
        "P8": {
            "passed": len(p8_models) >= minimum_models,
            "passing_models": p8_models,
        },
        "P9": {
            "passed": len(healthy) >= minimum_numeric,
            "healthy_models": sorted(healthy),
            "by_model": numeric["by_model"],
        },
        "P10": {
            "passed": len(p10_models) >= minimum_cross_phase,
            "passing_models": p10_models,
        },
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]
    cross_surface_authorized = all(
        predictions[name]["passed"]
        for name in ("P1", "P2", "P3", "P4", "P5", "P6", "P9", "P10")
    )
    if cross_surface_authorized:
        decision = "authorize_phase1090_cross_surface_behavior_pilot"
    elif predictions["P5"]["passed"] and predictions["P6"]["passed"]:
        decision = (
            "retain_truth_matched_pair_map_but_do_not_start_cross_surface_scan; "
            "common_field_and_surface_or_cross_model_specificity_failed"
        )
    else:
        decision = (
            "truth_matched_null_explains_pair_identity; stop_color_escalation"
        )

    outputs = (
        ("shared_binding_field_audit.json", "phase1089_shared_binding_field_audit.v1", shared, "shared_binding_field_digest"),
        ("surface_control_audit.json", "phase1089_surface_control_audit.v1", controls, "surface_control_digest"),
        ("color_pair_residual_audit.json", "phase1089_color_pair_residual_audit.v1", residuals, "color_pair_residual_digest"),
        ("cross_model_geometry.json", "phase1089_cross_model_geometry.v1", cross_model, "cross_model_geometry_digest"),
        ("signed_decomposition.json", "phase1089_signed_decomposition.v1", decomposition, "signed_decomposition_digest"),
        ("physical_map.json", "phase1089_physical_map.v1", physical, "physical_map_digest"),
        ("projection_audit.json", "phase1089_projection_audit.v1", projection, "projection_gate_digest"),
        ("numeric_audit.json", "phase1089_numeric_audit.v1", numeric, "numeric_audit_digest"),
        ("cross_phase_pair_geometry.json", "phase1089_cross_phase_pair_geometry.v1", cross_phase, "cross_phase_pair_digest"),
    )
    for filename, schema, payload, digest_key in outputs:
        write_output(
            root, filename, schema, payload, digest_key,
            prereg["protocol_digest"],
        )

    automatic = {
        "schema_version": "phase1089_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "decision": decision,
        "cross_surface_behavior_pilot_authorized": cross_surface_authorized,
        "cross_surface_hidden_scan_authorized": False,
        "local_causal_authorized": False,
        "reason": (
            "Cross-surface work can begin automatically only after the shared "
            "field, held-out pair specificity, numeric, and cross-phase gates "
            "all pass. Phase1089 never authorizes component or neuron causality."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)
    protocol.write_json(root / "automatic_next.json", automatic)

    previous_shared = protocol.read_json(
        protocol.SOURCE_PHASE1088 / "analysis"
        / "shared_binding_field_audit.json"
    )
    summary = {
        "schema_version": "phase1089_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "predictions": predictions,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "decision": decision,
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
        "truth_matched_shared_summary": {
            name: row["median_signed_shared_fraction"]
            for name, row in shared["by_model"].items()
        },
        "phase1088_shared_summary": {
            name: row["median_signed_shared_fraction"]
            for name, row in previous_shared["by_model"].items()
        },
        "control_ratios": {
            name: {
                "template_to_content": row["median_surface_to_content_ratio"],
                "output_to_content": row["median_output_to_content_ratio"],
            }
            for name, row in controls["by_model"].items()
        },
        "pair_residual_gates": {
            name: {
                "split": row["split_gate_passed"],
                "heldout_world": row["heldout_gate_passed"],
            }
            for name, row in residuals["by_model"].items()
        },
        "cross_phase_pair_gate": {
            name: row["passed"]
            for name, row in cross_phase["by_model"].items()
        },
        "interpretation": [
            "Truth-marginal matching does not restore a content-specific common color-binding field.",
            "Pair residual identity remains repeatable within model, but the matched null is usually more stable at the common-field level.",
            "The surviving pair map is compatible with lexical-pair and context-conditioned routing; it is not abstract color semantics.",
            "No component, head, MLP, or neuron causal localization is authorized.",
        ],
        "automatic_next_digest": automatic["automatic_next_digest"],
    }
    summary["summary_digest"] = protocol.digest(summary)
    protocol.write_json(root / "final_summary.json", summary)
    print({
        "phase": protocol.PHASE,
        "passed": passed,
        "failed": failed,
        "decision": decision,
        "summary_digest": summary["summary_digest"],
    })


if __name__ == "__main__":
    main()
