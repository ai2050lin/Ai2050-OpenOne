#!/usr/bin/env python3
"""Finalize Phase1097 conditional-transition evidence gates."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1097_conditional_transition_protocol as protocol


EPSILON = 1e-12


def safe_mean(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    result = np.full(total.shape, np.nan, dtype=np.float64)
    valid = count > 0
    result[valid] = total[valid] / count[valid]
    return result


def finite_vector(value: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(value), value, 0.0).astype(np.float64, copy=False)


def unit(value: np.ndarray) -> np.ndarray:
    clean = finite_vector(value).reshape(-1)
    norm = float(np.linalg.norm(clean))
    return clean / norm if norm > EPSILON else np.zeros_like(clean)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = unit(left)
    right_unit = unit(right)
    denominator = float(np.linalg.norm(left_unit) * np.linalg.norm(right_unit))
    return float(left_unit @ right_unit) if denominator > EPSILON else 0.0


def trajectory_signature(
    amplitude: np.ndarray,
    gram: np.ndarray,
    local_margin: np.ndarray,
) -> np.ndarray:
    """Equal-weight three-block signature fixed before model results."""
    amplitude_block = unit(amplitude)
    upper = np.triu_indices(gram.shape[-1], k=1)
    gram_block = unit(gram[upper])
    margin_block = unit(local_margin)
    return np.concatenate((amplitude_block, gram_block, margin_block)) / math.sqrt(3.0)


def load_model(model_name: str) -> dict[str, Any]:
    root = protocol.OUT_ROOT / "atlas" / model_name
    summary = protocol.read_json(root / "summary.json")
    with np.load(root / "transition_aggregates.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return {
        "summary": summary,
        "amplitude": safe_mean(arrays["amplitude_sum"], arrays["amplitude_count"]),
        "local_margin": safe_mean(arrays["local_margin_sum"], arrays["local_margin_count"]),
        "gram": safe_mean(arrays["gram_sum"], arrays["gram_count"]),
        "panel_alignment": safe_mean(arrays["panel_alignment_sum"], arrays["panel_alignment_count"]),
        "ledger_alignment": safe_mean(arrays["ledger_alignment_sum"], arrays["ledger_alignment_count"]),
        "physical": safe_mean(arrays["physical_sum"], arrays["physical_count"]),
    }


def signature_for(
    data: dict[str, Any], relation: int, surface: int, split: int,
    field: int, role: int,
) -> np.ndarray:
    return trajectory_signature(
        data["amplitude"][relation, surface, split, field, role],
        data["gram"][relation, surface, split, field, role],
        data["local_margin"][relation, surface, split, field, role],
    )


def split_repeat(data: dict[str, Any]) -> dict[str, Any]:
    field_index = {name: index for index, name in enumerate(protocol.FIELDS)}
    role_index = {name: index for index, name in enumerate(protocol.CAPTURE_ROLES)}
    records = []
    for relation_index, relation in enumerate(protocol.RELATIONS):
        for surface_index, surface in enumerate(protocol.SURFACES):
            for role in ("query_end", "answer_boundary"):
                role_number = role_index[role]
                content = cosine(
                    signature_for(data, relation_index, surface_index, 0, field_index["relational_execution"], role_number),
                    signature_for(data, relation_index, surface_index, 1, field_index["relational_execution"], role_number),
                )
                carrier = cosine(
                    signature_for(data, relation_index, surface_index, 0, field_index["relational_carrier"], role_number),
                    signature_for(data, relation_index, surface_index, 1, field_index["relational_carrier"], role_number),
                )
                passed = content >= protocol.EVIDENCE_THRESHOLDS["minimum_split_trajectory_cosine"]
                records.append({
                    "relation": relation,
                    "surface": surface,
                    "role": role,
                    "content_cosine": content,
                    "carrier_cosine": carrier,
                    "passed": passed,
                })
    pass_count = sum(int(row["passed"]) for row in records)
    return {
        "records": records,
        "passing_records": pass_count,
        "record_count": len(records),
        "passed": pass_count >= protocol.EVIDENCE_THRESHOLDS["minimum_split_records"],
    }


def heldout_relation_prediction(data: dict[str, Any]) -> dict[str, Any]:
    field_index = {name: index for index, name in enumerate(protocol.FIELDS)}
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    records = []
    cells = []
    for surface_index, surface in enumerate(protocol.SURFACES):
        for split_index, split in enumerate(protocol.SPLITS):
            passing = 0
            for heldout_index, relation in enumerate(protocol.RELATIONS):
                train_indices = [index for index in range(len(protocol.RELATIONS)) if index != heldout_index]
                content_center = unit(np.mean([
                    signature_for(data, index, surface_index, split_index, field_index["relational_execution"], role)
                    for index in train_indices
                ], axis=0))
                carrier_center = unit(np.mean([
                    signature_for(data, index, surface_index, split_index, field_index["relational_carrier"], role)
                    for index in train_indices
                ], axis=0))
                content_target = signature_for(data, heldout_index, surface_index, split_index, field_index["relational_execution"], role)
                carrier_target = signature_for(data, heldout_index, surface_index, split_index, field_index["relational_carrier"], role)
                content_score = cosine(content_center, content_target)
                carrier_score = cosine(carrier_center, carrier_target)
                advantage = content_score - carrier_score
                passed = (
                    content_score >= protocol.EVIDENCE_THRESHOLDS["minimum_split_trajectory_cosine"]
                    and advantage >= protocol.EVIDENCE_THRESHOLDS["minimum_content_over_carrier_advantage"]
                )
                passing += int(passed)
                records.append({
                    "surface": surface,
                    "split": split,
                    "heldout_relation": relation,
                    "content_cosine": content_score,
                    "carrier_cosine": carrier_score,
                    "content_advantage": advantage,
                    "passed": passed,
                })
            cell_passed = passing >= protocol.EVIDENCE_THRESHOLDS["minimum_heldout_relations"]
            cells.append({"surface": surface, "split": split, "passing_relations": passing, "passed": cell_passed})
    return {"records": records, "cells": cells, "passed": all(row["passed"] for row in cells)}


def behavior_anchor(data: dict[str, Any]) -> dict[str, Any]:
    field_index = {name: index for index, name in enumerate(protocol.FIELDS)}
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    records = []
    for relation_index, relation in enumerate(protocol.RELATIONS):
        for surface_index, surface in enumerate(protocol.SURFACES):
            for split_index, split in enumerate(protocol.SPLITS):
                execution = float(data["local_margin"][relation_index, surface_index, split_index, field_index["relational_execution"], role, -1])
                carrier = float(data["local_margin"][relation_index, surface_index, split_index, field_index["relational_carrier"], role, -1])
                ratio = abs(carrier) / max(abs(execution), EPSILON)
                passed = execution > 0.0 and ratio <= protocol.EVIDENCE_THRESHOLDS["maximum_carrier_to_execution_ratio"]
                records.append({
                    "relation": relation,
                    "surface": surface,
                    "split": split,
                    "relational_execution_margin_interaction": execution,
                    "relational_carrier_margin_interaction": carrier,
                    "carrier_to_execution_ratio": ratio,
                    "passed": passed,
                })
    pass_count = sum(int(row["passed"]) for row in records)
    return {
        "records": records,
        "passing_cells": pass_count,
        "cell_count": len(records),
        "passed": pass_count >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_anchor_cells"],
    }


def panel_convergence(data: dict[str, Any]) -> dict[str, Any]:
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    fractions = np.asarray(protocol.DEPTH_ANCHORS)
    early = np.where((fractions >= 0.25) & (fractions <= 0.42))[0]
    late = np.where(fractions >= 0.83)[0]
    records = []
    for relation_index, relation in enumerate(protocol.RELATIONS):
        for surface_index, surface in enumerate(protocol.SURFACES):
            for split_index, split in enumerate(protocol.SPLITS):
                execution_curve = data["panel_alignment"][relation_index, surface_index, split_index, 0, role]
                carrier_curve = data["panel_alignment"][relation_index, surface_index, split_index, 1, role]
                execution_early = float(np.nanmean(execution_curve[early]))
                execution_late = float(np.nanmean(execution_curve[late]))
                carrier_early = float(np.nanmean(carrier_curve[early]))
                carrier_late = float(np.nanmean(carrier_curve[late]))
                execution_rise = execution_late - execution_early
                carrier_rise = carrier_late - carrier_early
                advantage = execution_rise - carrier_rise
                passed = (
                    execution_rise >= protocol.EVIDENCE_THRESHOLDS["minimum_panel_convergence_rise"]
                    and advantage >= protocol.EVIDENCE_THRESHOLDS["minimum_panel_convergence_advantage"]
                )
                records.append({
                    "relation": relation,
                    "surface": surface,
                    "split": split,
                    "execution_early": execution_early,
                    "execution_late": execution_late,
                    "execution_rise": execution_rise,
                    "carrier_rise": carrier_rise,
                    "rise_advantage": advantage,
                    "passed": passed,
                })
    pass_count = sum(int(row["passed"]) for row in records)
    return {
        "records": records,
        "passing_cells": pass_count,
        "cell_count": len(records),
        "passed": pass_count >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_anchor_cells"],
    }


def cross_language(data: dict[str, Any]) -> dict[str, Any]:
    field_index = {name: index for index, name in enumerate(protocol.FIELDS)}
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    direction_records = []
    for source_surface, target_surface in ((0, 1), (1, 0)):
        split_records = []
        for split_index, split in enumerate(protocol.SPLITS):
            source_content = [
                signature_for(data, relation, source_surface, split_index, field_index["relational_execution"], role)
                for relation in range(len(protocol.RELATIONS))
            ]
            target_content = [
                signature_for(data, relation, target_surface, split_index, field_index["relational_execution"], role)
                for relation in range(len(protocol.RELATIONS))
            ]
            source_carrier = [
                signature_for(data, relation, source_surface, split_index, field_index["relational_carrier"], role)
                for relation in range(len(protocol.RELATIONS))
            ]
            target_carrier = [
                signature_for(data, relation, target_surface, split_index, field_index["relational_carrier"], role)
                for relation in range(len(protocol.RELATIONS))
            ]
            matrix = np.asarray([[cosine(left, right) for right in target_content] for left in source_content])
            carrier_diagonal = [cosine(source_carrier[index], target_carrier[index]) for index in range(len(protocol.RELATIONS))]
            passing = 0
            identities = []
            for relation_index, relation in enumerate(protocol.RELATIONS):
                predicted = int(np.argmax(matrix[relation_index]))
                advantage = float(matrix[relation_index, relation_index] - carrier_diagonal[relation_index])
                passed = predicted == relation_index and advantage >= protocol.EVIDENCE_THRESHOLDS["minimum_content_over_carrier_advantage"]
                passing += int(passed)
                identities.append({
                    "relation": relation,
                    "predicted_relation": protocol.RELATIONS[predicted],
                    "content_cosine": float(matrix[relation_index, relation_index]),
                    "carrier_cosine": float(carrier_diagonal[relation_index]),
                    "content_advantage": advantage,
                    "passed": passed,
                })
            split_records.append({
                "split": split,
                "passing_relations": passing,
                "passed": passing >= protocol.EVIDENCE_THRESHOLDS["minimum_heldout_relations"],
                "identities": identities,
            })
        direction_records.append({
            "source_surface": protocol.SURFACES[source_surface],
            "target_surface": protocol.SURFACES[target_surface],
            "splits": split_records,
            "passed": all(row["passed"] for row in split_records),
        })
    passing_directions = sum(int(row["passed"]) for row in direction_records)
    return {
        "directions": direction_records,
        "passing_directions": passing_directions,
        "passed": passing_directions >= protocol.EVIDENCE_THRESHOLDS["minimum_cross_language_directions"],
    }


def phase_profile(data: dict[str, Any], field_name: str) -> np.ndarray:
    field = protocol.FIELDS.index(field_name)
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    amplitude = np.nanmean(data["amplitude"][:, :, :, field, role, :], axis=(0, 1, 2))
    margin = np.nanmean(np.abs(data["local_margin"][:, :, :, field, role, :]), axis=(0, 1, 2))
    return np.concatenate((unit(amplitude), unit(margin))) / math.sqrt(2.0)


def cross_model_profiles(all_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        content = cosine(phase_profile(all_data[left], "relational_execution"), phase_profile(all_data[right], "relational_execution"))
        carrier = cosine(phase_profile(all_data[left], "relational_carrier"), phase_profile(all_data[right], "relational_carrier"))
        advantage = content - carrier
        records.append({
            "model_pair": [left, right],
            "content_profile_cosine": content,
            "carrier_profile_cosine": carrier,
            "content_advantage": advantage,
            "descriptive_threshold_passed": content >= protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_profile_cosine"] and advantage >= protocol.EVIDENCE_THRESHOLDS["minimum_content_over_carrier_advantage"],
        })
    return {"records": records, "descriptive_only": True}


def physical_hotspots(data: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    summary = data["summary"]
    field = protocol.FIELDS.index("relational_execution")
    dynamic_roles = [protocol.CAPTURE_ROLES.index(role) for role in protocol.DYNAMIC_ROLES]
    profile = np.nanmean(data["physical"][:, :, :, :, dynamic_roles, field], axis=(0, 1, 2))
    candidates = []
    for event_index, event in enumerate(summary["events"]):
        for local_role, role_number in enumerate(dynamic_roles):
            value = float(profile[event_index, local_role])
            if math.isfinite(value):
                candidates.append({
                    "event_id": event["event_id"],
                    "component": event["component"],
                    "depth": event["depth"],
                    "relative_depth": event["relative_depth"],
                    "role": protocol.CAPTURE_ROLES[role_number],
                    "mean_relative_magnitude": value,
                })
    return sorted(candidates, key=lambda row: row["mean_relative_magnitude"], reverse=True)[:limit]


def model_instrument_gate(data: dict[str, Any]) -> dict[str, Any]:
    summary = data["summary"]
    passed = (
        summary["hidden_finite_fraction_lower_bound"] >= protocol.EVIDENCE_THRESHOLDS["minimum_hidden_finite_fraction"]
        and summary["local_readout_finite_fraction"] >= protocol.EVIDENCE_THRESHOLDS["minimum_hidden_finite_fraction"]
        and summary["identity_maximum"] <= protocol.EVIDENCE_THRESHOLDS["pre_task_tolerance"]
        and summary["pre_task_control_execution_maximum"] <= protocol.EVIDENCE_THRESHOLDS["pre_task_tolerance"]
        and summary["local_readout_maximum_native_margin_error"] <= protocol.EVIDENCE_THRESHOLDS["local_readout_tolerance"]
    )
    return {
        "behavior_formal": summary["behavior_formal"],
        "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
        "local_readout_finite_fraction": summary["local_readout_finite_fraction"],
        "identity_maximum": summary["identity_maximum"],
        "pre_task_maximum": summary["pre_task_control_execution_maximum"],
        "local_readout_maximum_native_margin_error": summary["local_readout_maximum_native_margin_error"],
        "instrument_passed": passed,
        "formal_instrument_passed": passed and bool(summary["behavior_formal"]),
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    all_data = {model: load_model(model) for model in protocol.MODELS}
    model_results = {}
    for model_name, data in all_data.items():
        model_results[model_name] = {
            "instrument": model_instrument_gate(data),
            "split_repeat": split_repeat(data),
            "heldout_relation_prediction": heldout_relation_prediction(data),
            "behavior_anchor": behavior_anchor(data),
            "panel_convergence": panel_convergence(data),
            "cross_language": cross_language(data),
            "physical_hotspots": physical_hotspots(data),
            "summary_digest": data["summary"]["summary_digest"],
        }

    formal_models = [
        model for model in protocol.MODELS
        if behavior["models"][model]["model_behavior_passed"]
    ]
    p3_models = [model for model in formal_models if model_results[model]["instrument"]["formal_instrument_passed"]]
    p4_models = [model for model in formal_models if model_results[model]["split_repeat"]["passed"]]
    p5_models = [model for model in formal_models if model_results[model]["heldout_relation_prediction"]["passed"]]
    p6_models = [model for model in formal_models if model_results[model]["behavior_anchor"]["passed"]]
    p7_models = [model for model in formal_models if model_results[model]["panel_convergence"]["passed"]]
    p8_models = [model for model in formal_models if model_results[model]["cross_language"]["passed"]]
    required_models = protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"]
    gates = {
        "P1_protocol": bool(audit["all_checks_passed"]),
        "P2_behavior": bool(behavior["hidden_scan_authorized"]),
        "P3_instrument": len(p3_models) >= required_models,
        "P4_split_transition_repeat": len(p4_models) >= required_models,
        "P5_heldout_relation_transition": len(p5_models) >= required_models,
        "P6_behavior_anchored_execution": len(p6_models) >= required_models,
        "P7_early_late_panel_convergence": len(p7_models) >= required_models,
        "P8_cross_language_transition": len(p8_models) >= required_models,
        "P9_causal_localization_authorized": False,
    }
    independent_replication = all(gates[key] for key in (
        "P5_heldout_relation_transition",
        "P6_behavior_anchored_execution",
        "P7_early_late_panel_convergence",
        "P8_cross_language_transition",
    ))
    cross_model = cross_model_profiles(all_data)
    conclusion = {
        "fixed_vector_hypothesis_status": "Phase1096 rejected the tested fixed-coordinate execution form; capability was not rejected.",
        "transition_hypothesis_status": (
            "prospective transition evidence warrants independent replication"
            if independent_replication
            else "conditional transition structure is descriptive or partial; no nearby protocol variants are authorized"
        ),
        "mathematics_status": "Basic exact differences, per-item norms, depth Gram relations, cosines, and local output margins remain sufficient. No new mathematical theory is warranted.",
        "causal_status": "No Phase1097 result is a causal edge or a neuron-level mechanism.",
    }
    result = {
        "schema_version": "phase1097_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_authorization_digest": behavior["authorization_digest"],
        "formal_models": formal_models,
        "instrument_models": p3_models,
        "split_repeat_models": p4_models,
        "heldout_transition_models": p5_models,
        "behavior_anchor_models": p6_models,
        "panel_convergence_models": p7_models,
        "cross_language_models": p8_models,
        "models": model_results,
        "cross_model_profiles": cross_model,
        "gates": gates,
        "independent_replication_authorized": independent_replication,
        "causal_localization_authorized": False,
        "automatic_next_required": independent_replication,
        "automatic_next_route": "Phase1098 independent conditional-transition replication" if independent_replication else "",
        "conclusion": conclusion,
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print({
        "phase": protocol.PHASE,
        "formal_models": formal_models,
        "gates": gates,
        "automatic_next_required": independent_replication,
        "summary_digest": result["summary_digest"],
    })


if __name__ == "__main__":
    main()
