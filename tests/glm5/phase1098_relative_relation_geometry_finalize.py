#!/usr/bin/env python3
"""Finalize prospective Phase1098 relative-relation geometry gates."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1098_relative_relation_geometry_protocol as protocol


EPSILON = 1e-12
DEPTH_FRACTIONS = (0.25, 1.0 / 3.0, 5.0 / 12.0, 0.5, 7.0 / 12.0, 2.0 / 3.0, 0.75)
COMPONENTS = ("residual", "attention_output", "mlp_output")
RELATION_PERMUTATIONS = tuple(itertools.permutations(range(len(protocol.RELATIONS))))
IDENTITY_PERMUTATION = tuple(range(len(protocol.RELATIONS)))


def unit(value: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(value), value, 0.0).astype(np.float64, copy=False).reshape(-1)
    norm = float(np.linalg.norm(clean))
    return clean / norm if norm > EPSILON else np.zeros_like(clean)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = unit(left), unit(right)
    if np.linalg.norm(a) <= EPSILON or np.linalg.norm(b) <= EPSILON:
        return 0.0
    return float(a @ b)


def finite_mean(value: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(value)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, value, 0.0), axis=axis)
    result = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def graph_vector(graph: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(len(protocol.RELATIONS), k=1)
    return graph[..., upper[0], upper[1]].reshape(-1)


def permute_graph(graph: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    index = np.asarray(permutation, dtype=np.int64)
    return graph[..., index, :][..., :, index]


def permutation_scores(source: np.ndarray, target: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    return [
        (permutation, cosine(graph_vector(source), graph_vector(permute_graph(target, permutation))))
        for permutation in RELATION_PERMUTATIONS
    ]


def load_model(model_name: str) -> dict[str, Any]:
    root = protocol.OUT_ROOT / "atlas" / model_name
    summary = protocol.read_json(root / "summary.json")
    index = protocol.read_jsonl(root / "superunit_index.jsonl")
    with np.load(root / "relative_relation_geometry.npz") as data:
        arrays = {key: data[key] for key in data.files}
    surface_index = {value: number for number, value in enumerate(protocol.SURFACES)}
    split_index = {value: number for number, value in enumerate(protocol.SPLITS)}
    aggregate_gram = np.full(
        (len(protocol.SURFACES), len(protocol.SPLITS)) + arrays["relation_gram"].shape[1:],
        np.nan, dtype=np.float64,
    )
    aggregate_shared = np.full(
        (len(protocol.SURFACES), len(protocol.SPLITS)) + arrays["shared_energy"].shape[1:],
        np.nan, dtype=np.float64,
    )
    aggregate_differential = np.full_like(aggregate_shared, np.nan)
    cell_counts = np.zeros((len(protocol.SURFACES), len(protocol.SPLITS)), dtype=np.int32)
    for surface, si in surface_index.items():
        for split, qi in split_index.items():
            selected = [row["superunit_index"] for row in index if row["surface"] == surface and row["split"] == split]
            cell_counts[si, qi] = len(selected)
            aggregate_gram[si, qi] = finite_mean(arrays["relation_gram"][selected], axis=0)
            aggregate_shared[si, qi] = finite_mean(arrays["shared_energy"][selected], axis=0)
            aggregate_differential[si, qi] = finite_mean(arrays["differential_energy"][selected], axis=0)
    event_lookup = {(row["component"], int(row["depth"])): int(row["event_index"]) for row in summary["events"]}
    selected_events = []
    for component in COMPONENTS:
        candidates = [row for row in summary["events"] if row["component"] == component]
        for fraction in DEPTH_FRACTIONS:
            event = min(candidates, key=lambda row: (abs(float(row["relative_depth"]) - fraction), int(row["depth"])))
            selected_events.append(int(event["event_index"]))
    return {
        "summary": summary,
        "index": index,
        "raw": arrays,
        "gram": aggregate_gram,
        "shared": aggregate_shared,
        "differential": aggregate_differential,
        "cell_counts": cell_counts,
        "selected_events": selected_events,
        "event_lookup": event_lookup,
    }


def tensor_for(data: dict[str, Any], surface: int, split: int, field: str, role: str) -> np.ndarray:
    field_index = protocol.FIELDS.index(field)
    role_index = protocol.CAPTURE_ROLES.index(role)
    return data["gram"][surface, split, data["selected_events"], field_index, role_index]


def evaluate_match(
    source_data: dict[str, Any],
    target_data: dict[str, Any],
    source_surface: int,
    source_split: int,
    target_surface: int,
    target_split: int,
    role: str = "answer_boundary",
) -> dict[str, Any]:
    source = tensor_for(source_data, source_surface, source_split, protocol.PRIMARY_FIELD, role)
    target = tensor_for(target_data, target_surface, target_split, protocol.PRIMARY_FIELD, role)
    scores = sorted(permutation_scores(source, target), key=lambda row: row[1], reverse=True)
    identity_score = next(score for permutation, score in scores if permutation == IDENTITY_PERMUTATION)
    best_nonidentity = max(score for permutation, score in scores if permutation != IDENTITY_PERMUTATION)
    identity_rank = 1 + next(index for index, row in enumerate(scores) if row[0] == IDENTITY_PERMUTATION)
    control_records = []
    for field in protocol.CONTROL_FIELDS:
        control = tensor_for(target_data, target_surface, target_split, field, role)
        best_control = max(score for _, score in permutation_scores(source, control))
        control_records.append({"field": field, "best_permuted_score": best_control})
    maximum_control = max(row["best_permuted_score"] for row in control_records)
    permutation_margin = identity_score - best_nonidentity
    field_advantage = identity_score - maximum_control
    thresholds = protocol.EVIDENCE_THRESHOLDS
    passed = (
        identity_score >= thresholds["minimum_geometry_cosine"]
        and identity_rank == 1
        and permutation_margin >= thresholds["minimum_permutation_margin"]
        and field_advantage >= thresholds["minimum_field_specificity_advantage"]
    )
    return {
        "source_surface": protocol.SURFACES[source_surface],
        "source_split": protocol.SPLITS[source_split],
        "target_surface": protocol.SURFACES[target_surface],
        "target_split": protocol.SPLITS[target_split],
        "role": role,
        "identity_score": identity_score,
        "identity_rank": identity_rank,
        "best_nonidentity_score": best_nonidentity,
        "permutation_margin": permutation_margin,
        "best_permutation": list(scores[0][0]),
        "maximum_control_score": maximum_control,
        "field_specificity_advantage": field_advantage,
        "controls": control_records,
        "passed": passed,
    }


def split_gate(data: dict[str, Any]) -> dict[str, Any]:
    records = []
    for surface in range(len(protocol.SURFACES)):
        records.append(evaluate_match(data, data, surface, 0, surface, 1))
        records.append(evaluate_match(data, data, surface, 1, surface, 0))
    return {"records": records, "passing_records": sum(row["passed"] for row in records), "passed": all(row["passed"] for row in records)}


def cross_language_gate(data: dict[str, Any]) -> dict[str, Any]:
    records = []
    for split in range(len(protocol.SPLITS)):
        records.append(evaluate_match(data, data, 0, split, 1, split))
        records.append(evaluate_match(data, data, 1, split, 0, split))
    return {"records": records, "passing_records": sum(row["passed"] for row in records), "passed": all(row["passed"] for row in records)}


def energy_summary(data: dict[str, Any]) -> dict[str, Any]:
    field = protocol.FIELDS.index(protocol.PRIMARY_FIELD)
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    selected = data["selected_events"]
    shared = data["shared"][:, :, selected, field, role]
    differential = data["differential"][:, :, selected, field, role]
    closure = np.abs(shared + differential - 1.0)
    return {
        "shared_median": float(np.nanmedian(shared)),
        "shared_minimum": float(np.nanmin(shared)),
        "shared_maximum": float(np.nanmax(shared)),
        "differential_median": float(np.nanmedian(differential)),
        "differential_minimum": float(np.nanmin(differential)),
        "differential_maximum": float(np.nanmax(differential)),
        "maximum_energy_closure_error": float(np.nanmax(closure)),
        "differential_above_floor": bool(np.nanmedian(differential) >= protocol.EVIDENCE_THRESHOLDS["minimum_differential_energy"]),
    }


def physical_hotspots(data: dict[str, Any]) -> list[dict[str, Any]]:
    field = protocol.FIELDS.index(protocol.PRIMARY_FIELD)
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    control_indices = [protocol.FIELDS.index(value) for value in protocol.CONTROL_FIELDS]
    rows = []
    for event in data["summary"]["events"]:
        event_index = int(event["event_index"])
        surface_rows = []
        for surface in range(len(protocol.SURFACES)):
            source = data["gram"][surface, 0, event_index, field, role][None, ...]
            target = data["gram"][surface, 1, event_index, field, role][None, ...]
            scores = sorted(permutation_scores(source, target), key=lambda row: row[1], reverse=True)
            identity = next(score for permutation, score in scores if permutation == IDENTITY_PERMUTATION)
            nonidentity = max(score for permutation, score in scores if permutation != IDENTITY_PERMUTATION)
            controls = []
            for control in control_indices:
                control_graph = data["gram"][surface, 1, event_index, control, role][None, ...]
                controls.append(max(score for _, score in permutation_scores(source, control_graph)))
            surface_rows.append({
                "surface": protocol.SURFACES[surface],
                "identity_score": identity,
                "permutation_margin": identity - nonidentity,
                "field_advantage": identity - max(controls),
            })
        rows.append({
            "event_index": event_index,
            "event_id": event["event_id"],
            "component": event["component"],
            "depth": event["depth"],
            "relative_depth": event["relative_depth"],
            "mean_identity_score": float(np.mean([row["identity_score"] for row in surface_rows])),
            "minimum_permutation_margin": float(min(row["permutation_margin"] for row in surface_rows)),
            "minimum_field_advantage": float(min(row["field_advantage"] for row in surface_rows)),
            "surfaces": surface_rows,
        })
    rows.sort(key=lambda row: (row["minimum_field_advantage"], row["minimum_permutation_margin"], row["mean_identity_score"]), reverse=True)
    return rows[:20]


def cross_model_gate(all_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pair_records = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        cells = []
        for surface in range(len(protocol.SURFACES)):
            for split in range(len(protocol.SPLITS)):
                forward = evaluate_match(all_data[left], all_data[right], surface, split, surface, split)
                reverse = evaluate_match(all_data[right], all_data[left], surface, split, surface, split)
                cells.append({
                    "surface": protocol.SURFACES[surface],
                    "split": protocol.SPLITS[split],
                    "forward": forward,
                    "reverse": reverse,
                    "passed": forward["passed"] and reverse["passed"],
                })
        passing = sum(row["passed"] for row in cells)
        pair_records.append({
            "left": left,
            "right": right,
            "cells": cells,
            "passing_cells": passing,
            "passed": passing >= protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_cells"],
        })
    passing_pairs = sum(row["passed"] for row in pair_records)
    return {
        "pairs": pair_records,
        "passing_pairs": passing_pairs,
        "passed": passing_pairs >= protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_pairs"],
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    all_data = {model: load_model(model) for model in protocol.MODELS}
    model_results = {}
    for model, data in all_data.items():
        summary = data["summary"]
        instrument_passed = (
            summary["hidden_finite_fraction"] >= protocol.EVIDENCE_THRESHOLDS["minimum_hidden_finite_fraction"]
            and summary["identity_maximum_error"] <= protocol.EVIDENCE_THRESHOLDS["pre_task_tolerance"]
            and summary["pre_task_maximum_error"] <= protocol.EVIDENCE_THRESHOLDS["pre_task_tolerance"]
            and summary["primary_signature_excludes_output_interaction"]
        )
        model_results[model] = {
            "behavior_formal": bool(authorization["models"][model]["model_behavior_passed"]),
            "instrument_passed": instrument_passed,
            "split_gate": split_gate(data),
            "cross_language_gate": cross_language_gate(data),
            "energy": energy_summary(data),
            "physical_hotspots": physical_hotspots(data),
            "summary_digest": summary["summary_digest"],
        }
    formal_models = [model for model in protocol.MODELS if model_results[model]["behavior_formal"]]
    p3_models = [model for model in formal_models if model_results[model]["instrument_passed"]]
    p4_models = [model for model in p3_models if model_results[model]["split_gate"]["passed"]]
    p5_models = [model for model in p4_models if model_results[model]["cross_language_gate"]["passed"]]
    cross_model = cross_model_gate(all_data)
    gates = {
        "P1_protocol_and_source_audit": bool(audit["all_checks_passed"]),
        "P2_behavior_base": len(formal_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P3_instrument_validity": len(p3_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P4_split_content_specific_relation_geometry": len(p4_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P5_cross_language_content_specific_relation_geometry": len(p5_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P6_cross_model_content_specific_relation_geometry": cross_model["passed"],
        "P8_primary_signature_excludes_output_margin": all(data["summary"]["primary_signature_excludes_output_interaction"] for data in all_data.values()),
    }
    automatic_next = all(gates[key] for key in (
        "P1_protocol_and_source_audit",
        "P2_behavior_base",
        "P3_instrument_validity",
        "P4_split_content_specific_relation_geometry",
        "P5_cross_language_content_specific_relation_geometry",
        "P6_cross_model_content_specific_relation_geometry",
        "P8_primary_signature_excludes_output_margin",
    ))
    block_audit = protocol.read_json(protocol.SOURCE_BLOCK_AUDIT)
    diagnostic_path = protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json"
    diagnostic = protocol.read_json(diagnostic_path) if diagnostic_path.exists() else {}
    result = {
        "schema_version": "phase1098_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "source_signature_block_audit": block_audit,
        "models": model_results,
        "formal_models": formal_models,
        "instrument_models": p3_models,
        "split_geometry_models": p4_models,
        "cross_language_models": p5_models,
        "cross_model": cross_model,
        "posthoc_failure_diagnostic": {
            "present": bool(diagnostic),
            "status": diagnostic.get("status"),
            "diagnostic_digest": diagnostic.get("diagnostic_digest"),
            "comparison_count": diagnostic.get("comparison_count"),
            "identity_rank_one_count": diagnostic.get("identity_rank_one_count"),
            "best_permutation_fixed_counts": diagnostic.get("best_permutation_fixed_counts"),
            "near_best_fixed_fraction_mean": diagnostic.get("near_best_fixed_fraction_mean"),
            "median_near_best_permutation_count": diagnostic.get("median_near_best_permutation_count"),
            "guardrail": diagnostic.get("guardrail"),
        },
        "gates": gates,
        "automatic_next_required": automatic_next,
        "automatic_next_decision": (
            "authorize_phase1099_minimal_causal_transport_map"
            if automatic_next else
            "stop_automatic_continuation; retain descriptive atlas and redesign only from observed failures"
        ),
        "theory_status": (
            "Phase1097 stability is now separated into a carrier-like amplitude/depth geometry and an output-margin-dominated content advantage. Phase1098 tests, without output margins, whether relation identity survives as a relative graph. No additive direct-sum, rotation group, fiber-bundle, or new mathematics claim is licensed."
        ),
        "mathematics_status": (
            "Signed factorial differences, Euclidean centering, Gram relations, exact 5! label permutations, and matched field controls are sufficient for this test."
        ),
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "formal_models": formal_models,
        "p3_models": p3_models,
        "p4_models": p4_models,
        "p5_models": p5_models,
        "cross_model_passing_pairs": cross_model["passing_pairs"],
        "gates": gates,
        "automatic_next_required": automatic_next,
        "summary_digest": result["summary_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
