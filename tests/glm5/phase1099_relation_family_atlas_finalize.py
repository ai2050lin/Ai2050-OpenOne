#!/usr/bin/env python3
"""Finalize the prospective Phase1099 held-out relation-family gates."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1099_relation_family_atlas_protocol as protocol


EPSILON = 1e-12
FAMILY_PERMUTATIONS = tuple(itertools.permutations(range(len(protocol.FAMILIES))))
IDENTITY_PERMUTATION = tuple(range(len(protocol.FAMILIES)))


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


def family_block(graph: np.ndarray, relation_split: str) -> np.ndarray:
    """Average a relation Gram into a 5x5 family block graph."""
    groups = []
    for family in protocol.FAMILIES:
        groups.append([
            index for index, relation in enumerate(protocol.RELATIONS)
            if protocol.RELATION_FAMILY[relation] == family
            and protocol.RELATION_SPLIT[relation] == relation_split
        ])
    output = np.full(graph.shape[:-2] + (len(groups), len(groups)), np.nan, dtype=np.float64)
    for left, left_indices in enumerate(groups):
        for right, right_indices in enumerate(groups):
            values = graph[..., left_indices, :][..., :, right_indices]
            values = values.reshape(values.shape[:-2] + (-1,))
            if left == right:
                mask = (~np.eye(len(left_indices), dtype=bool)).reshape(-1)
                values = values[..., mask]
            output[..., left, right] = finite_mean(values, axis=-1)
    return output


def block_vector(block: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(len(protocol.FAMILIES), k=0)
    return block[..., upper[0], upper[1]].reshape(-1)


def permute_block(block: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    index = np.asarray(permutation, dtype=np.int64)
    return block[..., index, :][..., :, index]


def permutation_scores(source: np.ndarray, target: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    return [
        (permutation, cosine(block_vector(source), block_vector(permute_block(target, permutation))))
        for permutation in FAMILY_PERMUTATIONS
    ]


def load_model(model_name: str) -> dict[str, Any]:
    root = protocol.OUT_ROOT / "atlas" / model_name
    summary = protocol.read_json(root / "summary.json")
    index = protocol.read_jsonl(root / "superunit_index.jsonl")
    with np.load(root / "relative_relation_geometry.npz") as handle:
        arrays = {key: handle[key] for key in handle.files}
    aggregate_shape = (len(protocol.SURFACES), len(protocol.TEMPLATES)) + arrays["relation_gram"].shape[1:]
    aggregate_gram = np.full(aggregate_shape, np.nan, dtype=np.float64)
    aggregate_shared = np.full(
        (len(protocol.SURFACES), len(protocol.TEMPLATES)) + arrays["shared_energy"].shape[1:],
        np.nan,
        dtype=np.float64,
    )
    aggregate_differential = np.full_like(aggregate_shared, np.nan)
    counts = np.zeros((len(protocol.SURFACES), len(protocol.TEMPLATES)), dtype=np.int32)
    for surface_index, surface in enumerate(protocol.SURFACES):
        for template in protocol.TEMPLATES:
            selected = [
                int(row["superunit_index"])
                for row in index
                if row["surface"] == surface and int(row["template"]) == template
            ]
            counts[surface_index, template] = len(selected)
            aggregate_gram[surface_index, template] = finite_mean(arrays["relation_gram"][selected], axis=0)
            aggregate_shared[surface_index, template] = finite_mean(arrays["shared_energy"][selected], axis=0)
            aggregate_differential[surface_index, template] = finite_mean(arrays["differential_energy"][selected], axis=0)
    return {
        "summary": summary,
        "index": index,
        "raw": arrays,
        "gram": aggregate_gram,
        "shared": aggregate_shared,
        "differential": aggregate_differential,
        "counts": counts,
    }


def tensor_for(
    data: dict[str, Any],
    surface: int,
    template: int,
    relation_split: str,
    field: str,
    role: str = "answer_boundary",
) -> np.ndarray:
    field_index = protocol.FIELDS.index(field)
    role_index = protocol.CAPTURE_ROLES.index(role)
    graph = data["gram"][surface, template, :, field_index, role_index]
    return family_block(graph, relation_split)


def evaluate_match(
    source_data: dict[str, Any],
    target_data: dict[str, Any],
    source_surface: int,
    source_template: int,
    source_relation_split: str,
    target_surface: int,
    target_template: int,
    target_relation_split: str,
    role: str = "answer_boundary",
    event_index: int | None = None,
) -> dict[str, Any]:
    source = tensor_for(
        source_data, source_surface, source_template, source_relation_split,
        protocol.PRIMARY_FIELD, role,
    )
    target = tensor_for(
        target_data, target_surface, target_template, target_relation_split,
        protocol.PRIMARY_FIELD, role,
    )
    if event_index is not None:
        source = source[event_index:event_index + 1]
        target = target[event_index:event_index + 1]
    scores = sorted(permutation_scores(source, target), key=lambda row: row[1], reverse=True)
    identity_score = next(score for permutation, score in scores if permutation == IDENTITY_PERMUTATION)
    best_nonidentity = max(score for permutation, score in scores if permutation != IDENTITY_PERMUTATION)
    identity_rank = 1 + next(index for index, row in enumerate(scores) if row[0] == IDENTITY_PERMUTATION)
    control_records = []
    for field in protocol.CONTROL_FIELDS:
        control = tensor_for(
            target_data, target_surface, target_template, target_relation_split,
            field, role,
        )
        if event_index is not None:
            control = control[event_index:event_index + 1]
        best_control = max(score for _, score in permutation_scores(source, control))
        control_records.append({"field": field, "best_permuted_score": best_control})
    maximum_control = max(row["best_permuted_score"] for row in control_records)
    permutation_margin = identity_score - best_nonidentity
    field_advantage = identity_score - maximum_control
    threshold = protocol.EVIDENCE_THRESHOLDS
    passed = (
        identity_score >= threshold["minimum_family_geometry_cosine"]
        and identity_rank == 1
        and permutation_margin >= threshold["minimum_family_permutation_margin"]
        and field_advantage >= threshold["minimum_field_specificity_advantage"]
    )
    return {
        "source_surface": protocol.SURFACES[source_surface],
        "source_template": source_template,
        "source_relation_split": source_relation_split,
        "target_surface": protocol.SURFACES[target_surface],
        "target_template": target_template,
        "target_relation_split": target_relation_split,
        "role": role,
        "event_index": event_index,
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


TEMPLATE_RELATION_PAIRS = (
    (0, "discovery", 2, "confirmation"),
    (2, "confirmation", 0, "discovery"),
    (1, "discovery", 3, "confirmation"),
    (3, "confirmation", 1, "discovery"),
)


def heldout_split_gate(data: dict[str, Any]) -> dict[str, Any]:
    records = []
    by_surface = {}
    for surface in range(len(protocol.SURFACES)):
        surface_records = [
            evaluate_match(data, data, surface, source_template, source_split, surface, target_template, target_split)
            for source_template, source_split, target_template, target_split in TEMPLATE_RELATION_PAIRS
        ]
        records.extend(surface_records)
        by_surface[protocol.SURFACES[surface]] = sum(row["passed"] for row in surface_records)
    passing = sum(row["passed"] for row in records)
    return {
        "records": records,
        "passing_records": passing,
        "passing_by_surface": by_surface,
        "passed": passing >= protocol.EVIDENCE_THRESHOLDS["minimum_split_records"] and all(value >= 3 for value in by_surface.values()),
    }


def cross_language_gate(data: dict[str, Any]) -> dict[str, Any]:
    records = []
    for source_surface, target_surface in ((0, 1), (1, 0)):
        for source_template, source_split, target_template, target_split in TEMPLATE_RELATION_PAIRS:
            records.append(evaluate_match(
                data, data,
                source_surface, source_template, source_split,
                target_surface, target_template, target_split,
            ))
    passing = sum(row["passed"] for row in records)
    required = math.ceil(len(records) * protocol.EVIDENCE_THRESHOLDS["minimum_cross_language_fraction"])
    return {"records": records, "passing_records": passing, "required_records": required, "passed": passing >= required}


def cross_model_gate(all_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pair_records = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        cells = []
        for surface in range(len(protocol.SURFACES)):
            for template in protocol.TEMPLATES:
                for relation_split in protocol.RELATION_SPLITS:
                    forward = evaluate_match(
                        all_data[left], all_data[right],
                        surface, template, relation_split,
                        surface, template, relation_split,
                    )
                    reverse = evaluate_match(
                        all_data[right], all_data[left],
                        surface, template, relation_split,
                        surface, template, relation_split,
                    )
                    cells.append({
                        "surface": protocol.SURFACES[surface],
                        "template": template,
                        "relation_split": relation_split,
                        "forward": forward,
                        "reverse": reverse,
                        "passed": forward["passed"] and reverse["passed"],
                    })
        passing = sum(row["passed"] for row in cells)
        required = math.ceil(len(cells) * protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_fraction"])
        pair_records.append({
            "left": left,
            "right": right,
            "cells": cells,
            "passing_cells": passing,
            "required_cells": required,
            "passed": passing >= required,
        })
    passing_pairs = sum(row["passed"] for row in pair_records)
    return {
        "pairs": pair_records,
        "passing_pairs": passing_pairs,
        "passed": passing_pairs >= protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_pairs"],
    }


def energy_summary(data: dict[str, Any]) -> dict[str, float]:
    field = protocol.FIELDS.index(protocol.PRIMARY_FIELD)
    role = protocol.CAPTURE_ROLES.index("answer_boundary")
    event_indices = [
        int(row["event_index"])
        for row in data["summary"]["events"]
        if 0.2 <= float(row["relative_depth"]) <= 0.8
    ]
    shared = data["shared"][:, :, event_indices, field, role]
    differential = data["differential"][:, :, event_indices, field, role]
    return {
        "shared_median": float(np.nanmedian(shared)),
        "differential_median": float(np.nanmedian(differential)),
        "maximum_energy_closure_error": float(np.nanmax(np.abs(shared + differential - 1.0))),
    }


def cohesion_summary(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    result = {}
    for field in (protocol.PRIMARY_FIELD,) + protocol.CONTROL_FIELDS:
        values = []
        for surface in range(len(protocol.SURFACES)):
            for template in protocol.TEMPLATES:
                for relation_split in protocol.RELATION_SPLITS:
                    block = tensor_for(data, surface, template, relation_split, field)
                    diagonal = np.diagonal(block, axis1=-2, axis2=-1)
                    off_mask = ~np.eye(len(protocol.FAMILIES), dtype=bool)
                    off = block[..., off_mask]
                    values.append(float(np.nanmean(diagonal) - np.nanmean(off)))
        result[field] = {
            "mean_within_minus_between": float(np.mean(values)),
            "minimum_within_minus_between": float(np.min(values)),
            "maximum_within_minus_between": float(np.max(values)),
        }
    return result


def physical_hotspots(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for event in data["summary"]["events"]:
        event_index = int(event["event_index"])
        records = []
        for surface in range(len(protocol.SURFACES)):
            records.append(evaluate_match(
                data, data, surface, 0, "discovery", surface, 2, "confirmation",
                event_index=event_index,
            ))
            records.append(evaluate_match(
                data, data, surface, 1, "discovery", surface, 3, "confirmation",
                event_index=event_index,
            ))
        rows.append({
            "event_index": event_index,
            "event_id": event["event_id"],
            "component": event["component"],
            "depth": event["depth"],
            "relative_depth": event["relative_depth"],
            "mean_identity_score": float(np.mean([row["identity_score"] for row in records])),
            "minimum_permutation_margin": float(min(row["permutation_margin"] for row in records)),
            "minimum_field_advantage": float(min(row["field_specificity_advantage"] for row in records)),
            "passing_records": sum(row["passed"] for row in records),
        })
    rows.sort(key=lambda row: (row["passing_records"], row["minimum_field_advantage"], row["minimum_permutation_margin"]), reverse=True)
    return rows[:20]


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    all_data = {model: load_model(model) for model in protocol.MODELS}
    model_results = {}
    for model_name, data in all_data.items():
        summary = data["summary"]
        instrument_passed = (
            summary["hidden_finite_fraction"] >= protocol.EVIDENCE_THRESHOLDS["minimum_hidden_finite_fraction"]
            and summary["identity_maximum_error"] <= protocol.EVIDENCE_THRESHOLDS["pre_task_tolerance"]
            and summary["pre_task_maximum_error"] <= protocol.EVIDENCE_THRESHOLDS["pre_task_tolerance"]
            and summary["primary_signature_excludes_output_interaction"]
        )
        model_results[model_name] = {
            "behavior_formal": bool(authorization["models"][model_name]["model_behavior_passed"]),
            "instrument_passed": instrument_passed,
            "heldout_split_gate": heldout_split_gate(data),
            "cross_language_gate": cross_language_gate(data),
            "energy": energy_summary(data),
            "cohesion": cohesion_summary(data),
            "physical_hotspots": physical_hotspots(data),
            "summary_digest": summary["summary_digest"],
        }
    formal_models = [name for name in protocol.MODELS if model_results[name]["behavior_formal"]]
    instrument_models = [name for name in formal_models if model_results[name]["instrument_passed"]]
    split_models = [name for name in instrument_models if model_results[name]["heldout_split_gate"]["passed"]]
    language_models = [name for name in split_models if model_results[name]["cross_language_gate"]["passed"]]
    cross_model = cross_model_gate(all_data)
    gates = {
        "P1_protocol_and_source_audit": bool(audit["all_checks_passed"]),
        "P2_family_balanced_behavior": len(formal_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P3_instrument_validity": len(instrument_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P4_relation_heldout_family_geometry": len(split_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P5_cross_language_family_geometry": len(language_models) >= protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"],
        "P6_cross_model_family_geometry": cross_model["passed"],
        "P7_primary_signature_excludes_output_margin": all(data["summary"]["primary_signature_excludes_output_interaction"] for data in all_data.values()),
    }
    automatic_next = all(gates.values())
    result = {
        "schema_version": "phase1099_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "models": model_results,
        "formal_models": formal_models,
        "instrument_models": instrument_models,
        "relation_heldout_family_models": split_models,
        "cross_language_family_models": language_models,
        "cross_model": cross_model,
        "gates": gates,
        "automatic_next_required": automatic_next,
        "automatic_next_decision": (
            "authorize_independent_family_selective_causal_test"
            if automatic_next else
            "stop_automatic_continuation; retain descriptive family atlas and diagnose only from frozen outputs"
        ),
        "theory_status": "This phase tests a prospective family-level invariant rather than assuming one. A stable task shell, family graph, or hotspot is not a causal processor and does not establish an additive decomposition.",
        "mathematics_status": "Signed factorial differences, relation centering, block averaging, exact 5! family permutations, and matched field controls are sufficient for the registered question.",
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print({"phase": protocol.PHASE, "gates": gates, "automatic_next_required": automatic_next, "summary_digest": result["summary_digest"]})


if __name__ == "__main__":
    main()
