#!/usr/bin/env python3
"""Posthoc, non-upgrading diagnostics for Phase1098 gate failures."""

from __future__ import annotations

import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1098_relative_relation_geometry_finalize as finalize
import phase1098_relative_relation_geometry_protocol as protocol


NEAR_BEST_TOLERANCE = 0.02
DEPTH_BINS = {
    "early": (0.0, 0.25),
    "middle": (0.25, 0.75),
    "late": (0.75, 1.000001),
}


def tensor_custom(
    data: dict[str, Any], surface: int, split: int, field: str, role: str,
    component: str, lower: float, upper: float,
) -> np.ndarray:
    events = [
        int(row["event_index"])
        for row in data["summary"]["events"]
        if row["component"] == component and lower <= float(row["relative_depth"]) <= upper
    ]
    field_index = protocol.FIELDS.index(field)
    role_index = protocol.CAPTURE_ROLES.index(role)
    return data["gram"][surface, split, events, field_index, role_index]


def evaluate_custom(
    source_data: dict[str, Any], target_data: dict[str, Any],
    source_surface: int, source_split: int, target_surface: int, target_split: int,
    role: str, component: str, lower: float, upper: float,
) -> dict[str, Any]:
    source = tensor_custom(source_data, source_surface, source_split, protocol.PRIMARY_FIELD, role, component, lower, upper)
    target = tensor_custom(target_data, target_surface, target_split, protocol.PRIMARY_FIELD, role, component, lower, upper)
    scores = sorted(finalize.permutation_scores(source, target), key=lambda row: row[1], reverse=True)
    identity = next(score for permutation, score in scores if permutation == finalize.IDENTITY_PERMUTATION)
    controls = []
    for field in protocol.CONTROL_FIELDS:
        control = tensor_custom(target_data, target_surface, target_split, field, role, component, lower, upper)
        controls.append(max(score for _, score in finalize.permutation_scores(source, control)))
    return {
        "identity_score": identity,
        "identity_rank": 1 + next(index for index, row in enumerate(scores) if row[0] == finalize.IDENTITY_PERMUTATION),
        "permutation_margin": identity - max(score for permutation, score in scores if permutation != finalize.IDENTITY_PERMUTATION),
        "field_advantage": identity - max(controls),
        "best_permutation": list(scores[0][0]),
    }


def ambiguity_record(label: str, source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    scores = sorted(finalize.permutation_scores(source, target), key=lambda row: row[1], reverse=True)
    best_score = scores[0][1]
    near = [row for row in scores if best_score - row[1] <= NEAR_BEST_TOLERANCE]
    identity_score = next(score for permutation, score in scores if permutation == finalize.IDENTITY_PERMUTATION)
    best = scores[0][0]
    fixed_near = {
        relation: sum(permutation[index] == index for permutation, _ in near) / len(near)
        for index, relation in enumerate(protocol.RELATIONS)
    }
    source_edges = finalize.graph_vector(source)
    target_edges = finalize.graph_vector(target)
    return {
        "label": label,
        "identity_score": identity_score,
        "identity_rank": 1 + next(index for index, row in enumerate(scores) if row[0] == finalize.IDENTITY_PERMUTATION),
        "best_score": best_score,
        "best_permutation": list(best),
        "best_minus_identity": best_score - identity_score,
        "score_range": best_score - scores[-1][1],
        "near_best_permutation_count": len(near),
        "best_fixed_relations": [protocol.RELATIONS[index] for index, value in enumerate(best) if value == index],
        "near_best_fixed_fraction": fixed_near,
        "source_edge_std": float(np.nanstd(source_edges)),
        "target_edge_std": float(np.nanstd(target_edges)),
    }


def main() -> None:
    all_data = {model: finalize.load_model(model) for model in protocol.MODELS}
    ambiguity = []
    for model, data in all_data.items():
        for surface in range(len(protocol.SURFACES)):
            source = finalize.tensor_for(data, surface, 0, protocol.PRIMARY_FIELD, "answer_boundary")
            target = finalize.tensor_for(data, surface, 1, protocol.PRIMARY_FIELD, "answer_boundary")
            ambiguity.append(ambiguity_record(f"split|{model}|{protocol.SURFACES[surface]}", source, target))
        for split in range(len(protocol.SPLITS)):
            source = finalize.tensor_for(data, 0, split, protocol.PRIMARY_FIELD, "answer_boundary")
            target = finalize.tensor_for(data, 1, split, protocol.PRIMARY_FIELD, "answer_boundary")
            ambiguity.append(ambiguity_record(f"language|{model}|{protocol.SPLITS[split]}", source, target))
    for left, right in itertools.combinations(protocol.MODELS, 2):
        for surface in range(len(protocol.SURFACES)):
            for split in range(len(protocol.SPLITS)):
                source = finalize.tensor_for(all_data[left], surface, split, protocol.PRIMARY_FIELD, "answer_boundary")
                target = finalize.tensor_for(all_data[right], surface, split, protocol.PRIMARY_FIELD, "answer_boundary")
                ambiguity.append(ambiguity_record(
                    f"model|{left}|{right}|{protocol.SURFACES[surface]}|{protocol.SPLITS[split]}", source, target
                ))
    best_fixed = Counter(relation for row in ambiguity for relation in row["best_fixed_relations"])
    near_fixed = {
        relation: float(np.mean([row["near_best_fixed_fraction"][relation] for row in ambiguity]))
        for relation in protocol.RELATIONS
    }
    band_records = []
    for model, data in all_data.items():
        for role in ("query_end", "answer_boundary"):
            for component in finalize.COMPONENTS:
                for band, (lower, upper) in DEPTH_BINS.items():
                    cells = []
                    for surface in range(len(protocol.SURFACES)):
                        cells.append(evaluate_custom(data, data, surface, 0, surface, 1, role, component, lower, upper))
                    band_records.append({
                        "model": model,
                        "role": role,
                        "component": component,
                        "depth_band": band,
                        "cells": cells,
                        "minimum_identity_score": min(row["identity_score"] for row in cells),
                        "maximum_identity_rank": max(row["identity_rank"] for row in cells),
                        "minimum_permutation_margin": min(row["permutation_margin"] for row in cells),
                        "minimum_field_advantage": min(row["field_advantage"] for row in cells),
                    })
    band_records.sort(
        key=lambda row: (
            row["minimum_field_advantage"], row["minimum_permutation_margin"], row["minimum_identity_score"]
        ), reverse=True,
    )
    result = {
        "schema_version": "phase1098_failure_diagnostic.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_descriptive_only",
        "near_best_tolerance": NEAR_BEST_TOLERANCE,
        "comparison_count": len(ambiguity),
        "ambiguity_records": ambiguity,
        "best_permutation_fixed_counts": dict(best_fixed),
        "near_best_fixed_fraction_mean": near_fixed,
        "identity_rank_one_count": sum(row["identity_rank"] == 1 for row in ambiguity),
        "median_best_minus_identity": float(np.median([row["best_minus_identity"] for row in ambiguity])),
        "median_permutation_score_range": float(np.median([row["score_range"] for row in ambiguity])),
        "median_near_best_permutation_count": float(np.median([row["near_best_permutation_count"] for row in ambiguity])),
        "top_depth_role_component_bands": band_records[:20],
        "guardrail": "These diagnostics were selected after the primary gates failed and cannot upgrade Phase1098 evidence.",
    }
    result["diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json", result)
    print({
        "phase": protocol.PHASE,
        "comparisons": len(ambiguity),
        "identity_rank_one": result["identity_rank_one_count"],
        "best_fixed_counts": result["best_permutation_fixed_counts"],
        "near_fixed": result["near_best_fixed_fraction_mean"],
        "median_near_best_count": result["median_near_best_permutation_count"],
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
