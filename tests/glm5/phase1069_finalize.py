#!/usr/bin/env python3
"""Aggregate Phase1069 local-coordinate evidence and frozen gates."""

from __future__ import annotations

import itertools
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1069_local_coordinate_protocol as protocol


def median(values: list[float]) -> float | None:
    finite = [
        float(value) for value in values
        if math.isfinite(float(value))
    ]
    return float(statistics.median(finite)) if finite else None


def weighted_mean(
    rows: list[dict[str, Any]],
    value_key: str,
    count_key: str,
) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = row.get(value_key)
        count = int(row.get(count_key, 0))
        if value is None or not math.isfinite(float(value)) or count <= 0:
            continue
        numerator += float(value) * count
        denominator += count
    return numerator / denominator if denominator else None


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def late_relation_evidence(
    relation: str,
    summary: dict[str, Any],
    responses: list[dict[str, Any]],
    readouts: list[dict[str, Any]],
    late_start: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "relation": relation,
        "behavior": summary["relations"][relation],
        "splits": {},
    }
    for split in protocol.SPLITS:
        response_rows = [
            row for row in responses
            if row["relation"] == relation
            and row["split"] == split
            and row["task_kind"] == "transitive"
            and row["role"] == "answer_boundary"
            and row["conditioning"] == "behavior_conditioned"
            and float(row["relative_depth"]) >= late_start
            and int(row["complete_factorial_count"]) > 0
        ]
        readout_rows = [
            row for row in readouts
            if row["relation"] == relation
            and row["split"] == split
            and row["task_kind"] == "transitive"
            and row["conditioning"] == "behavior_conditioned"
            and float(row["relative_depth"]) >= late_start
            and int(row["semantic_pair_count"]) > 0
        ]
        result["splits"][split] = {
            "late_lexical_semantic_cosine_median": median([
                row["mean_lexical_semantic_cosine"]
                for row in response_rows
            ]),
            "late_interaction_relative_magnitude_median": median([
                row["mean_interaction_relative_magnitude"]
                for row in response_rows
            ]),
            "late_matched_readout_shift": weighted_mean(
                readout_rows,
                "mean_matched_readout_shift",
                "semantic_pair_count",
            ),
            "late_mismatched_readout_shift": weighted_mean(
                readout_rows,
                "mean_mismatched_readout_shift",
                "semantic_pair_count",
            ),
            "late_matched_readout_positive_rate": weighted_mean(
                readout_rows,
                "matched_readout_positive_rate",
                "semantic_pair_count",
            ),
            "late_mismatched_readout_positive_rate": weighted_mean(
                readout_rows,
                "mismatched_readout_positive_rate",
                "semantic_pair_count",
            ),
            "late_positive_rate_gap": weighted_mean(
                readout_rows,
                "positive_rate_gap",
                "semantic_pair_count",
            ),
            "late_matched_answer_axis_cosine": weighted_mean(
                readout_rows,
                "mean_matched_answer_axis_cosine",
                "semantic_pair_count",
            ),
            "late_mismatched_answer_axis_cosine": weighted_mean(
                readout_rows,
                "mean_mismatched_answer_axis_cosine",
                "semantic_pair_count",
            ),
            "late_absolute_surface_readout_shift": weighted_mean(
                readout_rows,
                "mean_absolute_surface_readout_shift",
                "surface_observation_count",
            ),
            "readout_pair_observations": sum(
                int(row["semantic_pair_count"])
                for row in readout_rows
            ),
        }
    return result


def model_operation_gate(
    model: str,
    relation_rows: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    strong = [
        row for row in relation_rows
        if row["behavior"]["strong_behavior_gate_passed"]
    ]
    selected = strong
    split_values = {}
    for split in protocol.SPLITS:
        lexical = [
            row["splits"][split][
                "late_lexical_semantic_cosine_median"
            ]
            for row in selected
            if row["splits"][split][
                "late_lexical_semantic_cosine_median"
            ] is not None
        ]
        matched_positive = [
            row["splits"][split][
                "late_matched_readout_positive_rate"
            ]
            for row in selected
            if row["splits"][split][
                "late_matched_readout_positive_rate"
            ] is not None
        ]
        gaps = [
            row["splits"][split]["late_positive_rate_gap"]
            for row in selected
            if row["splits"][split]["late_positive_rate_gap"]
            is not None
        ]
        split_values[split] = {
            "selected_relation_count": len(selected),
            "lexical_semantic_cosine_median": median(lexical),
            "matched_readout_positive_rate_median": median(
                matched_positive
            ),
            "positive_rate_gap_median": median(gaps),
        }
    enough_relations = (
        len(selected)
        >= int(gates["minimum_strong_relations_per_model"])
    )
    split_gate = all(
        split_values[split]["lexical_semantic_cosine_median"]
        is not None
        and split_values[split]["lexical_semantic_cosine_median"]
        >= float(gates["late_lexical_semantic_cosine_min"])
        and split_values[split][
            "matched_readout_positive_rate_median"
        ] is not None
        and split_values[split][
            "matched_readout_positive_rate_median"
        ] >= float(
            gates["late_matched_readout_positive_rate_min"]
        )
        and split_values[split]["positive_rate_gap_median"]
        is not None
        and split_values[split]["positive_rate_gap_median"]
        >= float(
            gates[
                "late_matched_vs_mismatch_positive_rate_gap_min"
            ]
        )
        for split in protocol.SPLITS
    )
    return {
        "schema_version": "phase1069_model_operation_gate.v1",
        "phase": protocol.PHASE,
        "model": model,
        "strong_behavior_relations": [
            row["relation"] for row in strong
        ],
        "strong_behavior_relation_count": len(strong),
        "split_evidence": split_values,
        "enough_strong_relations": enough_relations,
        "split_internal_gate_passed": split_gate,
        "shared_order_operation_gate_passed": bool(
            enough_relations and split_gate
        ),
    }


def relation_fingerprint(
    rows: list[dict[str, Any]],
    roles: tuple[str, ...],
) -> dict[str, Any]:
    metrics = (
        "mean_semantic_relative_magnitude",
        "mean_surface_relative_magnitude",
        "mean_lexical_semantic_cosine",
        "mean_interaction_relative_magnitude",
    )
    selected = [
        row for row in rows
        if row["task_kind"] == "transitive"
        and row["conditioning"] == "behavior_conditioned"
        and row["role"] in roles
        and int(row["complete_factorial_count"]) > 0
    ]
    values: dict[
        tuple[str, str, tuple[Any, ...]], float
    ] = {}
    coordinates = set()
    for row in selected:
        for metric in metrics:
            coordinate = (
                row["event_id"],
                row["role"],
                metric,
            )
            values[
                (row["split"], row["relation"], coordinate)
            ] = float(row[metric])
            coordinates.add(coordinate)
    usable = [
        coordinate for coordinate in sorted(coordinates)
        if all(
            (split, relation, coordinate) in values
            for split in protocol.SPLITS
            for relation in protocol.RELATION_NAMES
        )
    ]
    fingerprints: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in protocol.SPLITS
    }
    for split in protocol.SPLITS:
        matrix = np.array([
            [
                values[(split, relation, coordinate)]
                for coordinate in usable
            ]
            for relation in protocol.RELATION_NAMES
        ], dtype=np.float64)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scale = np.sqrt(
            np.mean(centered * centered, axis=0, keepdims=True)
        )
        scale[scale <= 1e-12] = 1.0
        standardized = centered / scale
        for index, relation in enumerate(protocol.RELATION_NAMES):
            fingerprints[split][relation] = standardized[index]

    matrix_rows = []
    matched = []
    mismatched = []
    correct = 0
    for left_relation in protocol.RELATION_NAMES:
        candidates = []
        for right_relation in protocol.RELATION_NAMES:
            value = cosine(
                fingerprints["discovery"][left_relation],
                fingerprints["confirmation"][right_relation],
            )
            matrix_rows.append({
                "discovery_relation": left_relation,
                "confirmation_relation": right_relation,
                "cosine": value,
            })
            candidates.append((value, right_relation))
            if value is not None:
                if left_relation == right_relation:
                    matched.append(value)
                else:
                    mismatched.append(value)
        best = max(
            candidates,
            key=lambda item: (
                -float("inf")
                if item[0] is None else float(item[0])
            ),
        )[1]
        correct += int(best == left_relation)
    matched_median = median(matched)
    mismatched_median = median(mismatched)
    gap = (
        matched_median - mismatched_median
        if matched_median is not None
        and mismatched_median is not None
        else None
    )
    return {
        "roles": list(roles),
        "coordinate_count": len(usable),
        "cosine_matrix": matrix_rows,
        "matched_cosine_median": matched_median,
        "mismatched_cosine_median": mismatched_median,
        "specificity_gap": gap,
        "retrieval_accuracy": (
            correct / len(protocol.RELATION_NAMES)
        ),
    }


def operation_profile(
    readouts: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    by_depth: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in readouts:
        if (
            row["task_kind"] == "transitive"
            and row["conditioning"] == "behavior_conditioned"
            and int(row["semantic_pair_count"]) > 0
        ):
            by_depth[float(row["relative_depth"])].append(row)
    depths = np.array(sorted(by_depth), dtype=np.float64)
    values = np.array([
        weighted_mean(
            by_depth[depth],
            "positive_rate_gap",
            "semantic_pair_count",
        ) or 0.0
        for depth in depths
    ], dtype=np.float64)
    return depths, values


def posthoc_task_and_selection_controls(
    responses: list[dict[str, Any]],
    readouts: list[dict[str, Any]],
    late_start: float,
) -> dict[str, Any]:
    """Describe direct-chain and behavior-selection alternatives."""
    cells = {}
    for task_kind in ("direct", "transitive"):
        cells[task_kind] = {}
        for conditioning in ("all", "behavior_conditioned"):
            response_rows = [
                row for row in responses
                if row["task_kind"] == task_kind
                and row["conditioning"] == conditioning
                and row["role"] == "answer_boundary"
                and float(row["relative_depth"]) >= late_start
                and int(row["complete_factorial_count"]) > 0
            ]
            readout_rows = [
                row for row in readouts
                if row["task_kind"] == task_kind
                and row["conditioning"] == conditioning
                and float(row["relative_depth"]) >= late_start
                and int(row["semantic_pair_count"]) > 0
            ]
            cells[task_kind][conditioning] = {
                "late_lexical_semantic_cosine_median": median([
                    row["mean_lexical_semantic_cosine"]
                    for row in response_rows
                ]),
                "late_matched_readout_positive_rate": weighted_mean(
                    readout_rows,
                    "matched_readout_positive_rate",
                    "semantic_pair_count",
                ),
                "late_mismatched_readout_positive_rate": weighted_mean(
                    readout_rows,
                    "mismatched_readout_positive_rate",
                    "semantic_pair_count",
                ),
                "late_positive_rate_gap": weighted_mean(
                    readout_rows,
                    "positive_rate_gap",
                    "semantic_pair_count",
                ),
                "late_matched_readout_shift": weighted_mean(
                    readout_rows,
                    "mean_matched_readout_shift",
                    "semantic_pair_count",
                ),
                "semantic_pair_observations": sum(
                    int(row["semantic_pair_count"])
                    for row in readout_rows
                ),
            }

    conditioned_direct = cells["direct"]["behavior_conditioned"]
    conditioned_transitive = cells["transitive"][
        "behavior_conditioned"
    ]
    all_transitive = cells["transitive"]["all"]
    direct_transitive = {
        "positive_rate_gap_difference": (
            conditioned_transitive["late_positive_rate_gap"]
            - conditioned_direct["late_positive_rate_gap"]
        ),
        "matched_readout_shift_difference": (
            conditioned_transitive["late_matched_readout_shift"]
            - conditioned_direct["late_matched_readout_shift"]
        ),
        "lexical_semantic_cosine_difference": (
            conditioned_transitive[
                "late_lexical_semantic_cosine_median"
            ]
            - conditioned_direct[
                "late_lexical_semantic_cosine_median"
            ]
        ),
    }
    selection_effect = {
        "transitive_positive_rate_gap_increase": (
            conditioned_transitive["late_positive_rate_gap"]
            - all_transitive["late_positive_rate_gap"]
        ),
        "transitive_matched_positive_rate_increase": (
            conditioned_transitive[
                "late_matched_readout_positive_rate"
            ]
            - all_transitive[
                "late_matched_readout_positive_rate"
            ]
        ),
        "transitive_lexical_cosine_increase": (
            conditioned_transitive[
                "late_lexical_semantic_cosine_median"
            ]
            - all_transitive[
                "late_lexical_semantic_cosine_median"
            ]
        ),
    }

    by_depth: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in readouts:
        if (
            row["task_kind"] == "transitive"
            and row["conditioning"] == "behavior_conditioned"
            and int(row["semantic_pair_count"]) > 0
        ):
            by_depth[float(row["relative_depth"])].append(row)
    depth_rows = []
    for depth in sorted(by_depth):
        values = by_depth[depth]
        depth_rows.append({
            "relative_depth": depth,
            "matched_readout_positive_rate": weighted_mean(
                values,
                "matched_readout_positive_rate",
                "semantic_pair_count",
            ),
            "positive_rate_gap": weighted_mean(
                values,
                "positive_rate_gap",
                "semantic_pair_count",
            ),
        })
    emergence = None
    for index in range(max(0, len(depth_rows) - 2)):
        window = depth_rows[index:index + 3]
        if all(
            row["matched_readout_positive_rate"] is not None
            and row["matched_readout_positive_rate"] >= 0.80
            and row["positive_rate_gap"] is not None
            and row["positive_rate_gap"] >= 0.30
            for row in window
        ):
            emergence = window[0]["relative_depth"]
            break
    return {
        "schema_version": "phase1069_posthoc_controls.v1",
        "phase": protocol.PHASE,
        "status": (
            "posthoc_descriptive_control_not_a_preregistered_gate"
        ),
        "cells": cells,
        "direct_vs_transitive": direct_transitive,
        "behavior_conditioning_effect": selection_effect,
        "local_readout_emergence_rule": (
            "first of three consecutive depths with matched positive "
            "rate >= 0.80 and matched-minus-mismatched gap >= 0.30"
        ),
        "local_readout_emergence_relative_depth": emergence,
        "depth_profile": depth_rows,
    }


def cross_model_profiles(
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    grid = np.linspace(0.0, 1.0, 101)
    for left, right in itertools.combinations(protocol.MODELS, 2):
        left_depths, left_values = profiles[left]
        right_depths, right_values = profiles[right]
        value = (
            cosine(
                np.interp(grid, left_depths, left_values),
                np.interp(grid, right_depths, right_values),
            )
            if len(left_depths) and len(right_depths)
            else None
        )
        rows.append({
            "schema_version": (
                "phase1069_cross_model_operation_profile.v1"
            ),
            "phase": protocol.PHASE,
            "left_model": left,
            "right_model": right,
            "profile": (
                "behavior_conditioned_transitive_"
                "matched_minus_mismatch_positive_rate"
            ),
            "profile_cosine": value,
        })
    return rows


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {}
    relation_evidence_rows = []
    model_gate_rows = []
    fingerprints = {}
    profiles = {}
    posthoc_controls = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(atlas / "summary.json")
        responses = protocol.read_jsonl(
            atlas / "response_metrics.jsonl"
        )
        readouts = protocol.read_jsonl(
            atlas / "local_readout_metrics.jsonl"
        )
        summaries[model] = summary
        model_relation_rows = [
            late_relation_evidence(
                relation,
                summary,
                responses,
                readouts,
                float(prereg["gates"]["late_depth_start"]),
            )
            for relation in protocol.RELATION_NAMES
        ]
        for row in model_relation_rows:
            relation_evidence_rows.append({
                "schema_version": (
                    "phase1069_relation_evidence.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                **row,
            })
        model_gate_rows.append(model_operation_gate(
            model, model_relation_rows, prereg["gates"]
        ))
        fingerprints[model] = {
            "source_and_query": relation_fingerprint(
                responses,
                (
                    "logical_first",
                    "logical_last",
                    "query_near_premise",
                    "query",
                ),
            ),
            "answer_boundary": relation_fingerprint(
                responses, ("answer_boundary",)
            ),
            "all_roles": relation_fingerprint(
                responses, tuple(protocol.CAPTURE_ROLES)
            ),
        }
        for name, result in fingerprints[model].items():
            result["relation_domain_gate_passed"] = bool(
                result["retrieval_accuracy"]
                >= float(prereg["gates"][
                    "relation_fingerprint_retrieval_accuracy_min"
                ])
                and result["specificity_gap"] is not None
                and result["specificity_gap"]
                >= float(prereg["gates"][
                    "relation_fingerprint_specificity_gap_min"
                ])
            )
            result["fingerprint_name"] = name
        profiles[model] = operation_profile(readouts)
        posthoc_controls[model] = (
            posthoc_task_and_selection_controls(
                responses,
                readouts,
                float(prereg["gates"]["late_depth_start"]),
            )
        )
        posthoc_controls[model]["model"] = model

    repeated_models = [
        row["model"] for row in model_gate_rows
        if row["shared_order_operation_gate_passed"]
    ]
    should_continue = (
        len(repeated_models)
        >= int(prereg["gates"]["minimum_repeated_models"])
    )
    automatic_next = {
        "schema_version": "phase1069_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": should_continue,
        "selected_models": repeated_models,
        "repeated_model_count": len(repeated_models),
        "route": (
            "phase1070_operation_invariant_causal_localization"
            if should_continue
            else "stop_at_local_coordinate_atlas"
        ),
        "rationale": (
            "The decision uses the frozen shared-operation behavior, "
            "lexical-reuse, matched local-readout, and mismatched-axis "
            "controls. Relation-domain fingerprint gates are reported "
            "separately and cannot rescue a failed shared-operation gate."
        ),
    }
    cross_profiles = cross_model_profiles(profiles)

    protocol.write_jsonl(
        protocol.OUT_ROOT / "analysis" / "relation_evidence.jsonl",
        relation_evidence_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT / "analysis" / "model_operation_gates.jsonl",
        model_gate_rows,
    )
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "relation_fingerprints.json",
        {
            "schema_version": "phase1069_relation_fingerprints.v1",
            "phase": protocol.PHASE,
            "models": fingerprints,
        },
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "cross_model_operation_profiles.jsonl",
        cross_profiles,
    )
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "posthoc_task_and_selection_controls.json",
        {
            "schema_version": (
                "phase1069_posthoc_control_collection.v1"
            ),
            "phase": protocol.PHASE,
            "models": posthoc_controls,
        },
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic_next,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json",
        {
            "schema_version": "phase1069_aggregate.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model_summaries": summaries,
            "relation_evidence": relation_evidence_rows,
            "model_operation_gates": model_gate_rows,
            "relation_fingerprints": fingerprints,
            "cross_model_operation_profiles": cross_profiles,
            "posthoc_task_and_selection_controls": posthoc_controls,
            "automatic_next": automatic_next,
        },
    )
    print({
        "phase": protocol.PHASE,
        "model_operation_gates": {
            row["model"]: row[
                "shared_order_operation_gate_passed"
            ]
            for row in model_gate_rows
        },
        "automatic_next": automatic_next,
    })


if __name__ == "__main__":
    main()
