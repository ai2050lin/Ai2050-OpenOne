#!/usr/bin/env python3
"""Finalize Phase1079 without assigning causal evidence."""

from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1079_output_orthogonal_pattern_protocol as protocol


COMPONENTS = ("residual", "attention_output", "mlp_output")
DEPTH_GRID = np.linspace(0.0, 1.0, 13)
EPSILON = 1e-12


FIELD_COLUMNS = {
    "operation": "mean_operation_relative_magnitude",
    "controlled_semantic_answer": (
        "mean_controlled_answer_relative_magnitude"
    ),
    "natural_answer": "mean_natural_answer_relative_magnitude",
}


def safe_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def safe_median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(
        np.linalg.norm(left.astype(np.float64, copy=False))
        * np.linalg.norm(right.astype(np.float64, copy=False))
    )
    if denominator <= EPSILON:
        return 0.0
    return float(np.dot(
        left.astype(np.float64, copy=False),
        right.astype(np.float64, copy=False),
    ) / denominator)


def channel_normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values.astype(np.float64, copy=False)))
    return values / norm if norm > EPSILON else values


def interpolate_channel(
    rows: list[dict[str, Any]],
    *,
    family: str,
    split: str,
    component: str,
    role: str,
    column: str,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["conditioning"] == "all_finite"
        and row["family"] == family
        and row["split"] == split
        and row["component"] == component
        and row["role"] == role
        and row[column] is not None
    ]
    if not selected:
        return np.zeros(len(DEPTH_GRID), dtype=np.float64)
    by_depth: dict[float, list[float]] = defaultdict(list)
    for row in selected:
        by_depth[float(row["relative_depth"])].append(
            float(row[column])
        )
    depths = np.array(sorted(by_depth), dtype=np.float64)
    values = np.array([
        float(np.mean(by_depth[depth])) for depth in depths
    ], dtype=np.float64)
    if len(depths) == 1:
        result = np.full(len(DEPTH_GRID), values[0], dtype=np.float64)
    else:
        result = np.interp(DEPTH_GRID, depths, values)
    return channel_normalize(result)


def build_profile(
    rows: list[dict[str, Any]],
    family: str,
    split: str,
    field: str,
    *,
    roles: tuple[str, ...] | None = None,
) -> np.ndarray:
    column = FIELD_COLUMNS[field]
    selected_roles = roles or protocol.CAPTURE_ROLES
    channels = []
    for component in COMPONENTS:
        for role in selected_roles:
            channels.append(interpolate_channel(
                rows,
                family=family,
                split=split,
                component=component,
                role=role,
                column=column,
            ))
    return np.concatenate(channels)


def profile_bank(
    rows: list[dict[str, Any]],
    families: tuple[str, ...],
    split: str,
    field: str,
    *,
    roles: tuple[str, ...] | None = None,
    centered: bool,
) -> np.ndarray:
    values = np.stack([
        build_profile(
            rows, family, split, field, roles=roles
        )
        for family in families
    ])
    if centered:
        values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > EPSILON,
    )


def similarity_matrix(
    source_values: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    return source_values @ target_values.T


def exact_assignment_test(matrix: np.ndarray) -> dict[str, Any]:
    size = matrix.shape[0]
    identity_score = float(np.trace(matrix) / size)
    scores_at_least_identity = 0
    permutation_count = 0
    row_axis = np.arange(size)
    best_other = -math.inf
    for permutation in itertools.permutations(range(size)):
        score = float(
            matrix[row_axis, np.array(permutation)].mean()
        )
        permutation_count += 1
        if score >= identity_score - 1e-12:
            scores_at_least_identity += 1
        if permutation != tuple(range(size)):
            best_other = max(best_other, score)
    return {
        "identity_mean_score": identity_score,
        "permutation_count": permutation_count,
        "scores_at_least_identity": scores_at_least_identity,
        "exact_upper_tail_p": (
            scores_at_least_identity / permutation_count
        ),
        "best_nonidentity_mean_score": best_other,
        "identity_margin_over_best_other": identity_score - best_other,
    }


def assignment_record(
    *,
    comparison: str,
    field: str,
    profile: str,
    source_model: str,
    target_model: str,
    families: tuple[str, ...],
    source_values: np.ndarray,
    target_values: np.ndarray,
) -> dict[str, Any]:
    matrix = similarity_matrix(source_values, target_values)
    exact = exact_assignment_test(matrix)
    rows = []
    for index, family in enumerate(families):
        order = np.argsort(-matrix[index])
        predicted = int(order[0])
        second = int(order[1])
        rows.append({
            "family": family,
            "predicted_family": families[predicted],
            "correct": predicted == index,
            "correct_similarity": float(matrix[index, index]),
            "best_other_family": families[
                second if predicted == index else predicted
            ],
            "best_other_similarity": float(
                matrix[index, second if predicted == index else predicted]
            ),
        })
    return {
        "schema_version": "phase1079_assignment.v1",
        "phase": protocol.PHASE,
        "comparison": comparison,
        "field": field,
        "profile": profile,
        "source_model": source_model,
        "target_model": target_model,
        "families": list(families),
        "family_count": len(families),
        "top1_correct": sum(int(row["correct"]) for row in rows),
        "rows": rows,
        "similarity_matrix": matrix.tolist(),
        **exact,
    }


def behavior_models(
    summaries: dict[str, dict[str, Any]],
    family: str,
) -> list[str]:
    passing = []
    candidate_threshold = protocol.EVIDENCE_THRESHOLDS[
        "candidate_accuracy_for_behavior_annotation"
    ]
    generation_threshold = protocol.EVIDENCE_THRESHOLDS[
        "natural_generation_first_accuracy"
    ]
    for model_name, summary in summaries.items():
        split_passes = []
        for split in protocol.SPLITS:
            rows = summary["behavior_summary"][family][split]
            candidate_values = [
                rows["controlled_semantic"]["candidate_accuracy"],
                rows["natural_natural"]["candidate_accuracy"],
            ]
            generation_value = rows["natural_generation"][
                "semantic_first_accuracy"
            ]
            split_passes.append(
                all(
                    value is not None
                    and float(value) >= candidate_threshold
                    for value in candidate_values
                )
                and generation_value is not None
                and float(generation_value) >= generation_threshold
            )
        if all(split_passes):
            passing.append(model_name)
    return passing


def find_assignment(
    assignments: list[dict[str, Any]],
    **criteria,
) -> list[dict[str, Any]]:
    return [
        row
        for row in assignments
        if all(row.get(key) == value for key, value in criteria.items())
    ]


def family_correct(
    assignment: dict[str, Any],
    family: str,
) -> bool:
    return next(
        row["correct"]
        for row in assignment["rows"]
        if row["family"] == family
    )


def answer_alignment(
    source_phase1078: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": "phase1079_answer_alignment.v1",
        "phase": protocol.PHASE,
        "by_model": {},
    }
    for model_name in protocol.MODELS:
        path = (
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "selected_mean_directions.fp16.npz"
        )
        data = np.load(path)
        families = [str(value) for value in data["families"]]
        splits = [str(value) for value in data["splits"]]
        depths = data["residual_depths"].astype(np.float64)
        vectors = data["natural_answer"].astype(np.float64)
        conditioning = 0
        split = splits.index("confirmation")
        role = 1
        family_indices = [
            families.index(family)
            for family in protocol.BASE_FAMILIES
        ]
        depth_profile = []
        for depth_index, depth in enumerate(depths):
            values = vectors[
                conditioning,
                family_indices,
                split,
                depth_index,
                role,
                :,
            ]
            pair_values = []
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    denominator = float(
                        np.linalg.norm(values[left])
                        * np.linalg.norm(values[right])
                    )
                    if denominator > EPSILON:
                        pair_values.append(float(
                            np.dot(values[left], values[right])
                            / denominator
                        ))
            depth_profile.append({
                "depth": int(depth),
                "relative_depth": (
                    float(depth / depths.max())
                    if depths.max() > 0 else 0.0
                ),
                "mean_cross_family_natural_answer_cosine": (
                    safe_mean(pair_values)
                ),
                "pair_count": len(pair_values),
            })
        early = [
            float(row["mean_cross_family_natural_answer_cosine"])
            for row in depth_profile
            if row["mean_cross_family_natural_answer_cosine"] is not None
            and 0.0 < row["relative_depth"] <= 0.25
        ]
        late = [
            float(row["mean_cross_family_natural_answer_cosine"])
            for row in depth_profile
            if row["mean_cross_family_natural_answer_cosine"] is not None
            and row["relative_depth"] >= 0.75
        ]
        early_mean = safe_mean(early)
        late_mean = safe_mean(late)
        old_late = float(
            source_phase1078["generic_truth_alignment"]["by_model"][
                model_name
            ]["late_mean"]
        )
        reduction = (
            old_late - late_mean
            if late_mean is not None else None
        )
        result["by_model"][model_name] = {
            "early_mean": early_mean,
            "late_mean": late_mean,
            "late_minus_early": (
                late_mean - early_mean
                if early_mean is not None and late_mean is not None
                else None
            ),
            "phase1078_late_mean": old_late,
            "phase1078_to_phase1079_reduction": reduction,
            "depth_profile": depth_profile,
        }
    threshold = protocol.EVIDENCE_THRESHOLDS[
        "phase1078_alignment_drop_min"
    ]
    result["passing_models"] = [
        model
        for model, row in result["by_model"].items()
        if row["phase1078_to_phase1079_reduction"] is not None
        and row["phase1078_to_phase1079_reduction"] >= threshold
    ]
    return result


def factor_ratios(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_model = {}
    for model_name, rows in metrics_by_model.items():
        family_rows = {}
        pooled = []
        for family in protocol.FAMILIES:
            ratios = []
            for row in rows:
                if (
                    row["conditioning"] != "all_finite"
                    or row["family"] != family
                ):
                    continue
                operation = row["mean_operation_relative_magnitude"]
                surface = row["mean_surface_relative_magnitude"]
                shell = row["mean_shell_relative_magnitude"]
                if (
                    operation is None
                    or float(operation) <= EPSILON
                    or surface is None
                    or shell is None
                ):
                    continue
                ratio = max(float(surface), float(shell)) / float(
                    operation
                )
                if math.isfinite(ratio):
                    ratios.append(ratio)
                    pooled.append(ratio)
            family_rows[family] = {
                "median_max_control_to_operation": safe_median(ratios),
                "observation_count": len(ratios),
            }
        by_model[model_name] = {
            "families": family_rows,
            "pooled_median_max_control_to_operation": (
                safe_median(pooled)
            ),
        }
    return {
        "schema_version": "phase1079_factor_ratios.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
    }


def top_regions(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for model_name, rows in metrics_by_model.items():
        for family in protocol.FAMILIES:
            for field, column in FIELD_COLUMNS.items():
                candidates = [
                    row
                    for row in rows
                    if row["conditioning"] == "all_finite"
                    and row["family"] == family
                    and row["split"] == "confirmation"
                    and row[column] is not None
                ]
                candidates.sort(
                    key=lambda row: float(row[column]),
                    reverse=True,
                )
                for rank, row in enumerate(candidates[:5], 1):
                    output.append({
                        "schema_version": (
                            "phase1079_top_region.v1"
                        ),
                        "phase": protocol.PHASE,
                        "model": model_name,
                        "family": family,
                        "field": field,
                        "rank": rank,
                        "component": row["component"],
                        "depth": row["depth"],
                        "relative_depth": row["relative_depth"],
                        "role": row["role"],
                        "mean_relative_magnitude": float(row[column]),
                    })
    return output


def heldout_audit(
    metrics_by_model: dict[str, list[dict[str, Any]]],
    top_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_model = {}
    for model_name, rows in metrics_by_model.items():
        families = protocol.FAMILIES
        values = profile_bank(
            rows,
            families,
            "confirmation",
            "operation",
            roles=("answer_boundary",),
            centered=True,
        )
        heldout_index = families.index(protocol.HELDOUT_FAMILY)
        similarities = [
            (families[index], float(values[heldout_index] @ values[index]))
            for index in range(len(families))
            if index != heldout_index
        ]
        similarities.sort(key=lambda row: row[1], reverse=True)
        peak = next(
            row
            for row in top_rows
            if row["model"] == model_name
            and row["family"] == protocol.HELDOUT_FAMILY
            and row["field"] == "operation"
            and row["rank"] == 1
        )
        nearest_pass = similarities[0][0] == "contrast_conjunction"
        peak_pass = (
            float(peak["relative_depth"]) >= 0.4
            and peak["component"] in {
                "attention_output",
                "mlp_output",
            }
        )
        by_model[model_name] = {
            "nearest_base_family": similarities[0][0],
            "nearest_similarity": similarities[0][1],
            "all_base_similarities": [
                {"family": family, "similarity": similarity}
                for family, similarity in similarities
            ],
            "operation_peak": peak,
            "nearest_prediction_passed": nearest_pass,
            "peak_prediction_passed": peak_pass,
            "joint_prediction_passed": nearest_pass and peak_pass,
        }
    return {
        "schema_version": "phase1079_heldout_audit.v1",
        "phase": protocol.PHASE,
        "heldout_family": protocol.HELDOUT_FAMILY,
        "predicted_nearest_family": "contrast_conjunction",
        "by_model": by_model,
        "passing_models": [
            model
            for model, row in by_model.items()
            if row["joint_prediction_passed"]
        ],
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    metrics_by_model = {
        model: protocol.read_jsonl(
            protocol.OUT_ROOT
            / "atlas"
            / model
            / "response_metrics.jsonl"
        )
        for model in protocol.MODELS
    }
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    source_phase1078 = protocol.read_json(protocol.SOURCE_PHASE1078)
    threshold_p = float(
        prereg["evidence_thresholds"]["permutation_p_max"]
    )
    minimum = int(
        prereg["evidence_thresholds"][
            "minimum_repeated_models_or_pairs"
        ]
    )
    minimum_top1 = int(
        prereg["evidence_thresholds"]["minimum_base_family_top1"]
    )

    assignments = []
    families = protocol.BASE_FAMILIES
    for model_name, rows in metrics_by_model.items():
        for field in (
            "operation",
            "controlled_semantic_answer",
            "natural_answer",
        ):
            for centered in (False, True):
                profile_name = "family_centered" if centered else "raw"
                assignments.append(assignment_record(
                    comparison="within_model_discovery_to_confirmation",
                    field=field,
                    profile=profile_name,
                    source_model=model_name,
                    target_model=model_name,
                    families=families,
                    source_values=profile_bank(
                        rows, families, "discovery", field,
                        centered=centered,
                    ),
                    target_values=profile_bank(
                        rows, families, "confirmation", field,
                        centered=centered,
                    ),
                ))

        for source_field, target_field, name in (
            (
                "controlled_semantic_answer",
                "natural_answer",
                "controlled_to_natural_transfer",
            ),
            (
                "natural_answer",
                "controlled_semantic_answer",
                "natural_to_controlled_transfer",
            ),
        ):
            for centered in (False, True):
                profile_name = "family_centered" if centered else "raw"
                assignments.append(assignment_record(
                    comparison=name,
                    field=f"{source_field}__{target_field}",
                    profile=profile_name,
                    source_model=model_name,
                    target_model=model_name,
                    families=families,
                    source_values=profile_bank(
                        rows,
                        families,
                        "discovery",
                        source_field,
                        centered=centered,
                    ),
                    target_values=profile_bank(
                        rows,
                        families,
                        "confirmation",
                        target_field,
                        centered=centered,
                    ),
                ))

    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            for field in ("operation", "natural_answer"):
                for centered in (False, True):
                    profile_name = (
                        "family_centered" if centered else "raw"
                    )
                    assignments.append(assignment_record(
                        comparison="cross_model_confirmation",
                        field=field,
                        profile=profile_name,
                        source_model=source_model,
                        target_model=target_model,
                        families=families,
                        source_values=profile_bank(
                            metrics_by_model[source_model],
                            families,
                            "confirmation",
                            field,
                            centered=centered,
                        ),
                        target_values=profile_bank(
                            metrics_by_model[target_model],
                            families,
                            "confirmation",
                            field,
                            centered=centered,
                        ),
                    ))

    alignment = answer_alignment(source_phase1078)
    factors = factor_ratios(metrics_by_model)
    regions = top_regions(metrics_by_model)
    heldout = heldout_audit(metrics_by_model, regions)

    p1_models = []
    for model in protocol.MODELS:
        candidates = find_assignment(
            assignments,
            comparison="within_model_discovery_to_confirmation",
            field="operation",
            profile="family_centered",
            source_model=model,
        )
        if (
            len(candidates) == 1
            and candidates[0]["exact_upper_tail_p"] <= threshold_p
            and candidates[0]["top1_correct"] >= minimum_top1
        ):
            p1_models.append(model)

    p2_pairs = [
        f"{row['source_model']}__{row['target_model']}"
        for row in assignments
        if row["comparison"] == "cross_model_confirmation"
        and row["field"] == "operation"
        and row["profile"] == "family_centered"
        and row["exact_upper_tail_p"] <= threshold_p
        and row["top1_correct"] >= minimum_top1
    ]

    transfer_threshold = int(
        prereg["evidence_thresholds"][
            "natural_controlled_transfer_top1"
        ]
    )
    p4_models = []
    for model in protocol.MODELS:
        candidates = [
            row
            for row in assignments
            if row["comparison"]
            == "controlled_to_natural_transfer"
            and row["profile"] == "family_centered"
            and row["source_model"] == model
            and row["exact_upper_tail_p"] <= threshold_p
            and row["top1_correct"] >= transfer_threshold
        ]
        if candidates:
            p4_models.append(model)

    pre_mode_tolerance = float(
        prereg["evidence_thresholds"][
            "pre_mode_operation_tolerance"
        ]
    )
    p6_models = [
        model
        for model, summary in summaries.items()
        if float(summary["pre_mode_operation_max_abs"])
        <= pre_mode_tolerance
    ]
    prediction_audit = {
        "schema_version": "phase1079_prediction_audit.v1",
        "phase": protocol.PHASE,
        "predictions": {
            "P1": {
                "passed": len(p1_models) >= minimum,
                "passing_models": p1_models,
            },
            "P2": {
                "passed": len(p2_pairs) >= minimum,
                "passing_directed_pairs": p2_pairs,
            },
            "P3": {
                "passed": len(alignment["passing_models"]) >= minimum,
                "passing_models": alignment["passing_models"],
            },
            "P4": {
                "passed": len(p4_models) >= minimum,
                "passing_models": p4_models,
            },
            "P5": {
                "passed": len(heldout["passing_models"]) >= minimum,
                "passing_models": heldout["passing_models"],
            },
            "P6": {
                "passed": len(p6_models) == len(protocol.MODELS),
                "passing_models": p6_models,
                "by_model": {
                    model: summaries[model][
                        "pre_mode_operation_max_abs"
                    ]
                    for model in protocol.MODELS
                },
            },
        },
    }
    prediction_audit["passed_count"] = sum(
        int(row["passed"])
        for row in prediction_audit["predictions"].values()
    )
    prediction_audit["prediction_digest"] = protocol.digest(
        prediction_audit
    )

    family_evidence = {}
    evidence_rows = []
    for family in protocol.FAMILIES:
        if family == protocol.HELDOUT_FAMILY:
            operation_within = [
                model
                for model in protocol.MODELS
                if heldout["by_model"][model][
                    "nearest_prediction_passed"
                ]
            ]
            cross_hits = []
            transfer_models = []
        else:
            operation_within = []
            for model in protocol.MODELS:
                row = next(
                    value
                    for value in assignments
                    if value["comparison"]
                    == "within_model_discovery_to_confirmation"
                    and value["field"] == "operation"
                    and value["profile"] == "family_centered"
                    and value["source_model"] == model
                )
                if (
                    row["exact_upper_tail_p"] <= threshold_p
                    and family_correct(row, family)
                ):
                    operation_within.append(model)
            cross_hits = [
                f"{row['source_model']}__{row['target_model']}"
                for row in assignments
                if row["comparison"] == "cross_model_confirmation"
                and row["field"] == "operation"
                and row["profile"] == "family_centered"
                and row["exact_upper_tail_p"] <= threshold_p
                and family_correct(row, family)
            ]
            transfer_models = []
            for model in protocol.MODELS:
                row = next(
                    value
                    for value in assignments
                    if value["comparison"]
                    == "controlled_to_natural_transfer"
                    and value["profile"] == "family_centered"
                    and value["source_model"] == model
                )
                if (
                    row["exact_upper_tail_p"] <= threshold_p
                    and family_correct(row, family)
                ):
                    transfer_models.append(model)
        behavior = behavior_models(summaries, family)
        l1 = len(operation_within) >= minimum
        l2 = l1 and len(cross_hits) >= minimum
        l3 = l2 and len(transfer_models) >= minimum
        l4 = l3 and len(behavior) >= minimum
        highest = (
            "L4" if l4
            else "L3" if l3
            else "L2" if l2
            else "L1" if l1
            else "L0"
        )
        if family == protocol.HELDOUT_FAMILY:
            highest = "L0"
        row = {
            "highest_evidence_level": highest,
            "within_model_operation_hits": operation_within,
            "cross_model_operation_hits": cross_hits,
            "controlled_to_natural_transfer_models": transfer_models,
            "behavior_annotation_models": behavior,
            "descriptive_status": (
                "output_independent_operation_and_natural_topology_repeated"
                if l3
                else "mapped_without_complete_output_independent_transfer"
            ),
            "causal_status": "not_tested",
            "retained_in_atlas": True,
        }
        family_evidence[family] = row
        evidence_rows.append({
            "schema_version": "phase1079_family_evidence.v1",
            "phase": protocol.PHASE,
            "family": family,
            **row,
        })

    required_predictions = ("P1", "P2", "P4", "P5", "P6")
    l3_base_count = sum(
        family_evidence[family]["highest_evidence_level"]
        in {"L3", "L4"}
        for family in protocol.BASE_FAMILIES
    )
    empirical_continue = (
        all(
            prediction_audit["predictions"][key]["passed"]
            for key in required_predictions
        )
        and l3_base_count >= 5
    )
    automatic_next = {
        "schema_version": "phase1079_automatic_next.v1",
        "phase": protocol.PHASE,
        "continue": empirical_continue,
        "integrity_audit_pending": True,
        "required_predictions": list(required_predictions),
        "l3_base_family_count": l3_base_count,
        "reason": (
            "The frozen empirical continuation gate passed; integrity "
            "audit remains required."
            if empirical_continue
            else "The frozen empirical continuation gate did not pass. "
            "Do not select components or neurons from response peaks."
        ),
        "recommended_next_task": (
            prereg["automatic_next"]["next_task_if_passed"]
            if empirical_continue
            else prereg["automatic_next"]["stop_if_failed"]
        ),
    }
    automatic_next["decision_digest"] = protocol.digest(automatic_next)

    hypothesis_audit = {
        "language_as_pattern_collection": {
            "status": "compatible_not_exhaustive",
            "reason": (
                "Nine families are operationalized, but no experiment "
                "proves that they form a complete language basis."
            ),
        },
        "relative_encoding": {
            "status": "directly_tested_as_conditional_differences",
            "reason": (
                "Operation is varied at fixed answer and answer is varied "
                "at fixed operation. This supports only an operational "
                "relative description."
            ),
        },
        "reuse_plus_minimal_difference": {
            "status": "reuse_tested_minimality_unmeasured",
            "reason": (
                "Cross-answer and cross-surface operation consistency "
                "measure reuse; no minimum-code proof is present."
            ),
        },
        "efficient_or_optimal_distribution": {
            "status": "unsupported",
            "reason": (
                "No comparative training, energy, capacity, robustness, "
                "or compression optimum is measured."
            ),
        },
        "unique_word_ecological_niche": {
            "status": "rare_word_behavior_and_topology_only",
            "reason": (
                "The rare-word family includes candidate-list-free natural "
                "behavior, but does not identify a complete physical niche "
                "for each token."
            ),
        },
        "joint_style_logic_grammar_selection": {
            "status": "partial_operation_output_orthogonalization",
            "reason": (
                "Operation and answer identity are separated here; style "
                "is not fully crossed with logic and syntax."
            ),
        },
        "small_model_roughness": {
            "status": "live_limit_not_a_posthoc_explanation",
            "reason": (
                "Architecture, tokenizer, data, and scale remain confounded "
                "across the three local models."
            ),
        },
    }

    analysis_root = protocol.OUT_ROOT / "analysis"
    assignment_payload = {
        "schema_version": "phase1079_assignment_collection.v1",
        "phase": protocol.PHASE,
        "rows": assignments,
    }
    assignment_payload["assignment_digest"] = protocol.digest(
        assignment_payload
    )
    protocol.write_json(
        analysis_root / "exact_assignments.json",
        assignment_payload,
    )
    protocol.write_json(
        analysis_root / "answer_alignment.json", alignment
    )
    protocol.write_json(
        analysis_root / "factor_ratios.json", factors
    )
    protocol.write_jsonl(
        analysis_root / "top_regions.jsonl", regions
    )
    protocol.write_json(
        analysis_root / "heldout_prediction.json", heldout
    )
    protocol.write_json(
        analysis_root / "prediction_audit.json", prediction_audit
    )
    protocol.write_jsonl(
        analysis_root / "family_evidence_ledger.jsonl",
        evidence_rows,
    )
    protocol.write_json(
        analysis_root / "automatic_next.json", automatic_next
    )

    final = {
        "schema_version": "phase1079_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "case_count_total": sum(
            int(row["case_count"]) for row in summaries.values()
        ),
        "unit_count_total": sum(
            int(row["unit_count"]) for row in summaries.values()
        ),
        "primary_population": prereg["primary_population"],
        "secondary_population": prereg["secondary_population"],
        "model_summaries": summaries,
        "family_evidence": family_evidence,
        "exact_assignment_summary": [{
            "comparison": row["comparison"],
            "field": row["field"],
            "profile": row["profile"],
            "source_model": row["source_model"],
            "target_model": row["target_model"],
            "top1_correct": row["top1_correct"],
            "family_count": row["family_count"],
            "exact_upper_tail_p": row["exact_upper_tail_p"],
            "identity_margin_over_best_other": row[
                "identity_margin_over_best_other"
            ],
        } for row in assignments],
        "answer_alignment": alignment,
        "factor_ratios": factors,
        "heldout_prediction": heldout,
        "prospective_prediction_audit": prediction_audit,
        "hypothesis_audit": hypothesis_audit,
        "mechanism_status": {
            family: {
                "observed": (
                    "Output-matched semantic-versus-index response field, "
                    "same-operation answer field, and candidate-list-free "
                    "natural answer field."
                ),
                "descriptive_evidence": row[
                    "highest_evidence_level"
                ],
                "not_established": (
                    "No head, neuron, transport edge, necessary path, "
                    "sufficient state, minimal code, or complete algorithm."
                ),
            }
            for family, row in family_evidence.items()
        },
        "mathematical_status": {
            "current_tools_sufficient_for": [
                "factorial conditional differences",
                "normalized-depth response topology",
                "exact label-permutation assignments",
                "output-matched operation controls",
                "natural-versus-controlled transfer tests",
            ],
            "not_yet_recovered": [
                "a complete language pattern ontology",
                "a predictive component-level transition law",
                "causal transport routes",
                "minimality or optimality",
                "brain-model homology",
            ],
            "new_mathematics_needed_now": False,
            "reason": (
                "Identification and protocol transfer remain the limiting "
                "problems. No empirical result currently requires a new "
                "mathematical primitive."
            ),
        },
        "hard_limits": list(prereg["interpretation_limits"]) + [
            (
                "The semantic/index routing shell can itself induce a "
                "generic mode-switch response."
            ),
            (
                "Candidate-list-free does not mean the target string is "
                "absent from source evidence in entity-selection tasks."
            ),
            (
                "Global exact-assignment significance can coexist with "
                "individual family confusion."
            ),
            (
                "Behavior-conditioned evidence remains secondary because "
                "conditioning on correctness changes the population."
            ),
        ],
        "automatic_next": automatic_next,
    }
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(analysis_root / "final_summary.json", final)
    print({
        "phase": protocol.PHASE,
        "status": "finalized",
        "case_count_total": final["case_count_total"],
        "passed_predictions": prediction_audit["passed_count"],
        "automatic_continue": empirical_continue,
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
