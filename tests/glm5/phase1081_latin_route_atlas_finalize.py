#!/usr/bin/env python3
"""Finalize Phase1081 as descriptive evidence, never causal evidence."""

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

import phase1081_latin_route_atlas_protocol as protocol


COMPONENTS = ("residual", "attention_output", "mlp_output")
DEPTH_GRID = np.linspace(0.0, 1.0, 13)
EPSILON = 1e-12
FIELD_COLUMNS = {
    "active_route": "mean_active_route_relative_magnitude",
    "duplicate_route": "mean_duplicate_route_relative_magnitude",
    "content_route": "mean_content_route_relative_magnitude",
    "content_label0": "mean_content_label0_relative_magnitude",
    "content_label1": "mean_content_label1_relative_magnitude",
    "answer": "mean_answer_relative_magnitude",
    "query_active": "mean_query_active_relative_magnitude",
    "query_duplicate": "mean_query_duplicate_relative_magnitude",
}


def safe_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def safe_median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


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
        row for row in rows
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
        by_depth[float(row["relative_depth"])].append(float(row[column]))
    depths = np.array(sorted(by_depth), dtype=np.float64)
    values = np.array(
        [float(np.mean(by_depth[depth])) for depth in depths],
        dtype=np.float64,
    )
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
        build_profile(rows, family, split, field, roles=roles)
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


def exact_assignment_test(matrix: np.ndarray) -> dict[str, Any]:
    size = int(matrix.shape[0])
    row_axis = np.arange(size)
    identity_score = float(np.trace(matrix) / size)
    scores_at_least_identity = 0
    permutation_count = 0
    best_other = -math.inf
    identity = tuple(range(size))
    for permutation in itertools.permutations(range(size)):
        score = float(matrix[row_axis, np.array(permutation)].mean())
        permutation_count += 1
        scores_at_least_identity += int(score >= identity_score - 1e-12)
        if permutation != identity:
            best_other = max(best_other, score)
    return {
        "identity_mean_score": identity_score,
        "permutation_count": permutation_count,
        "scores_at_least_identity": scores_at_least_identity,
        "exact_upper_tail_p": scores_at_least_identity / permutation_count,
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
    matrix = source_values @ target_values.T
    details = []
    for index, family in enumerate(families):
        order = np.argsort(-matrix[index])
        predicted = int(order[0])
        competitor = int(order[1] if predicted == index else predicted)
        details.append({
            "family": family,
            "predicted_family": families[predicted],
            "correct": predicted == index,
            "correct_similarity": float(matrix[index, index]),
            "best_other_family": families[competitor],
            "best_other_similarity": float(matrix[index, competitor]),
        })
    return {
        "schema_version": "phase1081_assignment.v1",
        "phase": protocol.PHASE,
        "comparison": comparison,
        "field": field,
        "profile": profile,
        "source_model": source_model,
        "target_model": target_model,
        "families": list(families),
        "family_count": len(families),
        "top1_correct": sum(int(row["correct"]) for row in details),
        "rows": details,
        "similarity_matrix": matrix.tolist(),
        **exact_assignment_test(matrix),
    }


def find_assignment(
    rows: list[dict[str, Any]], **criteria: Any
) -> dict[str, Any]:
    matches = [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"assignment lookup is not unique: {criteria}")
    return matches[0]


def family_correct(row: dict[str, Any], family: str) -> bool:
    return bool(next(
        value["correct"] for value in row["rows"]
        if value["family"] == family
    ))


def behavior_audit(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_threshold = float(
        protocol.EVIDENCE_THRESHOLDS[
            "candidate_accuracy_for_family_behavior"
        ]
    )
    generation_threshold = float(
        protocol.EVIDENCE_THRESHOLDS[
            "generation_target_before_distractor_accuracy"
        ]
    )
    by_model: dict[str, Any] = {}
    for model_name, summary in summaries.items():
        families: dict[str, Any] = {}
        for family in protocol.FAMILIES:
            candidate_count = candidate_hits = 0
            generation_count = generation_hits = 0
            for split in protocol.SPLITS:
                row = summary["behavior_summary"][family][split]
                candidate_count += int(row["active"]["candidate_count"])
                candidate_hits += int(row["active"]["candidate_hit_count"])
                generation = row["natural_generation"]
                generation_count += int(generation["generation_case_count"])
                generation_hits += int(
                    generation[
                        "generation_target_before_distractor_count"
                    ]
                )
            candidate_accuracy = (
                candidate_hits / candidate_count if candidate_count else None
            )
            generation_accuracy = (
                generation_hits / generation_count if generation_count else None
            )
            passed = (
                candidate_accuracy is not None
                and candidate_accuracy >= candidate_threshold
                and generation_accuracy is not None
                and generation_accuracy >= generation_threshold
            )
            families[family] = {
                "candidate_count": candidate_count,
                "candidate_accuracy": candidate_accuracy,
                "generation_count": generation_count,
                "generation_target_before_distractor_accuracy": generation_accuracy,
                "passed": passed,
            }
        base_passes = [
            family for family in protocol.BASE_FAMILIES
            if families[family]["passed"]
        ]
        by_model[model_name] = {
            "families": families,
            "passing_base_families": base_passes,
            "passing_base_family_count": len(base_passes),
        }
    minimum = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_behavior_families"]
    )
    passing_models = [
        model for model, row in by_model.items()
        if row["passing_base_family_count"] >= minimum
    ]
    return {
        "schema_version": "phase1081_behavior_audit.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "passing_models": passing_models,
    }


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
                content = row["mean_content_route_relative_magnitude"]
                label = row["mean_label_swap"]
                shell = row["mean_shell"]
                if (
                    content is None or float(content) <= EPSILON
                    or label is None or shell is None
                ):
                    continue
                ratio = max(float(label), float(shell)) / float(content)
                if math.isfinite(ratio):
                    ratios.append(ratio)
                    pooled.append(ratio)
            family_rows[family] = {
                "median_max_control_to_content": safe_median(ratios),
                "observation_count": len(ratios),
            }
        by_model[model_name] = {
            "families": family_rows,
            "pooled_median_max_control_to_content": safe_median(pooled),
        }
    threshold = float(
        protocol.EVIDENCE_THRESHOLDS[
            "maximum_control_to_content_ratio"
        ]
    )
    return {
        "schema_version": "phase1081_factor_ratios.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "passing_models": [
            model for model, row in by_model.items()
            if row["pooled_median_max_control_to_content"] is not None
            and row["pooled_median_max_control_to_content"] <= threshold
        ],
    }


def top_regions(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for model_name, rows in metrics_by_model.items():
        for family in protocol.FAMILIES:
            for field, column in FIELD_COLUMNS.items():
                candidates = [
                    row for row in rows
                    if row["conditioning"] == "all_finite"
                    and row["family"] == family
                    and row["split"] == "confirmation"
                    and row[column] is not None
                ]
                candidates.sort(key=lambda row: float(row[column]), reverse=True)
                for rank, row in enumerate(candidates[:5], 1):
                    output.append({
                        "schema_version": "phase1081_top_region.v1",
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
) -> dict[str, Any]:
    predicted = set(
        protocol.read_json(
            protocol.OUT_ROOT / "protocol" / "preregistration.json"
        )["heldout_prediction"]["nearest_family_set"]
    )
    by_model = {}
    comparison_families = protocol.BASE_FAMILIES + (
        protocol.HELDOUT_FAMILY,
    )
    for model_name, rows in metrics_by_model.items():
        values = profile_bank(
            rows,
            comparison_families,
            "confirmation",
            "content_route",
            centered=True,
        )
        heldout_index = comparison_families.index(protocol.HELDOUT_FAMILY)
        similarities = [
            (family, float(values[heldout_index] @ values[index]))
            for index, family in enumerate(comparison_families)
            if family != protocol.HELDOUT_FAMILY
        ]
        similarities.sort(key=lambda value: value[1], reverse=True)
        request_rows = [
            row for row in rows
            if row["conditioning"] == "all_finite"
            and row["family"] == protocol.HELDOUT_FAMILY
            and row["split"] == "confirmation"
            and row["role"] == "request_end"
            and row["mean_content_route_relative_magnitude"] is not None
        ]
        peak = max(
            request_rows,
            key=lambda row: float(
                row["mean_content_route_relative_magnitude"]
            ),
        )
        nearest_pass = similarities[0][0] in predicted
        peak_pass = (
            peak["component"] in {"attention_output", "mlp_output"}
            and 1 / 3 <= float(peak["relative_depth"]) <= 2 / 3
        )
        by_model[model_name] = {
            "nearest_base_family": similarities[0][0],
            "nearest_similarity": similarities[0][1],
            "all_base_similarities": [
                {"family": family, "similarity": similarity}
                for family, similarity in similarities
            ],
            "request_end_peak": {
                "component": peak["component"],
                "depth": peak["depth"],
                "relative_depth": peak["relative_depth"],
                "mean_relative_magnitude": peak[
                    "mean_content_route_relative_magnitude"
                ],
            },
            "nearest_prediction_passed": nearest_pass,
            "peak_prediction_passed": peak_pass,
            "joint_prediction_passed": nearest_pass and peak_pass,
        }
    return {
        "schema_version": "phase1081_heldout_audit.v1",
        "phase": protocol.PHASE,
        "heldout_family": protocol.HELDOUT_FAMILY,
        "predicted_nearest_family_set": sorted(predicted),
        "by_model": by_model,
        "passing_models": [
            model for model, row in by_model.items()
            if row["joint_prediction_passed"]
        ],
    }


def integrity_audit(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_model = {}
    for model_name, summary in summaries.items():
        candidate_total = int(summary["case_count"])
        candidate_finite_fraction = (
            1.0
            - int(summary["nonfinite_candidate_count"]) / candidate_total
        )
        hidden_total = (
            int(summary["unit_count"])
            * int(summary["event_count"])
            * len(protocol.CAPTURE_ROLES)
            * 36
        )
        hidden_finite_fraction = (
            1.0
            - int(summary["nonfinite_hidden_magnitude_role_count"])
            / hidden_total
        )
        passed = (
            candidate_finite_fraction
            >= protocol.EVIDENCE_THRESHOLDS[
                "minimum_candidate_finite_fraction"
            ]
            and hidden_finite_fraction
            >= protocol.EVIDENCE_THRESHOLDS[
                "minimum_hidden_finite_fraction"
            ]
            and float(summary["identity_maximum"]) <= 1e-8
            and float(summary["pre_query_global_max_abs"])
            <= protocol.EVIDENCE_THRESHOLDS["pre_query_tolerance"]
        )
        by_model[model_name] = {
            "candidate_finite_fraction": candidate_finite_fraction,
            "hidden_finite_fraction": hidden_finite_fraction,
            "identity_maximum": summary["identity_maximum"],
            "pre_query_global_max_abs": summary[
                "pre_query_global_max_abs"
            ],
            "passed": passed,
        }
    return {
        "schema_version": "phase1081_integrity_audit.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "all_models_passed": all(row["passed"] for row in by_model.values()),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    metrics_by_model = {
        model: protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        for model in protocol.MODELS
    }
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    families = protocol.BASE_FAMILIES
    assignments: list[dict[str, Any]] = []

    for model_name, rows in metrics_by_model.items():
        for field in (
            "active_route",
            "duplicate_route",
            "content_route",
            "answer",
            "query_duplicate",
        ):
            for centered in (False, True):
                assignments.append(assignment_record(
                    comparison="within_model_discovery_to_confirmation",
                    field=field,
                    profile="family_centered" if centered else "raw",
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
        for centered in (False, True):
            assignments.append(assignment_record(
                comparison="within_model_label_assignment_transfer",
                field="content_label0__content_label1",
                profile="family_centered" if centered else "raw",
                source_model=model_name,
                target_model=model_name,
                families=families,
                source_values=profile_bank(
                    rows, families, "confirmation", "content_label0",
                    centered=centered,
                ),
                target_values=profile_bank(
                    rows, families, "confirmation", "content_label1",
                    centered=centered,
                ),
            ))

    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            for field in ("content_route", "duplicate_route", "answer"):
                for centered in (False, True):
                    assignments.append(assignment_record(
                        comparison="cross_model_confirmation",
                        field=field,
                        profile="family_centered" if centered else "raw",
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

    threshold_p = float(
        prereg["evidence_thresholds"]["permutation_p_max"]
    )
    minimum_top1 = int(
        prereg["evidence_thresholds"]["minimum_base_family_top1"]
    )
    minimum_repeat = int(
        prereg["evidence_thresholds"][
            "minimum_repeated_models_or_pairs"
        ]
    )
    behavior = behavior_audit(summaries)
    factors = factor_ratios(metrics_by_model)
    regions = top_regions(metrics_by_model)
    heldout = heldout_audit(metrics_by_model)
    integrity = integrity_audit(summaries)

    within_models = []
    label_models = []
    for model in protocol.MODELS:
        within = find_assignment(
            assignments,
            comparison="within_model_discovery_to_confirmation",
            field="content_route",
            profile="family_centered",
            source_model=model,
        )
        if (
            within["exact_upper_tail_p"] <= threshold_p
            and within["top1_correct"] >= minimum_top1
        ):
            within_models.append(model)
        label = find_assignment(
            assignments,
            comparison="within_model_label_assignment_transfer",
            field="content_label0__content_label1",
            profile="family_centered",
            source_model=model,
        )
        if (
            label["exact_upper_tail_p"] <= threshold_p
            and label["top1_correct"] >= minimum_top1
        ):
            label_models.append(model)

    cross_pairs = []
    advantage_rows = []
    advantage_threshold = float(
        prereg["evidence_thresholds"][
            "minimum_cross_model_content_advantage"
        ]
    )
    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            content = find_assignment(
                assignments,
                comparison="cross_model_confirmation",
                field="content_route",
                profile="family_centered",
                source_model=source_model,
                target_model=target_model,
            )
            duplicate = find_assignment(
                assignments,
                comparison="cross_model_confirmation",
                field="duplicate_route",
                profile="family_centered",
                source_model=source_model,
                target_model=target_model,
            )
            pair = f"{source_model}__{target_model}"
            if (
                content["exact_upper_tail_p"] <= threshold_p
                and content["top1_correct"] >= minimum_top1
            ):
                cross_pairs.append(pair)
            advantage = (
                float(content["identity_mean_score"])
                - float(duplicate["identity_mean_score"])
            )
            advantage_rows.append({
                "pair": pair,
                "content_identity_score": content["identity_mean_score"],
                "duplicate_identity_score": duplicate["identity_mean_score"],
                "content_advantage": advantage,
                "passed": advantage >= advantage_threshold,
            })
    advantage_pairs = [
        row["pair"] for row in advantage_rows if row["passed"]
    ]

    p1 = bool(
        protocol.read_json(
            protocol.OUT_ROOT / "protocol" / "audit.json"
        )["all_checks_passed"]
    )
    prediction_audit = {
        "schema_version": "phase1081_prediction_audit.v1",
        "phase": protocol.PHASE,
        "predictions": {
            "P1": {"passed": p1},
            "P2": {
                "passed": len(behavior["passing_models"]) >= minimum_repeat,
                "passing_models": behavior["passing_models"],
            },
            "P3": {
                "passed": len(within_models) >= minimum_repeat,
                "passing_models": within_models,
            },
            "P4": {
                "passed": len(cross_pairs) >= minimum_repeat,
                "passing_directed_pairs": cross_pairs,
            },
            "P5": {
                "passed": len(advantage_pairs) >= int(
                    prereg["evidence_thresholds"][
                        "minimum_content_advantage_pairs"
                    ]
                ),
                "passing_directed_pairs": advantage_pairs,
                "rows": advantage_rows,
            },
            "P6": {
                "passed": len(label_models) >= minimum_repeat,
                "passing_models": label_models,
            },
            "P7": {
                "passed": len(factors["passing_models"]) >= minimum_repeat,
                "passing_models": factors["passing_models"],
            },
            "P8": {
                "passed": len(heldout["passing_models"]) >= minimum_repeat,
                "passing_models": heldout["passing_models"],
            },
            "P9": {
                "passed": integrity["all_models_passed"],
                "by_model": integrity["by_model"],
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

    evidence_rows = []
    family_evidence: dict[str, Any] = {}
    for family in protocol.FAMILIES:
        if family in protocol.BASE_FAMILIES:
            within_hits = []
            label_hits = []
            for model in protocol.MODELS:
                within = find_assignment(
                    assignments,
                    comparison="within_model_discovery_to_confirmation",
                    field="content_route",
                    profile="family_centered",
                    source_model=model,
                )
                if (
                    within["exact_upper_tail_p"] <= threshold_p
                    and family_correct(within, family)
                ):
                    within_hits.append(model)
                label = find_assignment(
                    assignments,
                    comparison="within_model_label_assignment_transfer",
                    field="content_label0__content_label1",
                    profile="family_centered",
                    source_model=model,
                )
                if (
                    label["exact_upper_tail_p"] <= threshold_p
                    and family_correct(label, family)
                ):
                    label_hits.append(model)
            family_cross = []
            for row in assignments:
                if (
                    row["comparison"] == "cross_model_confirmation"
                    and row["field"] == "content_route"
                    and row["profile"] == "family_centered"
                    and row["exact_upper_tail_p"] <= threshold_p
                    and family_correct(row, family)
                ):
                    family_cross.append(
                        f"{row['source_model']}__{row['target_model']}"
                    )
        else:
            within_hits = []
            label_hits = []
            family_cross = []
        behavior_models = [
            model for model in protocol.MODELS
            if behavior["by_model"][model]["families"][family]["passed"]
        ]
        l1 = len(within_hits) >= minimum_repeat
        l2 = l1 and len(label_hits) >= minimum_repeat
        l3 = (
            l2
            and len(family_cross) >= minimum_repeat
            and len(advantage_pairs) >= int(
                prereg["evidence_thresholds"][
                    "minimum_content_advantage_pairs"
                ]
            )
        )
        l4 = l3 and len(behavior_models) >= minimum_repeat
        highest = (
            "L4" if l4 else "L3" if l3 else "L2" if l2
            else "L1" if l1 else "L0"
        )
        row = {
            "highest_evidence_level": highest,
            "within_model_content_hits": within_hits,
            "cross_label_hits": label_hits,
            "cross_model_content_hits": family_cross,
            "behavior_annotation_models": behavior_models,
            "descriptive_status": "mapped_retained",
            "causal_status": "not_tested",
            "retained_in_atlas": True,
        }
        family_evidence[family] = row
        evidence_rows.append({
            "schema_version": "phase1081_family_evidence.v1",
            "phase": protocol.PHASE,
            "family": family,
            **row,
        })

    l3_base_count = sum(
        family_evidence[family]["highest_evidence_level"] in {"L3", "L4"}
        for family in protocol.BASE_FAMILIES
    )
    empirical_continue = (
        all(
            prediction_audit["predictions"][f"P{index}"]["passed"]
            for index in range(1, 10)
        )
        and l3_base_count >= 5
    )
    automatic_next = {
        "schema_version": "phase1081_automatic_next.v1",
        "phase": protocol.PHASE,
        "continue_to_local_causal_sampling": empirical_continue,
        "continue_global_atlas": True,
        "l3_base_family_count": l3_base_count,
        "reason": (
            "All frozen descriptive, behavior, held-out, and integrity gates passed."
            if empirical_continue
            else "At least one frozen gate failed. Retain and diagnose the atlas; do not choose neurons or causal edges from peaks."
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
            "status": "compatible_not_complete",
            "reason": "Eleven operational families are mapped; they are not a basis for all language.",
        },
        "relational_encoding": {
            "status": "directly_tested_as_matched_differences",
            "reason": "Content routing is estimated as an active-minus-duplicate difference under output matching.",
        },
        "reuse_plus_minimal_difference": {
            "status": "reuse_tested_minimality_unmeasured",
            "reason": "Cross-label and cross-model repetition test reuse; no minimum code is proven.",
        },
        "efficient_or_optimal_distribution": {
            "status": "unsupported",
            "reason": "No energy, compression, capacity, robustness, or training comparison measures optimality.",
        },
        "unique_word_ecological_niche": {
            "status": "rare_family_only",
            "reason": "Rare meanings are included, but no complete token-specific physical niche is identified.",
        },
        "joint_style_logic_grammar_selection": {
            "status": "partial",
            "reason": "Semantic, relation, translation, tense, contrast, and punctuation tasks are observed, but style is not fully crossed.",
        },
        "small_model_roughness": {
            "status": "measured_as_cross_model_heterogeneity",
            "reason": "Behavior and normalized topology differ across models; architecture, tokenizer, scale, and data remain confounded.",
        },
    }

    analysis_root = protocol.OUT_ROOT / "analysis"
    assignment_payload = {
        "schema_version": "phase1081_assignment_collection.v1",
        "phase": protocol.PHASE,
        "rows": assignments,
    }
    assignment_payload["assignment_digest"] = protocol.digest(
        assignment_payload
    )
    protocol.write_json(
        analysis_root / "exact_assignments.json", assignment_payload
    )
    protocol.write_json(analysis_root / "behavior_audit.json", behavior)
    protocol.write_json(analysis_root / "factor_ratios.json", factors)
    protocol.write_jsonl(analysis_root / "top_regions.jsonl", regions)
    protocol.write_json(
        analysis_root / "heldout_prediction.json", heldout
    )
    protocol.write_json(
        analysis_root / "integrity_audit.json", integrity
    )
    protocol.write_json(
        analysis_root / "prediction_audit.json", prediction_audit
    )
    protocol.write_jsonl(
        analysis_root / "family_evidence_ledger.jsonl", evidence_rows
    )
    protocol.write_json(
        analysis_root / "automatic_next.json", automatic_next
    )

    final = {
        "schema_version": "phase1081_final_summary.v1",
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
        "behavior_audit": behavior,
        "family_evidence": family_evidence,
        "exact_assignment_summary": [{
            "comparison": row["comparison"],
            "field": row["field"],
            "profile": row["profile"],
            "source_model": row["source_model"],
            "target_model": row["target_model"],
            "top1_correct": row["top1_correct"],
            "family_count": row["family_count"],
            "identity_mean_score": row["identity_mean_score"],
            "exact_upper_tail_p": row["exact_upper_tail_p"],
        } for row in assignments],
        "factor_ratios": factors,
        "heldout_prediction": heldout,
        "integrity_audit": integrity,
        "prospective_prediction_audit": prediction_audit,
        "hypothesis_audit": hypothesis_audit,
        "mechanism_status": {
            family: {
                "observed": "Output-matched active route, matched duplicate route, and their difference-in-differences topology.",
                "descriptive_evidence": row["highest_evidence_level"],
                "not_established": "No causal edge, necessary path, sufficient state, neuron code, minimal code, or complete algorithm.",
            }
            for family, row in family_evidence.items()
        },
        "mathematical_status": {
            "current_tools_sufficient_for": [
                "matched factorial differences",
                "normalized-depth response topology",
                "exact finite permutation tests",
                "negative-control subtraction",
            ],
            "not_yet_recovered": [
                "complete language ontology",
                "predictive component transition law",
                "causal transport route",
                "minimality or optimality",
                "brain-model homology",
            ],
            "new_mathematics_needed_now": False,
            "reason": "Identification and behavioral validity remain limiting; the present evidence does not force a new mathematical primitive.",
        },
        "hard_limits": list(prereg["interpretation_limits"]) + [
            "Difference-in-differences removes a matched duplicate baseline only under an approximate additivity assumption.",
            "Family retrieval can still contain answer-pair and cloze-task structure.",
            "Cross-model normalized depth is functional similarity, not coordinate homology.",
            "A failed behavior gate makes a mapped topology uninterpretable as successful task computation.",
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
        "automatic_causal_continue": empirical_continue,
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
