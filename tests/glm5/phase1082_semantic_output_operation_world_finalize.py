#!/usr/bin/env python3
"""Finalize Phase1082 revision 2 as descriptive, non-causal evidence."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1081_latin_route_atlas_finalize as base
import phase1082_semantic_output_operation_world_protocol as protocol


base.protocol = protocol
EPSILON = 1e-12
PRIMARY_ROLES = tuple(protocol.PRIMARY_PROFILE_ROLES)


def finite_mean(values: list[float]) -> float | None:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else None


def finite_median(values: list[float]) -> float | None:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(selected)) if selected else None


def row_normalize(values: np.ndarray, *, centered: bool) -> np.ndarray:
    output = values.astype(np.float64, copy=True)
    if centered:
        output -= output.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(
        output,
        norms,
        out=np.zeros_like(output),
        where=norms > EPSILON,
    )


def operation_profile(
    rows: list[dict[str, Any]],
    operation: str,
    worlds: tuple[str, ...],
    split: str,
    field: str,
) -> np.ndarray:
    profiles = [
        base.build_profile(
            rows,
            f"{operation}__{world}",
            split,
            field,
            roles=PRIMARY_ROLES,
        )
        for world in worlds
    ]
    return np.mean(np.stack(profiles), axis=0)


def operation_bank(
    rows: list[dict[str, Any]],
    worlds: tuple[str, ...],
    split: str,
    field: str,
    *,
    centered: bool,
) -> np.ndarray:
    values = np.stack([
        operation_profile(rows, operation, worlds, split, field)
        for operation in protocol.OPERATIONS
    ])
    return row_normalize(values, centered=centered)


def assignment(
    *,
    comparison: str,
    field: str,
    profile: str,
    source_model: str,
    target_model: str,
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_world: str | None = None,
    target_world: str | None = None,
) -> dict[str, Any]:
    row = base.assignment_record(
        comparison=comparison,
        field=field,
        profile=profile,
        source_model=source_model,
        target_model=target_model,
        families=tuple(protocol.OPERATIONS),
        source_values=source_values,
        target_values=target_values,
    )
    row["schema_version"] = "phase1082_operation_assignment.v2"
    row["source_world"] = source_world
    row["target_world"] = target_world
    for detail in row["rows"]:
        detail["operation"] = detail.pop("family")
        detail["predicted_operation"] = detail.pop("predicted_family")
        detail["best_other_operation"] = detail.pop("best_other_family")
    row["operations"] = row.pop("families")
    row["operation_count"] = row.pop("family_count")
    return row


def find_assignment(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    selected = [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(selected) != 1:
        raise RuntimeError(f"assignment lookup is not unique: {criteria}")
    return selected[0]


def correct_for(row: dict[str, Any], operation: str) -> bool:
    return bool(next(
        detail["correct"] for detail in row["rows"]
        if detail["operation"] == operation
    ))


def behavior_audit(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_threshold = float(
        protocol.EVIDENCE_THRESHOLDS[
            "candidate_accuracy_for_operation_behavior"
        ]
    )
    generation_threshold = float(
        protocol.EVIDENCE_THRESHOLDS[
            "generation_target_before_distractor_accuracy"
        ]
    )
    minimum_worlds = int(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_behavior_worlds_per_operation"
        ]
    )
    minimum_operations = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_behavior_operations"]
    )
    by_model: dict[str, Any] = {}
    for model_name, summary in summaries.items():
        cells: dict[str, Any] = {}
        operations: dict[str, Any] = {}
        for operation in protocol.OPERATIONS:
            passing_worlds = []
            for world in protocol.WORLDS:
                cell = f"{operation}__{world}"
                candidate_count = candidate_hits = 0
                generation_count = generation_hits = 0
                for split in protocol.SPLITS:
                    row = summary["behavior_summary"][cell][split]
                    candidate_count += int(row["active"]["candidate_count"])
                    candidate_hits += int(row["active"]["candidate_hit_count"])
                    generation = row["natural_generation"]
                    generation_count += int(generation["generation_case_count"])
                    generation_hits += int(
                        generation["generation_target_before_distractor_count"]
                    )
                candidate_accuracy = (
                    candidate_hits / candidate_count if candidate_count else None
                )
                generation_accuracy = (
                    generation_hits / generation_count if generation_count else None
                )
                passed = bool(
                    candidate_accuracy is not None
                    and candidate_accuracy >= candidate_threshold
                    and generation_accuracy is not None
                    and generation_accuracy >= generation_threshold
                )
                if passed:
                    passing_worlds.append(world)
                cells[cell] = {
                    "candidate_count": candidate_count,
                    "candidate_accuracy": candidate_accuracy,
                    "generation_count": generation_count,
                    "generation_accuracy": generation_accuracy,
                    "passed": passed,
                }
            operations[operation] = {
                "passing_worlds": passing_worlds,
                "passing_world_count": len(passing_worlds),
                "passed": len(passing_worlds) >= minimum_worlds,
            }
        passing_operations = [
            operation for operation, row in operations.items() if row["passed"]
        ]
        by_model[model_name] = {
            "cells": cells,
            "operations": operations,
            "passing_operations": passing_operations,
            "passing_operation_count": len(passing_operations),
            "passed": len(passing_operations) >= minimum_operations,
        }
    return {
        "schema_version": "phase1082_behavior_audit.v2",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "passing_models": [
            model for model, row in by_model.items() if row["passed"]
        ],
    }


def factor_ratio_audit(
    metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scopes = {
        "all_roles": set(protocol.CAPTURE_ROLES),
        "downstream_roles": set(protocol.PRIMARY_PROFILE_ROLES),
    }
    by_model: dict[str, Any] = {}
    for model_name, rows in metrics.items():
        model_scopes = {}
        for scope, roles in scopes.items():
            pooled = []
            by_operation = {}
            for operation in protocol.OPERATIONS:
                ratios = []
                for row in rows:
                    if (
                        row["conditioning"] != "all_finite"
                        or row["role"] not in roles
                        or not row["family"].startswith(f"{operation}__")
                    ):
                        continue
                    content = row["mean_content_route_relative_magnitude"]
                    output = row["mean_label_swap"]
                    shell = row["mean_shell"]
                    if (
                        content is None or float(content) <= EPSILON
                        or output is None or shell is None
                    ):
                        continue
                    ratio = max(float(output), float(shell)) / float(content)
                    if math.isfinite(ratio):
                        ratios.append(ratio)
                        pooled.append(ratio)
                by_operation[operation] = {
                    "median_max_control_to_content": finite_median(ratios),
                    "observation_count": len(ratios),
                }
            model_scopes[scope] = {
                "operations": by_operation,
                "pooled_median_max_control_to_content": finite_median(pooled),
                "observation_count": len(pooled),
            }
        by_model[model_name] = model_scopes
    threshold = float(
        protocol.EVIDENCE_THRESHOLDS["maximum_control_to_content_ratio"]
    )
    passing = [
        model for model, row in by_model.items()
        if row["all_roles"]["pooled_median_max_control_to_content"] is not None
        and row["all_roles"]["pooled_median_max_control_to_content"] <= threshold
    ]
    return {
        "schema_version": "phase1082_factor_ratio_audit.v2",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "passing_models": passing,
    }


def integrity_audit(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_model = {}
    for model_name, summary in summaries.items():
        candidate_total = int(summary["case_count"])
        candidate_finite_fraction = (
            1.0 - int(summary["nonfinite_candidate_count"]) / candidate_total
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
            / max(hidden_total, 1)
        )
        passed = bool(
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
            "hidden_finite_fraction_lower_bound": hidden_finite_fraction,
            "nonfinite_candidate_count": summary["nonfinite_candidate_count"],
            "nonfinite_hidden_count": summary[
                "nonfinite_hidden_magnitude_role_count"
            ],
            "identity_maximum": summary["identity_maximum"],
            "pre_query_global_max_abs": summary["pre_query_global_max_abs"],
            "passed": passed,
        }
    return {
        "schema_version": "phase1082_integrity_audit.v2",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "all_models_passed": all(row["passed"] for row in by_model.values()),
    }


def decomposition_audit(
    metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_model = {}
    for model_name, rows in metrics.items():
        tensor = np.stack([
            np.stack([
                operation_profile(
                    rows, operation, (world,), "confirmation", "content_route"
                )
                for world in protocol.WORLDS
            ])
            for operation in protocol.OPERATIONS
        ])
        grand = tensor.mean(axis=(0, 1), keepdims=True)
        operation_main = tensor.mean(axis=1, keepdims=True) - grand
        world_main = tensor.mean(axis=0, keepdims=True) - grand
        interaction = tensor - grand - operation_main - world_main
        energies = {
            "operation_main": float(np.square(operation_main).sum()),
            "world_main": float(np.square(world_main).sum()),
            "operation_world_interaction": float(np.square(interaction).sum()),
        }
        total = sum(energies.values())
        by_model[model_name] = {
            "energies": energies,
            "fractions": {
                key: value / total if total > EPSILON else None
                for key, value in energies.items()
            },
            "interpretation": (
                "Descriptive energy partition of normalized profiles; not an "
                "independence test or causal decomposition."
            ),
        }
    return {
        "schema_version": "phase1082_operation_world_decomposition.v2",
        "phase": protocol.PHASE,
        "by_model": by_model,
    }


def top_regions(
    metrics: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for model_name, rows in metrics.items():
        for operation in protocol.OPERATIONS:
            grouped: dict[tuple[str, int, float, str], list[float]] = defaultdict(list)
            for row in rows:
                if (
                    row["conditioning"] == "all_finite"
                    and row["split"] == "confirmation"
                    and row["role"] in PRIMARY_ROLES
                    and row["family"].startswith(f"{operation}__")
                    and row["mean_content_route_relative_magnitude"] is not None
                ):
                    key = (
                        row["component"], int(row["depth"]),
                        float(row["relative_depth"]), row["role"],
                    )
                    grouped[key].append(
                        float(row["mean_content_route_relative_magnitude"])
                    )
            ranked = sorted(
                (
                    (finite_mean(values), key, len(values))
                    for key, values in grouped.items()
                ),
                key=lambda value: float(value[0] or -math.inf),
                reverse=True,
            )[:5]
            for rank, (magnitude, key, count) in enumerate(ranked, 1):
                component, depth, relative_depth, role = key
                output.append({
                    "schema_version": "phase1082_top_region.v2",
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "operation": operation,
                    "rank": rank,
                    "component": component,
                    "depth": depth,
                    "relative_depth": relative_depth,
                    "role": role,
                    "mean_content_relative_magnitude": magnitude,
                    "world_split_observation_count": count,
                })
    return output


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    metrics = {
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
    assignments: list[dict[str, Any]] = []

    for model, rows in metrics.items():
        for field in ("content_route", "duplicate_route"):
            for centered in (False, True):
                assignments.append(assignment(
                    comparison="within_model_item_split",
                    field=field,
                    profile="operation_centered" if centered else "raw",
                    source_model=model,
                    target_model=model,
                    source_values=operation_bank(
                        rows, tuple(protocol.WORLDS), "discovery", field,
                        centered=centered,
                    ),
                    target_values=operation_bank(
                        rows, tuple(protocol.WORLDS), "confirmation", field,
                        centered=centered,
                    ),
                ))
        for centered in (False, True):
            assignments.append(assignment(
                comparison="within_model_output_vocabulary_transfer",
                field="content_label0__content_label1",
                profile="operation_centered" if centered else "raw",
                source_model=model,
                target_model=model,
                source_values=operation_bank(
                    rows, tuple(protocol.WORLDS), "confirmation",
                    "content_label0", centered=centered,
                ),
                target_values=operation_bank(
                    rows, tuple(protocol.WORLDS), "confirmation",
                    "content_label1", centered=centered,
                ),
            ))
        for heldout in protocol.WORLDS:
            source_worlds = tuple(
                world for world in protocol.WORLDS if world != heldout
            )
            for field in ("content_route", "duplicate_route"):
                assignments.append(assignment(
                    comparison="within_model_heldout_world",
                    field=field,
                    profile="operation_centered",
                    source_model=model,
                    target_model=model,
                    source_world="+".join(source_worlds),
                    target_world=heldout,
                    source_values=operation_bank(
                        rows, source_worlds, "discovery", field, centered=True
                    ),
                    target_values=operation_bank(
                        rows, (heldout,), "confirmation", field, centered=True
                    ),
                ))
        for source_world in protocol.WORLDS:
            for target_world in protocol.WORLDS:
                if source_world == target_world:
                    continue
                for field in ("content_route", "duplicate_route"):
                    assignments.append(assignment(
                        comparison="within_model_directed_cross_world",
                        field=field,
                        profile="operation_centered",
                        source_model=model,
                        target_model=model,
                        source_world=source_world,
                        target_world=target_world,
                        source_values=operation_bank(
                            rows, (source_world,), "discovery", field,
                            centered=True,
                        ),
                        target_values=operation_bank(
                            rows, (target_world,), "confirmation", field,
                            centered=True,
                        ),
                    ))

    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            for field in ("content_route", "duplicate_route"):
                assignments.append(assignment(
                    comparison="cross_model_world_averaged",
                    field=field,
                    profile="operation_centered",
                    source_model=source_model,
                    target_model=target_model,
                    source_values=operation_bank(
                        metrics[source_model], tuple(protocol.WORLDS),
                        "confirmation", field, centered=True,
                    ),
                    target_values=operation_bank(
                        metrics[target_model], tuple(protocol.WORLDS),
                        "confirmation", field, centered=True,
                    ),
                ))

    threshold_p = float(prereg["evidence_thresholds"]["permutation_p_max"])
    minimum_top1 = int(prereg["evidence_thresholds"]["minimum_operation_top1"])
    minimum_repeat = int(
        prereg["evidence_thresholds"]["minimum_repeated_models_or_pairs"]
    )
    behavior = behavior_audit(summaries)
    factors = factor_ratio_audit(metrics)
    integrity = integrity_audit(summaries)
    decomposition = decomposition_audit(metrics)
    regions = top_regions(metrics)

    def passes_assignment(row: dict[str, Any]) -> bool:
        return bool(
            row["top1_correct"] >= minimum_top1
            and row["exact_upper_tail_p"] <= threshold_p
        )

    within_models = []
    output_models = []
    heldout_by_model: dict[str, Any] = {}
    cross_world_by_model: dict[str, Any] = {}
    for model in protocol.MODELS:
        within = find_assignment(
            assignments,
            comparison="within_model_item_split",
            field="content_route",
            profile="operation_centered",
            source_model=model,
            target_model=model,
        )
        if passes_assignment(within):
            within_models.append(model)
        output = find_assignment(
            assignments,
            comparison="within_model_output_vocabulary_transfer",
            field="content_label0__content_label1",
            profile="operation_centered",
            source_model=model,
            target_model=model,
        )
        if passes_assignment(output):
            output_models.append(model)
        folds = []
        for world in protocol.WORLDS:
            row = find_assignment(
                assignments,
                comparison="within_model_heldout_world",
                field="content_route",
                source_model=model,
                target_model=model,
                target_world=world,
            )
            folds.append({
                "world": world,
                "top1_correct": row["top1_correct"],
                "exact_upper_tail_p": row["exact_upper_tail_p"],
                "passed": passes_assignment(row),
            })
        heldout_by_model[model] = {
            "folds": folds,
            "passing_fold_count": sum(int(row["passed"]) for row in folds),
        }
        pair_rows = []
        for source_world in protocol.WORLDS:
            for target_world in protocol.WORLDS:
                if source_world == target_world:
                    continue
                content = find_assignment(
                    assignments,
                    comparison="within_model_directed_cross_world",
                    field="content_route",
                    source_model=model,
                    target_model=model,
                    source_world=source_world,
                    target_world=target_world,
                )
                duplicate = find_assignment(
                    assignments,
                    comparison="within_model_directed_cross_world",
                    field="duplicate_route",
                    source_model=model,
                    target_model=model,
                    source_world=source_world,
                    target_world=target_world,
                )
                advantage = (
                    float(content["identity_mean_score"])
                    - float(duplicate["identity_mean_score"])
                )
                pair_rows.append({
                    "source_world": source_world,
                    "target_world": target_world,
                    "content_top1": content["top1_correct"],
                    "content_identity": content["identity_mean_score"],
                    "duplicate_identity": duplicate["identity_mean_score"],
                    "content_advantage": advantage,
                    "passed": advantage >= float(
                        prereg["evidence_thresholds"][
                            "minimum_cross_world_content_advantage"
                        ]
                    ),
                })
        cross_world_by_model[model] = {
            "pairs": pair_rows,
            "passing_advantage_pair_count": sum(
                int(row["passed"]) for row in pair_rows
            ),
        }

    cross_model_rows = []
    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            content = find_assignment(
                assignments,
                comparison="cross_model_world_averaged",
                field="content_route",
                source_model=source_model,
                target_model=target_model,
            )
            duplicate = find_assignment(
                assignments,
                comparison="cross_model_world_averaged",
                field="duplicate_route",
                source_model=source_model,
                target_model=target_model,
            )
            advantage = (
                float(content["identity_mean_score"])
                - float(duplicate["identity_mean_score"])
            )
            cross_model_rows.append({
                "source_model": source_model,
                "target_model": target_model,
                "content_top1": content["top1_correct"],
                "content_exact_p": content["exact_upper_tail_p"],
                "content_identity": content["identity_mean_score"],
                "duplicate_identity": duplicate["identity_mean_score"],
                "content_advantage": advantage,
                "retrieval_passed": passes_assignment(content),
                "advantage_passed": advantage >= float(
                    prereg["evidence_thresholds"][
                        "minimum_cross_model_content_advantage"
                    ]
                ),
            })

    p1 = bool(protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )["all_checks_passed"])
    p2 = len(behavior["passing_models"]) >= minimum_repeat
    p3 = len(within_models) >= minimum_repeat
    heldout_models = [
        model for model, row in heldout_by_model.items()
        if row["passing_fold_count"]
        >= prereg["evidence_thresholds"]["minimum_heldout_world_folds"]
    ]
    p4 = len(heldout_models) >= minimum_repeat
    p5 = len(output_models) >= minimum_repeat
    cross_world_models = [
        model for model, row in cross_world_by_model.items()
        if row["passing_advantage_pair_count"]
        >= prereg["evidence_thresholds"][
            "minimum_cross_world_advantage_pairs"
        ]
    ]
    p6 = len(cross_world_models) >= minimum_repeat
    p7 = len(factors["passing_models"]) >= minimum_repeat
    cross_model_retrieval = [
        row for row in cross_model_rows if row["retrieval_passed"]
    ]
    cross_model_advantage = [
        row for row in cross_model_rows if row["advantage_passed"]
    ]
    p8 = bool(
        len(cross_model_retrieval) >= minimum_repeat
        and len(cross_model_advantage)
        >= prereg["evidence_thresholds"][
            "minimum_cross_model_advantage_pairs"
        ]
    )
    p9 = bool(integrity["all_models_passed"])

    predictions = {
        "schema_version": "phase1082_prediction_audit.v2",
        "phase": protocol.PHASE,
        "predictions": {
            "P1": {"passed": p1},
            "P2": {"passed": p2, "passing_models": behavior["passing_models"]},
            "P3": {"passed": p3, "passing_models": within_models},
            "P4": {"passed": p4, "passing_models": heldout_models},
            "P5": {"passed": p5, "passing_models": output_models},
            "P6": {"passed": p6, "passing_models": cross_world_models},
            "P7": {"passed": p7, "passing_models": factors["passing_models"]},
            "P8": {
                "passed": p8,
                "retrieval_pair_count": len(cross_model_retrieval),
                "advantage_pair_count": len(cross_model_advantage),
            },
            "P9": {"passed": p9},
        },
        "passed_count": sum(map(int, (p1, p2, p3, p4, p5, p6, p7, p8, p9))),
        "total_count": 9,
    }

    operation_evidence = {}
    for operation in protocol.OPERATIONS:
        behavior_models = [
            model for model in protocol.MODELS
            if behavior["by_model"][model]["operations"][operation]["passed"]
        ]
        within_correct = []
        output_correct = []
        for model in protocol.MODELS:
            within = find_assignment(
                assignments,
                comparison="within_model_item_split",
                field="content_route",
                profile="operation_centered",
                source_model=model,
                target_model=model,
            )
            output = find_assignment(
                assignments,
                comparison="within_model_output_vocabulary_transfer",
                field="content_label0__content_label1",
                profile="operation_centered",
                source_model=model,
                target_model=model,
            )
            if correct_for(within, operation):
                within_correct.append(model)
            if correct_for(output, operation):
                output_correct.append(model)
        cross_model_correct = [
            f"{row['source_model']}->{row['target_model']}"
            for row in cross_model_rows
            if correct_for(find_assignment(
                assignments,
                comparison="cross_model_world_averaged",
                field="content_route",
                source_model=row["source_model"],
                target_model=row["target_model"],
            ), operation)
        ]
        level = "L0"
        if len(within_correct) >= 2:
            level = "L1"
        if level == "L1" and len(output_correct) >= 2 and p4:
            level = "L2"
        if level == "L2" and p6:
            level = "L3"
        if level == "L3" and len(cross_model_correct) >= 2:
            level = "L4"
        if level == "L4" and len(behavior_models) >= 2:
            level = "L5"
        operation_evidence[operation] = {
            "evidence_level": level,
            "behavior_models": behavior_models,
            "within_split_correct_models": within_correct,
            "output_vocabulary_correct_models": output_correct,
            "cross_model_correct_pairs": cross_model_correct,
            "causal_status": "not_tested",
        }

    l4_operations = [
        operation for operation, row in operation_evidence.items()
        if row["evidence_level"] in ("L4", "L5")
    ]
    all_predictions = all(
        row["passed"] for row in predictions["predictions"].values()
    )
    local_causal = all_predictions and len(l4_operations) >= 6
    global_atlas = p4 and p6 and p7 and not p8
    if local_causal:
        decision = "continue_to_preregistered_local_causal_validation"
    elif global_atlas:
        decision = "continue_global_atlas_with_cross_model_alignment"
    else:
        decision = "stop_hidden_escalation_and_diagnose_controls_or_transfer"
    automatic_next = {
        "schema_version": "phase1082_automatic_next.v2",
        "phase": protocol.PHASE,
        "decision": decision,
        "local_causal_authorized": local_causal,
        "global_atlas_authorized": global_atlas,
        "l4_or_higher_operations": l4_operations,
        "reason": (
            "Causal escalation requires all preregistered gates; descriptive "
            "mapping remains cumulative and is never deleted by a failed gate."
        ),
    }

    analysis_root = protocol.OUT_ROOT / "analysis"
    payloads = {
        "exact_assignments.json": {
            "schema_version": "phase1082_assignment_collection.v2",
            "phase": protocol.PHASE,
            "rows": assignments,
        },
        "behavior_audit.json": behavior,
        "factor_ratio_audit.json": factors,
        "integrity_audit.json": integrity,
        "operation_world_decomposition.json": decomposition,
        "prediction_audit.json": predictions,
        "operation_evidence.json": {
            "schema_version": "phase1082_operation_evidence.v2",
            "phase": protocol.PHASE,
            "operations": operation_evidence,
        },
        "heldout_world_audit.json": {
            "schema_version": "phase1082_heldout_world_audit.v2",
            "phase": protocol.PHASE,
            "by_model": heldout_by_model,
        },
        "cross_world_advantage.json": {
            "schema_version": "phase1082_cross_world_advantage.v2",
            "phase": protocol.PHASE,
            "by_model": cross_world_by_model,
        },
        "cross_model_audit.json": {
            "schema_version": "phase1082_cross_model_audit.v2",
            "phase": protocol.PHASE,
            "rows": cross_model_rows,
        },
        "automatic_next.json": automatic_next,
    }
    for filename, payload in payloads.items():
        payload[f"{filename.removesuffix('.json')}_digest"] = protocol.digest(payload)
        protocol.write_json(analysis_root / filename, payload)
    protocol.write_jsonl(analysis_root / "top_regions.jsonl", regions)

    final = {
        "schema_version": "phase1082_final_summary.v2",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "status": "complete_descriptive_noncausal",
        "model_order": list(protocol.MODELS),
        "case_count_total": sum(int(row["case_count"]) for row in summaries.values()),
        "unit_count_total": sum(int(row["unit_count"]) for row in summaries.values()),
        "predictions": predictions,
        "behavior": behavior,
        "within_split_passing_models": within_models,
        "heldout_world_passing_models": heldout_models,
        "output_vocabulary_passing_models": output_models,
        "cross_world_advantage_passing_models": cross_world_models,
        "factor_ratio_passing_models": factors["passing_models"],
        "cross_model": cross_model_rows,
        "integrity": integrity,
        "decomposition": decomposition,
        "operation_evidence": operation_evidence,
        "automatic_next": automatic_next,
        "interpretation_limits": prereg["interpretation_limits"] + [
            "Revision 1 arbitrary-code behavior results are not pooled with revision 2.",
            "Operation retrieval is descriptive identity, not a transported causal variable.",
            "A small content field does not prove minimal coding, optimal reuse, or a brain analogue.",
        ],
        "result_files": sorted(payloads) + ["top_regions.jsonl"],
    }
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(analysis_root / "final_summary.json", final)
    print({
        "phase": protocol.PHASE,
        "status": final["status"],
        "predictions": f"{predictions['passed_count']}/9",
        "decision": automatic_next["decision"],
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
