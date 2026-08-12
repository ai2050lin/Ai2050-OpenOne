#!/usr/bin/env python3
"""Finalize Phase1078 without converting response repetition into causality."""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1078_shared_shell_pattern_atlas_protocol as protocol


GRID = np.linspace(0.0, 1.0, 21, dtype=np.float64)
PROFILE_METRICS = (
    "mean_truth_relative_magnitude",
    "mean_surface_relative_magnitude",
    "mean_shell_relative_magnitude",
    "mean_truth_surface_interaction",
    "mean_truth_shell_interaction",
    "truth_direction_consistency",
    "mean_truth_cross_surface_cosine",
    "mean_truth_cross_shell_cosine",
)
COMPONENTS = ("residual", "attention_output", "mlp_output")
MODEL_PAIRS = tuple(itertools.combinations(protocol.MODELS, 2))
EPSILON = 1e-12


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(
        np.linalg.norm(left.astype(np.float64))
        * np.linalg.norm(right.astype(np.float64))
    )
    if denominator <= EPSILON:
        return None
    return float(np.dot(
        left.astype(np.float64),
        right.astype(np.float64),
    ) / denominator)


def safe_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def safe_median(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def channel_normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > EPSILON else values


def interpolate_channel(
    rows: list[dict[str, Any]],
    component: str,
    role: str,
    metric: str,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["component"] == component
        and row["role"] == role
        and row.get(metric) is not None
    ]
    if not selected:
        return np.zeros_like(GRID)
    points: dict[float, list[float]] = defaultdict(list)
    for row in selected:
        value = float(row[metric])
        if math.isfinite(value):
            points[float(row["relative_depth"])].append(value)
    if not points:
        return np.zeros_like(GRID)
    xs = sorted(points)
    ys = [float(np.mean(points[x])) for x in xs]
    if component != "residual" and xs[0] > 0.0:
        xs = [0.0, *xs]
        ys = [0.0, *ys]
    return np.interp(GRID, xs, ys)


def build_profile(
    rows: list[dict[str, Any]],
    *,
    family: str,
    split: str,
    conditioning: str = "all_finite",
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and row["split"] == split
        and row["conditioning"] == conditioning
    ]
    channels = []
    for metric in PROFILE_METRICS:
        values = []
        for component in COMPONENTS:
            for role in protocol.CAPTURE_ROLES:
                values.append(interpolate_channel(
                    selected,
                    component,
                    role,
                    metric,
                ))
        channels.append(channel_normalize(np.concatenate(values)))
    return np.concatenate(channels).astype(np.float64)


def depth_band(relative_depth: float) -> str:
    if relative_depth < 0.30:
        return "early"
    if relative_depth < 0.70:
        return "middle"
    return "late"


def behavior_annotation_models(
    summaries: dict[str, dict[str, Any]],
    family: str,
) -> list[str]:
    return [
        model
        for model, summary in summaries.items()
        if summary["families"][family]["behavior_annotation_passed"]
    ]


def similarity_matrix(
    profiles: dict[tuple[str, str, str], np.ndarray],
    *,
    source_model: str,
    source_split: str,
    target_model: str,
    target_split: str,
) -> np.ndarray:
    matrix = np.zeros(
        (len(protocol.FAMILIES), len(protocol.FAMILIES)),
        dtype=np.float64,
    )
    for left_index, left in enumerate(protocol.FAMILIES):
        for right_index, right in enumerate(protocol.FAMILIES):
            value = cosine(
                profiles[(source_model, left, source_split)],
                profiles[(target_model, right, target_split)],
            )
            matrix[left_index, right_index] = (
                value if value is not None else -1.0
            )
    return matrix


def exact_assignment_test(matrix: np.ndarray) -> dict[str, Any]:
    family_count = len(protocol.FAMILIES)
    observed = float(np.mean(np.diag(matrix)))
    permutation_values = []
    for permutation in itertools.permutations(range(family_count)):
        permutation_values.append(float(np.mean([
            matrix[index, permutation[index]]
            for index in range(family_count)
        ])))
    null = np.asarray(permutation_values, dtype=np.float64)
    exceed = int(np.sum(null >= observed - 1e-12))
    predictions = np.argmax(matrix, axis=1)
    rows = []
    margins = []
    for index, family in enumerate(protocol.FAMILIES):
        alternatives = np.delete(matrix[index], index)
        margin = float(matrix[index, index] - np.max(alternatives))
        margins.append(margin)
        rows.append({
            "family": family,
            "prediction": protocol.FAMILIES[int(predictions[index])],
            "correct": int(predictions[index]) == index,
            "same_family_score": float(matrix[index, index]),
            "margin_over_best_other": margin,
        })
    return {
        "observed_diagonal_mean": observed,
        "exact_permutation_count": int(null.size),
        "exact_upper_tail_p": exceed / int(null.size),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_max": float(np.max(null)),
        "top1_correct": int(np.sum(
            predictions == np.arange(family_count)
        )),
        "family_count": family_count,
        "top1_accuracy": float(np.mean(
            predictions == np.arange(family_count)
        )),
        "mean_margin_over_best_other": float(np.mean(margins)),
        "minimum_margin_over_best_other": float(np.min(margins)),
        "rows": rows,
    }


def assignment_row(
    profiles: dict[tuple[str, str, str], np.ndarray],
    *,
    profile_name: str,
    comparison: str,
    source_model: str,
    source_split: str,
    target_model: str,
    target_split: str,
) -> dict[str, Any]:
    matrix = similarity_matrix(
        profiles,
        source_model=source_model,
        source_split=source_split,
        target_model=target_model,
        target_split=target_split,
    )
    return {
        "schema_version": "phase1078_exact_assignment.v1",
        "phase": protocol.PHASE,
        "profile": profile_name,
        "comparison": comparison,
        "source_model": source_model,
        "source_split": source_split,
        "target_model": target_model,
        "target_split": target_split,
        "similarity_matrix": matrix.tolist(),
        **exact_assignment_test(matrix),
    }


def build_consensus_regions(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    aggregate: dict[
        tuple[str, str, str, str],
        dict[str, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    for model_name, rows in metrics_by_model.items():
        selected = [
            row
            for row in rows
            if row["conditioning"] == "all_finite"
            and row["split"] == "confirmation"
            and row["mean_truth_relative_magnitude"] is not None
        ]
        maxima = defaultdict(float)
        for row in selected:
            maxima[row["family"]] = max(
                maxima[row["family"]],
                float(row["mean_truth_relative_magnitude"]),
            )
        local: dict[
            tuple[str, str, str, str],
            list[float],
        ] = defaultdict(list)
        for row in selected:
            family = str(row["family"])
            maximum = maxima[family]
            normalized = (
                float(row["mean_truth_relative_magnitude"]) / maximum
                if maximum > EPSILON
                else 0.0
            )
            key = (
                family,
                str(row["component"]),
                str(row["role"]),
                depth_band(float(row["relative_depth"])),
            )
            local[key].append(normalized)
        for key, values in local.items():
            aggregate[key][model_name].append(float(np.mean(values)))

    output = []
    for key, by_model in aggregate.items():
        family, component, role, band = key
        model_scores = {
            model: float(np.mean(values))
            for model, values in by_model.items()
        }
        output.append({
            "schema_version": "phase1078_consensus_region.v1",
            "phase": protocol.PHASE,
            "family": family,
            "component": component,
            "role": role,
            "depth_band": band,
            "normalized_response_by_model": model_scores,
            "model_count": len(model_scores),
            "mean_normalized_response": safe_mean(
                list(model_scores.values())
            ),
            "minimum_normalized_response": (
                min(model_scores.values()) if model_scores else None
            ),
        })
    ranked = []
    for family in protocol.FAMILIES:
        values = [row for row in output if row["family"] == family]
        values.sort(
            key=lambda row: (
                row["model_count"],
                row["minimum_normalized_response"] or 0.0,
                row["mean_normalized_response"] or 0.0,
            ),
            reverse=True,
        )
        for rank, row in enumerate(values[:10], 1):
            row["family_rank"] = rank
            ranked.append(row)
    return ranked


def factor_control_ratios(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_model = {}
    for model_name, rows in metrics_by_model.items():
        by_family = {}
        pooled = []
        for family in protocol.FAMILIES:
            selected = [
                row
                for row in rows
                if row["conditioning"] == "all_finite"
                and row["split"] == "confirmation"
                and row["family"] == family
                and row["mean_truth_relative_magnitude"] is not None
                and float(row["mean_truth_relative_magnitude"]) > EPSILON
            ]
            surface_ratios = [
                float(row["mean_surface_relative_magnitude"])
                / float(row["mean_truth_relative_magnitude"])
                for row in selected
                if row["mean_surface_relative_magnitude"] is not None
            ]
            shell_ratios = [
                float(row["mean_shell_relative_magnitude"])
                / float(row["mean_truth_relative_magnitude"])
                for row in selected
                if row["mean_shell_relative_magnitude"] is not None
            ]
            maximum_ratios = [
                max(surface, shell)
                for surface, shell in zip(
                    surface_ratios,
                    shell_ratios,
                )
            ]
            pooled.extend(maximum_ratios)
            by_family[family] = {
                "median_surface_to_truth": safe_median(surface_ratios),
                "median_shell_to_truth": safe_median(shell_ratios),
                "median_max_control_to_truth": safe_median(
                    maximum_ratios
                ),
                "observation_count": len(maximum_ratios),
            }
        by_model[model_name] = {
            "families": by_family,
            "pooled_median_max_control_to_truth": safe_median(pooled),
        }
    threshold = protocol.EVIDENCE_THRESHOLDS[
        "control_to_truth_ratio_nontrivial_min"
    ]
    nontrivial_models = [
        model
        for model, row in by_model.items()
        if row["pooled_median_max_control_to_truth"] is not None
        and row["pooled_median_max_control_to_truth"] >= threshold
    ]
    return {
        "schema_version": "phase1078_factor_control_ratios.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "nontrivial_threshold": threshold,
        "nontrivial_models": nontrivial_models,
    }


def generic_truth_alignment() -> dict[str, Any]:
    by_model = {}
    for model_name in protocol.MODELS:
        path = (
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "residual_mean_truth_directions.fp16.npz"
        )
        arrays = np.load(path)
        directions = arrays["mean_directions"].astype(np.float64)
        counts = arrays["counts"]
        depths = arrays["relative_depth"].astype(np.float64)
        # all_finite, confirmation, answer_boundary
        vectors = directions[
            0,
            :,
            1,
            :,
            protocol.CAPTURE_ROLES.index("answer_boundary"),
            :,
        ]
        count_values = counts[
            0,
            :,
            1,
            :,
            protocol.CAPTURE_ROLES.index("answer_boundary"),
        ]
        depth_rows = []
        for depth_index, relative_depth in enumerate(depths):
            values = []
            for left, right in itertools.combinations(
                range(len(protocol.FAMILIES)),
                2,
            ):
                if (
                    count_values[left, depth_index] <= 0
                    or count_values[right, depth_index] <= 0
                ):
                    continue
                score = cosine(
                    vectors[left, depth_index],
                    vectors[right, depth_index],
                )
                if score is not None:
                    values.append(score)
            depth_rows.append({
                "relative_depth": float(relative_depth),
                "mean_cross_family_truth_direction_cosine": (
                    safe_mean(values)
                ),
                "pair_count": len(values),
            })
        early = [
            row["mean_cross_family_truth_direction_cosine"]
            for row in depth_rows
            if row["relative_depth"] < 0.30
            and row["mean_cross_family_truth_direction_cosine"]
            is not None
        ]
        late = [
            row["mean_cross_family_truth_direction_cosine"]
            for row in depth_rows
            if row["relative_depth"] >= 0.70
            and row["mean_cross_family_truth_direction_cosine"]
            is not None
        ]
        early_mean = safe_mean(early)
        late_mean = safe_mean(late)
        gap = (
            late_mean - early_mean
            if late_mean is not None and early_mean is not None
            else None
        )
        by_model[model_name] = {
            "early_mean": early_mean,
            "late_mean": late_mean,
            "late_minus_early": gap,
            "depth_profile": depth_rows,
        }
    threshold = protocol.EVIDENCE_THRESHOLDS[
        "late_truth_alignment_gap_min"
    ]
    passing = [
        model
        for model, row in by_model.items()
        if row["late_minus_early"] is not None
        and row["late_minus_early"] >= threshold
    ]
    return {
        "schema_version": "phase1078_generic_truth_alignment.v1",
        "phase": protocol.PHASE,
        "by_model": by_model,
        "gap_threshold": threshold,
        "passing_models": passing,
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1078 protocol audit failed")

    summaries = {}
    metrics_by_model = {}
    split_directions_by_model = {}
    profiles: dict[tuple[str, str, str], np.ndarray] = {}
    for model_name in protocol.MODELS:
        atlas_root = protocol.OUT_ROOT / "atlas" / model_name
        summaries[model_name] = protocol.read_json(
            atlas_root / "summary.json"
        )
        if (
            summaries[model_name]["protocol_digest"]
            != prereg["protocol_digest"]
        ):
            raise RuntimeError(f"protocol drift for {model_name}")
        metrics_by_model[model_name] = protocol.read_jsonl(
            atlas_root / "response_metrics.jsonl"
        )
        split_directions_by_model[model_name] = protocol.read_jsonl(
            atlas_root / "split_direction_repeat.jsonl"
        )
        for family in protocol.FAMILIES:
            for split in protocol.SPLITS:
                profiles[(model_name, family, split)] = build_profile(
                    metrics_by_model[model_name],
                    family=family,
                    split=split,
                )

    centered_profiles = {}
    for model_name in protocol.MODELS:
        for split in protocol.SPLITS:
            matrix = np.stack([
                profiles[(model_name, family, split)]
                for family in protocol.FAMILIES
            ])
            mean = matrix.mean(axis=0)
            for family_index, family in enumerate(protocol.FAMILIES):
                centered_profiles[
                    (model_name, family, split)
                ] = matrix[family_index] - mean

    assignment_rows = []
    for model_name in protocol.MODELS:
        for profile_name, source_profiles in (
            ("raw", profiles),
            ("family_centered", centered_profiles),
        ):
            assignment_rows.append(assignment_row(
                source_profiles,
                profile_name=profile_name,
                comparison="within_model_discovery_to_confirmation",
                source_model=model_name,
                source_split="discovery",
                target_model=model_name,
                target_split="confirmation",
            ))
    for left, right in MODEL_PAIRS:
        for source_model, target_model in (
            (left, right),
            (right, left),
        ):
            for profile_name, source_profiles in (
                ("raw", profiles),
                ("family_centered", centered_profiles),
            ):
                assignment_rows.append(assignment_row(
                    source_profiles,
                    profile_name=profile_name,
                    comparison="cross_model_confirmation",
                    source_model=source_model,
                    source_split="confirmation",
                    target_model=target_model,
                    target_split="confirmation",
                ))

    threshold_p = float(
        prereg["evidence_thresholds"]["permutation_p_max"]
    )
    minimum = int(
        prereg["evidence_thresholds"][
            "minimum_repeated_models_or_pairs"
        ]
    )
    family_evidence = {}
    evidence_rows = []
    for family in protocol.FAMILIES:
        within_hits = [
            row["source_model"]
            for row in assignment_rows
            if row["comparison"]
            == "within_model_discovery_to_confirmation"
            and row["profile"] == "family_centered"
            and row["exact_upper_tail_p"] <= threshold_p
            and next(
                value
                for value in row["rows"]
                if value["family"] == family
            )["correct"]
        ]
        raw_cross_hits = [
            f"{row['source_model']}__{row['target_model']}"
            for row in assignment_rows
            if row["comparison"] == "cross_model_confirmation"
            and row["profile"] == "raw"
            and row["exact_upper_tail_p"] <= threshold_p
            and next(
                value
                for value in row["rows"]
                if value["family"] == family
            )["correct"]
        ]
        centered_cross_hits = [
            f"{row['source_model']}__{row['target_model']}"
            for row in assignment_rows
            if row["comparison"] == "cross_model_confirmation"
            and row["profile"] == "family_centered"
            and row["exact_upper_tail_p"] <= threshold_p
            and next(
                value
                for value in row["rows"]
                if value["family"] == family
            )["correct"]
        ]
        behavior_models = behavior_annotation_models(
            summaries,
            family,
        )
        l1 = len(within_hits) >= minimum
        l2 = l1 and len(raw_cross_hits) >= minimum
        l3 = l2 and len(centered_cross_hits) >= minimum
        l4 = l3 and len(behavior_models) >= minimum
        highest = (
            "L4" if l4
            else "L3" if l3
            else "L2" if l2
            else "L1" if l1
            else "L0"
        )
        direction_repeat = {}
        for model_name, rows in split_directions_by_model.items():
            values = [
                float(row[
                    "discovery_confirmation_truth_direction_cosine"
                ])
                for row in rows
                if row["conditioning"] == "all_finite"
                and row["family"] == family
                and row[
                    "discovery_confirmation_truth_direction_cosine"
                ] is not None
            ]
            direction_repeat[model_name] = {
                "mean": safe_mean(values),
                "median": safe_median(values),
                "observation_count": len(values),
            }
        row = {
            "highest_evidence_level": highest,
            "within_model_centered_assignment_hits": within_hits,
            "cross_model_raw_assignment_hits": raw_cross_hits,
            "cross_model_centered_assignment_hits": centered_cross_hits,
            "behavior_annotation_models": behavior_models,
            "discovery_confirmation_truth_direction_repeat": (
                direction_repeat
            ),
            "descriptive_status": (
                "shared_shell_family_specific_response_repeated"
                if l3
                else "mapped_with_incomplete_shared_shell_repetition"
            ),
            "causal_status": "not_tested",
            "retained_in_atlas": True,
        }
        family_evidence[family] = row
        evidence_rows.append({
            "schema_version": "phase1078_family_evidence.v1",
            "phase": protocol.PHASE,
            "family": family,
            **row,
        })

    consensus = build_consensus_regions(metrics_by_model)
    controls = factor_control_ratios(metrics_by_model)
    truth_alignment = generic_truth_alignment()
    source_1077 = protocol.read_json(protocol.SOURCE_PHASE1077)

    translation_means = {
        model: family_evidence["translation_equivalence"][
            "discovery_confirmation_truth_direction_repeat"
        ][model]["mean"]
        for model in protocol.MODELS
    }
    translation_pass_models = [
        model
        for model, value in translation_means.items()
        if value is not None
        and value
        < prereg["evidence_thresholds"][
            "translation_direction_repeat_max"
        ]
    ]
    old_to_new = {}
    mapping = {
        "height_relation": "height_polarity",
        "punctuation_rule": "punctuation",
    }
    for new_family, old_family in mapping.items():
        old_to_new[new_family] = {}
        for model in protocol.MODELS:
            old_value = source_1077["family_evidence"][old_family][
                "discovery_confirmation_direction_repeat"
            ][model]["mean"]
            new_value = family_evidence[new_family][
                "discovery_confirmation_truth_direction_repeat"
            ][model]["mean"]
            old_to_new[new_family][model] = {
                "phase1077": old_value,
                "phase1078": new_value,
                "drop": (
                    old_value - new_value
                    if old_value is not None and new_value is not None
                    else None
                ),
            }
    drop_threshold = prereg["evidence_thresholds"][
        "phase1077_direction_drop_min"
    ]
    direction_drop_pass = any(
        sum(
            1
            for row in by_model.values()
            if row["drop"] is not None
            and row["drop"] >= drop_threshold
        ) >= minimum
        for by_model in old_to_new.values()
    )

    within_centered_significant = [
        row
        for row in assignment_rows
        if row["comparison"]
        == "within_model_discovery_to_confirmation"
        and row["profile"] == "family_centered"
        and row["exact_upper_tail_p"] <= threshold_p
    ]
    prediction_audit = {
        "schema_version": "phase1078_prediction_audit.v1",
        "phase": protocol.PHASE,
        "predictions": {
            "P1": {
                "passed": len(within_centered_significant) >= minimum,
                "significant_models": [
                    row["source_model"]
                    for row in within_centered_significant
                ],
            },
            "P2": {
                "passed": len(
                    truth_alignment["passing_models"]
                ) >= minimum,
                "passing_models": truth_alignment["passing_models"],
            },
            "P3": {
                "passed": len(translation_pass_models) >= minimum,
                "translation_direction_means": translation_means,
                "passing_models": translation_pass_models,
            },
            "P4": {
                "passed": direction_drop_pass,
                "phase1077_to_phase1078": old_to_new,
            },
            "P5": {
                "passed": len(
                    controls["nontrivial_models"]
                ) >= minimum,
                "nontrivial_models": controls["nontrivial_models"],
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

    mechanism_status = {
        family: {
            "what_is_observed": (
                "A shared-shell true/false differential field with "
                "independent surface and shell controls across component, "
                "token role, and normalized depth."
            ),
            "what_repetition_supports": row["descriptive_status"],
            "what_is_not_established": (
                "No head, neuron, transport edge, necessary path, "
                "sufficient state, complete token algorithm, minimal code, "
                "or optimality law is established."
            ),
        }
        for family, row in family_evidence.items()
    }
    hypothesis_audit = {
        "language_as_pattern_collection": {
            "status": "compatible_but_not_a_complete_ontology",
            "reason": (
                "Eight families can be mapped under one protocol, but the "
                "set is neither exhaustive nor proven to be language's "
                "fundamental basis."
            ),
        },
        "relative_encoding": {
            "status": "operational_conditional_difference_support",
            "reason": (
                "Truth, surface, and shell-relative changes are separately "
                "measured. This does not prove that concepts exist only as "
                "relations."
            ),
        },
        "reuse_plus_minimal_difference": {
            "status": "reuse_is_tested_minimality_is_not",
            "reason": (
                "Repeated topologies and direction reuse estimate sharing; "
                "no compression optimum or minimum-description proof exists."
            ),
        },
        "globally_optimal_distribution": {
            "status": "unsupported",
            "reason": (
                "Optimality requires comparative training, capacity, "
                "energy, robustness, and ablation evidence."
            ),
        },
        "unique_word_ecological_niche": {
            "status": "not_established",
            "reason": (
                "Rare terms are paired with supplied definitions. This "
                "tests contextual integration, not a complete unique niche "
                "for each vocabulary item."
            ),
        },
        "style_logic_grammar_joint_selection": {
            "status": "compatible_but_not_recovered",
            "reason": (
                "Multiple families share a yes/no arbitration protocol, "
                "but style, logic, syntax, and content were not fully "
                "orthogonalized in free generation."
            ),
        },
        "small_model_roughness": {
            "status": "live_limit_not_an_explanation",
            "reason": (
                "Cross-model disagreement is preserved. Scale, data, "
                "tokenizer, and architecture are confounded."
            ),
        },
    }
    automatic_next = {
        "schema_version": "phase1078_automatic_next.v1",
        "phase": protocol.PHASE,
        "continue": False,
        "reason": prereg["automatic_next"]["reason"],
        "recommended_large_next_task": (
            "Build an independent natural-generation atlas that removes "
            "the shared yes/no output axis, then use its preregistered "
            "predictions to select component-level tests in a separate "
            "causal ledger."
        ),
    }
    automatic_next["decision_digest"] = protocol.digest(automatic_next)

    analysis_root = protocol.OUT_ROOT / "analysis"
    protocol.write_jsonl(
        analysis_root / "family_evidence_ledger.jsonl",
        evidence_rows,
    )
    protocol.write_jsonl(
        analysis_root / "consensus_regions.jsonl",
        consensus,
    )
    assignment_payload = {
        "schema_version": "phase1078_exact_assignment_collection.v1",
        "phase": protocol.PHASE,
        "rows": assignment_rows,
    }
    assignment_payload["assignment_digest"] = protocol.digest(
        assignment_payload
    )
    protocol.write_json(
        analysis_root / "exact_permutation_assignment.json",
        assignment_payload,
    )
    protocol.write_json(
        analysis_root / "factor_control_ratios.json",
        controls,
    )
    protocol.write_json(
        analysis_root / "generic_truth_alignment.json",
        truth_alignment,
    )
    protocol.write_json(
        analysis_root / "prediction_audit.json",
        prediction_audit,
    )
    protocol.write_json(
        analysis_root / "automatic_next.json",
        automatic_next,
    )

    final = {
        "schema_version": "phase1078_final_summary.v1",
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
        "measurement_integrity": {
            model: {
                "candidate_finite_coverage": (
                    1.0
                    - summary["nonfinite_candidate_count"]
                    / summary["case_count"]
                ),
                "nonfinite_candidate_count": summary[
                    "nonfinite_candidate_count"
                ],
                "nonfinite_hidden_truth_role_count": summary[
                    "nonfinite_hidden_truth_role_count"
                ],
            }
            for model, summary in summaries.items()
        },
        "family_evidence": family_evidence,
        "exact_assignment_summary": [{
            "comparison": row["comparison"],
            "profile": row["profile"],
            "source_model": row["source_model"],
            "target_model": row["target_model"],
            "top1_correct": row["top1_correct"],
            "family_count": row["family_count"],
            "exact_upper_tail_p": row["exact_upper_tail_p"],
            "minimum_margin_over_best_other": row[
                "minimum_margin_over_best_other"
            ],
        } for row in assignment_rows],
        "factor_control_ratios": controls,
        "generic_truth_alignment": truth_alignment,
        "prospective_prediction_audit": prediction_audit,
        "mechanism_status": mechanism_status,
        "hypothesis_audit": hypothesis_audit,
        "mathematical_status": {
            "current_tools_sufficient_for": [
                "three-factor controlled differences",
                "normalized-depth response profiles",
                "exact family-label permutation tests",
                "direction and topology separation",
                "generic-axis versus family-centered decomposition",
            ],
            "not_yet_recovered": [
                "a complete language ontology",
                "a predictive natural-generation state transition law",
                "head/neuron transport routes",
                "causal necessity and sufficiency",
                "minimality or efficiency optimality",
                "brain-model mechanism homology",
            ],
            "new_mathematics_needed_now": False,
            "reason": (
                "The limiting problem remains empirical identification. "
                "Existing elementary differences, norms, cosines, and exact "
                "permutations are adequate for the present descriptive step."
            ),
        },
        "claim_boundary": (
            "Phase1078 tests whether family response topologies survive a "
            "shared yes/no shell while separately recording surface and "
            "shell controls. It does not close a language mechanism."
        ),
        "automatic_next_decision": automatic_next,
    }
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(
        analysis_root / "final_summary.json",
        final,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "evidence_levels": {
            family: row["highest_evidence_level"]
            for family, row in family_evidence.items()
        },
        "prediction_passed_count": prediction_audit["passed_count"],
        "automatic_next": automatic_next["continue"],
        "summary_digest": final["summary_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
