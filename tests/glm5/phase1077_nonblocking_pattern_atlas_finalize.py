#!/usr/bin/env python3
"""Finalize Phase1077 without turning descriptive repetition into causality."""

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

import phase1077_nonblocking_pattern_atlas_protocol as protocol


GRID = np.linspace(0.0, 1.0, 21, dtype=np.float64)
PROFILE_METRICS = (
    "mean_semantic_relative_magnitude",
    "mean_lexical_relative_magnitude",
    "mean_interaction_relative_magnitude",
    "semantic_direction_consistency",
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
        row for row in rows
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
        row for row in rows
        if row["family"] == family
        and row["split"] == split
        and row["conditioning"] == conditioning
    ]
    metric_channels = []
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
        metric_channels.append(channel_normalize(np.concatenate(values)))
    return np.concatenate(metric_channels).astype(np.float64)


def depth_band(relative_depth: float) -> str:
    if relative_depth < 0.30:
        return "early"
    if relative_depth < 0.70:
        return "middle"
    return "late"


def build_consensus_regions(
    metrics_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    aggregate: dict[
        tuple[str, str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for model_name, rows in metrics_by_model.items():
        selected = [
            row for row in rows
            if row["conditioning"] == "all_finite"
            and row["split"] == "confirmation"
            and row["mean_semantic_relative_magnitude"] is not None
        ]
        maxima = defaultdict(float)
        for row in selected:
            maxima[row["family"]] = max(
                maxima[row["family"]],
                float(row["mean_semantic_relative_magnitude"]),
            )
        local: dict[
            tuple[str, str, str, str], list[float]
        ] = defaultdict(list)
        for row in selected:
            family = str(row["family"])
            maximum = maxima[family]
            normalized = (
                float(row["mean_semantic_relative_magnitude"]) / maximum
                if maximum > EPSILON else 0.0
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
            "schema_version": "phase1077_consensus_region.v1",
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
        values = [
            row for row in output if row["family"] == family
        ]
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


def behavior_annotation_models(
    summaries: dict[str, dict[str, Any]],
    family: str,
) -> list[str]:
    return [
        model
        for model, summary in summaries.items()
        if summary["families"][family]["behavior_annotation_passed"]
    ]


def retrieval_diagnostic(
    source: dict[tuple[str, str, str], np.ndarray],
    *,
    source_model: str,
    source_split: str,
    target_model: str,
    target_split: str,
) -> dict[str, Any]:
    hits = 0
    margins = []
    rows = []
    for family in protocol.FAMILIES:
        scores = {
            candidate: cosine(
                source[(source_model, family, source_split)],
                source[(target_model, candidate, target_split)],
            )
            for candidate in protocol.FAMILIES
        }
        finite_scores = {
            key: value for key, value in scores.items()
            if value is not None
        }
        prediction = max(finite_scores, key=finite_scores.get)
        alternatives = [
            value for candidate, value in finite_scores.items()
            if candidate != family
        ]
        margin = (
            finite_scores[family] - max(alternatives)
            if alternatives else None
        )
        hits += int(prediction == family)
        if margin is not None:
            margins.append(margin)
        rows.append({
            "family": family,
            "prediction": prediction,
            "correct": prediction == family,
            "same_family_score": finite_scores.get(family),
            "margin_over_best_other": margin,
        })
    return {
        "source_model": source_model,
        "source_split": source_split,
        "target_model": target_model,
        "target_split": target_split,
        "top1_correct": hits,
        "family_count": len(protocol.FAMILIES),
        "top1_accuracy": hits / len(protocol.FAMILIES),
        "mean_margin_over_best_other": safe_mean(margins),
        "minimum_margin_over_best_other": (
            min(margins) if margins else None
        ),
        "rows": rows,
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1077 protocol audit failed")

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

    profile_rows = []
    family_evidence = {}
    thresholds = prereg["evidence_thresholds"]
    for family in protocol.FAMILIES:
        within_model = {}
        repeated_models = []
        for model_name in protocol.MODELS:
            score = cosine(
                profiles[(model_name, family, "discovery")],
                profiles[(model_name, family, "confirmation")],
            )
            within_model[model_name] = score
            if (
                score is not None
                and score
                >= thresholds[
                    "within_model_split_profile_cosine_l1"
                ]
            ):
                repeated_models.append(model_name)

        raw_cross_model = {}
        centered_cross_model = {}
        raw_repeated_pairs = []
        centered_repeated_pairs = []
        for left, right in MODEL_PAIRS:
            pair_name = f"{left}__{right}"
            raw_score = cosine(
                profiles[(left, family, "confirmation")],
                profiles[(right, family, "confirmation")],
            )
            centered_score = cosine(
                centered_profiles[(left, family, "confirmation")],
                centered_profiles[(right, family, "confirmation")],
            )
            raw_cross_model[pair_name] = raw_score
            centered_cross_model[pair_name] = centered_score
            if (
                raw_score is not None
                and raw_score
                >= thresholds[
                    "cross_model_raw_profile_cosine_l2"
                ]
            ):
                raw_repeated_pairs.append(pair_name)
            if (
                centered_score is not None
                and centered_score
                >= thresholds[
                    "cross_model_centered_profile_cosine_l3"
                ]
            ):
                centered_repeated_pairs.append(pair_name)

        behavior_models = behavior_annotation_models(
            summaries,
            family,
        )
        minimum = int(
            thresholds["minimum_repeated_models_or_pairs"]
        )
        l1 = len(repeated_models) >= minimum
        l2 = l1 and len(raw_repeated_pairs) >= minimum
        l3 = l2 and len(centered_repeated_pairs) >= minimum
        l4 = l3 and len(behavior_models) >= minimum
        highest = "L4" if l4 else (
            "L3" if l3 else (
                "L2" if l2 else ("L1" if l1 else "L0")
            )
        )

        direction_repeat = {}
        for model_name, rows in split_directions_by_model.items():
            values = [
                float(row[
                    "discovery_confirmation_direction_cosine"
                ])
                for row in rows
                if row["conditioning"] == "all_finite"
                and row["family"] == family
                and row[
                    "discovery_confirmation_direction_cosine"
                ] is not None
            ]
            direction_repeat[model_name] = {
                "mean": safe_mean(values),
                "median": (
                    float(np.median(values)) if values else None
                ),
                "observation_count": len(values),
            }

        if l3:
            descriptive_status = (
                "cross_model_family_specific_response_repeated"
            )
        elif l2:
            descriptive_status = (
                "cross_model_generic_shape_repeated_but_family_specificity_weak"
            )
        elif l1:
            descriptive_status = "within_model_response_repeated"
        else:
            descriptive_status = "mapped_but_not_repeated_at_frozen_thresholds"
        family_evidence[family] = {
            "highest_evidence_level": highest,
            "within_model_split_profile_cosine": within_model,
            "within_model_repeated_models": repeated_models,
            "cross_model_raw_profile_cosine": raw_cross_model,
            "cross_model_raw_repeated_pairs": raw_repeated_pairs,
            "cross_model_centered_profile_cosine": (
                centered_cross_model
            ),
            "cross_model_centered_repeated_pairs": (
                centered_repeated_pairs
            ),
            "behavior_annotation_models": behavior_models,
            "discovery_confirmation_direction_repeat": (
                direction_repeat
            ),
            "descriptive_status": descriptive_status,
            "causal_status": "not_tested",
            "retained_in_atlas": True,
        }
        profile_rows.append({
            "schema_version": "phase1077_family_profile_evidence.v1",
            "phase": protocol.PHASE,
            "family": family,
            **family_evidence[family],
        })

    reuse_rows = []
    for model_name in protocol.MODELS:
        for left, right in itertools.combinations(protocol.FAMILIES, 2):
            reuse_rows.append({
                "model": model_name,
                "family_left": left,
                "family_right": right,
                "raw_profile_cosine": cosine(
                    profiles[(model_name, left, "confirmation")],
                    profiles[(model_name, right, "confirmation")],
                ),
                "centered_profile_cosine": cosine(
                    centered_profiles[
                        (model_name, left, "confirmation")
                    ],
                    centered_profiles[
                        (model_name, right, "confirmation")
                    ],
                ),
            })
    pair_summary = {}
    for left, right in itertools.combinations(protocol.FAMILIES, 2):
        selected = [
            row for row in reuse_rows
            if row["family_left"] == left
            and row["family_right"] == right
        ]
        pair_summary[f"{left}__{right}"] = {
            "mean_raw_profile_cosine": safe_mean([
                row["raw_profile_cosine"]
                for row in selected
                if row["raw_profile_cosine"] is not None
            ]),
            "mean_centered_profile_cosine": safe_mean([
                row["centered_profile_cosine"]
                for row in selected
                if row["centered_profile_cosine"] is not None
            ]),
            "by_model": {
                row["model"]: {
                    "raw": row["raw_profile_cosine"],
                    "centered": row["centered_profile_cosine"],
                }
                for row in selected
            },
        }

    retrieval_rows = []
    for model_name in protocol.MODELS:
        for profile_name, source in (
            ("raw", profiles),
            ("family_centered", centered_profiles),
        ):
            retrieval_rows.append({
                "comparison": "within_model_discovery_to_confirmation",
                "profile": profile_name,
                **retrieval_diagnostic(
                    source,
                    source_model=model_name,
                    source_split="discovery",
                    target_model=model_name,
                    target_split="confirmation",
                ),
            })
    for left, right in MODEL_PAIRS:
        for source_model, target_model in (
            (left, right),
            (right, left),
        ):
            for profile_name, source in (
                ("raw", profiles),
                ("family_centered", centered_profiles),
            ):
                retrieval_rows.append({
                    "comparison": "cross_model_confirmation",
                    "profile": profile_name,
                    **retrieval_diagnostic(
                        source,
                        source_model=source_model,
                        source_split="confirmation",
                        target_model=target_model,
                        target_split="confirmation",
                    ),
                })

    consensus_regions = build_consensus_regions(metrics_by_model)
    mechanism_status = {
        family: {
            "what_is_observed": (
                "A semantic-branch differential field across residual, "
                "Attention output, MLP output, token role, and normalized depth."
            ),
            "what_repetition_supports": row["descriptive_status"],
            "what_is_not_established": (
                "No specific neuron, head, transport edge, necessary path, "
                "sufficient state, full token algorithm, or optimality law."
            ),
        }
        for family, row in family_evidence.items()
    }
    hypothesis_audit = {
        "language_as_pattern_collection": {
            "status": "compatible_but_not_identified_as_complete_ontology",
            "reason": (
                "The six controlled operations produce distinguishable "
                "response fields, but six tasks cannot exhaust language."
            ),
        },
        "relative_encoding": {
            "status": "operationally_supported_as_conditional_differences",
            "reason": (
                "The atlas measures stable branch-relative changes. It does "
                "not prove that every concept exists only relationally."
            ),
        },
        "reuse_plus_minimal_difference": {
            "status": "testable_partial_support_only",
            "reason": (
                "Raw profile similarity estimates reuse and centered profiles "
                "estimate family-specific deviations; minimality is not proven."
            ),
        },
        "globally_optimal_distribution": {
            "status": "unsupported",
            "reason": (
                "Efficiency or evolutionary optimality requires comparative "
                "training, ablation, capacity, and energy evidence."
            ),
        },
        "unique_word_ecological_niche": {
            "status": "not_directly_tested",
            "reason": (
                "Rare-word versus definition surfaces probe contextual "
                "equivalence, not a complete unique niche for every word."
            ),
        },
        "style_logic_grammar_joint_selection": {
            "status": "compatible_with_interaction_fields_not_closed",
            "reason": (
                "Semantic-surface interaction is measured, but the complete "
                "next-token arbitration mechanism remains unresolved."
            ),
        },
        "small_model_roughness": {
            "status": "live_limit",
            "reason": (
                "Cross-model disagreement is recorded rather than averaged "
                "away; it may reflect scale, data, tokenizer, or architecture."
            ),
        },
    }

    analysis_root = protocol.OUT_ROOT / "analysis"
    protocol.write_jsonl(
        analysis_root / "family_evidence_ledger.jsonl",
        profile_rows,
    )
    protocol.write_jsonl(
        analysis_root / "consensus_regions.jsonl",
        consensus_regions,
    )
    protocol.write_json(
        analysis_root / "reuse_matrix.json",
        {
            "schema_version": "phase1077_reuse_matrix.v1",
            "phase": protocol.PHASE,
            "rows": reuse_rows,
            "pair_summary": pair_summary,
            "interpretation": (
                "Raw similarity includes common depth dynamics. Centered "
                "similarity compares deviations from each model's six-family "
                "mean and is therefore the stricter reuse indicator."
            ),
        },
    )
    retrieval_payload = {
        "schema_version": "phase1077_posthoc_family_retrieval.v1",
        "phase": protocol.PHASE,
        "preregistered_evidence_gate": False,
        "rows": retrieval_rows,
        "interpretation": (
            "This posthoc diagnostic asks whether response profiles identify "
            "their own protocol family better than the other five families. "
            "Success demonstrates a repeatable task-family signature, not a "
            "causal language mechanism; prompt and role geometry remain live "
            "alternative explanations."
        ),
    }
    retrieval_payload["diagnostic_digest"] = protocol.digest(
        retrieval_payload
    )
    protocol.write_json(
        analysis_root / "posthoc_family_retrieval.json",
        retrieval_payload,
    )
    automatic_next = {
        "schema_version": "phase1077_automatic_next.v1",
        "phase": protocol.PHASE,
        "continue": False,
        "reason": prereg["automatic_next"]["reason"],
        "recommended_large_next_task": (
            "Expand the nonblocking atlas with additional independent "
            "pattern families and counterfactual controls, while selecting "
            "causal tests through a separate evidence ledger."
        ),
    }
    automatic_next["decision_digest"] = protocol.digest(
        automatic_next
    )
    protocol.write_json(
        analysis_root / "automatic_next.json",
        automatic_next,
    )

    final = {
        "schema_version": "phase1077_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "case_count_total": sum(
            summary["case_count"] for summary in summaries.values()
        ),
        "unit_count_total": sum(
            summary["unit_count"] for summary in summaries.values()
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
                "hidden_semantic_observation_finite_coverage": (
                    1.0
                    - summary["nonfinite_hidden_role_count"]
                    / (
                        2
                        * summary["unit_count"]
                        * summary["event_count"]
                        * len(protocol.CAPTURE_ROLES)
                    )
                ),
                "nonfinite_candidate_count": summary[
                    "nonfinite_candidate_count"
                ],
                "nonfinite_hidden_role_count": summary[
                    "nonfinite_hidden_role_count"
                ],
            }
            for model, summary in summaries.items()
        },
        "family_evidence": family_evidence,
        "mechanism_status": mechanism_status,
        "hypothesis_audit": hypothesis_audit,
        "posthoc_family_retrieval": {
            "all_comparisons_top1_perfect": all(
                row["top1_accuracy"] == 1.0
                for row in retrieval_rows
            ),
            "comparison_count": len(retrieval_rows),
            "minimum_margin_over_best_other": min(
                row["minimum_margin_over_best_other"]
                for row in retrieval_rows
                if row["minimum_margin_over_best_other"] is not None
            ),
            "claim_limit": retrieval_payload["interpretation"],
        },
        "mathematical_status": {
            "current_tools_sufficient_for": [
                "controlled differences",
                "relative magnitudes",
                "cosine repetition",
                "normalized-depth profiles",
                "factor interaction residuals",
            ],
            "current_tools_not_yet_sufficient_for": [
                "a complete language ontology",
                "a recovered routing algorithm",
                "causal necessity and sufficiency",
                "global efficiency optimality",
                "brain-model mechanism homology",
            ],
            "new_mathematics_needed_now": False,
            "reason": (
                "The immediate bottleneck is empirical identifiability and "
                "controlled repetition, not the absence of a named advanced "
                "mathematical formalism."
            ),
        },
        "claim_boundary": (
            "Phase1077 maps repeated conditional response fields. It does "
            "not close any language mechanism and does not treat a response "
            "profile as a causal route."
        ),
        "automatic_next_decision": automatic_next,
    }
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(analysis_root / "final_summary.json", final)
    print(json.dumps({
        "phase": protocol.PHASE,
        "family_levels": {
            family: row["highest_evidence_level"]
            for family, row in family_evidence.items()
        },
        "automatic_next": automatic_next["continue"],
        "summary_digest": final["summary_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
