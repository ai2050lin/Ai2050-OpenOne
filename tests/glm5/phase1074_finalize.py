#!/usr/bin/env python3
"""Finalize Phase1074 residual-transition and routing evidence."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1074_polarity_dynamics_protocol as protocol


EPSILON = 1e-12


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    left = left[finite].astype(np.float64)
    right = right[finite].astype(np.float64)
    denominator = float(
        np.linalg.norm(left) * np.linalg.norm(right)
    )
    if denominator <= EPSILON:
        return 0.0
    return float(np.dot(left, right) / denominator)


def mean(values: list[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def profile(
    rows: list[dict[str, Any]],
    *,
    field: str,
    filters: dict[str, str],
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    grouped: dict[tuple[int, str], list[float | None]] = (
        defaultdict(list)
    )
    for row in rows:
        if any(str(row[key]) != value for key, value in filters.items()):
            continue
        if row["role"] not in protocol.PRIMARY_DYNAMIC_ROLES:
            continue
        grouped[(int(row["depth"]), str(row["role"]))].append(
            row.get(field)
        )
    keys = [
        (depth, role)
        for depth in sorted({key[0] for key in grouped})
        for role in protocol.PRIMARY_DYNAMIC_ROLES
    ]
    values = np.asarray(
        [
            mean(grouped.get(key, []))
            if mean(grouped.get(key, [])) is not None
            else np.nan
            for key in keys
        ],
        dtype=np.float64,
    )
    return values, keys


def routing_means(
    archive,
) -> tuple[np.ndarray, np.ndarray]:
    sums = archive["sums"].astype(np.float64)
    counts = archive["counts"].astype(np.int64)
    positive = archive["positive_counts"].astype(np.int64)
    values = np.full_like(sums, np.nan, dtype=np.float64)
    positive_rate = np.full_like(sums, np.nan, dtype=np.float64)
    valid = counts > 0
    values[valid] = sums[valid] / counts[valid]
    positive_rate[valid] = positive[valid] / counts[valid]
    return values, positive_rate


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "behavior_decision.json"
    )
    if not behavior["should_run_internal_dynamics"]:
        payload = {
            "schema_version": "phase1074_automatic_next.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "selected_models": [],
            "selected_model_count": 0,
            "should_continue_automatically": False,
            "next_phase": None,
            "route": "stop_at_behavior_foundation",
            "reason": (
                "The behavior gate did not authorize internal dynamics."
            ),
        }
        protocol.write_json(
            protocol.OUT_ROOT
            / "analysis"
            / "automatic_next.json",
            payload,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    gates = prereg["gates"]
    model_summaries = {}
    head_records = []
    dynamic_models = []
    normalized_profiles = {}

    for model in protocol.MODELS:
        scan_summary = protocol.read_json(
            protocol.OUT_ROOT
            / "dynamics"
            / model
            / "summary.json"
        )
        residual_rows = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "dynamics"
            / model
            / "residual_unit_metrics.jsonl"
        )
        relation_results = {}
        global_prebranch = max(
            (
                float(row["selection_relative_magnitude"])
                for row in residual_rows
                if row["role"] in protocol.PRE_BRANCH_ROLES
                and row["selection_relative_magnitude"] is not None
            ),
            default=float("nan"),
        )
        embedding_max = max(
            (
                float(row["selection_relative_magnitude"])
                for row in residual_rows
                if int(row["depth"]) == 0
                and row["selection_relative_magnitude"] is not None
            ),
            default=float("nan"),
        )

        archive = np.load(
            protocol.OUT_ROOT
            / "dynamics"
            / model
            / "routing_aggregates.npz",
            allow_pickle=False,
        )
        routing, routing_positive = routing_means(archive)
        relation_idx = {
            value: index
            for index, value in enumerate(protocol.RELATIONS)
        }
        split_idx = {
            value: index
            for index, value in enumerate(protocol.SPLITS)
        }
        pair_idx = {
            value[0]: index
            for index, value in enumerate(protocol.SOURCE_PAIRS)
        }
        metric_idx = {"attention_mass": 0, "av_norm": 1}
        all_conditioning = 0
        discovery_fact = routing[
            :,
            split_idx["discovery"],
            :,
            :,
            :,
            :,
            :,
            pair_idx["fact"],
            metric_idx["attention_mass"],
            all_conditioning,
        ]
        # Average relation, path, layout, retaining layer/head/destination.
        discovery_head_score = np.nanmean(
            discovery_fact, axis=(0, 1, 2)
        )
        flat = []
        for depth in range(discovery_head_score.shape[0]):
            for head in range(discovery_head_score.shape[1]):
                for destination in range(
                    discovery_head_score.shape[2]
                ):
                    value = float(
                        discovery_head_score[
                            depth, head, destination
                        ]
                    )
                    if math.isfinite(value):
                        flat.append(
                            (value, depth, head, destination)
                        )
        flat.sort(reverse=True)
        selected_heads = flat[
            : int(gates["attention_selected_head_count"])
        ]

        confirmation_fact = routing[
            :,
            split_idx["confirmation"],
            :,
            :,
            :,
            :,
            :,
            pair_idx["fact"],
            metric_idx["attention_mass"],
            all_conditioning,
        ]
        confirmation_null = routing[
            :,
            split_idx["confirmation"],
            :,
            :,
            :,
            :,
            :,
            pair_idx["null_control"],
            metric_idx["attention_mass"],
            all_conditioning,
        ]
        confirmation_positive = routing_positive[
            :,
            split_idx["confirmation"],
            :,
            :,
            :,
            :,
            :,
            pair_idx["fact"],
            metric_idx["attention_mass"],
            all_conditioning,
        ]

        for rank, (score, depth, head, destination) in enumerate(
            selected_heads, 1
        ):
            fact_values = confirmation_fact[
                :, :, :, depth, head, destination
            ]
            null_values = confirmation_null[
                :, :, :, depth, head, destination
            ]
            positive_values = confirmation_positive[
                :, :, :, depth, head, destination
            ]
            head_records.append({
                "schema_version": (
                    "phase1074_frozen_head_candidate.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "rank": rank,
                "depth": depth + 1,
                "relative_depth": (depth + 1) / scan_summary[
                    "n_layers"
                ],
                "head": head,
                "destination": protocol.ATTENTION_DESTINATIONS[
                    destination
                ],
                "discovery_fact_route_score": score,
                "confirmation_fact_route_score": float(
                    np.nanmean(fact_values)
                ),
                "confirmation_positive_fraction": float(
                    np.nanmean(positive_values)
                ),
                "confirmation_null_control_abs": float(
                    np.nanmean(np.abs(null_values))
                ),
            })

        selected_coordinates = [
            (depth, head, destination)
            for _score, depth, head, destination in selected_heads
        ]
        relation_route = {}
        for relation in protocol.RELATIONS:
            ridx = relation_idx[relation]
            fact_values = []
            null_values = []
            positive_values = []
            for depth, head, destination in selected_coordinates:
                fact_values.append(
                    confirmation_fact[
                        ridx, :, :, depth, head, destination
                    ].reshape(-1)
                )
                null_values.append(
                    confirmation_null[
                        ridx, :, :, depth, head, destination
                    ].reshape(-1)
                )
                positive_values.append(
                    confirmation_positive[
                        ridx, :, :, depth, head, destination
                    ].reshape(-1)
                )
            fact_array = (
                np.concatenate(fact_values)
                if fact_values
                else np.asarray([], dtype=np.float64)
            )
            null_array = (
                np.concatenate(null_values)
                if null_values
                else np.asarray([], dtype=np.float64)
            )
            positive_array = (
                np.concatenate(positive_values)
                if positive_values
                else np.asarray([], dtype=np.float64)
            )
            fact_mean = float(np.nanmean(fact_array))
            null_abs = float(np.nanmean(np.abs(null_array)))
            positive_fraction = float(
                np.nanmean(positive_array)
            )
            effect_ratio = abs(fact_mean) / (
                null_abs + EPSILON
            )
            relation_route[relation] = {
                "confirmation_fact_route_score": fact_mean,
                "confirmation_positive_fraction": positive_fraction,
                "confirmation_null_control_abs": null_abs,
                "fact_to_null_effect_ratio": effect_ratio,
            }

        layer_profile = np.nanmean(
            discovery_fact, axis=(0, 1, 2, 4, 5)
        )
        bins = 12
        binned = np.full(bins, np.nan, dtype=np.float64)
        for index in range(bins):
            start = int(
                math.floor(index * len(layer_profile) / bins)
            )
            end = int(
                math.floor((index + 1) * len(layer_profile) / bins)
            )
            end = max(end, start + 1)
            binned[index] = float(
                np.nanmean(layer_profile[start:end])
            )
        normalized_profiles[model] = binned

        dynamic_relations = []
        for relation in protocol.RELATIONS:
            relation_rows = [
                row
                for row in residual_rows
                if row["relation"] == relation
            ]
            selection_peak = max(
                (
                    float(row["selection_relative_magnitude"])
                    for row in relation_rows
                    if row["role"]
                    in protocol.PRIMARY_DYNAMIC_ROLES
                    and row["selection_relative_magnitude"]
                    is not None
                ),
                default=float("nan"),
            )
            transition_peak = max(
                (
                    float(row["transition_relative_magnitude"])
                    for row in relation_rows
                    if row["role"]
                    in protocol.PRIMARY_DYNAMIC_ROLES
                    and row["transition_relative_magnitude"]
                    is not None
                ),
                default=float("nan"),
            )
            strong_rows = [
                row
                for row in relation_rows
                if row["role"] in protocol.PRIMARY_DYNAMIC_ROLES
                and row["selection_relative_magnitude"] is not None
                and float(row["selection_relative_magnitude"])
                >= gates[
                    "selection_interaction_relative_magnitude_min"
                ]
            ]
            lexical_reuse = mean([
                row["selection_lexical_reuse_cosine"]
                for row in strong_rows
            ])
            discovery_profile, _ = profile(
                relation_rows,
                field="selection_relative_magnitude",
                filters={"split": "discovery"},
            )
            confirmation_profile, _ = profile(
                relation_rows,
                field="selection_relative_magnitude",
                filters={"split": "confirmation"},
            )
            direct_profile, _ = profile(
                relation_rows,
                field="selection_relative_magnitude",
                filters={"path": "direct"},
            )
            transitive_profile, _ = profile(
                relation_rows,
                field="selection_relative_magnitude",
                filters={"path": "transitive"},
            )
            split_cosine = cosine(
                discovery_profile, confirmation_profile
            )
            path_cosine = cosine(
                direct_profile, transitive_profile
            )
            route = relation_route[relation]
            behavior_relation_valid = bool(
                protocol.read_json(
                    protocol.OUT_ROOT
                    / "behavior"
                    / model
                    / "summary.json"
                )["relations"][relation][
                    "strong_relation_gate_passed"
                ]
            )
            gate_checks = {
                "behavior_relation_valid": behavior_relation_valid,
                "numerical": (
                    scan_summary["residual_metric_finite_rate"]
                    >= gates["internal_finite_rate_min"]
                    and scan_summary["routing_metric_finite_rate"]
                    >= gates["internal_finite_rate_min"]
                ),
                "prebranch_zero": (
                    global_prebranch
                    <= gates[
                        "prebranch_selection_interaction_max"
                    ]
                ),
                "embedding_zero": (
                    embedding_max
                    <= gates[
                        "embedding_selection_interaction_max"
                    ]
                ),
                "selection_magnitude": (
                    selection_peak
                    >= gates[
                        "selection_interaction_relative_magnitude_min"
                    ]
                ),
                "transition_magnitude": (
                    transition_peak
                    >= gates[
                        "transition_interaction_relative_magnitude_min"
                    ]
                ),
                "lexical_reuse": (
                    lexical_reuse is not None
                    and lexical_reuse
                    >= gates[
                        "selection_lexical_reuse_cosine_min"
                    ]
                ),
                "split_replication": (
                    split_cosine
                    >= gates["selection_split_profile_cosine_min"]
                ),
                "path_replication": (
                    path_cosine
                    >= gates["selection_path_profile_cosine_min"]
                ),
                "heldout_attention_route": (
                    route["confirmation_fact_route_score"] > 0.0
                    and route[
                        "confirmation_positive_fraction"
                    ]
                    >= gates[
                        "attention_confirmation_positive_fraction_min"
                    ]
                    and route["fact_to_null_effect_ratio"]
                    >= gates[
                        "attention_fact_to_null_effect_ratio_min"
                    ]
                ),
            }
            passed = all(gate_checks.values())
            if passed:
                dynamic_relations.append(relation)
            relation_results[relation] = {
                "selection_interaction_peak": selection_peak,
                "transition_interaction_peak": transition_peak,
                "selection_lexical_reuse_cosine": lexical_reuse,
                "discovery_confirmation_profile_cosine": (
                    split_cosine
                ),
                "direct_transitive_profile_cosine": path_cosine,
                "routing": route,
                "gate_checks": gate_checks,
                "dynamic_relation_gate_passed": passed,
            }

        model_gate = bool(
            model in behavior["selected_models"]
            and len(dynamic_relations)
            >= gates["minimum_dynamic_relations_per_model"]
        )
        if model_gate:
            dynamic_models.append(model)
        model_summaries[model] = {
            "behavior_model_selected": (
                model in behavior["selected_models"]
            ),
            "residual_metric_finite_rate": scan_summary[
                "residual_metric_finite_rate"
            ],
            "routing_metric_finite_rate": scan_summary[
                "routing_metric_finite_rate"
            ],
            "prebranch_selection_interaction_max": (
                global_prebranch
            ),
            "embedding_selection_interaction_max": embedding_max,
            "selected_heads": [
                {
                    "depth": depth + 1,
                    "head": head,
                    "destination": (
                        protocol.ATTENTION_DESTINATIONS[destination]
                    ),
                    "discovery_score": score,
                }
                for score, depth, head, destination in selected_heads
            ],
            "relations": relation_results,
            "dynamic_relations": dynamic_relations,
            "model_dynamic_gate_passed": model_gate,
        }

    cross_model_profile_cosines = {}
    for left_index, left in enumerate(protocol.MODELS):
        for right in protocol.MODELS[left_index + 1:]:
            cross_model_profile_cosines[
                f"{left}::{right}"
            ] = cosine(
                normalized_profiles[left],
                normalized_profiles[right],
            )

    should_continue = (
        len(dynamic_models) >= gates["minimum_dynamic_models"]
    )
    summary = {
        "schema_version": "phase1074_analysis_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_selected_models": behavior["selected_models"],
        "dynamic_models": dynamic_models,
        "cross_model_attention_profile_cosines": (
            cross_model_profile_cosines
        ),
        "models": model_summaries,
    }
    automatic = {
        "schema_version": "phase1074_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selected_models": dynamic_models,
        "selected_model_count": len(dynamic_models),
        "should_continue_automatically": should_continue,
        "next_phase": 1075 if should_continue else None,
        "route": (
            "run_frozen_polarity_component_causal_validation"
            if should_continue
            else "stop_at_descriptive_polarity_dynamics"
        ),
        "reason": (
            "At least two behavior-valid models passed the frozen "
            "residual-transition and held-out Attention-routing gates."
            if should_continue
            else (
                "The frozen cross-model dynamics gate was not met; "
                "retain the polarity atlas without a causal mechanism "
                "claim."
            )
        ),
    }
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "dynamics_summary.json",
        summary,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT / "analysis" / "head_candidates.jsonl",
        head_records,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic,
    )
    print(json.dumps({
        "dynamic_models": dynamic_models,
        "cross_model_attention_profile_cosines": (
            cross_model_profile_cosines
        ),
        "automatic_next": automatic,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
