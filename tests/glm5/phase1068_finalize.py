#!/usr/bin/env python3
"""Aggregate Phase1068 without fitting a mechanism formula."""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1068_reasoning_generalization_protocol as protocol


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(
        np.linalg.norm(left) * np.linalg.norm(right)
    )
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def model_profile(
    metrics: list[dict[str, Any]],
    bucket_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_depth: dict[float, list[float]] = {}
    for row in metrics:
        if (
            row["bucket_id"] != bucket_id
            or row["role"] != "answer_boundary"
            or row["mean_semantic_relative_magnitude"] is None
        ):
            continue
        by_depth.setdefault(
            float(row["relative_depth"]), []
        ).append(float(
            row["mean_semantic_relative_magnitude"]
        ))
    depths = np.array(sorted(by_depth), dtype=np.float64)
    values = np.array(
        [
            sum(by_depth[depth]) / len(by_depth[depth])
            for depth in depths
        ],
        dtype=np.float64,
    )
    return depths, values


def interpolated_profile_cosine(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> float | None:
    if not len(left[0]) or not len(right[0]):
        return None
    grid = np.linspace(0.0, 1.0, 41)
    left_values = np.interp(grid, left[0], left[1])
    right_values = np.interp(grid, right[0], right[1])
    return cosine(left_values, right_values)


def late_internal_repeat(
    rows: list[dict[str, Any]],
    relation: str,
) -> dict[str, Any]:
    values = [
        float(row[
            "discovery_confirmation_direction_cosine"
        ])
        for row in rows
        if row["bucket_id"].startswith(
            f"relation_query:{relation}:"
        )
        and float(row["relative_depth"]) >= 0.65
        and row[
            "discovery_confirmation_direction_cosine"
        ] is not None
        and int(row["discovery_pair_count"]) >= 20
        and int(row["confirmation_pair_count"]) >= 20
    ]
    median = statistics.median(values) if values else None
    return {
        "late_event_count": len(values),
        "median_late_answer_discovery_confirmation_cosine": (
            median
        ),
    }


def within_model_relation_directions(
    model_name: str,
) -> list[dict[str, Any]]:
    path = (
        protocol.OUT_ROOT
        / "atlas"
        / model_name
        / "answer_directions.fp16.npz"
    )
    archive = np.load(path)
    directions = archive["mean_directions"].astype(np.float32)
    counts = archive["direction_counts"]
    bucket_ids = [str(value) for value in archive["bucket_ids"]]
    depths = archive["relative_depths"].astype(np.float64)
    bucket_index = {
        value: index for index, value in enumerate(bucket_ids)
    }
    rows = []
    for query_type in protocol.QUERY_TYPES:
        for left_relation, right_relation in itertools.combinations(
            protocol.RELATION_NAMES, 2
        ):
            left_bucket = bucket_index[
                f"relation_query:{left_relation}:{query_type}"
            ]
            right_bucket = bucket_index[
                f"relation_query:{right_relation}:{query_type}"
            ]
            values = []
            for event_index, relative_depth in enumerate(depths):
                if relative_depth < 0.65:
                    continue
                if (
                    counts[left_bucket, :, event_index].min() < 20
                    or counts[right_bucket, :, event_index].min() < 20
                ):
                    continue
                left = directions[
                    left_bucket, :, event_index, :
                ].mean(axis=0)
                right = directions[
                    right_bucket, :, event_index, :
                ].mean(axis=0)
                value = cosine(left, right)
                if value is not None:
                    values.append(value)
            rows.append({
                "schema_version": (
                    "phase1068_within_model_relation_direction.v1"
                ),
                "phase": protocol.PHASE,
                "model": model_name,
                "query_type": query_type,
                "left_relation": left_relation,
                "right_relation": right_relation,
                "late_event_count": len(values),
                "median_late_direction_cosine": (
                    statistics.median(values)
                    if values
                    else None
                ),
            })
    return rows


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {}
    metrics = {}
    directions = {}
    internal_repeat = {}
    for model_name in protocol.MODELS:
        root = protocol.OUT_ROOT / "atlas" / model_name
        summary = protocol.read_json(root / "summary.json")
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model_name} protocol digest drift")
        if summary["case_count"] != 2400:
            raise RuntimeError(f"{model_name} case count drift")
        if float(summary["identity_maximum"]) != 0.0:
            raise RuntimeError(f"{model_name} identity control failed")
        precision = summary["precision"]
        if (
            not precision["has_fp16_parameters"]
            or precision["has_bf16_parameters"]
            or precision["has_quantized_modules"]
        ):
            raise RuntimeError(f"{model_name} precision audit failed")
        summaries[model_name] = summary
        metrics[model_name] = protocol.read_jsonl(
            root / "response_metrics.jsonl"
        )
        directions[model_name] = protocol.read_jsonl(
            root / "cross_template_directions.jsonl"
        )
        internal_repeat[model_name] = {
            relation: late_internal_repeat(
                directions[model_name], relation
            )
            for relation in protocol.RELATION_NAMES
        }

    cross_model_rows = []
    for relation in protocol.RELATION_NAMES:
        for left_model, right_model in itertools.combinations(
            protocol.MODELS, 2
        ):
            value = interpolated_profile_cosine(
                model_profile(
                    metrics[left_model],
                    f"relation:{relation}",
                ),
                model_profile(
                    metrics[right_model],
                    f"relation:{relation}",
                ),
            )
            cross_model_rows.append({
                "schema_version": (
                    "phase1068_cross_model_profile.v1"
                ),
                "phase": protocol.PHASE,
                "relation": relation,
                "left_model": left_model,
                "right_model": right_model,
                "answer_residual_depth_profile_cosine": value,
            })

    within_model_rows = []
    for model_name in protocol.MODELS:
        within_model_rows.extend(
            within_model_relation_directions(model_name)
        )

    evidence_rows = []
    for relation in protocol.RELATION_NAMES:
        strong_models = [
            model_name
            for model_name in protocol.MODELS
            if summaries[model_name]["relations"][relation][
                "strong_behavior_gate_passed"
            ]
        ]
        internal_models = [
            model_name
            for model_name in strong_models
            if internal_repeat[model_name][relation][
                "median_late_answer_discovery_confirmation_cosine"
            ] is not None
            and internal_repeat[model_name][relation][
                "median_late_answer_discovery_confirmation_cosine"
            ]
            >= prereg["gates"][
                "late_answer_discovery_confirmation_cosine_min"
            ]
        ]
        profile_candidates = [
            row
            for row in cross_model_rows
            if row["relation"] == relation
            and row["left_model"] in strong_models
            and row["right_model"] in strong_models
            and row[
                "answer_residual_depth_profile_cosine"
            ] is not None
        ]
        best_profile = max(
            profile_candidates,
            key=lambda row: row[
                "answer_residual_depth_profile_cosine"
            ],
            default=None,
        )
        profile_value = (
            best_profile[
                "answer_residual_depth_profile_cosine"
            ]
            if best_profile
            else None
        )
        selectable = bool(
            len(strong_models)
            >= prereg["gates"]["minimum_repeated_models"]
            and len(internal_models)
            >= prereg["gates"]["minimum_repeated_models"]
            and profile_value is not None
            and profile_value
            >= prereg["gates"][
                "cross_model_depth_profile_cosine_min"
            ]
        )
        internal_values = [
            internal_repeat[model_name][relation][
                "median_late_answer_discovery_confirmation_cosine"
            ]
            for model_name in internal_models
        ]
        evidence_rows.append({
            "schema_version": "phase1068_relation_evidence.v1",
            "phase": protocol.PHASE,
            "relation": relation,
            "strong_behavior_models": strong_models,
            "internal_repeat_models": internal_models,
            "median_internal_repeat_across_passing_models": (
                statistics.median(internal_values)
                if internal_values
                else None
            ),
            "best_cross_model_profile": best_profile,
            "selectable_for_generalized_causal_test": selectable,
            "selection_score": (
                len(strong_models)
                + statistics.median(internal_values)
                + float(profile_value)
                if selectable
                else None
            ),
        })

    selectable = sorted(
        [
            row
            for row in evidence_rows
            if row["selectable_for_generalized_causal_test"]
        ],
        key=lambda row: float(row["selection_score"]),
        reverse=True,
    )
    should_continue = (
        len(selectable)
        >= prereg["gates"]["minimum_strong_relations"]
    )
    selected_relations = [
        row["relation"] for row in selectable[:3]
    ]
    automatic_next = {
        "schema_version": "phase1068_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": should_continue,
        "selected_relations": selected_relations,
        "selectable_relation_count": len(selectable),
        "route": (
            "build_phase1069_role_position_orthogonal_causal_test"
            if should_continue
            else "stop_and_repair_reasoning_behavior_protocol"
        ),
        "rationale": (
            "Continuation follows frozen strong-behavior, "
            "cross-template direction, and cross-model residual-profile "
            "gates; no K/V channel formula is assumed."
        ),
    }
    analysis_root = protocol.OUT_ROOT / "analysis"
    protocol.write_jsonl(
        analysis_root / "relation_evidence.jsonl",
        evidence_rows,
    )
    protocol.write_jsonl(
        analysis_root / "cross_model_depth_profiles.jsonl",
        cross_model_rows,
    )
    protocol.write_jsonl(
        analysis_root
        / "within_model_relation_direction_cosines.jsonl",
        within_model_rows,
    )
    protocol.write_json(
        analysis_root / "automatic_next.json",
        automatic_next,
    )
    aggregate = {
        "schema_version": "phase1068_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_summaries": summaries,
        "internal_repeat": internal_repeat,
        "relation_evidence": evidence_rows,
        "cross_model_depth_profiles": cross_model_rows,
        "within_model_relation_direction_cosines": (
            within_model_rows
        ),
        "automatic_next_decision": automatic_next,
        "interpretation": {
            "response_repeat_is_not_causal_transport": True,
            "shared_direction_is_not_a_reasoning_algorithm": True,
            "cross_model_profiles_do_not_align_hidden_bases": True,
        },
    }
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json",
        aggregate,
    )
    print(json.dumps(
        {
            "phase": protocol.PHASE,
            "evidence": evidence_rows,
            "automatic_next": automatic_next,
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
