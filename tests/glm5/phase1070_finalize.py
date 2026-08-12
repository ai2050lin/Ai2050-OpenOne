#!/usr/bin/env python3
"""Finalize the Phase1070 process/answer orthogonal atlas."""

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

import phase1070_process_answer_protocol as protocol


PROFILE_METRICS = (
    "mean_process_did_relative_magnitude",
    "mean_process_lexical_reuse_cosine",
    "mean_process_answer_invariance_cosine",
)


def finite_values(values: list[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def median(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return float(np.median(clean)) if clean else None


def maximum(values: list[Any]) -> float | None:
    clean = finite_values(values)
    return max(clean) if clean else None


def cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return float(np.dot(a, b) / denominator)


def normalized_metric(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [float(value) for value in array / norm]


def relation_profile(
    rows: list[dict[str, Any]],
    relation: str,
    split: str,
    reverse_depth: bool = False,
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["relation"] == relation
        and row["split"] == split
        and row["conditioning"] == "all"
        and row["role"] in protocol.PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[(row["query_type"], row["role"])].append(row)
    profile = []
    for metric in PROFILE_METRICS:
        metric_values = []
        for key in sorted(groups):
            ordered = sorted(
                groups[key],
                key=lambda row: float(row["relative_depth"]),
                reverse=reverse_depth,
            )
            metric_values.extend([
                float(row[metric])
                if row[metric] is not None
                and math.isfinite(float(row[metric]))
                else 0.0
                for row in ordered
            ])
        profile.extend(normalized_metric(metric_values))
    return profile


def split_evidence(
    response_rows: list[dict[str, Any]],
    readout_rows: list[dict[str, Any]],
    relation: str,
    split: str,
) -> dict[str, Any]:
    process_rows = [
        row
        for row in response_rows
        if row["relation"] == relation
        and row["split"] == split
        and row["conditioning"] == "all"
        and row["role"] in protocol.PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    embedding_rows = [
        row
        for row in response_rows
        if row["relation"] == relation
        and row["split"] == split
        and row["conditioning"] == "all"
        and int(row["depth"]) == 0
    ]
    late_readout = [
        row
        for row in readout_rows
        if row["relation"] == relation
        and row["split"] == split
        and row["conditioning"] == "all"
        and float(row["relative_depth"])
        >= protocol.GATES["late_depth_start"]
    ]
    did = median([
        row["mean_process_did_relative_magnitude"]
        for row in process_rows
    ])
    lexical = median([
        row["mean_process_lexical_reuse_cosine"]
        for row in process_rows
    ])
    answer_invariance = median([
        row["mean_process_answer_invariance_cosine"]
        for row in process_rows
    ])
    process_answer_overlap = median([
        row["mean_process_answer_absolute_cosine"]
        for row in process_rows
    ])
    embedding_max = maximum([
        row["mean_process_did_relative_magnitude"]
        for row in embedding_rows
    ])
    readout_ratio = median([
        row["absolute_process_to_answer_readout_ratio"]
        for row in late_readout
    ])
    checks = {
        "process_did_magnitude": (
            did is not None
            and did
            >= protocol.GATES[
                "process_did_relative_magnitude_min"
            ]
        ),
        "process_lexical_reuse": (
            lexical is not None
            and lexical
            >= protocol.GATES[
                "process_lexical_reuse_cosine_min"
            ]
        ),
        "process_answer_invariance": (
            answer_invariance is not None
            and answer_invariance
            >= protocol.GATES[
                "process_answer_invariance_cosine_min"
            ]
        ),
        "embedding_control": (
            embedding_max is not None
            and embedding_max
            <= protocol.GATES[
                "embedding_process_did_relative_magnitude_max"
            ]
        ),
        "readout_separation": (
            readout_ratio is not None
            and readout_ratio
            <= protocol.GATES[
                "process_to_answer_readout_ratio_max"
            ]
        ),
    }
    return {
        "split": split,
        "process_window_row_count": len(process_rows),
        "median_process_did_relative_magnitude": did,
        "median_process_lexical_reuse_cosine": lexical,
        "median_process_answer_invariance_cosine": (
            answer_invariance
        ),
        "median_process_answer_absolute_cosine": (
            process_answer_overlap
        ),
        "embedding_process_did_max": embedding_max,
        "late_process_to_answer_readout_ratio_median": (
            readout_ratio
        ),
        "checks": checks,
        "all_split_checks_passed": all(checks.values()),
    }


def pooled_model_profile(
    rows: list[dict[str, Any]],
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["conditioning"] == "all"
        and row["role"] in protocol.PROCESS_ROLES
    ]
    bins = tuple(range(11))
    profile = []
    for metric in PROFILE_METRICS:
        values = []
        for role in protocol.PROCESS_ROLES:
            for depth_bin in bins:
                bucket = [
                    row[metric]
                    for row in selected
                    if row["role"] == role
                    and min(
                        10,
                        int(round(
                            float(row["relative_depth"]) * 10
                        )),
                    ) == depth_bin
                ]
                value = median(bucket)
                values.append(value if value is not None else 0.0)
        profile.extend(normalized_metric(values))
    return profile


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1070 protocol audit failed")

    relation_rows = []
    model_rows = []
    model_profiles = {}
    all_response_rows: dict[str, list[dict[str, Any]]] = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(atlas / "summary.json")
        response_rows = protocol.read_jsonl(
            atlas / "response_metrics.jsonl"
        )
        readout_rows = protocol.read_jsonl(
            atlas / "local_readout_metrics.jsonl"
        )
        all_response_rows[model] = response_rows
        numerical_checks = {
            "candidate_finite": (
                float(summary["candidate_finite_rate"])
                >= prereg["gates"]["candidate_finite_rate_min"]
            ),
            "residual_metric_finite": (
                float(summary["residual_metric_finite_rate"])
                >= prereg["gates"]["internal_finite_rate_min"]
            ),
            "internal_readout_finite": (
                float(summary["internal_readout_finite_rate"])
                >= prereg["gates"]["internal_finite_rate_min"]
            ),
        }
        numerical_passed = all(numerical_checks.values())
        selected_relations = []
        for relation in protocol.RELATION_NAMES:
            discovery = split_evidence(
                response_rows,
                readout_rows,
                relation,
                "discovery",
            )
            confirmation = split_evidence(
                response_rows,
                readout_rows,
                relation,
                "confirmation",
            )
            discovery_profile = relation_profile(
                response_rows, relation, "discovery"
            )
            confirmation_profile = relation_profile(
                response_rows, relation, "confirmation"
            )
            reversed_profile = relation_profile(
                response_rows,
                relation,
                "confirmation",
                reverse_depth=True,
            )
            matched_cosine = cosine(
                discovery_profile, confirmation_profile
            )
            reversed_cosine = cosine(
                discovery_profile, reversed_profile
            )
            reversal_gap = (
                matched_cosine - reversed_cosine
                if matched_cosine is not None
                and reversed_cosine is not None
                else None
            )
            profile_checks = {
                "discovery_confirmation_profile": (
                    matched_cosine is not None
                    and matched_cosine
                    >= prereg["gates"][
                        "process_discovery_confirmation_profile_cosine_min"
                    ]
                ),
                "depth_reversal_specificity": (
                    reversal_gap is not None
                    and reversal_gap
                    >= prereg["gates"][
                        "process_depth_reversal_gap_min"
                    ]
                ),
            }
            behavior_passed = bool(
                summary["relations"][relation][
                    "strong_behavior_gate_passed"
                ]
            )
            relation_passed = bool(
                behavior_passed
                and numerical_passed
                and discovery["all_split_checks_passed"]
                and confirmation["all_split_checks_passed"]
                and all(profile_checks.values())
            )
            if relation_passed:
                selected_relations.append(relation)
            relation_rows.append({
                "schema_version": (
                    "phase1070_relation_evidence.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "relation": relation,
                "behavior_gate_passed": behavior_passed,
                "numerical_gate_passed": numerical_passed,
                "numerical_checks": numerical_checks,
                "discovery": discovery,
                "confirmation": confirmation,
                "discovery_confirmation_profile_cosine": (
                    matched_cosine
                ),
                "depth_reversed_confirmation_cosine": (
                    reversed_cosine
                ),
                "profile_depth_reversal_gap": reversal_gap,
                "profile_checks": profile_checks,
                "process_relation_gate_passed": relation_passed,
            })
        model_passed = (
            len(selected_relations)
            >= prereg["gates"][
                "minimum_strong_relations_per_model"
            ]
        )
        model_rows.append({
            "schema_version": "phase1070_model_gate.v1",
            "phase": protocol.PHASE,
            "model": model,
            "candidate_finite_rate": summary[
                "candidate_finite_rate"
            ],
            "residual_metric_finite_rate": summary[
                "residual_metric_finite_rate"
            ],
            "internal_readout_finite_rate": summary[
                "internal_readout_finite_rate"
            ],
            "numerical_checks": numerical_checks,
            "numerical_gate_passed": numerical_passed,
            "selected_relations": selected_relations,
            "selected_relation_count": len(selected_relations),
            "process_model_gate_passed": model_passed,
        })
        model_profiles[model] = pooled_model_profile(response_rows)

    cross_model_rows = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        cross_model_rows.append({
            "schema_version": (
                "phase1070_cross_model_process_profile.v1"
            ),
            "phase": protocol.PHASE,
            "left_model": left,
            "right_model": right,
            "pooled_depth_role_profile_cosine": cosine(
                model_profiles[left], model_profiles[right]
            ),
            "interpretation": (
                "Descriptive normalized scalar-profile similarity; it is "
                "not neuron correspondence or causal homology."
            ),
        })

    selected_models = [
        row["model"]
        for row in model_rows
        if row["process_model_gate_passed"]
    ]
    should_continue = (
        len(selected_models)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    automatic_next = {
        "schema_version": "phase1070_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": should_continue,
        "selected_models": selected_models,
        "repeated_model_count": len(selected_models),
        "route": (
            "authorize_component_localization"
            if should_continue
            else "stop_at_process_answer_atlas"
        ),
        "next_phase": (
            prereg["automatic_next"]["next_phase"]
            if should_continue else None
        ),
        "rationale": (
            "Automation requires the frozen behavior, numerical, matched "
            "difference-in-differences, answer/lexical reuse, split "
            "replication, depth-specificity, and readout-separation gates "
            "in at least two relations for at least two models."
        ),
    }
    atlas_summary = {
        "schema_version": "phase1070_atlas_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "primary_evidence": (
            "All-pair residual atlas. Behavior-conditioned rows are "
            "secondary strata and cannot rescue a failed all-pair gate."
        ),
        "core_measurement": prereg["core_contrast"],
        "model_gates": model_rows,
        "cross_model_profiles": cross_model_rows,
        "automatic_next": automatic_next,
        "interpretation_limits": prereg["interpretation_limits"],
    }
    protocol.write_jsonl(
        protocol.OUT_ROOT / "analysis" / "relation_evidence.jsonl",
        relation_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT / "analysis" / "model_gates.jsonl",
        model_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "cross_model_process_profiles.jsonl",
        cross_model_rows,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic_next,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "atlas_summary.json",
        atlas_summary,
    )
    print({
        "phase": protocol.PHASE,
        "selected_models": selected_models,
        "automatic_next": automatic_next,
    })


if __name__ == "__main__":
    main()
