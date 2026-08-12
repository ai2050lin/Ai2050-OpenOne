#!/usr/bin/env python3
"""Finalize the Phase1071 causal-exposure-aware pattern-family atlas."""

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

import phase1071_exposure_pattern_protocol as protocol


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


def normalized(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [float(value) for value in array / norm]


def relation_profile(
    rows: list[dict[str, Any]],
    relation: str,
    split: str | None = None,
    reverse_depth: bool = False,
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["relation"] == relation
        and (split is None or row["split"] == split)
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in selected:
        groups[(row["query_type"], row["role"])].append(row)
    profile = []
    for metric in PROFILE_METRICS:
        values = []
        for key in sorted(groups):
            ordered = sorted(
                groups[key],
                key=lambda row: float(row["relative_depth"]),
                reverse=reverse_depth,
            )
            values.extend([
                float(row[metric])
                if row[metric] is not None
                and math.isfinite(float(row[metric]))
                else 0.0
                for row in ordered
            ])
        profile.extend(normalized(values))
    return profile


def role_median(
    rows: list[dict[str, Any]],
    role: str,
    metric: str,
) -> float | None:
    return median([
        row[metric] for row in rows if row["role"] == role
    ])


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
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
        and float(row["relative_depth"])
        >= protocol.GATES["process_window_start"]
    ]
    hard_negative_rows = [
        row
        for row in response_rows
        if row["relation"] == relation
        and row["split"] == split
        and row["conditioning"] == "all"
        and row["role"] in protocol.HARD_NEGATIVE_ROLES
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
    evidence_did = role_median(
        process_rows,
        "evidence_probe",
        "mean_process_did_relative_magnitude",
    )
    answer_did = role_median(
        process_rows,
        "answer_boundary",
        "mean_process_did_relative_magnitude",
    )
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
    surface_difference = median([
        row["mean_surface_relative_magnitude"]
        for row in process_rows
    ])
    hard_negative_max = maximum([
        row["mean_process_did_relative_magnitude"]
        for row in hard_negative_rows
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
        "evidence_probe_did": (
            evidence_did is not None
            and evidence_did
            >= protocol.GATES[
                "evidence_probe_did_relative_magnitude_min"
            ]
        ),
        "answer_boundary_did": (
            answer_did is not None
            and answer_did
            >= protocol.GATES[
                "answer_boundary_did_relative_magnitude_min"
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
        "hard_negative_control": (
            hard_negative_max is not None
            and hard_negative_max
            <= protocol.GATES["hard_negative_process_did_max"]
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
        "primary_process_row_count": len(process_rows),
        "hard_negative_row_count": len(hard_negative_rows),
        "median_process_did_relative_magnitude": did,
        "median_evidence_probe_did_relative_magnitude": evidence_did,
        "median_answer_boundary_did_relative_magnitude": answer_did,
        "median_process_lexical_reuse_cosine": lexical,
        "median_process_answer_invariance_cosine": (
            answer_invariance
        ),
        "median_process_answer_absolute_cosine": (
            process_answer_overlap
        ),
        "median_lexical_surface_relative_magnitude": (
            surface_difference
        ),
        "hard_negative_process_did_max": hard_negative_max,
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
    values = []
    bins = tuple(range(11))
    selected = [
        row
        for row in rows
        if row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
    ]
    for metric in PROFILE_METRICS:
        metric_values = []
        for role in protocol.PRIMARY_PROCESS_ROLES:
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
                metric_values.append(
                    value if value is not None else 0.0
                )
        values.extend(normalized(metric_values))
    return values


def binned_relation_profile(
    rows: list[dict[str, Any]],
    relation: str,
) -> list[float]:
    """Create a fixed-width profile for cross-model layer-count mismatch."""
    selected = [
        row
        for row in rows
        if row["relation"] == relation
        and row["conditioning"] == "all"
        and row["role"] in protocol.PRIMARY_PROCESS_ROLES
    ]
    values = []
    for metric in PROFILE_METRICS:
        metric_values = []
        for role in protocol.PRIMARY_PROCESS_ROLES:
            for depth_bin in range(11):
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
                metric_values.append(
                    value if value is not None else 0.0
                )
        values.extend(normalized(metric_values))
    return values


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1071 protocol audit failed")

    relation_rows = []
    model_rows = []
    response_by_model = {}
    profile_by_model = {}
    relation_profiles: dict[tuple[str, str], list[float]] = {}
    binned_relation_profiles: dict[
        tuple[str, str], list[float]
    ] = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(atlas / "summary.json")
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"Phase1071 digest drift: {model}")
        response_rows = protocol.read_jsonl(
            atlas / "response_metrics.jsonl"
        )
        readout_rows = protocol.read_jsonl(
            atlas / "local_readout_metrics.jsonl"
        )
        response_by_model[model] = response_rows
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
            relation_profiles[(model, relation)] = relation_profile(
                response_rows, relation, split=None
            )
            binned_relation_profiles[(model, relation)] = (
                binned_relation_profile(response_rows, relation)
            )
            relation_rows.append({
                "schema_version": (
                    "phase1071_relation_evidence.v1"
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
            "schema_version": "phase1071_model_gate.v1",
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
        profile_by_model[model] = pooled_model_profile(response_rows)

    within_model_family_rows = []
    for model in protocol.MODELS:
        for left, right in itertools.combinations(
            protocol.RELATION_NAMES, 2
        ):
            within_model_family_rows.append({
                "schema_version": (
                    "phase1071_within_model_family_profile.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "left_relation": left,
                "right_relation": right,
                "normalized_profile_cosine": cosine(
                    relation_profiles[(model, left)],
                    relation_profiles[(model, right)],
                ),
                "interpretation": (
                    "Descriptive similarity of scalar depth-role "
                    "profiles; not proof of a shared circuit."
                ),
            })

    cross_model_rows = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        cross_model_rows.append({
            "schema_version": (
                "phase1071_cross_model_process_profile.v1"
            ),
            "phase": protocol.PHASE,
            "left_model": left,
            "right_model": right,
            "pooled_depth_role_profile_cosine": cosine(
                profile_by_model[left], profile_by_model[right]
            ),
            "per_relation_profile_cosine": {
                relation: cosine(
                    binned_relation_profiles[(left, relation)],
                    binned_relation_profiles[(right, relation)],
                )
                for relation in protocol.RELATION_NAMES
            },
            "interpretation": (
                "Functional depth-role similarity only; not neuron "
                "correspondence or causal homology."
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
        "schema_version": "phase1071_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": should_continue,
        "selected_models": selected_models,
        "repeated_model_count": len(selected_models),
        "route": (
            "authorize_exposure_preserving_component_localization"
            if should_continue
            else "stop_at_exposure_pattern_atlas"
        ),
        "next_phase": (
            prereg["automatic_next"]["next_phase"]
            if should_continue else None
        ),
        "rationale": (
            "Automation requires frozen behavior, numerical, exact "
            "hard-negative, post-evidence DiD, lexical/answer reuse, "
            "split-depth, and readout-separation gates in at least two "
            "relations for at least two models."
        ),
    }
    atlas_summary = {
        "schema_version": "phase1071_atlas_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "calibration_protocol_digest": prereg[
            "calibration_protocol_digest"
        ],
        "selected_prompt_style": prereg["selected_prompt_style"],
        "primary_evidence": (
            "All-pair residual atlas at fixed post-evidence probes. "
            "Behavior-conditioned rows remain secondary strata."
        ),
        "core_measurement": prereg["core_contrast"],
        "role_exposure": prereg["role_exposure"],
        "model_gates": model_rows,
        "within_model_family_profiles": within_model_family_rows,
        "cross_model_profiles": cross_model_rows,
        "automatic_next": automatic_next,
        "interpretation_limits": prereg["interpretation_limits"],
    }
    analysis_root = protocol.OUT_ROOT / "analysis"
    protocol.write_jsonl(
        analysis_root / "relation_evidence.jsonl",
        relation_rows,
    )
    protocol.write_jsonl(
        analysis_root / "model_gates.jsonl",
        model_rows,
    )
    protocol.write_jsonl(
        analysis_root / "within_model_family_profiles.jsonl",
        within_model_family_rows,
    )
    protocol.write_jsonl(
        analysis_root / "cross_model_process_profiles.jsonl",
        cross_model_rows,
    )
    protocol.write_json(
        analysis_root / "automatic_next.json",
        automatic_next,
    )
    protocol.write_json(
        analysis_root / "atlas_summary.json",
        atlas_summary,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "selected_models": selected_models,
        "automatic_next": automatic_next,
    }), flush=True)


if __name__ == "__main__":
    main()
