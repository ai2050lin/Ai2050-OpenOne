#!/usr/bin/env python3
"""Finalize Phase1065 without fitting a language-mechanism formula."""

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

import phase1065_multimode_response_atlas_protocol as protocol


def finite_median(values: list[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(finite)) if finite else None


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    numerator = float(np.dot(
        a.astype(np.float64, copy=False),
        b.astype(np.float64, copy=False),
    ))
    denominator = float(
        np.linalg.norm(a.astype(np.float64, copy=False))
        * np.linalg.norm(b.astype(np.float64, copy=False))
    )
    return numerator / denominator if denominator > 1e-12 else None


def profile_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    return vector_cosine(a, b)


def resampled_profile(
    rows: list[dict[str, Any]],
    component: str,
    role: str,
) -> np.ndarray | None:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["component"] != component or row["role"] != role:
            continue
        value = row["mean_semantic_relative_magnitude"]
        if value is None:
            continue
        grouped[float(row["relative_depth"])].append(float(value))
    if len(grouped) < 2:
        return None
    x = np.array(sorted(grouped), dtype=np.float64)
    y = np.array([
        float(np.mean(grouped[value])) for value in x
    ], dtype=np.float64)
    grid = np.linspace(float(x.min()), float(x.max()), 21)
    return np.interp(grid, x, y)


def load_model(model_name: str) -> dict[str, Any]:
    root = protocol.OUT_ROOT / "atlas" / model_name
    summary = protocol.read_json(root / "summary.json")
    metrics = protocol.read_jsonl(root / "response_metrics.jsonl")
    arrays = np.load(root / "mean_directions.fp16.npz")
    return {
        "summary": summary,
        "metrics": metrics,
        "directions": arrays["mean_directions"].astype(np.float32),
        "counts": arrays["semantic_counts"],
        "families": [str(value) for value in arrays["family_names"]],
        "splits": [str(value) for value in arrays["split_names"]],
        "roles": [str(value) for value in arrays["role_names"]],
        "events": [str(value) for value in arrays["event_ids"]],
    }


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1065 protocol audit failed")
    model_data = {
        model: load_model(model) for model in protocol.MODELS
    }
    for model, data in model_data.items():
        summary = data["summary"]
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"protocol drift for {model}")
        precision = summary["precision"]
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError(f"precision drift for {model}")
        if float(summary["identity_maximum"]) != 0.0:
            raise RuntimeError(f"identity control failed for {model}")

    cross_template_rows = []
    internal_summary: dict[str, dict[str, dict[str, Any]]] = {
        model: {} for model in protocol.MODELS
    }
    for model, data in model_data.items():
        family_index = {
            value: index for index, value in enumerate(data["families"])
        }
        split_index = {
            value: index for index, value in enumerate(data["splits"])
        }
        role_index = {
            value: index for index, value in enumerate(data["roles"])
        }
        event_index = {
            value: index for index, value in enumerate(data["events"])
        }
        metric_index = {
            (
                str(row["family"]),
                str(row["split"]),
                str(row["event_id"]),
                str(row["role"]),
            ): row
            for row in data["metrics"]
        }
        for family in protocol.FAMILIES:
            late_answer_cosines = []
            late_branch_cosines = []
            late_specificity = []
            repeated_event_count = 0
            for event_id in data["events"]:
                discovery_metric = metric_index[
                    (family, "discovery", event_id, "answer_boundary")
                ]
                confirmation_metric = metric_index[
                    (family, "confirmation", event_id, "answer_boundary")
                ]
                discovery = data["directions"][
                    family_index[family],
                    split_index["discovery"],
                    event_index[event_id],
                    role_index["answer_boundary"],
                ]
                confirmation = data["directions"][
                    family_index[family],
                    split_index["confirmation"],
                    event_index[event_id],
                    role_index["answer_boundary"],
                ]
                cosine = vector_cosine(discovery, confirmation)
                row = {
                    "schema_version": (
                        "phase1065_cross_template_direction.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model,
                    "family": family,
                    "event_id": event_id,
                    "component": discovery_metric["component"],
                    "depth": discovery_metric["depth"],
                    "relative_depth": discovery_metric["relative_depth"],
                    "role": "answer_boundary",
                    "discovery_confirmation_direction_cosine": cosine,
                    "discovery_pair_count": discovery_metric[
                        "semantic_pair_count"
                    ],
                    "confirmation_pair_count": confirmation_metric[
                        "semantic_pair_count"
                    ],
                    "discovery_surface_branch_cosine": discovery_metric[
                        "mean_surface_branch_semantic_cosine"
                    ],
                    "confirmation_surface_branch_cosine": (
                        confirmation_metric[
                            "mean_surface_branch_semantic_cosine"
                        ]
                    ),
                }
                cross_template_rows.append(row)
                if float(row["relative_depth"]) < 0.5:
                    continue
                if cosine is not None:
                    late_answer_cosines.append(cosine)
                    if cosine >= prereg["gates"][
                        "internal_discovery_confirmation_cosine_min"
                    ]:
                        repeated_event_count += 1
                late_branch_cosines.extend([
                    discovery_metric[
                        "mean_surface_branch_semantic_cosine"
                    ],
                    confirmation_metric[
                        "mean_surface_branch_semantic_cosine"
                    ],
                ])
                for metric in (discovery_metric, confirmation_metric):
                    semantic = metric[
                        "mean_semantic_relative_magnitude"
                    ]
                    surface = metric[
                        "mean_surface_relative_magnitude"
                    ]
                    if (
                        semantic is not None
                        and surface is not None
                        and float(surface) > 1e-12
                    ):
                        late_specificity.append(
                            float(semantic) / float(surface)
                        )
            median_direction = finite_median(late_answer_cosines)
            median_branch = finite_median(late_branch_cosines)
            behavior_row = data["summary"]["families"][family]
            repeat_gate = bool(
                behavior_row["behavior_gate_passed"]
                and median_direction is not None
                and median_direction
                >= prereg["gates"][
                    "internal_discovery_confirmation_cosine_min"
                ]
                and median_branch is not None
                and median_branch
                >= prereg["gates"][
                    "internal_discovery_confirmation_cosine_min"
                ]
            )
            internal_summary[model][family] = {
                "median_late_answer_discovery_confirmation_cosine": (
                    median_direction
                ),
                "median_late_answer_surface_branch_cosine": (
                    median_branch
                ),
                "median_late_semantic_surface_magnitude_ratio": (
                    finite_median(late_specificity)
                ),
                "late_repeated_event_count": repeated_event_count,
                "internal_repeat_gate_passed": repeat_gate,
            }

    cross_model_rows = []
    profiles: dict[
        tuple[str, str, str, str], np.ndarray
    ] = {}
    for model, data in model_data.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in data["metrics"]:
            grouped[str(row["family"])].append(row)
        for family in protocol.FAMILIES:
            for component in (
                "residual",
                "attention_output",
                "mlp_output",
            ):
                for role in protocol.CAPTURE_ROLES:
                    profile = resampled_profile(
                        grouped[family], component, role
                    )
                    if profile is not None:
                        profiles[
                            (model, family, component, role)
                        ] = profile
    for family in protocol.FAMILIES:
        for left_index, left in enumerate(protocol.MODELS):
            for right in protocol.MODELS[left_index + 1:]:
                for component in (
                    "residual",
                    "attention_output",
                    "mlp_output",
                ):
                    for role in protocol.CAPTURE_ROLES:
                        a = profiles.get((left, family, component, role))
                        b = profiles.get((right, family, component, role))
                        if a is None or b is None:
                            continue
                        cross_model_rows.append({
                            "schema_version": (
                                "phase1065_cross_model_profile.v1"
                            ),
                            "phase": protocol.PHASE,
                            "family": family,
                            "model_left": left,
                            "model_right": right,
                            "component": component,
                            "role": role,
                            "normalized_depth_profile_cosine": (
                                profile_cosine(a, b)
                            ),
                        })

    family_evidence = []
    selectable = []
    for family in protocol.FAMILIES:
        behavior_models = [
            model for model in protocol.MODELS
            if model_data[model]["summary"]["families"][family][
                "behavior_gate_passed"
            ]
        ]
        strong_models = [
            model for model in protocol.MODELS
            if model_data[model]["summary"]["families"][family][
                "strong_behavior_gate_passed"
            ]
        ]
        internal_models = [
            model for model in protocol.MODELS
            if internal_summary[model][family][
                "internal_repeat_gate_passed"
            ]
        ]
        pair_rows = [
            row for row in cross_model_rows
            if row["family"] == family
            and row["component"] == "residual"
            and row["role"] == "answer_boundary"
            and row["model_left"] in strong_models
            and row["model_right"] in strong_models
        ]
        best_profile = max(
            (
                float(row["normalized_depth_profile_cosine"])
                for row in pair_rows
                if row["normalized_depth_profile_cosine"] is not None
            ),
            default=None,
        )
        cross_model_gate = bool(
            best_profile is not None
            and best_profile
            >= prereg["gates"][
                "cross_model_depth_profile_cosine_min"
            ]
        )
        selectable_family = bool(
            family != "translation_word"
            and len(strong_models)
            >= prereg["gates"]["minimum_repeated_models"]
            and len(set(strong_models) & set(internal_models))
            >= prereg["gates"]["minimum_repeated_models"]
            and cross_model_gate
        )
        model_scores = [
            internal_summary[model][family][
                "median_late_answer_discovery_confirmation_cosine"
            ]
            for model in set(strong_models) & set(internal_models)
        ]
        score = (
            min(float(value) for value in model_scores if value is not None)
            + float(best_profile)
            if selectable_family and best_profile is not None
            else None
        )
        evidence = {
            "schema_version": "phase1065_family_evidence.v1",
            "phase": protocol.PHASE,
            "family": family,
            "behavior_gate_models": behavior_models,
            "strong_behavior_gate_models": strong_models,
            "internal_repeat_gate_models": internal_models,
            "best_cross_model_answer_residual_profile_cosine": (
                best_profile
            ),
            "cross_model_profile_gate_passed": cross_model_gate,
            "selectable_for_causal_followup": selectable_family,
            "selection_score": score,
        }
        family_evidence.append(evidence)
        if selectable_family:
            selectable.append(evidence)

    selected = None
    if selectable:
        selected = max(
            selectable,
            key=lambda row: (
                float(row["selection_score"]),
                row["family"],
            ),
        )["family"]
    automatic = {
        "schema_version": "phase1065_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": selected is not None,
        "selected_family": selected,
        "route": (
            "build_phase1066_independent_role_conditioned_causal_test"
            if selected is not None
            else "stop_after_descriptive_atlas_and_repair_behavior_or_repeat"
        ),
        "rationale": (
            "Selection follows frozen behavior, natural-generation, "
            "cross-template, and cross-model profile gates. Translation "
            "K/V assumptions were not used to select the family."
        ),
    }
    aggregate = {
        "schema_version": "phase1065_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_summaries": {
            model: model_data[model]["summary"]
            for model in protocol.MODELS
        },
        "internal_repeat_summary": internal_summary,
        "family_evidence": family_evidence,
        "automatic_next_decision": automatic,
        "claim_boundary": {
            "supported": [
                "Behavior-qualified semantic branch differences can be mapped across complete model depth.",
                "Discovery-confirmation and cross-model depth-profile repetition are directly measured.",
            ],
            "not_supported": [
                "A repeated response field is a causal mechanism.",
                "All language patterns use translation K/V transport.",
                "A rare-word differential is the word's complete meaning.",
                "Any measured structure is biologically optimal.",
                "The current atlas is a closed mathematical theory.",
            ],
        },
    }
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "cross_template_directions.jsonl",
        cross_template_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT
        / "analysis"
        / "cross_model_depth_profiles.jsonl",
        cross_model_rows,
    )
    protocol.write_jsonl(
        protocol.OUT_ROOT / "analysis" / "family_evidence.jsonl",
        family_evidence,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json",
        aggregate,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "family_evidence": family_evidence,
        "automatic_next": automatic,
    }, ensure_ascii=False, indent=2))
    return aggregate


def main() -> None:
    finalize()


if __name__ == "__main__":
    main()
