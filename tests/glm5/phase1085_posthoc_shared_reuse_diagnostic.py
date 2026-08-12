#!/usr/bin/env python3
"""Describe the shared response skeleton in frozen Phase1085 aggregates.

The registered identity tests center profiles across attributes.  That is the
right test for attribute-specific structure, but it deliberately removes any
response shape shared by all attributes.  This offline diagnostic measures
that removed common shape without making new model calls or upgrading gates.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1085_direct_entity_attribute_protocol as protocol

sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol
import phase1082_semantic_output_operation_world_finalize as analysis


analysis.protocol = protocol
analysis.base.protocol = protocol
analysis.base.DEPTH_GRID = np.linspace(
    protocol.TARGET_RELATIVE_DEPTH_MIN,
    protocol.TARGET_RELATIVE_DEPTH_MAX,
    7,
)

EPSILON = 1e-12
ROLE = "answer_boundary"
FIELDS = ("content_route", "duplicate_route")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > EPSILON else np.zeros_like(vector)


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= EPSILON or right_norm <= EPSILON:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def safe_median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def profile(
    rows: list[dict[str, Any]],
    operation: str,
    world: str,
    split: str,
    field: str,
) -> np.ndarray:
    value = analysis.base.build_profile(
        rows,
        f"{operation}__{world}",
        split,
        field,
        roles=(ROLE,),
    )
    return unit(value)


def world_bank(
    rows: list[dict[str, Any]], world: str, split: str, field: str
) -> np.ndarray:
    return np.stack([
        profile(rows, operation, world, split, field)
        for operation in protocol.OPERATIONS
    ])


def centroid(bank: np.ndarray) -> np.ndarray:
    return np.mean(bank, axis=0)


def pairwise_coherence(bank: np.ndarray) -> float | None:
    values = [
        cosine(bank[left], bank[right])
        for left, right in itertools.combinations(range(len(bank)), 2)
    ]
    return safe_mean(values)


def shared_differential_energy(bank: np.ndarray) -> dict[str, float | None]:
    common = centroid(bank)
    total = float(np.mean(np.sum(bank * bank, axis=1)))
    shared = float(np.dot(common, common))
    differential = float(np.mean(np.sum((bank - common) ** 2, axis=1)))
    if total <= EPSILON:
        return {
            "shared_fraction": None,
            "attribute_differential_fraction": None,
        }
    return {
        "shared_fraction": shared / total,
        "attribute_differential_fraction": differential / total,
    }


def split_direction_summary(
    rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    column = f"{field}_discovery_confirmation_cosine"
    selected = [
        float(row[column])
        for row in rows
        if row["conditioning"] == "all_finite"
        and row["role"] == ROLE
        and row[column] is not None
        and math.isfinite(float(row[column]))
    ]
    by_component = {}
    for component in ("residual", "attention_output", "mlp_output"):
        values = [
            float(row[column])
            for row in rows
            if row["conditioning"] == "all_finite"
            and row["role"] == ROLE
            and row["component"] == component
            and row[column] is not None
            and math.isfinite(float(row[column]))
        ]
        by_component[component] = {
            "count": len(values),
            "mean": safe_mean(values),
            "median": safe_median(values),
            "positive_fraction": (
                sum(value > 0 for value in values) / len(values) if values else None
            ),
        }
    return {
        "count": len(selected),
        "mean": safe_mean(selected),
        "median": safe_median(selected),
        "positive_fraction": (
            sum(value > 0 for value in selected) / len(selected)
            if selected else None
        ),
        "by_component": by_component,
    }


def content_answer_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["mean_content_answer_cosine"])
        for row in rows
        if row["conditioning"] == "all_finite"
        and row["role"] == ROLE
        and row["mean_content_answer_cosine"] is not None
        and math.isfinite(float(row["mean_content_answer_cosine"]))
    ]
    return {
        "count": len(values),
        "mean": safe_mean(values),
        "median": safe_median(values),
        "positive_fraction": (
            sum(value > 0 for value in values) / len(values) if values else None
        ),
    }


def model_diagnostic(model: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    atlas_root = protocol.OUT_ROOT / "atlas" / model
    rows = protocol.read_jsonl(atlas_root / "response_metrics.jsonl")
    direction_rows = protocol.read_jsonl(atlas_root / "split_direction_repeat.jsonl")

    field_results = {}
    global_centroids = {}
    for field in FIELDS:
        banks = {
            (world, split): world_bank(rows, world, split, field)
            for world in protocol.WORLDS
            for split in protocol.SPLITS
        }
        same_world = []
        world_details = {}
        for world in protocol.WORLDS:
            discovery = banks[(world, "discovery")]
            confirmation = banks[(world, "confirmation")]
            repeat = cosine(centroid(discovery), centroid(confirmation))
            same_world.append(repeat)
            world_details[world] = {
                "shared_centroid_split_repeat": repeat,
                "discovery_pairwise_attribute_coherence": pairwise_coherence(discovery),
                "confirmation_pairwise_attribute_coherence": pairwise_coherence(confirmation),
                "discovery_energy": shared_differential_energy(discovery),
                "confirmation_energy": shared_differential_energy(confirmation),
            }

        cross_world = []
        for source_world in protocol.WORLDS:
            for target_world in protocol.WORLDS:
                if source_world == target_world:
                    continue
                cross_world.append({
                    "source_world": source_world,
                    "target_world": target_world,
                    "shared_centroid_cosine": cosine(
                        centroid(banks[(source_world, "discovery")]),
                        centroid(banks[(target_world, "confirmation")]),
                    ),
                })

        all_profiles = np.concatenate([
            banks[(world, split)]
            for world in protocol.WORLDS
            for split in protocol.SPLITS
        ])
        global_centroids[field] = unit(centroid(all_profiles))
        field_results[field] = {
            "same_world_shared_centroid_repeat_mean": safe_mean(same_world),
            "cross_world_shared_centroid_cosine_mean": safe_mean([
                row["shared_centroid_cosine"] for row in cross_world
            ]),
            "cross_world_shared_centroid_threshold_count": sum(
                row["shared_centroid_cosine"] is not None
                and row["shared_centroid_cosine"] >= 0.90
                for row in cross_world
            ),
            "cross_world_pair_count": len(cross_world),
            "worlds": world_details,
            "cross_world_pairs": cross_world,
            "split_direction_repeat": split_direction_summary(direction_rows, field),
        }

    return ({
        "fields": field_results,
        "content_answer_alignment": content_answer_alignment(rows),
        "content_minus_duplicate": {
            "split_direction_repeat_mean_gap": (
                field_results["content_route"]["split_direction_repeat"]["mean"]
                - field_results["duplicate_route"]["split_direction_repeat"]["mean"]
            ),
            "cross_world_shared_centroid_cosine_gap": (
                field_results["content_route"]["cross_world_shared_centroid_cosine_mean"]
                - field_results["duplicate_route"]["cross_world_shared_centroid_cosine_mean"]
            ),
        },
    }, global_centroids)


def main() -> None:
    by_model = {}
    centroids = {}
    for model in protocol.MODELS:
        by_model[model], centroids[model] = model_diagnostic(model)

    cross_model = []
    for source, target in itertools.permutations(protocol.MODELS, 2):
        cross_model.append({
            "source_model": source,
            "target_model": target,
            "content_shared_centroid_cosine": cosine(
                centroids[source]["content_route"],
                centroids[target]["content_route"],
            ),
            "duplicate_shared_centroid_cosine": cosine(
                centroids[source]["duplicate_route"],
                centroids[target]["duplicate_route"],
            ),
        })

    result = {
        "schema_version": "phase1085_posthoc_shared_reuse_diagnostic.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_descriptive_no_new_model_calls_no_gate_upgrade",
        "scope": {
            "role": ROLE,
            "relative_depth_min": protocol.TARGET_RELATIVE_DEPTH_MIN,
            "relative_depth_max": protocol.TARGET_RELATIVE_DEPTH_MAX,
            "components": ["residual", "attention_output", "mlp_output"],
        },
        "by_model": by_model,
        "cross_model_shared_centroids": cross_model,
        "interpretation": [
            "The registered centered retrieval tests attribute-specific residuals; this diagnostic restores the removed uncentered common profile.",
            "A repeated common magnitude profile is compatible with shared routing reuse, but also with a generic depth or task-shell response.",
            "Content specificity requires the common content profile to exceed the matched duplicate profile; otherwise the shared skeleton is not semantic evidence.",
            "Direction repeat is measured only within the same attribute-world family across item splits and does not establish cross-attribute direction reuse.",
        ],
        "limits": [
            "All calculations are post-hoc and cannot upgrade P4, P6, P7, or P8.",
            "Profiles contain nonnegative magnitudes and normalized depth channels, which can inflate common-shape cosine.",
            "No raw cross-attribute hidden vectors were stored, so shared signed directions cannot be tested from these aggregates.",
            "No new model inference, intervention, component search, or neuron localization was performed.",
        ],
    }
    result["diagnostic_digest"] = protocol.digest(result)
    output = protocol.OUT_ROOT / "analysis" / "posthoc_shared_reuse_diagnostic.json"
    protocol.write_json(output, result)
    print({
        "phase": protocol.PHASE,
        "by_model": {
            model: {
                "content_crossworld_common": data["fields"]["content_route"]["cross_world_shared_centroid_cosine_mean"],
                "duplicate_crossworld_common": data["fields"]["duplicate_route"]["cross_world_shared_centroid_cosine_mean"],
                "content_direction_repeat": data["fields"]["content_route"]["split_direction_repeat"]["mean"],
                "duplicate_direction_repeat": data["fields"]["duplicate_route"]["split_direction_repeat"]["mean"],
            }
            for model, data in by_model.items()
        },
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
