#!/usr/bin/env python3
"""Finalize Phase1092 natural bilingual attribute-pattern mapping.

The measurements remain descriptive.  They test repeated signed structure,
matched-null advantage, and relation geometry before any causal localization.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_protocol as protocol


EPSILON = 1e-12
PERMUTATIONS = np.asarray(
    list(itertools.permutations(range(8))), dtype=np.int16
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > EPSILON else np.zeros_like(values)


def row_normalize(values: np.ndarray, *, centered: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if centered:
        values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms > EPSILON,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit_vector(left), unit_vector(right)))


def operation_names(attribute: str) -> tuple[str, ...]:
    return tuple(
        value for value in protocol.OPERATIONS
        if value.startswith(f"{attribute}_")
    )


def surface_worlds(surface: str) -> tuple[str, ...]:
    return tuple(f"{world}@{surface}" for world in protocol.BASE_WORLDS)


def load_model(model_name: str) -> dict[str, Any]:
    atlas = protocol.OUT_ROOT / "atlas" / model_name
    summary = protocol.read_json(atlas / "summary.json")
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    if summary["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError(f"protocol digest mismatch for {model_name}")
    path = atlas / "signed_fields.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    counts = arrays["direction_count"]
    direction_mean = np.divide(
        arrays["direction_sum"],
        counts[..., None],
        out=np.zeros_like(arrays["direction_sum"], dtype=np.float32),
        where=counts[..., None] > 0,
    )
    relative_mean = np.divide(
        arrays["relative_sum"],
        arrays["relative_count"],
        out=np.zeros_like(arrays["relative_sum"], dtype=np.float64),
        where=arrays["relative_count"] > 0,
    )
    surface_mean = np.divide(
        arrays["surface_relative_sum"],
        arrays["surface_relative_count"],
        out=np.zeros_like(arrays["surface_relative_sum"], dtype=np.float64),
        where=arrays["surface_relative_count"] > 0,
    )
    expected = (
        len(protocol.FAMILIES), len(protocol.SPLITS), summary["event_count"],
        len(protocol.CAPTURE_ROLES), len(protocol.SIGNED_FIELDS),
        len(protocol.TEMPLATE_IDS), len(protocol.OUTPUT_SET_IDS),
        protocol.SIGNED_PROJECTION_REPLICATES,
        protocol.SIGNED_PROJECTION_DIM,
    )
    if direction_mean.shape != expected:
        raise RuntimeError(
            f"unexpected direction shape for {model_name}: "
            f"{direction_mean.shape} != {expected}"
        )
    return {
        "summary": summary,
        "arrays": arrays,
        "direction_mean": direction_mean,
        "relative_mean": relative_mean,
        "surface_mean": surface_mean,
        "npz_sha256": sha256_file(path),
    }


def profile(
    data: dict[str, Any],
    operation: str,
    worlds: tuple[str, ...],
    split: str,
    field: str,
    replicate: int,
    *,
    role: str = "answer_boundary",
    event_index: int | None = None,
    template: int | None = None,
) -> np.ndarray:
    values = []
    for world in worlds:
        family = f"{operation}__{world}"
        fi = protocol.FAMILIES.index(family)
        si = protocol.SPLITS.index(split)
        ri = protocol.CAPTURE_ROLES.index(role)
        di = protocol.SIGNED_FIELDS.index(field)
        current = data["direction_mean"][fi, si, :, ri, di, :, :, replicate, :]
        if event_index is not None:
            current = current[event_index:event_index + 1]
        if template is not None:
            ti = protocol.TEMPLATE_IDS.index(template)
            current = current[:, ti:ti + 1]
        current = current.mean(axis=(1, 2))
        values.append(current.reshape(-1))
    return np.mean(np.stack(values), axis=0)


def bank(
    data: dict[str, Any],
    attribute: str,
    surface: str,
    split: str,
    field: str,
    replicate: int,
    *,
    worlds: tuple[str, ...] | None = None,
    role: str = "answer_boundary",
    event_index: int | None = None,
    template: int | None = None,
    centered: bool = True,
) -> np.ndarray:
    world_names = worlds or protocol.BASE_WORLDS
    encoded_worlds = tuple(f"{world}@{surface}" for world in world_names)
    values = np.stack([
        profile(
            data, operation, encoded_worlds, split, field, replicate,
            role=role, event_index=event_index, template=template,
        )
        for operation in operation_names(attribute)
    ])
    return row_normalize(values, centered=centered)


def exact_assignment(
    source: np.ndarray,
    target: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    matrix = source @ target.T
    identity = float(np.trace(matrix) / len(labels))
    scores = matrix[np.arange(len(labels))[None, :], PERMUTATIONS].mean(axis=1)
    top1 = np.argmax(matrix, axis=1)
    rows = []
    for index, label in enumerate(labels):
        predicted = int(top1[index])
        rows.append({
            "operation": label,
            "predicted_operation": labels[predicted],
            "correct": predicted == index,
            "correct_similarity": float(matrix[index, index]),
            "best_other_similarity": float(np.max(np.delete(matrix[index], index))),
        })
    return {
        "top1_correct": int(np.sum(top1 == np.arange(len(labels)))),
        "identity_mean_score": identity,
        "permutation_count": int(scores.size),
        "scores_at_least_identity": int(np.sum(scores >= identity - 1e-12)),
        "exact_upper_tail_p": float(np.mean(scores >= identity - 1e-12)),
        "similarity_matrix": matrix.tolist(),
        "rows": rows,
    }


def quick_assignment(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    matrix = source @ target.T
    top1 = np.argmax(matrix, axis=1)
    return {
        "top1_correct": int(np.sum(top1 == np.arange(matrix.shape[0]))),
        "identity_mean_score": float(np.trace(matrix) / matrix.shape[0]),
    }


def relation_vector(values: np.ndarray) -> np.ndarray:
    gram = values @ values.T
    upper = gram[np.triu_indices(values.shape[0], k=1)]
    return unit_vector(upper - upper.mean())


def identity_pass(content: dict[str, Any], null: dict[str, Any]) -> tuple[bool, float]:
    threshold = protocol.EVIDENCE_THRESHOLDS
    advantage = float(content["identity_mean_score"] - null["identity_mean_score"])
    return (
        int(content["top1_correct"]) >= int(threshold["minimum_pair_top1"])
        and float(content["exact_upper_tail_p"]) <= float(threshold["permutation_p_max"])
        and int(content["top1_correct"]) > int(null["top1_correct"])
        and advantage >= float(threshold["minimum_content_identity_advantage"])
    ), advantage


def gram_record(
    content_source: np.ndarray,
    content_target: np.ndarray,
    null_source: np.ndarray,
    null_target: np.ndarray,
) -> dict[str, Any]:
    content = cosine(relation_vector(content_source), relation_vector(content_target))
    null = cosine(relation_vector(null_source), relation_vector(null_target))
    advantage = content - null
    threshold = protocol.EVIDENCE_THRESHOLDS
    return {
        "content_gram_cosine": content,
        "field_null_gram_cosine": null,
        "content_over_null_advantage": advantage,
        "passed": (
            content >= float(threshold["minimum_cross_language_gram_cosine"])
            and advantage >= float(threshold["minimum_cross_language_gram_advantage"])
        ),
    }


def within_language_analysis(
    models: dict[str, dict[str, Any]], behavior: dict[str, Any]
) -> dict[str, Any]:
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for surface in protocol.SURFACES:
                    content = exact_assignment(
                        bank(data, attribute, surface, "discovery", "content", replicate),
                        bank(data, attribute, surface, "confirmation", "content", replicate),
                        operation_names(attribute),
                    )
                    null = exact_assignment(
                        bank(data, attribute, surface, "discovery", "field_null", replicate),
                        bank(data, attribute, surface, "confirmation", "field_null", replicate),
                        operation_names(attribute),
                    )
                    passed, advantage = identity_pass(content, null)
                    rows.append({
                        "replicate": replicate,
                        "surface": surface,
                        "content": content,
                        "field_null": null,
                        "identity_advantage": advantage,
                        "passed": passed,
                    })
            behavior_passed = attribute in behavior["models"][model_name][
                "passing_attributes"
            ]
            attributes[attribute] = {
                "behavior_passed": behavior_passed,
                "rows": rows,
                "passed": behavior_passed and all(row["passed"] for row in rows),
            }
        by_model[model_name] = {"attributes": attributes}
    return {"by_model": by_model}


def cross_language_analysis(
    models: dict[str, dict[str, Any]], behavior: dict[str, Any]
) -> dict[str, Any]:
    directions = (("en", "zh"), ("zh", "en"))
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for source_surface, target_surface in directions:
                    content_source = bank(
                        data, attribute, source_surface, "discovery", "content", replicate
                    )
                    content_target = bank(
                        data, attribute, target_surface, "confirmation", "content", replicate
                    )
                    null_source = bank(
                        data, attribute, source_surface, "discovery", "field_null", replicate
                    )
                    null_target = bank(
                        data, attribute, target_surface, "confirmation", "field_null", replicate
                    )
                    content = exact_assignment(
                        content_source, content_target, operation_names(attribute)
                    )
                    null = exact_assignment(
                        null_source, null_target, operation_names(attribute)
                    )
                    passed, advantage = identity_pass(content, null)
                    rows.append({
                        "replicate": replicate,
                        "source_surface": source_surface,
                        "target_surface": target_surface,
                        "content_assignment": content,
                        "field_null_assignment": null,
                        "identity_advantage": advantage,
                        "identity_passed": passed,
                        "gram": gram_record(
                            content_source, content_target, null_source, null_target
                        ),
                    })
            behavior_passed = attribute in behavior["models"][model_name][
                "passing_attributes"
            ]
            attributes[attribute] = {
                "behavior_passed": behavior_passed,
                "rows": rows,
                "identity_passed": behavior_passed and all(
                    row["identity_passed"] for row in rows
                ),
                "gram_passed": behavior_passed and all(
                    row["gram"]["passed"] for row in rows
                ),
            }
        by_model[model_name] = {"attributes": attributes}
    return {"by_model": by_model}


def heldout_world_analysis(
    models: dict[str, dict[str, Any]], behavior: dict[str, Any]
) -> dict[str, Any]:
    directions = (("en", "zh"), ("zh", "en"))
    minimum_worlds = int(protocol.EVIDENCE_THRESHOLDS["minimum_heldout_worlds"])
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for source_surface, target_surface in directions:
                    for heldout in protocol.BASE_WORLDS:
                        train_worlds = tuple(
                            world for world in protocol.BASE_WORLDS if world != heldout
                        )
                        content_source = bank(
                            data, attribute, source_surface, "discovery", "content",
                            replicate, worlds=train_worlds,
                        )
                        content_target = bank(
                            data, attribute, target_surface, "confirmation", "content",
                            replicate, worlds=(heldout,),
                        )
                        null_source = bank(
                            data, attribute, source_surface, "discovery", "field_null",
                            replicate, worlds=train_worlds,
                        )
                        null_target = bank(
                            data, attribute, target_surface, "confirmation", "field_null",
                            replicate, worlds=(heldout,),
                        )
                        content = exact_assignment(
                            content_source, content_target, operation_names(attribute)
                        )
                        null = exact_assignment(
                            null_source, null_target, operation_names(attribute)
                        )
                        identity_ok, identity_advantage = identity_pass(content, null)
                        rows.append({
                            "replicate": replicate,
                            "source_surface": source_surface,
                            "target_surface": target_surface,
                            "heldout_world": heldout,
                            "content_assignment": content,
                            "field_null_assignment": null,
                            "identity_advantage": identity_advantage,
                            "identity_passed": identity_ok,
                            "gram": gram_record(
                                content_source, content_target, null_source, null_target
                            ),
                        })
            group_counts = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for source_surface, target_surface in directions:
                    group = [
                        row for row in rows
                        if row["replicate"] == replicate
                        and row["source_surface"] == source_surface
                        and row["target_surface"] == target_surface
                    ]
                    group_counts.append({
                        "replicate": replicate,
                        "source_surface": source_surface,
                        "target_surface": target_surface,
                        "identity_passing_worlds": sum(
                            int(row["identity_passed"]) for row in group
                        ),
                        "gram_passing_worlds": sum(
                            int(row["gram"]["passed"]) for row in group
                        ),
                    })
            behavior_passed = attribute in behavior["models"][model_name][
                "passing_attributes"
            ]
            attributes[attribute] = {
                "behavior_passed": behavior_passed,
                "groups": group_counts,
                "rows": rows,
                "passed": behavior_passed and all(
                    row["gram_passing_worlds"] >= minimum_worlds
                    for row in group_counts
                ),
            }
        by_model[model_name] = {"attributes": attributes}
    return {"by_model": by_model}


def cross_model_analysis(
    models: dict[str, dict[str, Any]], behavior: dict[str, Any], healthy: set[str]
) -> dict[str, Any]:
    rows = []
    directions = (("en", "zh"), ("zh", "en"))
    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            common_attributes = sorted(set(
                behavior["models"][source_model]["passing_attributes"]
            ).intersection(
                behavior["models"][target_model]["passing_attributes"]
            ))
            attribute_rows = {}
            for attribute in protocol.ATTRIBUTES:
                comparisons = []
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    for source_surface, target_surface in directions:
                        content_source = bank(
                            models[source_model], attribute, source_surface,
                            "discovery", "content", replicate,
                        )
                        content_target = bank(
                            models[target_model], attribute, target_surface,
                            "confirmation", "content", replicate,
                        )
                        null_source = bank(
                            models[source_model], attribute, source_surface,
                            "discovery", "field_null", replicate,
                        )
                        null_target = bank(
                            models[target_model], attribute, target_surface,
                            "confirmation", "field_null", replicate,
                        )
                        comparisons.append({
                            "replicate": replicate,
                            "source_surface": source_surface,
                            "target_surface": target_surface,
                            **gram_record(
                                content_source, content_target, null_source, null_target
                            ),
                        })
                attribute_rows[attribute] = {
                    "formal_for_pair": attribute in common_attributes,
                    "comparisons": comparisons,
                    "passed": (
                        attribute in common_attributes
                        and all(row["passed"] for row in comparisons)
                    ),
                }
            passing_attributes = [
                name for name, row in attribute_rows.items() if row["passed"]
            ]
            rows.append({
                "source_model": source_model,
                "target_model": target_model,
                "healthy_pair": source_model in healthy and target_model in healthy,
                "common_behavior_attributes": common_attributes,
                "passing_attributes": passing_attributes,
                "attributes": attribute_rows,
                "passed": (
                    source_model in healthy
                    and target_model in healthy
                    and len(passing_attributes) >= int(
                        protocol.EVIDENCE_THRESHOLDS[
                            "minimum_cross_language_attributes"
                        ]
                    )
                ),
            })
    return {"rows": rows}


def shared_field_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    directions = (("en", "zh"), ("zh", "en"))
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for source_surface, target_surface in directions:
                    content_source = bank(
                        data, attribute, source_surface, "discovery", "content",
                        replicate, centered=False,
                    )
                    content_target = bank(
                        data, attribute, target_surface, "confirmation", "content",
                        replicate, centered=False,
                    )
                    null_source = bank(
                        data, attribute, source_surface, "discovery", "field_null",
                        replicate, centered=False,
                    )
                    null_target = bank(
                        data, attribute, target_surface, "confirmation", "field_null",
                        replicate, centered=False,
                    )
                    content_centroid_source = content_source.mean(axis=0)
                    content_centroid_target = content_target.mean(axis=0)
                    null_centroid_source = null_source.mean(axis=0)
                    null_centroid_target = null_target.mean(axis=0)
                    content_cosine = cosine(
                        content_centroid_source, content_centroid_target
                    )
                    null_cosine = cosine(null_centroid_source, null_centroid_target)
                    rows.append({
                        "replicate": replicate,
                        "source_surface": source_surface,
                        "target_surface": target_surface,
                        "content_shared_fraction_source": float(np.dot(
                            content_centroid_source, content_centroid_source
                        )),
                        "content_shared_fraction_target": float(np.dot(
                            content_centroid_target, content_centroid_target
                        )),
                        "field_null_shared_fraction_source": float(np.dot(
                            null_centroid_source, null_centroid_source
                        )),
                        "field_null_shared_fraction_target": float(np.dot(
                            null_centroid_target, null_centroid_target
                        )),
                        "content_centroid_cosine": content_cosine,
                        "field_null_centroid_cosine": null_cosine,
                        "content_over_null_advantage": content_cosine - null_cosine,
                    })
            attributes[attribute] = {"rows": rows}

        cross_attribute_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            for surface in protocol.SURFACES:
                for split in protocol.SPLITS:
                    content_centroids = []
                    null_centroids = []
                    for attribute in protocol.ATTRIBUTES:
                        content_centroids.append(unit_vector(bank(
                            data, attribute, surface, split, "content", replicate,
                            centered=False,
                        ).mean(axis=0)))
                        null_centroids.append(unit_vector(bank(
                            data, attribute, surface, split, "field_null", replicate,
                            centered=False,
                        ).mean(axis=0)))
                    content_gram = np.stack(content_centroids) @ np.stack(content_centroids).T
                    null_gram = np.stack(null_centroids) @ np.stack(null_centroids).T
                    upper = np.triu_indices(len(protocol.ATTRIBUTES), k=1)
                    cross_attribute_rows.append({
                        "replicate": replicate,
                        "surface": surface,
                        "split": split,
                        "mean_content_attribute_centroid_cosine": float(
                            content_gram[upper].mean()
                        ),
                        "mean_field_null_attribute_centroid_cosine": float(
                            null_gram[upper].mean()
                        ),
                        "content_over_null_advantage": float(
                            content_gram[upper].mean() - null_gram[upper].mean()
                        ),
                    })
        by_model[model_name] = {
            "attributes": attributes,
            "cross_attribute_rows": cross_attribute_rows,
        }
    return {"by_model": by_model}


def control_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    field_index = protocol.SIGNED_FIELDS.index("content")
    for model_name, data in models.items():
        roles = {}
        for role in ("query_end", "answer_boundary"):
            role_index = protocol.CAPTURE_ROLES.index(role)
            content = data["relative_mean"][:, :, :, role_index, field_index, :, :]
            content = content.mean(axis=(-1, -2))
            surface = data["surface_mean"][:, :, :, role_index]
            valid = np.isfinite(content) & np.isfinite(surface) & (content > EPSILON)
            ratios = surface[valid] / content[valid]
            roles[role] = {
                "observation_count": int(np.sum(valid)),
                "median_content_relative_magnitude": float(np.median(content[valid])),
                "median_template_relative_magnitude": float(np.median(surface[valid])),
                "median_template_to_content_ratio": float(np.median(ratios)),
                "p90_template_to_content_ratio": float(np.percentile(ratios, 90)),
                "fraction_template_larger_than_content": float(np.mean(ratios > 1.0)),
            }
        by_model[model_name] = {"roles": roles}
    return {"by_model": by_model}


def decomposition_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model_name, data in models.items():
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for surface in protocol.SURFACES:
                    values = []
                    for operation in operation_names(attribute):
                        operation_values = []
                        for world in protocol.BASE_WORLDS:
                            operation_values.append(unit_vector(profile(
                                data, operation, (f"{world}@{surface}",),
                                "confirmation", "content", replicate,
                            )))
                        values.append(operation_values)
                    array = np.asarray(values)
                    global_mean = array.mean(axis=(0, 1), keepdims=True)
                    operation_effect = array.mean(axis=1, keepdims=True) - global_mean
                    world_effect = array.mean(axis=0, keepdims=True) - global_mean
                    interaction = array - global_mean - operation_effect - world_effect
                    energies = {
                        "operation": float(np.sum(operation_effect ** 2) * len(protocol.BASE_WORLDS)),
                        "world": float(np.sum(world_effect ** 2) * 8),
                        "interaction": float(np.sum(interaction ** 2)),
                    }
                    total = sum(energies.values())
                    rows.append({
                        "replicate": replicate,
                        "surface": surface,
                        "energy_fractions": {
                            key: value / total if total > EPSILON else 0.0
                            for key, value in energies.items()
                        },
                    })
            attributes[attribute] = {"rows": rows}
        by_model[model_name] = {"attributes": attributes}
    return {
        "by_model": by_model,
        "interpretation": "Descriptive variance allocation only; not a causal or independent-module decomposition.",
    }


def physical_map(
    models: dict[str, dict[str, Any]], behavior: dict[str, Any]
) -> dict[str, Any]:
    directions = (("en", "zh"), ("zh", "en"))
    by_model = {}
    threshold = protocol.EVIDENCE_THRESHOLDS
    for model_name, data in models.items():
        formal_attributes = tuple(behavior["models"][model_name]["passing_attributes"])
        rows = []
        for event_index, event in enumerate(data["summary"]["events"]):
            for role in ("query_end", "answer_boundary"):
                replicate_rows = []
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    cells = []
                    for attribute in protocol.ATTRIBUTES:
                        for source_surface, target_surface in directions:
                            content_source = bank(
                                data, attribute, source_surface, "discovery", "content",
                                replicate, role=role, event_index=event_index,
                            )
                            content_target = bank(
                                data, attribute, target_surface, "confirmation", "content",
                                replicate, role=role, event_index=event_index,
                            )
                            null_source = bank(
                                data, attribute, source_surface, "discovery", "field_null",
                                replicate, role=role, event_index=event_index,
                            )
                            null_target = bank(
                                data, attribute, target_surface, "confirmation", "field_null",
                                replicate, role=role, event_index=event_index,
                            )
                            content_assignment = quick_assignment(
                                content_source, content_target
                            )
                            null_assignment = quick_assignment(null_source, null_target)
                            identity_advantage = (
                                content_assignment["identity_mean_score"]
                                - null_assignment["identity_mean_score"]
                            )
                            gram = gram_record(
                                content_source, content_target, null_source, null_target
                            )
                            local_passed = (
                                attribute in formal_attributes
                                and content_assignment["top1_correct"]
                                >= int(threshold["minimum_pair_top1"])
                                and content_assignment["top1_correct"]
                                > null_assignment["top1_correct"]
                                and identity_advantage
                                >= float(threshold["minimum_content_identity_advantage"])
                                and gram["passed"]
                            )
                            cells.append({
                                "attribute": attribute,
                                "source_surface": source_surface,
                                "target_surface": target_surface,
                                "formal_attribute": attribute in formal_attributes,
                                "content_top1": content_assignment["top1_correct"],
                                "field_null_top1": null_assignment["top1_correct"],
                                "identity_advantage": identity_advantage,
                                "gram": gram,
                                "passed": local_passed,
                            })
                    formal_cells = [cell for cell in cells if cell["formal_attribute"]]
                    replicate_rows.append({
                        "replicate": replicate,
                        "passing_formal_cells": sum(
                            int(cell["passed"]) for cell in formal_cells
                        ),
                        "formal_cell_count": len(formal_cells),
                        "mean_identity_advantage": float(np.mean([
                            cell["identity_advantage"] for cell in formal_cells
                        ])) if formal_cells else 0.0,
                        "mean_gram_advantage": float(np.mean([
                            cell["gram"]["content_over_null_advantage"]
                            for cell in formal_cells
                        ])) if formal_cells else 0.0,
                        "cells": cells,
                    })
                complete = bool(formal_attributes) and all(
                    row["passing_formal_cells"] == row["formal_cell_count"]
                    for row in replicate_rows
                )
                rows.append({
                    "event_index": event_index,
                    "event_id": event["event_id"],
                    "component": event["component"],
                    "depth": event["depth"],
                    "relative_depth": event["relative_depth"],
                    "role": role,
                    "replicates": replicate_rows,
                    "complete_formal_map": complete,
                })
        ranked = sorted(
            rows,
            key=lambda row: (
                min(value["passing_formal_cells"] for value in row["replicates"]),
                min(value["mean_gram_advantage"] for value in row["replicates"]),
                min(value["mean_identity_advantage"] for value in row["replicates"]),
            ),
            reverse=True,
        )
        by_model[model_name] = {
            "formal_attributes": list(formal_attributes),
            "rows": rows,
            "top_rows": ranked[:12],
            "passed": any(row["complete_formal_map"] for row in rows),
        }
    return {
        "by_model": by_model,
        "scope": "preregistered_relative_depth_0.20_to_0.60_descriptive_map",
        "causal_selection_authorized": False,
    }


def projection_gate(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    threshold = protocol.EVIDENCE_THRESHOLDS
    by_model = {}
    for model_name, data in models.items():
        rows = []
        for raw in data["summary"]["projection_audit"]["replicates"]:
            passed = (
                float(raw["median_abs_norm_error"])
                <= float(threshold["maximum_projection_median_abs_norm_error"])
                and float(raw["p95_abs_norm_error"])
                <= float(threshold["maximum_projection_p95_abs_norm_error"])
            )
            rows.append({**raw, "passed": passed})
        by_model[model_name] = {
            "replicates": rows,
            "passed": all(row["passed"] for row in rows),
        }
    return {"by_model": by_model}


def numeric_gate(
    models: dict[str, dict[str, Any]], behavior: dict[str, Any]
) -> dict[str, Any]:
    threshold = protocol.EVIDENCE_THRESHOLDS
    by_model = {}
    for model_name, data in models.items():
        summary = data["summary"]
        precision = summary["precision"]
        checks = {
            "fp16_parameters": bool(precision["has_fp16_parameters"]),
            "no_bf16_parameters": not bool(precision["has_bf16_parameters"]),
            "no_quantized_modules": not bool(precision["has_quantized_modules"]),
            "behavior_candidate_finite": (
                behavior["models"][model_name]["candidate_finite_fraction"]
                >= float(threshold["minimum_candidate_finite_fraction"])
            ),
            "scan_candidate_finite": (
                summary["candidate_finite_fraction"]
                >= float(threshold["minimum_candidate_finite_fraction"])
            ),
            "hidden_finite": (
                summary["hidden_finite_fraction_lower_bound"]
                >= float(threshold["minimum_hidden_finite_fraction"])
            ),
            "pre_query": (
                summary["pre_query_global_max_abs"]
                <= float(threshold["pre_query_tolerance"])
            ),
            "identity": (
                summary["identity_maximum"]
                <= float(threshold["pre_query_tolerance"])
            ),
        }
        by_model[model_name] = {
            "checks": checks,
            "behavior_authorized": behavior["models"][model_name]["model_authorized"],
            "behavior_candidate_finite_fraction": behavior["models"][model_name][
                "candidate_finite_fraction"
            ],
            "scan_candidate_finite_fraction": summary["candidate_finite_fraction"],
            "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
            "pre_query_global_max_abs": summary["pre_query_global_max_abs"],
            "passed": all(checks.values()),
        }
    return {
        "by_model": by_model,
        "numeric_healthy_models": [
            name for name, row in by_model.items() if row["passed"]
        ],
    }


def write_output(
    root: Path,
    filename: str,
    schema: str,
    payload: dict[str, Any],
    digest_key: str,
    protocol_digest: str,
) -> None:
    row = {
        "schema_version": schema,
        "phase": protocol.PHASE,
        "protocol_digest": protocol_digest,
        **payload,
    }
    row[digest_key] = protocol.digest(row)
    protocol.write_json(root / filename, row)


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {name: load_model(name) for name in protocol.MODELS}
    root = protocol.OUT_ROOT / "analysis"
    root.mkdir(parents=True, exist_ok=True)

    projection = projection_gate(models)
    numeric = numeric_gate(models, behavior)
    healthy = set(numeric["numeric_healthy_models"])
    formal_models = healthy.intersection(behavior["authorized_models"])
    within = within_language_analysis(models, behavior)
    cross_language = cross_language_analysis(models, behavior)
    heldout = heldout_world_analysis(models, behavior)
    cross_model = cross_model_analysis(models, behavior, formal_models)
    shared = shared_field_analysis(models)
    controls = control_analysis(models)
    decomposition = decomposition_analysis(models)
    physical = physical_map(models, behavior)

    minimum_models = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_language_models"]
    )
    minimum_attributes = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_language_attributes"]
    )

    p3_models = [
        name for name in formal_models if projection["by_model"][name]["passed"]
    ]
    p4_models = []
    p5_models = []
    p6_models = []
    p7_models = []
    p9_models = []
    model_attribute_counts = {}
    for model_name in formal_models:
        counts = {
            "within": sum(
                int(row["passed"])
                for row in within["by_model"][model_name]["attributes"].values()
            ),
            "identity": sum(
                int(row["identity_passed"])
                for row in cross_language["by_model"][model_name]["attributes"].values()
            ),
            "gram": sum(
                int(row["gram_passed"])
                for row in cross_language["by_model"][model_name]["attributes"].values()
            ),
            "heldout": sum(
                int(row["passed"])
                for row in heldout["by_model"][model_name]["attributes"].values()
            ),
        }
        model_attribute_counts[model_name] = counts
        if counts["within"] >= minimum_attributes:
            p4_models.append(model_name)
        if counts["identity"] >= minimum_attributes:
            p5_models.append(model_name)
        if counts["gram"] >= minimum_attributes:
            p6_models.append(model_name)
        if counts["heldout"] >= minimum_attributes:
            p7_models.append(model_name)
        if physical["by_model"][model_name]["passed"]:
            p9_models.append(model_name)

    p8_rows = [row for row in cross_model["rows"] if row["passed"]]
    predictions = {
        "P1": {
            "passed": bool(protocol.read_json(
                protocol.OUT_ROOT / "protocol" / "audit.json"
            )["all_checks_passed"]),
        },
        "P2": {
            "passed": bool(behavior["hidden_scan_authorized"]),
            "authorized_models": behavior["authorized_models"],
        },
        "P3": {
            "passed": len(p3_models) >= minimum_models,
            "passing_models": sorted(p3_models),
        },
        "P4": {
            "passed": len(p4_models) >= minimum_models,
            "passing_models": sorted(p4_models),
            "attribute_counts": model_attribute_counts,
        },
        "P5": {
            "passed": len(p5_models) >= minimum_models,
            "passing_models": sorted(p5_models),
        },
        "P6": {
            "passed": len(p6_models) >= minimum_models,
            "passing_models": sorted(p6_models),
        },
        "P7": {
            "passed": len(p7_models) >= minimum_models,
            "passing_models": sorted(p7_models),
        },
        "P8": {
            "passed": len(p8_rows) >= 2,
            "passing_directed_model_pairs": len(p8_rows),
            "rows": p8_rows,
        },
        "P9": {
            "passed": len(p9_models) >= minimum_models,
            "passing_models": sorted(p9_models),
        },
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]

    cross_language_semantic_candidate = all(
        predictions[name]["passed"] for name in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
    )
    if cross_language_semantic_candidate:
        decision = (
            "retain_cross_language_attribute_relation_candidate; next require "
            "unseen_value_pairs_and_natural_corpus_replication"
        )
        automatic_replication_authorized = True
    elif predictions["P6"]["passed"]:
        decision = (
            "retain_cross_language_relation_geometry_without_identity; next "
            "study language-conditioned coordinate transforms"
        )
        automatic_replication_authorized = False
    else:
        decision = (
            "retain_language-conditioned_attribute_atlas; cross-language semantic "
            "invariant_not_confirmed"
        )
        automatic_replication_authorized = False

    outputs = (
        ("within_language_identity.json", "phase1092_within_language_identity.v1", within, "within_digest"),
        ("cross_language_identity_geometry.json", "phase1092_cross_language_identity_geometry.v1", cross_language, "cross_language_digest"),
        ("heldout_world_geometry.json", "phase1092_heldout_world_geometry.v1", heldout, "heldout_digest"),
        ("cross_model_geometry.json", "phase1092_cross_model_geometry.v1", cross_model, "cross_model_digest"),
        ("shared_field.json", "phase1092_shared_field.v1", shared, "shared_field_digest"),
        ("control_audit.json", "phase1092_control_audit.v1", controls, "control_digest"),
        ("decomposition.json", "phase1092_decomposition.v1", decomposition, "decomposition_digest"),
        ("physical_map.json", "phase1092_physical_map.v1", physical, "physical_map_digest"),
        ("projection_audit.json", "phase1092_projection_audit.v1", projection, "projection_digest"),
        ("numeric_audit.json", "phase1092_numeric_audit.v1", numeric, "numeric_digest"),
    )
    for filename, schema, payload, digest_key in outputs:
        write_output(
            root, filename, schema, payload, digest_key, prereg["protocol_digest"]
        )

    automatic = {
        "schema_version": "phase1092_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "decision": decision,
        "automatic_replication_authorized": automatic_replication_authorized,
        "automatic_hidden_extension_authorized": False,
        "local_causal_authorized": False,
        "reason": (
            "Phase1092 is a descriptive signed-field map. Causal localization is "
            "forbidden from this phase alone. A new model run is automatic only "
            "after the full cross-language identity, Gram, and held-out-world gates pass."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)
    protocol.write_json(root / "automatic_next.json", automatic)

    summary = {
        "schema_version": "phase1092_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_authorization_digest": behavior["summary_digest"],
        "numeric_healthy_models": sorted(healthy),
        "formal_models": sorted(formal_models),
        "model_attribute_counts": model_attribute_counts,
        "predictions": predictions,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "cross_language_semantic_candidate": cross_language_semantic_candidate,
        "decision": decision,
        "models": {
            name: {
                "summary_digest": data["summary"]["summary_digest"],
                "npz_sha256": data["npz_sha256"],
                "behavior_status": data["summary"]["behavior_status"],
                "candidate_accuracy": data["summary"]["candidate_accuracy"],
                "candidate_finite_fraction": data["summary"]["candidate_finite_fraction"],
                "hidden_finite_fraction": data["summary"]["hidden_finite_fraction_lower_bound"],
                "event_count": data["summary"]["event_count"],
                "confirmatory": name in formal_models,
            }
            for name, data in models.items()
        },
        "interpretation_limits": [
            "Phase1091 did not use fully Chinese natural prompts; Phase1092 is the first full English/Chinese natural-shell test in this chain.",
            "A passing within-language pair fingerprint is not an abstract semantic invariant.",
            "A passing Gram relation is coordinate-rotation tolerant but still requires matched-null advantage.",
            "The controlled natural narratives are not naturally sampled corpus text.",
            "Descriptive physical bands are repeated response locations, not causal transport paths.",
            "No percentage completion of the full language mechanism is scientifically defined.",
        ],
        "automatic_next_digest": automatic["automatic_next_digest"],
    }
    summary["summary_digest"] = protocol.digest(summary)
    protocol.write_json(root / "final_summary.json", summary)
    print({
        "phase": protocol.PHASE,
        "formal_models": sorted(formal_models),
        "passed": passed,
        "failed": failed,
        "decision": decision,
        "summary_digest": summary["summary_digest"],
    })


if __name__ == "__main__":
    main()
