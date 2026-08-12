#!/usr/bin/env python3
"""Finalize the preregistered Phase1086 signed-field analysis.

The analysis keeps three questions separate:
1. Is there a signed response shared by the eight attribute questions?
2. Does that response exceed the matched field-null task route?
3. Do centered attribute residuals transfer to held-out lexical worlds?

Projected coordinates are never compared across models.  Cross-model tests
compare only within-model operation-relation Gram geometry.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1086_signed_shared_field_protocol as protocol


EPSILON = 1e-12
MODEL_NAMES = tuple(protocol.MODELS)
OPERATIONS = tuple(protocol.OPERATIONS)
WORLDS = tuple(protocol.WORLDS)
SPLITS = tuple(protocol.SPLITS)
FIELDS = tuple(protocol.SIGNED_FIELDS)
ROLES = tuple(protocol.CAPTURE_ROLES)
PERMUTATIONS = np.asarray(
    list(itertools.permutations(range(len(OPERATIONS)))), dtype=np.int16
)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(
        values,
        counts,
        out=np.zeros_like(values, dtype=np.float64),
        where=counts > 0,
    )


def unit_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > EPSILON else np.zeros_like(vector)


def row_normalize(values: np.ndarray, *, centered: bool) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if centered:
        output -= output.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(
        output,
        norms,
        out=np.zeros_like(output),
        where=norms > EPSILON,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = unit_vector(left)
    right_unit = unit_vector(right)
    if not np.any(left_unit) or not np.any(right_unit):
        return 0.0
    return float(np.dot(left_unit, right_unit))


def exact_assignment(matrix: np.ndarray) -> dict[str, Any]:
    size = matrix.shape[0]
    row_axis = np.arange(size)[:, None]
    scores = matrix[row_axis, PERMUTATIONS.T].mean(axis=0)
    identity = float(np.trace(matrix) / size)
    top1 = int(np.sum(np.argmax(matrix, axis=1) == np.arange(size)))
    scores_at_least = int(np.sum(scores >= identity - 1e-12))
    nonidentity = scores[1:]
    best_other = float(np.max(nonidentity))
    return {
        "top1_correct": top1,
        "identity_mean_score": identity,
        "best_nonidentity_mean_score": best_other,
        "identity_margin_over_best_other": identity - best_other,
        "permutation_count": int(scores.size),
        "scores_at_least_identity": scores_at_least,
        "exact_upper_tail_p": float(scores_at_least / scores.size),
        "similarity_matrix": matrix.tolist(),
    }


def load_model(model_name: str) -> dict[str, Any]:
    atlas_root = protocol.OUT_ROOT / "atlas" / model_name
    summary = protocol.read_json(atlas_root / "summary.json")
    preregistration = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    if summary["protocol_digest"] != preregistration["protocol_digest"]:
        raise RuntimeError(f"protocol digest mismatch for {model_name}")
    path = atlas_root / "signed_fields.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    direction_count = arrays["direction_count"]
    direction_mean = np.divide(
        arrays["direction_sum"],
        direction_count[..., None],
        out=np.zeros_like(arrays["direction_sum"], dtype=np.float32),
        where=direction_count[..., None] > 0,
    )
    relative_mean = safe_mean(
        arrays["relative_sum"], arrays["relative_count"]
    )
    surface_mean = safe_mean(
        arrays["surface_relative_sum"], arrays["surface_relative_count"]
    )
    output_mean = safe_mean(
        arrays["output_relative_sum"], arrays["output_relative_count"]
    )
    template_ids = tuple(getattr(protocol, "TEMPLATE_IDS", (0, 1)))
    output_set_ids = tuple(getattr(protocol, "OUTPUT_SET_IDS", (0, 1)))
    expected_prefix = (
        len(protocol.FAMILIES), len(SPLITS), int(summary["event_count"]),
        len(ROLES), len(FIELDS), len(template_ids), len(output_set_ids),
        protocol.SIGNED_PROJECTION_REPLICATES,
        protocol.SIGNED_PROJECTION_DIM,
    )
    if direction_mean.shape != expected_prefix:
        raise RuntimeError(
            f"unexpected direction shape for {model_name}: "
            f"{direction_mean.shape} != {expected_prefix}"
        )
    return {
        "summary": summary,
        "arrays": arrays,
        "direction_mean": direction_mean,
        "relative_mean": relative_mean,
        "surface_mean": surface_mean,
        "output_mean": output_mean,
        "npz_sha256": file_sha256(path),
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
    template: int | None = None,
    output_set: int | None = None,
    event_index: int | None = None,
) -> np.ndarray:
    values = []
    for world in worlds:
        family = f"{operation}__{world}"
        family_index = protocol.FAMILIES.index(family)
        split_index = SPLITS.index(split)
        role_index = ROLES.index(role)
        field_index = FIELDS.index(field)
        current = data["direction_mean"][
            family_index, split_index, :, role_index, field_index,
            :, :, replicate, :
        ]
        if event_index is not None:
            current = current[event_index:event_index + 1]
        if template is not None:
            current = current[:, template:template + 1]
        if output_set is not None:
            current = current[:, :, output_set:output_set + 1]
        current = current.mean(axis=(1, 2))
        values.append(current.reshape(-1))
    return np.mean(np.stack(values), axis=0)


def operation_bank(
    data: dict[str, Any],
    worlds: tuple[str, ...],
    split: str,
    field: str,
    replicate: int,
    *,
    centered: bool,
    role: str = "answer_boundary",
    template: int | None = None,
    output_set: int | None = None,
    event_index: int | None = None,
) -> np.ndarray:
    values = np.stack([
        profile(
            data, operation, worlds, split, field, replicate,
            role=role, template=template, output_set=output_set,
            event_index=event_index,
        )
        for operation in OPERATIONS
    ])
    return row_normalize(values, centered=centered)


def shared_centroid(
    data: dict[str, Any],
    worlds: tuple[str, ...],
    split: str,
    field: str,
    replicate: int,
    **filters: Any,
) -> tuple[np.ndarray, float]:
    bank = operation_bank(
        data, worlds, split, field, replicate, centered=False, **filters
    )
    centroid = bank.mean(axis=0)
    shared_fraction = float(np.dot(centroid, centroid))
    return centroid, shared_fraction


def assignment_record(
    source: np.ndarray,
    target: np.ndarray,
    **metadata: Any,
) -> dict[str, Any]:
    matrix = source @ target.T
    result = {**metadata, **exact_assignment(matrix)}
    details = []
    for index, operation in enumerate(OPERATIONS):
        predicted = int(np.argmax(matrix[index]))
        details.append({
            "operation": operation,
            "predicted_operation": OPERATIONS[predicted],
            "correct": predicted == index,
            "correct_similarity": float(matrix[index, index]),
            "best_other_similarity": float(np.max(np.delete(matrix[index], index))),
        })
    result["rows"] = details
    return result


def shared_field_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    threshold_cosine = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_split_cosine"]
    )
    threshold_advantage = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_shared_content_over_null_advantage"
        ]
    )
    minimum_worlds = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_worlds"]
    )
    minimum_pairs = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_world_pairs"]
    )
    by_model: dict[str, Any] = {}
    for model_name, data in models.items():
        split_rows = []
        cross_rows = []
        fractions = {"content": [], "field_null": []}
        replicate_gates = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            passing_worlds = 0
            passing_pairs = 0
            for world in WORLDS:
                centroids: dict[tuple[str, str], np.ndarray] = {}
                for field in ("content", "field_null"):
                    for split in SPLITS:
                        centroid, fraction = shared_centroid(
                            data, (world,), split, field, replicate
                        )
                        centroids[(field, split)] = centroid
                        fractions[field].append(fraction)
                content_cosine = cosine(
                    centroids[("content", "discovery")],
                    centroids[("content", "confirmation")],
                )
                null_cosine = cosine(
                    centroids[("field_null", "discovery")],
                    centroids[("field_null", "confirmation")],
                )
                advantage = content_cosine - null_cosine
                passed = (
                    content_cosine >= threshold_cosine
                    and advantage >= threshold_advantage
                )
                passing_worlds += int(passed)
                split_rows.append({
                    "model": model_name,
                    "replicate": replicate,
                    "world": world,
                    "content_cosine": content_cosine,
                    "field_null_cosine": null_cosine,
                    "content_over_null_advantage": advantage,
                    "passed": passed,
                })
            for source_world in WORLDS:
                for target_world in WORLDS:
                    if source_world == target_world:
                        continue
                    content_source, _ = shared_centroid(
                        data, (source_world,), "discovery", "content", replicate
                    )
                    content_target, _ = shared_centroid(
                        data, (target_world,), "confirmation", "content", replicate
                    )
                    null_source, _ = shared_centroid(
                        data, (source_world,), "discovery", "field_null", replicate
                    )
                    null_target, _ = shared_centroid(
                        data, (target_world,), "confirmation", "field_null", replicate
                    )
                    content_cosine = cosine(content_source, content_target)
                    null_cosine = cosine(null_source, null_target)
                    advantage = content_cosine - null_cosine
                    passed = (
                        content_cosine >= threshold_cosine
                        and advantage >= threshold_advantage
                    )
                    passing_pairs += int(passed)
                    cross_rows.append({
                        "model": model_name,
                        "replicate": replicate,
                        "source_world": source_world,
                        "target_world": target_world,
                        "content_cosine": content_cosine,
                        "field_null_cosine": null_cosine,
                        "content_over_null_advantage": advantage,
                        "passed": passed,
                    })
            replicate_gates.append({
                "replicate": replicate,
                "passing_split_worlds": passing_worlds,
                "split_gate_passed": passing_worlds >= minimum_worlds,
                "passing_directed_world_pairs": passing_pairs,
                "cross_world_gate_passed": passing_pairs >= minimum_pairs,
            })
        by_model[model_name] = {
            "replicate_gates": replicate_gates,
            "split_gate_passed": all(
                row["split_gate_passed"] for row in replicate_gates
            ),
            "cross_world_gate_passed": all(
                row["cross_world_gate_passed"] for row in replicate_gates
            ),
            "median_signed_shared_fraction": {
                field: float(np.median(values))
                for field, values in fractions.items()
            },
            "split_rows": split_rows,
            "cross_world_rows": cross_rows,
        }
    return {"by_model": by_model}


def control_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    transfer_surface = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_surface_transfer_cosine"]
    )
    transfer_output = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_output_transfer_cosine"]
    )
    maximum_ratio = float(
        protocol.EVIDENCE_THRESHOLDS["maximum_surface_to_content_ratio"]
    )
    minimum_worlds = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_worlds"]
    )
    by_model: dict[str, Any] = {}
    for model_name, data in models.items():
        role_index = ROLES.index("answer_boundary")
        content_index = FIELDS.index("content")
        content = data["relative_mean"][:, :, :, role_index, content_index]
        content = content.mean(axis=(-1, -2))
        surface = data["surface_mean"][:, :, :, role_index]
        output = data["output_mean"][:, :, :, role_index]
        valid = content > EPSILON
        surface_ratios = np.divide(
            surface, content, out=np.full_like(surface, np.nan), where=valid
        )
        output_ratios = np.divide(
            output, content, out=np.full_like(output, np.nan), where=valid
        )
        median_surface_ratio = float(np.nanmedian(surface_ratios))
        median_output_ratio = float(np.nanmedian(output_ratios))
        ratio_passed = max(median_surface_ratio, median_output_ratio) <= maximum_ratio

        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            surface_worlds = 0
            output_worlds = 0
            world_rows = []
            for world in WORLDS:
                surface_cosines = []
                output_cosines = []
                for source_level, target_level in ((0, 1), (1, 0)):
                    source, _ = shared_centroid(
                        data, (world,), "discovery", "content", replicate,
                        template=source_level,
                    )
                    target, _ = shared_centroid(
                        data, (world,), "confirmation", "content", replicate,
                        template=target_level,
                    )
                    surface_cosines.append(cosine(source, target))
                    source, _ = shared_centroid(
                        data, (world,), "discovery", "content", replicate,
                        output_set=source_level,
                    )
                    target, _ = shared_centroid(
                        data, (world,), "confirmation", "content", replicate,
                        output_set=target_level,
                    )
                    output_cosines.append(cosine(source, target))
                surface_passed = min(surface_cosines) >= transfer_surface
                output_passed = min(output_cosines) >= transfer_output
                surface_worlds += int(surface_passed)
                output_worlds += int(output_passed)
                world_rows.append({
                    "world": world,
                    "surface_cosines": surface_cosines,
                    "output_cosines": output_cosines,
                    "surface_passed": surface_passed,
                    "output_passed": output_passed,
                })
            replicate_rows.append({
                "replicate": replicate,
                "surface_passing_worlds": surface_worlds,
                "output_passing_worlds": output_worlds,
                "surface_transfer_passed": surface_worlds >= minimum_worlds,
                "output_transfer_passed": output_worlds >= minimum_worlds,
                "worlds": world_rows,
            })
        transfer_passed = all(
            row["surface_transfer_passed"] and row["output_transfer_passed"]
            for row in replicate_rows
        )
        by_model[model_name] = {
            "replicates": replicate_rows,
            "surface_transfer_passed": transfer_passed,
            "median_surface_to_content_ratio": median_surface_ratio,
            "median_output_to_content_ratio": median_output_ratio,
            "control_ratio_passed": ratio_passed,
            "combined_gate_passed": transfer_passed and ratio_passed,
        }
    return {"by_model": by_model}


def attribute_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    minimum_top1 = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_attribute_top1"]
    )
    maximum_p = float(protocol.EVIDENCE_THRESHOLDS["permutation_p_max"])
    minimum_worlds = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_attribute_heldout_worlds"]
    )
    minimum_advantage = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_shared_content_over_null_advantage"
        ]
    )
    by_model: dict[str, Any] = {}
    all_assignments = []
    for model_name, data in models.items():
        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            discovery = operation_bank(
                data, WORLDS, "discovery", "content", replicate, centered=True
            )
            confirmation = operation_bank(
                data, WORLDS, "confirmation", "content", replicate, centered=True
            )
            split_assignment = assignment_record(
                discovery, confirmation,
                comparison="independent_item_split",
                model=model_name,
                replicate=replicate,
                field="content",
            )
            split_assignment["passed"] = (
                split_assignment["top1_correct"] >= minimum_top1
                and split_assignment["exact_upper_tail_p"] <= maximum_p
            )
            all_assignments.append(split_assignment)
            heldout_rows = []
            for heldout in WORLDS:
                source_worlds = tuple(world for world in WORLDS if world != heldout)
                content_source = operation_bank(
                    data, source_worlds, "discovery", "content", replicate,
                    centered=True,
                )
                content_target = operation_bank(
                    data, (heldout,), "confirmation", "content", replicate,
                    centered=True,
                )
                null_source = operation_bank(
                    data, source_worlds, "discovery", "field_null", replicate,
                    centered=True,
                )
                null_target = operation_bank(
                    data, (heldout,), "confirmation", "field_null", replicate,
                    centered=True,
                )
                content_result = assignment_record(
                    content_source, content_target,
                    comparison="heldout_world",
                    model=model_name,
                    replicate=replicate,
                    field="content",
                    heldout_world=heldout,
                )
                null_result = assignment_record(
                    null_source, null_target,
                    comparison="heldout_world",
                    model=model_name,
                    replicate=replicate,
                    field="field_null",
                    heldout_world=heldout,
                )
                identity_advantage = (
                    content_result["identity_mean_score"]
                    - null_result["identity_mean_score"]
                )
                passed = (
                    content_result["top1_correct"] >= minimum_top1
                    and content_result["exact_upper_tail_p"] <= maximum_p
                    and content_result["top1_correct"] > null_result["top1_correct"]
                    and identity_advantage >= minimum_advantage
                )
                heldout_rows.append({
                    "heldout_world": heldout,
                    "content": content_result,
                    "field_null": null_result,
                    "content_identity_advantage": identity_advantage,
                    "passed": passed,
                })
                all_assignments.extend((content_result, null_result))
            replicate_rows.append({
                "replicate": replicate,
                "split_assignment": split_assignment,
                "passing_heldout_worlds": sum(
                    int(row["passed"]) for row in heldout_rows
                ),
                "heldout_gate_passed": sum(
                    int(row["passed"]) for row in heldout_rows
                ) >= minimum_worlds,
                "heldout_rows": heldout_rows,
            })
        by_model[model_name] = {
            "replicates": replicate_rows,
            "split_gate_passed": all(
                row["split_assignment"]["passed"] for row in replicate_rows
            ),
            "heldout_gate_passed": all(
                row["heldout_gate_passed"] for row in replicate_rows
            ),
        }
    return {"by_model": by_model, "assignments": all_assignments}


def relation_vector(bank: np.ndarray) -> np.ndarray:
    gram = bank @ bank.T
    values = gram[np.triu_indices(len(OPERATIONS), k=1)]
    values = values - values.mean()
    return unit_vector(values)


def cross_model_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    threshold = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_geometry_cosine"]
    )
    minimum_pairs = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_model_geometry_pairs"]
    )
    rows = []
    for source_name in MODEL_NAMES:
        for target_name in MODEL_NAMES:
            if source_name == target_name:
                continue
            replicate_cosines = []
            null_replicate_cosines = []
            geometry_advantages = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                source_bank = operation_bank(
                    models[source_name], WORLDS, "discovery", "content",
                    replicate, centered=True,
                )
                target_bank = operation_bank(
                    models[target_name], WORLDS, "confirmation", "content",
                    replicate, centered=True,
                )
                content_cosine = cosine(
                    relation_vector(source_bank), relation_vector(target_bank)
                )
                null_source_bank = operation_bank(
                    models[source_name], WORLDS, "discovery", "field_null",
                    replicate, centered=True,
                )
                null_target_bank = operation_bank(
                    models[target_name], WORLDS, "confirmation", "field_null",
                    replicate, centered=True,
                )
                null_cosine = cosine(
                    relation_vector(null_source_bank),
                    relation_vector(null_target_bank),
                )
                replicate_cosines.append(content_cosine)
                null_replicate_cosines.append(null_cosine)
                geometry_advantages.append(content_cosine - null_cosine)
            passed = min(replicate_cosines) >= threshold
            rows.append({
                "source_model": source_name,
                "target_model": target_name,
                "replicate_geometry_cosines": replicate_cosines,
                "field_null_replicate_geometry_cosines": null_replicate_cosines,
                "content_over_null_geometry_advantages": geometry_advantages,
                "minimum_cosine": min(replicate_cosines),
                "passed": passed,
                "posthoc_content_specific_advantage_passed": (
                    min(geometry_advantages) >= 0.10
                ),
            })
    content_specific_pairs = sum(
        int(row["posthoc_content_specific_advantage_passed"])
        for row in rows
    )
    return {
        "rows": rows,
        "passing_directed_pairs": sum(int(row["passed"]) for row in rows),
        "gate_passed": sum(int(row["passed"]) for row in rows) >= minimum_pairs,
        "posthoc_content_specific_passing_pairs": content_specific_pairs,
        "posthoc_content_specific_gate_passed": content_specific_pairs >= minimum_pairs,
        "coordinate_warning": (
            "Only within-model centered operation Gram geometry is compared; "
            "random projected coordinates are never compared across models."
        ),
        "posthoc_warning": (
            "The content-over-null geometry advantage is a diagnostic and does "
            "not alter the preregistered P9 result."
        ),
    }


def decomposition_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name, data in models.items():
        rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            for split in SPLITS:
                for field in ("content", "field_null"):
                    matrix = np.empty(
                        (len(OPERATIONS), len(WORLDS)), dtype=object
                    )
                    for oi, operation in enumerate(OPERATIONS):
                        for wi, world in enumerate(WORLDS):
                            matrix[oi, wi] = unit_vector(profile(
                                data, operation, (world,), split, field, replicate
                            ))
                    stacked = np.stack([
                        np.stack([matrix[oi, wi] for wi in range(len(WORLDS))])
                        for oi in range(len(OPERATIONS))
                    ])
                    grand = stacked.mean(axis=(0, 1), keepdims=True)
                    op_main = stacked.mean(axis=1, keepdims=True) - grand
                    world_main = stacked.mean(axis=0, keepdims=True) - grand
                    interaction = stacked - grand - op_main - world_main
                    energies = {
                        "operation": float(np.sum(op_main ** 2) * len(WORLDS)),
                        "world": float(np.sum(world_main ** 2) * len(OPERATIONS)),
                        "interaction": float(np.sum(interaction ** 2)),
                    }
                    total = sum(energies.values())
                    rows.append({
                        "model": model_name,
                        "replicate": replicate,
                        "split": split,
                        "field": field,
                        "energy": energies,
                        "fraction": {
                            key: value / total if total > EPSILON else 0.0
                            for key, value in energies.items()
                        },
                    })
        by_model[model_name] = rows
    return {"by_model": by_model, "interpretation": "descriptive_not_causal"}


def physical_map(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name, data in models.items():
        summary = data["summary"]
        role_rows = []
        for event_index, event in enumerate(summary["events"]):
            for role in ROLES:
                role_index = ROLES.index(role)
                content_index = FIELDS.index("content")
                magnitude = data["relative_mean"][
                    :, :, event_index, role_index, content_index
                ].mean()
                shared_content = []
                shared_null = []
                split_cosines = []
                split_advantages = []
                residual_top1 = []
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    for world in WORLDS:
                        content_discovery, fraction = shared_centroid(
                            data, (world,), "discovery", "content", replicate,
                            role=role, event_index=event_index,
                        )
                        shared_content.append(fraction)
                        content_confirmation, fraction = shared_centroid(
                            data, (world,), "confirmation", "content", replicate,
                            role=role, event_index=event_index,
                        )
                        shared_content.append(fraction)
                        null_discovery, fraction = shared_centroid(
                            data, (world,), "discovery", "field_null", replicate,
                            role=role, event_index=event_index,
                        )
                        shared_null.append(fraction)
                        null_confirmation, fraction = shared_centroid(
                            data, (world,), "confirmation", "field_null", replicate,
                            role=role, event_index=event_index,
                        )
                        shared_null.append(fraction)
                        content_cosine = cosine(
                            content_discovery, content_confirmation
                        )
                        null_cosine = cosine(null_discovery, null_confirmation)
                        split_cosines.append(content_cosine)
                        split_advantages.append(content_cosine - null_cosine)
                    source = operation_bank(
                        data, WORLDS, "discovery", "content", replicate,
                        centered=True, role=role, event_index=event_index,
                    )
                    target = operation_bank(
                        data, WORLDS, "confirmation", "content", replicate,
                        centered=True, role=role, event_index=event_index,
                    )
                    residual_top1.append(int(np.sum(
                        np.argmax(source @ target.T, axis=1)
                        == np.arange(len(OPERATIONS))
                    )))
                role_rows.append({
                    **event,
                    "role": role,
                    "mean_content_relative_magnitude": float(magnitude),
                    "median_content_signed_shared_fraction": float(
                        np.median(shared_content)
                    ),
                    "median_null_signed_shared_fraction": float(
                        np.median(shared_null)
                    ),
                    "mean_content_split_cosine": float(np.mean(split_cosines)),
                    "mean_content_over_null_split_advantage": float(
                        np.mean(split_advantages)
                    ),
                    "attribute_split_top1_by_replicate": residual_top1,
                })
        by_model[model_name] = {
            "rows": role_rows,
            "top_shared_advantage": sorted(
                role_rows,
                key=lambda row: row[
                    "mean_content_over_null_split_advantage"
                ],
                reverse=True,
            )[:12],
            "top_attribute_residual": sorted(
                role_rows,
                key=lambda row: (
                    min(row["attribute_split_top1_by_replicate"]),
                    row["mean_content_over_null_split_advantage"],
                ),
                reverse=True,
            )[:12],
        }
    return {
        "by_model": by_model,
        "scope": "preregistered_middle_band_descriptive_map",
        "causal_selection_authorized": False,
    }


def projection_gate(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    median_limit = float(
        protocol.EVIDENCE_THRESHOLDS[
            "maximum_projection_median_abs_norm_error"
        ]
    )
    p95_limit = float(
        protocol.EVIDENCE_THRESHOLDS[
            "maximum_projection_p95_abs_norm_error"
        ]
    )
    by_model = {}
    for model_name, data in models.items():
        rows = []
        for row in data["summary"]["projection_audit"]["replicates"]:
            passed = (
                row["median_abs_norm_error"] <= median_limit
                and row["p95_abs_norm_error"] <= p95_limit
            )
            rows.append({**row, "passed": passed})
        by_model[model_name] = {
            "replicates": rows,
            "passed": all(row["passed"] for row in rows),
        }
    return {
        "by_model": by_model,
        "passed": all(row["passed"] for row in by_model.values()),
    }


def numeric_gate(
    models: dict[str, dict[str, Any]],
    behavior_authorization: dict[str, Any],
) -> dict[str, Any]:
    candidate_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_candidate_finite_fraction"]
    )
    hidden_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_hidden_finite_fraction"]
    )
    prequery_max = float(
        protocol.EVIDENCE_THRESHOLDS["pre_query_tolerance"]
    )
    by_model = {}
    for model_name, data in models.items():
        summary = data["summary"]
        precision = summary["precision"]
        behavior_finite = behavior_authorization["models"][model_name][
            "candidate_finite_fraction"
        ]
        scan_finite = summary["candidate_finite_fraction"]
        checks = {
            "fp16_parameters": bool(precision["has_fp16_parameters"]),
            "no_quantized_modules": not bool(precision["has_quantized_modules"]),
            "behavior_candidate_finite": behavior_finite >= candidate_min,
            "scan_candidate_finite": scan_finite >= candidate_min,
            "hidden_finite": summary["hidden_finite_fraction_lower_bound"] >= hidden_min,
            "pre_query": summary["pre_query_global_max_abs"] <= prequery_max,
            "identity": summary["identity_maximum"] <= prequery_max,
        }
        by_model[model_name] = {
            "behavior_candidate_finite_fraction": behavior_finite,
            "scan_candidate_finite_fraction": scan_finite,
            "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
            "pre_query_global_max_abs": summary["pre_query_global_max_abs"],
            "identity_maximum": summary["identity_maximum"],
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "by_model": by_model,
        "passed": all(row["passed"] for row in by_model.values()),
        "healthy_models": [
            name for name, row in by_model.items() if row["passed"]
        ],
    }


def write_output(path: Path, payload: dict[str, Any], digest_key: str) -> None:
    payload[digest_key] = protocol.digest(payload)
    protocol.write_json(path, payload)


def main() -> None:
    protocol_payload = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior_authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {name: load_model(name) for name in MODEL_NAMES}
    analysis_root = protocol.OUT_ROOT / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    shared = shared_field_analysis(models)
    controls = control_analysis(models)
    attributes = attribute_analysis(models)
    cross_model = cross_model_analysis(models)
    decomposition = decomposition_analysis(models)
    physical = physical_map(models)
    projection = projection_gate(models)
    numeric = numeric_gate(models, behavior_authorization)

    minimum_models = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_behavior_models"]
    )
    p4_models = [
        name for name, row in shared["by_model"].items()
        if row["split_gate_passed"]
    ]
    p5_models = [
        name for name, row in shared["by_model"].items()
        if row["cross_world_gate_passed"]
    ]
    p6_models = [
        name for name, row in controls["by_model"].items()
        if row["combined_gate_passed"]
    ]
    p7_models = [
        name for name, row in attributes["by_model"].items()
        if row["split_gate_passed"]
    ]
    p8_models = [
        name for name, row in attributes["by_model"].items()
        if row["heldout_gate_passed"]
    ]
    predictions = {
        "P1": behavior_authorization["predictions"]["P1"],
        "P2": behavior_authorization["predictions"]["P2"],
        "P3": {"passed": projection["passed"], "by_model": projection["by_model"]},
        "P4": {"passed": len(p4_models) >= minimum_models, "passing_models": p4_models},
        "P5": {"passed": len(p5_models) >= minimum_models, "passing_models": p5_models},
        "P6": {"passed": len(p6_models) >= minimum_models, "passing_models": p6_models},
        "P7": {"passed": len(p7_models) >= minimum_models, "passing_models": p7_models},
        "P8": {"passed": len(p8_models) >= minimum_models, "passing_models": p8_models},
        "P9": {
            "passed": cross_model["gate_passed"],
            "passing_directed_pairs": cross_model["passing_directed_pairs"],
        },
        "P10": {"passed": numeric["passed"], "by_model": numeric["by_model"]},
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]
    discovery_gates = tuple(f"P{index}" for index in range(1, 9)) + ("P10",)
    full_atlas_authorized = all(predictions[name]["passed"] for name in discovery_gates)
    local_causal_authorized = full_atlas_authorized
    if full_atlas_authorized:
        decision = "continue_to_full_atlas_and_minimum_component_alliance"
    elif predictions["P4"]["passed"] and predictions["P5"]["passed"]:
        decision = "retain_shared_field_map_and_repair_residual_or_numeric_gates"
    else:
        decision = "stop_escalation_and_revise_shared_field_or_control_protocol"
    automatic = {
        "schema_version": "phase1086_automatic_next.v1",
        "phase": protocol.PHASE,
        "decision": decision,
        "offline_diagnostics_authorized": True,
        "full_atlas_authorized": full_atlas_authorized,
        "local_causal_authorized": local_causal_authorized,
        "failed_predictions": failed,
        "allowed_next": (
            "Use only the frozen signed atlas and independent protocol audit. "
            "Do not select components or neurons unless P1-P8 and P10 pass."
        ),
    }

    outputs = {
        "shared_field_audit.json": (
            {
                "schema_version": "phase1086_shared_field_audit.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                **shared,
            },
            "shared_field_digest",
        ),
        "surface_control_audit.json": (
            {
                "schema_version": "phase1086_surface_control_audit.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                **controls,
            },
            "surface_control_digest",
        ),
        "attribute_residual_audit.json": (
            {
                "schema_version": "phase1086_attribute_residual_audit.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                **attributes,
            },
            "attribute_residual_digest",
        ),
        "cross_model_geometry.json": (
            {
                "schema_version": "phase1086_cross_model_geometry.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                **cross_model,
            },
            "cross_model_geometry_digest",
        ),
        "signed_decomposition.json": (
            {
                "schema_version": "phase1086_signed_decomposition.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                **decomposition,
            },
            "signed_decomposition_digest",
        ),
        "physical_map.json": (
            {
                "schema_version": "phase1086_physical_map.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                **physical,
            },
            "physical_map_digest",
        ),
        "prediction_audit.json": (
            {
                "schema_version": "phase1086_prediction_audit.v1",
                "phase": protocol.PHASE,
                "protocol_digest": protocol_payload["protocol_digest"],
                "predictions": predictions,
                "passed_predictions": passed,
                "failed_predictions": failed,
            },
            "prediction_audit_digest",
        ),
        "automatic_next.json": (automatic, "automatic_next_digest"),
    }
    for filename, (payload, digest_key) in outputs.items():
        write_output(analysis_root / filename, payload, digest_key)

    final = {
        "schema_version": "phase1086_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": protocol_payload["protocol_digest"],
        "case_count_per_model": protocol_payload["case_count_per_model"],
        "unit_count_per_model": protocol_payload["unit_count_per_model"],
        "models": {
            name: {
                "summary_digest": data["summary"]["summary_digest"],
                "npz_sha256": data["npz_sha256"],
                "candidate_accuracy": data["summary"]["candidate_accuracy"],
                "candidate_finite_fraction": data["summary"]["candidate_finite_fraction"],
                "hidden_finite_fraction": data["summary"]["hidden_finite_fraction_lower_bound"],
                "event_count": data["summary"]["event_count"],
            }
            for name, data in models.items()
        },
        "predictions": predictions,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "shared_field_summary": {
            name: {
                "split_gate_passed": row["split_gate_passed"],
                "cross_world_gate_passed": row["cross_world_gate_passed"],
                "median_signed_shared_fraction": row[
                    "median_signed_shared_fraction"
                ],
            }
            for name, row in shared["by_model"].items()
        },
        "control_summary": controls["by_model"],
        "attribute_summary": {
            name: {
                "split_gate_passed": row["split_gate_passed"],
                "heldout_gate_passed": row["heldout_gate_passed"],
                "replicate_split_top1": [
                    value["split_assignment"]["top1_correct"]
                    for value in row["replicates"]
                ],
                "replicate_heldout_pass_counts": [
                    value["passing_heldout_worlds"]
                    for value in row["replicates"]
                ],
            }
            for name, row in attributes["by_model"].items()
        },
        "cross_model_geometry": cross_model,
        "numeric_integrity": numeric,
        "automatic_next": automatic,
        "evidence_boundary": {
            "supports": (
                "Descriptive signed middle-band mapping with independent "
                "negative controls and held-out lexical-world tests."
            ),
            "does_not_support": [
                "a complete language mechanism",
                "a shared semantic direction without content-over-null advantage",
                "direct neuron correspondence across models",
                "brain homology or evolutionary optimality",
                "causal component or neuron selection when frozen gates fail",
            ],
        },
    }
    write_output(analysis_root / "final_summary.json", final, "summary_digest")
    print({
        "phase": protocol.PHASE,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "decision": decision,
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
