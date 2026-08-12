#!/usr/bin/env python3
"""Finalize Phase1093 independent bilingual relation replication."""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_finalize as common
import phase1092_natural_bilingual_attribute_protocol as prior_protocol
import phase1093_independent_relation_protocol as protocol


EPSILON = 1e-12


@contextlib.contextmanager
def using_protocol(value) -> Iterator[None]:
    previous = common.protocol
    common.protocol = value
    try:
        yield
    finally:
        common.protocol = previous


def load_current_models() -> dict[str, dict[str, Any]]:
    with using_protocol(protocol):
        return {name: common.load_model(name) for name in protocol.MODELS}


def load_prior_models() -> dict[str, dict[str, Any]]:
    with using_protocol(prior_protocol):
        return {name: common.load_model(name) for name in prior_protocol.MODELS}


def operation_names(attribute: str, module=protocol) -> tuple[str, ...]:
    return tuple(
        value for value in module.OPERATIONS
        if value.startswith(f"{attribute}_")
    )


def _event_indices(
    data: dict[str, Any], minimum: float, maximum: float
) -> tuple[int, ...]:
    return tuple(
        index for index, event in enumerate(data["summary"]["events"])
        if minimum <= float(event["relative_depth"]) <= maximum
    )


def band_bank(
    data: dict[str, Any],
    module,
    attribute: str,
    surface: str,
    split: str,
    field: str,
    replicate: int,
    minimum: float,
    maximum: float,
    *,
    worlds: tuple[str, ...] | None = None,
    role: str = "answer_boundary",
) -> np.ndarray:
    indices = _event_indices(data, minimum, maximum)
    if not indices:
        raise RuntimeError(f"no events in normalized band {minimum}-{maximum}")
    encoded_worlds = tuple(
        f"{world}@{surface}" for world in (worlds or module.BASE_WORLDS)
    )
    values = []
    for operation in operation_names(attribute, module):
        world_values = []
        for world in encoded_worlds:
            family = f"{operation}__{world}"
            fi = module.FAMILIES.index(family)
            si = module.SPLITS.index(split)
            ri = module.CAPTURE_ROLES.index(role)
            di = module.SIGNED_FIELDS.index(field)
            current = data["direction_mean"][
                fi, si, list(indices), ri, di, :, :, replicate, :
            ]
            current = current.mean(axis=(1, 2)).reshape(-1)
            world_values.append(current)
        values.append(np.mean(np.stack(world_values), axis=0))
    return common.row_normalize(np.stack(values), centered=True)


def gram_metrics(
    content_source: np.ndarray,
    content_target: np.ndarray,
    null_source: np.ndarray,
    null_target: np.ndarray,
    *,
    cross_phase: bool = False,
) -> dict[str, Any]:
    content = common.cosine(
        common.relation_vector(content_source),
        common.relation_vector(content_target),
    )
    null = common.cosine(
        common.relation_vector(null_source),
        common.relation_vector(null_target),
    )
    advantage = content - null
    cosine_threshold = float(protocol.EVIDENCE_THRESHOLDS[
        "minimum_cross_phase_gram_cosine"
        if cross_phase else "minimum_cross_language_gram_cosine"
    ])
    advantage_threshold = float(protocol.EVIDENCE_THRESHOLDS[
        "minimum_cross_phase_gram_advantage"
        if cross_phase else "minimum_cross_language_gram_advantage"
    ])
    return {
        "content_gram_cosine": content,
        "field_null_gram_cosine": null,
        "content_over_null_advantage": advantage,
        "passed": content >= cosine_threshold and advantage >= advantage_threshold,
    }


def cross_phase_analysis(
    current_models: dict[str, dict[str, Any]],
    prior_models: dict[str, dict[str, Any]],
    current_behavior: dict[str, Any],
) -> dict[str, Any]:
    prior_behavior = prior_protocol.read_json(
        prior_protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    minimum, maximum = (0.45, 0.62)
    fraction = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_phase_cell_fraction"]
    )
    by_model = {}
    for model_name in protocol.MODELS:
        attributes = {}
        for attribute in protocol.ATTRIBUTES:
            rows = []
            for prior_replicate in range(prior_protocol.SIGNED_PROJECTION_REPLICATES):
                for current_replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    for prior_surface in prior_protocol.SURFACES:
                        for current_surface in protocol.SURFACES:
                            for current_split in protocol.SPLITS:
                                old_content = band_bank(
                                    prior_models[model_name], prior_protocol, attribute,
                                    prior_surface, "confirmation", "content",
                                    prior_replicate, minimum, maximum,
                                )
                                new_content = band_bank(
                                    current_models[model_name], protocol, attribute,
                                    current_surface, current_split, "content",
                                    current_replicate, minimum, maximum,
                                )
                                old_null = band_bank(
                                    prior_models[model_name], prior_protocol, attribute,
                                    prior_surface, "confirmation", "field_null",
                                    prior_replicate, minimum, maximum,
                                )
                                new_null = band_bank(
                                    current_models[model_name], protocol, attribute,
                                    current_surface, current_split, "field_null",
                                    current_replicate, minimum, maximum,
                                )
                                rows.append({
                                    "prior_replicate": prior_replicate,
                                    "current_replicate": current_replicate,
                                    "prior_surface": prior_surface,
                                    "current_surface": current_surface,
                                    "current_split": current_split,
                                    **gram_metrics(
                                        old_content, new_content, old_null, new_null,
                                        cross_phase=True,
                                    ),
                                })
            formal = (
                attribute in prior_behavior["models"][model_name]["passing_attributes"]
                and attribute in current_behavior["models"][model_name]["passing_attributes"]
            )
            minimum_count = math.ceil(len(rows) * fraction)
            split_counts = {
                split: {
                    "passing": sum(
                        int(row["passed"]) for row in rows
                        if row["current_split"] == split
                    ),
                    "total": sum(
                        1 for row in rows if row["current_split"] == split
                    ),
                }
                for split in protocol.SPLITS
            }
            passed_count = sum(int(row["passed"]) for row in rows)
            mean_content = float(np.mean([
                row["content_gram_cosine"] for row in rows
            ]))
            mean_advantage = float(np.mean([
                row["content_over_null_advantage"] for row in rows
            ]))
            attributes[attribute] = {
                "formal": formal,
                "row_count": len(rows),
                "passing_count": passed_count,
                "minimum_passing_count": minimum_count,
                "split_counts": split_counts,
                "mean_content_gram_cosine": mean_content,
                "mean_content_over_null_advantage": mean_advantage,
                "rows": rows,
                "passed": (
                    formal
                    and passed_count >= minimum_count
                    and all(
                        value["passing"] >= math.ceil(value["total"] * fraction)
                        for value in split_counts.values()
                    )
                    and mean_content >= float(
                        protocol.EVIDENCE_THRESHOLDS["minimum_cross_phase_gram_cosine"]
                    )
                    and mean_advantage >= float(
                        protocol.EVIDENCE_THRESHOLDS["minimum_cross_phase_gram_advantage"]
                    )
                ),
            }
        by_model[model_name] = {"attributes": attributes}
    return {
        "normalized_band": [minimum, maximum],
        "prior_phase": 1092,
        "current_phase": protocol.PHASE,
        "by_model": by_model,
    }


def raw_bank(
    data: dict[str, Any],
    attribute: str,
    surface: str,
    split: str,
    field: str,
    replicate: int,
    worlds: tuple[str, ...],
) -> np.ndarray:
    encoded = tuple(f"{world}@{surface}" for world in worlds)
    return np.stack([
        common.profile(
            data, operation, encoded, split, field, replicate,
            role="answer_boundary",
        )
        for operation in operation_names(attribute)
    ])


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > EPSILON else np.zeros_like(value)


def dictionary_prediction(
    train_source_raw: np.ndarray,
    train_target_raw: np.ndarray,
    query_source_raw: np.ndarray,
    target_candidates_raw: np.ndarray,
    heldout_index: int,
    ridge: float,
) -> dict[str, Any]:
    train_indices = [index for index in range(8) if index != heldout_index]
    source_mean = train_source_raw[train_indices].mean(axis=0)
    target_mean = train_target_raw[train_indices].mean(axis=0)
    source_train = common.row_normalize(
        train_source_raw[train_indices] - source_mean, centered=False
    )
    target_train = common.row_normalize(
        train_target_raw[train_indices] - target_mean, centered=False
    )
    query = _normalize_vector(query_source_raw - source_mean)
    candidates = common.row_normalize(
        target_candidates_raw - target_mean, centered=False
    )
    kernel = source_train @ source_train.T
    coefficients = np.linalg.solve(
        kernel + ridge * np.eye(kernel.shape[0]),
        source_train @ query,
    )
    prediction = _normalize_vector(coefficients @ target_train)
    scores = prediction @ candidates.T
    baseline_scores = query @ candidates.T
    predicted = int(np.argmax(scores))
    return {
        "heldout_pair_index": heldout_index,
        "predicted_pair_index": predicted,
        "correct": predicted == heldout_index,
        "correct_cosine": float(scores[heldout_index]),
        "best_other_cosine": float(np.max(np.delete(scores, heldout_index))),
        "baseline_correct_cosine": float(baseline_scores[heldout_index]),
        "alignment_gain": float(
            scores[heldout_index] - baseline_scores[heldout_index]
        ),
        "coefficient_norm": float(np.linalg.norm(coefficients)),
    }


def alignment_analysis(
    models: dict[str, dict[str, Any]],
    behavior: dict[str, Any],
    primary_prerequisite: bool,
) -> dict[str, Any]:
    ridge = float(protocol.EVIDENCE_THRESHOLDS["alignment_ridge"])
    directions = (("en", "zh"), ("zh", "en"))
    by_model = {}
    for model_name, data in models.items():
        groups = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            for source_surface, target_surface in directions:
                rows = []
                for heldout_world in protocol.BASE_WORLDS:
                    train_worlds = tuple(
                        world for world in protocol.BASE_WORLDS
                        if world != heldout_world
                    )
                    content_source_train = raw_bank(
                        data, "size", source_surface, "discovery", "content",
                        replicate, train_worlds,
                    )
                    content_target_train = raw_bank(
                        data, "size", target_surface, "discovery", "content",
                        replicate, train_worlds,
                    )
                    null_source_train = raw_bank(
                        data, "size", source_surface, "discovery", "field_null",
                        replicate, train_worlds,
                    )
                    null_target_train = raw_bank(
                        data, "size", target_surface, "discovery", "field_null",
                        replicate, train_worlds,
                    )
                    content_source_test = raw_bank(
                        data, "size", source_surface, "confirmation", "content",
                        replicate, (heldout_world,),
                    )
                    content_target_test = raw_bank(
                        data, "size", target_surface, "confirmation", "content",
                        replicate, (heldout_world,),
                    )
                    null_source_test = raw_bank(
                        data, "size", source_surface, "confirmation", "field_null",
                        replicate, (heldout_world,),
                    )
                    null_target_test = raw_bank(
                        data, "size", target_surface, "confirmation", "field_null",
                        replicate, (heldout_world,),
                    )
                    for heldout_index in range(8):
                        content = dictionary_prediction(
                            content_source_train, content_target_train,
                            content_source_test[heldout_index], content_target_test,
                            heldout_index, ridge,
                        )
                        null = dictionary_prediction(
                            null_source_train, null_target_train,
                            null_source_test[heldout_index], null_target_test,
                            heldout_index, ridge,
                        )
                        rows.append({
                            "heldout_world": heldout_world,
                            "content": content,
                            "field_null": null,
                        })
                top1_fraction = float(np.mean([
                    row["content"]["correct"] for row in rows
                ]))
                null_top1_fraction = float(np.mean([
                    row["field_null"]["correct"] for row in rows
                ]))
                mean_cosine = float(np.mean([
                    row["content"]["correct_cosine"] for row in rows
                ]))
                null_mean_cosine = float(np.mean([
                    row["field_null"]["correct_cosine"] for row in rows
                ]))
                mean_gain = float(np.mean([
                    row["content"]["alignment_gain"] for row in rows
                ]))
                passed = (
                    top1_fraction >= float(protocol.EVIDENCE_THRESHOLDS[
                        "minimum_alignment_top1_fraction"
                    ])
                    and top1_fraction > null_top1_fraction
                    and mean_cosine - null_mean_cosine >= float(
                        protocol.EVIDENCE_THRESHOLDS[
                            "minimum_alignment_cosine_advantage"
                        ]
                    )
                    and mean_gain >= float(
                        protocol.EVIDENCE_THRESHOLDS["minimum_alignment_gain"]
                    )
                )
                groups.append({
                    "replicate": replicate,
                    "source_surface": source_surface,
                    "target_surface": target_surface,
                    "case_count": len(rows),
                    "content_top1_fraction": top1_fraction,
                    "field_null_top1_fraction": null_top1_fraction,
                    "mean_content_correct_cosine": mean_cosine,
                    "mean_field_null_correct_cosine": null_mean_cosine,
                    "content_over_null_cosine_advantage": mean_cosine - null_mean_cosine,
                    "mean_alignment_gain": mean_gain,
                    "rows": rows,
                    "passed": passed,
                })
        behavior_passed = "size" in behavior["models"][model_name][
            "passing_attributes"
        ]
        by_model[model_name] = {
            "behavior_passed": behavior_passed,
            "primary_prerequisite_met": primary_prerequisite,
            "groups": groups,
            "passed": (
                primary_prerequisite
                and behavior_passed
                and all(group["passed"] for group in groups)
            ),
        }
    return {
        "method": "leave_one_pair_and_one_world_out_kernel_dictionary_alignment",
        "ridge": ridge,
        "fit_scope": "discovery split and four non-heldout worlds only",
        "test_scope": "confirmation synonyms, heldout pair, and heldout world",
        "by_model": by_model,
    }


def size_physical_map(physical: dict[str, Any]) -> dict[str, Any]:
    by_model = {}
    for model_name, model_row in physical["by_model"].items():
        rows = []
        for row in model_row["rows"]:
            replicate_rows = []
            for replicate in row["replicates"]:
                cells = [
                    cell for cell in replicate["cells"]
                    if cell["attribute"] == "size" and cell["formal_attribute"]
                ]
                replicate_rows.append({
                    "replicate": replicate["replicate"],
                    "passing_cells": sum(int(cell["passed"]) for cell in cells),
                    "cell_count": len(cells),
                    "mean_gram_advantage": float(np.mean([
                        cell["gram"]["content_over_null_advantage"] for cell in cells
                    ])) if cells else 0.0,
                    "mean_identity_advantage": float(np.mean([
                        cell["identity_advantage"] for cell in cells
                    ])) if cells else 0.0,
                })
            complete = bool(replicate_rows) and all(
                value["cell_count"] > 0
                and value["passing_cells"] == value["cell_count"]
                for value in replicate_rows
            )
            rows.append({
                "event_index": row["event_index"],
                "event_id": row["event_id"],
                "component": row["component"],
                "depth": row["depth"],
                "relative_depth": row["relative_depth"],
                "role": row["role"],
                "replicates": replicate_rows,
                "complete_size_map": complete,
                "inside_phase1092_candidate_band": (
                    0.45 <= float(row["relative_depth"]) <= 0.62
                ),
            })
        ranked = sorted(
            rows,
            key=lambda row: (
                min((value["passing_cells"] for value in row["replicates"]), default=0),
                min((value["mean_gram_advantage"] for value in row["replicates"]), default=-1e9),
                min((value["mean_identity_advantage"] for value in row["replicates"]), default=-1e9),
            ),
            reverse=True,
        )
        by_model[model_name] = {
            "top_rows": ranked[:20],
            "complete_row_count": sum(int(row["complete_size_map"]) for row in rows),
            "candidate_band_complete_row_count": sum(
                int(row["complete_size_map"] and row["inside_phase1092_candidate_band"])
                for row in rows
            ),
            "passed": any(
                row["complete_size_map"] and row["inside_phase1092_candidate_band"]
                for row in rows
            ),
        }
    return {
        "phase1092_candidate_band": [0.45, 0.62],
        "by_model": by_model,
        "causal_selection_authorized": False,
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
    common.protocol = protocol
    models = load_current_models()
    prior_models = load_prior_models()
    common.protocol = protocol
    root = protocol.OUT_ROOT / "analysis"
    root.mkdir(parents=True, exist_ok=True)

    projection = common.projection_gate(models)
    numeric = common.numeric_gate(models, behavior)
    healthy = set(numeric["numeric_healthy_models"])
    formal_models = healthy.intersection(behavior["authorized_models"])
    within = common.within_language_analysis(models, behavior)
    cross_language = common.cross_language_analysis(models, behavior)
    heldout = common.heldout_world_analysis(models, behavior)
    cross_model = common.cross_model_analysis(models, behavior, formal_models)
    shared = common.shared_field_analysis(models)
    controls = common.control_analysis(models)
    decomposition = common.decomposition_analysis(models)
    physical = common.physical_map(models, behavior)
    physical["scope"] = "preregistered_relative_depth_0.15_to_0.80_descriptive_map"
    cross_phase = cross_phase_analysis(models, prior_models, behavior)

    p3_models = [
        name for name in formal_models if projection["by_model"][name]["passed"]
    ]
    p4_models = [
        name for name in formal_models
        if all(
            within["by_model"][name]["attributes"][attribute]["passed"]
            for attribute in protocol.ATTRIBUTES
        )
    ]
    size_gram_models = [
        name for name in formal_models
        if cross_language["by_model"][name]["attributes"]["size"]["gram_passed"]
    ]
    qg_size_cross_model_rows = [
        row for row in cross_model["rows"]
        if {row["source_model"], row["target_model"]} == {"qwen3", "glm4"}
        and row["attributes"]["size"]["passed"]
    ]
    p5_passed = (
        len(size_gram_models) >= 2 and len(qg_size_cross_model_rows) == 2
    )
    cross_phase_size_models = [
        name for name in ("qwen3", "glm4")
        if cross_phase["by_model"][name]["attributes"]["size"]["passed"]
    ]
    p6_passed = len(cross_phase_size_models) == 2
    glm_color = cross_language["by_model"]["glm4"]["attributes"]["color"]
    glm_color_full = (
        "glm4" in formal_models
        and glm_color["identity_passed"]
        and glm_color["gram_passed"]
        and heldout["by_model"]["glm4"]["attributes"]["color"]["passed"]
    )
    color_gram_models = [
        name for name in formal_models
        if cross_language["by_model"][name]["attributes"]["color"]["gram_passed"]
    ]
    p7_passed = glm_color_full and len(color_gram_models) >= 2
    size_heldout_models = [
        name for name in formal_models
        if heldout["by_model"][name]["attributes"]["size"]["passed"]
    ]
    p8_passed = len(size_heldout_models) >= 2
    alignment = alignment_analysis(models, behavior, p5_passed)
    alignment_models = [
        name for name in formal_models if alignment["by_model"][name]["passed"]
    ]
    p9_passed = len(alignment_models) >= 2
    size_map = size_physical_map(physical)
    size_map_models = [
        name for name in formal_models if size_map["by_model"][name]["passed"]
    ]
    p10_passed = len(size_map_models) >= 2

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
            "passed": len(p3_models) >= 2,
            "passing_models": sorted(p3_models),
        },
        "P4": {
            "passed": len(p4_models) >= 2,
            "passing_models": sorted(p4_models),
        },
        "P5": {
            "passed": p5_passed,
            "cross_language_size_models": sorted(size_gram_models),
            "qwen_glm_directed_cross_model_rows": len(qg_size_cross_model_rows),
        },
        "P6": {
            "passed": p6_passed,
            "passing_qwen_glm_models": cross_phase_size_models,
        },
        "P7": {
            "passed": p7_passed,
            "glm4_color_full_replication": glm_color_full,
            "color_gram_models": sorted(color_gram_models),
        },
        "P8": {
            "passed": p8_passed,
            "passing_size_heldout_models": sorted(size_heldout_models),
        },
        "P9": {
            "passed": p9_passed,
            "prerequisite_P5": p5_passed,
            "passing_alignment_models": sorted(alignment_models),
        },
        "P10": {
            "passed": p10_passed,
            "passing_size_band_models": sorted(size_map_models),
        },
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]

    if all(predictions[name]["passed"] for name in (
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"
    )):
        decision = (
            "two_family_cross_language_relation_candidate_replicated; authorize "
            "a new full_depth_component_replication, not neuron causality"
        )
        automatic_full_depth_authorized = True
    elif all(predictions[name]["passed"] for name in (
        "P1", "P2", "P3", "P4", "P5", "P6", "P8"
    )):
        decision = (
            "size_relation_geometry_independently_replicated_but_second_family_missing; "
            "retain one_family invariant candidate"
        )
        automatic_full_depth_authorized = False
    elif predictions["P5"]["passed"] and not predictions["P6"]["passed"]:
        decision = (
            "phase1093_size_geometry_repeats_internally_but_not_across_phase; "
            "do_not_promote the Phase1092 candidate"
        )
        automatic_full_depth_authorized = False
    else:
        decision = (
            "independent_size_candidate_not_confirmed; retain conditional "
            "language_and_lexical_relation atlas"
        )
        automatic_full_depth_authorized = False

    outputs = (
        ("within_language_identity.json", "phase1093_within_language_identity.v1", within, "within_digest"),
        ("cross_language_identity_geometry.json", "phase1093_cross_language_identity_geometry.v1", cross_language, "cross_language_digest"),
        ("heldout_world_geometry.json", "phase1093_heldout_world_geometry.v1", heldout, "heldout_digest"),
        ("cross_model_geometry.json", "phase1093_cross_model_geometry.v1", cross_model, "cross_model_digest"),
        ("cross_phase_geometry.json", "phase1093_cross_phase_geometry.v1", cross_phase, "cross_phase_digest"),
        ("dictionary_alignment.json", "phase1093_dictionary_alignment.v1", alignment, "alignment_digest"),
        ("shared_field.json", "phase1093_shared_field.v1", shared, "shared_field_digest"),
        ("control_audit.json", "phase1093_control_audit.v1", controls, "control_digest"),
        ("decomposition.json", "phase1093_decomposition.v1", decomposition, "decomposition_digest"),
        ("physical_map.json", "phase1093_physical_map.v1", physical, "physical_map_digest"),
        ("size_physical_map.json", "phase1093_size_physical_map.v1", size_map, "size_map_digest"),
        ("projection_audit.json", "phase1093_projection_audit.v1", projection, "projection_digest"),
        ("numeric_audit.json", "phase1093_numeric_audit.v1", numeric, "numeric_digest"),
    )
    for filename, schema, payload, digest_key in outputs:
        write_output(
            root, filename, schema, payload, digest_key, prereg["protocol_digest"]
        )

    automatic = {
        "schema_version": "phase1093_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "decision": decision,
        "automatic_full_depth_replication_authorized": automatic_full_depth_authorized,
        "local_causal_authorized": False,
        "reason": (
            "A descriptive signed map and low-rank prediction cannot authorize "
            "head, MLP, or neuron causality. Full-depth replication requires the "
            "two-family P1-P9 gate; causality requires a later frozen intervention."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)
    protocol.write_json(root / "automatic_next.json", automatic)

    summary = {
        "schema_version": "phase1093_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1092_summary_digest": prereg["source_phase1092_summary_digest"],
        "behavior_authorization_digest": behavior["summary_digest"],
        "numeric_healthy_models": sorted(healthy),
        "formal_models": sorted(formal_models),
        "predictions": predictions,
        "passed_predictions": passed,
        "failed_predictions": failed,
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
            "Passing synonym retrieval would support a family relation, not a complete concept code.",
            "A Gram match preserves pairwise geometry but is indifferent to coordinate rotation.",
            "The dictionary alignment has only seven training relations and remains a local predictive test.",
            "The field-null route retains lexical and task processing and is not a blank baseline.",
            "A repeated normalized band is descriptive, not a causal transport path.",
            "No completion percentage for language or AGI is scientifically defined.",
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
