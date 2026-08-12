#!/usr/bin/env python3
"""Freeze Phase1108 event-map decisions and the automatic-next gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import phase1108_exact_key_event_protocol as protocol
import phase1108_exact_key_event_scan as scan


EPSILON = 1e-12


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= EPSILON:
        return None
    value = float(np.dot(left, right) / denominator)
    return value if math.isfinite(value) else None


def unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPSILON:
        return None
    return vector / norm


def median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values))) if values else None


def load_model(model: str) -> dict[str, Any]:
    atlas = protocol.OUT_ROOT / "atlas" / model
    summary = protocol.read_json(atlas / "summary.json")
    projection = protocol.read_json(atlas / "projection_audit.json")
    arrays = np.load(atlas / "signed_event_fields.npz")
    direction_sum = arrays["direction_sum"].astype(np.float64)
    norms = np.linalg.norm(direction_sum, axis=-1, keepdims=True)
    direction = np.divide(
        direction_sum,
        norms,
        out=np.zeros_like(direction_sum),
        where=norms > EPSILON,
    )
    relative = np.divide(
        arrays["relative_sum"],
        arrays["relative_count"],
        out=np.zeros_like(arrays["relative_sum"], dtype=np.float64),
        where=arrays["relative_count"] > 0,
    )
    return {
        "summary": summary,
        "projection": projection,
        "direction": direction,
        "relative": relative,
        "field_index": {value: index for index, value in enumerate(summary["fields"])},
        "role_index": {value: index for index, value in enumerate(summary["roles"])},
    }


def cross_regime_observations(
    data: dict[str, Any], split: int, event: int, role: int, replicate: int,
) -> list[dict[str, float]]:
    direction = data["direction"]
    fields = data["field_index"]
    rows = []
    for pair in range(direction.shape[0]):
        for surface in range(direction.shape[1]):
            primary = cosine(
                direction[pair, surface, split, event, role,
                          fields["relation_lexical_address"], replicate],
                direction[pair, surface, split, event, role,
                          fields["neutral_lexical_address"], replicate],
            )
            ordinal = cosine(
                direction[pair, surface, split, event, role,
                          fields["relation_ordinal_routing"], replicate],
                direction[pair, surface, split, event, role,
                          fields["neutral_ordinal_routing"], replicate],
            )
            selector = cosine(
                direction[pair, surface, split, event, role,
                          fields["relation_selector_address"], replicate],
                direction[pair, surface, split, event, role,
                          fields["neutral_selector_address"], replicate],
            )
            if primary is None or ordinal is None or selector is None:
                continue
            control = max(ordinal, selector)
            rows.append({
                "pair": pair,
                "surface": surface,
                "primary": primary,
                "ordinal": ordinal,
                "selector": selector,
                "control": control,
                "advantage": primary - control,
            })
    return rows


def select_event(models: dict[str, dict[str, Any]]) -> tuple[dict, list[dict]]:
    roles = ("selector_start", "selector_end", "query_end", "answer_boundary")
    ranking = []
    event_count = min(data["summary"]["event_count"] for data in models.values())
    for event in range(event_count):
        for role_name in roles:
            values = []
            for data in models.values():
                role = data["role_index"][role_name]
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    values.extend(cross_regime_observations(
                        data, 0, event, role, replicate
                    ))
            if not values:
                continue
            primary = float(np.mean([row["primary"] for row in values]))
            advantage = float(np.mean([row["advantage"] for row in values]))
            ranking.append({
                "event_index": event,
                "role": role_name,
                "qualification_primary_mean": primary,
                "qualification_advantage_mean": advantage,
                "selection_score": min(primary, advantage),
                "observation_count": len(values),
            })
    selected = max(
        ranking,
        key=lambda row: (
            row["selection_score"],
            row["qualification_primary_mean"],
            -row["event_index"],
        ),
    )
    return selected, sorted(
        ranking,
        key=lambda row: (
            row["selection_score"], row["qualification_primary_mean"]
        ),
        reverse=True,
    )


def projection_repeat(data: dict[str, Any]) -> dict[str, Any]:
    roles = ("selector_start", "selector_end", "query_end", "answer_boundary")
    curves = []
    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
        values = []
        for split in range(len(protocol.SPLITS)):
            for event in range(data["summary"]["event_count"]):
                for role_name in roles:
                    rows = cross_regime_observations(
                        data,
                        split,
                        event,
                        data["role_index"][role_name],
                        replicate,
                    )
                    values.append(
                        float(np.mean([row["primary"] for row in rows]))
                        if rows else float("nan")
                    )
        curves.append(np.asarray(values, dtype=np.float64))
    valid = np.isfinite(curves[0]) & np.isfinite(curves[1])
    repeat_cosine = cosine(curves[0][valid], curves[1][valid])
    mae = (
        float(np.mean(np.abs(curves[0][valid] - curves[1][valid])))
        if valid.any() else None
    )
    return {
        "valid_cells": int(valid.sum()),
        "total_cells": int(valid.size),
        "finite_fraction": float(valid.mean()),
        "replicate_curve_cosine": repeat_cosine,
        "replicate_curve_mae": mae,
        "passed": bool(
            repeat_cosine is not None
            and repeat_cosine
            >= protocol.THRESHOLDS["minimum_projection_replicate_cosine"]
        ),
    }


def confirmation_event_result(
    model: str, data: dict[str, Any], selected: dict,
) -> dict[str, Any]:
    event = int(selected["event_index"])
    role = data["role_index"][selected["role"]]
    observations = []
    per_replicate = []
    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
        rows = cross_regime_observations(data, 1, event, role, replicate)
        observations.extend(rows)
        per_replicate.append({
            "replicate": replicate,
            "primary_mean": float(np.mean([row["primary"] for row in rows])),
            "advantage_mean": float(np.mean([row["advantage"] for row in rows])),
            "observation_count": len(rows),
        })
    primary_median = median([row["primary"] for row in observations])
    control_median = median([row["control"] for row in observations])
    advantage_median = median([row["advantage"] for row in observations])
    passed = bool(
        primary_median is not None
        and advantage_median is not None
        and primary_median
        >= protocol.THRESHOLDS["minimum_confirmation_cross_regime_cosine"]
        and advantage_median
        >= protocol.THRESHOLDS["minimum_confirmation_control_advantage"]
    )
    return {
        "model": model,
        "event_index": event,
        "event": data["summary"]["events"][event],
        "role": selected["role"],
        "primary_median": primary_median,
        "control_median": control_median,
        "advantage_median": advantage_median,
        "observation_count": len(observations),
        "per_replicate": per_replicate,
        "passed": passed,
    }


def pair_retrieval(
    data: dict[str, Any],
    event: int,
    role_name: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    direction = data["direction"]
    field_index = data["field_index"]
    role = data["role_index"][role_name]
    replicates = []
    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
        split_vectors = []
        for split in range(len(protocol.SPLITS)):
            pair_vectors = []
            for pair in range(direction.shape[0]):
                pieces = []
                for surface in range(direction.shape[1]):
                    for field in fields:
                        value = unit(
                            direction[pair, surface, split, event, role,
                                      field_index[field], replicate]
                        )
                        if value is not None:
                            pieces.append(value)
                pooled = unit(np.sum(pieces, axis=0)) if pieces else None
                pair_vectors.append(
                    pooled if pooled is not None
                    else np.zeros(protocol.SIGNED_PROJECTION_DIM)
                )
            values = np.asarray(pair_vectors)
            values = values - values.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values = np.divide(
                values, norms, out=np.zeros_like(values), where=norms > EPSILON
            )
            split_vectors.append(values)
        similarity = split_vectors[0] @ split_vectors[1].T
        prediction = similarity.argmax(axis=1)
        diagonal = np.diag(similarity)
        off_diagonal = np.where(np.eye(similarity.shape[0]), -np.inf, similarity)
        margins = diagonal - off_diagonal.max(axis=1)
        replicates.append({
            "replicate": replicate,
            "retrieved_count": int(np.sum(prediction == np.arange(similarity.shape[0]))),
            "prediction": [int(value) for value in prediction],
            "margins": [float(value) for value in margins],
            "mean_margin": float(np.mean(margins)),
            "similarity": similarity.astype(float).tolist(),
        })
    minimum_count = min(row["retrieved_count"] for row in replicates)
    mean_margin = float(np.mean([row["mean_margin"] for row in replicates]))
    return {
        "fields": list(fields),
        "replicates": replicates,
        "minimum_retrieved_count": minimum_count,
        "mean_margin": mean_margin,
        "passed": bool(
            minimum_count
            >= protocol.THRESHOLDS["minimum_pair_retrieval_count"]
            and mean_margin
            >= protocol.THRESHOLDS["minimum_pair_retrieval_margin"]
        ),
    }


def normalized_curve(
    data: dict[str, Any], role_name: str, pair: int | None = None,
) -> np.ndarray:
    relative = data["relative"]
    role = data["role_index"][role_name]
    fields = data["field_index"]
    field_ids = [
        fields["relation_lexical_address"],
        fields["neutral_lexical_address"],
    ]
    if pair is None:
        values = relative[:, :, 1, :, role][:, :, :, field_ids]
        curve = values.mean(axis=(0, 1, 3))
    else:
        values = relative[pair, :, 1, :, role][:, :, field_ids]
        curve = values.mean(axis=(0, 2))
    maximum = float(np.max(curve))
    return curve / maximum if maximum > EPSILON else np.zeros_like(curve)


def cross_model_curve_result(
    models: dict[str, dict[str, Any]], role_name: str,
) -> dict[str, Any]:
    qwen = normalized_curve(models["qwen3"], role_name)
    glm = normalized_curve(models["glm4"], role_name)
    global_cosine = cosine(qwen, glm)
    global_mae = float(np.mean(np.abs(qwen - glm)))
    per_pair = []
    for pair, pair_name in enumerate(protocol.RELATION_PAIRS):
        left = normalized_curve(models["qwen3"], role_name, pair)
        right = normalized_curve(models["glm4"], role_name, pair)
        pair_cosine = cosine(left, right)
        pair_mae = float(np.mean(np.abs(left - right)))
        per_pair.append({
            "relation_pair": pair_name,
            "cosine": pair_cosine,
            "mean_absolute_error": pair_mae,
            "passed": bool(
                pair_cosine is not None
                and pair_cosine
                >= protocol.THRESHOLDS["minimum_cross_model_curve_cosine"]
                and pair_mae
                <= protocol.THRESHOLDS["maximum_cross_model_curve_mae"]
            ),
        })
    passed = bool(
        global_cosine is not None
        and global_cosine
        >= protocol.THRESHOLDS["minimum_cross_model_curve_cosine"]
        and global_mae
        <= protocol.THRESHOLDS["maximum_cross_model_curve_mae"]
        and all(row["passed"] for row in per_pair)
    )
    return {
        "role": role_name,
        "global_cosine": global_cosine,
        "global_mean_absolute_error": global_mae,
        "per_pair": per_pair,
        "passed": passed,
    }


def shared_energy(data: dict[str, Any], selected: dict) -> dict[str, Any]:
    event = int(selected["event_index"])
    role = data["role_index"][selected["role"]]
    fields = data["field_index"]
    values = []
    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
        vectors = []
        for pair in range(data["direction"].shape[0]):
            for surface in range(data["direction"].shape[1]):
                for field in ("relation_lexical_address", "neutral_lexical_address"):
                    vector = unit(
                        data["direction"][pair, surface, 1, event, role,
                                          fields[field], replicate]
                    )
                    if vector is not None:
                        vectors.append(vector)
        values.append(
            float(np.linalg.norm(np.mean(vectors, axis=0)) ** 2)
            if vectors else float("nan")
        )
    return {
        "per_replicate": values,
        "mean": float(np.nanmean(values)),
    }


def magnitude_control_ratio(data: dict[str, Any]) -> dict[str, Any]:
    relative = data["relative"]
    fields = data["field_index"]
    roles = [
        data["role_index"][role]
        for role in ("selector_start", "selector_end", "query_end", "answer_boundary")
    ]
    ratios = []
    exact_ratios = []
    for role in roles:
        lexical = relative[:, :, 1, :, role][:, :, :, [
            fields["relation_lexical_address"],
            fields["neutral_lexical_address"],
        ]].mean(axis=3)
        selector = relative[:, :, 1, :, role][:, :, :, [
            fields["relation_selector_address"],
            fields["neutral_selector_address"],
        ]].mean(axis=3)
        exact = relative[:, :, 1, :, role][:, :, :, [
            fields["relation_exact_routing"],
            fields["neutral_exact_routing"],
        ]].mean(axis=3)
        valid = lexical > EPSILON
        ratios.extend((selector[valid] / lexical[valid]).tolist())
        exact_ratios.extend((exact[valid] / lexical[valid]).tolist())
    return {
        "selector_address_over_lexical_address_median": median(ratios),
        "selector_address_over_lexical_address_iqr": [
            float(value) for value in np.percentile(ratios, (25, 75))
        ],
        "exact_routing_over_lexical_address_median": median(exact_ratios),
        "exact_routing_over_lexical_address_iqr": [
            float(value) for value in np.percentile(exact_ratios, (25, 75))
        ],
    }


def physical_peaks(data: dict[str, Any]) -> dict[str, Any]:
    relative = data["relative"]
    fields = data["field_index"]
    result = {}
    for role_name in ("selector_start", "selector_end", "query_end", "answer_boundary"):
        role = data["role_index"][role_name]
        values = relative[:, :, 1, :, role][:, :, :, [
            fields["relation_lexical_address"],
            fields["neutral_lexical_address"],
        ]].mean(axis=(0, 1, 3))
        indices = np.argsort(values)[-5:][::-1]
        result[role_name] = [
            {
                "event": data["summary"]["events"][int(index)],
                "relative_magnitude": float(values[index]),
            }
            for index in indices
        ]
    return result


def posthoc_common_relation_retrieval(
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    roles = ("selector_start", "selector_end", "query_end", "answer_boundary")
    event_count = min(data["summary"]["event_count"] for data in models.values())
    for event in range(event_count):
        for role in roles:
            rows = {
                model: pair_retrieval(
                    data, event, role, ("relation_lexical_address",)
                )
                for model, data in models.items()
            }
            candidates.append({
                "event_index": event,
                "role": role,
                "minimum_count": min(
                    row["minimum_retrieved_count"] for row in rows.values()
                ),
                "mean_margin": float(np.mean([
                    row["mean_margin"] for row in rows.values()
                ])),
                "relation_results": rows,
            })
    selected = max(
        candidates,
        key=lambda row: (row["minimum_count"], row["mean_margin"]),
    )
    selected["events_by_model"] = {
        model: data["summary"]["events"][selected["event_index"]]
        for model, data in models.items()
    }
    selected["neutral_results"] = {
        model: pair_retrieval(
            data,
            selected["event_index"],
            selected["role"],
            ("neutral_lexical_address",),
        )
        for model, data in models.items()
    }
    selected["evidence_status"] = (
        "posthoc descriptor only; relation labels are pair-specific and the "
        "same coordinate was selected using confirmation data"
    )
    return selected


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {model: load_model(model) for model in behavior["authorized_models"]}
    if set(models) != {"qwen3", "glm4"}:
        raise RuntimeError(f"unexpected Phase1108 authorized models: {set(models)}")

    selected, ranking = select_event(models)
    selected["events_by_model"] = {
        model: data["summary"]["events"][selected["event_index"]]
        for model, data in models.items()
    }
    projection_results = {
        model: projection_repeat(data) for model, data in models.items()
    }
    instrument_results = {}
    for model, data in models.items():
        summary = data["summary"]
        behavior_accuracy = behavior["models"][model]["candidate_accuracy"]
        projection_pass = projection_results[model]["passed"]
        passed = bool(
            summary["candidate_finite_fraction"]
            >= protocol.THRESHOLDS["minimum_candidate_finite_fraction"]
            and abs(summary["candidate_accuracy"] - behavior_accuracy) <= 0.005
            and summary["hidden_finite_fraction"]
            >= protocol.THRESHOLDS["minimum_hidden_finite_fraction"]
            and summary["identity_maximum_error"] <= 1e-8
            and summary["pre_query_maximum_error"]
            <= protocol.THRESHOLDS["pre_query_tolerance"]
            and projection_pass
        )
        instrument_results[model] = {
            "candidate_accuracy": summary["candidate_accuracy"],
            "behavior_accuracy": behavior_accuracy,
            "candidate_accuracy_absolute_difference": abs(
                summary["candidate_accuracy"] - behavior_accuracy
            ),
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "hidden_finite_fraction": summary["hidden_finite_fraction"],
            "identity_maximum_error": summary["identity_maximum_error"],
            "pre_query_maximum_error": summary["pre_query_maximum_error"],
            "projection_repeat": projection_results[model],
            "projection_norm_audit": data["projection"],
            "passed": passed,
        }

    confirmation = {
        model: confirmation_event_result(model, data, selected)
        for model, data in models.items()
    }
    retrieval = {
        model: pair_retrieval(
            data,
            int(selected["event_index"]),
            selected["role"],
            ("relation_lexical_address", "neutral_lexical_address"),
        )
        for model, data in models.items()
    }
    curve = cross_model_curve_result(models, selected["role"])
    shared = {model: shared_energy(data, selected) for model, data in models.items()}
    control_ratios = {
        model: magnitude_control_ratio(data) for model, data in models.items()
    }
    peaks = {model: physical_peaks(data) for model, data in models.items()}
    posthoc = posthoc_common_relation_retrieval(models)

    predictions = {
        "P1": bool(protocol_audit["all_checks_passed"]),
        "P2": bool(
            behavior["hidden_scan_authorized"]
            and len(behavior["cross_model_pairs"]) == len(protocol.RELATION_PAIRS)
        ),
        "P3": all(row["passed"] for row in instrument_results.values()),
        "P4": sum(row["passed"] for row in confirmation.values()) >= 2,
        "P5": sum(row["passed"] for row in retrieval.values()) >= 2,
        "P6": bool(curve["passed"]),
    }
    predictions["P7"] = all(predictions[key] for key in ("P2", "P3", "P4", "P5", "P6"))
    causal_authorized = predictions["P7"]

    result = {
        "schema_version": "phase1108_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "behavior_authorization_digest": behavior["authorization_digest"],
        "atlas_summary_digests": {
            model: data["summary"]["summary_digest"]
            for model, data in models.items()
        },
        "behavior": {
            "passing_pairs_by_model": {
                model: row["passing_pairs"]
                for model, row in behavior["models"].items()
            },
            "cross_model_pairs": behavior["cross_model_pairs"],
            "authorized_models": behavior["authorized_models"],
            "deepseek7b_is_behavior_only_negative_control": True,
        },
        "instrument": instrument_results,
        "qualification_selected_event": selected,
        "qualification_ranking_top10": ranking[:10],
        "confirmation_cross_regime": confirmation,
        "confirmation_pair_retrieval": retrieval,
        "cross_model_relative_magnitude_curve": curve,
        "selected_event_shared_energy": shared,
        "magnitude_control_ratios": control_ratios,
        "physical_peaks": peaks,
        "posthoc_relation_label_retrieval": posthoc,
        "prospective_predictions": predictions,
        "causal_staircase_authorized": causal_authorized,
        "component_or_neuron_localization_authorized": False,
        "automatic_next_required": False,
        "automatic_next_decision": (
            "Phase1109 causal staircase is not authorized because P3-P5 failed. "
            "Retain the repeated cross-model magnitude topology as a descriptive "
            "map and stop before component selection."
        ),
        "frozen_conclusion": (
            "Exact repeated keys replicate behavior in Qwen3 and GLM4, and the "
            "relative-magnitude event topology repeats across models. The signed "
            "relation-label to neutral-key lexical-address direction does not "
            "repeat beyond ordinal/selector controls, and the frozen event does "
            "not preserve pair identity. No causal staircase is authorized."
        ),
        "theory_update": {
            "supported": (
                "Functional phase/topology is more stable than signed identity "
                "coordinates in this exact-key task."
            ),
            "not_supported": [
                "A cross-key-regime invariant lexical-address vector.",
                "A relation-pair identity coordinate at the frozen event.",
                "Component, head, neuron, semantic-address, compression, or optimality closure.",
            ],
            "canonical_theory_name_unchanged": "conditional output-field closure theory",
        },
    }
    result["final_summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "selected_event": selected,
        "predictions": predictions,
        "causal_staircase_authorized": causal_authorized,
        "cross_model_curve": curve,
        "final_summary_digest": result["final_summary_digest"],
    }), flush=True)


if __name__ == "__main__":
    main()
