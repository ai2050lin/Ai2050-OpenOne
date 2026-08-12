#!/usr/bin/env python3
"""Evaluate frozen Phase1100 lexical inheritance and interface gates."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import phase1099_relation_family_atlas_protocol as source_protocol
import phase1100_relation_graph_inheritance_protocol as protocol


EPSILON = 1e-12


def centered_gram(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    normalized = np.zeros_like(values)
    valid = np.isfinite(norms) & (norms > EPSILON)
    normalized[valid] = values[valid] / norms[valid, None]
    gram = normalized @ normalized.T
    invalid = ~(valid[:, None] & valid[None, :])
    gram[invalid] = np.nan
    return gram


def graph_vector(graph: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(graph, dtype=np.float64)
    indices = np.triu_indices(values.shape[0], 1)
    vector = values[indices]
    finite = np.isfinite(vector)
    finite_fraction = float(finite.mean()) if vector.size else 0.0
    if not finite.any():
        return np.zeros_like(vector), finite_fraction
    mean = float(vector[finite].mean())
    vector = np.where(finite, vector - mean, 0.0)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPSILON:
        return np.zeros_like(vector), finite_fraction
    return vector / norm, finite_fraction


def mappings() -> tuple[list[np.ndarray], list[np.ndarray]]:
    identity = np.arange(15, dtype=np.int64)
    family = []
    for permutation in itertools.permutations(range(5)):
        mapping = np.concatenate([np.arange(index * 3, index * 3 + 3) for index in permutation])
        if not np.array_equal(mapping, identity):
            family.append(mapping)
    local = list(itertools.permutations(range(3)))
    within = []
    for choices in itertools.product(local, repeat=5):
        mapping = np.concatenate([np.asarray(choice, dtype=np.int64) + family_index * 3 for family_index, choice in enumerate(choices)])
        if not np.array_equal(mapping, identity):
            within.append(mapping)
    return family, within


FAMILY_MAPPINGS, WITHIN_MAPPINGS = mappings()


def permutation_bank(graph: np.ndarray) -> dict[str, Any]:
    identity, finite_fraction = graph_vector(graph)

    def bank(values: list[np.ndarray]) -> np.ndarray:
        return np.stack([graph_vector(graph[np.ix_(mapping, mapping)])[0] for mapping in values])

    return {
        "identity": identity,
        "finite_fraction": finite_fraction,
        "family": bank(FAMILY_MAPPINGS),
        "within": bank(WITHIN_MAPPINGS),
    }


def score_bank(bank: dict[str, Any], target: np.ndarray) -> dict[str, float]:
    target_vector, target_finite = graph_vector(target)
    identity = float(bank["identity"] @ target_vector)
    family_best = float(np.max(bank["family"] @ target_vector))
    within_best = float(np.max(bank["within"] @ target_vector))
    return {
        "identity_score": identity,
        "best_wrong_family_score": family_best,
        "best_wrong_within_family_score": within_best,
        "family_permutation_margin": identity - family_best,
        "within_family_permutation_margin": identity - within_best,
        "source_finite_fraction": float(bank["finite_fraction"]),
        "target_finite_fraction": target_finite,
    }


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    numerator = np.where(finite, values, 0.0).sum(axis=axis)
    denominator = finite.sum(axis=axis)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def load_target(model: str) -> dict[str, Any]:
    root = protocol.SOURCE_ROOT / "atlas" / model
    summary = protocol.read_json(root / "summary.json")
    index = protocol.read_jsonl(root / "superunit_index.jsonl")
    with np.load(root / "relative_relation_geometry.npz", allow_pickle=False) as archive:
        gram = archive["relation_gram"].astype(np.float64)
    aggregate: dict[tuple[str, str], np.ndarray] = {}
    for surface in protocol.SURFACES:
        for split in protocol.SAMPLE_SPLITS:
            selected = [row_index for row_index, row in enumerate(index) if row["surface"] == surface and row["split"] == split]
            if not selected:
                raise RuntimeError(f"missing target cell {model} {surface} {split}")
            aggregate[(surface, split)] = finite_mean(gram[selected], axis=0)
    return {"summary": summary, "aggregate": aggregate}


def source_graphs(model: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = protocol.OUT_ROOT / "source" / model / "lexical_source.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].astype(np.float64) for key in archive.files}
    result = {}
    for surface_index, surface in enumerate(protocol.SURFACES):
        for relation_split in protocol.RELATION_SPLITS:
            indices = [
                index for index, relation in enumerate(source_protocol.RELATIONS)
                if source_protocol.RELATION_SPLIT[relation] == relation_split
            ]
            for source_name, array in arrays.items():
                graph = centered_gram(array[surface_index, indices])
                result[(surface, relation_split, source_name)] = permutation_bank(graph)
    return result


def relation_indices(relation_split: str) -> list[int]:
    return [
        index for index, relation in enumerate(source_protocol.RELATIONS)
        if source_protocol.RELATION_SPLIT[relation] == relation_split
    ]


def target_graph(target: dict[str, Any], surface: str, sample_split: str, event: int, field: str, role: str, relation_split: str) -> np.ndarray:
    summary = target["summary"]
    field_index = summary["fields"].index(field)
    role_index = summary["roles"].index(role)
    indices = relation_indices(relation_split)
    graph = target["aggregate"][(surface, sample_split)][event, field_index, role_index]
    return graph[np.ix_(indices, indices)]


def evaluate_cell(
    banks: dict[tuple[str, str, str], dict[str, Any]],
    target: dict[str, Any],
    surface: str,
    sample_split: str,
    relation_split: str,
    event: int,
) -> dict[str, Any]:
    primary_bank = banks[(surface, relation_split, protocol.PRIMARY_SOURCE)]
    primary_target = target_graph(target, surface, sample_split, event, protocol.PRIMARY_TARGET_FIELD, protocol.PRIMARY_TARGET_ROLE, relation_split)
    primary = score_bank(primary_bank, primary_target)
    controls = []
    for field in protocol.MATCHED_TARGET_CONTROLS:
        graph = target_graph(target, surface, sample_split, event, field, protocol.PRIMARY_TARGET_ROLE, relation_split)
        record = score_bank(primary_bank, graph)
        controls.append({"control": field, "maximum_alignment": max(record["identity_score"], record["best_wrong_family_score"], record["best_wrong_within_family_score"])})
    form_bank = banks[(surface, relation_split, protocol.FORM_SOURCE)]
    form_record = score_bank(form_bank, primary_target)
    controls.append({"control": protocol.FORM_SOURCE, "maximum_alignment": max(form_record["identity_score"], form_record["best_wrong_family_score"], form_record["best_wrong_within_family_score"])})
    maximum_control = max(value["maximum_alignment"] for value in controls)
    specificity = primary["identity_score"] - maximum_control
    thresholds = protocol.THRESHOLDS
    inheritance_pass = bool(
        primary["source_finite_fraction"] >= thresholds["minimum_graph_finite_fraction"]
        and primary["target_finite_fraction"] >= thresholds["minimum_graph_finite_fraction"]
        and primary["identity_score"] >= thresholds["minimum_inheritance_cosine"]
        and primary["family_permutation_margin"] >= thresholds["minimum_family_permutation_margin"]
        and primary["within_family_permutation_margin"] >= thresholds["minimum_within_family_permutation_margin"]
    )
    specificity_pass = bool(inheritance_pass and specificity >= thresholds["minimum_execution_specificity_advantage"])
    diagnostic = {}
    for field in protocol.DIAGNOSTIC_TARGET_FIELDS:
        graph = target_graph(target, surface, sample_split, event, field, protocol.PRIMARY_TARGET_ROLE, relation_split)
        diagnostic[field] = score_bank(primary_bank, graph)["identity_score"]
    alternative = score_bank(banks[(surface, relation_split, protocol.ALTERNATIVE_SOURCE)], primary_target)
    slacks = [
        primary["identity_score"] - thresholds["minimum_inheritance_cosine"],
        primary["family_permutation_margin"] - thresholds["minimum_family_permutation_margin"],
        primary["within_family_permutation_margin"] - thresholds["minimum_within_family_permutation_margin"],
        specificity - thresholds["minimum_execution_specificity_advantage"],
    ]
    return {
        "surface": surface,
        "sample_split": sample_split,
        "relation_split": relation_split,
        "event_index": event,
        **primary,
        "maximum_control_alignment": maximum_control,
        "execution_specificity_advantage": specificity,
        "controls": controls,
        "diagnostic_identity_scores": diagnostic,
        "output_source_identity_score": alternative["identity_score"],
        "minimum_gate_slack": min(slacks),
        "inheritance_pass": inheritance_pass,
        "specificity_pass": specificity_pass,
    }


def select_event(banks: dict[tuple[str, str, str], dict[str, Any]], target: dict[str, Any], surface: str) -> tuple[int, list[dict[str, Any]]]:
    records = [
        evaluate_cell(banks, target, surface, "discovery", "discovery", event)
        for event in range(len(target["summary"]["events"]))
    ]
    records.sort(key=lambda row: (row["minimum_gate_slack"], row["identity_score"], -row["event_index"]), reverse=True)
    return int(records[0]["event_index"]), records


def interpolate_curve(target: dict[str, Any], banks: dict[tuple[str, str, str], dict[str, Any]], surface: str, component: str) -> np.ndarray:
    events = [row for row in target["summary"]["events"] if row["component"] == component]
    xs = np.asarray([float(row["relative_depth"]) for row in events], dtype=np.float64)
    ys = np.asarray([
        evaluate_cell(banks, target, surface, "confirmation", "confirmation", int(row["event_index"]))["identity_score"]
        for row in events
    ], dtype=np.float64)
    order = np.argsort(xs)
    return np.interp(np.linspace(0.0, 1.0, 11), xs[order], ys[order])


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    right = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > EPSILON else 0.0


def main() -> None:
    preregistration = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1100 protocol audit failed")

    targets = {model: load_target(model) for model in protocol.MODELS}
    banks = {model: source_graphs(model) for model in protocol.MODELS}
    model_results = {}
    for model in protocol.MODELS:
        source_summary = protocol.read_json(protocol.OUT_ROOT / "source" / model / "summary.json")
        surfaces = {}
        for surface in protocol.SURFACES:
            selected_event, discovery_ranking = select_event(banks[model], targets[model], surface)
            discovery = next(row for row in discovery_ranking if row["event_index"] == selected_event)
            confirmation = [
                evaluate_cell(banks[model], targets[model], surface, sample_split, relation_split, selected_event)
                for sample_split, relation_split in (
                    ("confirmation", "discovery"),
                    ("discovery", "confirmation"),
                    ("confirmation", "confirmation"),
                )
            ]
            inheritance_count = sum(row["inheritance_pass"] for row in confirmation)
            specificity_count = sum(row["specificity_pass"] for row in confirmation)
            surfaces[surface] = {
                "selected_event": targets[model]["summary"]["events"][selected_event],
                "selection_used_only_discovery_sample_and_relations": True,
                "discovery_record": discovery,
                "confirmation_records": confirmation,
                "inheritance_confirmation_passes": inheritance_count,
                "specificity_confirmation_passes": specificity_count,
                "inheritance_surface_passed": inheritance_count >= protocol.THRESHOLDS["minimum_confirmation_cells_per_surface"],
                "specificity_surface_passed": specificity_count >= protocol.THRESHOLDS["minimum_confirmation_cells_per_surface"],
                "discovery_top_five": discovery_ranking[:5],
            }
        model_results[model] = {
            "behavior_formal": bool(targets[model]["summary"]["behavior_formal"]),
            "source_finite_fraction": source_summary["source_finite_fraction"],
            "hidden_finite_fraction": targets[model]["summary"]["hidden_finite_fraction"],
            "surfaces": surfaces,
            "inheritance_model_passed": sum(row["inheritance_surface_passed"] for row in surfaces.values()) >= protocol.THRESHOLDS["minimum_surface_passes_per_formal_model"],
            "specificity_model_passed": sum(row["specificity_surface_passed"] for row in surfaces.values()) >= protocol.THRESHOLDS["minimum_surface_passes_per_formal_model"],
        }

    cross_model_curves = []
    for surface in protocol.SURFACES:
        for component in ("residual", "attention_output", "mlp_output"):
            qwen = interpolate_curve(targets["qwen3"], banks["qwen3"], surface, component)
            glm = interpolate_curve(targets["glm4"], banks["glm4"], surface, component)
            cosine = centered_cosine(qwen, glm)
            error = float(np.mean(np.abs(qwen - glm)))
            passed = bool(
                cosine >= protocol.THRESHOLDS["minimum_cross_model_curve_cosine"]
                and error <= protocol.THRESHOLDS["maximum_cross_model_curve_mean_absolute_error"]
            )
            cross_model_curves.append(
                {
                    "surface": surface,
                    "component": component,
                    "qwen3_curve": qwen.tolist(),
                    "glm4_curve": glm.tolist(),
                    "centered_curve_cosine": cosine,
                    "mean_absolute_error": error,
                    "passed": passed,
                }
            )

    gates = {
        "P1": bool(protocol_audit["all_checks_passed"]),
        "P2": all(
            model_results[model]["source_finite_fraction"] >= protocol.THRESHOLDS["minimum_source_finite_fraction"]
            for model in protocol.MODELS
        ),
        "P3": all(
            model_results[model]["surfaces"][surface]["discovery_record"]["inheritance_pass"]
            for model in protocol.FORMAL_MODELS for surface in protocol.SURFACES
        ),
        "P4": all(model_results[model]["inheritance_model_passed"] for model in protocol.FORMAL_MODELS),
        "P5": all(model_results[model]["specificity_model_passed"] for model in protocol.FORMAL_MODELS),
        "P6": sum(row["passed"] for row in cross_model_curves) >= protocol.THRESHOLDS["minimum_cross_model_curve_cells"],
        "P7": True,
    }
    automatic_next = all(gates.values())
    result = {
        "schema_version": "phase1100_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "source_phase": 1099,
        "models": model_results,
        "cross_model_functional_trajectories": cross_model_curves,
        "gates": gates,
        "automatic_next_required": automatic_next,
        "decision": "authorize_independent_interface_phase" if automatic_next else "stop_without_interface_or_causal_claim",
        "frozen_interpretation": (
            "A lexical-to-execution inheritance interface is authorized only if identity-aligned graph prediction survives unseen relations and independent templates, beats exact label permutations and every matched control, and repeats functionally across Qwen3 and GLM4."
            if automatic_next
            else "The Phase1099 lexical-inheritance hypothesis did not clear all prospective gates; any positive graph similarity remains descriptive task/representation inheritance and cannot locate semantic content or authorize an interface/causal stage."
        ),
        "mathematics_status": "Centered Gram vectors, exact finite permutations, held-out split replication, matched controls, and depth-trajectory comparison are sufficient for this inheritance question; no new mathematics is needed.",
        "registered_family_permutations": len(FAMILY_MAPPINGS),
        "registered_within_family_permutations": len(WITHIN_MAPPINGS),
    }
    result["final_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print(json.dumps({"phase": protocol.PHASE, "gates": gates, "automatic_next_required": automatic_next, "final_digest": result["final_digest"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
