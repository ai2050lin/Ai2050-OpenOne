#!/usr/bin/env python3
"""Evaluate frozen Phase1101 relation-address routing and inheritance gates."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from typing import Any

import numpy as np

import phase1101_relation_identity_routing_protocol as protocol


EPSILON = 1e-12


def centered_gram(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1)
    normalized = np.zeros_like(values)
    valid = np.isfinite(norms) & (norms > EPSILON)
    normalized[valid] = values[valid] / norms[valid, None]
    gram = normalized @ normalized.T
    gram[~(valid[:, None] & valid[None, :])] = np.nan
    return gram


def graph_vector(graph: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(graph, dtype=np.float64)
    indices = np.triu_indices(values.shape[0], 1)
    vector = values[indices]
    finite = np.isfinite(vector)
    finite_fraction = float(finite.mean()) if vector.size else 0.0
    if not finite.any():
        return np.zeros_like(vector), finite_fraction
    vector = np.where(finite, vector - float(vector[finite].mean()), 0.0)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPSILON:
        return np.zeros_like(vector), finite_fraction
    return vector / norm, finite_fraction


def centered_graph_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_vector, _ = graph_vector(left)
    right_vector, _ = graph_vector(right)
    return float(left_vector @ right_vector)


def mappings() -> tuple[list[np.ndarray], list[np.ndarray]]:
    identity = np.arange(15, dtype=np.int64)
    family = []
    for permutation in itertools.permutations(range(5)):
        mapping = np.concatenate([
            np.arange(index * 3, index * 3 + 3) for index in permutation
        ])
        if not np.array_equal(mapping, identity):
            family.append(mapping)
    local = list(itertools.permutations(range(3)))
    within = []
    for choices in itertools.product(local, repeat=5):
        mapping = np.concatenate([
            np.asarray(choice, dtype=np.int64) + family_index * 3
            for family_index, choice in enumerate(choices)
        ])
        if not np.array_equal(mapping, identity):
            within.append(mapping)
    return family, within


FAMILY_MAPPINGS, WITHIN_MAPPINGS = mappings()


def permutation_bank(graph: np.ndarray) -> dict[str, Any]:
    identity, finite_fraction = graph_vector(graph)

    def bank(mappings_: list[np.ndarray]) -> np.ndarray:
        return np.stack([
            graph_vector(graph[np.ix_(mapping, mapping)])[0]
            for mapping in mappings_
        ])

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
    root = protocol.OUT_ROOT / "atlas" / model
    summary = protocol.read_json(root / "summary.json")
    index = protocol.read_jsonl(root / "superunit_index.jsonl")
    with np.load(root / "relation_identity_routing_geometry.npz", allow_pickle=False) as archive:
        gram = archive["pair_gram"].astype(np.float64)
        shared = archive["shared_energy"].astype(np.float64)
        differential = archive["differential_energy"].astype(np.float64)
    aggregate = {}
    for surface in protocol.SURFACES:
        for split in protocol.SPLITS:
            selected = [
                row_index for row_index, row in enumerate(index)
                if row["surface"] == surface and row["split"] == split
            ]
            if not selected:
                raise RuntimeError(f"missing Phase1101 target cell {model} {surface} {split}")
            aggregate[(surface, split)] = {
                "gram": finite_mean(gram[selected], axis=0),
                "shared": finite_mean(shared[selected], axis=0),
                "differential": finite_mean(differential[selected], axis=0),
            }
    return {"summary": summary, "aggregate": aggregate}


def source_banks(model: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = protocol.SOURCE_ROOT / "source" / model / "lexical_source.npz"
    summary = protocol.read_json(protocol.SOURCE_ROOT / "source" / model / "summary.json")
    if tuple(summary["relations"]) != protocol.RELATIONS:
        raise RuntimeError(f"Phase1100 source relation order drift for {model}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].astype(np.float64) for key in archive.files}
    result = {}
    for surface_index, surface in enumerate(protocol.SURFACES):
        for source_name, array in arrays.items():
            pair_vectors = np.stack([
                array[surface_index, relation_index + 1]
                - array[surface_index, relation_index]
                for relation_index in range(0, len(protocol.RELATIONS), 2)
            ])
            result[(surface, source_name)] = permutation_bank(
                centered_gram(pair_vectors)
            )
    return result


def target_graph(target: dict[str, Any], surface: str, split: str, event: int, field: str, role: str) -> np.ndarray:
    summary = target["summary"]
    field_index = summary["fields"].index(field)
    role_index = summary["roles"].index(role)
    return target["aggregate"][(surface, split)]["gram"][event, field_index, role_index]


def evaluate_cell(
    banks: dict[tuple[str, str], dict[str, Any]],
    target: dict[str, Any],
    surface: str,
    split: str,
    event: int,
) -> dict[str, Any]:
    primary_bank = banks[(surface, "input_query_polarity")]
    primary_target = target_graph(
        target, surface, split, event, protocol.PRIMARY_FIELD, protocol.PRIMARY_ROLE
    )
    primary = score_bank(primary_bank, primary_target)
    controls = []
    for field in protocol.MATCHED_CONTROLS:
        graph = target_graph(target, surface, split, event, field, protocol.PRIMARY_ROLE)
        record = score_bank(primary_bank, graph)
        controls.append({
            "control": field,
            "identity_score": record["identity_score"],
            "maximum_alignment": max(
                record["identity_score"], record["best_wrong_family_score"],
                record["best_wrong_within_family_score"],
            ),
        })
    form_record = score_bank(banks[(surface, "query_token_form")], primary_target)
    controls.append({
        "control": "query_token_form",
        "identity_score": form_record["identity_score"],
        "maximum_alignment": max(
            form_record["identity_score"], form_record["best_wrong_family_score"],
            form_record["best_wrong_within_family_score"],
        ),
    })
    maximum_control = max(row["maximum_alignment"] for row in controls)
    specificity = primary["identity_score"] - maximum_control
    output_source = score_bank(
        banks[(surface, "output_query_polarity")], primary_target
    )
    thresholds = protocol.THRESHOLDS
    inheritance_pass = bool(
        primary["source_finite_fraction"] >= thresholds["minimum_graph_finite_fraction"]
        and primary["target_finite_fraction"] >= thresholds["minimum_graph_finite_fraction"]
        and primary["identity_score"] >= thresholds["minimum_inheritance_cosine"]
        and primary["family_permutation_margin"] >= thresholds["minimum_family_permutation_margin"]
        and primary["within_family_permutation_margin"] >= thresholds["minimum_within_family_permutation_margin"]
    )
    specificity_pass = bool(
        inheritance_pass and specificity >= thresholds["minimum_specificity_advantage"]
    )
    slacks = (
        primary["identity_score"] - thresholds["minimum_inheritance_cosine"],
        primary["family_permutation_margin"] - thresholds["minimum_family_permutation_margin"],
        primary["within_family_permutation_margin"] - thresholds["minimum_within_family_permutation_margin"],
        specificity - thresholds["minimum_specificity_advantage"],
    )
    return {
        "surface": surface,
        "split": split,
        "event_index": event,
        **primary,
        "maximum_control_alignment": maximum_control,
        "semantic_specificity_advantage": specificity,
        "controls": controls,
        "output_source_identity_score": output_source["identity_score"],
        "minimum_gate_slack": min(slacks),
        "inheritance_pass": inheritance_pass,
        "specificity_pass": specificity_pass,
    }


def select_event(banks: dict, target: dict, surface: str) -> tuple[int, list[dict[str, Any]]]:
    records = [
        evaluate_cell(banks, target, surface, "discovery", event)
        for event in range(len(target["summary"]["events"]))
    ]
    records.sort(
        key=lambda row: (
            row["minimum_gate_slack"], row["identity_score"], -row["event_index"]
        ),
        reverse=True,
    )
    return int(records[0]["event_index"]), records


def field_replications(target: dict, surface: str, event: int) -> dict[str, float]:
    return {
        field: centered_graph_cosine(
            target_graph(target, surface, "discovery", event, field, protocol.PRIMARY_ROLE),
            target_graph(target, surface, "confirmation", event, field, protocol.PRIMARY_ROLE),
        )
        for field in protocol.FIELDS
    }


def interpolate_curve(target: dict, banks: dict, surface: str, component: str) -> np.ndarray:
    events = [
        row for row in target["summary"]["events"] if row["component"] == component
    ]
    xs = np.asarray([float(row["relative_depth"]) for row in events])
    ys = np.asarray([
        evaluate_cell(
            banks, target, surface, "confirmation", int(row["event_index"])
        )["identity_score"]
        for row in events
    ])
    order = np.argsort(xs)
    return np.interp(np.linspace(0.0, 1.0, 11), xs[order], ys[order])


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    right = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > EPSILON else 0.0


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1101 protocol audit failed")
    if not behavior["hidden_scan_authorized"]:
        raise RuntimeError("Phase1101 hidden scan was not authorized")

    targets = {model: load_target(model) for model in protocol.MODELS}
    banks = {model: source_banks(model) for model in protocol.MODELS}
    model_results = {}
    strongest_controls = Counter()
    for model in protocol.MODELS:
        surfaces = {}
        for surface in protocol.SURFACES:
            selected_event, discovery_ranking = select_event(
                banks[model], targets[model], surface
            )
            discovery = discovery_ranking[0]
            confirmation = evaluate_cell(
                banks[model], targets[model], surface, "confirmation", selected_event
            )
            replications = field_replications(targets[model], surface, selected_event)
            strongest = max(
                confirmation["controls"], key=lambda row: row["maximum_alignment"]
            )["control"]
            strongest_controls[strongest] += 1
            field_index = targets[model]["summary"]["fields"].index(protocol.PRIMARY_FIELD)
            role_index = targets[model]["summary"]["roles"].index(protocol.PRIMARY_ROLE)
            shared = float(targets[model]["aggregate"][(surface, "confirmation")]["shared"][selected_event, field_index, role_index])
            differential = float(targets[model]["aggregate"][(surface, "confirmation")]["differential"][selected_event, field_index, role_index])
            surfaces[surface] = {
                "selected_event": targets[model]["summary"]["events"][selected_event],
                "selection_used_discovery_only": True,
                "discovery_record": discovery,
                "confirmation_record": confirmation,
                "target_graph_replication_by_field": replications,
                "semantic_routing_shared_energy": shared,
                "semantic_routing_differential_energy": differential,
                "inheritance_surface_passed": bool(confirmation["inheritance_pass"]),
                "specificity_surface_passed": bool(confirmation["specificity_pass"]),
                "discovery_top_five": discovery_ranking[:5],
            }
        model_results[model] = {
            "behavior_formal": bool(behavior["models"][model]["model_behavior_passed"]),
            "hidden_finite_fraction": targets[model]["summary"]["hidden_finite_fraction"],
            "pre_query_maximum_error": targets[model]["summary"]["pre_query_maximum_error"],
            "identity_maximum_error": targets[model]["summary"]["identity_maximum_error"],
            "surfaces": surfaces,
            "inheritance_model_passed": all(
                row["inheritance_surface_passed"] for row in surfaces.values()
            ),
            "specificity_model_passed": all(
                row["specificity_surface_passed"] for row in surfaces.values()
            ),
        }

    cross_model_curves = []
    for surface in protocol.SURFACES:
        for component in protocol.COMPONENTS:
            qwen = interpolate_curve(
                targets["qwen3"], banks["qwen3"], surface, component
            )
            glm = interpolate_curve(
                targets["glm4"], banks["glm4"], surface, component
            )
            cosine = centered_cosine(qwen, glm)
            error = float(np.mean(np.abs(qwen - glm)))
            passed = bool(
                cosine >= protocol.THRESHOLDS["minimum_cross_model_curve_cosine"]
                and error <= protocol.THRESHOLDS["maximum_cross_model_curve_mean_absolute_error"]
            )
            cross_model_curves.append({
                "surface": surface,
                "component": component,
                "qwen3_curve": qwen.tolist(),
                "glm4_curve": glm.tolist(),
                "centered_curve_cosine": cosine,
                "mean_absolute_error": error,
                "passed": passed,
            })

    behavior_pass_count = sum(
        row["model_behavior_passed"] for row in behavior["models"].values()
    )
    hidden_integrity_count = sum(
        model_results[model]["hidden_finite_fraction"]
        >= protocol.THRESHOLDS["minimum_hidden_finite_fraction"]
        and model_results[model]["pre_query_maximum_error"]
        <= protocol.THRESHOLDS["pre_query_tolerance"]
        for model in protocol.MODELS
    )
    gates = {
        "P1": bool(protocol_audit["all_checks_passed"]),
        "P2": behavior_pass_count >= protocol.THRESHOLDS["minimum_behavior_models"],
        "P3": hidden_integrity_count >= 2,
        "P4": all(
            model_results[model]["inheritance_model_passed"]
            for model in protocol.FORMAL_MODELS
        ),
        "P5": all(
            model_results[model]["specificity_model_passed"]
            for model in protocol.FORMAL_MODELS
        ),
        "P6": sum(row["passed"] for row in cross_model_curves)
        >= protocol.THRESHOLDS["minimum_cross_model_curve_cells"],
        "P7": all(
            targets[model]["summary"]["primary_signature_excludes_output_gram"]
            and targets[model]["summary"]["exact_full_d_model_gram"]
            for model in protocol.MODELS
        ),
    }
    automatic_next = all(gates.values())
    confirmation_records = [
        model_results[model]["surfaces"][surface]["confirmation_record"]
        for model in protocol.FORMAL_MODELS for surface in protocol.SURFACES
    ]
    result = {
        "schema_version": "phase1101_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "behavior_authorization_digest": behavior["authorization_digest"],
        "source_phase": 1100,
        "models": model_results,
        "cross_model_functional_trajectories": cross_model_curves,
        "formal_confirmation_summary": {
            "cell_count": len(confirmation_records),
            "inheritance_passes": sum(row["inheritance_pass"] for row in confirmation_records),
            "specificity_passes": sum(row["specificity_pass"] for row in confirmation_records),
            "median_identity_score": float(np.median([row["identity_score"] for row in confirmation_records])),
            "median_family_permutation_margin": float(np.median([row["family_permutation_margin"] for row in confirmation_records])),
            "median_within_family_permutation_margin": float(np.median([row["within_family_permutation_margin"] for row in confirmation_records])),
            "median_specificity_advantage": float(np.median([row["semantic_specificity_advantage"] for row in confirmation_records])),
            "strongest_control_counts_all_models": dict(strongest_controls),
        },
        "gates": gates,
        "automatic_next_required": automatic_next,
        "decision": (
            "authorize_independent_causal_interface_phase"
            if automatic_next else "stop_without_lexical_semantic_interface_claim"
        ),
        "frozen_interpretation": (
            "The relation-token pair graph predicts a behavior-necessary semantic-address routing graph beyond ordinal and selector controls in two formal models; an independent causal interface phase is required next."
            if automatic_next
            else "Behaviorally necessary relation-address routing was tested, but lexical graph inheritance did not clear every prospective identity, permutation, specificity, and cross-model gate. Any repeated routing structure remains descriptive and cannot be called a lexical-semantic interface."
        ),
        "mathematics_status": "Exact full-state centered Gram geometry, complete 120 family permutations, complete 7775 nonidentity within-family permutations, independent templates, and matched semantic-versus-ordinal routing controls are sufficient for this test; no new mathematics is needed.",
        "registered_family_permutations": len(FAMILY_MAPPINGS),
        "registered_within_family_permutations": len(WITHIN_MAPPINGS),
    }
    result["final_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "gates": gates,
        "automatic_next_required": automatic_next,
        "formal_confirmation_summary": result["formal_confirmation_summary"],
        "final_digest": result["final_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
