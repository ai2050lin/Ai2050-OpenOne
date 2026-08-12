#!/usr/bin/env python3
"""Finalize Phase1096 three-ledger predictive tests and physical map."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1096_comparison_dynamics_protocol as protocol


EPSILON = 1e-12
DEPTH_BINS = 12


def unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if math.isfinite(norm) and norm > EPSILON else np.zeros_like(values)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = unit(left.reshape(-1).astype(np.float64, copy=False))
    b = unit(right.reshape(-1).astype(np.float64, copy=False))
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b))


def finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def load_models() -> dict[str, dict[str, Any]]:
    result = {}
    for model_name in protocol.MODELS:
        root = protocol.OUT_ROOT / "atlas" / model_name
        summary = protocol.read_json(root / "summary.json")
        with np.load(root / "three_ledger_fields.npz") as payload:
            arrays = {key: payload[key] for key in payload.files}
        counts = arrays["direction_count"].astype(np.float64)
        directions = np.divide(
            arrays["direction_sum"].astype(np.float64),
            counts[..., None],
            out=np.zeros_like(arrays["direction_sum"], dtype=np.float64),
            where=counts[..., None] > 0,
        )
        relative_counts = arrays["relative_count"].astype(np.float64)
        relative = np.divide(
            arrays["relative_sum"].astype(np.float64),
            relative_counts,
            out=np.zeros_like(arrays["relative_sum"], dtype=np.float64),
            where=relative_counts > 0,
        )
        result[model_name] = {
            "summary": summary,
            "directions": directions,
            "relative": relative,
        }
    return result


def indices(summary: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        "relation": {value: index for index, value in enumerate(summary["relations"])},
        "surface": {value: index for index, value in enumerate(summary["surfaces"])},
        "split": {value: index for index, value in enumerate(summary["splits"])},
        "role": {value: index for index, value in enumerate(summary["roles"])},
        "field": {value: index for index, value in enumerate(summary["fields"])},
    }


def signed_vector(
    model: dict[str, Any], relation: str, surface: str, split: str,
    field: str, replicate: int,
) -> np.ndarray:
    summary = model["summary"]
    lookup = indices(summary)
    values = model["directions"][
        lookup["relation"][relation],
        lookup["surface"][surface],
        lookup["split"][split],
        :,
        [lookup["role"][role] for role in protocol.DYNAMIC_ROLES],
        lookup["field"][field],
        replicate,
        :,
    ]
    return unit(np.asarray(values, dtype=np.float64).reshape(-1))


def heldout_records(
    model: dict[str, Any], field: str, null_field: str,
    source_surface: str, target_surface: str,
) -> list[dict[str, Any]]:
    records = []
    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
        source_content = {
            relation: signed_vector(
                model, relation, source_surface, "discovery", field, replicate
            )
            for relation in protocol.RELATIONS
        }
        source_null = {
            relation: signed_vector(
                model, relation, source_surface, "discovery", null_field, replicate
            )
            for relation in protocol.RELATIONS
        }
        for target in protocol.RELATIONS:
            others = [relation for relation in protocol.RELATIONS if relation != target]
            predicted_content = unit(np.mean([source_content[value] for value in others], axis=0))
            predicted_null = unit(np.mean([source_null[value] for value in others], axis=0))
            target_content = signed_vector(
                model, target, target_surface, "confirmation", field, replicate
            )
            target_null = signed_vector(
                model, target, target_surface, "confirmation", null_field, replicate
            )
            content_cosine = cosine(predicted_content, target_content)
            null_cosine = cosine(predicted_null, target_null)
            records.append({
                "replicate": replicate,
                "target_relation": target,
                "source_surface": source_surface,
                "target_surface": target_surface,
                "content_cosine": content_cosine,
                "carrier_cosine": null_cosine,
                "content_over_carrier_advantage": content_cosine - null_cosine,
            })
    return records


def direction_gate(records: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    per_relation = {}
    passing_relations = []
    for relation in protocol.RELATIONS:
        rows = [row for row in records if row["target_relation"] == relation]
        passed = all(
            row["content_cosine"] >= thresholds["minimum_split_cosine"]
            and row["content_over_carrier_advantage"]
            >= thresholds["minimum_content_over_carrier_advantage"]
            for row in rows
        )
        if passed:
            passing_relations.append(relation)
        per_relation[relation] = {
            "replicates": rows,
            "passed_both_sketches": passed,
        }
    return {
        "passing_relations": passing_relations,
        "passing_relation_count": len(passing_relations),
        "passed": len(passing_relations) >= thresholds["minimum_heldout_relations"],
        "per_relation": per_relation,
        "mean_content_cosine": finite_mean(np.asarray([row["content_cosine"] for row in records])),
        "mean_carrier_cosine": finite_mean(np.asarray([row["carrier_cosine"] for row in records])),
        "mean_advantage": finite_mean(np.asarray([row["content_over_carrier_advantage"] for row in records])),
    }


def split_repeat(model: dict[str, Any], field: str) -> dict[str, Any]:
    records = []
    for relation in protocol.RELATIONS:
        for surface in protocol.SURFACES:
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                score = cosine(
                    signed_vector(model, relation, surface, "discovery", field, replicate),
                    signed_vector(model, relation, surface, "confirmation", field, replicate),
                )
                records.append({
                    "relation": relation,
                    "surface": surface,
                    "replicate": replicate,
                    "cosine": score,
                })
    return {
        "field": field,
        "records": records,
        "mean_cosine": finite_mean(np.asarray([row["cosine"] for row in records])),
        "passing_record_count": sum(
            row["cosine"] >= protocol.EVIDENCE_THRESHOLDS["minimum_split_cosine"]
            for row in records
        ),
        "record_count": len(records),
    }


def hidden_audit(models: dict[str, dict[str, Any]], behavior: dict[str, Any]) -> dict[str, Any]:
    thresholds = protocol.EVIDENCE_THRESHOLDS
    records = {}
    passing_formal = []
    for model_name, model in models.items():
        summary = model["summary"]
        projection_passed = all(
            row["median_abs_norm_error"] <= thresholds["maximum_projection_median_abs_norm_error"]
            and row["p95_abs_norm_error"] <= thresholds["maximum_projection_p95_abs_norm_error"]
            for row in summary["projection_audit"]["replicates"]
        )
        checks = {
            "candidate_finite": summary["candidate_finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"],
            "hidden_finite": summary["hidden_finite_fraction_lower_bound"] >= thresholds["minimum_hidden_finite_fraction"],
            "identity_exact": summary["identity_maximum"] == 0.0,
            "pre_task_zero": summary["pre_task_control_execution_maximum"] <= thresholds["pre_task_tolerance"],
            "dual_projection": projection_passed,
            "behavior_formal": bool(behavior["models"][model_name]["model_behavior_passed"]),
        }
        passed = all(checks.values())
        if passed:
            passing_formal.append(model_name)
        records[model_name] = {"checks": checks, "passed": passed, "summary": summary}
    return {
        "models": records,
        "passing_formal_models": passing_formal,
        "passed": len(passing_formal) >= thresholds["minimum_behavior_models"],
    }


def functional_profile(model: dict[str, Any], field: str) -> np.ndarray:
    summary = model["summary"]
    lookup = indices(summary)
    values = model["relative"][..., lookup["field"][field]]
    # Average relation, surface, and split, leaving event x role.
    averaged = values.mean(axis=(0, 1, 2))
    rows = []
    for component in ("residual", "attention_output", "mlp_output"):
        for depth_bin in range(DEPTH_BINS):
            eligible = [
                index for index, event in enumerate(summary["events"])
                if event["component"] == component
                and min(DEPTH_BINS - 1, int(float(event["relative_depth"]) * DEPTH_BINS)) == depth_bin
            ]
            for role in protocol.DYNAMIC_ROLES:
                role_index = lookup["role"][role]
                rows.append(
                    float(averaged[eligible, role_index].mean()) if eligible else 0.0
                )
    return unit(np.asarray(rows, dtype=np.float64))


def cross_model_profiles(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records = []
    for left, right in itertools.combinations(protocol.MODELS, 2):
        content = cosine(
            functional_profile(models[left], "comparison_execution"),
            functional_profile(models[right], "comparison_execution"),
        )
        carrier = cosine(
            functional_profile(models[left], "comparison_carrier"),
            functional_profile(models[right], "comparison_carrier"),
        )
        advantage = content - carrier
        passed = (
            content >= protocol.EVIDENCE_THRESHOLDS["minimum_functional_profile_cosine"]
            and advantage >= protocol.EVIDENCE_THRESHOLDS["minimum_functional_profile_advantage"]
        )
        records.append({
            "left_model": left,
            "right_model": right,
            "content_profile_cosine": content,
            "carrier_profile_cosine": carrier,
            "content_over_carrier_advantage": advantage,
            "passed": passed,
        })
    formal_pairs = [
        row for row in records
        if row["left_model"] in ("qwen3", "glm4")
        and row["right_model"] in ("qwen3", "glm4")
    ]
    return {
        "records": records,
        "formal_pair_passed": bool(formal_pairs and formal_pairs[0]["passed"]),
    }


def magnitude_ratio(model: dict[str, Any], numerator: str, denominator: str) -> float:
    lookup = indices(model["summary"])
    dynamic_roles = [lookup["role"][role] for role in protocol.DYNAMIC_ROLES]
    numerator_values = model["relative"][..., dynamic_roles, lookup["field"][numerator]]
    denominator_values = model["relative"][..., dynamic_roles, lookup["field"][denominator]]
    numerator_median = float(np.median(numerator_values[numerator_values > 0])) if np.any(numerator_values > 0) else 0.0
    denominator_median = float(np.median(denominator_values[denominator_values > 0])) if np.any(denominator_values > 0) else 0.0
    return numerator_median / denominator_median if denominator_median > EPSILON else 0.0


def physical_map(model: dict[str, Any]) -> list[dict[str, Any]]:
    summary = model["summary"]
    lookup = indices(summary)
    field = lookup["field"]["comparison_execution"]
    relative = model["relative"][..., field]
    rows = []
    for event_index, event in enumerate(summary["events"]):
        for role in protocol.DYNAMIC_ROLES:
            role_number = lookup["role"][role]
            magnitude = float(relative[:, :, :, event_index, role_number].mean())
            split_scores = []
            for relation in protocol.RELATIONS:
                for surface in protocol.SURFACES:
                    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                        left = model["directions"][
                            lookup["relation"][relation], lookup["surface"][surface],
                            lookup["split"]["discovery"], event_index, role_number,
                            field, replicate,
                        ]
                        right = model["directions"][
                            lookup["relation"][relation], lookup["surface"][surface],
                            lookup["split"]["confirmation"], event_index, role_number,
                            field, replicate,
                        ]
                        split_scores.append(cosine(left, right))
            repeat = float(np.mean(split_scores)) if split_scores else 0.0
            rows.append({
                "event_id": event["event_id"],
                "component": event["component"],
                "depth": event["depth"],
                "relative_depth": event["relative_depth"],
                "role": role,
                "mean_relative_magnitude": magnitude,
                "mean_split_cosine": repeat,
                "descriptive_score": magnitude * max(0.0, repeat),
            })
    return sorted(rows, key=lambda row: row["descriptive_score"], reverse=True)[:12]


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json")
    models = load_models()
    hidden = hidden_audit(models, behavior)
    thresholds = prereg["evidence_thresholds"]
    ledgers = {}
    execution_gates = {}
    cross_language_gates = {}
    control_gates = {}
    for model_name, model in models.items():
        representation = split_repeat(model, "relational_representation")
        control_repeat = split_repeat(model, "comparison_control")
        execution_repeat = split_repeat(model, "comparison_execution")
        carrier_repeat = split_repeat(model, "comparison_carrier")
        within = {}
        control_within = {}
        for surface in protocol.SURFACES:
            within[surface] = direction_gate(
                heldout_records(
                    model, "comparison_execution", "comparison_carrier",
                    surface, surface,
                ),
                thresholds,
            )
            control_within[surface] = direction_gate(
                heldout_records(
                    model, "comparison_control", "comparison_carrier",
                    surface, surface,
                ),
                thresholds,
            )
        execution_passed = all(row["passed"] for row in within.values())
        control_passed = all(row["passed"] for row in control_within.values())
        cross_language = {}
        for source_surface, target_surface in (("en", "zh"), ("zh", "en")):
            key = f"{source_surface}_to_{target_surface}"
            cross_language[key] = direction_gate(
                heldout_records(
                    model, "comparison_execution", "comparison_carrier",
                    source_surface, target_surface,
                ),
                thresholds,
            )
        cross_language_passed = sum(row["passed"] for row in cross_language.values()) >= thresholds["minimum_cross_language_directions"]
        ledgers[model_name] = {
            "behavior_formal": models[model_name]["summary"]["behavior_formal"],
            "representation": representation,
            "control": control_repeat,
            "execution": execution_repeat,
            "carrier": carrier_repeat,
            "within_language_execution_prediction": within,
            "within_language_control_prediction": control_within,
            "cross_language_execution_prediction": cross_language,
            "comparison_execution_to_relational_execution_magnitude_ratio": magnitude_ratio(
                model, "comparison_execution", "relational_execution"
            ),
            "comparison_carrier_to_relational_carrier_magnitude_ratio": magnitude_ratio(
                model, "comparison_carrier", "relational_carrier"
            ),
            "physical_map": physical_map(model),
        }
        execution_gates[model_name] = execution_passed
        control_gates[model_name] = control_passed
        cross_language_gates[model_name] = cross_language_passed

    formal_models = behavior["passing_models"]
    p4_models = [
        model for model in formal_models
        if ledgers[model]["representation"]["passing_record_count"] >= 16
    ]
    p5_models = [model for model in formal_models if control_gates[model]]
    p6_models = [model for model in formal_models if execution_gates[model]]
    p7_models = [model for model in formal_models if cross_language_gates[model]]
    profiles = cross_model_profiles(models)
    gates = {
        "P1": {
            "passed": bool(protocol_audit["all_checks_passed"]),
            "criterion": prereg["prospective_predictions"]["P1"],
        },
        "P2": {
            "passed": bool(behavior["hidden_scan_authorized"]),
            "passing_models": behavior["passing_models"],
            "criterion": prereg["prospective_predictions"]["P2"],
        },
        "P3": {
            "passed": hidden["passed"],
            "passing_models": hidden["passing_formal_models"],
            "criterion": prereg["prospective_predictions"]["P3"],
        },
        "P4": {
            "passed": len(p4_models) >= 2,
            "passing_models": p4_models,
            "criterion": prereg["prospective_predictions"]["P4"],
        },
        "P5": {
            "passed": len(p5_models) >= 2,
            "passing_models": p5_models,
            "criterion": prereg["prospective_predictions"]["P5"],
        },
        "P6": {
            "passed": len(p6_models) >= 2,
            "passing_models": p6_models,
            "criterion": prereg["prospective_predictions"]["P6"],
        },
        "P7": {
            "passed": len(p7_models) >= 2,
            "passing_models": p7_models,
            "criterion": prereg["prospective_predictions"]["P7"],
        },
        "P8": {
            "passed": profiles["formal_pair_passed"],
            "criterion": prereg["prospective_predictions"]["P8"],
        },
    }
    gates["P9"] = {
        "passed": all(gates[key]["passed"] for key in ("P6", "P7", "P8")),
        "criterion": prereg["prospective_predictions"]["P9"],
    }
    independent_replication = gates["P6"]["passed"] and gates["P7"]["passed"]
    summary = {
        "schema_version": "phase1096_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_authorization_digest": behavior["authorization_digest"],
        "gates": gates,
        "hidden_audit": hidden,
        "ledgers": ledgers,
        "cross_model_functional_profiles": profiles,
        "decision": (
            "comparison_execution_primitive_candidate_requires_independent_replication"
            if independent_replication
            else "representation_control_execution_separated_no_predictive_comparison_primitive"
        ),
        "automatic_next_required": independent_replication,
        "automatic_next_route": (
            "independent_comparison_primitive_replication"
            if independent_replication else None
        ),
        "causal_authorized": bool(gates["P9"]["passed"]),
        "theory_status": (
            "No new mathematical theory is licensed. The contrasts are measurement definitions; only repeated predictive transitions may later motivate a dynamical law."
        ),
        "interpretation_limits": prereg["interpretation_limits"],
    }
    summary["summary_digest"] = protocol.digest(summary)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", summary)
    print({
        "phase": protocol.PHASE,
        "gates": {key: value["passed"] for key, value in gates.items()},
        "decision": summary["decision"],
        "automatic_next_required": summary["automatic_next_required"],
        "causal_authorized": summary["causal_authorized"],
        "summary_digest": summary["summary_digest"],
    })


if __name__ == "__main__":
    main()
