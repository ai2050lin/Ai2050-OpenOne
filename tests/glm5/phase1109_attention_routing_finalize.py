#!/usr/bin/env python3
"""Freeze Phase1109 attention-routing decisions and the causal gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import phase1109_attention_routing_protocol as protocol


EPSILON = 1e-12


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= EPSILON:
        return None
    value = float(np.dot(left, right) / denominator)
    return value if math.isfinite(value) else None


def fields(values: np.ndarray) -> dict[str, np.ndarray]:
    # [unit, regime, route, congruence, layer, head, role]
    exact_conflict = values[:, :, 0, 0]
    exact_congruent = values[:, :, 0, 1]
    ordinal_conflict = values[:, :, 1, 0]
    ordinal_congruent = values[:, :, 1, 1]
    lexical_conflict = exact_conflict - ordinal_conflict
    lexical_congruent = exact_congruent - ordinal_congruent
    return {
        "exact_following": 0.5 * (exact_conflict + exact_congruent),
        "lexical_advantage": 0.5 * (lexical_conflict + lexical_congruent),
        "execution_modulation": lexical_conflict - lexical_congruent,
        "exact_conflict": exact_conflict,
        "exact_congruent": exact_congruent,
        "ordinal_conflict": ordinal_conflict,
        "ordinal_congruent": ordinal_congruent,
    }


def load_model(model: str) -> dict[str, Any]:
    root = protocol.OUT_ROOT / "atlas" / model
    summary = protocol.read_json(root / "summary.json")
    units = protocol.read_json(root / "units.json")
    arrays = np.load(root / "attention_routing_fields.npz")
    return {
        "summary": summary,
        "units": units,
        "key": fields(arrays["key_follow"].astype(np.float64)),
        "key_total": arrays["key_total"].astype(np.float64),
        "record": fields(arrays["record_follow"].astype(np.float64)),
        "record_total": arrays["record_total"].astype(np.float64),
    }


def event_dict(layer: int, head: int, role: int, layer_count: int) -> dict[str, Any]:
    return {
        "layer_index": int(layer),
        "relative_depth": float((layer + 1) / layer_count),
        "head_index": int(head),
        "query_role": protocol.QUERY_ROLES[role],
    }


def select_events(data: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    units = data["units"]
    mask = np.asarray([row["split"] == "qualification" for row in units], dtype=bool)
    key = data["key"]
    exact = np.mean(key["exact_following"][mask], axis=0)
    advantage = np.mean(key["lexical_advantage"][mask], axis=0)
    exact_total = 0.5 * (
        data["key_total"][:, :, 0, 0] + data["key_total"][:, :, 0, 1]
    )
    mass = np.mean(exact_total[mask], axis=0)
    candidates = []
    for layer in range(exact.shape[1]):
        for head in range(exact.shape[2]):
            best = None
            for role in range(exact.shape[3]):
                exact_floor = float(np.min(exact[:, layer, head, role]))
                advantage_floor = float(np.min(advantage[:, layer, head, role]))
                mass_floor = float(np.min(mass[:, layer, head, role]))
                eligible = (
                    exact_floor >= thresholds["minimum_exact_key_following"]
                    and advantage_floor >= thresholds["minimum_lexical_over_ordinal_advantage"]
                    and mass_floor >= thresholds["minimum_total_key_attention_mass"]
                )
                score = min(exact_floor, advantage_floor, mass_floor)
                row = {
                    **event_dict(layer, head, role, data["summary"]["layer_count"]),
                    "qualification_exact_following_floor": exact_floor,
                    "qualification_lexical_advantage_floor": advantage_floor,
                    "qualification_key_mass_floor": mass_floor,
                    "qualification_score": score,
                    "eligible": eligible,
                }
                if best is None or row["qualification_score"] > best["qualification_score"]:
                    best = row
            if best is not None:
                candidates.append(best)
    eligible = sorted(
        (row for row in candidates if row["eligible"]),
        key=lambda row: row["qualification_score"],
        reverse=True,
    )
    selected = eligible[:protocol.MAX_SELECTED_EVENTS]
    top_diagnostic = sorted(
        candidates, key=lambda row: row["qualification_score"], reverse=True
    )[:10]
    return {
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected_events": selected,
        "top_diagnostic_events": top_diagnostic,
        "no_candidate": not selected,
    }


def ensemble_values(array: np.ndarray, events: list[dict[str, Any]]) -> np.ndarray:
    values = [
        array[:, :, row["layer_index"], row["head_index"],
              protocol.QUERY_ROLES.index(row["query_role"])]
        for row in events
    ]
    if not values:
        return np.full(array.shape[:2], np.nan, dtype=np.float64)
    return np.mean(values, axis=0)


def pair_counts(values: np.ndarray, units: list[dict[str, Any]], mask: np.ndarray) -> dict[str, Any]:
    by_pair = {}
    for pair in protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")["relation_pairs"]:
        pair_mask = mask & np.asarray([row["relation_pair"] == pair for row in units], dtype=bool)
        by_pair[pair] = float(np.mean(values[pair_mask]))
    return {
        "means": by_pair,
        "positive_count": sum(value > 0.0 for value in by_pair.values()),
    }


def decide_model(data: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    selection = select_events(data, thresholds)
    units = data["units"]
    confirmation = np.asarray([row["split"] == "confirmation" for row in units], dtype=bool)
    events = selection["selected_events"]
    if not events:
        return {
            "model": data["summary"]["model"],
            "instrument_passed": bool(data["summary"]["all_checks_passed"]),
            "selection": selection,
            "confirmation": {},
            "P4_attention_address_confirmation": False,
            "P5_pair_breadth": False,
            "P6_execution_modulation": False,
        }
    exact = ensemble_values(data["key"]["exact_following"], events)
    advantage = ensemble_values(data["key"]["lexical_advantage"], events)
    modulation = ensemble_values(data["key"]["execution_modulation"], events)
    record_advantage = ensemble_values(data["record"]["lexical_advantage"], events)
    exact_total = 0.5 * (
        data["key_total"][:, :, 0, 0] + data["key_total"][:, :, 0, 1]
    )
    mass = ensemble_values(exact_total, events)
    regimes = {}
    for regime_index, regime in enumerate(protocol.LABEL_REGIMES):
        address_pairs = pair_counts(advantage[:, regime_index], units, confirmation)
        modulation_pairs = pair_counts(modulation[:, regime_index], units, confirmation)
        regimes[regime] = {
            "exact_following_mean": float(np.nanmean(exact[confirmation, regime_index])),
            "lexical_advantage_mean": float(np.nanmean(advantage[confirmation, regime_index])),
            "execution_modulation_mean": float(np.nanmean(modulation[confirmation, regime_index])),
            "key_attention_mass_mean": float(np.nanmean(mass[confirmation, regime_index])),
            "record_lexical_advantage_mean": float(np.nanmean(
                record_advantage[confirmation, regime_index]
            )),
            "address_pair_results": address_pairs,
            "modulation_pair_results": modulation_pairs,
        }
    p4 = bool(events) and all(
        row["exact_following_mean"] >= thresholds["minimum_exact_key_following"]
        and row["lexical_advantage_mean"]
        >= thresholds["minimum_lexical_over_ordinal_advantage"]
        and row["key_attention_mass_mean"]
        >= thresholds["minimum_total_key_attention_mass"]
        for row in regimes.values()
    )
    p5 = p4 and all(
        row["address_pair_results"]["positive_count"]
        >= thresholds["minimum_positive_relation_pairs"]
        for row in regimes.values()
    )
    p6 = p5 and all(
        row["execution_modulation_mean"]
        >= thresholds["minimum_execution_modulation"]
        and row["modulation_pair_results"]["positive_count"]
        >= thresholds["minimum_positive_relation_pairs"]
        for row in regimes.values()
    )
    return {
        "model": data["summary"]["model"],
        "instrument_passed": bool(data["summary"]["all_checks_passed"]),
        "selection": selection,
        "confirmation": regimes,
        "P4_attention_address_confirmation": p4,
        "P5_pair_breadth": p5,
        "P6_execution_modulation": p6,
    }


def normalized_depth_curve(data: dict[str, Any]) -> list[float]:
    units = data["units"]
    confirmation = np.asarray([row["split"] == "confirmation" for row in units], dtype=bool)
    advantage = np.mean(data["key"]["lexical_advantage"][confirmation], axis=(0, 1))
    # [layer, head, role]. Retain the strongest positive address event per depth.
    layer_curve = np.maximum(np.max(advantage, axis=(1, 2)), 0.0)
    source_x = (np.arange(len(layer_curve), dtype=np.float64) + 1) / len(layer_curve)
    target_x = np.linspace(0.0, 1.0, 21)
    curve = np.interp(target_x, source_x, layer_curve, left=layer_curve[0], right=layer_curve[-1])
    maximum = float(np.max(curve))
    if maximum > EPSILON:
        curve = curve / maximum
    return [float(value) for value in curve]


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1109 protocol audit failed")
    thresholds = prereg["thresholds"]
    models = {model: load_model(model) for model in protocol.AUTHORIZED_MODELS}
    decisions = {model: decide_model(data, thresholds) for model, data in models.items()}
    p3 = all(row["instrument_passed"] for row in decisions.values())
    p4 = sum(row["P4_attention_address_confirmation"] for row in decisions.values()) >= thresholds["minimum_models"]
    p5 = sum(row["P5_pair_breadth"] for row in decisions.values()) >= thresholds["minimum_models"]
    p6 = sum(row["P6_execution_modulation"] for row in decisions.values()) >= thresholds["minimum_models"]

    curves = {model: normalized_depth_curve(data) for model, data in models.items()}
    qwen = np.asarray(curves["qwen3"], dtype=np.float64)
    glm = np.asarray(curves["glm4"], dtype=np.float64)
    curve_cosine = cosine(qwen, glm)
    curve_mae = float(np.mean(np.abs(qwen - glm)))
    p7 = bool(
        curve_cosine is not None
        and curve_cosine >= thresholds["minimum_cross_model_curve_cosine"]
        and curve_mae <= thresholds["maximum_cross_model_curve_mae"]
    )
    predictions = {
        "P1": True,
        "P2": True,
        "P3": p3,
        "P4": p4,
        "P5": p5,
        "P6": p6,
        "P7": p7,
    }
    causal_authorized = all(predictions.values())
    predictions["P8"] = causal_authorized
    evidence = {
        "attention_routing_observable": (
            "E2_candidate" if p4 and p5 else "E3_operator_boundary"
        ),
        "content_conditioned_execution": (
            "E2_candidate" if p6 else "E3_not_established"
        ),
        "cross_model_attention_topology": (
            "E2" if p7 else "E1"
        ),
        "causal_edge": "not_added" if not causal_authorized else "eligible_for_separate_preregistration",
    }
    cross_model = {
        "schema_version": "phase1109_cross_model_attention_topology.v1",
        "phase": protocol.PHASE,
        "curves": curves,
        "cosine": curve_cosine,
        "mae": curve_mae,
        "passed": p7,
    }
    cross_model["cross_model_digest"] = protocol.digest(cross_model)
    analysis_root = protocol.OUT_ROOT / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    protocol.write_json(analysis_root / "model_decisions.json", decisions)
    protocol.write_json(analysis_root / "cross_model_topology.json", cross_model)
    final = {
        "schema_version": "phase1109_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "source_behavior_authorization_digest": prereg["source"]["behavior_authorization_digest"],
        "authorized_models": list(protocol.AUTHORIZED_MODELS),
        "denied_models": list(protocol.DENIED_MODELS),
        "model_summary_digests": {
            model: data["summary"]["summary_digest"] for model, data in models.items()
        },
        "model_decisions": decisions,
        "cross_model_topology": cross_model,
        "prospective_predictions": predictions,
        "evidence": evidence,
        "causal_staircase_authorized": causal_authorized,
        "component_head_qkv_neuron_localization_authorized": causal_authorized,
        "automatic_next_required": causal_authorized,
        "automatic_next_decision": (
            "A separate causal preregistration is required before intervention."
            if causal_authorized else
            "Stop before head/QKV/neuron intervention; at least one frozen attention-specificity gate failed."
        ),
        "frozen_conclusion": (
            "Phase1109 measures attention routing weights as a new descriptive observable. "
            "Exact token matching, content-conditioned execution modulation, and causal use "
            "remain separate claims and are decided by P4-P8."
        ),
        "canonical_theory_name_unchanged": "conditional output-field closure theory",
    }
    final["final_summary_digest"] = protocol.digest(final)
    protocol.write_json(analysis_root / "final_summary.json", final)
    print(json.dumps({
        "phase": protocol.PHASE,
        "prospective_predictions": predictions,
        "cross_model_cosine": curve_cosine,
        "cross_model_mae": curve_mae,
        "causal_staircase_authorized": causal_authorized,
        "automatic_next_required": causal_authorized,
        "final_summary_digest": final["final_summary_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
