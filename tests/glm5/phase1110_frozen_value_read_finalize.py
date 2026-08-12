#!/usr/bin/env python3
"""Freeze Phase1110 key/body value-read decisions and claim boundary."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1108_exact_key_event_protocol as source
import phase1110_frozen_value_read_protocol as protocol


EPSILON = 1e-12


def normalized_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(left - right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) + np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > EPSILON,
    )


def state_index() -> dict[tuple[str, str, str, int, int, int], int]:
    return {
        source.state_factors(state): index
        for index, state in enumerate(source.STATES)
    }


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_model(model: str, prereg: dict[str, Any]) -> dict[str, Any]:
    atlas_root = protocol.OUT_ROOT / "atlas" / model
    summary = protocol.read_json(atlas_root / "summary.json")
    units = protocol.read_json(atlas_root / "units.json")
    with np.load(atlas_root / "frozen_value_read_fields.npz") as data:
        mass = np.asarray(data["attention_mass"], dtype=np.float32)
        av = np.asarray(data["av_vectors"], dtype=np.float32)
        raw = np.asarray(data["raw_value_means"], dtype=np.float32)
        readout = np.asarray(data["readout_alignment"], dtype=np.float32)
        reconstruction = np.asarray(data["reconstruction_relative_error"], dtype=np.float32)
    if mass.shape[:3] != (48, 64, 4):
        raise RuntimeError(f"{model} field shape drift: {mass.shape}")
    index = state_index()
    source_slot = {name: slot for slot, name in enumerate(protocol.SOURCE_NAMES)}
    raw_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_body: dict[tuple[str, str], list[float]] = defaultdict(list)
    av_distance: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    readout_advantage: dict[tuple[str, str], list[float]] = defaultdict(list)
    target_mass: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    target_av_norm: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for unit_index, unit in enumerate(units):
        pair = str(unit["relation_pair"])
        for regime in source.LABEL_REGIMES:
            # Relation1 is the only payload whose winner changes across the
            # matched conflict/congruent panels. Its key precedes that payload.
            for route in source.ROUTE_TYPES:
                for target in source.TARGET_RELATIONS:
                    for order in source.RELATION_ORDERS:
                        for orientation in source.ORIENTATIONS:
                            conflict = index[(regime, route, "conflict", target, order, orientation)]
                            congruent = index[(regime, route, "congruent", target, order, orientation)]
                            key_distance = normalized_distance(
                                raw[unit_index, conflict, :, source_slot["key1"], :],
                                raw[unit_index, congruent, :, source_slot["key1"], :],
                            )
                            body_distance = normalized_distance(
                                raw[unit_index, conflict, :, source_slot["body1"], :],
                                raw[unit_index, congruent, :, source_slot["body1"], :],
                            )
                            body_av_distance = normalized_distance(
                                av[unit_index, conflict, :, source_slot["body1"], :],
                                av[unit_index, congruent, :, source_slot["body1"], :],
                            )
                            raw_key[(regime, pair)].extend(key_distance.tolist())
                            raw_body[(regime, pair)].extend(body_distance.tolist())
                            av_distance[(regime, pair, route, target)].extend(body_av_distance.tolist())
            for route in source.ROUTE_TYPES:
                for congruence in source.CONGRUENCES:
                    for target in source.TARGET_RELATIONS:
                        for order in source.RELATION_ORDERS:
                            for orientation in source.ORIENTATIONS:
                                slot = index[(regime, route, congruence, target, order, orientation)]
                                target_key = source_slot[f"key{target}"]
                                target_body = source_slot[f"body{target}"]
                                body_readout = readout[unit_index, slot, :, target_body]
                                key_readout = readout[unit_index, slot, :, target_key]
                                readout_advantage[(regime, pair)].extend((body_readout - key_readout).tolist())
                                target_mass[(regime, route, "key")].extend(mass[unit_index, slot, :, target_key].tolist())
                                target_mass[(regime, route, "body")].extend(mass[unit_index, slot, :, target_body].tolist())
                                target_av_norm[(regime, route, "key")].extend(
                                    np.linalg.norm(av[unit_index, slot, :, target_key, :], axis=-1).tolist()
                                )
                                target_av_norm[(regime, route, "body")].extend(
                                    np.linalg.norm(av[unit_index, slot, :, target_body, :], axis=-1).tolist()
                                )

    thresholds = prereg["thresholds"]
    regimes = {}
    for regime in source.LABEL_REGIMES:
        per_pair = {}
        for pair in source.RELATION_PAIRS:
            key_distance = finite_mean(raw_key[(regime, pair)])
            body_distance = finite_mean(raw_body[(regime, pair)])
            selected = {
                route: finite_mean(av_distance[(regime, pair, route, 1)])
                for route in source.ROUTE_TYPES
            }
            distractor = {
                route: finite_mean(av_distance[(regime, pair, route, 0)])
                for route in source.ROUTE_TYPES
            }
            selection = {
                route: selected[route] - distractor[route]
                for route in source.ROUTE_TYPES
            }
            per_pair[pair] = {
                "key_value_matched_distance": key_distance,
                "body_value_matched_distance": body_distance,
                "body_over_key_distance_advantage": body_distance - key_distance,
                "selected_body_av_distance": selected,
                "distractor_body_av_distance": distractor,
                "body_selection_advantage": selection,
                "exact_over_ordinal_selection_advantage": selection["exact"] - selection["ordinal"],
                "body_over_key_readout_alignment": finite_mean(readout_advantage[(regime, pair)]),
            }
        aggregate = {
            "key_value_matched_distance": finite_mean([row["key_value_matched_distance"] for row in per_pair.values()]),
            "maximum_pair_key_value_matched_distance": max(row["key_value_matched_distance"] for row in per_pair.values()),
            "body_value_matched_distance": finite_mean([row["body_value_matched_distance"] for row in per_pair.values()]),
            "body_over_key_distance_advantage": finite_mean([row["body_over_key_distance_advantage"] for row in per_pair.values()]),
            "exact_body_selection_advantage": finite_mean([row["body_selection_advantage"]["exact"] for row in per_pair.values()]),
            "ordinal_body_selection_advantage": finite_mean([row["body_selection_advantage"]["ordinal"] for row in per_pair.values()]),
            "exact_over_ordinal_selection_advantage": finite_mean([row["exact_over_ordinal_selection_advantage"] for row in per_pair.values()]),
            "body_over_key_readout_alignment": finite_mean([row["body_over_key_readout_alignment"] for row in per_pair.values()]),
            "target_attention_mass": {
                route: {kind: finite_mean(target_mass[(regime, route, kind)]) for kind in ("key", "body")}
                for route in source.ROUTE_TYPES
            },
            "target_av_norm": {
                route: {kind: finite_mean(target_av_norm[(regime, route, kind)]) for kind in ("key", "body")}
                for route in source.ROUTE_TYPES
            },
        }
        positive_counts = {
            "body_payload": sum(
                row["body_value_matched_distance"] >= thresholds["minimum_body_value_matched_distance"]
                and row["key_value_matched_distance"] <= thresholds["maximum_key_value_matched_distance"]
                and row["body_over_key_distance_advantage"] >= thresholds["minimum_body_over_key_distance_advantage"]
                for row in per_pair.values()
            ),
            "selected_body": sum(
                row["body_selection_advantage"]["exact"] >= thresholds["minimum_selected_body_av_distance_advantage"]
                for row in per_pair.values()
            ),
            "exact_over_ordinal": sum(
                row["exact_over_ordinal_selection_advantage"] >= thresholds["minimum_exact_over_ordinal_selection_advantage"]
                for row in per_pair.values()
            ),
            "readout_alignment": sum(
                row["body_over_key_readout_alignment"] >= thresholds["minimum_body_over_key_readout_alignment"]
                for row in per_pair.values()
            ),
        }
        gates = {
            "key_causal_invariance": aggregate["maximum_pair_key_value_matched_distance"] <= thresholds["maximum_key_value_matched_distance"],
            "body_payload_changes": aggregate["body_value_matched_distance"] >= thresholds["minimum_body_value_matched_distance"],
            "body_over_key": aggregate["body_over_key_distance_advantage"] >= thresholds["minimum_body_over_key_distance_advantage"],
            "body_payload_pair_breadth": positive_counts["body_payload"] >= thresholds["minimum_positive_relation_pairs"],
            "selected_body_av": aggregate["exact_body_selection_advantage"] >= thresholds["minimum_selected_body_av_distance_advantage"],
            "selected_body_pair_breadth": positive_counts["selected_body"] >= thresholds["minimum_positive_relation_pairs"],
            "exact_over_ordinal": aggregate["exact_over_ordinal_selection_advantage"] >= thresholds["minimum_exact_over_ordinal_selection_advantage"],
            "exact_over_ordinal_pair_breadth": positive_counts["exact_over_ordinal"] >= thresholds["minimum_positive_relation_pairs"],
            "body_readout_alignment": aggregate["body_over_key_readout_alignment"] >= thresholds["minimum_body_over_key_readout_alignment"],
            "readout_pair_breadth": positive_counts["readout_alignment"] >= thresholds["minimum_positive_relation_pairs"],
        }
        regimes[regime] = {
            "per_pair": per_pair,
            "aggregate": aggregate,
            "positive_pair_counts": positive_counts,
            "gates": gates,
        }

    model_gates = {
        "instrument": bool(summary["all_checks_passed"]),
        "reconstruction": float(np.max(reconstruction)) <= thresholds["maximum_head_reconstruction_relative_error"],
        "P4_key_body_causal_order": all(
            row["gates"][key]
            for row in regimes.values()
            for key in ("key_causal_invariance", "body_payload_changes", "body_over_key", "body_payload_pair_breadth")
        ),
        "P5_selected_body_av": all(
            row["gates"][key]
            for row in regimes.values()
            for key in ("selected_body_av", "selected_body_pair_breadth")
        ),
        "P6_exact_over_ordinal": all(
            row["gates"][key]
            for row in regimes.values()
            for key in ("exact_over_ordinal", "exact_over_ordinal_pair_breadth")
        ),
        "P7_direct_readout": all(
            row["gates"][key]
            for row in regimes.values()
            for key in ("body_readout_alignment", "readout_pair_breadth")
        ),
    }
    return {
        "model": model,
        "model_summary_digest": summary["summary_digest"],
        "maximum_head_reconstruction_relative_error": float(np.max(reconstruction)),
        "regimes": regimes,
        "model_gates": model_gates,
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1110 protocol audit failed")
    decisions = {
        model: summarize_model(model, prereg)
        for model in protocol.AUTHORIZED_MODELS
    }
    thresholds = prereg["thresholds"]
    predictions = {
        "P1": bool(audit["all_checks_passed"]),
        "P2": all(row["model_gates"]["instrument"] for row in decisions.values())
        and (protocol.OUT_ROOT / "atlas" / "deepseek7b" / "denial.json").exists(),
        "P3": all(row["model_gates"]["reconstruction"] for row in decisions.values()),
        "P4": sum(row["model_gates"]["P4_key_body_causal_order"] for row in decisions.values()) >= thresholds["minimum_models"],
        "P5": sum(row["model_gates"]["P5_selected_body_av"] for row in decisions.values()) >= thresholds["minimum_models"],
        "P6": sum(row["model_gates"]["P6_exact_over_ordinal"] for row in decisions.values()) >= thresholds["minimum_models"],
        "P7": sum(row["model_gates"]["P7_direct_readout"] for row in decisions.values()) >= thresholds["minimum_models"],
        "P8": False,
    }
    analysis_root = protocol.OUT_ROOT / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    protocol.write_json(analysis_root / "model_value_decisions.json", decisions)
    final = {
        "schema_version": "phase1110_frozen_value_read_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "authorized_models": list(protocol.AUTHORIZED_MODELS),
        "denied_models": list(protocol.DENIED_MODELS),
        "model_summary_digests": {
            model: row["model_summary_digest"] for model, row in decisions.items()
        },
        "prospective_predictions": predictions,
        "evidence": {
            "frozen_key_body_value_decomposition": "E2_candidate" if predictions["P4"] else "E1",
            "selective_body_read_path": "E2_candidate" if predictions["P5"] and predictions["P6"] else "E1",
            "direct_answer_compatibility": "E2_candidate" if predictions["P7"] else "E1",
            "content_conditioned_execution": "E3_not_established",
            "semantic_addressing": "not_tested",
            "causal_edge": "not_added",
        },
        "causal_staircase_authorized": False,
        "component_head_qkv_neuron_localization_authorized": False,
        "automatic_next_required": False,
        "automatic_next_decision": (
            "No automatic model run. Phase1110 resolves the key/body descriptive ambiguity, "
            "but inherited Phase1109 execution and cross-model topology gates still fail. "
            "A next phase must change the evidence axis rather than deepen the same exact-key registry."
        ),
        "frozen_conclusion": (
            "The frozen exact-key heads were tested for physical key/body value transport on an "
            "independent confirmation split. Any positive P4-P7 result remains a descriptive read-path "
            "observation; it is not semantic addressing, behavioral necessity, or causal closure."
        ),
        "canonical_theory_name_unchanged": "条件化输出场闭合理论",
    }
    final["final_summary_digest"] = protocol.digest(final)
    protocol.write_json(analysis_root / "final_summary.json", final)
    print(json.dumps({
        "phase": protocol.PHASE,
        "prospective_predictions": predictions,
        "model_gates": {model: row["model_gates"] for model, row in decisions.items()},
        "automatic_next_required": final["automatic_next_required"],
        "final_summary_digest": final["final_summary_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
