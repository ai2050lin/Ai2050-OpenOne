#!/usr/bin/env python3
"""Discover repeated Phase578 natural structures without assuming a mechanism."""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase578_choice_world"
PROTOCOL_PATH = OUT_DIR / "phase578_natural_trace_protocol.json"
ANALYSIS_PATH = OUT_DIR / "phase578_natural_structure_analysis.json"
DECISION_PATH = OUT_DIR / "phase578_natural_structure_decision.json"

SPLITS = ("causal_discovery", "causal_confirmation")
RELATIONS = ("category", "outer_color")
VARIANTS = ("target_first", "target_second")
PHYSICAL_CHANNELS = (
    "option_score_margin",
    "option_weight_margin",
    "option_message_norm_margin",
)
OBSERVER_CHANNELS = (
    "candidate_input_logit_margin",
    "candidate_output_logit_margin",
)
CHANNELS = PHYSICAL_CHANNELS + OBSERVER_CHANNELS
SOURCES = ("object", "relation", "target_option", "foil_option")
SOURCE_FIELDS = ("score_mean", "weight_mass_mean", "message_norm")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def compact_bands(layers: list[int]) -> list[dict[str, int]]:
    if not layers:
        return []
    ordered = sorted(set(layers))
    bands: list[dict[str, int]] = []
    start = previous = ordered[0]
    for layer in ordered[1:]:
        if layer != previous + 1:
            bands.append({"start": start, "end": previous})
            start = layer
        previous = layer
    bands.append({"start": start, "end": previous})
    return bands


def summarize_channel(rows: list[dict[str, Any]], channel: str) -> dict[str, Any]:
    first_values: list[float] = []
    second_values: list[float] = []
    both_target: list[bool] = []
    same_sign: list[bool] = []
    for row in rows:
        first = float(row["variants"]["target_first"][channel])
        second = float(row["variants"]["target_second"][channel])
        first_values.append(first)
        second_values.append(second)
        both_target.append(first > 0.0 and second > 0.0)
        same_sign.append((first > 0.0) == (second > 0.0))
    values = first_values + second_values
    return {
        "world_count": len(rows),
        "observation_count": len(values),
        "target_direction_observation_rate": rate([value > 0.0 for value in values]),
        "target_direction_both_orders_world_rate": rate(both_target),
        "same_sign_across_orders_world_rate": rate(same_sign),
        "margin_mean": mean(values),
        "margin_median": median(values),
        "target_first_margin_mean": mean(first_values),
        "target_second_margin_mean": mean(second_values),
        "order_absolute_delta_mean": mean(
            [abs(left - right) for left, right in zip(first_values, second_values, strict=True)]
        ),
    }


def summarize_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in SOURCES:
        result[source] = {}
        for field in SOURCE_FIELDS:
            first_values = [
                float(row["variants"]["target_first"]["sources"][source][field])
                for row in rows
            ]
            second_values = [
                float(row["variants"]["target_second"]["sources"][source][field])
                for row in rows
            ]
            result[source][field] = {
                "mean_across_orders": mean(first_values + second_values),
                "target_first_mean": mean(first_values),
                "target_second_mean": mean(second_values),
                "order_absolute_delta_mean": mean(
                    [
                        abs(left - right)
                        for left, right in zip(first_values, second_values, strict=True)
                    ]
                ),
            }
    return result


def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["replicated_natural_event"],
        item["gate_score"],
        item["margin_mean_floor"],
        -item["layer"],
    )


def analyze_model(
    model: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = OUT_DIR / f"phase578_{model}_natural_trace_summary.json"
    rows_path = OUT_DIR / f"phase578_{model}_natural_trace_rows.jsonl.gz"
    summary = read_json(summary_path)
    if summary["status"] != "complete":
        raise RuntimeError(f"{model} natural trace is incomplete")
    if not summary["natural_structure_analysis_authorized"]:
        raise RuntimeError(f"{model} natural structure analysis is not authorized")
    if summary["rows_sha256"] != sha256_file(rows_path):
        raise RuntimeError(f"{model} natural trace hash mismatch")
    if summary["causal_holdout_internal_state_read"] or summary["sealed_split_read"]:
        raise RuntimeError(f"{model} crossed a frozen evidence boundary")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    for row in read_jsonl_gz(rows_path):
        if row["model"] != model or row["sealed"]:
            raise RuntimeError(f"invalid {model} natural trace row")
        key = (row["split"], row["relation"], int(row["layer"]))
        grouped[key].append(row)
        row_count += 1
    if row_count != int(summary["trace_row_count"]):
        raise RuntimeError(f"{model} natural trace row count mismatch")

    gate = protocol["natural_event_gate"]
    direction_threshold = float(gate["minimum_target_direction_rate_each_split"])
    order_threshold = float(gate["minimum_option_order_preservation_rate_each_split"])
    minimum_world_count = int(gate["minimum_relation_specific_world_count_each_split"])
    duplicate_floor = float(summary["duplicate_trace_max_abs_delta"])
    minimum_effect = max(
        1e-12,
        duplicate_floor * float(gate["effect_must_exceed_duplicate_floor_multiplier"]),
    )
    layer_count = int(summary["layer_count"])

    coordinates: list[dict[str, Any]] = []
    event_layers: dict[str, dict[str, list[int]]] = {
        channel: {relation: [] for relation in RELATIONS} for channel in CHANNELS
    }
    for layer in range(layer_count):
        for relation in RELATIONS:
            split_channel_metrics: dict[str, Any] = {}
            split_source_profiles: dict[str, Any] = {}
            for split in SPLITS:
                rows = grouped[(split, relation, layer)]
                split_channel_metrics[split] = {
                    channel: summarize_channel(rows, channel) for channel in CHANNELS
                }
                split_source_profiles[split] = summarize_sources(rows)
            for channel in CHANNELS:
                metrics = [split_channel_metrics[split][channel] for split in SPLITS]
                direction_floor = min(
                    item["target_direction_observation_rate"] for item in metrics
                )
                order_floor = min(
                    item["target_direction_both_orders_world_rate"] for item in metrics
                )
                same_sign_floor = min(
                    item["same_sign_across_orders_world_rate"] for item in metrics
                )
                margin_floor = min(item["margin_mean"] for item in metrics)
                count_floor = min(item["world_count"] for item in metrics)
                replicated = (
                    count_floor >= minimum_world_count
                    and direction_floor >= direction_threshold
                    and order_floor >= order_threshold
                    and margin_floor > minimum_effect
                )
                if replicated:
                    event_layers[channel][relation].append(layer)
                coordinates.append(
                    {
                        "model": model,
                        "layer": layer,
                        "normalized_depth": layer / max(1, layer_count - 1),
                        "relation": relation,
                        "channel": channel,
                        "channel_class": (
                            "physical" if channel in PHYSICAL_CHANNELS else "observer"
                        ),
                        "world_count_floor": count_floor,
                        "target_direction_rate_floor": direction_floor,
                        "both_orders_target_rate_floor": order_floor,
                        "same_sign_order_rate_floor": same_sign_floor,
                        "margin_mean_floor": margin_floor,
                        "gate_score": min(direction_floor, order_floor),
                        "replicated_natural_event": replicated,
                        "split_metrics": {
                            split: split_channel_metrics[split][channel]
                            for split in SPLITS
                        },
                    }
                )

    ranked = sorted(coordinates, key=rank_key, reverse=True)
    bands = {
        channel: {
            relation: compact_bands(event_layers[channel][relation])
            for relation in RELATIONS
        }
        for channel in CHANNELS
    }

    source_profiles: list[dict[str, Any]] = []
    for layer in range(layer_count):
        for relation in RELATIONS:
            source_profiles.append(
                {
                    "layer": layer,
                    "normalized_depth": layer / max(1, layer_count - 1),
                    "relation": relation,
                    "by_split": {
                        split: summarize_sources(grouped[(split, relation, layer)])
                        for split in SPLITS
                    },
                }
            )

    physical_events = [
        item
        for item in ranked
        if item["channel_class"] == "physical" and item["replicated_natural_event"]
    ]
    observer_events = [
        item
        for item in ranked
        if item["channel_class"] == "observer" and item["replicated_natural_event"]
    ]
    score_weight_intersections: dict[str, list[int]] = {}
    for relation in RELATIONS:
        score_layers = set(event_layers["option_score_margin"][relation])
        weight_layers = set(event_layers["option_weight_margin"][relation])
        score_weight_intersections[relation] = sorted(score_layers & weight_layers)

    candidates: list[dict[str, Any]] = []
    for relation in RELATIONS:
        intersections = score_weight_intersections[relation]
        if not intersections:
            continue
        candidate_rows = [
            item
            for item in coordinates
            if item["relation"] == relation
            and item["layer"] in intersections
            and item["channel"] in ("option_score_margin", "option_weight_margin")
        ]
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in candidate_rows:
            by_layer[int(item["layer"])].append(item)
        ranked_layers = sorted(
            (
                {
                    "model": model,
                    "relation": relation,
                    "layer": layer,
                    "normalized_depth": layer / max(1, layer_count - 1),
                    "joint_gate_score": min(row["gate_score"] for row in rows),
                    "joint_margin_floor": min(row["margin_mean_floor"] for row in rows),
                }
                for layer, rows in by_layer.items()
            ),
            key=lambda item: (
                item["joint_gate_score"],
                item["joint_margin_floor"],
                -item["layer"],
            ),
            reverse=True,
        )
        candidates.append(ranked_layers[0])

    analysis = {
        "schema_version": "phase578_natural_structure_analysis_model.v1",
        "phase_id": "Phase578",
        "created_at": now(),
        "status": "complete",
        "model": model,
        "analysis_principle": (
            "rank stable repeated natural structures before naming a mechanism or "
            "defining a causal operation"
        ),
        "world_count": summary["world_count"],
        "trace_row_count": row_count,
        "layer_count": layer_count,
        "replication_thresholds": {
            "target_direction_rate": direction_threshold,
            "both_option_orders_world_rate": order_threshold,
            "minimum_relation_world_count": minimum_world_count,
            "minimum_effect_above_duplicate_floor": minimum_effect,
        },
        "coordinate_count": len(coordinates),
        "coordinate_rows": coordinates,
        "ranked_coordinates": ranked,
        "replicated_event_bands": bands,
        "source_profiles": source_profiles,
        "replicated_physical_event_count": len(physical_events),
        "replicated_observer_event_count": len(observer_events),
        "score_weight_intersection_layers": score_weight_intersections,
        "quality_gates": summary["quality_gates"],
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
        "rows_sha256": sha256_file(rows_path),
        "summary_sha256": sha256_file(summary_path),
    }
    decision = {
        "model": model,
        "natural_physical_structure_found": bool(physical_events),
        "score_weight_joint_structure_found": bool(candidates),
        "causal_candidate_coordinates": candidates,
        "causal_protocol_authorized": bool(candidates),
        "candidate_not_yet_a_mechanism": True,
        "observer_event_is_not_causal_evidence": True,
    }
    return analysis, decision


def main() -> None:
    protocol = read_json(PROTOCOL_PATH)
    models = list(protocol["authorized_models"])
    if not models:
        raise RuntimeError("Phase578 has no model authorized for natural analysis")

    model_analyses: dict[str, Any] = {}
    model_decisions: dict[str, Any] = {}
    for model in models:
        analysis, decision = analyze_model(model, protocol)
        model_analyses[model] = analysis
        model_decisions[model] = decision

    shared_shape: dict[str, Any] = {}
    for channel in CHANNELS:
        shared_shape[channel] = {}
        for relation in RELATIONS:
            shared_shape[channel][relation] = {
                model: [
                    {
                        **band,
                        "normalized_start": band["start"]
                        / max(1, model_analyses[model]["layer_count"] - 1),
                        "normalized_end": band["end"]
                        / max(1, model_analyses[model]["layer_count"] - 1),
                    }
                    for band in model_analyses[model]["replicated_event_bands"][
                        channel
                    ][relation]
                ]
                for model in models
            }

    combined_analysis = {
        "schema_version": "phase578_natural_structure_analysis.v1",
        "phase_id": "Phase578",
        "created_at": now(),
        "status": "complete",
        "analysis_principle": (
            "discover real stable repeated structures first; theory and formulas remain "
            "downstream of observation and causal validation"
        ),
        "authorized_models": models,
        "excluded_models": ["deepseek7b"],
        "excluded_model_reason": "natural behavior gate failed before internal tracing",
        "model_analyses": model_analyses,
        "normalized_cross_model_event_shape": shared_shape,
        "cross_model_coordinate_identity_assumed": False,
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    write_json(ANALYSIS_PATH, combined_analysis)

    authorized_models = [
        model
        for model, decision in model_decisions.items()
        if decision["causal_protocol_authorized"]
    ]
    combined_decision = {
        "schema_version": "phase578_natural_structure_decision.v1",
        "phase_id": "Phase578",
        "created_at": now(),
        "status": "complete",
        "model_decisions": model_decisions,
        "causal_protocol_authorized_models": authorized_models,
        "causal_protocol_authorized": bool(authorized_models),
        "selection_rule": (
            "for each model and relation, freeze the strongest layer where raw attention "
            "score and normalized attention weight both favor the behaviorally correct "
            "option across both option orders and both open splits"
        ),
        "theory_formula_frozen_before_discovery": False,
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
        "analysis_sha256": sha256_file(ANALYSIS_PATH),
    }
    write_json(DECISION_PATH, combined_decision)

    print(
        json.dumps(
            {
                "model_results": {
                    model: {
                        "physical_events": model_analyses[model][
                            "replicated_physical_event_count"
                        ],
                        "observer_events": model_analyses[model][
                            "replicated_observer_event_count"
                        ],
                        "score_weight_intersections": model_analyses[model][
                            "score_weight_intersection_layers"
                        ],
                        "causal_candidates": model_decisions[model][
                            "causal_candidate_coordinates"
                        ],
                    }
                    for model in models
                },
                "causal_protocol_authorized_models": authorized_models,
                "causal_holdout_internal_state_read": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
