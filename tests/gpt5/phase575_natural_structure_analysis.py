#!/usr/bin/env python3
"""Find repeated Phase575 structures before defining a causal mechanism."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase575_source_competition"
PROTOCOL_PATH = OUT_DIR / "phase575_natural_ledger_protocol.json"
SUMMARY_PATH = OUT_DIR / "phase575_qwen3_natural_ledger_summary.json"
ROWS_PATH = OUT_DIR / "phase575_qwen3_natural_ledger_rows.jsonl.gz"
ANALYSIS_PATH = OUT_DIR / "phase575_qwen3_natural_structure_analysis.json"
DECISION_PATH = OUT_DIR / "phase575_natural_structure_decision.json"

SPLITS = (
    "structure_discovery",
    "structure_confirmation",
    "heldout_recombination",
)
RECEIVERS = ("query_terminal", "answer_boundary")
VARIANTS = (
    "base",
    "relation_swap",
    "object_swap",
    "relation_object_swap",
    "order_swap",
)
CHANNELS = {
    "score": ("semantic_score_margin", "anchor_score_margin"),
    "weight": ("semantic_weight_margin", "anchor_weight_margin"),
    "message_norm": (
        "semantic_message_norm_margin",
        "anchor_message_norm_margin",
    ),
}
PAIR_TYPES = ("relation", "object", "relation_object", "order")
VECTOR_COMPONENTS = (
    "post_rotary_query_relative_delta",
    "attention_output_relative_delta",
    "layer_input_state_relative_delta",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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


def sign_preserved(left: float, right: float) -> bool:
    return (left > 0.0 and right > 0.0) or (left < 0.0 and right < 0.0)


def compact_band(layers: list[int]) -> list[dict[str, int]]:
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


def summarize_coordinate(rows: list[dict[str, Any]], receiver: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for channel, (semantic_field, anchor_field) in CHANNELS.items():
        semantic_values: list[float] = []
        semantic_all_positive: list[bool] = []
        order_positive: list[bool] = []
        relation_direction: list[bool] = []
        relation_flip: list[bool] = []
        object_anchor_preservation: list[bool] = []
        effects: list[float] = []
        for row in rows:
            values = {
                variant: float(row["variants"][variant][receiver][semantic_field])
                for variant in VARIANTS
            }
            anchors = {
                variant: float(row["variants"][variant][receiver][anchor_field])
                for variant in VARIANTS
            }
            semantic_values.extend(values.values())
            semantic_all_positive.append(all(value > 0.0 for value in values.values()))
            order_positive.append(values["order_swap"] > 0.0)
            first_effect = anchors["base"] - anchors["relation_swap"]
            second_effect = (
                anchors["object_swap"] - anchors["relation_object_swap"]
            )
            relation_direction.append(first_effect > 0.0 and second_effect > 0.0)
            relation_flip.append(
                anchors["base"] > 0.0
                and anchors["relation_swap"] < 0.0
                and anchors["object_swap"] > 0.0
                and anchors["relation_object_swap"] < 0.0
            )
            object_anchor_preservation.append(
                sign_preserved(anchors["base"], anchors["object_swap"])
                and sign_preserved(
                    anchors["relation_swap"],
                    anchors["relation_object_swap"],
                )
            )
            effects.append((first_effect + second_effect) / 2.0)
        result[channel] = {
            "semantic_positive_observation_rate": rate(
                [value > 0.0 for value in semantic_values]
            ),
            "semantic_all_five_variants_world_rate": rate(semantic_all_positive),
            "order_semantic_positive_world_rate": rate(order_positive),
            "relation_anchor_direction_world_rate": rate(relation_direction),
            "relation_anchor_strict_flip_world_rate": rate(relation_flip),
            "object_anchor_sign_preservation_world_rate": rate(
                object_anchor_preservation
            ),
            "relation_anchor_effect_mean": mean(effects),
            "relation_anchor_effect_median": median(effects),
        }

    vector_result: dict[str, Any] = {}
    for pair_type in PAIR_TYPES:
        vector_result[pair_type] = {}
        for component in VECTOR_COMPONENTS:
            values = [
                float(
                    row["pair_vector_deltas_if_snapshotted"][pair_type][receiver][
                        component
                    ]
                )
                for row in rows
            ]
            vector_result[pair_type][component] = {
                "mean": mean(values),
                "median": median(values),
                "positive_rate": rate([value > 0.0 for value in values]),
                "maximum": max(values) if values else 0.0,
            }
    result["natural_vector_deltas"] = vector_result

    source_invariance: dict[str, Any] = {}
    for source in ("anchor_base_selected", "anchor_base_other_relation"):
        source_invariance[source] = {}
        for field in (
            "source_post_rotary_key_norm",
            "source_value_norm",
        ):
            deltas = [
                abs(
                    float(
                        row["variants"]["base"][receiver]["sources"][source][field]
                    )
                    - float(
                        row["variants"]["relation_swap"][receiver]["sources"][
                            source
                        ][field]
                    )
                )
                for row in rows
            ]
            source_invariance[source][field] = {
                "mean_absolute_delta": mean(deltas),
                "maximum_absolute_delta": max(deltas) if deltas else 0.0,
                "exact_equal_rate": rate([value == 0.0 for value in deltas]),
            }
    result["fixed_source_relation_invariance"] = source_invariance
    return result


def analyze() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    summary = read_json(SUMMARY_PATH)
    if protocol["authorized_models"] != ["qwen3"]:
        raise RuntimeError("Phase575 authorization changed after protocol freeze")
    if not summary["natural_structure_analysis_authorized"]:
        raise RuntimeError("Phase575 natural ledger did not pass quality gates")
    if summary["rows_sha256"] != sha256_file(ROWS_PATH):
        raise RuntimeError("Phase575 natural ledger hash mismatch")
    if summary["causal_splits_read"] or summary["sealed_split_read"]:
        raise RuntimeError("Phase575 natural ledger crossed a frozen evidence boundary")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    row_count = 0
    for row in read_jsonl_gz(ROWS_PATH):
        if row["sealed"]:
            raise RuntimeError("sealed row found in Phase575 natural ledger")
        if row["split"] not in SPLITS:
            raise RuntimeError(f"unexpected Phase575 split: {row['split']}")
        grouped[(row["split"], int(row["layer"]))].append(row)
        row_count += 1
    if row_count != summary["ledger_row_count"]:
        raise RuntimeError("Phase575 natural ledger row count mismatch")

    threshold = float(
        protocol["natural_event_discovery"][
            "minimum_world_direction_rate_each_split"
        ]
    )
    duplicate_floor = float(summary["duplicate_trace_max_abs_delta"])
    minimum_effect = max(1e-12, duplicate_floor * 10.0)
    coordinate_rows: list[dict[str, Any]] = []
    by_coordinate: dict[tuple[int, str], dict[str, Any]] = {}
    for layer in range(int(summary["layer_count"])):
        for receiver in RECEIVERS:
            split_metrics = {
                split: summarize_coordinate(grouped[(split, layer)], receiver)
                for split in SPLITS
            }
            replicated_channels: dict[str, Any] = {}
            for channel in CHANNELS:
                semantic_floor = min(
                    split_metrics[split][channel][
                        "semantic_positive_observation_rate"
                    ]
                    for split in SPLITS
                )
                order_floor = min(
                    split_metrics[split][channel][
                        "order_semantic_positive_world_rate"
                    ]
                    for split in SPLITS
                )
                direction_floor = min(
                    split_metrics[split][channel][
                        "relation_anchor_direction_world_rate"
                    ]
                    for split in SPLITS
                )
                effect_floor = min(
                    split_metrics[split][channel]["relation_anchor_effect_mean"]
                    for split in SPLITS
                )
                strict_flip_floor = min(
                    split_metrics[split][channel][
                        "relation_anchor_strict_flip_world_rate"
                    ]
                    for split in SPLITS
                )
                gate_score = min(semantic_floor, order_floor, direction_floor)
                replicated_channels[channel] = {
                    "semantic_rate_floor": semantic_floor,
                    "order_rate_floor": order_floor,
                    "direction_rate_floor": direction_floor,
                    "strict_flip_rate_floor": strict_flip_floor,
                    "effect_mean_floor": effect_floor,
                    "gate_score": gate_score,
                    "replicated_routing_event": (
                        semantic_floor >= threshold
                        and order_floor >= threshold
                        and direction_floor >= threshold
                        and effect_floor > minimum_effect
                    ),
                }
            coordinate = {
                "layer": layer,
                "normalized_depth": layer / max(1, int(summary["layer_count"]) - 1),
                "receiver": receiver,
                "world_count_each_split": len(grouped[(SPLITS[0], layer)]),
                "split_metrics": split_metrics,
                "replicated_channels": replicated_channels,
            }
            coordinate_rows.append(coordinate)
            by_coordinate[(layer, receiver)] = coordinate

    ranked: list[dict[str, Any]] = []
    bands: dict[str, dict[str, list[dict[str, int]]]] = {}
    for channel in CHANNELS:
        bands[channel] = {}
        for receiver in RECEIVERS:
            layers = []
            for layer in range(int(summary["layer_count"])):
                gate = by_coordinate[(layer, receiver)]["replicated_channels"][channel]
                ranked.append(
                    {
                        "channel": channel,
                        "receiver": receiver,
                        "layer": layer,
                        **gate,
                    }
                )
                if gate["replicated_routing_event"]:
                    layers.append(layer)
            bands[channel][receiver] = compact_band(layers)
    ranked.sort(
        key=lambda item: (
            item["replicated_routing_event"],
            item["gate_score"],
            item["effect_mean_floor"],
            -item["layer"],
        ),
        reverse=True,
    )

    q_relation_onsets: dict[str, Any] = {}
    for receiver in RECEIVERS:
        qualifying_layers = []
        layer_rows = []
        for layer in range(int(summary["layer_count"])):
            split_values = [
                by_coordinate[(layer, receiver)]["split_metrics"][split][
                    "natural_vector_deltas"
                ]["relation"]["post_rotary_query_relative_delta"]["mean"]
                for split in SPLITS
            ]
            floor = min(split_values)
            qualifies = floor > minimum_effect
            if qualifies:
                qualifying_layers.append(layer)
            layer_rows.append(
                {
                    "layer": layer,
                    "mean_by_split": dict(zip(SPLITS, split_values, strict=True)),
                    "mean_floor": floor,
                    "above_duplicate_floor": qualifies,
                }
            )
        q_relation_onsets[receiver] = {
            "first_repeated_nonzero_layer": (
                min(qualifying_layers) if qualifying_layers else None
            ),
            "bands": compact_band(qualifying_layers),
            "layers": layer_rows,
        }

    analysis = {
        "schema_version": "phase575_natural_structure_analysis.v1",
        "phase_id": "Phase575",
        "created_at": now(),
        "status": "complete",
        "model": "qwen3",
        "analysis_principle": (
            "discover stable repeated natural structures before defining theory or "
            "causal mechanism"
        ),
        "world_count": summary["world_count"],
        "ledger_row_count": row_count,
        "split_count": len(SPLITS),
        "layer_count": summary["layer_count"],
        "receiver_count": len(RECEIVERS),
        "replication_threshold": threshold,
        "duplicate_trace_floor": duplicate_floor,
        "minimum_effect_above_floor": minimum_effect,
        "coordinate_count": len(coordinate_rows) * len(CHANNELS),
        "coordinate_rows": coordinate_rows,
        "replicated_event_bands": bands,
        "ranked_coordinates": ranked,
        "query_relation_natural_onsets": q_relation_onsets,
        "quality_gates": summary["quality_gates"],
        "output_embedding_direction_used": False,
        "causal_intervention_executed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
        "rows_sha256": sha256_file(ROWS_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }

    replicated = [item for item in ranked if item["replicated_routing_event"]]
    first_by_channel_receiver = {}
    for item in replicated:
        key = f"{item['channel']}__{item['receiver']}"
        current = first_by_channel_receiver.get(key)
        if current is None or item["layer"] < current["layer"]:
            first_by_channel_receiver[key] = item
    decision = {
        "schema_version": "phase575_natural_structure_decision.v1",
        "phase_id": "Phase575",
        "created_at": now(),
        "status": "complete",
        "model": "qwen3",
        "replicated_routing_coordinate_count": len(replicated),
        "replicated_routing_bands": bands,
        "earliest_replicated_coordinate_by_channel_and_receiver": (
            first_by_channel_receiver
        ),
        "natural_structure_found": bool(replicated),
        "causal_protocol_authorized": bool(replicated),
        "causal_candidate_not_yet_a_mechanism": True,
        "selection_rule": (
            "freeze only coordinates that independently pass semantic selection, "
            "order preservation, relation-direction, and duplicate-floor gates in "
            "all three open structure splits"
        ),
        "theory_formula_frozen_before_discovery": False,
        "causal_intervention_executed": False,
        "causal_splits_read": False,
        "sealed_split_read": False,
        "analysis_sha256": None,
    }
    write_json(ANALYSIS_PATH, analysis)
    decision["analysis_sha256"] = sha256_file(ANALYSIS_PATH)
    write_json(DECISION_PATH, decision)
    return analysis, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    analysis, decision = analyze()
    print(
        json.dumps(
            {
                "model": analysis["model"],
                "world_count": analysis["world_count"],
                "ledger_row_count": analysis["ledger_row_count"],
                "replicated_event_bands": analysis["replicated_event_bands"],
                "query_relation_first_nonzero_layer": {
                    receiver: payload["first_repeated_nonzero_layer"]
                    for receiver, payload in analysis[
                        "query_relation_natural_onsets"
                    ].items()
                },
                "replicated_routing_coordinate_count": decision[
                    "replicated_routing_coordinate_count"
                ],
                "causal_protocol_authorized": decision[
                    "causal_protocol_authorized"
                ],
                "causal_splits_read": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
