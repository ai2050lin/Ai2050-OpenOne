#!/usr/bin/env python3
"""Finalize frozen Phase1126 behavior gates."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1126_semeval_lexsub_natural_cloze_protocol as protocol


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def meets(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def interaction(table: dict[tuple[str, str], float]) -> float:
    return 0.5 * (
        (table[("a", "a")] - table[("a", "b")])
        + (table[("b", "b")] - table[("b", "a")])
    )


def summarize_partition(units: list[dict[str, Any]], finite_rate: float, thresholds: dict[str, float]) -> dict[str, Any]:
    complete_units = [row for row in units if row["complete"]]
    active = [row["active_z"] for row in complete_units]
    matched = [row["matched_deranged_z"] for row in complete_units]
    lexical = [row["lexical_active_z"] for row in complete_units]
    advantages = [a - abs(n) for a, n in zip(active, matched)]
    active_median = median(active)
    lexical_abs_median = median([abs(value) for value in lexical])
    metrics = {
        "unit_count": len(units),
        "complete_unit_count": len(complete_units),
        "finite_rate": finite_rate,
        "active_mean": mean(active),
        "active_median": active_median,
        "active_positive_rate": sum(value > 0.0 for value in active) / len(active) if active else 0.0,
        "matched_deranged_mean": mean(matched),
        "matched_deranged_abs_median": median([abs(value) for value in matched]),
        "matched_advantage_mean": mean(advantages),
        "matched_advantage_median": median(advantages),
        "matched_advantage_positive_rate": sum(value > 0.0 for value in advantages) / len(advantages) if advantages else 0.0,
        "lexical_active_abs_median": lexical_abs_median,
        "lexical_zero_advantage": (
            active_median - lexical_abs_median
            if active_median is not None and lexical_abs_median is not None
            else None
        ),
    }
    gates = {
        "finite": finite_rate >= thresholds["finite_rate_min"],
        "active_positive": metrics["active_positive_rate"] >= thresholds["active_positive_rate_min"],
        "active_median": meets(active_median, thresholds["active_median_min"]),
        "matched_advantage_median": meets(
            metrics["matched_advantage_median"], thresholds["matched_advantage_median_min"]
        ),
        "matched_advantage_positive": metrics["matched_advantage_positive_rate"] >= thresholds["matched_advantage_positive_rate_min"],
        "lexical_zero_advantage": meets(
            metrics["lexical_zero_advantage"], thresholds["lexical_zero_advantage_min"]
        ),
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"] or audit["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1126 protocol is not authorized")

    thresholds = prereg["thresholds"]
    model_results: dict[str, Any] = {}
    all_units: list[dict[str, Any]] = []
    for model_name in protocol.MODELS:
        summary_path = protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
        detail_path = protocol.OUT_ROOT / "behavior" / model_name / "scores.jsonl"
        if not summary_path.exists() or not detail_path.exists():
            raise RuntimeError(f"missing model output: {model_name}")
        summary = protocol.read_json(summary_path)
        details = protocol.read_jsonl(detail_path)
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"protocol mismatch: {model_name}")
        if protocol.digest(details) != summary["detail_digest"]:
            raise RuntimeError(f"detail digest mismatch: {model_name}")

        grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            grouped[(row["partition"], row["panel_index"], row["replica"], row["route"])].append(row)
        units_by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for partition in protocol.BEHAVIOR_PARTITIONS:
            keys = sorted(key for key in grouped if key[0] == partition)
            panel_replicas = sorted({(key[1], key[2]) for key in keys})
            for panel_index, replica in panel_replicas:
                route_values: dict[str, float | None] = {}
                lexical_values: dict[str, float] = {}
                item = None
                pos = None
                for route in protocol.ROUTES:
                    rows = grouped[(partition, panel_index, replica, route)]
                    if len(rows) != 4:
                        raise RuntimeError(f"incomplete unit: {model_name}/{partition}/{panel_index}/{replica}/{route}")
                    score_table = {
                        (row["context_sense"], row["candidate_side"]): float(row["total_score"])
                        for row in rows
                    }
                    lexical_table = {
                        (row["context_sense"], row["candidate_side"]): float(row["lexical_overlap"])
                        for row in rows
                    }
                    route_values[route] = interaction(score_table) if all(
                        bool(row["finite"]) and math.isfinite(float(row["total_score"])) for row in rows
                    ) else None
                    lexical_values[route] = interaction(lexical_table)
                    item = rows[0]["item"]
                    pos = rows[0]["pos"]
                complete = all(
                    route_values[route] is not None and math.isfinite(route_values[route])
                    for route in protocol.ROUTES
                )
                unit = {
                    "model": model_name,
                    "partition": partition,
                    "panel_index": panel_index,
                    "replica": replica,
                    "item": item,
                    "pos": pos,
                    "active_z": route_values["active"],
                    "matched_deranged_z": route_values["matched_deranged"],
                    "matched_advantage": (
                        route_values["active"] - abs(route_values["matched_deranged"])
                        if complete else None
                    ),
                    "lexical_active_z": lexical_values["active"],
                    "lexical_matched_z": lexical_values["matched_deranged"],
                    "complete": complete,
                }
                units_by_partition[partition].append(unit)
                all_units.append(unit)

        partition_results = {}
        for partition in protocol.BEHAVIOR_PARTITIONS:
            partition_details = [row for row in details if row["partition"] == partition]
            finite_rate = sum(bool(row["finite"]) for row in partition_details) / len(partition_details)
            partition_results[partition] = summarize_partition(
                units_by_partition[partition],
                finite_rate,
                thresholds,
            )
        model_passed = all(result["passed"] for result in partition_results.values())
        model_results[model_name] = {
            "source_summary": summary,
            "partitions": partition_results,
            "passed_both_partitions": model_passed,
        }

    authorized_models = [
        model for model, result in model_results.items()
        if result["passed_both_partitions"]
    ]
    behavior_gate = len(authorized_models) >= int(thresholds["models_required"])
    predictions = {
        "P1_protocol_and_numerics": bool(audit["all_checks_passed"]) and all(
            result["source_summary"]["finite_rate"] >= thresholds["finite_rate_min"]
            for result in model_results.values()
        ),
        "P2_cross_resource_behavior": behavior_gate,
        "P3_hidden_holdout_authorized": behavior_gate,
        "P4_stop_if_behavior_fails": not behavior_gate,
    }
    auto_continue = {
        "value": bool(behavior_gate),
        "next_phase": "Phase1127 separately frozen hidden-use event protocol on hidden_holdout" if behavior_gate else None,
        "restrictions": [
            "authorization is for hidden description only",
            "no component hotspot selection",
            "no causal claim",
            "hidden_holdout behavior must be measured before hidden interpretation",
        ] if behavior_gate else [
            "do not scan hidden states",
            "retain K57 as resource-bound",
        ],
    }
    final = {
        "schema_version": "phase1126_semeval_lexsub_natural_cloze_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "material_digest": prereg["material_digest"],
        "model_results": model_results,
        "authorized_models": authorized_models,
        "predictions": predictions,
        "auto_continue": auto_continue,
        "evidence_boundary": (
            "A passing result is a behavior-level external-material replication. It is not a hidden representation, "
            "attention-use, component, causal, or training-formation result."
        ),
        "unit_digest": protocol.digest(all_units),
    }
    final["final_digest"] = protocol.digest(final)
    protocol.write_jsonl(protocol.OUT_ROOT / "analysis" / "interaction_units.jsonl", all_units)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
