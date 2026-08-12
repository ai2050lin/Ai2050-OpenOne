#!/usr/bin/env python3
"""Freeze Phase1113 behavior gates and automatic continuation decision."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1113_wordnet_semantic_quadrant_protocol as protocol


def metric(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    panel = list(rows)
    finite = [row for row in panel if row["finite"]]
    return {
        "count": len(panel),
        "candidate_finite_fraction": len(finite) / max(len(panel), 1),
        "candidate_accuracy": sum(bool(row["hit"]) for row in finite) / max(len(finite), 1),
        "direct_candidate_output_rate": sum(
            bool(row["direct_candidate"]) for row in panel
        ) / max(len(panel), 1),
        "direct_exact_accuracy": sum(bool(row["direct_hit"]) for row in panel) / max(len(panel), 1),
    }


def grouped(
    rows: list[dict[str, Any]], fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        buckets[key].append(row)
    return {"|".join(key): metric(values) for key, values in sorted(buckets.items())}


def minimum(values: Iterable[float]) -> float:
    panel = list(values)
    return min(panel) if panel else 0.0


def analyze_model(model_name: str, thresholds: dict[str, float]) -> dict[str, Any]:
    summary = protocol.read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    rows = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"
    ))
    by_split = grouped(rows, ("split",))
    by_quadrant = grouped(rows, ("quadrant",))
    by_split_quadrant = grouped(rows, ("split", "quadrant"))
    by_template_quadrant = grouped(rows, ("template", "quadrant"))
    by_answer_order = grouped(rows, ("answer_order",))
    by_split_quadrant_order = grouped(rows, ("split", "quadrant", "answer_order"))
    order_buckets: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for key, value in by_split_quadrant_order.items():
        split, quadrant, order = key.split("|")
        order_buckets[(split, quadrant)][int(order)] = value["candidate_accuracy"]
    order_gaps = {
        f"{split}|{quadrant}": abs(values.get(0, 0.0) - values.get(1, 0.0))
        for (split, quadrant), values in sorted(order_buckets.items())
    }
    critical = (
        "same_surface_different_sense",
        "different_surface_same_sense",
    )
    checks = {
        "precision_fp16_no_quantization": (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        ),
        "case_count": len(rows) == 864 == summary["candidate_count"],
        "candidate_finite_fraction": (
            summary["candidate_finite_fraction"]
            >= thresholds["minimum_candidate_finite_fraction"]
            and minimum(
                value["candidate_finite_fraction"] for value in by_split.values()
            ) >= thresholds["minimum_candidate_finite_fraction"]
        ),
        "overall_candidate_accuracy": (
            summary["candidate_accuracy"]
            >= thresholds["minimum_overall_candidate_accuracy"]
        ),
        "split_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_split.values()
        ) >= thresholds["minimum_split_candidate_accuracy"],
        "quadrant_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_quadrant.values()
        ) >= thresholds["minimum_quadrant_candidate_accuracy"],
        "split_quadrant_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_split_quadrant.values()
        ) >= thresholds["minimum_split_quadrant_candidate_accuracy"],
        "template_quadrant_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_template_quadrant.values()
        ) >= thresholds["minimum_template_quadrant_candidate_accuracy"],
        "anti_shortcut_quadrants": all(
            by_quadrant[quadrant]["candidate_accuracy"]
            >= thresholds["minimum_anti_shortcut_quadrant_accuracy"]
            for quadrant in critical
        ),
        "answer_order_gap": max(order_gaps.values(), default=1.0)
        <= thresholds["maximum_answer_order_accuracy_gap"],
    }
    return {
        "model": model_name,
        "summary_digest": summary["summary_digest"],
        "overall": metric(rows),
        "by_split": by_split,
        "by_quadrant": by_quadrant,
        "by_split_quadrant": by_split_quadrant,
        "by_template_quadrant": by_template_quadrant,
        "by_answer_order": by_answer_order,
        "answer_order_gaps_by_split_quadrant": order_gaps,
        "maximum_answer_order_gap": max(order_gaps.values(), default=1.0),
        "checks": checks,
        "behavior_qualified": all(checks.values()),
    }


def main() -> None:
    preregistration = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1113 protocol audit failed")
    thresholds = preregistration["thresholds"]
    models = {
        model_name: analyze_model(model_name, thresholds)
        for model_name in protocol.MODELS
    }
    qualified_models = [
        model_name for model_name, row in models.items() if row["behavior_qualified"]
    ]
    cross_model_behavior_qualified = (
        len(qualified_models) >= thresholds["minimum_behavior_qualified_models"]
    )
    anti_shortcut_model_count = sum(
        row["checks"]["anti_shortcut_quadrants"] for row in models.values()
    )
    answer_order_model_count = sum(
        row["checks"]["answer_order_gap"] for row in models.values()
    )
    predictions = {
        "P1": {
            "passed": bool(protocol_audit["all_checks_passed"]),
            "reason": "All public-source, lexical-isolation, factorial, tokenization, and digest checks passed.",
        },
        "P2": {
            "passed": cross_model_behavior_qualified,
            "qualified_models": qualified_models,
            "required_models": thresholds["minimum_behavior_qualified_models"],
        },
        "P3": {
            "passed": anti_shortcut_model_count >= thresholds["minimum_behavior_qualified_models"],
            "passing_model_count": anti_shortcut_model_count,
        },
        "P4": {
            "passed": answer_order_model_count >= thresholds["minimum_behavior_qualified_models"],
            "passing_model_count": answer_order_model_count,
        },
        "P5": {
            "passed": True,
            "hidden_state_accessed": False,
            "causal_intervention_accessed": False,
            "reason": "Phase1113 is behavior-only by preregistration regardless of P2-P4.",
        },
    }
    if cross_model_behavior_qualified:
        continuation = {
            "automatic_hidden_scan": False,
            "decision": "behavior_object_qualified_hidden_scan_not_authorized",
            "next_stage": (
                "Freeze a separate natural-context semantic-routing protocol using the "
                "qualified source senses and matched nonsemantic controls. Do not reuse this "
                "metalinguistic classification response as hidden semantic evidence."
            ),
        }
    else:
        continuation = {
            "automatic_hidden_scan": False,
            "decision": "behavior_object_not_cross_model_qualified",
            "next_stage": (
                "Do not scan hidden states. Treat failed split/template/quadrant cells as an "
                "interface-domain boundary and move the scale or training-dynamics arm before "
                "constructing another semantic hidden-state atlas."
            ),
        }
    final = {
        "schema_version": "phase1113_wordnet_semantic_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "thresholds": thresholds,
        "models": models,
        "qualified_models": qualified_models,
        "cross_model_behavior_qualified": cross_model_behavior_qualified,
        "predictions": predictions,
        "automatic_continuation": continuation,
        "interpretation": {
            "positive_limit": (
                "A pass establishes behaviorally usable WordNet noun-sense identity across a "
                "surface-overlap four-quadrant design and independent interfaces."
            ),
            "negative_limit": (
                "A failure constrains this interface and model panel; it does not show that "
                "semantic relations or content readers do not exist."
            ),
            "not_claimed": [
                "natural-language semantic-address invariance",
                "a hidden semantic coordinate",
                "payload transport or behavioral necessity",
                "cross-model physical conservation",
            ],
        },
    }
    final["final_digest"] = protocol.digest(final)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json",
        {
            "phase": protocol.PHASE,
            "protocol_digest": preregistration["protocol_digest"],
            "qualified_models": qualified_models,
            "cross_model_behavior_qualified": cross_model_behavior_qualified,
            "hidden_state_authorized": False,
            "reason": continuation["decision"],
            "authorization_digest": protocol.digest({
                "qualified_models": qualified_models,
                "cross_model_behavior_qualified": cross_model_behavior_qualified,
                "hidden_state_authorized": False,
            }),
        },
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "qualified_models": qualified_models,
        "cross_model_behavior_qualified": cross_model_behavior_qualified,
        "predictions": predictions,
        "automatic_continuation": continuation,
        "final_digest": final["final_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
