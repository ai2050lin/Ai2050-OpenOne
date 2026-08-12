#!/usr/bin/env python3
"""Apply frozen Phase1114 behavior gates and continuation decision."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1114_wordnet_contextual_hypernym_protocol as protocol


def minimum(values: Iterable[float]) -> float:
    panel = list(values)
    return min(panel) if panel else 0.0


def case_metric(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    panel = list(rows)
    finite = [row for row in panel if row["finite"]]
    return {
        "count": len(panel),
        "candidate_finite_fraction": len(finite) / max(len(panel), 1),
        "candidate_accuracy": sum(bool(row["hit"]) for row in finite)
        / max(len(finite), 1),
        "sense0_preference_rate": sum(
            row["sense0_minus_sense1"] > 0.0 for row in finite
        )
        / max(len(finite), 1),
        "median_expected_margin": statistics.median(
            [row["expected_margin"] for row in finite]
        )
        if finite
        else None,
        "direct_candidate_output_rate": sum(
            bool(row["direct_candidate"]) for row in panel
        )
        / max(len(panel), 1),
        "direct_exact_accuracy": sum(bool(row["direct_hit"]) for row in panel)
        / max(len(panel), 1),
    }


def grouped_cases(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        buckets[key].append(row)
    return {
        "|".join(key): case_metric(values)
        for key, values in sorted(buckets.items())
    }


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["pair_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for pair_id, values in sorted(buckets.items()):
        by_sense = {int(row["sense"]): row for row in values}
        if set(by_sense) != {0, 1}:
            raise RuntimeError(f"incomplete Phase1114 pair {pair_id}")
        left, right = by_sense[0], by_sense[1]
        finite = bool(left["finite"] and right["finite"])
        context_effect = (
            left["sense0_minus_sense1"] - right["sense0_minus_sense1"]
            if finite
            else None
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "concept_id": left["concept_id"],
                "split": left["split"],
                "template": int(left["template"]),
                "base": left["base"],
                "candidate_labels": left["candidate_labels"],
                "finite": finite,
                "sense0_log_odds": left["sense0_minus_sense1"] if finite else None,
                "sense1_log_odds": right["sense0_minus_sense1"] if finite else None,
                "context_effect": context_effect,
                "context_direction_hit": bool(finite and context_effect > 0.0),
                "bidirectional_hit": bool(
                    finite
                    and left["sense0_minus_sense1"] > 0.0
                    and right["sense0_minus_sense1"] < 0.0
                ),
            }
        )
    return pairs


def pair_metric(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    panel = list(rows)
    finite = [row for row in panel if row["finite"]]
    effects = [row["context_effect"] for row in finite]
    return {
        "count": len(panel),
        "finite_fraction": len(finite) / max(len(panel), 1),
        "context_direction_accuracy": sum(
            bool(row["context_direction_hit"]) for row in finite
        )
        / max(len(finite), 1),
        "bidirectional_pair_accuracy": sum(
            bool(row["bidirectional_hit"]) for row in finite
        )
        / max(len(finite), 1),
        "median_context_effect": statistics.median(effects) if effects else None,
    }


def grouped_pairs(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        buckets[key].append(row)
    return {
        "|".join(key): pair_metric(values)
        for key, values in sorted(buckets.items())
    }


def analyze_model(model_name: str, thresholds: dict[str, float]) -> dict[str, Any]:
    summary = protocol.read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    rows = list(
        protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"
        )
    )
    pairs = build_pairs(rows)
    by_split = grouped_cases(rows, ("split",))
    by_sense = grouped_cases(rows, ("sense",))
    by_split_sense = grouped_cases(rows, ("split", "sense"))
    by_template = grouped_cases(rows, ("template",))
    pairs_by_split = grouped_pairs(pairs, ("split",))
    pairs_by_template = grouped_pairs(pairs, ("template",))
    sense_gap = abs(
        by_sense["0"]["candidate_accuracy"]
        - by_sense["1"]["candidate_accuracy"]
    )
    overall_pairs = pair_metric(pairs)
    checks = {
        "precision_fp16_no_quantization": (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        ),
        "case_and_pair_counts": len(rows) == 432 and len(pairs) == 216,
        "candidate_finite_fraction": (
            summary["candidate_finite_fraction"]
            >= thresholds["minimum_candidate_finite_fraction"]
            and minimum(
                value["candidate_finite_fraction"] for value in by_split.values()
            )
            >= thresholds["minimum_candidate_finite_fraction"]
        ),
        "overall_candidate_accuracy": summary["candidate_accuracy"]
        >= thresholds["minimum_overall_candidate_accuracy"],
        "split_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_split.values()
        )
        >= thresholds["minimum_split_candidate_accuracy"],
        "sense_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_sense.values()
        )
        >= thresholds["minimum_sense_candidate_accuracy"],
        "split_sense_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_split_sense.values()
        )
        >= thresholds["minimum_split_sense_candidate_accuracy"],
        "template_candidate_accuracy": minimum(
            value["candidate_accuracy"] for value in by_template.values()
        )
        >= thresholds["minimum_template_candidate_accuracy"],
        "overall_context_direction": overall_pairs["context_direction_accuracy"]
        >= thresholds["minimum_context_direction_accuracy"],
        "split_context_direction": minimum(
            value["context_direction_accuracy"] for value in pairs_by_split.values()
        )
        >= thresholds["minimum_split_context_direction_accuracy"],
        "template_context_direction": minimum(
            value["context_direction_accuracy"]
            for value in pairs_by_template.values()
        )
        >= thresholds["minimum_template_context_direction_accuracy"],
        "overall_bidirectional_pair": overall_pairs[
            "bidirectional_pair_accuracy"
        ]
        >= thresholds["minimum_bidirectional_pair_accuracy"],
        "split_bidirectional_pair": minimum(
            value["bidirectional_pair_accuracy"]
            for value in pairs_by_split.values()
        )
        >= thresholds["minimum_split_bidirectional_pair_accuracy"],
        "sense_accuracy_gap": sense_gap
        <= thresholds["maximum_sense_accuracy_gap"],
    }
    return {
        "model": model_name,
        "summary_digest": summary["summary_digest"],
        "overall": case_metric(rows),
        "overall_pairs": overall_pairs,
        "by_split": by_split,
        "by_sense": by_sense,
        "by_split_sense": by_split_sense,
        "by_template": by_template,
        "pairs_by_split": pairs_by_split,
        "pairs_by_template": pairs_by_template,
        "sense_accuracy_gap": sense_gap,
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
        raise RuntimeError("Phase1114 protocol audit failed")
    thresholds = preregistration["thresholds"]
    models = {
        model_name: analyze_model(model_name, thresholds)
        for model_name in protocol.MODELS
    }
    qualified_models = [
        model_name
        for model_name, row in models.items()
        if row["behavior_qualified"]
    ]
    cross_model_behavior_qualified = len(qualified_models) >= thresholds[
        "minimum_behavior_qualified_models"
    ]
    context_direction_models = [
        model_name
        for model_name, row in models.items()
        if row["checks"]["overall_context_direction"]
        and row["checks"]["split_context_direction"]
        and row["checks"]["template_context_direction"]
    ]
    bidirectional_models = [
        model_name
        for model_name, row in models.items()
        if row["checks"]["overall_bidirectional_pair"]
        and row["checks"]["split_bidirectional_pair"]
    ]
    predictions = {
        "P1": {
            "passed": bool(protocol_audit["all_checks_passed"]),
            "reason": "All source, native-example, isolation, nonleakage, tokenization, and digest audits passed.",
        },
        "P2": {
            "passed": cross_model_behavior_qualified,
            "qualified_models": qualified_models,
            "required_models": thresholds["minimum_behavior_qualified_models"],
        },
        "P3": {
            "passed": len(context_direction_models)
            >= thresholds["minimum_behavior_qualified_models"],
            "passing_models": context_direction_models,
        },
        "P4": {
            "passed": len(bidirectional_models)
            >= thresholds["minimum_behavior_qualified_models"],
            "passing_models": bidirectional_models,
        },
        "P5": {
            "passed": True,
            "direct_output_used_as_gate": False,
            "reason": "Direct top-token behavior is reported only as a diagnostic.",
        },
        "P6": {
            "passed": True,
            "hidden_state_accessed": False,
            "causal_intervention_accessed": False,
        },
    }
    if cross_model_behavior_qualified:
        continuation = {
            "automatic_hidden_scan": False,
            "decision": "contextual_behavior_qualified_but_hidden_contrast_not_pure",
            "next_stage": (
                "Freeze an independent matched-context protocol that separates semantic "
                "sense from the many surface words differing between the two native examples."
            ),
        }
    else:
        continuation = {
            "automatic_hidden_scan": False,
            "decision": "contextual_behavior_not_cross_model_qualified",
            "next_stage": (
                "Deny hidden-state access. Do not revise this frozen small-model protocol; "
                "move to the scale or training-dynamics arm, or obtain an independently "
                "validated natural-semantic material family."
            ),
        }
    final = {
        "schema_version": "phase1114_contextual_hypernym_final.v1",
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
                "A pass establishes behaviorally usable context-conditioned selection "
                "between two hidden WordNet hypernym candidates across independent concepts "
                "and prompt families."
            ),
            "negative_limit": (
                "A failure constrains this source, candidate construction, interface, and "
                "small-model panel; it does not show that contextual meaning is absent."
            ),
            "not_claimed": [
                "free-generation semantic competence",
                "cross-surface synonym invariance",
                "a hidden semantic coordinate",
                "payload transport or behavioral necessity",
                "cross-model physical conservation",
            ],
        },
    }
    final["final_digest"] = protocol.digest(final)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    authorization = {
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "qualified_models": qualified_models,
        "cross_model_behavior_qualified": cross_model_behavior_qualified,
        "hidden_state_authorized": False,
        "reason": continuation["decision"],
    }
    authorization["authorization_digest"] = protocol.digest(authorization)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json",
        authorization,
    )
    print(
        json.dumps(
            {
                "phase": protocol.PHASE,
                "qualified_models": qualified_models,
                "cross_model_behavior_qualified": cross_model_behavior_qualified,
                "predictions": predictions,
                "automatic_continuation": continuation,
                "final_digest": final["final_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
