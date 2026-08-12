#!/usr/bin/env python3
"""Finalize the frozen Phase1119 Qwen3 behavior scale comparison."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any

import phase1119_qwen3_scale_protocol as protocol


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in pairs if row["finite"]]
    count = max(len(finite), 1)
    return {
        "pair_count": len(pairs),
        "finite_pair_count": len(finite),
        "finite_fraction": len(finite) / max(len(pairs), 1),
        "direction_accuracy": sum(row["true_d"] > 0.0 for row in finite) / count,
        "control_direction_accuracy": sum(row["control_d"] > 0.0 for row in finite) / count,
        "control_advantage": (
            sum(row["true_d"] > 0.0 for row in finite)
            - sum(row["control_d"] > 0.0 for row in finite)
        )
        / count,
        "bidirectional_accuracy": sum(row["bidirectional"] for row in finite) / count,
        "median_true_d": statistics.median(row["true_d"] for row in finite) if finite else None,
        "median_control_d": statistics.median(row["control_d"] for row in finite) if finite else None,
    }


def compute_model(model_name: str, details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[row["pair_id"]].append(row)
    pairs: list[dict[str, Any]] = []
    for pair_id, panel in sorted(grouped.items()):
        panel = sorted(panel, key=lambda row: row["sense"])
        if len(panel) != 2 or [row["sense"] for row in panel] != [0, 1]:
            raise RuntimeError(f"malformed pair: {pair_id}")
        finite = all(row["finite"] for row in panel)
        pairs.append(
            {
                "pair_id": pair_id,
                "concept_id": panel[0]["concept_id"],
                "split": panel[0]["split"],
                "template": panel[0]["template"],
                "finite": finite,
                "true_d": float(panel[0]["true_z"] - panel[1]["true_z"]) if finite else math.nan,
                "control_d": float(panel[0]["control_z"] - panel[1]["control_z"])
                if finite
                else math.nan,
                "bidirectional": finite
                and panel[0]["true_z"] > 0.0
                and panel[1]["true_z"] < 0.0,
            }
        )

    overall = summarize_pairs(pairs)
    by_split = {
        split: summarize_pairs([row for row in pairs if row["split"] == split])
        for split in protocol.SPLITS
    }
    by_template = {
        str(template): summarize_pairs([row for row in pairs if row["template"] == template])
        for template in range(protocol.TEMPLATE_COUNT)
    }
    concept_panels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        concept_panels[row["concept_id"]].append(row)
    concepts: dict[str, Any] = {}
    for concept_id, panel in sorted(concept_panels.items()):
        finite = [row for row in panel if row["finite"]]
        median_d = statistics.median(row["true_d"] for row in finite) if finite else None
        concepts[concept_id] = {
            "split": panel[0]["split"],
            "finite_fraction": len(finite) / max(len(panel), 1),
            "median_true_d": median_d,
            "positive_median": median_d is not None and median_d > 0.0,
        }
    positive_count = sum(row["positive_median"] for row in concepts.values())
    positive_by_split = {
        split: sum(
            row["positive_median"]
            for row in concepts.values()
            if row["split"] == split
        )
        for split in protocol.SPLITS
    }
    concept_summary = {
        "concept_count": len(concepts),
        "positive_median_count": positive_count,
        "positive_median_fraction": positive_count / max(len(concepts), 1),
        "positive_by_split": positive_by_split,
        "concepts": concepts,
    }
    thresholds = protocol.ABSOLUTE_THRESHOLDS
    checks = {
        "finite_fraction": overall["finite_fraction"] >= thresholds["minimum_finite_fraction"],
        "overall_direction": overall["direction_accuracy"]
        >= thresholds["minimum_overall_direction_accuracy"],
        "split_direction": all(
            value["direction_accuracy"] >= thresholds["minimum_split_direction_accuracy"]
            for value in by_split.values()
        ),
        "template_direction": all(
            value["direction_accuracy"] >= thresholds["minimum_template_direction_accuracy"]
            for value in by_template.values()
        ),
        "overall_control_advantage": overall["control_advantage"]
        >= thresholds["minimum_overall_control_advantage"],
        "split_control_advantage": all(
            value["control_advantage"] >= thresholds["minimum_split_control_advantage"]
            for value in by_split.values()
        ),
        "template_control_advantage": all(
            value["control_advantage"] >= thresholds["minimum_template_control_advantage"]
            for value in by_template.values()
        ),
        "concept_positive_fraction": concept_summary["positive_median_fraction"]
        >= thresholds["minimum_concept_positive_fraction"],
        "concept_split_counts": all(
            value >= thresholds["minimum_positive_concepts_per_split"]
            for value in positive_by_split.values()
        ),
    }
    return {
        "schema_version": "phase1119_qwen3_scale_model_metrics.v1",
        "phase": protocol.PHASE,
        "model": model_name,
        "overall": overall,
        "by_split": by_split,
        "by_template": by_template,
        "concept_summary": concept_summary,
        "absolute_checks": checks,
        "absolute_qualified": all(checks.values()),
        "pair_digest": protocol.digest(pairs),
    }


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("protocol audit failed")

    metrics: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    details_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_name in prereg["models"]:
        root = protocol.OUT_ROOT / "behavior" / model_name
        details = list(protocol.read_jsonl(root / "candidate_detail.jsonl"))
        summary = protocol.read_json(root / "summary.json")
        if protocol.digest(details) != summary["detail_digest"]:
            raise RuntimeError(f"detail digest mismatch: {model_name}")
        details_by_model[model_name] = details
        summaries[model_name] = summary
        metrics[model_name] = compute_model(model_name, details)

    small = metrics["qwen3_4b"]
    large = metrics["qwen3_14b"]
    gains = {
        "direction_accuracy": large["overall"]["direction_accuracy"]
        - small["overall"]["direction_accuracy"],
        "control_advantage": large["overall"]["control_advantage"]
        - small["overall"]["control_advantage"],
        "bidirectional_accuracy": large["overall"]["bidirectional_accuracy"]
        - small["overall"]["bidirectional_accuracy"],
        "concept_positive_fraction": large["concept_summary"]["positive_median_fraction"]
        - small["concept_summary"]["positive_median_fraction"],
        "candidate_accuracy": summaries["qwen3_14b"]["candidate_accuracy"]
        - summaries["qwen3_4b"]["candidate_accuracy"],
        "split_direction": {
            split: large["by_split"][split]["direction_accuracy"]
            - small["by_split"][split]["direction_accuracy"]
            for split in protocol.SPLITS
        },
        "split_control_advantage": {
            split: large["by_split"][split]["control_advantage"]
            - small["by_split"][split]["control_advantage"]
            for split in protocol.SPLITS
        },
    }
    thresholds = protocol.SCALE_THRESHOLDS
    scale_checks = {
        "direction_gain": gains["direction_accuracy"] >= thresholds["minimum_direction_gain"],
        "control_advantage_gain": gains["control_advantage"]
        >= thresholds["minimum_control_advantage_gain"],
        "bidirectional_gain": gains["bidirectional_accuracy"]
        >= thresholds["minimum_bidirectional_gain"],
        "concept_fraction_gain": gains["concept_positive_fraction"]
        >= thresholds["minimum_concept_fraction_gain"],
        "split_direction_nonregression": min(gains["split_direction"].values())
        >= -thresholds["maximum_split_direction_regression"],
        "split_control_nonregression": min(gains["split_control_advantage"].values())
        >= -thresholds["maximum_split_control_advantage_regression"],
    }
    numerical_gate = all(
        summary["finite_fraction"] >= protocol.ABSOLUTE_THRESHOLDS["minimum_finite_fraction"]
        and summary["precision"]["has_fp16_parameters"]
        and not summary["precision"]["has_bf16_parameters"]
        and not summary["precision"]["has_quantized_modules"]
        for summary in summaries.values()
    )
    predictions = {
        "P1": "pass",
        "P2": "pass" if numerical_gate else "fail",
        "P3": "pass" if large["absolute_qualified"] else "fail",
        "P4": "pass" if all(scale_checks.values()) else "fail",
        "P5": "pass",
    }
    core = {
        "schema_version": "phase1119_qwen3_scale_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": metrics,
        "behavior_summary_digests": {
            name: summary["summary_digest"] for name, summary in summaries.items()
        },
        "gains_14b_minus_4b": gains,
        "scale_checks": scale_checks,
        "scale_gate_passed": all(scale_checks.values()),
        "prospective_predictions": predictions,
        "hidden_or_causal_authorized": False,
        "automatic_continuation": {
            "run_hidden_or_causal": False,
            "decision": (
                "seek an independently frozen third same-family size before proposing a scale law"
                if all(scale_checks.values())
                else "do not claim monotone scale improvement; audit which registered dimensions diverged"
            ),
        },
        "interpretation": {
            "positive_limit": (
                "A pass is one matched Qwen3 4B-to-14B behavior interval under a fixed tokenizer, "
                "material set, base-LM interface, FP16 precision, and matched candidate control."
            ),
            "not_claimed": [
                "a universal parameter-count causal effect",
                "a monotone multi-point scale law",
                "pure semantic modulation",
                "hidden representation conservation",
                "component or neuron causality",
            ],
        },
    }
    final = dict(core)
    final["final_digest"] = protocol.digest(core)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    return final


if __name__ == "__main__":
    print(json.dumps(finalize(), ensure_ascii=False, indent=2))
