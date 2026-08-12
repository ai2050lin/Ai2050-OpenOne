#!/usr/bin/env python3
"""Apply the frozen Phase1115 context-modulation confirmation gates."""

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

import phase1114_wordnet_contextual_hypernym_finalize as metrics
import phase1115_wordnet_context_modulation_confirmation_protocol as protocol


def minimum(values: Iterable[float]) -> float:
    panel = list(values)
    return min(panel) if panel else 0.0


def concept_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        buckets[row["concept_id"]].append(row)
    output: list[dict[str, Any]] = []
    for concept_id, values in sorted(buckets.items()):
        finite = [row for row in values if row["finite"]]
        effects = [row["context_effect"] for row in finite]
        median_effect = statistics.median(effects) if effects else None
        output.append(
            {
                "concept_id": concept_id,
                "base": values[0]["base"],
                "split": values[0]["split"],
                "template_count": len(values),
                "finite_fraction": len(finite) / max(len(values), 1),
                "median_context_effect": median_effect,
                "positive_median": bool(
                    len(finite) == len(values)
                    and median_effect is not None
                    and median_effect > 0.0
                ),
                "all_templates_positive": bool(
                    len(finite) == len(values)
                    and all(row["context_effect"] > 0.0 for row in finite)
                ),
            }
        )
    return output


def concept_metric(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    panel = list(rows)
    return {
        "count": len(panel),
        "positive_median_fraction": sum(row["positive_median"] for row in panel)
        / max(len(panel), 1),
        "all_templates_positive_fraction": sum(
            row["all_templates_positive"] for row in panel
        )
        / max(len(panel), 1),
        "minimum_finite_fraction": minimum(
            row["finite_fraction"] for row in panel
        ),
    }


def analyze_model(model_name: str, thresholds: dict[str, float]) -> dict[str, Any]:
    summary = protocol.base.read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    rows = list(
        protocol.base.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"
        )
    )
    pairs = metrics.build_pairs(rows)
    concepts = concept_rows(pairs)
    overall_pairs = metrics.pair_metric(pairs)
    by_split = metrics.grouped_pairs(pairs, ("split",))
    by_template = metrics.grouped_pairs(pairs, ("template",))
    overall_concepts = concept_metric(concepts)
    concepts_by_split = {
        split: concept_metric(row for row in concepts if row["split"] == split)
        for split in protocol.SPLITS
    }
    checks = {
        "precision_fp16_no_quantization": (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        ),
        "case_pair_concept_counts": len(rows) == 252
        and len(pairs) == 126
        and len(concepts) == 21,
        "candidate_finite_fraction": summary["candidate_finite_fraction"]
        >= thresholds["minimum_candidate_finite_fraction"]
        and minimum(value["finite_fraction"] for value in by_split.values())
        >= thresholds["minimum_candidate_finite_fraction"],
        "overall_context_direction": overall_pairs["context_direction_accuracy"]
        >= thresholds["minimum_context_direction_accuracy"],
        "split_context_direction": minimum(
            value["context_direction_accuracy"] for value in by_split.values()
        )
        >= thresholds["minimum_split_context_direction_accuracy"],
        "template_context_direction": minimum(
            value["context_direction_accuracy"] for value in by_template.values()
        )
        >= thresholds["minimum_template_context_direction_accuracy"],
        "concept_context_direction": overall_concepts[
            "positive_median_fraction"
        ]
        >= thresholds["minimum_concept_direction_accuracy"],
        "split_concept_context_direction": minimum(
            value["positive_median_fraction"]
            for value in concepts_by_split.values()
        )
        >= thresholds["minimum_split_concept_direction_accuracy"],
    }
    return {
        "model": model_name,
        "summary_digest": summary["summary_digest"],
        "case_diagnostics": metrics.case_metric(rows),
        "overall_pairs": overall_pairs,
        "pairs_by_split": by_split,
        "pairs_by_template": by_template,
        "overall_concepts": overall_concepts,
        "concepts_by_split": concepts_by_split,
        "concept_signs": {
            row["concept_id"]: {
                "base": row["base"],
                "split": row["split"],
                "positive_median": row["positive_median"],
                "all_templates_positive": row["all_templates_positive"],
                "median_context_effect": row["median_context_effect"],
            }
            for row in concepts
        },
        "checks": checks,
        "modulation_qualified": all(checks.values()),
    }


def main() -> None:
    preregistration = protocol.base.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.base.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1115 protocol audit failed")
    thresholds = preregistration["thresholds"]
    models = {
        model_name: analyze_model(model_name, thresholds)
        for model_name in protocol.MODELS
    }
    qualified_models = [
        model_name
        for model_name, row in models.items()
        if row["modulation_qualified"]
    ]
    all_concepts = sorted(
        next(iter(models.values()))["concept_signs"].keys()
    )
    shared_positive = [
        concept_id
        for concept_id in all_concepts
        if sum(
            bool(models[model]["concept_signs"][concept_id]["positive_median"])
            for model in qualified_models
        )
        >= 2
    ]
    shared_fraction = len(shared_positive) / max(len(all_concepts), 1)
    shared_gate = shared_fraction >= thresholds[
        "minimum_shared_two_model_concept_fraction"
    ]
    cross_model_confirmed = (
        len(qualified_models) >= thresholds["minimum_modulation_qualified_models"]
        and shared_gate
    )
    predictions = {
        "P1": {
            "passed": bool(protocol_audit["all_checks_passed"]),
            "reason": "All source, Phase1114-disjointness, template, tokenization, and digest checks passed.",
        },
        "P2": {
            "passed": len(qualified_models)
            >= thresholds["minimum_modulation_qualified_models"],
            "qualified_models": qualified_models,
        },
        "P3": {
            "passed": shared_gate and len(qualified_models) >= 2,
            "shared_positive_concept_count": len(shared_positive),
            "concept_count": len(all_concepts),
            "shared_positive_fraction": shared_fraction,
        },
        "P4": {
            "passed": True,
            "candidate_accuracy_used_as_gate": False,
            "bidirectional_use_claimed": False,
        },
        "P5": {
            "passed": True,
            "hidden_state_accessed": False,
            "causal_intervention_accessed": False,
        },
    }
    if cross_model_confirmed:
        continuation = {
            "automatic_hidden_scan": False,
            "decision": "context_modulation_independently_confirmed_hidden_scan_denied",
            "next_stage": (
                "Register the behavior-level context-modulation primitive. Any internal "
                "study must first create a matched-surface contrast or use training/scale "
                "data; the current native sentence difference is not a pure hidden contrast."
            ),
        }
    else:
        continuation = {
            "automatic_hidden_scan": False,
            "decision": "context_modulation_not_independently_confirmed",
            "next_stage": (
                "Do not scan hidden states or reopen prompt search. Move to the scale, "
                "training-dynamics, or independently annotated material arm."
            ),
        }
    final = {
        "schema_version": "phase1115_context_modulation_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "thresholds": thresholds,
        "models": models,
        "qualified_models": qualified_models,
        "shared_positive_concepts": shared_positive,
        "shared_positive_fraction": shared_fraction,
        "cross_model_context_modulation_confirmed": cross_model_confirmed,
        "predictions": predictions,
        "automatic_continuation": continuation,
        "interpretation": {
            "positive_limit": (
                "A pass confirms that natural source contexts repeatedly shift a hidden "
                "candidate pair's output margin in the WordNet-consistent direction across "
                "new concepts, interfaces, and at least two models."
            ),
            "not_claimed": [
                "reliable sense classification",
                "candidate-free natural generation",
                "cross-surface semantic invariance",
                "a hidden semantic coordinate",
                "causal necessity",
            ],
        },
    }
    final["final_digest"] = protocol.base.digest(final)
    protocol.base.write_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json", final
    )
    authorization = {
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "qualified_models": qualified_models,
        "cross_model_context_modulation_confirmed": cross_model_confirmed,
        "hidden_state_authorized": False,
        "reason": continuation["decision"],
    }
    authorization["authorization_digest"] = protocol.base.digest(authorization)
    protocol.base.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json",
        authorization,
    )
    print(
        json.dumps(
            {
                "phase": protocol.PHASE,
                "qualified_models": qualified_models,
                "shared_positive_concept_count": len(shared_positive),
                "shared_positive_fraction": shared_fraction,
                "cross_model_context_modulation_confirmed": cross_model_confirmed,
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
