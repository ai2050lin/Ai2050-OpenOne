#!/usr/bin/env python3
"""Finalize the Phase1067 necessity stress and coalition map."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1067_reasoning_necessity_coalition_protocol as protocol


def finalize() -> dict[str, Any]:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1067 protocol audit failed")
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    model_maps = {}
    for model, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"protocol drift for {model}")
        if (
            summary["precision"]["has_quantized_modules"]
            or summary["precision"]["has_bf16_parameters"]
            or not summary["precision"]["has_fp16_parameters"]
        ):
            raise RuntimeError(f"precision drift for {model}")
        conditions = summary["condition_results"]
        baseline = conditions["semantic_swap_kv"][
            "bidirectional_opposite_class_flip_rate"
        ]
        coalition_rows = {}
        for name, row in conditions.items():
            if row["kind"] != "group_coalition":
                continue
            rate = row["bidirectional_opposite_class_flip_rate"]
            coalition_rows[name] = {
                "bidirectional_flip_rate": rate,
                "retention_vs_all_groups": (
                    rate / baseline if baseline > 0 else None
                ),
                "loss_vs_all_groups": baseline - rate,
                "group_count": len(row["groups"]),
                "groups": row["groups"],
            }
        model_maps[model] = {
            "semantic_swap": {
                "kv": baseline,
                "k_only": conditions["semantic_swap_k_only"][
                    "bidirectional_opposite_class_flip_rate"
                ],
                "v_only": conditions["semantic_swap_v_only"][
                    "bidirectional_opposite_class_flip_rate"
                ],
            },
            "neutralization": {
                "both_kv_accuracy_drop": conditions[
                    "semantic_mean_kv"
                ]["own_class_accuracy_drop"],
                "both_k_accuracy_drop": conditions[
                    "semantic_mean_k_only"
                ]["own_class_accuracy_drop"],
                "both_v_accuracy_drop": conditions[
                    "semantic_mean_v_only"
                ]["own_class_accuracy_drop"],
                "premise1_kv_accuracy_drop": conditions[
                    "semantic_mean_premise1_kv"
                ]["own_class_accuracy_drop"],
                "premise2_kv_accuracy_drop": conditions[
                    "semantic_mean_premise2_kv"
                ]["own_class_accuracy_drop"],
            },
            "semantic_preserving_surface_swap_accuracy_drop": (
                conditions["surface_preserving_swap_kv"][
                    "own_class_accuracy_drop"
                ]
            ),
            "group_coalitions": coalition_rows,
            "necessity_stress_gate_passed": summary[
                "necessity_stress_gate_passed"
            ],
        }

    passing_models = [
        model for model, summary in summaries.items()
        if summary["necessity_stress_gate_passed"]
    ]
    repeated = (
        len(passing_models)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    all_k_dominant = all(
        model_maps[model]["semantic_swap"]["k_only"]
        > model_maps[model]["semantic_swap"]["v_only"]
        for model in protocol.MODELS
    )
    all_premise2_dominant = all(
        model_maps[model]["neutralization"][
            "premise2_kv_accuracy_drop"
        ]
        >= model_maps[model]["neutralization"][
            "premise1_kv_accuracy_drop"
        ]
        for model in protocol.MODELS
    )
    automatic = {
        "schema_version": "phase1067_automatic_next.v1",
        "phase": protocol.PHASE,
        "should_continue_automatically": False,
        "passing_necessity_stress_models": passing_models,
        "cross_model_necessity_stress_repetition": repeated,
        "route": "stop_at_controlled_reasoning_milestone",
        "next_large_task": (
            "Build an independently frozen reasoning atlas that varies "
            "relation type, chain length, query, answer position, and "
            "distractors before any further physical compression."
        ),
        "rationale": (
            "Phase1067 exhausts the preregistered controlled three-person "
            "task. More group compression would optimize one toy surface "
            "rather than test a reusable reasoning mechanism."
        ),
    }
    aggregate = {
        "schema_version": "phase1067_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_summaries": summaries,
        "model_functional_maps": model_maps,
        "passing_necessity_stress_models": passing_models,
        "cross_model_necessity_stress_repetition": repeated,
        "cross_model_descriptive_ordering": {
            "k_only_greater_than_v_only_all_models": all_k_dominant,
            "premise2_neutralization_not_less_than_premise1_all_models": (
                all_premise2_dominant
            ),
        },
        "automatic_next_decision": automatic,
        "claim_boundary": {
            "supported_if_repeated": (
                "The paired semantic difference at the two premise "
                "endpoints contributes to the controlled transitive-answer "
                "decision over a broad middle-late K/V coalition."
            ),
            "not_supported": [
                "The paired mean is a natural network state.",
                "The mapped coalition is the complete reasoning circuit.",
                "K is a semantic address or V is semantic content.",
                "The result generalizes to other relations or chain lengths.",
                "The observed topology is an optimal or biological code.",
            ],
        },
    }
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json", aggregate
    )
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "model_functional_maps.json",
        model_maps,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json",
        automatic,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "model_maps": model_maps,
        "automatic_next": automatic,
    }, ensure_ascii=False, indent=2))
    return aggregate


def main() -> None:
    finalize()


if __name__ == "__main__":
    main()
