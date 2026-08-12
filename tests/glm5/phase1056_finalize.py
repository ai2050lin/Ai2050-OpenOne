#!/usr/bin/env python3
"""Aggregate Phase1056 translation phase and coalition results."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1056_translation_phase_coalition_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    for model, row in summaries.items():
        if row["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model} protocol digest mismatch")
    coalition_models = [
        model for model, row in summaries.items()
        if row["coalition_gate_passed"]
    ]
    rollout_models = [
        model for model, row in summaries.items()
        if row["rollout_gate_passed"]
    ]
    suppression_models = [
        model for model, row in summaries.items()
        if row["behavior_eligible"]
        and row["early_only_rate"] <= 0.10
        and row["early_plus_postsource_suppression_contrast"] >= 0.30
    ]
    minimum = prereg["gates"]["minimum_repeated_models"]
    repeated = len(coalition_models) >= minimum
    decision = {
        "route": (
            "stop_at_translation_coalition_milestone"
            if repeated
            else "stop_with_model_specific_translation_phase"
        ),
        "should_continue_automatically": False,
        "rationale": (
            "A lexically held-out translation coalition repeated; freeze "
            "the milestone before choosing another language family."
            if repeated
            else
            "A compact translation coalition did not repeat in two "
            "behavior-qualified models."
        ),
    }
    aggregate = {
        "schema_version": "phase1056_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "coalition_passing_models": coalition_models,
        "rollout_passing_models": rollout_models,
        "early_suppression_models": suppression_models,
        "model_results": {
            model: {
                key: value for key, value in row.items()
                if key not in ("rollouts", "beam_trace", "placement")
            }
            for model, row in summaries.items()
        },
        "automatic_next_decision": decision,
        "interpretation_limits": [
            "The milestone is one English-to-French lexical task.",
            "Coalitions are compact rectangles, not minimal circuits.",
            "Model-specific phase differences remain real evidence.",
            "No further family should start without a new frozen protocol.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} coalition={coalition_models} "
        f"rollout={rollout_models} suppression={suppression_models} "
        f"next={decision['route']}"
    )


if __name__ == "__main__":
    main()
