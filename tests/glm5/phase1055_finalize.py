#!/usr/bin/env python3
"""Aggregate Phase1055 natural translation transfer."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1055_pattern_family_transfer_protocol as protocol


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
    for model, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(f"{model} protocol digest mismatch")
    behavior_models = [
        model for model, row in summaries.items()
        if row["behavior_gate_passed"]
    ]
    broad_models = [
        model for model, row in summaries.items()
        if row["broad_bridge_gate_passed"]
    ]
    fact_reuse_models = [
        model for model, row in summaries.items()
        if row["fact_rectangle_reuse_gate_passed"]
    ]
    rollout_models = [
        model for model, row in summaries.items()
        if row["rollout_gate_passed"]
    ]
    minimum = prereg["gates"]["minimum_repeated_models"]
    if len(broad_models) >= minimum:
        route = "phase1056_translation_coalition_localization"
        should_continue = True
        rationale = (
            "A source-term K/V graph cut controlled natural translation "
            "in at least two behavior-qualified models."
        )
    else:
        route = "stop_with_pattern_specific_difference"
        should_continue = False
        rationale = (
            "Natural translation behavior or source-term K/V transport "
            "did not repeat in two models."
        )
    aggregate = {
        "schema_version": "phase1055_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "behavior_passing_models": behavior_models,
        "broad_bridge_passing_models": broad_models,
        "fact_rectangle_reuse_models": fact_reuse_models,
        "rollout_passing_models": rollout_models,
        "model_results": {
            model: {
                key: value for key, value in row.items()
                if key not in ("rollouts", "placement")
            }
            for model, row in summaries.items()
        },
        "automatic_next_decision": {
            "route": route,
            "should_continue_automatically": should_continue,
            "rationale": rationale,
        },
        "interpretation_limits": [
            "The task covers one lexical translation direction.",
            "Broad source transport is not a localized algorithm.",
            "Physical reuse and functional topology reuse are distinct.",
            "Failed transfer does not imply that translation lacks structure.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} behavior={behavior_models} "
        f"broad={broad_models} fact={fact_reuse_models} "
        f"rollout={rollout_models} next={route}"
    )


if __name__ == "__main__":
    main()
