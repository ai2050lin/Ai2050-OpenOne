#!/usr/bin/env python3
"""Aggregate Phase1053 distributed coalition localization."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1053_output_bridge_localization_protocol as protocol


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
            raise RuntimeError(f"{model} protocol digest drift")
    coalition_models = [
        model for model, row in summaries.items()
        if row["coalition_gate_passed"]
    ]
    rollout_models = [
        model for model, row in summaries.items()
        if row["rollout_gate_passed"]
    ]
    repeated = (
        len(coalition_models)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    decision = {
        "route": (
            "phase1054_pattern_family_extension"
            if repeated
            else "stop_with_distributed_coalition_map"
        ),
        "should_continue_automatically": repeated,
        "rationale": (
            "A discovery-frozen group-depth coalition retained the broad "
            "natural full-vocabulary effect in at least two models."
            if repeated
            else
            "The reduced coalition did not independently repeat in two "
            "behavior-eligible models."
        ),
    }
    aggregate = {
        "schema_version": "phase1053_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "coalition_passing_models": coalition_models,
        "rollout_passing_models": rollout_models,
        "model_results": {
            model: {
                key: value for key, value in summary.items()
                if key not in ("rollouts", "placement", "precision")
            }
            for model, summary in summaries.items()
        },
        "automatic_next_decision": decision,
        "interpretation_limits": [
            "The coalitions are greedy, not globally minimal.",
            "Cross-model repetition is functional, not coordinate identity.",
            "Broad graph cuts remain stronger than reduced coalitions.",
            "Failure to reduce is evidence for distribution, not proof.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} coalition={coalition_models} "
        f"rollout={rollout_models} next={decision['route']}"
    )


if __name__ == "__main__":
    main()
