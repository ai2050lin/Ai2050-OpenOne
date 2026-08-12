#!/usr/bin/env python3
"""Aggregate Phase1057 fresh translation trajectory results."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1057_translation_trajectory_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    rows = {
        model_name: protocol.read_json(
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "summary.json"
        )
        for model_name in prereg["models"]
    }
    for model_name, row in rows.items():
        if row["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(
                f"protocol digest drift for {model_name}"
            )
    behavior_models = [
        model_name for model_name, row in rows.items()
        if row["behavior_gate_passed"]
    ]
    bridge_models = [
        model_name for model_name, row in rows.items()
        if row["fresh_bridge_gate_passed"]
    ]
    rollout_models = [
        model_name for model_name, row in rows.items()
        if row["rollout_gate_passed"]
    ]
    phase_classes = {
        model_name: row["phase_class"]
        for model_name, row in rows.items()
        if row["behavior_gate_passed"]
    }
    should_continue = (
        len(bridge_models)
        >= prereg["gates"]["minimum_repeated_models"]
        and len(rollout_models) >= 1
    )
    if should_continue:
        route = "phase1058_multitoken_translation"
        rationale = (
            "The fresh lexical bridge repeated in two models, trajectory "
            "capture completed, and at least one EOS-aware rollout passed."
        )
    else:
        route = "stop_with_unstable_fresh_translation_bridge"
        rationale = (
            "Fresh lexical transport or EOS-aware generation did not "
            "meet the preregistered repetition boundary."
        )

    aggregate = {
        "schema_version": "phase1057_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(prereg["models"]),
        "behavior_passing_models": behavior_models,
        "fresh_bridge_passing_models": bridge_models,
        "rollout_passing_models": rollout_models,
        "phase_classes": phase_classes,
        "model_results": {
            model_name: {
                key: value for key, value in row.items()
                if key not in (
                    "placement",
                    "rollouts",
                    "trajectory",
                    "condition_results",
                )
            }
            for model_name, row in rows.items()
        },
        "automatic_next_decision": {
            "route": route,
            "should_continue_automatically": should_continue,
            "rationale": rationale,
        },
        "interpretation_limits": [
            "Phase classes describe interventions, not natural modules.",
            "Trajectory distance is not a semantic coordinate system.",
            "A fresh lexical bridge is not sentence translation.",
            "Physical phase differences can remain model-specific.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} behavior={behavior_models} "
        f"bridge={bridge_models} rollout={rollout_models} "
        f"classes={phase_classes} next={route}"
    )


if __name__ == "__main__":
    main()
