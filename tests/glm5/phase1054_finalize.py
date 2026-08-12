#!/usr/bin/env python3
"""Aggregate Phase1054 joint K/V and EOS-aware rollout results."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1054_joint_kv_rollout_protocol as protocol


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

    coalition_models = [
        model for model, row in summaries.items()
        if row["coalition_gate_passed"]
    ]
    rollout_models = [
        model for model, row in summaries.items()
        if row["rollout_gate_passed"]
    ]
    eligible_broad_models = [
        model for model, row in summaries.items()
        if row["behavior_eligible"]
        and row["confirmation_results"]["selected_full"][
            "both_counterfactual_top1_rate"
        ] >= prereg["gates"]["joint_both_counterfactual_rate_min"]
    ]
    minimum = prereg["gates"]["minimum_repeated_models"]
    if (
        len(coalition_models) >= minimum
        and len(rollout_models) >= 1
    ):
        route = "phase1055_pattern_family_transfer"
        should_continue = True
        rationale = (
            "A compact joint rectangle repeated in two eligible models "
            "and EOS-aware output matching passed in at least one."
        )
    elif len(eligible_broad_models) >= minimum:
        route = "phase1055_nonrectangular_block_atlas"
        should_continue = True
        rationale = (
            "The broad selected-position graph cut repeated, but a "
            "compact rectangular coalition did not repeat."
        )
    else:
        route = "stop_and_reassess_output_bridge"
        should_continue = False
        rationale = (
            "The broad full-vocabulary bridge no longer repeated on the "
            "causal confirmation reserve."
        )
    aggregate = {
        "schema_version": "phase1054_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "coalition_passing_models": coalition_models,
        "rollout_passing_models": rollout_models,
        "eligible_broad_graph_cut_models": eligible_broad_models,
        "model_results": {
            model: {
                key: value for key, value in row.items()
                if key not in ("rollouts", "beam_trace", "placement")
            }
            for model, row in summaries.items()
        },
        "automatic_next_decision": {
            "route": route,
            "should_continue_automatically": should_continue,
            "rationale": rationale,
        },
        "interpretation_limits": [
            "EOS correction can revise the old rollout conclusion.",
            "A compact rectangle is not a minimum causal circuit.",
            "Only behavior-eligible models count toward repetition.",
            "One-token label termination is not free-form generation.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} coalition={coalition_models} "
        f"rollout={rollout_models} broad={eligible_broad_models} "
        f"next={route}"
    )


if __name__ == "__main__":
    main()
