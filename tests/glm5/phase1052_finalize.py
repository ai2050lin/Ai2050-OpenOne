#!/usr/bin/env python3
"""Aggregate natural full-vocabulary K/V bridge evidence."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1052_full_vocab_kv_bridge_protocol as protocol


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
    local_models = [
        model for model, row in summaries.items()
        if row["local_bridge_gate_passed"]
    ]
    broad_models = [
        model for model, row in summaries.items()
        if row["broad_graph_cut_gate_passed"]
    ]
    rollout_models = [
        model for model, row in summaries.items()
        if row["rollout_gate_passed"]
    ]
    minimum = prereg["gates"]["minimum_repeated_models"]
    if len(local_models) >= minimum and rollout_models:
        route = "phase1053_pattern_family_extension"
        should_continue = True
        rationale = (
            "A local frozen K/V bridge repeated across models and at "
            "least one model reproduced the other natural trajectory."
        )
    elif len(broad_models) >= minimum:
        route = "phase1053_output_bridge_localization"
        should_continue = True
        rationale = (
            "Only the broad selected-position graph cut repeated; "
            "localize the missing depth/head/output subroute."
        )
    else:
        route = "stop_and_reassess_transport"
        should_continue = False
        rationale = (
            "Natural full-vocabulary transport did not repeat in two "
            "behavior-eligible models."
        )
    aggregate = {
        "schema_version": "phase1052_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "local_bridge_passing_models": local_models,
        "broad_graph_cut_passing_models": broad_models,
        "rollout_passing_models": rollout_models,
        "model_results": {
            model: {
                key: value for key, value in summary.items()
                if key not in ("rollouts", "placement", "precision")
            }
            for model, summary in summaries.items()
        },
        "automatic_next_decision": {
            "route": route,
            "should_continue_automatically": should_continue,
            "rationale": rationale,
        },
        "interpretation_limits": [
            "Local bridge and broad graph-cut evidence are separate.",
            "Only Phase1051 behavior-eligible models count as repeats.",
            "Candidate margins are diagnostics, not closure gates.",
            "Full trajectory matching is stricter than first-token flipping.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} local={local_models} broad={broad_models} "
        f"rollout={rollout_models} next={route}"
    )


if __name__ == "__main__":
    main()
