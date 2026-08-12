#!/usr/bin/env python3
"""Aggregate Phase1063 and enforce its automatic-next gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1063_lexical_behavior_atlas_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model_name: protocol.read_json(
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "summary.json"
        )
        for model_name in protocol.MODELS
    }
    for model_name, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(
                f"protocol digest drift for {model_name}"
            )
    primary_passing = [
        model_name
        for model_name, summary in summaries.items()
        if summary["primary_behavior_gate_passed"]
    ]
    panel_passing = {
        panel: [
            model_name
            for model_name, summary in summaries.items()
            if summary["panel_behavior_gates"][panel]
        ]
        for panel in protocol.PANELS
    }
    repeated = (
        len(primary_passing)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    if repeated:
        decision = {
            "route": "continue_to_phase1064_cross_panel_transport",
            "should_continue_automatically": True,
            "rationale": (
                "At least two models formed behavior-qualified anchor and "
                "novel-noun pair pools; causal transport is authorized."
            ),
        }
    else:
        decision = {
            "route": "stop_with_lexical_behavior_atlas",
            "should_continue_automatically": False,
            "rationale": (
                "Fewer than two models formed both required pair pools; "
                "causal transport would be behavior-confounded."
            ),
        }
    aggregate = {
        "schema_version": "phase1063_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_results": summaries,
        "panel_passing_models": panel_passing,
        "primary_passing_models": primary_passing,
        "cross_model_behavior_repetition": repeated,
        "automatic_next_decision": decision,
        "interpretation_limits": prereg["interpretation_limits"],
    }
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json", aggregate
    )
    print(
        f"Phase{protocol.PHASE} finalized: "
        f"repeated={repeated} passing={primary_passing} "
        f"route={decision['route']}"
    )


if __name__ == "__main__":
    main()
