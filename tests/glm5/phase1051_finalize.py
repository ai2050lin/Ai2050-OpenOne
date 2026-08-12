#!/usr/bin/env python3
"""Aggregate Phase1051 and decide whether causal bridging is authorized."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1051_natural_behavior_protocol as protocol


def strip_masks(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_masks(item)
            for key, item in value.items()
            if key != "correct_pair_mask"
        }
    if isinstance(value, list):
        return [strip_masks(item) for item in value]
    return value


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
    passing = [
        model for model, summary in summaries.items()
        if summary["behavior_gate_passed"]
    ]
    should_continue = (
        len(passing)
        >= prereg["gates"]["minimum_passing_models"]
    )
    decision = {
        "route": (
            "phase1052_full_vocab_kv_bridge"
            if should_continue
            else "stop_and_redesign_behavior_protocol"
        ),
        "should_continue_automatically": should_continue,
        "rationale": (
            "At least two models passed a discovery-selected, held-out "
            "full-vocabulary behavior protocol with at least 100 exact "
            "counterfactual pairs."
            if should_continue
            else
            "Fewer than two models passed the held-out natural behavior "
            "gate; causal transport would be uninterpretable."
        ),
    }
    aggregate = {
        "schema_version": "phase1051_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "passing_models": passing,
        "model_results": {
            model: {
                "frozen_variant": summary["frozen_variant"],
                "confirmation": strip_masks(summary["confirmation"]),
                "behavior_gate_passed": summary[
                    "behavior_gate_passed"
                ],
                "clean_rollout_summary": summary[
                    "clean_rollout_summary"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for model, summary in summaries.items()
        },
        "automatic_next_decision": decision,
        "interpretation": [
            "This phase validates an output protocol, not an internal route.",
            "Format selection used discovery units only.",
            "Confirmation and causal holdout units are disjoint.",
            "A passing model authorizes but does not guarantee causal closure.",
        ],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} passing={passing} "
        f"next={decision['route']}"
    )


if __name__ == "__main__":
    main()
