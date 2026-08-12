#!/usr/bin/env python3
"""Aggregate Phase1059 without changing preregistered criteria."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1059_lexically_heldout_composition_protocol as protocol


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
    behavior_models = [
        model_name
        for model_name, row in summaries.items()
        if row["behavior_gate_passed"]
    ]
    repeated_models = [
        model_name
        for model_name, row in summaries.items()
        if row["fully_heldout_composition_gate_passed"]
    ]
    repeated = (
        len(repeated_models)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    route = {
        "route": (
            "phase1060_sentence_role_transport"
            if repeated
            else "stop_and_revise_behavior_or_holdout_protocol"
        ),
        "should_continue_automatically": repeated,
        "rationale": (
            "Fully held-out phrase transport repeated in at least two "
            "behavior-qualified models; sentence-level role transport is "
            "the preregistered next complexity step."
            if repeated
            else
            "The fully held-out phrase result did not repeat in two "
            "behavior-qualified models; sentence-level extension is not "
            "authorized."
        ),
    }
    payload = {
        "schema_version": "phase1059_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "behavior_passing_models": behavior_models,
        "fully_heldout_passing_models": repeated_models,
        "cross_model_fully_heldout_repetition": repeated,
        "model_results": {
            model_name: {
                "precision": row["precision"],
                "clean_behavior": row["clean_behavior"],
                "valid_pair_counts": row["valid_pair_counts"],
                "behavior_gate_passed": row[
                    "behavior_gate_passed"
                ],
                "component_rates": row["component_rates"],
                "phase_rates": row["phase_rates"],
                "phase_class": row["phase_class"],
                "channel_rates": row["channel_rates"],
                "support_map": row["support_map"],
                "maximum_role_control_rate": row[
                    "maximum_role_control_rate"
                ],
                "fully_heldout_composition_gate_passed": row[
                    "fully_heldout_composition_gate_passed"
                ],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for model_name, row in summaries.items()
        },
        "automatic_next_decision": route,
        "interpretation_limits": prereg["interpretation_limits"],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", payload)
    print(
        f"Phase{protocol.PHASE} finalized: repeated={repeated} "
        f"models={repeated_models} route={route['route']}"
    )


if __name__ == "__main__":
    main()
