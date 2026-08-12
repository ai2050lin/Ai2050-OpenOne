#!/usr/bin/env python3
"""Aggregate decoded-text transport without hiding token results."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1062_text_equivalence_protocol as protocol


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
    passing = [
        model_name
        for model_name, row in summaries.items()
        if row["text_equivalence_transport_gate_passed"]
    ]
    repeated = len(passing) >= prereg["gates"]["minimum_repeated_models"]
    route = {
        "route": (
            "phase1063_sentence_role_transport"
            if repeated
            else "stop_with_real_lexical_behavior_limit"
        ),
        "should_continue_automatically": repeated,
        "rationale": (
            "Frozen decoded-text equivalence yielded behavior-qualified "
            "donor-clean transport in at least two models."
            if repeated
            else
            "After decoded-text repair, fewer than two models passed the "
            "held-out lexical transport gate; sentence extension stops."
        ),
    }
    payload = {
        "schema_version": "phase1062_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "passing_models": passing,
        "cross_model_repetition": repeated,
        "model_results": {
            model_name: {
                "precision": row["precision"],
                "behavior_summary": row["behavior_summary"],
                "valid_pair_counts": row["valid_pair_counts"],
                "behavior_gate_passed": row["behavior_gate_passed"],
                "component_text_rates": row["component_text_rates"],
                "component_token_rates": row["component_token_rates"],
                "phase_text_rates": row["phase_text_rates"],
                "phase_token_rates": row["phase_token_rates"],
                "phase_class": row["phase_class"],
                "channel_text_rates": row["channel_text_rates"],
                "support_map": row["support_map"],
                "maximum_role_control_text_rate": row[
                    "maximum_role_control_text_rate"
                ],
                "text_equivalence_transport_gate_passed": row[
                    "text_equivalence_transport_gate_passed"
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
        f"models={passing} route={route['route']}"
    )


if __name__ == "__main__":
    main()
