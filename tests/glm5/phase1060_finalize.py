#!/usr/bin/env python3
"""Aggregate the Phase1060 factorial repair."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1060_lexicon_template_factorial_protocol as protocol


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
        if row["new_lexicon_old_template_gate_passed"]
    ]
    repeated = len(passing) >= prereg["gates"]["minimum_repeated_models"]
    route = {
        "route": (
            "phase1061_sentence_role_transport"
            if repeated
            else "stop_with_lexicon_or_protocol_limit"
        ),
        "should_continue_automatically": repeated,
        "rationale": (
            "The new lexicon transported under the established templates "
            "in at least two models, so sentence-role transport is the "
            "predeclared next complexity step."
            if repeated
            else
            "The orthogonal repair did not yield two behavior-qualified "
            "models with new-lexicon transport; sentence extension stops."
        ),
    }
    payload = {
        "schema_version": "phase1060_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "new_lexicon_old_template_passing_models": passing,
        "cross_model_repetition": repeated,
        "model_results": {
            model_name: {
                "precision": row["precision"],
                "exact_case_counts_by_cell": row[
                    "exact_case_counts_by_cell"
                ],
                "exact_case_rates_by_cell": row[
                    "exact_case_rates_by_cell"
                ],
                "cell_behavior_passed": row["cell_behavior_passed"],
                "behavior_factor_contrasts": row[
                    "behavior_factor_contrasts"
                ],
                "valid_pair_counts": row["valid_pair_counts"],
                "component_rates": row["component_rates"],
                "phase_rates": row["phase_rates"],
                "phase_class": row["phase_class"],
                "channel_rates": row["channel_rates"],
                "support_map": row["support_map"],
                "maximum_role_control_rate": row[
                    "maximum_role_control_rate"
                ],
                "new_lexicon_old_template_gate_passed": row[
                    "new_lexicon_old_template_gate_passed"
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
