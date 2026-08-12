#!/usr/bin/env python3
"""Aggregate Phase1058 without promoting descriptive results to theory."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1058_multitoken_translation_protocol as protocol


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
    parity_audits = {
        model_name: protocol.read_json(
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "cache_parity_audit.json"
        )
        for model_name in protocol.MODELS
    }
    for model_name, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(
                f"protocol digest drift for {model_name}"
            )
        parity = parity_audits[model_name]
        if parity["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(
                f"parity protocol digest drift for {model_name}"
            )
        if not parity["passed"]:
            raise RuntimeError(
                f"full-sequence cache parity failed for {model_name}"
            )
    behavior_models = [
        name for name, row in summaries.items()
        if row["behavior_gate_passed"]
    ]
    composition_models = [
        name for name, row in summaries.items()
        if row["composition_gate_passed"]
    ]
    repeated = (
        len(composition_models)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    common_phase_classes = {
        name: row["phase_class"]
        for name, row in summaries.items()
        if row["behavior_gate_passed"]
    }
    aggregate = {
        "schema_version": "phase1058_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "behavior_passing_models": behavior_models,
        "composition_passing_models": composition_models,
        "cross_model_compositional_repetition": repeated,
        "phase_classes": common_phase_classes,
        "model_results": {
            name: {
                key: row[key]
                for key in (
                    "precision",
                    "clean_behavior",
                    "valid_pair_counts",
                    "behavior_gate_passed",
                    "cache_parity",
                    "cache_parity_passed",
                    "component_rates",
                    "phase_rates",
                    "phase_class",
                    "channel_rates",
                    "frozen_rectangle_rate",
                    "maximum_role_control_rate",
                    "composition_gate_passed",
                    "elapsed_seconds",
                )
            }
            for name, row in summaries.items()
        },
        "full_recompute_cache_parity": {
            name: {
                key: value for key, value in row.items()
                if key not in ("placement", "records")
            }
            for name, row in parity_audits.items()
        },
        "automatic_next_decision": {
            "route": "stop_at_compositional_translation_milestone",
            "should_continue_automatically": False,
            "rationale": prereg["automatic_next"]["decision"],
        },
        "interpretation_limits": prereg["interpretation_limits"],
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    print(
        f"Phase{protocol.PHASE} behavior={behavior_models} "
        f"composition={composition_models} repeated={repeated} "
        "next=stop_at_compositional_translation_milestone"
    )


if __name__ == "__main__":
    main()
