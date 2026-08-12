#!/usr/bin/env python3
"""Finalize the Phase1074 cross-model behavior gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1074_polarity_dynamics_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "summary.json"
        )
        for model in protocol.MODELS
    }
    selected = [
        model
        for model, summary in summaries.items()
        if summary["model_behavior_gate_passed"]
    ]
    repeated_relations = {}
    for relation in protocol.RELATIONS:
        models = [
            model
            for model, summary in summaries.items()
            if summary["relations"][relation][
                "strong_relation_gate_passed"
            ]
        ]
        repeated_relations[relation] = models
    should_continue = (
        len(selected)
        >= prereg["gates"]["minimum_behavior_models"]
    )
    payload = {
        "schema_version": "phase1074_behavior_decision.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selected_models": selected,
        "selected_model_count": len(selected),
        "repeated_relations": repeated_relations,
        "should_run_internal_dynamics": should_continue,
        "route": (
            "run_late_polarity_dynamics"
            if should_continue
            else "stop_at_behavior_foundation"
        ),
        "reason": (
            "At least two models passed the frozen behavior gate."
            if should_continue
            else (
                "Fewer than two models passed the frozen behavior gate; "
                "an internal operation-selection interpretation would be "
                "invalid."
            )
        ),
        "model_summaries": {
            model: {
                "candidate_accuracy": summary[
                    "candidate_accuracy"
                ],
                "by_task": summary["by_task"],
                "by_path": summary["by_path"],
                "by_split": summary["by_split"],
                "natural_semantic_first_rate": summary[
                    "natural_semantic_first_rate"
                ],
                "strong_relations": summary["strong_relations"],
                "model_behavior_gate_passed": summary[
                    "model_behavior_gate_passed"
                ],
            }
            for model, summary in summaries.items()
        },
    }
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "behavior_decision.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
