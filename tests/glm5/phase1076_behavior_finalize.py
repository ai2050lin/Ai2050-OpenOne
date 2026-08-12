#!/usr/bin/env python3
"""Finalize the frozen Phase1076 behavior authorization."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1076_polarity_head_causal_protocol as protocol


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
    authorized_models = [
        model
        for model, summary in summaries.items()
        if summary["model_behavior_gate_passed"]
    ]
    should_run = (
        len(authorized_models)
        >= prereg["gates"]["minimum_behavior_models"]
        and set(authorized_models) == set(protocol.MODELS)
    )
    payload = {
        "schema_version": "phase1076_behavior_decision.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "authorized_models": authorized_models,
        "should_run_causal_validation": should_run,
        "route": (
            "run_frozen_head_output_causal_validation"
            if should_run
            else "stop_at_behavior_gate"
        ),
        "reason": (
            "Both independently repeated models passed the polarity "
            "and surface-control behavior gates."
            if should_run
            else (
                "The preregistered two-model behavior gate failed; "
                "head-output causal claims are not authorized."
            )
        ),
        "model_summary": {
            model: {
                "overall_candidate_finite_rate": summary[
                    "overall_candidate_finite_rate"
                ],
                "overall_candidate_accuracy": summary[
                    "overall_candidate_accuracy"
                ],
                "model_behavior_gate_passed": summary[
                    "model_behavior_gate_passed"
                ],
                "contrasts": {
                    contrast: {
                        "candidate_finite_rate": value[
                            "candidate_finite_rate"
                        ],
                        "candidate_accuracy": value[
                            "candidate_accuracy"
                        ],
                        "by_task": value["by_task"],
                        "by_path": value["by_path"],
                        "natural_semantic_first_rate": value[
                            "natural_semantic_first_rate"
                        ],
                        "gate_passed": value[
                            "contrast_behavior_gate_passed"
                        ],
                    }
                    for contrast, value in summary[
                        "contrasts"
                    ].items()
                },
            }
            for model, summary in summaries.items()
        },
    }
    payload["decision_digest"] = protocol.digest(payload)
    protocol.write_json(
        protocol.OUT_ROOT
        / "analysis"
        / "behavior_decision.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
