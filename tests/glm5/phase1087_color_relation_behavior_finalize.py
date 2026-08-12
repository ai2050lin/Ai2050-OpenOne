#!/usr/bin/env python3
"""Freeze Phase1087 behavior authorization before hidden-state collection."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1087_color_relation_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    pilots = {
        model: protocol.read_json(protocol.OUT_ROOT / "pilot" / f"{model}.json")
        for model in protocol.MODELS
    }
    passing_models = [
        model for model, row in pilots.items()
        if row["model_behavior_gate_passed"]
    ]
    p1 = bool(static["all_checks_passed"])
    p2 = len(passing_models) >= int(
        prereg["evidence_thresholds"]["minimum_behavior_models"]
    )
    numeric_subgate = {
        model: (
            row["candidate_finite_fraction"]
            >= prereg["evidence_thresholds"]["minimum_candidate_finite_fraction"]
            and row["precision"]["has_fp16_parameters"]
            and not row["precision"]["has_bf16_parameters"]
            and not row["precision"]["has_quantized_modules"]
        )
        for model, row in pilots.items()
    }
    result = {
        "schema_version": "phase1087_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "predictions": {
            "P1": {"passed": p1},
            "P2": {"passed": p2, "passing_models": passing_models},
            "P10_behavior_numeric_subgate": {
                "passed_models": [
                    model for model, passed in numeric_subgate.items() if passed
                ],
                "failed_models": [
                    model for model, passed in numeric_subgate.items() if not passed
                ],
            },
        },
        "models": {
            model: {
                "passing_operations": row["passing_operations"],
                "passing_operation_count": row["passing_operation_count"],
                "model_behavior_gate_passed": row["model_behavior_gate_passed"],
                "candidate_finite_fraction": row["candidate_finite_fraction"],
                "elapsed_seconds": row["elapsed_seconds"],
                "precision": row["precision"],
                "result_digest": row["result_digest"],
            }
            for model, row in pilots.items()
        },
        "hidden_scan_authorized": p1 and p2,
        "full_atlas_authorized": False,
        "causal_authorized": False,
        "reason": (
            "Static and two-model behavior gates authorize only the frozen "
            "middle-band signed scan. A model with FP16 failures remains "
            "exploratory and cannot contribute to content-specific gates."
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result
    )
    print({
        "phase": protocol.PHASE,
        "passing_models": passing_models,
        "numeric_subgate": numeric_subgate,
        "hidden_scan_authorized": result["hidden_scan_authorized"],
        "authorization_digest": result["authorization_digest"],
    })


if __name__ == "__main__":
    main()
