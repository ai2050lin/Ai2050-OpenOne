#!/usr/bin/env python3
"""Finalize Phase1084 after its preregistered behavior hard stop."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1084_two_entity_attribute_protocol as protocol


def main() -> None:
    protocol_root = protocol.OUT_ROOT / "protocol"
    analysis_root = protocol.OUT_ROOT / "analysis"
    prereg = protocol.read_json(protocol_root / "preregistration.json")
    static = protocol.read_json(protocol_root / "audit.json")
    pilots = {
        model: protocol.read_json(protocol.OUT_ROOT / "pilot" / f"{model}.json")
        for model in protocol.MODELS
    }
    passing_models = [
        model for model, row in pilots.items()
        if row["model_behavior_gate_passed"]
    ]
    model_rows = {}
    for model, row in pilots.items():
        operations = {}
        for operation in protocol.OPERATIONS:
            cells = [
                value for key, value in row["per_cell"].items()
                if key.startswith(f"{operation}__")
            ]
            operations[operation] = {
                "mean_candidate_accuracy": sum(
                    value["candidate_accuracy"] for value in cells
                ) / len(cells),
                "mean_generation_accuracy": sum(
                    value["generation_target_before_distractor_accuracy"]
                    for value in cells
                ) / len(cells),
                "passing_world_count": sum(int(value["passes"]) for value in cells),
                "passed": operation in row["passing_operations"],
            }
        model_rows[model] = {
            "model_behavior_gate_passed": row["model_behavior_gate_passed"],
            "passing_operations": row["passing_operations"],
            "passing_operation_count": row["passing_operation_count"],
            "candidate_finite_fraction": row["candidate_finite_fraction"],
            "elapsed_seconds": row["elapsed_seconds"],
            "operations": operations,
            "precision": row["precision"],
            "result_digest": row["result_digest"],
        }
    p1 = bool(static["all_checks_passed"])
    p2 = len(passing_models) >= int(
        prereg["evidence_thresholds"]["minimum_repeated_models_or_pairs"]
    )
    p9_behavior = all(
        row["candidate_finite_fraction"]
        >= prereg["evidence_thresholds"]["minimum_candidate_finite_fraction"]
        and row["precision"]["has_fp16_parameters"]
        and not row["precision"]["has_bf16_parameters"]
        and not row["precision"]["has_quantized_modules"]
        for row in pilots.values()
    )
    behavior = {
        "schema_version": "phase1084_behavior_audit.v1",
        "phase": protocol.PHASE,
        "passing_models": passing_models,
        "models": model_rows,
    }
    behavior["behavior_audit_digest"] = protocol.digest(behavior)
    prediction = {
        "schema_version": "phase1084_prediction_audit.v1",
        "phase": protocol.PHASE,
        "predictions": {
            "P1": {"passed": p1},
            "P2": {"passed": p2, "passing_models": passing_models},
            "P3": {"passed": None, "status": "not_run_due_to_P2_hard_stop"},
            "P4": {"passed": None, "status": "not_run_due_to_P2_hard_stop"},
            "P5": {"passed": None, "status": "not_run_due_to_P2_hard_stop"},
            "P6": {"passed": None, "status": "not_run_due_to_P2_hard_stop"},
            "P7": {"passed": None, "status": "not_run_due_to_P2_hard_stop"},
            "P8": {"passed": None, "status": "not_run_due_to_P2_hard_stop"},
            "P9": {
                "passed": None,
                "behavior_numeric_subgate_passed": p9_behavior,
                "status": "hidden_numeric_subgate_not_run_due_to_P2_hard_stop",
            },
        },
        "behavior_gate_passed": p2,
    }
    prediction["prediction_audit_digest"] = protocol.digest(prediction)
    automatic = {
        "schema_version": "phase1084_automatic_next.v1",
        "phase": protocol.PHASE,
        "decision": "continue_to_natural_direct_entity_behavior_revision",
        "hidden_scan_authorized": False,
        "full_atlas_authorized": False,
        "local_causal_authorized": False,
        "reason": (
            "Only Qwen3 passed at least six attributes. GLM4 and DS7B failures "
            "were concentrated in candidate relation accuracy rather than "
            "generation formatting, so the frozen two-model behavior gate failed."
        ),
        "next_protocol_constraints": [
            "preserve the two distinct entities and shared all-attribute dossier",
            "preserve the matched duplicate complete-profile control",
            "replace A/B indirection with a late natural selected-entity name",
            "use fresh phase and digest; never rewrite Phase1084 conclusions",
            "run all three behavior gates sequentially before hidden collection",
        ],
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)
    final = {
        "schema_version": "phase1084_behavior_stop_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "status": "stopped_before_hidden_scan_behavior_gate",
        "model_order": list(protocol.MODELS),
        "case_count_total": prereg["case_count_per_model"] * len(protocol.MODELS),
        "unit_count_total": prereg["unit_count_per_model"] * len(protocol.MODELS),
        "behavior": behavior,
        "predictions": prediction,
        "automatic_next": automatic,
        "interpretation_limits": prereg["interpretation_limits"] + [
            "No hidden-state response, component, head, neuron, or causal result was collected.",
            "A failed behavior gate diagnoses the protocol/model pair, not the absence of an attribute mechanism.",
        ],
    }
    final["summary_digest"] = protocol.digest(final)
    for filename, payload in (
        ("behavior_audit.json", behavior),
        ("prediction_audit.json", prediction),
        ("automatic_next.json", automatic),
        ("behavior_stop_summary.json", final),
    ):
        protocol.write_json(analysis_root / filename, payload)
    print({
        "phase": protocol.PHASE,
        "status": final["status"],
        "passing_models": passing_models,
        "decision": automatic["decision"],
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
