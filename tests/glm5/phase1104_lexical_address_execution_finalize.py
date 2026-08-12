#!/usr/bin/env python3
"""Assemble the frozen Phase1104 result and next-task decision."""

from __future__ import annotations

import json

import phase1104_lexical_address_execution_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    causal = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "causal_authorization.json"
    )
    behavior_summary_digests = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )["summary_digest"]
        for model in protocol.MODELS
    }
    behavior_pass = behavior["model_specific_causal_scan_authorized"]
    causal_pass = causal["component_localization_authorized"]
    cross_model_pass = causal.get("cross_model_mechanism_upgrade", False)
    prospective = {
        "P1": audit["all_checks_passed"],
        "P2": behavior_pass,
        "P3": True,
        "P4": bool(causal.get("model_cells")),
        "P5": causal_pass,
        "P6": causal_pass,
        "P7": causal_pass,
        "P8": True,
        "P9": True,
    }
    if causal_pass:
        next_task = (
            "Phase1105: localize the confirmed model-specific lexical routing "
            "interface into attention/MLP component coalitions; keep the "
            "semantic-equivalence ledger separate."
        )
        conclusion = (
            "At least one independently confirmed model/pair/surface cell "
            "passed cross-regime active-minus-congruent transport and matched "
            "controls. This is model-specific lexical execution evidence, not "
            "semantic-address or cross-model closure."
        )
    else:
        next_task = (
            "Phase1105: build a relation-specific natural paraphrase behavior "
            "calibration ledger before any semantic hidden-state intervention."
        )
        conclusion = (
            "No content-conditioned lexical execution cell passed all frozen "
            "confirmation and control gates. Raw effects, if present, remain "
            "selector-transport instrument checks."
        )
    result = {
        "schema_version": "phase1104_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "behavior_summary_digests": behavior_summary_digests,
        "behavior_authorization_digest": behavior["authorization_digest"],
        "causal_authorization_digest": causal["causal_authorization_digest"],
        "behavior": {
            "per_model": {
                model: {
                    "candidate_finite_fraction": row[
                        "candidate_finite_fraction"
                    ],
                    "candidate_accuracy": row["candidate_accuracy"],
                    "passing_pairs": row["model_specific_passing_pairs"],
                    "causal_selected_pairs": row["causal_selected_pairs"],
                }
                for model, row in behavior["models"].items()
            },
            "cross_model_behavior_pairs": behavior[
                "cross_model_behavior_pairs"
            ],
        },
        "causal": {
            "scan_authorized": causal["causal_scan_authorized"],
            "model_specific_confirmed_cells": causal[
                "model_specific_confirmed_cells"
            ],
            "cross_model_confirmed_cells": causal[
                "cross_model_confirmed_cells"
            ],
            "component_localization_authorized": causal_pass,
            "cross_model_mechanism_upgrade": cross_model_pass,
        },
        "prospective_predictions": prospective,
        "all_predictions_passed": all(prospective.values()),
        "frozen_conclusion": conclusion,
        "theory_status": {
            "lexical_address": (
                "Tested as a cross-key-regime, content-conditioned routing "
                "event rather than a fixed relation vector."
            ),
            "semantic_address": (
                "Not tested in Phase1104; paraphrase equivalence remains a "
                "separate behavior ledger."
            ),
            "coordinate_claim": (
                "Any pass is local to the selected residual interface and "
                "does not imply a global or cross-model coordinate."
            ),
            "compression_and_optimality": "Not tested.",
        },
        "phase1103_frozen_decision_unchanged": True,
        "automatic_next_required": True,
        "automatic_next_task": next_task,
    }
    result["final_summary_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json", result
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "behavior_pass": behavior_pass,
        "model_specific_causal_pass": causal_pass,
        "cross_model_mechanism_upgrade": cross_model_pass,
        "automatic_next_task": next_task,
        "final_summary_digest": result["final_summary_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
