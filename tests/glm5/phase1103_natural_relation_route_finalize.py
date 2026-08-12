#!/usr/bin/env python3
"""Assemble the frozen Phase1103 result without changing any gate."""

from __future__ import annotations

import json

import phase1103_natural_relation_route_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    causal = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "causal_authorization.json"
    )
    behavior_summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    causal_summaries = {}
    for model in protocol.MODELS:
        path = protocol.OUT_ROOT / "causal" / model / "summary.json"
        if path.exists():
            causal_summaries[model] = protocol.read_json(path)
    diagnostic_path = (
        protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json"
    )
    diagnostic = (
        protocol.read_json(diagnostic_path)
        if diagnostic_path.exists() else None
    )
    predictions = {
        "P1": protocol_audit["all_checks_passed"],
        "P2": behavior["causal_scan_authorized"],
        "P3": all(
            set(summary.get("eligible_pairs", []))
            <= set(behavior["causally_eligible_pairs"])
            for summary in causal_summaries.values()
        ),
        "P4": any(
            row["confirmation_gates"]["median_recovery"]
            for row in causal.get("model_cells", {}).values()
        ),
        "P5": any(
            row["confirmation_gates"]["specificity"]
            and row["confirmation_gates"]["positive_fraction"]
            for row in causal.get("model_cells", {}).values()
        ),
        "P6": any(
            row["confirmation_gates"]["flip_rate"]
            and row["confirmation_gates"]["congruent_collateral"]
            for row in causal.get("model_cells", {}).values()
        ),
        "P7": causal["component_scan_authorized"],
        "P8": (
            (behavior["causal_scan_authorized"] or not causal_summaries)
            and (
                causal["component_scan_authorized"]
                or causal["decision"]
                in (
                    "retain_pair_specific_response_map_without_mechanism_closure",
                    "stop_at_behavior_gate",
                )
            )
        ),
    }
    if not behavior["causal_scan_authorized"]:
        frozen_conclusion = (
            "The natural exact/paraphrase/ordinal interface did not yield a "
            "shared prospectively authorized relation pair. Hidden-state "
            "access stopped, so no coding-mechanism claim is permitted."
        )
    elif causal["component_scan_authorized"]:
        frozen_conclusion = (
            "At least one pair-surface cell independently transported late "
            "relation selection in two models while beating frozen controls. "
            "This authorizes component localization, not full mechanism closure."
        )
    else:
        frozen_conclusion = (
            "Natural relation routing produced pair-specific behavior and a "
            "signed residual response map, but no pair-surface cell passed the "
            "two-model independent causal/control gate. The map remains "
            "descriptive rather than a semantic transport mechanism."
        )
    result = {
        "schema_version": "phase1103_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "behavior_summary_digests": {
            model: row["summary_digest"]
            for model, row in behavior_summaries.items()
        },
        "behavior_authorization_digest": behavior[
            "authorization_digest"
        ],
        "causal_summary_digests": {
            model: row["summary_digest"]
            for model, row in causal_summaries.items()
        },
        "causal_authorization_digest": causal[
            "causal_authorization_digest"
        ],
        "failure_diagnostic_digest": (
            diagnostic["diagnostic_digest"] if diagnostic else None
        ),
        "behavior": {
            "per_model": {
                model: {
                    "candidate_finite_fraction": row[
                        "candidate_finite_fraction"
                    ],
                    "candidate_accuracy": row["candidate_accuracy"],
                    "generation_accuracy": row[
                        "generation_target_before_distractor_accuracy"
                    ],
                    "passing_pair_count": behavior["models"][model][
                        "passing_pair_count"
                    ],
                }
                for model, row in behavior_summaries.items()
            },
            "shared_behavior_authorized_pairs": behavior[
                "shared_behavior_authorized_pairs"
            ],
            "causally_eligible_pairs": behavior[
                "causally_eligible_pairs"
            ],
        },
        "causal": {
            "scan_authorized": causal["causal_scan_authorized"],
            "passing_model_cell_count": causal.get(
                "passing_model_cell_count", 0
            ),
            "shared_confirmed_cells": causal[
                "shared_confirmed_cells"
            ],
            "component_scan_authorized": causal[
                "component_scan_authorized"
            ],
        },
        "prospective_predictions": predictions,
        "all_predictions_passed": all(predictions.values()),
        "frozen_conclusion": frozen_conclusion,
        "theory_status": {
            "behavior_level": (
                "A relation phrase is treated as a contextual address only "
                "for the prospectively passing pairs."
            ),
            "coordinate_level": (
                "Phase1103 tests a local signed difference at a causal "
                "interface; it does not assume globally fixed coordinates."
            ),
            "compression_hypothesis": (
                "Not tested. Compression remains a candidate formation story, "
                "not an inference from transport or its failure."
            ),
            "optimality_hypothesis": (
                "Not tested because no architecture, training, capacity, or "
                "resource-matched comparison is present."
            ),
        },
        "automatic_next": causal["component_scan_authorized"],
        "automatic_next_task": (
            "Phase1104 preregistered attention/MLP decomposition of only the "
            "shared confirmed Phase1103 cells"
            if causal["component_scan_authorized"] else None
        ),
    }
    if diagnostic is not None:
        result["descriptive_route_pair_counts"] = {
            model: row["route_pair_counts_passing_both_splits"]
            for model, row in diagnostic["models"].items()
        }
    result["final_summary_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json", result
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "shared_behavior_pairs": result["behavior"][
            "shared_behavior_authorized_pairs"
        ],
        "shared_confirmed_causal_cells": result["causal"][
            "shared_confirmed_cells"
        ],
        "automatic_next": result["automatic_next"],
        "final_summary_digest": result["final_summary_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
