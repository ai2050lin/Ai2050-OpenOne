#!/usr/bin/env python3
"""Create a frozen, non-upgrading Phase1101 failure/success diagnostic."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import phase1101_relation_identity_routing_protocol as protocol


def main() -> None:
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    thresholds = protocol.THRESHOLDS
    records = []
    failure_reasons = Counter()
    strongest_controls = Counter()
    replications = {field: [] for field in protocol.FIELDS}
    selected_events = []
    for model in protocol.MODELS:
        for surface in protocol.SURFACES:
            row = final["models"][model]["surfaces"][surface]
            confirmation = row["confirmation_record"]
            reasons = []
            checks = {
                "raw_cosine": confirmation["identity_score"] >= thresholds["minimum_inheritance_cosine"],
                "family_margin": confirmation["family_permutation_margin"] >= thresholds["minimum_family_permutation_margin"],
                "within_family_margin": confirmation["within_family_permutation_margin"] >= thresholds["minimum_within_family_permutation_margin"],
                "specificity": confirmation["semantic_specificity_advantage"] >= thresholds["minimum_specificity_advantage"],
            }
            for key, passed in checks.items():
                if not passed:
                    reasons.append(key)
                    failure_reasons[key] += 1
            strongest = max(
                confirmation["controls"], key=lambda value: value["maximum_alignment"]
            )["control"]
            strongest_controls[strongest] += 1
            for field, value in row["target_graph_replication_by_field"].items():
                replications[field].append(float(value))
            selected_events.append({
                "model": model,
                "surface": surface,
                **row["selected_event"],
            })
            records.append({
                "model": model,
                "surface": surface,
                "formal": model in protocol.FORMAL_MODELS,
                "behavior_passed": behavior["models"][model]["model_behavior_passed"],
                "identity_score": confirmation["identity_score"],
                "family_margin": confirmation["family_permutation_margin"],
                "within_family_margin": confirmation["within_family_permutation_margin"],
                "specificity_advantage": confirmation["semantic_specificity_advantage"],
                "strongest_control": strongest,
                "inheritance_pass": confirmation["inheritance_pass"],
                "specificity_pass": confirmation["specificity_pass"],
                "failure_reasons": reasons,
            })
    formal = [row for row in records if row["formal"]]
    raw_nonidentifying = sum(
        row["identity_score"] >= thresholds["minimum_inheritance_cosine"]
        and not row["inheritance_pass"] for row in formal
    )
    diagnostic = {
        "schema_version": "phase1101_diagnostic.v1",
        "phase": protocol.PHASE,
        "final_digest": final["final_digest"],
        "evidence_status": "non_upgrading_post_frozen_diagnostic",
        "behaviorally_necessary_identity_task": True,
        "records": records,
        "formal_cell_count": len(formal),
        "formal_raw_high_but_nonidentifying_count": raw_nonidentifying,
        "failure_reason_counts_formal": dict(Counter(
            reason for row in formal for reason in row["failure_reasons"]
        )),
        "failure_reason_counts_all": dict(failure_reasons),
        "strongest_control_counts": dict(strongest_controls),
        "target_graph_replication_summary": {
            field: {
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
            for field, values in replications.items()
        },
        "selected_events": selected_events,
        "corrected_phase1100_question": "Phase1101 makes relation identity behaviorally necessary by placing two contradictory relation records in one fixed prefix and selecting one relation only at the query. This removes the Phase1099/1100 loophole in which a generic max/min endpoint operation could solve every item without distinguishing relation identities.",
        "interpretive_boundary": (
            "Passing behavior establishes relation-address use, not human-like semantic understanding. Passing target-graph repetition establishes stable internal relation-routing geometry. Only source-target identity, exact permutation, and matched-control specificity together support lexical inheritance."
        ),
        "next_decision": (
            "Run a new-data causal interface phase."
            if final["automatic_next_required"]
            else "Do not tune the same Gram gate or select neurons. Preserve the physical map and change the observational object only if a new protocol separates lexical key matching from natural relation semantics."
        ),
    }
    diagnostic["diagnostic_digest"] = protocol.digest(diagnostic)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json", diagnostic
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "formal_raw_high_but_nonidentifying_count": raw_nonidentifying,
        "failure_reason_counts_formal": diagnostic["failure_reason_counts_formal"],
        "diagnostic_digest": diagnostic["diagnostic_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
