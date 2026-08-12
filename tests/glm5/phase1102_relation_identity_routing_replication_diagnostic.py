#!/usr/bin/env python3
"""Create non-upgrading diagnostics for the Phase1101-1102 behavior chain."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

import phase1102_relation_identity_routing_replication_protocol as protocol


def aggregate_candidate_cells(model: str) -> dict[str, float]:
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "behavior" / model / "candidate_detail.jsonl"
    )
    totals = Counter()
    hits = Counter()
    finite = Counter()
    for row in rows:
        key = "|".join((
            str(row["surface"]), str(row["split"]),
            str(row["route_type"]), str(row["congruence"]),
        ))
        totals[key] += 1
        hits[key] += int(row["hit"])
        finite[key] += int(row["finite"])
    return {
        key: {
            "count": totals[key],
            "accuracy": hits[key] / totals[key],
            "finite_fraction": finite[key] / totals[key],
        }
        for key in sorted(totals)
    }


def main() -> None:
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    phase1101_revision1 = protocol.read_json(protocol.base.REVISION1_AUTHORIZATION)
    phase1101_revision2 = protocol.read_json(protocol.SOURCE_PHASE1101_AUTHORIZATION)
    model_details = {}
    for model in protocol.MODELS:
        current = authorization["models"][model]
        rev1 = phase1101_revision1["models"][model]
        rev2 = phase1101_revision2["models"][model]
        failed_current = {
            pair: row for pair, row in current["pair_results"].items()
            if not row["passed"]
        }
        failure_modes = {
            pair: (
                "finite_rate"
                if row["minimum_conflict_cell_finite_fraction"]
                < protocol.THRESHOLDS["minimum_candidate_finite_fraction"]
                else "accuracy"
            )
            for pair, row in failed_current.items()
        }
        model_details[model] = {
            "revision1": {
                "accuracy": rev1["candidate_accuracy"],
                "passing_pairs": rev1["passing_pairs"],
            },
            "revision2": {
                "accuracy": rev2["candidate_accuracy"],
                "passing_pairs": rev2["passing_pairs"],
            },
            "independent_replication": {
                "accuracy": current["candidate_accuracy"],
                "passing_pairs": current["passing_pairs"],
                "failed_pair_modes": failure_modes,
                "condition_cells": aggregate_candidate_cells(model),
            },
        }
    diagnostic = {
        "schema_version": "phase1102_diagnostic.v1",
        "phase": protocol.PHASE,
        "final_digest": final["final_digest"],
        "evidence_status": "non_upgrading_post_frozen_diagnostic",
        "model_details": model_details,
        "core_corrections_to_uploaded_reflections": [
            "Phase1100 had no behavior-geometry contradiction: a single visible relation plus max/min can be solved without distinguishing relation category identity.",
            "A zero or negative within-family permutation margin under the Phase1100 metric means a registered wrong assignment tied or beat identity; it is not a universal information-theoretic proof that relation identity is absent.",
            "The repeated coarse depth curve is a descriptive task/processing-phase topology, not yet a universal coarse-fine law or semantic interface.",
            "Information bottleneck and superposition remain candidate explanations. The present tests do not show that the network retains only one comparison bit.",
            "The suggested cross-dimensional gap comparison is ill-posed without a shared unit and scale. Phase1101 instead makes identity necessary by using two opposite-winner records and a late relation selector.",
            "The Phase1099-era proposal for unsupervised primitive discovery is now outdated and would still cluster carrier, template, and task-shell variance unless matched controls are built first.",
        ],
        "observed_behavior_structure": (
            "Relation words can act as contextual addresses: all three models exceed 0.93 aggregate semantic conflict-routing accuracy in the independent replication. The remaining hard cells are not shared by all models, and their identity changes across name worlds. This supports a reusable routing ability with model- and token-conditioned roughness, not a uniform 15-relation semantic coordinate system."
        ),
        "hard_limit": (
            "Because the 13-of-15 worst-cell gate failed in every model, no hidden-state object was measured. The study therefore cannot locate the routing mechanism, compare it with lexical embeddings, claim relative-code inheritance, or perform causal closure."
        ),
        "automatic_next_assessment": (
            "No automatic continuation is justified. A third prompt repair, threshold relaxation, or post-hoc subset scan would optimize the experiment around observed failures. The next phase must change the observational setting to a natural continuation task and be separately preregistered."
        ),
    }
    diagnostic["diagnostic_digest"] = protocol.digest(diagnostic)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json", diagnostic
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "automatic_next": False,
        "diagnostic_digest": diagnostic["diagnostic_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
