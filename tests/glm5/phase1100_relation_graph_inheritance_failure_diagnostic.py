#!/usr/bin/env python3
"""Post-hoc, non-upgrading diagnosis of the Phase1100 failed inheritance gates."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import phase1100_relation_graph_inheritance_finalize as finalize
import phase1100_relation_graph_inheritance_protocol as protocol


def rank_strict(identity: float, alternatives: np.ndarray, tolerance: float = 1e-10) -> int:
    return 1 + int(np.sum(alternatives > identity + tolerance))


def main() -> None:
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    targets = {model: finalize.load_target(model) for model in protocol.MODELS}
    banks = {model: finalize.source_graphs(model) for model in protocol.MODELS}
    records = []
    control_winners = Counter()
    for model in protocol.MODELS:
        for surface in protocol.SURFACES:
            event = int(final["models"][model]["surfaces"][surface]["selected_event"]["event_index"])
            for sample_split, relation_split in (
                ("discovery", "discovery"),
                ("confirmation", "discovery"),
                ("discovery", "confirmation"),
                ("confirmation", "confirmation"),
            ):
                row = finalize.evaluate_cell(banks[model], targets[model], surface, sample_split, relation_split, event)
                source_bank = banks[model][(surface, relation_split, protocol.PRIMARY_SOURCE)]
                target_graph = finalize.target_graph(
                    targets[model], surface, sample_split, event,
                    protocol.PRIMARY_TARGET_FIELD, protocol.PRIMARY_TARGET_ROLE, relation_split,
                )
                target_vector, _ = finalize.graph_vector(target_graph)
                identity = float(source_bank["identity"] @ target_vector)
                family_scores = source_bank["family"] @ target_vector
                within_scores = source_bank["within"] @ target_vector
                winner = max(row["controls"], key=lambda value: value["maximum_alignment"])
                control_winners[str(winner["control"])] += 1
                records.append(
                    {
                        "model": model,
                        "behavior_formal": bool(final["models"][model]["behavior_formal"]),
                        "surface": surface,
                        "sample_split": sample_split,
                        "relation_split": relation_split,
                        "event_index": event,
                        "identity_score": identity,
                        "family_rank_among_120": rank_strict(identity, family_scores),
                        "within_family_rank_among_7776": rank_strict(identity, within_scores),
                        "family_permutation_margin": row["family_permutation_margin"],
                        "within_family_permutation_margin": row["within_family_permutation_margin"],
                        "execution_specificity_advantage": row["execution_specificity_advantage"],
                        "winning_control": winner,
                        "high_similarity_but_nonidentifying": bool(
                            identity >= protocol.THRESHOLDS["minimum_inheritance_cosine"]
                            and row["within_family_permutation_margin"] < protocol.THRESHOLDS["minimum_within_family_permutation_margin"]
                        ),
                    }
                )

    formal = [row for row in records if row["behavior_formal"]]
    all_rows = records

    def summarize(rows):
        return {
            "cell_count": len(rows),
            "raw_cosine_passes": sum(row["identity_score"] >= protocol.THRESHOLDS["minimum_inheritance_cosine"] for row in rows),
            "family_margin_passes": sum(row["family_permutation_margin"] >= protocol.THRESHOLDS["minimum_family_permutation_margin"] for row in rows),
            "within_family_margin_passes": sum(row["within_family_permutation_margin"] >= protocol.THRESHOLDS["minimum_within_family_permutation_margin"] for row in rows),
            "specificity_advantage_passes": sum(row["execution_specificity_advantage"] >= protocol.THRESHOLDS["minimum_execution_specificity_advantage"] for row in rows),
            "high_similarity_but_nonidentifying": sum(row["high_similarity_but_nonidentifying"] for row in rows),
            "negative_specificity_cells": sum(row["execution_specificity_advantage"] < 0.0 for row in rows),
            "median_identity_score": float(np.median([row["identity_score"] for row in rows])),
            "median_family_rank": float(np.median([row["family_rank_among_120"] for row in rows])),
            "median_within_family_rank": float(np.median([row["within_family_rank_among_7776"] for row in rows])),
            "maximum_within_family_margin": float(max(row["within_family_permutation_margin"] for row in rows)),
            "median_specificity_advantage": float(np.median([row["execution_specificity_advantage"] for row in rows])),
        }

    result = {
        "schema_version": "phase1100_failure_diagnostic.v1",
        "phase": protocol.PHASE,
        "post_hoc_only": True,
        "cannot_upgrade_registered_gates": True,
        "final_digest": final["final_digest"],
        "formal_summary": summarize(formal),
        "all_model_summary": summarize(all_rows),
        "control_winner_counts": dict(sorted(control_winners.items())),
        "cross_model_curve_passes": sum(row["passed"] for row in final["cross_model_functional_trajectories"]),
        "cross_model_curve_cells": len(final["cross_model_functional_trajectories"]),
        "records": records,
        "interpretation": {
            "survives": "The coarse lexical-to-hidden graph similarity and its depth trajectory repeat, including five of six Qwen3/GLM4 trajectory cells.",
            "fails": "Correct relation identity is not selected over exact within-family relabelings, and relational execution is not stronger than matched lookup/carrier/form controls.",
            "boundary": "The result maps a generic task/lexical processing phase, not a semantic inheritance interface or a computation primitive.",
        },
    }
    result["diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json", result)
    print(json.dumps({"phase": protocol.PHASE, "formal_summary": result["formal_summary"], "diagnostic_digest": result["diagnostic_digest"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
