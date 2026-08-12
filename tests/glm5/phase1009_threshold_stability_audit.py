#!/usr/bin/env python3
"""Post-hoc threshold stability audit for Phase1009 repeated trajectories."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1009_crossfamily_response_finalize as finalize
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_jsonl,
    write_json,
)


THRESHOLDS = (0.85, 0.90, 0.95)


def motif_signature(row: dict) -> tuple:
    return (
        row["model"],
        row["family"],
        row["operation"],
        row["stage"],
        row["component"],
        row["role"],
    )


def cross_signature(row: dict) -> tuple:
    return (
        row["operation"],
        row["stage"],
        row["component"],
        row["role_class"],
        tuple(row["families"]),
        tuple(row["models"]),
        round(float(row["peak_relative_depth_median"]), 2),
    )


def run_threshold(threshold: float) -> tuple[dict, set[tuple], set[tuple]]:
    finalize.HIGH_FRACTION = threshold
    all_motifs = []
    for model_name in MODELS:
        for family in FAMILIES:
            root = OUT_ROOT / "scan" / model_name / family
            arrays = np.load(root / "response_scalars.npz")
            direction = np.load(root / "direction_consistency.npz")
            events = read_jsonl(root / "events.jsonl")
            units = read_jsonl(root / "units.jsonl")
            output_rows = read_jsonl(root / "output_pairs.jsonl")
            all_motifs.extend(finalize.trajectory_motifs(
                model_name=model_name,
                family=family,
                events=events,
                units=units,
                normalized=arrays["normalized_magnitude"],
                qualified=arrays["semantic_qualified"],
                rollout=arrays["rollout_qualified"],
                consistency=direction["direction_consistency"],
                outputs=finalize.output_arrays(units, output_rows),
            ))
    cross = finalize.cross_family_groups(all_motifs)
    eligible = {
        motif_signature(row)
        for row in all_motifs
        if row["refinement_eligible"]
    }
    strong = {
        cross_signature(row)
        for row in cross
        if row["strong_cross_family_cross_model"]
    }
    late_attention = [
        row for row in cross
        if row["strong_cross_family_cross_model"]
        and row["stage"] == "semantic0"
        and row["component"] == "attention_output"
        and row["operation"] in ("F", "Q", "FQ", "X")
        and float(row["peak_relative_depth_median"]) >= 2.0 / 3.0
    ]
    summary = {
        "high_fraction": threshold,
        "repeated_candidate_count": int(sum(
            row["repeated_candidate"] for row in all_motifs
        )),
        "refinement_eligible_count": len(eligible),
        "cross_family_motif_count": len(cross),
        "strong_cross_family_cross_model_count": len(strong),
        "late_semantic0_attention_candidate_count": len(late_attention),
        "late_semantic0_attention_operations": sorted({
            row["operation"] for row in late_attention
        }),
    }
    return summary, eligible, strong


def set_jaccard(left: set[tuple], right: set[tuple]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def main() -> None:
    runs = {}
    eligible_sets = {}
    strong_sets = {}
    for threshold in THRESHOLDS:
        summary, eligible, strong = run_threshold(threshold)
        key = f"{threshold:.2f}"
        runs[key] = summary
        eligible_sets[key] = eligible
        strong_sets[key] = strong
    common_eligible = set.intersection(*eligible_sets.values())
    common_strong = set.intersection(*strong_sets.values())
    result = {
        "schema_version": "phase1009_threshold_stability_audit.v1",
        "phase": PHASE,
        "posthoc_diagnostic": True,
        "thresholds": list(THRESHOLDS),
        "runs": runs,
        "eligible_common_all_thresholds": len(common_eligible),
        "strong_common_all_thresholds": len(common_strong),
        "eligible_0_85_vs_0_95_jaccard": set_jaccard(
            eligible_sets["0.85"],
            eligible_sets["0.95"],
        ),
        "strong_0_85_vs_0_95_jaccard": set_jaccard(
            strong_sets["0.85"],
            strong_sets["0.95"],
        ),
        "interpretation": (
            "This checks sensitivity to the trajectory peak-band threshold. "
            "It does not validate any motif or alter the frozen 0.90 atlas."
        ),
    }
    write_json(
        OUT_ROOT / "audit" / "threshold_stability.json",
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
