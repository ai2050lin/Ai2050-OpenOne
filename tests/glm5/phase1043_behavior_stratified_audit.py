#!/usr/bin/env python3
"""Post-hoc behavior-stratified audit of the Phase1043 negative result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import phase1043_late_readout_causal_protocol as protocol


EPS = 1e-8


def summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "median": (
            float(np.median(finite)) if finite.size else None
        ),
        "positive_rate": (
            float(np.mean(finite > 0)) if finite.size else None
        ),
    }


def main() -> None:
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    target_family = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_family = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    rows = np.arange(len(targets), dtype=np.int64)
    model_results = {}
    any_subset_gate_like = False
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        clean = np.load(
            atlas / "clean_target_candidate_logits.fp32.npy"
        )
        patched = np.load(
            atlas / "patched_candidate_logits.fp32.npy"
        )
        clean_margin = (
            clean[rows, cross_family]
            - clean[rows, target_family]
        )
        prediction = np.argmax(clean, axis=-1)
        masks = {
            "all": np.ones(len(targets), dtype=bool),
            "argmax_correct": prediction == target_family,
            "pair_correct": clean_margin < 0,
            "argmax_wrong": prediction != target_family,
        }
        candidate_rows = []
        for candidate_index in range(patched.shape[1]):
            shifts = []
            for intervention_index in range(patched.shape[2]):
                margin = (
                    patched[
                        rows,
                        candidate_index,
                        intervention_index,
                        cross_family,
                    ]
                    - patched[
                        rows,
                        candidate_index,
                        intervention_index,
                        target_family,
                    ]
                )
                shifts.append(margin - clean_margin)
            subset_rows = {}
            for subset, mask in masks.items():
                cross = shifts[0][mask]
                same = np.abs(shifts[1][mask])
                shuffled = np.abs(shifts[2][mask])
                full = shifts[4][mask]
                cross_summary = summary(cross)
                same_median = summary(same)["median"]
                shuffled_median = summary(shuffled)["median"]
                full_median = summary(full)["median"]
                cross_median = cross_summary["median"]
                ratio_same = (
                    abs(cross_median) / (same_median + EPS)
                    if cross_median is not None
                    and same_median is not None
                    else None
                )
                ratio_shuffled = (
                    abs(cross_median) / (shuffled_median + EPS)
                    if cross_median is not None
                    and shuffled_median is not None
                    else None
                )
                retention = (
                    cross_median / (full_median + EPS)
                    if cross_median is not None
                    and full_median is not None
                    and abs(full_median) > EPS
                    else None
                )
                gate_like = bool(
                    subset in {"argmax_correct", "pair_correct"}
                    and cross_median is not None
                    and cross_median > 0
                    and cross_summary["positive_rate"] >= 0.65
                    and ratio_same >= 1.5
                    and ratio_shuffled >= 1.5
                    and retention >= 0.05
                )
                any_subset_gate_like |= gate_like
                subset_rows[subset] = {
                    "row_count": int(np.sum(mask)),
                    "cross_shift": cross_summary,
                    "matched_to_same_absolute_ratio": ratio_same,
                    "matched_to_shuffled_absolute_ratio": ratio_shuffled,
                    "full_state_retention": retention,
                    "gate_like_without_group_median_check": gate_like,
                }
            candidate_rows.append({
                "candidate_index": candidate_index,
                "subsets": subset_rows,
            })
        model_results[model] = {
            "subset_counts": {
                name: int(np.sum(mask)) for name, mask in masks.items()
            },
            "candidate_rows": candidate_rows,
        }

    result = {
        "schema_version": "phase1043_behavior_stratified_audit.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_descriptive_only",
        "question": (
            "Could the preregistered negative result be explained solely "
            "by including behaviorally incorrect target prompts?"
        ),
        "model_results": model_results,
        "checks": {
            "all_three_models_present": (
                set(model_results) == set(protocol.MODELS)
            ),
            "no_correct_subset_gate_like_candidate": (
                not any_subset_gate_like
            ),
        },
        "conclusion": (
            "Behavioral errors reduce interpretability, but neither the "
            "argmax-correct nor pair-correct subset reveals a candidate "
            "that simultaneously meets the positive-rate, matched-control "
            "specificity, and full-state-retention thresholds. This audit "
            "does not replace the preregistered all-target result."
        ),
    }
    output = (
        protocol.OUT_ROOT
        / "posthoc"
        / "behavior_stratified_audit.json"
    )
    protocol.write_json(output, result)
    print(json.dumps(result["checks"], ensure_ascii=False, indent=2))
    print(json.dumps(result["conclusion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
