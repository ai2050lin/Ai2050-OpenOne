#!/usr/bin/env python3
"""Summarize the frozen behavior-stratified Phase1016 rescan."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1016_query_factorial_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


TARGET_ROOT = OUT_ROOT / "targeted_behavior_scan"
ANALYSIS_ROOT = OUT_ROOT / "analysis"


def strict_trace(row: dict[str, Any]) -> bool:
    return bool(
        row["n"] >= 8
        and row["semantic_direction_consistency"] is not None
        and row["semantic_direction_consistency"] >= 0.45
        and row["lexical_family_direction_alignment"] is not None
        and row["lexical_family_direction_alignment"] >= 0.40
        and row["semantic_over_lexical_prevalence"] is not None
        and row["semantic_over_lexical_prevalence"] >= 0.70
    )


def finite_median(values: list[float | None]) -> float | None:
    finite = [
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(np.median(finite))


def main() -> None:
    selection = read_json(TARGET_ROOT / "selection.json")
    all_rows = []
    model_summaries = {}
    for model_name in MODELS:
        summary = read_json(TARGET_ROOT / model_name / "summary.json")
        if summary["selection_digest"] != selection["selection_digest"]:
            raise RuntimeError(f"{model_name}: selection digest drift")
        model_summaries[model_name] = summary
        all_rows.extend(read_jsonl(
            TARGET_ROOT / model_name / "direction_results.jsonl"
        ))

    confirmation_correct = [
        row for row in all_rows
        if row["split"] == "confirmation"
        and row["population"] == "factorial_correct"
        and row["family_scope"] != "all_families"
    ]
    confirmation_failed = {
        (
            row["model"],
            row["event_id"],
            row["role"],
            row["family_scope"],
        ): row
        for row in all_rows
        if row["split"] == "confirmation"
        and row["population"] == "factorial_failed"
        and row["family_scope"] != "all_families"
    }
    qualified_rows = []
    pair_rows = []
    for correct in confirmation_correct:
        key = (
            correct["model"],
            correct["event_id"],
            correct["role"],
            correct["family_scope"],
        )
        failed = confirmation_failed.get(key)
        correct_pass = strict_trace(correct)
        failed_pass = bool(failed is not None and strict_trace(failed))
        if correct_pass:
            qualified_rows.append({
                **correct,
                "schema_version": (
                    "phase1016_targeted_heldout_trace.v1"
                ),
                "failed_population_also_passes": failed_pass,
                "behavior_specific_trace": bool(
                    correct_pass and not failed_pass
                ),
            })
        if (
            failed is not None
            and correct["n"] >= 4
            and failed["n"] >= 4
            and correct["semantic_direction_consistency"] is not None
            and failed["semantic_direction_consistency"] is not None
        ):
            pair_rows.append({
                "schema_version": (
                    "phase1016_correct_failed_trace_pair.v1"
                ),
                "phase": PHASE,
                "model": correct["model"],
                "event_id": correct["event_id"],
                "component": correct["component"],
                "depth": correct["depth"],
                "head": correct["head"],
                "role": correct["role"],
                "family": correct["family_scope"],
                "correct_n": correct["n"],
                "failed_n": failed["n"],
                "correct_direction_consistency": correct[
                    "semantic_direction_consistency"
                ],
                "failed_direction_consistency": failed[
                    "semantic_direction_consistency"
                ],
                "correct_lexical_alignment": correct[
                    "lexical_family_direction_alignment"
                ],
                "failed_lexical_alignment": failed[
                    "lexical_family_direction_alignment"
                ],
                "correct_semantic_median": correct["semantic_median"],
                "failed_semantic_median": failed["semantic_median"],
                "correct_failed_mean_direction_cosine": correct[
                    "correct_failed_mean_direction_cosine"
                ],
                "correct_passes_trace_gate": correct_pass,
                "failed_passes_trace_gate": failed_pass,
            })
    write_jsonl(
        ANALYSIS_ROOT / "targeted_heldout_correct_traces.jsonl",
        qualified_rows,
    )
    write_jsonl(
        ANALYSIS_ROOT / "targeted_correct_failed_pairs.jsonl",
        pair_rows,
    )

    by_model = {}
    for model_name in MODELS:
        model_qualified = [
            row for row in qualified_rows
            if row["model"] == model_name
        ]
        model_pairs = [
            row for row in pair_rows
            if row["model"] == model_name
        ]
        by_model[model_name] = {
            "selected_event_role_count": int(
                model_summaries[model_name]["selection_count"]
            ),
            "factorial_correct_count": int(
                model_summaries[model_name]["factorial_correct_count"]
            ),
            "heldout_correct_trace_count": len(model_qualified),
            "heldout_correct_trace_by_family": dict(Counter(
                row["family_scope"] for row in model_qualified
            )),
            "heldout_correct_trace_by_role": dict(Counter(
                row["role"] for row in model_qualified
            )),
            "behavior_specific_trace_count": sum(
                row["behavior_specific_trace"]
                for row in model_qualified
            ),
            "trace_also_present_in_failed_count": sum(
                row["failed_population_also_passes"]
                for row in model_qualified
            ),
            "correct_failed_pair_count": len(model_pairs),
            "correct_direction_consistency_median": finite_median([
                row["correct_direction_consistency"]
                for row in model_pairs
            ]),
            "failed_direction_consistency_median": finite_median([
                row["failed_direction_consistency"]
                for row in model_pairs
            ]),
            "correct_lexical_alignment_median": finite_median([
                row["correct_lexical_alignment"]
                for row in model_pairs
            ]),
            "failed_lexical_alignment_median": finite_median([
                row["failed_lexical_alignment"]
                for row in model_pairs
            ]),
            "correct_semantic_magnitude_median": finite_median([
                row["correct_semantic_median"]
                for row in model_pairs
            ]),
            "failed_semantic_magnitude_median": finite_median([
                row["failed_semantic_median"]
                for row in model_pairs
            ]),
            "correct_failed_mean_direction_cosine_median": finite_median([
                row["correct_failed_mean_direction_cosine"]
                for row in model_pairs
            ]),
            "correct_direction_more_consistent_count": sum(
                row["correct_direction_consistency"]
                > row["failed_direction_consistency"]
                for row in model_pairs
            ),
            "correct_magnitude_larger_count": sum(
                row["correct_semantic_median"]
                > row["failed_semantic_median"]
                for row in model_pairs
            ),
        }

    summary = {
        "schema_version": "phase1016_targeted_analysis_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "selection_digest": selection["selection_digest"],
        "selection_count": selection["selection_count"],
        "selection_used_discovery_only": bool(
            not selection["confirmation_metrics_used"]
        ),
        "selection_used_behavior": bool(
            selection["behavior_labels_used"]
        ),
        "model_summaries": by_model,
        "heldout_correct_trace_count": len(qualified_rows),
        "behavior_specific_trace_count": sum(
            row["behavior_specific_trace"] for row in qualified_rows
        ),
        "trace_also_present_in_failed_count": sum(
            row["failed_population_also_passes"]
            for row in qualified_rows
        ),
        "interpretation": {
            "supported": (
                "Within individual language families, frozen physical "
                "components repeatedly carry synonym-invariant query "
                "differences on heldout templates."
            ),
            "not_supported": (
                "A single invariant vector across language families, or a "
                "trace that uniquely separates correct from failed behavior."
            ),
            "best_current_model": (
                "Physical computation resources are reused across patterns, "
                "while their active directions remain pattern-conditioned."
            ),
        },
        "automatic_continuation_decision": {
            "continue_to_neuron_localization": False,
            "continue_to_causal_closure": False,
            "reason": (
                "The stable trace is largely present in failed computation, "
                "so it is not yet a behavior-specific decision mechanism."
            ),
            "next_large_task": (
                "Build matched correct-versus-error divergence maps after "
                "conditioning out the shared query trace, then search for "
                "answer-selection state changes rather than larger "
                "activations."
            ),
        },
        "hard_limits": [
            "Targeted scan used four-state batches; singleton and batch "
            "behavior counts differ slightly for Qwen3 and GLM4.",
            "Some family-conditioned correct subsets are small, especially "
            "for DS7B.",
            "Direction stability is observational and does not identify an "
            "edge or necessary component.",
            "Selection was discovery-only but still came from the same task "
            "families used in confirmation.",
        ],
    }
    write_json(ANALYSIS_ROOT / "targeted_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
