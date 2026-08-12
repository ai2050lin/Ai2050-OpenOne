#!/usr/bin/env python3
"""Produce descriptive Phase1011 map diagnostics from frozen outputs."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1011_native_semantic_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(
        str(row[field]) for row in rows
    ).items()))


def depth_band(relative_depth: float) -> str:
    if relative_depth < 1.0 / 3.0:
        return "early"
    if relative_depth < 2.0 / 3.0:
        return "middle"
    return "late"


def identity_without_axis(row: dict[str, Any]) -> tuple:
    return (
        row["model"],
        row["family"],
        row["output_mode"],
        row["operation"],
        row["event_id"],
    )


def top_counts(
    rows: list[dict[str, Any]],
    field: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    values = Counter(str(row[field]) for row in rows)
    return [
        {"value": value, "count": count}
        for value, count in values.most_common(limit)
    ]


def main() -> None:
    final_root = OUT_ROOT / "final"
    summary = read_json(final_root / "summary.json")
    motifs = read_jsonl(final_root / "repeated_events.jsonl")
    contours = read_jsonl(final_root / "response_contours.jsonl")
    output_alignments = read_jsonl(
        final_root / "cross_output_alignments.jsonl"
    )
    family_alignments = read_jsonl(
        final_root / "cross_family_alignments.jsonl"
    )
    model_alignments = read_jsonl(
        final_root / "cross_model_alignments.jsonl"
    )
    sensitivity = read_jsonl(
        final_root / "threshold_sensitivity.jsonl"
    )
    prompt = [row for row in motifs if row["stage"] == "prompt"]
    after_answer = [
        row for row in motifs if row["stage"] == "after_answer"
    ]
    axis_sets: dict[tuple, set[str]] = defaultdict(set)
    for row in prompt:
        axis_sets[identity_without_axis(row)].add(
            row["qualification_axis"]
        )
    both_axis = {
        key for key, axes in axis_sets.items() if len(axes) == 2
    }
    one_axis = {
        key for key, axes in axis_sets.items() if len(axes) == 1
    }
    threshold_totals = {}
    for direction, prevalence in (
        (0.85, 0.70),
        (0.90, 0.80),
        (0.95, 0.90),
    ):
        threshold_totals[f"d{direction:.2f}_p{prevalence:.2f}"] = int(sum(
            row["candidate_count"]
            for row in sensitivity
            if float(row["direction_threshold"]) == direction
            and float(row["prevalence_threshold"]) == prevalence
        ))

    model_details = []
    for model in MODELS:
        model_prompt = [row for row in prompt if row["model"] == model]
        model_after = [
            row for row in after_answer if row["model"] == model
        ]
        by_operation_role = {}
        for operation in ("F", "Q", "FQ", "X"):
            selected = [
                row for row in model_prompt
                if row["operation"] == operation
            ]
            by_operation_role[operation] = {
                "count": len(selected),
                "top_role_classes": top_counts(
                    selected, "role_class"
                ),
                "components": count_by(selected, "component"),
                "depth_bands": dict(sorted(Counter(
                    depth_band(float(row["relative_depth"]))
                    for row in selected
                ).items())),
            }
        model_details.append({
            "model": model,
            "prompt_event_count": len(model_prompt),
            "after_answer_event_count": len(model_after),
            "prompt_by_family": count_by(model_prompt, "family"),
            "prompt_by_output_mode": count_by(
                model_prompt, "output_mode"
            ),
            "prompt_by_axis": count_by(
                model_prompt, "qualification_axis"
            ),
            "prompt_by_operation": count_by(
                model_prompt, "operation"
            ),
            "prompt_by_component": count_by(
                model_prompt, "component"
            ),
            "prompt_by_role_class": count_by(
                model_prompt, "role_class"
            ),
            "prompt_by_depth_band": dict(sorted(Counter(
                depth_band(float(row["relative_depth"]))
                for row in model_prompt
            ).items())),
            "input_embedding_event_count": int(sum(
                row["component"] == "residual"
                and int(row["depth"]) == 0
                for row in model_prompt
            )),
            "operation_role_diagnostics": by_operation_role,
        })

    glm_l30 = [
        row for row in prompt
        if row["model"] == "glm4"
        and row["component"] == "attention_output"
        and int(row["depth"]) == 30
    ]
    glm_l30_boundary = [
        row for row in glm_l30
        if row["role_class"] == "answer_boundary"
    ]

    prompt_contours = [
        row for row in contours if row["stage"] == "prompt"
    ]
    ranked_contours = sorted(
        prompt_contours,
        key=lambda row: (
            int(row["span"]),
            int(row["event_count"]),
            float(row["minimum_confirmation_direction"]),
            float(row["minimum_confirmation_prevalence"]),
        ),
        reverse=True,
    )
    top_contours = ranked_contours[:100]
    write_jsonl(final_root / "top_prompt_contours.jsonl", top_contours)

    three_output = [
        row for row in output_alignments
        if len(row["distinct_values"]) == 3
    ]
    three_family = [
        row for row in family_alignments
        if len(row["distinct_values"]) == 3
    ]
    three_model = [
        row for row in model_alignments
        if len(row["distinct_values"]) == 3
    ]
    result = {
        "schema_version": "phase1011_native_analysis.v1",
        "phase": PHASE,
        "canonical_repeated_event_count": len(motifs),
        "prompt_event_count": len(prompt),
        "after_answer_event_count": len(after_answer),
        "prompt_fraction": len(prompt) / max(len(motifs), 1),
        "prompt_unique_event_operation_count": len(axis_sets),
        "prompt_shared_by_both_behavior_axes": len(both_axis),
        "prompt_only_one_behavior_axis": len(one_axis),
        "prompt_both_axis_fraction": (
            len(both_axis) / max(len(axis_sets), 1)
        ),
        "threshold_sensitivity_totals": threshold_totals,
        "model_details": model_details,
        "glm4_l30_attention_prompt_event_count": len(glm_l30),
        "glm4_l30_attention_answer_boundary_count": len(
            glm_l30_boundary
        ),
        "glm4_l30_attention_answer_boundary_breakdown": {
            "families": count_by(glm_l30_boundary, "family"),
            "output_modes": count_by(
                glm_l30_boundary, "output_mode"
            ),
            "operations": count_by(glm_l30_boundary, "operation"),
            "qualification_axes": count_by(
                glm_l30_boundary, "qualification_axis"
            ),
        },
        "prompt_contour_count": len(prompt_contours),
        "top_prompt_contours_written": len(top_contours),
        "cross_output_alignment_count": len(output_alignments),
        "all_three_output_mode_alignment_count": len(three_output),
        "cross_family_alignment_count": len(family_alignments),
        "all_three_family_alignment_count": len(three_family),
        "cross_model_alignment_count": len(model_alignments),
        "all_three_model_alignment_count": len(three_model),
        "interpretation": {
            "both_axis_overlap": (
                "same physical event and operation repeats under both "
                "controlled-candidate and natural-rollout qualification"
            ),
            "input_embedding_events": (
                "direct lexical/surface responses, retained as controls "
                "rather than interpreted as computed mechanisms"
            ),
            "glm4_l30": (
                "tests whether the historically emphasized region remains "
                "descriptively visible without selecting it in Phase1011"
            ),
            "alignment": (
                "functional co-response correspondence only; no transport "
                "or physical homology"
            ),
        },
        "source_summary_digest": summary["protocol_digest"],
    }
    write_json(final_root / "analysis_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
