#!/usr/bin/env python3
"""Post-hoc role ablations for failed Phase1081 frozen gates.

These diagnostics cannot alter any preregistered prediction or evidence level.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1081_latin_route_atlas_finalize as finalize
import phase1081_latin_route_atlas_protocol as protocol


ROLE_SETS = {
    "pre_query_records": ("record0_end", "record1_end"),
    "request_only": ("request_end",),
    "answer_only": ("answer_boundary",),
    "downstream_without_selected_label": (
        "request_end", "answer_boundary",
    ),
    "all_without_selected_label": (
        "record0_end", "record1_end", "request_end", "answer_boundary",
    ),
}


def assignment(
    rows_source: list[dict[str, Any]],
    rows_target: list[dict[str, Any]],
    *,
    comparison: str,
    source_model: str,
    target_model: str,
    role_set: str,
    source_split: str,
    target_split: str,
    field: str,
) -> dict[str, Any]:
    roles = ROLE_SETS[role_set]
    return finalize.assignment_record(
        comparison=comparison,
        field=field,
        profile=f"posthoc_family_centered__{role_set}",
        source_model=source_model,
        target_model=target_model,
        families=protocol.BASE_FAMILIES,
        source_values=finalize.profile_bank(
            rows_source,
            protocol.BASE_FAMILIES,
            source_split,
            field,
            roles=roles,
            centered=True,
        ),
        target_values=finalize.profile_bank(
            rows_target,
            protocol.BASE_FAMILIES,
            target_split,
            field,
            roles=roles,
            centered=True,
        ),
    )


def main() -> None:
    metrics = {
        model: protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        for model in protocol.MODELS
    }
    rows = []
    for role_set in ROLE_SETS:
        for model in protocol.MODELS:
            for field in ("content_route", "duplicate_route"):
                rows.append(assignment(
                    metrics[model],
                    metrics[model],
                    comparison="posthoc_within_model_split",
                    source_model=model,
                    target_model=model,
                    role_set=role_set,
                    source_split="discovery",
                    target_split="confirmation",
                    field=field,
                ))
        for source_model in protocol.MODELS:
            for target_model in protocol.MODELS:
                if source_model == target_model:
                    continue
                rows.append(assignment(
                    metrics[source_model],
                    metrics[target_model],
                    comparison="posthoc_cross_model_confirmation",
                    source_model=source_model,
                    target_model=target_model,
                    role_set=role_set,
                    source_split="confirmation",
                    target_split="confirmation",
                    field="content_route",
                ))

    control_by_role = {}
    for model, model_rows in metrics.items():
        role_rows = {}
        for role in protocol.CAPTURE_ROLES:
            ratios = []
            for row in model_rows:
                if (
                    row["conditioning"] != "all_finite"
                    or row["role"] != role
                    or row["mean_content_route_relative_magnitude"] is None
                    or row["mean_label_swap"] is None
                    or row["mean_shell"] is None
                    or float(row["mean_content_route_relative_magnitude"])
                    <= finalize.EPSILON
                ):
                    continue
                ratios.append(
                    max(float(row["mean_label_swap"]), float(row["mean_shell"]))
                    / float(row["mean_content_route_relative_magnitude"])
                )
            role_rows[role] = {
                "median_max_control_to_content": finalize.safe_median(ratios),
                "count": len(ratios),
            }
        control_by_role[model] = role_rows

    top_rows = protocol.read_jsonl(
        protocol.OUT_ROOT / "analysis" / "top_regions.jsonl"
    )
    peak_role_counts = {
        model: dict(Counter(
            row["role"] for row in top_rows
            if row["model"] == model
            and row["field"] == "content_route"
            and row["rank"] == 1
        ))
        for model in protocol.MODELS
    }
    threshold_p = float(
        protocol.EVIDENCE_THRESHOLDS["permutation_p_max"]
    )
    threshold_top1 = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_base_family_top1"]
    )
    summary = {}
    for role_set in ROLE_SETS:
        selected = [
            row for row in rows
            if row["profile"].endswith(role_set)
        ]
        summary[role_set] = {
            "within_content_passing_models": [
                row["source_model"] for row in selected
                if row["comparison"] == "posthoc_within_model_split"
                and row["field"] == "content_route"
                and row["exact_upper_tail_p"] <= threshold_p
                and row["top1_correct"] >= threshold_top1
            ],
            "within_duplicate_passing_models": [
                row["source_model"] for row in selected
                if row["comparison"] == "posthoc_within_model_split"
                and row["field"] == "duplicate_route"
                and row["exact_upper_tail_p"] <= threshold_p
                and row["top1_correct"] >= threshold_top1
            ],
            "cross_content_passing_pairs": [
                f"{row['source_model']}__{row['target_model']}"
                for row in selected
                if row["comparison"] == "posthoc_cross_model_confirmation"
                and row["exact_upper_tail_p"] <= threshold_p
                and row["top1_correct"] >= threshold_top1
            ],
        }

    payload = {
        "schema_version": "phase1081_posthoc_diagnostics.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_cannot_change_frozen_predictions",
        "role_sets": {key: list(value) for key, value in ROLE_SETS.items()},
        "summary": summary,
        "control_by_role": control_by_role,
        "content_peak_role_counts": peak_role_counts,
        "rows": rows,
        "interpretation": (
            "Role ablations diagnose whether selected-label topology drives "
            "within-model repetition. They are not confirmatory evidence."
        ),
    }
    payload["diagnostic_digest"] = protocol.digest(payload)
    output_path = (
        protocol.OUT_ROOT / "analysis" / "posthoc_role_diagnostics.json"
    )
    protocol.write_json(output_path, payload)

    final_path = protocol.OUT_ROOT / "analysis" / "final_summary.json"
    final = protocol.read_json(final_path)
    final["posthoc_role_diagnostics"] = {
        "path": str(output_path.relative_to(ROOT)),
        "diagnostic_digest": payload["diagnostic_digest"],
        "status": payload["status"],
        "summary": summary,
    }
    final.pop("summary_digest", None)
    final["summary_digest"] = protocol.digest(final)
    protocol.write_json(final_path, final)
    print({
        "phase": protocol.PHASE,
        "status": payload["status"],
        "summary": summary,
        "diagnostic_digest": payload["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
