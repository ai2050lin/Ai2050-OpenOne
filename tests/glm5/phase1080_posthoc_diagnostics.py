#!/usr/bin/env python3
"""Post-hoc diagnostics for Phase1080; never alters frozen predictions."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1079_output_orthogonal_pattern_finalize as atlas_math
import phase1080_natural_relevance_atlas_protocol as protocol


FIELD_COLUMNS = {
    "relevance": "mean_relevance_relative_magnitude",
    "presence": "mean_presence_relative_magnitude",
    "total": "mean_total_relative_magnitude",
}
atlas_math.protocol = protocol
atlas_math.FIELD_COLUMNS = FIELD_COLUMNS


def finite_mean(values: list[float]) -> float | None:
    selected = [value for value in values if math.isfinite(value)]
    return float(np.mean(selected)) if selected else None


def assignment(
    source: np.ndarray,
    target: np.ndarray,
    source_field: str,
    target_field: str,
    model: str,
    split: str,
) -> dict[str, Any]:
    matrix = source @ target.T
    exact = atlas_math.exact_assignment_test(matrix)
    rows = []
    for index, family in enumerate(protocol.BASE_FAMILIES):
        order = np.argsort(-matrix[index])
        predicted = int(order[0])
        rows.append({
            "family": family,
            "predicted_family": protocol.BASE_FAMILIES[predicted],
            "correct": predicted == index,
            "same_family_cosine": float(matrix[index, index]),
            "best_cosine": float(matrix[index, predicted]),
        })
    return {
        "model": model,
        "split": split,
        "source_field": source_field,
        "target_field": target_field,
        "top1_correct": sum(int(row["correct"]) for row in rows),
        "rows": rows,
        **exact,
    }


def behavior_table(summary: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for family in protocol.FAMILIES:
        candidate_values = []
        infer_generation = []
        split_rows = {}
        for split in protocol.SPLITS:
            row = summary["behavior_summary"][family][split]
            branch_candidates = {
                branch: row[branch]["candidate_accuracy"]
                for branch in protocol.BRANCHES
            }
            candidate_values.extend(
                float(value) for value in branch_candidates.values()
                if value is not None
            )
            generation_value = row["infer"][
                "generation_semantic_first_accuracy"
            ]
            if generation_value is not None:
                infer_generation.append(float(generation_value))
            split_rows[split] = {
                "candidate_accuracy_by_branch": branch_candidates,
                "infer_generation_accuracy": generation_value,
                "behavior_supported_unit_count": row[
                    "behavior_supported_unit_count"
                ],
            }
        output[family] = {
            "minimum_candidate_accuracy": (
                min(candidate_values) if candidate_values else None
            ),
            "minimum_infer_generation_accuracy": (
                min(infer_generation) if infer_generation else None
            ),
            "splits": split_rows,
        }
    return output


def pooled_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["conditioning"] != "all_finite" or row["split"] != "confirmation":
            continue
        depth = float(row["relative_depth"])
        zone = "early" if depth < 1 / 3 else "middle" if depth < 2 / 3 else "late"
        for field, column in FIELD_COLUMNS.items():
            value = row[column]
            if value is not None and math.isfinite(float(value)):
                buckets[(field, row["component"], row["role"], zone)].append(
                    float(value)
                )
    output = []
    for (field, component, role, zone), values in buckets.items():
        output.append({
            "field": field,
            "component": component,
            "role": role,
            "depth_zone": zone,
            "mean_relative_magnitude": float(np.mean(values)),
            "observation_count": len(values),
        })
    output.sort(
        key=lambda row: (row["field"], -row["mean_relative_magnitude"])
    )
    return output


def family_role_peaks(
    rows: list[dict[str, Any]],
    role: str,
    field: str,
) -> dict[str, dict[str, Any]]:
    """Return the strongest confirmation event for each family at one role."""
    column = FIELD_COLUMNS[field]
    output: dict[str, dict[str, Any]] = {}
    for family in protocol.FAMILIES:
        candidates = [
            row
            for row in rows
            if row["conditioning"] == "all_finite"
            and row["split"] == "confirmation"
            and row["family"] == family
            and row["role"] == role
            and row[column] is not None
            and math.isfinite(float(row[column]))
        ]
        if not candidates:
            continue
        peak = max(candidates, key=lambda row: float(row[column]))
        output[family] = {
            "event_id": peak["event_id"],
            "component": peak["component"],
            "depth": int(peak["depth"]),
            "relative_depth": float(peak["relative_depth"]),
            "mean_relative_magnitude": float(peak[column]),
        }
    return output


def main() -> None:
    exact_assignments = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "exact_assignments.json"
    )["rows"]
    diagnostics: dict[str, Any] = {
        "schema_version": "phase1080_posthoc_diagnostics.v1",
        "phase": protocol.PHASE,
        "status": "posthoc_not_preregistered",
        "changes_frozen_prediction_results": False,
        "purpose": (
            "Diagnose the P3 ceiling, behavior failures, and FP16 finite "
            "coverage without loading a model or selecting components."
        ),
        "cross_model_centered_assignments": [
            row
            for row in exact_assignments
            if row["comparison"] == "cross_model_confirmation"
            and row["profile"] == "family_centered"
            and row["field"] in {"relevance", "presence"}
        ],
        "by_model": {},
    }
    for model in protocol.MODELS:
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "atlas" / model / "response_metrics.jsonl"
        )
        summary = protocol.read_json(
            protocol.OUT_ROOT / "atlas" / model / "summary.json"
        )
        cross_field = []
        same_family_cosines: dict[str, Any] = {}
        for split in protocol.SPLITS:
            banks = {
                field: atlas_math.profile_bank(
                    rows,
                    protocol.BASE_FAMILIES,
                    split,
                    field,
                    centered=True,
                )
                for field in FIELD_COLUMNS
            }
            for source_field, target_field in (
                ("relevance", "presence"),
                ("presence", "relevance"),
                ("relevance", "total"),
                ("presence", "total"),
            ):
                cross_field.append(assignment(
                    banks[source_field],
                    banks[target_field],
                    source_field,
                    target_field,
                    model,
                    split,
                ))
            same_family_cosines[split] = {
                family: float(
                    banks["relevance"][index] @ banks["presence"][index]
                )
                for index, family in enumerate(protocol.BASE_FAMILIES)
            }

        vector_comparisons_per_event_unit_role = 36
        hidden_denominator = (
            int(summary["unit_count"])
            * int(summary["event_count"])
            * vector_comparisons_per_event_unit_role
            * len(protocol.CAPTURE_ROLES)
        )
        candidate_denominator = int(summary["case_count"])
        diagnostics["by_model"][model] = {
            "cross_field_assignments": cross_field,
            "same_family_relevance_presence_cosines": same_family_cosines,
            "mean_same_family_relevance_presence_cosine": finite_mean([
                value
                for split in same_family_cosines.values()
                for value in split.values()
            ]),
            "behavior": behavior_table(summary),
            "pooled_region_ranking": pooled_regions(rows),
            "request_end_family_peaks": {
                field: family_role_peaks(rows, "request_end", field)
                for field in FIELD_COLUMNS
            },
            "finite_coverage": {
                "nonfinite_candidate_count": summary[
                    "nonfinite_candidate_count"
                ],
                "candidate_observation_count": candidate_denominator,
                "nonfinite_candidate_fraction": (
                    summary["nonfinite_candidate_count"] / candidate_denominator
                ),
                "nonfinite_hidden_role_count": summary[
                    "nonfinite_hidden_magnitude_role_count"
                ],
                "hidden_role_observation_count": hidden_denominator,
                "nonfinite_hidden_role_fraction": (
                    summary["nonfinite_hidden_magnitude_role_count"]
                    / hidden_denominator
                ),
            },
        }

    diagnostics["interpretation"] = {
        "ceiling_rule": (
            "P3 cannot distinguish fields when both independently retrieve "
            "8/8. Cross-field assignments and same-family cosines diagnose "
            "whether the two maps are structurally interchangeable."
        ),
        "not_evidence_for": [
            "causal necessity or sufficiency",
            "minimal or optimal coding",
            "a token-level ecological niche",
            "a complete language mechanism",
        ],
    }
    diagnostics["diagnostic_digest"] = protocol.digest(diagnostics)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "posthoc_diagnostics.json",
        diagnostics,
    )
    print({
        "phase": protocol.PHASE,
        "status": diagnostics["status"],
        "digest": diagnostics["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
