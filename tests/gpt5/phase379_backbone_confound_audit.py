#!/usr/bin/env python3
"""Audit whether Phase379 raw layout similarity is a common-backbone confound."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "entity_recency",
    "number_agreement",
    "relation_binding",
    "target_vs_wrong",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / (left_norm * right_norm)


def norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def profile_vectors(split: str) -> tuple[dict[tuple[str, str, str], list[float]], list[tuple[str, str, int]]]:
    rows = read_jsonl(OUT / split / "phase379_function_profiles.jsonl")
    cells = sorted(
        {
            (row["component_type"], row["position_role"], row["depth_bin"])
            for row in rows
        }
    )
    scalar = {
        (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["component_type"],
            row["position_role"],
            row["depth_bin"],
        ): float(row["median_descriptive_layout_weight"])
        for row in rows
    }
    keys = sorted(
        {
            (row["model"], row["mechanism_id"], row["contrast_axis"])
            for row in rows
        }
    )
    return (
        {
            key: [
                scalar[(*key, component, role, depth)]
                for component, role, depth in cells
            ]
            for key in keys
        },
        cells,
    )


def residualize(
    vectors: dict[tuple[str, str, str], list[float]]
) -> tuple[
    dict[tuple[str, str, str], list[float]],
    dict[tuple[str, str], list[float]],
]:
    axes = sorted({key[2] for key in vectors})
    residuals = {}
    backbones = {}
    for model in MODELS:
        for axis in axes:
            source = [vectors[(model, mechanism, axis)] for mechanism in MECHANISMS]
            backbone = [mean(values) for values in zip(*source, strict=True)]
            backbones[(model, axis)] = backbone
            for mechanism in MECHANISMS:
                residuals[(model, mechanism, axis)] = [
                    value - base
                    for value, base in zip(
                        vectors[(model, mechanism, axis)], backbone, strict=True
                    )
                ]
    return residuals, backbones


def main() -> None:
    discovery, cells = profile_vectors("fresh_discovery")
    calibration, calibration_cells = profile_vectors("fresh_calibration")
    if cells != calibration_cells:
        raise RuntimeError("Profile cell contract changed")
    discovery_residual, discovery_backbone = residualize(discovery)
    calibration_residual, calibration_backbone = residualize(calibration)
    rows = []
    for key in sorted(discovery):
        model, mechanism, axis = key
        raw_discovery = discovery[key]
        raw_calibration = calibration[key]
        residual_discovery = discovery_residual[key]
        residual_calibration = calibration_residual[key]
        raw_norm = norm(raw_discovery)
        residual_norm = norm(residual_discovery)
        rows.append(
            {
                "schema_version": "52.5.0",
                "phase_id": "Phase379-BackboneConfoundAudit",
                "model": model,
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "raw_discovery_calibration_cosine": cosine(
                    raw_discovery, raw_calibration
                ),
                "backbone_discovery_calibration_cosine": cosine(
                    discovery_backbone[(model, axis)],
                    calibration_backbone[(model, axis)],
                ),
                "backbone_residual_discovery_calibration_cosine": cosine(
                    residual_discovery, residual_calibration
                ),
                "raw_profile_norm": raw_norm,
                "backbone_residual_norm": residual_norm,
                "backbone_residual_norm_fraction": residual_norm
                / max(raw_norm, 1e-12),
                "independent_validation_for_residual_metric": False,
                "causal_scan_authorized": False,
            }
        )
    crossmodel_rows = []
    axes = sorted({key[2] for key in discovery})
    for split, residuals in (
        ("fresh_discovery", discovery_residual),
        ("fresh_calibration", calibration_residual),
    ):
        for mechanism in MECHANISMS:
            for axis in axes:
                for left_index, left in enumerate(MODELS):
                    for right in MODELS[left_index + 1 :]:
                        crossmodel_rows.append(
                            {
                                "schema_version": "52.5.0",
                                "phase_id": "Phase379-BackboneConfoundAudit",
                                "split": split,
                                "mechanism_id": mechanism,
                                "contrast_axis": axis,
                                "left_model": left,
                                "right_model": right,
                                "backbone_residual_profile_cosine": cosine(
                                    residuals[(left, mechanism, axis)],
                                    residuals[(right, mechanism, axis)],
                                ),
                                "heterogeneous_pair": "glm4" in {left, right},
                                "causal_scan_authorized": False,
                            }
                        )
    row_path = OUT / "phase379_backbone_residual_rows.jsonl"
    crossmodel_path = OUT / "phase379_backbone_residual_crossmodel_rows.jsonl"
    write_jsonl(row_path, rows)
    write_jsonl(crossmodel_path, crossmodel_rows)
    raw_values = [row["raw_discovery_calibration_cosine"] for row in rows]
    residual_values = [
        row["backbone_residual_discovery_calibration_cosine"] for row in rows
    ]
    residual_fractions = [row["backbone_residual_norm_fraction"] for row in rows]
    heterogeneous_values = [
        row["backbone_residual_profile_cosine"]
        for row in crossmodel_rows
        if row["heterogeneous_pair"]
    ]
    summary = {
        "schema_version": "52.5.0",
        "phase_id": "Phase379-BackboneConfoundAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "separate_common_architecture_backbone_from_function_specific_layout_residuals",
        "method": {
            "backbone": "per_model_per_axis_cellwise_mean_across_four_mechanisms",
            "residual": "mechanism_profile_minus_frozen_common_backbone",
            "advanced_statistical_model_used": False,
            "posthoc_diagnostic": True,
            "calibration_was_already_opened_under_raw_metric": True,
        },
        "denominator": {
            "raw_profile_count": len(rows),
            "profile_width": len(cells),
            "crossmodel_residual_comparison_count": len(crossmodel_rows),
        },
        "results": {
            "raw_discovery_calibration_cosine": {
                "minimum": min(raw_values),
                "median": median(raw_values),
                "mean": mean(raw_values),
                "maximum": max(raw_values),
            },
            "backbone_residual_discovery_calibration_cosine": {
                "minimum": min(residual_values),
                "median": median(residual_values),
                "mean": mean(residual_values),
                "maximum": max(residual_values),
            },
            "backbone_residual_norm_fraction": {
                "minimum": min(residual_fractions),
                "median": median(residual_fractions),
                "mean": mean(residual_fractions),
                "maximum": max(residual_fractions),
            },
            "heterogeneous_crossmodel_residual_cosine": {
                "minimum": min(heterogeneous_values),
                "median": median(heterogeneous_values),
                "mean": mean(heterogeneous_values),
                "maximum": max(heterogeneous_values),
            },
            "raw_profile_replication_is_function_specific_evidence": False,
            "common_backbone_confound_detected": True,
            "causal_scan_authorized": False,
            "language_encoding_mechanism_closed": False,
        },
        "reason_causal_scan_closed": (
            "the_raw_metric_made_all_36_model_profiles_and_all_12_mechanism_axis_objects_pass; "
            "the_correction_was_defined_after_discovery_and_the_existing_calibration_was_consumed_by_the_old_metric"
        ),
        "next_stage": {
            "phase": 380,
            "objective": "freeze_backbone_residual_and_counterfactual_influence_metrics_before_a_new_independent_factorial_denominator",
            "reuse_current_calibration_for_new_metric_claims": False,
            "open_current_physical_holdout": False,
            "single_neuron_scan": False,
        },
    }
    write_json(OUT / "phase379_backbone_confound_audit.json", summary)
    write_json(
        OUT / "phase379_causal_authorization.json",
        {
            "schema_version": "52.5.0",
            "phase_id": "Phase379-CausalAuthorization",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "supersedes_raw_profile_authorization": True,
            "authorization": {
                "run_registered_natural_boundary_causal_scan": False,
                "open_physical_holdout": False,
                "run_single_neuron_scan": False,
            },
            "reason": "common_backbone_confound_and_no_independent_validation_for_the_posthoc_residual_metric",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
