#!/usr/bin/env python3
"""Summarize Phase1011 repeated response contours without causal labels."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1011_native_semantic_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_MODES,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLIT_INDEX = {"discovery": 0, "confirmation": 1}
DIRECTION_AXES = ("semantic_panel", "natural_rollout")
TARGET_OPERATIONS = ("F", "Q", "FQ", "X")
CONTROL_OPERATIONS = ("E", "O", "N")
DIRECTION_THRESHOLDS = (0.85, 0.90, 0.95)
PREVALENCE_THRESHOLDS = (0.70, 0.80, 0.90)
CANONICAL_DIRECTION = 0.90
CANONICAL_PREVALENCE = 0.80
MIN_QUALIFIED_PER_SPLIT = 8
MIN_NAME_POOLS = 2
MIN_TEMPLATES = 2
EPSILON = 1e-12


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def split_masks(
    units: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        split: np.asarray(
            [row["split"] == split for row in units],
            dtype=np.bool_,
        )
        for split in SPLIT_INDEX
    }


def coverage(
    units: list[dict[str, Any]],
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = [
        row for index, row in enumerate(units) if bool(mask[index])
    ]
    return {
        "n": len(selected),
        "name_pools": sorted({
            int(row["name_pool"]) for row in selected
        }),
        "templates": sorted({
            int(row["template"]) for row in selected
        }),
        "worlds": sorted({
            int(row["world_index"]) for row in selected
        }),
    }


def profile_event(
    *,
    values: np.ndarray,
    qualified: np.ndarray,
    split_mask: np.ndarray,
    operation_index: int,
    event_index: int,
) -> dict[str, float | int | None]:
    mask = qualified[:, operation_index] & split_mask
    target = values[mask, operation_index, event_index]
    if target.size == 0:
        return {
            "n": 0,
            "target_median": None,
            "control_median": None,
            "contrast_delta": None,
            "contrast_prevalence": None,
        }
    controls = values[
        mask, :, event_index
    ][:, [OP_INDEX[name] for name in CONTROL_OPERATIONS]]
    unit_background = np.nanmedian(controls, axis=1)
    return {
        "n": int(target.size),
        "target_median": finite(np.nanmedian(target)),
        "control_median": finite(np.nanmedian(unit_background)),
        "contrast_delta": finite(
            np.nanmedian(target - unit_background)
        ),
        "contrast_prevalence": finite(
            np.nanmean(target > unit_background)
        ),
    }


def repeated_at(
    *,
    profile: dict[str, Any],
    direction_threshold: float,
    prevalence_threshold: float,
) -> bool:
    discovery = profile["splits"]["discovery"]
    confirmation = profile["splits"]["confirmation"]
    for row in (discovery, confirmation):
        if row["coverage"]["n"] < MIN_QUALIFIED_PER_SPLIT:
            return False
        if len(row["coverage"]["name_pools"]) < MIN_NAME_POOLS:
            return False
        if len(row["coverage"]["templates"]) < MIN_TEMPLATES:
            return False
        if row["direction_consistency"] is None:
            return False
        if row["direction_consistency"] < direction_threshold:
            return False
        if row["contrast_prevalence"] is None:
            return False
        if row["contrast_prevalence"] < prevalence_threshold:
            return False
        if row["contrast_delta"] is None or row["contrast_delta"] <= 0:
            return False
    return True


def scan_panel_profiles(
    *,
    model: str,
    family: str,
    output_mode: str,
    panel_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = read_jsonl(panel_root / "events.jsonl")
    units = read_jsonl(panel_root / "units.jsonl")
    scalar_data = np.load(panel_root / "response_scalars.npz")
    direction_data = np.load(panel_root / "direction_consistency.npz")
    values = scalar_data["normalized_magnitude"]
    directions = direction_data["direction_consistency"]
    direction_counts = direction_data["direction_count"]
    if tuple(directions.shape[:3]) != (
        len(DIRECTION_AXES),
        len(ANALYSIS_OPERATIONS),
        len(SPLIT_INDEX),
    ):
        raise RuntimeError(
            f"direction shape drift in {model}/{family}/{output_mode}: "
            f"{directions.shape}"
        )
    masks = split_masks(units)
    profiles = []
    sensitivity = []
    for axis_index, axis in enumerate(DIRECTION_AXES):
        qualified = scalar_data[f"{axis}_qualified"]
        for operation in TARGET_OPERATIONS:
            operation_index = OP_INDEX[operation]
            coverage_by_split = {
                split: coverage(
                    units,
                    qualified[:, operation_index] & masks[split],
                )
                for split in SPLIT_INDEX
            }
            operation_profiles = []
            for event in events:
                event_index = int(event["event_index"])
                split_profiles = {}
                for split, split_index in SPLIT_INDEX.items():
                    response = profile_event(
                        values=values,
                        qualified=qualified,
                        split_mask=masks[split],
                        operation_index=operation_index,
                        event_index=event_index,
                    )
                    response["coverage"] = coverage_by_split[split]
                    response["direction_consistency"] = finite(
                        directions[
                            axis_index,
                            operation_index,
                            split_index,
                            event_index,
                        ]
                    )
                    response["direction_count"] = int(
                        direction_counts[
                            axis_index,
                            operation_index,
                            split_index,
                            event_index,
                        ]
                    )
                    split_profiles[split] = response
                profile = {
                    "schema_version": (
                        "phase1011_native_response_profile.v1"
                    ),
                    "phase": PHASE,
                    "model": model,
                    "family": family,
                    "output_mode": output_mode,
                    "qualification_axis": axis,
                    "operation": operation,
                    "event_id": event["event_id"],
                    "event_index": event_index,
                    "stage": event["stage"],
                    "component": event["component"],
                    "depth": int(event["depth"]),
                    "relative_depth": float(event["relative_depth"]),
                    "role": event["role"],
                    "role_class": event["role_class"],
                    "splits": split_profiles,
                    "interpretation_scope": (
                        "pre_output_internal_response"
                        if event["stage"] == "prompt"
                        else "teacher_forced_answer_surface_response"
                    ),
                    "edge_claim_allowed": "co_response_only",
                }
                profile["canonical_repeated"] = repeated_at(
                    profile=profile,
                    direction_threshold=CANONICAL_DIRECTION,
                    prevalence_threshold=CANONICAL_PREVALENCE,
                )
                operation_profiles.append(profile)
                if profile["canonical_repeated"]:
                    profiles.append(profile)
            for direction_threshold in DIRECTION_THRESHOLDS:
                for prevalence_threshold in PREVALENCE_THRESHOLDS:
                    sensitivity.append({
                        "schema_version": (
                            "phase1011_native_threshold_cell.v1"
                        ),
                        "phase": PHASE,
                        "model": model,
                        "family": family,
                        "output_mode": output_mode,
                        "qualification_axis": axis,
                        "operation": operation,
                        "direction_threshold": direction_threshold,
                        "prevalence_threshold": prevalence_threshold,
                        "candidate_count": int(sum(
                            repeated_at(
                                profile=row,
                                direction_threshold=direction_threshold,
                                prevalence_threshold=prevalence_threshold,
                            )
                            for row in operation_profiles
                        )),
                        "event_count": len(operation_profiles),
                    })
    scalar_data.close()
    direction_data.close()
    return profiles, sensitivity


def build_contours(
    motifs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in motifs:
        grouped[(
            row["model"],
            row["family"],
            row["output_mode"],
            row["qualification_axis"],
            row["operation"],
            row["stage"],
            row["component"],
            row["role_class"],
        )].append(row)
    contours = []
    for key, rows in sorted(grouped.items()):
        by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_depth[int(row["depth"])].append(row)
        depths = sorted(by_depth)
        if not depths:
            continue
        runs: list[list[int]] = [[depths[0]]]
        for depth in depths[1:]:
            if depth <= runs[-1][-1] + 1:
                runs[-1].append(depth)
            else:
                runs.append([depth])
        for run_index, run in enumerate(runs):
            members = [
                row
                for depth in run
                for row in by_depth[depth]
            ]
            model, family, output_mode, axis, operation, stage, component, role_class = key
            contours.append({
                "schema_version": "phase1011_native_response_contour.v1",
                "phase": PHASE,
                "contour_id": (
                    f"{model}.{family}.{output_mode}.{axis}.{operation}."
                    f"{stage}.{component}.{role_class}.r{run_index}"
                ),
                "model": model,
                "family": family,
                "output_mode": output_mode,
                "qualification_axis": axis,
                "operation": operation,
                "stage": stage,
                "component": component,
                "role_class": role_class,
                "start_depth": min(run),
                "end_depth": max(run),
                "span": max(run) - min(run) + 1,
                "event_count": len(members),
                "roles": sorted({row["role"] for row in members}),
                "event_ids": [row["event_id"] for row in members],
                "minimum_discovery_direction": finite(min(
                    row["splits"]["discovery"]["direction_consistency"]
                    for row in members
                )),
                "minimum_confirmation_direction": finite(min(
                    row["splits"]["confirmation"]["direction_consistency"]
                    for row in members
                )),
                "minimum_discovery_prevalence": finite(min(
                    row["splits"]["discovery"]["contrast_prevalence"]
                    for row in members
                )),
                "minimum_confirmation_prevalence": finite(min(
                    row["splits"]["confirmation"]["contrast_prevalence"]
                    for row in members
                )),
                "claim": "repeated_response_contour_only",
            })
    return contours


def alignment_rows(
    motifs: list[dict[str, Any]],
    *,
    varying: str,
    minimum_distinct: int = 2,
) -> list[dict[str, Any]]:
    if varying not in {"output_mode", "family", "model"}:
        raise KeyError(varying)
    stable_fields = [
        "qualification_axis",
        "operation",
        "stage",
        "component",
        "role_class",
    ]
    for candidate in ("model", "family", "output_mode"):
        if candidate != varying:
            stable_fields.insert(0, candidate)
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in motifs:
        if row["stage"] != "prompt":
            continue
        relative_bin = int(round(float(row["relative_depth"]) * 20.0))
        key = tuple(row[field] for field in stable_fields) + (
            relative_bin,
        )
        grouped[key].append(row)
    results = []
    for key, rows in sorted(grouped.items()):
        distinct = sorted({row[varying] for row in rows})
        if len(distinct) < minimum_distinct:
            continue
        results.append({
            "schema_version": "phase1011_native_functional_alignment.v1",
            "phase": PHASE,
            "alignment_axis": varying,
            "stable_fields": {
                field: rows[0][field] for field in stable_fields
            },
            "relative_depth_bin_center": key[-1] / 20.0,
            "distinct_values": distinct,
            "member_count": len(rows),
            "members": [{
                "model": row["model"],
                "family": row["family"],
                "output_mode": row["output_mode"],
                "event_id": row["event_id"],
                "relative_depth": row["relative_depth"],
            } for row in rows],
            "claim": (
                "relative_depth_functional_correspondence_only"
                if varying == "model"
                else "repeated_functional_correspondence_only"
            ),
        })
    return results


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("protocol revision drift")
    all_motifs = []
    all_sensitivity = []
    model_summaries = []
    for model in MODELS:
        scan_root = OUT_ROOT / "scan" / model
        scan_summary = read_json(scan_root / "summary.json")
        if scan_summary["protocol_digest"] != protocol[
            "preregistration_digest"
        ]:
            raise RuntimeError(f"{model}: scan digest drift")
        model_motifs = []
        model_sensitivity = []
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                motifs, sensitivity = scan_panel_profiles(
                    model=model,
                    family=family,
                    output_mode=output_mode,
                    panel_root=scan_root / family / output_mode,
                )
                model_motifs.extend(motifs)
                model_sensitivity.extend(sensitivity)
        all_motifs.extend(model_motifs)
        all_sensitivity.extend(model_sensitivity)
        model_summaries.append({
            "model": model,
            "canonical_repeated_event_count": len(model_motifs),
            "prompt_repeated_event_count": int(sum(
                row["stage"] == "prompt" for row in model_motifs
            )),
            "after_answer_repeated_event_count": int(sum(
                row["stage"] == "after_answer"
                for row in model_motifs
            )),
            "counts_by_axis": {
                axis: int(sum(
                    row["qualification_axis"] == axis
                    for row in model_motifs
                ))
                for axis in DIRECTION_AXES
            },
            "counts_by_operation": {
                operation: int(sum(
                    row["operation"] == operation
                    for row in model_motifs
                ))
                for operation in TARGET_OPERATIONS
            },
        })
    contours = build_contours(all_motifs)
    output_alignments = alignment_rows(
        all_motifs, varying="output_mode"
    )
    family_alignments = alignment_rows(
        all_motifs, varying="family"
    )
    model_alignments = alignment_rows(
        all_motifs, varying="model"
    )
    final_root = OUT_ROOT / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(final_root / "repeated_events.jsonl", all_motifs)
    write_jsonl(final_root / "response_contours.jsonl", contours)
    write_jsonl(
        final_root / "threshold_sensitivity.jsonl", all_sensitivity
    )
    write_jsonl(
        final_root / "cross_output_alignments.jsonl",
        output_alignments,
    )
    write_jsonl(
        final_root / "cross_family_alignments.jsonl",
        family_alignments,
    )
    write_jsonl(
        final_root / "cross_model_alignments.jsonl",
        model_alignments,
    )
    summary = {
        "schema_version": "phase1011_native_finalize_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": protocol["preregistration_digest"],
        "method": "descriptive_response_field_mapping",
        "canonical_descriptive_thresholds": {
            "minimum_direction_consistency_each_split": (
                CANONICAL_DIRECTION
            ),
            "minimum_contrast_prevalence_each_split": (
                CANONICAL_PREVALENCE
            ),
            "minimum_qualified_each_split": MIN_QUALIFIED_PER_SPLIT,
            "minimum_name_pools_each_split": MIN_NAME_POOLS,
            "minimum_templates_each_split": MIN_TEMPLATES,
            "control_operations": list(CONTROL_OPERATIONS),
        },
        "threshold_sensitivity_grid": {
            "direction": list(DIRECTION_THRESHOLDS),
            "prevalence": list(PREVALENCE_THRESHOLDS),
        },
        "model_summaries": model_summaries,
        "canonical_repeated_event_count": len(all_motifs),
        "prompt_repeated_event_count": int(sum(
            row["stage"] == "prompt" for row in all_motifs
        )),
        "teacher_forced_after_answer_event_count": int(sum(
            row["stage"] == "after_answer" for row in all_motifs
        )),
        "response_contour_count": len(contours),
        "cross_output_alignment_count": len(output_alignments),
        "cross_family_alignment_count": len(family_alignments),
        "cross_model_relative_depth_alignment_count": len(
            model_alignments
        ),
        "formula_status": {
            "normalized_magnitude": (
                "measurement identity, not a language mechanism"
            ),
            "mean_pairwise_direction": (
                "measurement identity, not a language mechanism"
            ),
            "interaction_X": (
                "descriptive non-additivity residual, not an assumed "
                "compositional law"
            ),
        },
        "claim_limits": [
            "repeated events are response observations, not causes",
            "contours do not establish transport edges",
            "after-answer events include the forced answer-token surface",
            "relative-depth cross-model alignment is not physical homology",
            "thresholds organize the map and are not laws of language",
        ],
    }
    write_json(final_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
