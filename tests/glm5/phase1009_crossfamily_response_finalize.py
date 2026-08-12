#!/usr/bin/env python3
"""Find repeated Phase1009 response trajectories without mechanism claims."""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1009_crossfamily_response_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLITS = ("discovery", "confirmation")
SPLIT_INDEX = {name: index for index, name in enumerate(SPLITS)}
STAGE_INDEX = {
    "prompt": 0,
    "semantic0": 1,
    "function0": 2,
    "termination": 3,
}
ABSOLUTE_FLOOR = 1e-8
HIGH_FRACTION = 0.90
MIN_QUALIFIED_PER_SPLIT = 8
MIN_POOLS_PER_SPLIT = 2
MIN_TEMPLATES = 2
CROSS_DEPTH_TOLERANCE = 0.15


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def safe_mean(values: np.ndarray) -> float | None:
    selected = finite(values)
    return None if selected.size == 0 else float(np.mean(selected))


def safe_median(values: np.ndarray) -> float | None:
    selected = finite(values)
    return None if selected.size == 0 else float(np.median(selected))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(mask)) < 3:
        return None
    x = left[mask].astype(np.float64)
    y = right[mask].astype(np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def depth_bin(relative_depth: float) -> int:
    return int(np.clip(np.floor(relative_depth * 10.0 + 1e-9), 0, 10))


def high_bins(profile: np.ndarray, relative_depths: np.ndarray) -> set[int]:
    selected = finite(profile)
    if selected.size == 0:
        return set()
    peak = float(np.max(selected))
    if peak <= ABSOLUTE_FLOOR:
        return set()
    mask = np.isfinite(profile) & (profile >= HIGH_FRACTION * peak)
    return {
        depth_bin(float(relative_depth))
        for relative_depth in relative_depths[mask]
    }


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def peak_for_group(
    values: np.ndarray,
    relative_depths: np.ndarray,
    mask: np.ndarray,
) -> tuple[int | None, float | None]:
    if int(np.sum(mask)) == 0:
        return None, None
    profile = np.nanmedian(values[mask], axis=0)
    if not np.any(np.isfinite(profile)):
        return None, None
    index = int(np.nanargmax(profile))
    return depth_bin(float(relative_depths[index])), float(profile[index])


def direct_edit_roles(family: str, operation: str) -> set[str]:
    edits = {
        "comparison": {
            "F": {
                "chain_left",
                "chain_bridge_0",
                "chain_bridge_1",
                "chain_right",
            },
            "Q": {"query_operator"},
            "E": {"nuisance_left", "nuisance_right"},
            "N": {"nuisance_left", "nuisance_right"},
        },
        "negation": {
            "F": {"focal_marker_0", "focal_marker_1"},
            "Q": {"query_operator"},
            "E": {"nuisance_entity"},
            "N": {"nuisance_marker"},
        },
        "semantic_role": {
            "F": {"focal_agent", "focal_patient", "query_anchor"},
            "Q": {"query_anchor", "query_operator"},
            "E": {"nuisance_agent", "nuisance_patient"},
            "N": {"nuisance_agent", "nuisance_patient"},
        },
    }
    if operation in ("FQ", "X"):
        return edits[family].get("F", set()) | edits[family].get("Q", set())
    return edits[family].get(operation, set())


def surface_confound_audit(motif: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if motif["component"] == "residual" and int(motif["peak_depth"]) == 0:
        reasons.append("embedding_or_direct_appended_token_difference")
    if (
        motif["stage"] in ("function0", "termination")
        and motif["operation"] in ("F", "Q", "X")
    ):
        reasons.append("teacher_forced_answer_surface_differs")
    if (
        motif["stage"] == "prompt"
        and motif["role"] in direct_edit_roles(
            motif["family"],
            motif["operation"],
        )
    ):
        reasons.append("role_token_is_directly_edited")
    return {
        "direct_surface_confounded": bool(reasons),
        "surface_confound_reasons": reasons,
        "refinement_eligible": bool(
            motif["repeated_candidate"] and not reasons
        ),
    }


def output_arrays(
    units: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    unit_index = {unit["unit_id"]: index for index, unit in enumerate(units)}
    result = {
        operation: {
            "probability_l1": np.full(len(units), np.nan, dtype=np.float32),
            "delta_margin": np.full(len(units), np.nan, dtype=np.float32),
        }
        for operation in ("F", "Q", "FQ", "E", "O", "N", "S", "I")
    }
    for row in rows:
        index = unit_index[row["unit_id"]]
        operation = row["operation"]
        result[operation]["probability_l1"][index] = float(
            row["fixed_panel_probability_l1"]
        )
        result[operation]["delta_margin"][index] = float(
            row["delta_fixed_choice_margin"]
        )
    return result


def trajectory_motifs(
    *,
    model_name: str,
    family: str,
    events: list[dict[str, Any]],
    units: list[dict[str, Any]],
    normalized: np.ndarray,
    qualified: np.ndarray,
    rollout: np.ndarray,
    consistency: np.ndarray,
    outputs: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for event in events:
        groups[
            (event["stage"], event["component"], event["role"])
        ].append(int(event["event_index"]))
    for indices in groups.values():
        indices.sort(key=lambda index: int(events[index]["depth"]))
    split_masks = {
        split: np.array(
            [unit["split"] == split for unit in units],
            dtype=np.bool_,
        )
        for split in SPLITS
    }
    template_values = sorted({int(unit["template"]) for unit in units})
    pool_values = {
        split: sorted({
            int(unit["name_pool"])
            for unit in units
            if unit["split"] == split
        })
        for split in SPLITS
    }
    motifs = []
    for operation in ("F", "Q", "FQ", "E", "O", "N", "S", "X"):
        operation_index = OP_INDEX[operation]
        qmask = qualified[:, operation_index]
        for (stage, component, role), indices in sorted(groups.items()):
            trajectory = normalized[:, operation_index, indices]
            relative_depths = np.array(
                [float(events[index]["relative_depth"]) for index in indices],
                dtype=np.float32,
            )
            depths = np.array(
                [int(events[index]["depth"]) for index in indices],
                dtype=np.int32,
            )
            split_profiles = {}
            split_high = {}
            qualified_counts = {}
            for split in SPLITS:
                mask = split_masks[split] & qmask
                qualified_counts[split] = int(np.sum(mask))
                split_profiles[split] = (
                    np.nanmedian(trajectory[mask], axis=0)
                    if qualified_counts[split] > 0
                    else np.full(len(indices), np.nan, dtype=np.float32)
                )
                split_high[split] = high_bins(
                    split_profiles[split],
                    relative_depths,
                )
            combined_profile = (
                np.nanmedian(trajectory[qmask], axis=0)
                if int(np.sum(qmask)) > 0
                else np.nanmedian(trajectory, axis=0)
            )
            if not np.any(np.isfinite(combined_profile)):
                continue
            peak_offset = int(np.nanargmax(combined_profile))
            peak_event_index = indices[peak_offset]
            peak_event = events[peak_event_index]
            peak_value = float(combined_profile[peak_offset])
            high_mask = (
                np.isfinite(combined_profile)
                & (combined_profile >= HIGH_FRACTION * max(peak_value, 0.0))
            )
            active_mask = (
                np.isfinite(combined_profile)
                & (combined_profile >= 0.10 * max(peak_value, 0.0))
            )
            active_depths = depths[active_mask]
            template_peak_bins = {}
            for template in template_values:
                mask = np.array(
                    [
                        int(unit["template"]) == template
                        for unit in units
                    ],
                    dtype=np.bool_,
                ) & qmask
                peak_bin_value, _ = peak_for_group(
                    trajectory,
                    relative_depths,
                    mask,
                )
                template_peak_bins[str(template)] = peak_bin_value
            valid_template_bins = [
                value
                for value in template_peak_bins.values()
                if value is not None
            ]
            modal_template_bin = (
                Counter(valid_template_bins).most_common(1)[0][0]
                if valid_template_bins else None
            )
            template_support = sum(
                modal_template_bin is not None
                and abs(value - modal_template_bin) <= 1
                for value in valid_template_bins
            )
            candidate_bins = (
                split_high["discovery"] | split_high["confirmation"]
            )
            pool_peak_bins: dict[str, dict[str, int | None]] = {}
            pool_support: dict[str, int] = {}
            for split in SPLITS:
                pool_peak_bins[split] = {}
                support = 0
                for pool in pool_values[split]:
                    mask = np.array(
                        [
                            unit["split"] == split
                            and int(unit["name_pool"]) == pool
                            for unit in units
                        ],
                        dtype=np.bool_,
                    ) & qmask
                    peak_bin_value, _ = peak_for_group(
                        trajectory,
                        relative_depths,
                        mask,
                    )
                    pool_peak_bins[split][str(pool)] = peak_bin_value
                    if (
                        peak_bin_value is not None
                        and peak_bin_value in candidate_bins
                    ):
                        support += 1
                pool_support[split] = support
            split_overlap = jaccard(
                split_high["discovery"],
                split_high["confirmation"],
            )
            repeated = bool(
                peak_value > ABSOLUTE_FLOOR
                and qualified_counts["discovery"] >= MIN_QUALIFIED_PER_SPLIT
                and qualified_counts["confirmation"] >= MIN_QUALIFIED_PER_SPLIT
                and split_overlap > 0.0
                and template_support >= MIN_TEMPLATES
                and pool_support["discovery"] >= MIN_POOLS_PER_SPLIT
                and pool_support["confirmation"] >= MIN_POOLS_PER_SPLIT
            )
            peak_values = trajectory[:, peak_offset]
            output_operation = "FQ" if operation == "X" else operation
            output_relation = outputs.get(output_operation)
            probability_corr = (
                None
                if output_relation is None
                else safe_corr(
                    peak_values[qmask],
                    output_relation["probability_l1"][qmask],
                )
            )
            margin_corr = (
                None
                if output_relation is None
                else safe_corr(
                    peak_values[qmask],
                    output_relation["delta_margin"][qmask],
                )
            )
            reference_operations = [
                reference
                for reference in ("E", "O", "N", "S")
                if reference != operation
            ]
            reference_medians = {
                reference: safe_median(
                    normalized[
                        :,
                        OP_INDEX[reference],
                        peak_event_index,
                    ]
                )
                for reference in reference_operations
            }
            finite_references = [
                value
                for value in reference_medians.values()
                if value is not None
            ]
            reference_median = (
                None
                if not finite_references
                else float(np.median(finite_references))
            )
            contrast_ratio = (
                None
                if reference_median is None
                else float(
                    peak_value / max(reference_median, ABSOLUTE_FLOOR)
                )
            )
            motif = {
                "schema_version": "phase1009_trajectory_motif.v1",
                "phase": PHASE,
                "model": model_name,
                "family": family,
                "motif_id": (
                    f"{model_name}.{family}.{operation}."
                    f"{stage}.{component}.{role}"
                ),
                "operation": operation,
                "stage": stage,
                "component": component,
                "role": role,
                "role_class": peak_event["role_class"],
                "trajectory_depth_count": len(indices),
                "peak_event_index": peak_event_index,
                "peak_event_id": peak_event["event_id"],
                "peak_depth": int(peak_event["depth"]),
                "peak_relative_depth": float(peak_event["relative_depth"]),
                "peak_depth_bin": depth_bin(
                    float(peak_event["relative_depth"])
                ),
                "qualified_peak_normalized_median": peak_value,
                "high90_depths": [
                    int(value)
                    for value in depths[high_mask].tolist()
                ],
                "onset10_depth": (
                    None
                    if active_depths.size == 0
                    else int(active_depths[0])
                ),
                "last10_depth": (
                    None
                    if active_depths.size == 0
                    else int(active_depths[-1])
                ),
                "persistence10_depth_count": int(active_depths.size),
                "discovery_high90_bins": sorted(
                    split_high["discovery"]
                ),
                "confirmation_high90_bins": sorted(
                    split_high["confirmation"]
                ),
                "split_high90_jaccard": split_overlap,
                "semantic_qualified_counts": qualified_counts,
                "rollout_qualified_count": int(np.sum(
                    rollout[:, operation_index]
                )),
                "rollout_qualified_rate": float(np.mean(
                    rollout[:, operation_index]
                )),
                "template_peak_bins": template_peak_bins,
                "template_modal_bin": modal_template_bin,
                "template_support_within_one_bin": int(template_support),
                "pool_peak_bins": pool_peak_bins,
                "pool_support_in_repeated_band": pool_support,
                "direction_consistency": {
                    split: (
                        None
                        if not np.isfinite(consistency[
                            operation_index,
                            SPLIT_INDEX[split],
                            peak_event_index,
                        ])
                        else float(consistency[
                            operation_index,
                            SPLIT_INDEX[split],
                            peak_event_index,
                        ])
                    )
                    for split in SPLITS
                },
                "candidate_competition": {
                    "peak_vs_fixed_probability_l1_correlation": (
                        probability_corr
                    ),
                    "peak_vs_fixed_margin_delta_correlation": margin_corr,
                    "is_causal_evidence": False,
                },
                "same_output_reference_medians": reference_medians,
                "operation_to_reference_median_ratio": contrast_ratio,
                "repeated_candidate": repeated,
                "edge_claim_allowed": "co_response_only",
                "selection_is_mechanism_proof": False,
            }
            motif.update(surface_confound_audit(motif))
            motifs.append(motif)
    return motifs


def best_depth_clusters(
    rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    remaining = list(rows)
    clusters = []
    while remaining:
        candidates = []
        ordered = sorted(
            remaining,
            key=lambda row: float(row["peak_relative_depth"]),
        )
        for start_index, start in enumerate(ordered):
            start_depth = float(start["peak_relative_depth"])
            window = [
                row
                for row in ordered[start_index:]
                if (
                    float(row["peak_relative_depth"]) - start_depth
                    <= CROSS_DEPTH_TOLERANCE
                )
            ]
            if not window:
                continue
            families = {row["family"] for row in window}
            models = {row["model"] for row in window}
            pairs = {(row["model"], row["family"]) for row in window}
            spread = (
                max(float(row["peak_relative_depth"]) for row in window)
                - min(float(row["peak_relative_depth"]) for row in window)
            )
            candidates.append((
                (len(families), len(models), len(pairs), len(window), -spread),
                window,
            ))
        if not candidates:
            break
        _, best = max(candidates, key=lambda item: item[0])
        clusters.append(best)
        selected_ids = {row["motif_id"] for row in best}
        remaining = [
            row for row in remaining
            if row["motif_id"] not in selected_ids
        ]
    return clusters


def cross_family_groups(
    all_motifs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for motif in all_motifs:
        if not motif["refinement_eligible"]:
            continue
        grouped[(
            motif["operation"],
            motif["stage"],
            motif["component"],
            motif["role_class"],
        )].append(motif)
    result = []
    group_index = 0
    for key, rows in sorted(grouped.items()):
        for cluster in best_depth_clusters(rows):
            families = sorted({row["family"] for row in cluster})
            models = sorted({row["model"] for row in cluster})
            pairs = sorted({
                f"{row['model']}:{row['family']}"
                for row in cluster
            })
            if len(families) < 2:
                continue
            depths = [
                float(row["peak_relative_depth"]) for row in cluster
            ]
            group_index += 1
            result.append({
                "schema_version": "phase1009_cross_family_motif.v1",
                "phase": PHASE,
                "cross_family_motif_id": (
                    f"cf{group_index:04d}."
                    + ".".join(key)
                ),
                "operation": key[0],
                "stage": key[1],
                "component": key[2],
                "role_class": key[3],
                "families": families,
                "family_count": len(families),
                "models": models,
                "model_count": len(models),
                "model_family_pairs": pairs,
                "model_family_pair_count": len(pairs),
                "member_motif_ids": [
                    row["motif_id"] for row in cluster
                ],
                "peak_relative_depth_min": min(depths),
                "peak_relative_depth_median": float(np.median(depths)),
                "peak_relative_depth_max": max(depths),
                "relative_depth_spread": max(depths) - min(depths),
                "strong_cross_family_cross_model": bool(
                    len(families) >= 2 and len(models) >= 2
                ),
                "alignment_basis": (
                    "operation + autoregressive stage + component + "
                    "role class + relative depth tolerance"
                ),
                "raw_coordinate_or_vector_comparison_used": False,
                "claim": (
                    "coordinate-free repeated response shape; not transport "
                    "or shared mechanism proof"
                ),
            })
    return result


def reuse_rows(
    motifs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for model_name in MODELS:
        for family in FAMILIES:
            selected = [
                row for row in motifs
                if row["model"] == model_name
                and row["family"] == family
                and row["refinement_eligible"]
            ]
            operation_sets = {}
            for operation in ("F", "Q", "FQ", "X"):
                operation_sets[operation] = {
                    (
                        row["stage"],
                        row["component"],
                        row["role_class"],
                        int(row["peak_depth_bin"]),
                    )
                    for row in selected
                    if row["operation"] == operation
                }
            for left, right in itertools.combinations(
                ("F", "Q", "FQ", "X"),
                2,
            ):
                result.append({
                    "schema_version": "phase1009_operation_reuse.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "family": family,
                    "left_operation": left,
                    "right_operation": right,
                    "left_count": len(operation_sets[left]),
                    "right_count": len(operation_sets[right]),
                    "intersection_count": len(
                        operation_sets[left] & operation_sets[right]
                    ),
                    "jaccard": jaccard(
                        operation_sets[left],
                        operation_sets[right],
                    ),
                    "interpretation": (
                        "physical-bin reuse after repetition and surface "
                        "audit; not shared causal function"
                    ),
                })
    return result


def stage_depth_summary(
    motifs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in motifs:
        if row["refinement_eligible"]:
            grouped[
                (
                    row["model"],
                    row["family"],
                    row["operation"],
                    row["stage"],
                )
            ].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        depths = np.array(
            [float(row["peak_relative_depth"]) for row in rows],
            dtype=np.float64,
        )
        result.append({
            "schema_version": "phase1009_stage_depth_summary.v1",
            "phase": PHASE,
            "model": key[0],
            "family": key[1],
            "operation": key[2],
            "stage": key[3],
            "eligible_motif_count": len(rows),
            "early_count": int(np.sum(depths < 1.0 / 3.0)),
            "middle_count": int(np.sum(
                (depths >= 1.0 / 3.0) & (depths < 2.0 / 3.0)
            )),
            "late_count": int(np.sum(depths >= 2.0 / 3.0)),
            "peak_relative_depth_median": float(np.median(depths)),
        })
    return result


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    if int(protocol["protocol_revision"]) != 1:
        raise RuntimeError("Phase1009 protocol revision drift")
    final_root = OUT_ROOT / "final"
    all_motifs = []
    model_family_summaries = {}
    scalar_total = 0
    for model_name in MODELS:
        scan_model_summary = read_json(
            OUT_ROOT / "scan" / model_name / "summary.json"
        )
        if scan_model_summary["protocol_digest"] != (
            protocol["preregistration_digest"]
        ):
            raise RuntimeError(f"{model_name}: scan protocol drift")
        scalar_total += int(scan_model_summary["scalar_measurement_count"])
        for family in FAMILIES:
            scan_root = OUT_ROOT / "scan" / model_name / family
            arrays = np.load(scan_root / "response_scalars.npz")
            direction = np.load(
                scan_root / "direction_consistency.npz"
            )
            events = read_jsonl(scan_root / "events.jsonl")
            units = read_jsonl(scan_root / "units.jsonl")
            output_rows = read_jsonl(scan_root / "output_pairs.jsonl")
            motifs = trajectory_motifs(
                model_name=model_name,
                family=family,
                events=events,
                units=units,
                normalized=arrays["normalized_magnitude"],
                qualified=arrays["semantic_qualified"],
                rollout=arrays["rollout_qualified"],
                consistency=direction["direction_consistency"],
                outputs=output_arrays(units, output_rows),
            )
            all_motifs.extend(motifs)
            target_root = final_root / model_name / family
            write_jsonl(target_root / "trajectory_motifs.jsonl", motifs)
            key = f"{model_name}:{family}"
            model_family_summaries[key] = {
                "trajectory_count": len(motifs),
                "repeated_candidate_count": int(sum(
                    row["repeated_candidate"] for row in motifs
                )),
                "surface_confounded_repeated_count": int(sum(
                    row["repeated_candidate"]
                    and row["direct_surface_confounded"]
                    for row in motifs
                )),
                "refinement_eligible_count": int(sum(
                    row["refinement_eligible"] for row in motifs
                )),
                "eligible_by_operation": {
                    operation: int(sum(
                        row["refinement_eligible"]
                        and row["operation"] == operation
                        for row in motifs
                    ))
                    for operation in (
                        "F",
                        "Q",
                        "FQ",
                        "E",
                        "O",
                        "N",
                        "S",
                        "X",
                    )
                },
            }
    cross_family = cross_family_groups(all_motifs)
    reuse = reuse_rows(all_motifs)
    stage_summary = stage_depth_summary(all_motifs)
    write_jsonl(final_root / "all_trajectory_motifs.jsonl", all_motifs)
    write_jsonl(final_root / "cross_family_motifs.jsonl", cross_family)
    write_jsonl(final_root / "operation_reuse.jsonl", reuse)
    write_jsonl(final_root / "stage_depth_summary.jsonl", stage_summary)
    strong = [
        row for row in cross_family
        if row["strong_cross_family_cross_model"]
    ]
    shared_decision_candidates = [
        row for row in strong
        if row["stage"] == "semantic0"
        and row["component"] == "attention_output"
        and float(row["peak_relative_depth_median"]) >= 2.0 / 3.0
        and row["operation"] in ("F", "Q", "FQ", "X")
    ]
    family_support = Counter(
        family
        for row in strong
        for family in row["families"]
    )
    model_support = Counter(
        model
        for row in strong
        for model in row["models"]
    )
    summary = {
        "schema_version": "phase1009_final_summary.v1",
        "phase": PHASE,
        "protocol_digest": protocol["preregistration_digest"],
        "scalar_measurement_count": scalar_total,
        "raw_hidden_tensors_persisted": 0,
        "trajectory_count": len(all_motifs),
        "repeated_candidate_count": int(sum(
            row["repeated_candidate"] for row in all_motifs
        )),
        "refinement_eligible_count": int(sum(
            row["refinement_eligible"] for row in all_motifs
        )),
        "model_family_summaries": model_family_summaries,
        "cross_family_motif_count": len(cross_family),
        "strong_cross_family_cross_model_count": len(strong),
        "strong_family_support": dict(sorted(family_support.items())),
        "strong_model_support": dict(sorted(model_support.items())),
        "late_semantic0_shared_decision_candidate_count": len(
            shared_decision_candidates
        ),
        "late_semantic0_shared_decision_candidate_ids": [
            row["cross_family_motif_id"]
            for row in shared_decision_candidates
        ],
        "operation_reuse_mean_jaccard": {
            model: {
                family: safe_mean(np.array([
                    row["jaccard"]
                    for row in reuse
                    if row["model"] == model
                    and row["family"] == family
                ], dtype=np.float64))
                for family in FAMILIES
            }
            for model in MODELS
        },
        "phase_gates": {
            "G1_behavior": (
                "reported per model/family; no global pass is inferred"
            ),
            "G2_more_than_one_million_scalars": bool(
                scalar_total > 1_000_000
            ),
            "G3_repeated_candidates_in_two_families": bool(
                sum(
                    model_family_summaries[
                        f"{model}:{family}"
                    ]["refinement_eligible_count"] > 0
                    for family in FAMILIES
                    for model in MODELS
                ) >= 2
            ),
        },
        "supported_claim_ceiling": (
            "repeated coordinate-free response shapes across controlled "
            "language families; no transport, mediation, shared decision "
            "mechanism, or language formula is established"
        ),
        "automatic_next_step_rule": {
            "eligible": bool(shared_decision_candidates),
            "action": (
                "replicate the Phase1008 discovery-frozen late semantic0 "
                "head sets on Phase1009 held-out confirmation families"
                if shared_decision_candidates
                else (
                    "do not run causal tests; improve family protocol or "
                    "response discovery first"
                )
            ),
        },
    }
    write_json(final_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
