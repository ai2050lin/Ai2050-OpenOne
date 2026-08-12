#!/usr/bin/env python3
"""Describe repeated Phase1010 response topology across output types.

All thresholds are treated as measurement settings and audited at three
values. The output intentionally separates scalar repetition, direction
similarity, and causal evidence.
"""
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

from phase1009_crossfamily_response_finalize import (
    ABSOLUTE_FLOOR,
    CROSS_DEPTH_TOLERANCE,
    MIN_POOLS_PER_SPLIT,
    MIN_QUALIFIED_PER_SPLIT,
    MIN_TEMPLATES,
    depth_bin,
    direct_edit_roles,
    jaccard,
    peak_for_group,
    safe_median,
)
from phase1010_output_type_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    OUTPUT_TYPES,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLITS = ("discovery", "confirmation")
SPLIT_INDEX = {name: index for index, name in enumerate(SPLITS)}
HIGH_FRACTIONS = (0.85, 0.90, 0.95)
CONTROL_OPERATIONS = ("E", "O", "N", "S")


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def high_bins(
    profile: np.ndarray,
    relative_depths: np.ndarray,
    fraction: float,
) -> set[int]:
    selected = finite(profile)
    if selected.size == 0:
        return set()
    peak = float(np.max(selected))
    if peak <= ABSOLUTE_FLOOR:
        return set()
    mask = np.isfinite(profile) & (profile >= fraction * peak)
    return {
        depth_bin(float(relative_depth))
        for relative_depth in relative_depths[mask]
    }


def surface_confound_reasons(
    family: str,
    operation: str,
    stage: str,
    component: str,
    role: str,
    peak_depth: int,
) -> list[str]:
    reasons: list[str] = []
    if component == "residual" and peak_depth == 0:
        reasons.append("embedding_or_direct_token_difference")
    if stage in ("function0", "termination") and operation in (
        "F",
        "Q",
        "X",
    ):
        reasons.append("teacher_forced_answer_surface_differs")
    if (
        stage == "prompt"
        and role in direct_edit_roles(family, operation)
    ):
        reasons.append("measured_role_is_directly_edited")
    return reasons


def panel_motifs(
    *,
    model_name: str,
    family: str,
    output_type: str,
    panel_root: Path,
) -> list[dict[str, Any]]:
    events = read_jsonl(panel_root / "events.jsonl")
    units = read_jsonl(panel_root / "units.jsonl")
    with np.load(panel_root / "response_scalars.npz") as payload:
        normalized = payload["normalized_magnitude"]
        qualified = payload["semantic_qualified"]
    with np.load(panel_root / "direction_consistency.npz") as payload:
        consistency = payload["direction_consistency"]

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
    motifs: list[dict[str, Any]] = []
    for operation in (
        "F",
        "Q",
        "FQ",
        "E",
        "O",
        "N",
        "S",
        "X",
    ):
        operation_index = OP_INDEX[operation]
        qmask = qualified[:, operation_index]
        for (stage, component, role), indices in sorted(groups.items()):
            trajectory = normalized[:, operation_index, indices]
            relative_depths = np.array(
                [float(events[index]["relative_depth"]) for index in indices],
                dtype=np.float32,
            )
            split_profiles: dict[str, np.ndarray] = {}
            qualified_counts: dict[str, int] = {}
            for split in SPLITS:
                mask = split_masks[split] & qmask
                qualified_counts[split] = int(np.sum(mask))
                split_profiles[split] = (
                    np.nanmedian(trajectory[mask], axis=0)
                    if qualified_counts[split] > 0
                    else np.full(len(indices), np.nan, dtype=np.float32)
                )
            if int(np.sum(qmask)) == 0:
                continue
            combined_profile = np.nanmedian(trajectory[qmask], axis=0)
            if not np.any(np.isfinite(combined_profile)):
                continue
            peak_offset = int(np.nanargmax(combined_profile))
            peak_event_index = int(indices[peak_offset])
            peak_event = events[peak_event_index]
            peak_value = float(combined_profile[peak_offset])

            template_peak_bins: dict[str, int | None] = {}
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
                if valid_template_bins
                else None
            )
            template_support = sum(
                modal_template_bin is not None
                and abs(value - modal_template_bin) <= 1
                for value in valid_template_bins
            )

            pool_peak_bins: dict[str, dict[str, int | None]] = {}
            for split in SPLITS:
                pool_peak_bins[split] = {}
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

            threshold_results: dict[str, dict[str, Any]] = {}
            for fraction in HIGH_FRACTIONS:
                discovery_high = high_bins(
                    split_profiles["discovery"],
                    relative_depths,
                    fraction,
                )
                confirmation_high = high_bins(
                    split_profiles["confirmation"],
                    relative_depths,
                    fraction,
                )
                candidate_bins = discovery_high | confirmation_high
                pool_support = {
                    split: int(sum(
                        value is not None and value in candidate_bins
                        for value in pool_peak_bins[split].values()
                    ))
                    for split in SPLITS
                }
                overlap = jaccard(discovery_high, confirmation_high)
                repeated = bool(
                    peak_value > ABSOLUTE_FLOOR
                    and qualified_counts["discovery"]
                    >= MIN_QUALIFIED_PER_SPLIT
                    and qualified_counts["confirmation"]
                    >= MIN_QUALIFIED_PER_SPLIT
                    and overlap > 0.0
                    and template_support >= MIN_TEMPLATES
                    and pool_support["discovery"]
                    >= MIN_POOLS_PER_SPLIT
                    and pool_support["confirmation"]
                    >= MIN_POOLS_PER_SPLIT
                )
                threshold_results[f"{fraction:.2f}"] = {
                    "discovery_high_bins": sorted(discovery_high),
                    "confirmation_high_bins": sorted(confirmation_high),
                    "split_high_bin_jaccard": float(overlap),
                    "pool_support": pool_support,
                    "repeated_candidate": repeated,
                }
            pool_support = threshold_results["0.90"]["pool_support"]

            control_values = []
            control_detail: dict[str, float | None] = {}
            for control in CONTROL_OPERATIONS:
                control_index = OP_INDEX[control]
                control_mask = qualified[:, control_index]
                value = safe_median(
                    normalized[
                        control_mask,
                        control_index,
                        peak_event_index,
                    ]
                )
                control_detail[control] = value
                if value is not None:
                    control_values.append(value)
            control_median = (
                float(np.median(control_values))
                if control_values
                else None
            )
            control_ratio = (
                None
                if control_median is None
                else float(
                    peak_value / max(control_median, ABSOLUTE_FLOOR)
                )
            )
            reasons = surface_confound_reasons(
                family,
                operation,
                stage,
                component,
                role,
                int(peak_event["depth"]),
            )
            motifs.append({
                "schema_version": "phase1010_response_motif.v1",
                "phase": PHASE,
                "model": model_name,
                "family": family,
                "output_type": output_type,
                "operation": operation,
                "stage": stage,
                "component": component,
                "role": role,
                "role_class": peak_event["role_class"],
                "peak_event_index": peak_event_index,
                "peak_depth": int(peak_event["depth"]),
                "peak_relative_depth": float(
                    peak_event["relative_depth"]
                ),
                "peak_normalized_magnitude": peak_value,
                "qualified_counts": qualified_counts,
                "template_peak_bins": template_peak_bins,
                "template_support": int(template_support),
                "pool_peak_bins": pool_peak_bins,
                "pool_support": pool_support,
                "threshold_results": threshold_results,
                "repeated_at_0_90": bool(
                    threshold_results["0.90"]["repeated_candidate"]
                ),
                "direction_consistency": {
                    split: (
                        None
                        if not np.isfinite(
                            consistency[
                                operation_index,
                                SPLIT_INDEX[split],
                                peak_event_index,
                            ]
                        )
                        else float(
                            consistency[
                                operation_index,
                                SPLIT_INDEX[split],
                                peak_event_index,
                            ]
                        )
                    )
                    for split in SPLITS
                },
                "control_peak_medians": control_detail,
                "control_median": control_median,
                "target_to_control_median_ratio": control_ratio,
                "direct_surface_confounded": bool(reasons),
                "surface_confound_reasons": reasons,
                "claim_limit": (
                    "repeated response topology only; not a transport edge "
                    "or a causal mechanism"
                ),
            })
    return motifs


def load_direction_rows(
    model_name: str,
    family: str,
    output_type: str,
    panel_root: Path,
) -> list[dict[str, Any]]:
    metadata = read_jsonl(
        panel_root / "peak_direction_metadata.jsonl"
    )
    with np.load(panel_root / "peak_direction_centroids.npz") as payload:
        centroids = payload["centroids"].astype(np.float32)
    rows = []
    for row in metadata:
        index = int(row["centroid_index"])
        result = dict(row)
        result["_centroid"] = centroids[index]
        rows.append(result)
    return rows


def cross_output_matches(
    motifs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for motif in motifs:
        groups[(
            motif["model"],
            motif["family"],
            motif["operation"],
            motif["stage"],
            motif["component"],
            motif["role"],
        )].append(motif)
    rows = []
    for key, values in sorted(groups.items()):
        repeated = {
            row["output_type"]: row
            for row in values
            if row["repeated_at_0_90"]
        }
        support = sorted(repeated)
        nonperson = sorted(set(support) - {"person"})
        depths = [
            float(row["peak_relative_depth"])
            for row in repeated.values()
        ]
        rows.append({
            "schema_version": "phase1010_cross_output_match.v1",
            "phase": PHASE,
            "model": key[0],
            "family": key[1],
            "operation": key[2],
            "stage": key[3],
            "component": key[4],
            "role": key[5],
            "output_type_support": support,
            "output_type_support_count": len(support),
            "nonperson_support": nonperson,
            "nonperson_support_count": len(nonperson),
            "person_supported": "person" in repeated,
            "peak_relative_depth_span": (
                None if not depths else float(max(depths) - min(depths))
            ),
            "depth_aligned_within_tolerance": bool(
                len(depths) >= 2
                and max(depths) - min(depths)
                <= CROSS_DEPTH_TOLERANCE
            ),
            "cross_output_response_candidate": bool(
                len(support) >= 2 and len(nonperson) >= 1
            ),
            "all_output_types_repeat": len(support) == len(OUTPUT_TYPES),
            "claim_limit": (
                "shared response location/shape; output-independent "
                "mechanism is not established"
            ),
        })
    return rows


def direction_pair_rows(
    direction_rows: list[dict[str, Any]],
    repeated_lookup: dict[tuple, bool],
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in direction_rows:
        groups[(
            row["model"],
            row["family"],
            row["operation"],
            row["split"],
            row["stage"],
            row["component"],
            row["role"],
        )].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        by_output = {row["output_type"]: row for row in rows}
        for left_type, right_type in itertools.combinations(
            sorted(by_output),
            2,
        ):
            left = by_output[left_type]
            right = by_output[right_type]
            cosine = float(np.dot(
                left["_centroid"],
                right["_centroid"],
            ))
            left_repeated = repeated_lookup.get((
                key[0],
                key[1],
                left_type,
                key[2],
                key[4],
                key[5],
                key[6],
            ), False)
            right_repeated = repeated_lookup.get((
                key[0],
                key[1],
                right_type,
                key[2],
                key[4],
                key[5],
                key[6],
            ), False)
            result.append({
                "schema_version": "phase1010_direction_pair.v1",
                "phase": PHASE,
                "model": key[0],
                "family": key[1],
                "operation": key[2],
                "split": key[3],
                "stage": key[4],
                "component": key[5],
                "role": key[6],
                "left_output_type": left_type,
                "right_output_type": right_type,
                "left_peak_relative_depth": float(
                    left["peak_relative_depth"]
                ),
                "right_peak_relative_depth": float(
                    right["peak_relative_depth"]
                ),
                "relative_depth_gap": float(abs(
                    left["peak_relative_depth"]
                    - right["peak_relative_depth"]
                )),
                "direction_cosine": cosine,
                "left_direction_concentration": float(
                    left["direction_concentration"]
                ),
                "right_direction_concentration": float(
                    right["direction_concentration"]
                ),
                "both_scalar_repeated_at_0_90": bool(
                    left_repeated and right_repeated
                ),
                "claim_limit": (
                    "aggregate direction similarity is not information "
                    "transport or causal equivalence"
                ),
            })
    return result


def cross_model_matches(
    motifs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for motif in motifs:
        if motif["repeated_at_0_90"]:
            groups[(
                motif["family"],
                motif["output_type"],
                motif["operation"],
                motif["stage"],
                motif["component"],
                motif["role"],
            )].append(motif)
    rows = []
    for key, values in sorted(groups.items()):
        by_model = {row["model"]: row for row in values}
        depths = [
            float(row["peak_relative_depth"])
            for row in by_model.values()
        ]
        rows.append({
            "schema_version": "phase1010_cross_model_match.v1",
            "phase": PHASE,
            "family": key[0],
            "output_type": key[1],
            "operation": key[2],
            "stage": key[3],
            "component": key[4],
            "role": key[5],
            "model_support": sorted(by_model),
            "model_support_count": len(by_model),
            "peak_relative_depth_span": float(
                max(depths) - min(depths)
            ),
            "depth_aligned_within_tolerance": bool(
                len(depths) >= 2
                and max(depths) - min(depths)
                <= CROSS_DEPTH_TOLERANCE
            ),
            "claim_limit": (
                "cross-model topological repetition only; hidden "
                "directions are not compared across coordinate systems"
            ),
        })
    return rows


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    motifs: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        model_root = OUT_ROOT / "scan" / model_name
        summary = read_json(model_root / "summary.json")
        if summary["protocol_digest"] != protocol["preregistration_digest"]:
            raise RuntimeError(f"{model_name}: scan/protocol drift")
        for output_type in OUTPUT_TYPES:
            for family in FAMILIES:
                panel_root = model_root / output_type / family
                motifs.extend(panel_motifs(
                    model_name=model_name,
                    family=family,
                    output_type=output_type,
                    panel_root=panel_root,
                ))
                direction_rows.extend(load_direction_rows(
                    model_name,
                    family,
                    output_type,
                    panel_root,
                ))

    repeated_lookup = {
        (
            row["model"],
            row["family"],
            row["output_type"],
            row["operation"],
            row["stage"],
            row["component"],
            row["role"],
        ): bool(row["repeated_at_0_90"])
        for row in motifs
    }
    output_matches = cross_output_matches(motifs)
    direction_pairs = direction_pair_rows(
        direction_rows,
        repeated_lookup,
    )
    model_matches = cross_model_matches(motifs)

    analysis_root = OUT_ROOT / "analysis"
    write_jsonl(analysis_root / "response_motifs.jsonl", motifs)
    write_jsonl(
        analysis_root / "cross_output_matches.jsonl",
        output_matches,
    )
    write_jsonl(
        analysis_root / "cross_output_direction_pairs.jsonl",
        direction_pairs,
    )
    write_jsonl(
        analysis_root / "cross_model_matches.jsonl",
        model_matches,
    )

    threshold_counts = {
        f"{fraction:.2f}": int(sum(
            row["threshold_results"][f"{fraction:.2f}"][
                "repeated_candidate"
            ]
            for row in motifs
        ))
        for fraction in HIGH_FRACTIONS
    }
    repeated_90 = [
        row for row in motifs if row["repeated_at_0_90"]
    ]
    unconfounded_repeated = [
        row
        for row in repeated_90
        if not row["direct_surface_confounded"]
    ]
    cross_output_candidates = [
        row
        for row in output_matches
        if row["cross_output_response_candidate"]
    ]
    all_output = [
        row for row in output_matches if row["all_output_types_repeat"]
    ]
    aligned_cross_model = [
        row
        for row in model_matches
        if row["model_support_count"] >= 2
        and row["depth_aligned_within_tolerance"]
    ]
    informative_direction_pairs = [
        row
        for row in direction_pairs
        if row["both_scalar_repeated_at_0_90"]
        and row["relative_depth_gap"] <= CROSS_DEPTH_TOLERANCE
    ]
    direction_cosines = np.array(
        [
            row["direction_cosine"]
            for row in informative_direction_pairs
        ],
        dtype=np.float64,
    )
    summary = {
        "schema_version": "phase1010_analysis_summary.v1",
        "phase": PHASE,
        "protocol_digest": protocol["preregistration_digest"],
        "motif_count": len(motifs),
        "threshold_sensitivity_repeated_counts": threshold_counts,
        "repeated_at_0_90_count": len(repeated_90),
        "unconfounded_repeated_at_0_90_count": len(
            unconfounded_repeated
        ),
        "cross_output_response_candidate_count": len(
            cross_output_candidates
        ),
        "all_four_output_types_repeat_count": len(all_output),
        "cross_model_depth_aligned_count": len(aligned_cross_model),
        "informative_cross_output_direction_pair_count": len(
            informative_direction_pairs
        ),
        "informative_direction_cosine": {
            "median": (
                None
                if direction_cosines.size == 0
                else float(np.median(direction_cosines))
            ),
            "q25": (
                None
                if direction_cosines.size == 0
                else float(np.quantile(direction_cosines, 0.25))
            ),
            "q75": (
                None
                if direction_cosines.size == 0
                else float(np.quantile(direction_cosines, 0.75))
            ),
            "minimum": (
                None
                if direction_cosines.size == 0
                else float(np.min(direction_cosines))
            ),
            "maximum": (
                None
                if direction_cosines.size == 0
                else float(np.max(direction_cosines))
            ),
        },
        "interpretation": {
            "scalar_repetition": (
                "stable location/shape of a within-output response"
            ),
            "direction_similarity": (
                "similar aggregate displacement in one model coordinate "
                "system; not transport"
            ),
            "causal_evidence": (
                "not measured by this finalizer and must remain separate"
            ),
            "formula_policy": (
                "thresholds describe observed repetition and were "
                "sensitivity-audited; no mechanism formula was fitted"
            ),
        },
    }
    write_json(analysis_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
