#!/usr/bin/env python3
"""Reconstruct repeated Phase1008 trajectories without causal overclaiming."""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1008_global_response_atlas_protocol import (
    ANALYSIS_OPERATIONS,
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
STAGE_ORDER = {
    "prompt": 0,
    "semantic0": 1,
    "semantic1": 2,
    "termination": 3,
}
ABSOLUTE_FLOOR = 1e-8
HIGH_FRACTION = 0.90
MIN_QUALIFIED_PER_SPLIT = 8
MIN_POOLS_PER_SPLIT = 2
MIN_TEMPLATES = 2


def direct_edit_roles(operation: str) -> set[str]:
    if operation == "B":
        return {
            "fact_value_0_word0",
            "fact_value_0_word1",
            "fact_value_1_word0",
            "fact_value_1_word1",
        }
    if operation in ("Q",):
        return {"query_entity"}
    if operation in ("BQ", "X"):
        return {
            "fact_value_0_word0",
            "fact_value_0_word1",
            "fact_value_1_word0",
            "fact_value_1_word1",
            "query_entity",
        }
    if operation == "E":
        return {
            "fact_entity_0",
            "fact_entity_1",
            "nuisance_entity",
            "query_entity",
        }
    if operation == "N":
        return {
            "nuisance_entity",
            "nuisance_value_word0",
            "nuisance_value_word1",
        }
    return set()


def surface_confound_audit(motif: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if motif["component"] == "residual" and int(motif["peak_depth"]) == 0:
        reasons.append("embedding_or_direct_appended_token_difference")
    if (
        motif["stage"] in ("semantic1", "termination")
        and motif["operation"] in ("B", "Q", "X")
    ):
        reasons.append("teacher_forced_answer_surface_differs")
    if (
        motif["stage"] == "prompt"
        and motif["role"] in direct_edit_roles(motif["operation"])
    ):
        reasons.append("role_is_directly_edited_by_operation")
    return {
        "direct_surface_confounded": bool(reasons),
        "surface_confound_reasons": reasons,
        "refinement_eligible": bool(
            motif["repeated_candidate"] and not reasons
        ),
    }


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def safe_mean(values: np.ndarray) -> float | None:
    selected = finite(values)
    return None if selected.size == 0 else float(np.mean(selected))


def safe_median(values: np.ndarray) -> float | None:
    selected = finite(values)
    return None if selected.size == 0 else float(np.median(selected))


def safe_quantile(values: np.ndarray, q: float) -> float | None:
    selected = finite(values)
    return None if selected.size == 0 else float(np.quantile(selected, q))


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
        depth_bin(float(value))
        for value in relative_depths[mask]
    }


def jaccard(left: set[int], right: set[int]) -> float:
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


def event_observations(
    model_name: str,
    events: list[dict[str, Any]],
    units: list[dict[str, Any]],
    normalized: np.ndarray,
    raw: np.ndarray,
    qualified: np.ndarray,
    consistency: np.ndarray,
) -> list[dict[str, Any]]:
    split_masks = {
        split: np.array(
            [unit["split"] == split for unit in units], dtype=np.bool_
        )
        for split in SPLITS
    }
    identity = normalized[:, OP_INDEX["I"], :]
    rows = []
    for operation in ANALYSIS_OPERATIONS:
        operation_index = OP_INDEX[operation]
        for event in events:
            event_index = int(event["event_index"])
            values = normalized[:, operation_index, event_index]
            raw_values = raw[:, operation_index, event_index]
            qmask = qualified[:, operation_index]
            identity_values = identity[:, event_index]
            numerical_floor = max(
                ABSOLUTE_FLOOR,
                float(np.nanquantile(identity_values, 0.99))
                if np.any(np.isfinite(identity_values))
                else ABSOLUTE_FLOOR,
            )
            rows.append({
                "schema_version": "phase1008_event_observation.v1",
                "phase": PHASE,
                "model": model_name,
                "operation": operation,
                **event,
                "n": int(np.sum(np.isfinite(values))),
                "semantic_qualified_n": int(np.sum(qmask)),
                "raw_magnitude_mean": safe_mean(raw_values),
                "normalized_magnitude_mean": safe_mean(values),
                "normalized_magnitude_median": safe_median(values),
                "normalized_magnitude_q25": safe_quantile(values, 0.25),
                "normalized_magnitude_q75": safe_quantile(values, 0.75),
                "qualified_normalized_median": safe_median(values[qmask]),
                "discovery_normalized_median": safe_median(
                    values[split_masks["discovery"]]
                ),
                "confirmation_normalized_median": safe_median(
                    values[split_masks["confirmation"]]
                ),
                "above_identity_floor_rate": float(np.mean(
                    values > numerical_floor
                )),
                "identity_q99_floor": numerical_floor,
                "direction_consistency_discovery": (
                    None
                    if not np.isfinite(
                        consistency[
                            operation_index,
                            SPLIT_INDEX["discovery"],
                            event_index,
                        ]
                    )
                    else float(consistency[
                        operation_index,
                        SPLIT_INDEX["discovery"],
                        event_index,
                    ])
                ),
                "direction_consistency_confirmation": (
                    None
                    if not np.isfinite(
                        consistency[
                            operation_index,
                            SPLIT_INDEX["confirmation"],
                            event_index,
                        ]
                    )
                    else float(consistency[
                        operation_index,
                        SPLIT_INDEX["confirmation"],
                        event_index,
                    ])
                ),
                "interpretation_limit": (
                    "response observation only; not a feature, transport "
                    "edge, mediator, or causal mechanism"
                ),
            })
    return rows


def output_arrays(
    units: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    unit_index = {unit["unit_id"]: index for index, unit in enumerate(units)}
    result = {
        operation: {
            "probability_l1": np.full(len(units), np.nan, dtype=np.float32),
            "delta_fixed_margin": np.full(len(units), np.nan, dtype=np.float32),
        }
        for operation in ("B", "Q", "BQ", "E", "O", "N", "I")
    }
    for row in rows:
        index = unit_index[row["unit_id"]]
        operation = row["operation"]
        result[operation]["probability_l1"][index] = float(
            row["fixed_panel_probability_l1"]
        )
        result[operation]["delta_fixed_margin"][index] = float(
            row["delta_fixed_choice_margin"]
        )
    return result


def trajectory_motifs(
    model_name: str,
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
    for values in groups.values():
        values.sort(key=lambda index: int(events[index]["depth"]))

    split_masks = {
        split: np.array(
            [unit["split"] == split for unit in units], dtype=np.bool_
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
    for operation in ("B", "Q", "BQ", "E", "O", "N", "X"):
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
                    split_profiles[split], relative_depths
                )
            combined_mask = qmask
            combined_profile = (
                np.nanmedian(trajectory[combined_mask], axis=0)
                if int(np.sum(combined_mask)) > 0
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
            ten_percent = (
                np.isfinite(combined_profile)
                & (combined_profile >= 0.10 * max(peak_value, 0.0))
            )
            active_depths = depths[ten_percent]

            template_peak_bins = {}
            for template in template_values:
                mask = np.array(
                    [int(unit["template"]) == template for unit in units],
                    dtype=np.bool_,
                ) & qmask
                peak_bin, _ = peak_for_group(
                    trajectory, relative_depths, mask
                )
                template_peak_bins[str(template)] = peak_bin
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

            pool_peak_bins: dict[str, dict[str, int | None]] = {}
            pool_support: dict[str, int] = {}
            candidate_bins = (
                split_high["discovery"] | split_high["confirmation"]
            )
            for split in SPLITS:
                pool_peak_bins[split] = {}
                support = 0
                for pool in pool_values[split]:
                    mask = np.array([
                        unit["split"] == split
                        and int(unit["name_pool"]) == pool
                        for unit in units
                    ], dtype=np.bool_) & qmask
                    peak_bin, _ = peak_for_group(
                        trajectory, relative_depths, mask
                    )
                    pool_peak_bins[split][str(pool)] = peak_bin
                    if peak_bin is not None and peak_bin in candidate_bins:
                        support += 1
                pool_support[split] = support

            split_overlap = jaccard(
                split_high["discovery"], split_high["confirmation"]
            )
            repeated_candidate = bool(
                peak_value > ABSOLUTE_FLOOR
                and qualified_counts["discovery"] >= MIN_QUALIFIED_PER_SPLIT
                and qualified_counts["confirmation"] >= MIN_QUALIFIED_PER_SPLIT
                and split_overlap > 0.0
                and template_support >= MIN_TEMPLATES
                and pool_support["discovery"] >= MIN_POOLS_PER_SPLIT
                and pool_support["confirmation"] >= MIN_POOLS_PER_SPLIT
            )
            peak_values = trajectory[:, peak_offset]
            output_operation = "BQ" if operation == "X" else operation
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
                    output_relation["delta_fixed_margin"][qmask],
                )
            )
            reference_medians = {}
            if not role.startswith("nuisance"):
                for reference in ("E", "O", "N"):
                    reference_medians[reference] = safe_median(
                        normalized[:, OP_INDEX[reference], peak_event_index]
                    )
            reference_values = [
                value
                for value in reference_medians.values()
                if value is not None
            ]
            reference_median = (
                float(np.median(reference_values))
                if reference_values else None
            )
            contrast_ratio = (
                None
                if reference_median is None
                else float(peak_value / max(reference_median, ABSOLUTE_FLOOR))
            )
            motif = {
                "schema_version": "phase1008_trajectory_motif.v1",
                "phase": PHASE,
                "model": model_name,
                "motif_id": (
                    f"{model_name}.{operation}.{stage}.{component}.{role}"
                ),
                "operation": operation,
                "stage": stage,
                "component": component,
                "role": role,
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
                    int(value) for value in depths[high_mask].tolist()
                ],
                "onset10_depth": (
                    None if active_depths.size == 0
                    else int(active_depths[0])
                ),
                "last10_depth": (
                    None if active_depths.size == 0
                    else int(active_depths[-1])
                ),
                "persistence10_depth_count": int(active_depths.size),
                "discovery_high90_bins": sorted(split_high["discovery"]),
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
                    "peak_response_vs_fixed_probability_l1_correlation": (
                        probability_corr
                    ),
                    "peak_response_vs_fixed_margin_delta_correlation": (
                        margin_corr
                    ),
                    "is_causal_evidence": False,
                },
                "same_output_reference_medians": reference_medians,
                "operation_to_reference_median_ratio": contrast_ratio,
                "repeated_candidate": repeated_candidate,
                "evidence_axes": {
                    "O_observation": {
                        "n": int(np.sum(np.isfinite(peak_values))),
                        "peak_normalized_median": peak_value,
                    },
                    "R_repetition": {
                        "split_jaccard": split_overlap,
                        "template_support": int(template_support),
                        "pool_support": pool_support,
                    },
                    "S_specificity": {
                        "reference_ratio": contrast_ratio,
                        "role_conditioned": True,
                    },
                    "C_candidate_competition": {
                        "probability_correlation": probability_corr,
                        "margin_correlation": margin_corr,
                    },
                    "N_natural_rollout": {
                        "qualified_rate": float(np.mean(
                            rollout[:, operation_index]
                        )),
                    },
                    "M_cross_model": {"support": 0},
                    "H_local_causality": {
                        "support": 0,
                        "status": "not_tested_in_global_scan",
                    },
                },
                "edge_claim_allowed": "co_response_only",
                "selection_is_mechanism_proof": False,
            }
            motif.update(surface_confound_audit(motif))
            motifs.append(motif)
    return motifs


def cross_model_groups(
    motifs_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for motifs in motifs_by_model.values():
        for motif in motifs:
            if not motif["repeated_candidate"]:
                continue
            grouped[(
                motif["operation"],
                motif["stage"],
                motif["component"],
                motif["role"],
            )].append(motif)
    result = []
    for key, rows in sorted(grouped.items()):
        models = sorted({row["model"] for row in rows})
        if len(models) < 2:
            continue
        relative_depths = [
            float(row["peak_relative_depth"]) for row in rows
        ]
        spread = max(relative_depths) - min(relative_depths)
        if spread > 0.15:
            continue
        group_id = "cross." + ".".join(key)
        result.append({
            "schema_version": "phase1008_cross_model_motif.v1",
            "phase": PHASE,
            "cross_motif_id": group_id,
            "operation": key[0],
            "stage": key[1],
            "component": key[2],
            "role": key[3],
            "models": models,
            "model_count": len(models),
            "peak_relative_depths": {
                row["model"]: row["peak_relative_depth"] for row in rows
            },
            "relative_depth_spread": spread,
            "alignment_basis": (
                "operation + stage + component + semantic role + "
                "relative-depth tolerance"
            ),
            "raw_vector_or_coordinate_comparison_used": False,
            "claim": "coordinate-free repeated functional response motif",
            "refinement_eligible_models": sorted(
                row["model"] for row in rows
                if row["refinement_eligible"]
            ),
        })
        for row in rows:
            row["evidence_axes"]["M_cross_model"] = {
                "support": len(models),
                "models": models,
                "cross_motif_id": group_id,
            }
    return result


def co_response_edges(
    model_name: str,
    motifs: list[dict[str, Any]],
    units: list[dict[str, Any]],
    normalized: np.ndarray,
    qualified: np.ndarray,
) -> list[dict[str, Any]]:
    selected = [row for row in motifs if row["refinement_eligible"]]
    split_masks = {
        split: np.array(
            [unit["split"] == split for unit in units], dtype=np.bool_
        )
        for split in SPLITS
    }
    candidates = []
    for left, right in itertools.combinations(selected, 2):
        if left["operation"] != right["operation"]:
            continue
        if (
            left["stage"],
            left["component"],
            left["role"],
        ) == (
            right["stage"],
            right["component"],
            right["role"],
        ):
            continue
        left_order = (
            STAGE_ORDER[left["stage"]],
            float(left["peak_relative_depth"]),
        )
        right_order = (
            STAGE_ORDER[right["stage"]],
            float(right["peak_relative_depth"]),
        )
        source, target = (
            (left, right) if left_order <= right_order else (right, left)
        )
        operation_index = OP_INDEX[source["operation"]]
        qmask = qualified[:, operation_index]
        source_values = normalized[
            :, operation_index, int(source["peak_event_index"])
        ]
        target_values = normalized[
            :, operation_index, int(target["peak_event_index"])
        ]
        correlations = {}
        counts = {}
        valid = True
        for split in SPLITS:
            mask = split_masks[split] & qmask
            counts[split] = int(np.sum(mask))
            correlation = safe_corr(
                source_values[mask], target_values[mask]
            )
            correlations[split] = correlation
            if correlation is None or correlation < 0.50:
                valid = False
        if not valid:
            continue
        candidates.append({
            "schema_version": "phase1008_co_response_edge.v1",
            "phase": PHASE,
            "model": model_name,
            "edge_id": (
                f"{model_name}.{source['operation']}."
                f"{source['motif_id']}->{target['motif_id']}"
            ),
            "operation": source["operation"],
            "source_motif_id": source["motif_id"],
            "target_motif_id": target["motif_id"],
            "source_stage": source["stage"],
            "target_stage": target["stage"],
            "source_peak_relative_depth": source["peak_relative_depth"],
            "target_peak_relative_depth": target["peak_relative_depth"],
            "correlations": correlations,
            "qualified_counts": counts,
            "edge_type": "co_response",
            "transport_claim": False,
            "mediation_claim": False,
            "causal_claim": False,
            "ordering_meaning": (
                "display order only; stage/depth precedence is not causality"
            ),
            "_rank": min(correlations.values()),
        })
    candidates.sort(key=lambda row: row["_rank"], reverse=True)
    result = candidates[:300]
    for row in result:
        row.pop("_rank", None)
    return result


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    final_root = OUT_ROOT / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    motifs_by_model: dict[str, list[dict[str, Any]]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    model_summaries = {}
    total_measurements = 0
    for model_name in MODELS:
        scan_root = OUT_ROOT / "scan" / model_name
        scan_summary = read_json(scan_root / "summary.json")
        if scan_summary["protocol_digest"] != protocol["preregistration_digest"]:
            raise RuntimeError(f"{model_name}: scan/protocol digest mismatch")
        events = read_jsonl(scan_root / "events.jsonl")
        units = read_jsonl(scan_root / "units.jsonl")
        scalars = np.load(scan_root / "response_scalars.npz")
        direction = np.load(scan_root / "direction_consistency.npz")
        raw = scalars["raw_magnitude"]
        normalized = scalars["normalized_magnitude"]
        qualified = scalars["semantic_qualified"]
        rollout = scalars["rollout_qualified"]
        consistency = direction["direction_consistency"]
        outputs = output_arrays(
            units, read_jsonl(scan_root / "output_pairs.jsonl")
        )
        observations = event_observations(
            model_name,
            events,
            units,
            normalized,
            raw,
            qualified,
            consistency,
        )
        motifs = trajectory_motifs(
            model_name,
            events,
            units,
            normalized,
            qualified,
            rollout,
            consistency,
            outputs,
        )
        motifs_by_model[model_name] = motifs
        runtime[model_name] = {
            "units": units,
            "normalized": normalized,
            "qualified": qualified,
        }
        write_jsonl(final_root / model_name / "event_observations.jsonl", observations)
        write_jsonl(final_root / model_name / "trajectory_motifs.jsonl", motifs)
        total_measurements += int(scan_summary["scalar_measurement_count"])
        model_summaries[model_name] = {
            "event_observation_count": len(observations),
            "trajectory_count": len(motifs),
            "repeated_candidate_count": sum(
                row["repeated_candidate"] for row in motifs
            ),
            "surface_confounded_repeated_count": sum(
                row["repeated_candidate"]
                and row["direct_surface_confounded"]
                for row in motifs
            ),
            "refinement_eligible_count": sum(
                row["refinement_eligible"] for row in motifs
            ),
            "repeated_by_operation": {
                operation: sum(
                    row["repeated_candidate"]
                    and row["operation"] == operation
                    for row in motifs
                )
                for operation in ("B", "Q", "BQ", "E", "O", "N", "X")
            },
        }

    cross = cross_model_groups(motifs_by_model)
    write_jsonl(final_root / "cross_model_motifs.jsonl", cross)
    edge_count = 0
    for model_name in MODELS:
        data = runtime[model_name]
        edges = co_response_edges(
            model_name,
            motifs_by_model[model_name],
            data["units"],
            data["normalized"],
            data["qualified"],
        )
        edge_count += len(edges)
        write_jsonl(final_root / model_name / "co_response_edges.jsonl", edges)
        # Re-write motifs after coordinate-free cross-model evidence is attached.
        write_jsonl(
            final_root / model_name / "trajectory_motifs.jsonl",
            motifs_by_model[model_name],
        )
        model_summaries[model_name]["co_response_edge_count"] = len(edges)
        model_summaries[model_name]["cross_model_supported_motifs"] = sum(
            row["evidence_axes"]["M_cross_model"]["support"] >= 2
            for row in motifs_by_model[model_name]
        )

    cross_core = [
        row for row in cross
        if set(row["models"]) >= {"qwen3", "glm4"}
        and set(row["refinement_eligible_models"]) >= {"qwen3", "glm4"}
        and row["operation"] in ("B", "Q", "BQ", "X")
    ]
    automatic_next = (
        "targeted_head_and_neuron_observation_warranted"
        if len(cross_core) >= 3
        else "expand_templates_and_behavior_before_fine_decomposition"
    )
    summary = {
        "schema_version": "phase1008_final_summary.v1",
        "phase": PHASE,
        "protocol_digest": protocol["preregistration_digest"],
        "models": list(MODELS),
        "total_scalar_internal_measurements": total_measurements,
        "raw_hidden_tensors_persisted": 0,
        "model_summaries": model_summaries,
        "cross_model_motif_count": len(cross),
        "qwen_glm_core_cross_motif_count": len(cross_core),
        "qwen_glm_core_definition": (
            "B/Q/BQ/X motif repeats in Qwen3 and GLM4 after excluding "
            "directly edited roles, residual depth 0, and answer-token "
            "teacher-forcing surface differences"
        ),
        "co_response_edge_count": edge_count,
        "edge_semantics": (
            "all generated edges are repeated co-response correlations; "
            "none are transport, mediation, or causal edges"
        ),
        "evidence_is_multi_axis": True,
        "single_pass_fail_gate_used": False,
        "pca_umap_tsne_used_for_selection": False,
        "cross_model_raw_coordinate_comparison_used": False,
        "automatic_next_action": automatic_next,
        "automatic_next_reason": (
            f"{len(cross_core)} coordinate-free B/Q/BQ/X motifs repeat in "
            "both Qwen3 and GLM4"
        ),
    }
    write_json(final_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
