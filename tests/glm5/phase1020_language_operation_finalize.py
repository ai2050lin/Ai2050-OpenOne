#!/usr/bin/env python3
"""Finalize Phase1020 behavior-qualified language-pattern measurements."""

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

import phase1018_language_pattern_finalize as base
import phase1020_language_operation_protocol as protocol


ANALYSIS_ROOT = protocol.OUT_ROOT / "analysis"
DISCOVERY_CONFIRMATION = 0.40
CROSS_CONCEPT = 0.30
PREVALENCE = 0.50
ACCURACY_GATE = 0.70
EPSILON = 1e-12


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "candidate_accuracy": (
            float(np.mean([row["candidate_hit"] for row in rows]))
            if rows else None
        ),
        "first_token_accuracy": (
            float(np.mean([row["first_token_hit"] for row in rows]))
            if rows else None
        ),
        "median_candidate_margin": (
            float(np.median([row["candidate_margin"] for row in rows]))
            if rows else None
        ),
    }


def behavior_analysis() -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    rows_out = []
    family_table = {}
    subgroup_table = {}
    for model in protocol.MODELS:
        rows = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "formal.jsonl"
        )
        for family in protocol.FAMILIES:
            value = metrics([
                row for row in rows if row["family"] == family
            ])
            family_table[(model, family)] = value
            rows_out.append({
                "schema_version": "phase1020_behavior_group.v1",
                "phase": protocol.PHASE,
                "model": model,
                "group_type": "family",
                "family": family,
                "group": family,
                **value,
            })
        for family in protocol.FAMILIES:
            subgroups = sorted({
                row["subgroup"]
                for row in rows
                if row["family"] == family
            })
            for subgroup in subgroups:
                value = metrics([
                    row
                    for row in rows
                    if row["family"] == family
                    and row["subgroup"] == subgroup
                ])
                subgroup_table[(model, subgroup)] = value
                rows_out.append({
                    "schema_version": "phase1020_behavior_group.v1",
                    "phase": protocol.PHASE,
                    "model": model,
                    "group_type": "subgroup",
                    "family": family,
                    "group": subgroup,
                    **value,
                })
        translation_items = defaultdict(list)
        rare_branches = defaultdict(list)
        for row in rows:
            if row["family"] == "translation_mode":
                spec = protocol.TRANSLATION_ITEMS[row["item_id"]]
                translation_items[
                    (
                        spec["profile"],
                        spec["direction"],
                        row["split"],
                    )
                ].append(row)
            elif row["family"] == "rare_knowledge":
                rare_branches[
                    (row["item_id"], row["state"][1], row["split"])
                ].append(row)
        for (profile, direction, split), subset in sorted(
            translation_items.items()
        ):
            value = metrics(subset)
            rows_out.append({
                "schema_version": "phase1020_behavior_group.v1",
                "phase": protocol.PHASE,
                "model": model,
                "group_type": "translation_profile_direction_split",
                "family": "translation_mode",
                "group": f"{profile}:{direction}:{split}",
                "profile": profile,
                "direction": direction,
                "split": split,
                **value,
            })
        for (item_id, branch, split), subset in sorted(
            rare_branches.items()
        ):
            value = metrics(subset)
            rows_out.append({
                "schema_version": "phase1020_behavior_group.v1",
                "phase": protocol.PHASE,
                "model": model,
                "group_type": "rare_pair_branch_split",
                "family": "rare_knowledge",
                "group": f"{item_id}:b{branch}:{split}",
                "item_id": item_id,
                "branch": int(branch),
                "split": split,
                **value,
            })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "behavior_groups.jsonl", rows_out
    )
    return rows_out, family_table, subgroup_table


def panel_catalog() -> dict[
    tuple[str, str, str, str], dict[str, Any]
]:
    catalog = {}
    for model in protocol.MODELS:
        model_root = protocol.OUT_ROOT / "formal_scan" / model
        summary_path = model_root / "summary.json"
        if not summary_path.exists():
            continue
        model_summary = protocol.read_json(summary_path)
        for family in model_summary.get("eligible_families", []):
            family_root = model_root / family
            if not family_root.exists():
                continue
            for item_root in sorted(
                path for path in family_root.iterdir() if path.is_dir()
            ):
                for split in protocol.SPLITS:
                    root = item_root / split
                    if not (root / "summary.json").exists():
                        continue
                    catalog[(model, family, item_root.name, split)] = {
                        "root": root,
                        "summary": protocol.read_json(
                            root / "summary.json"
                        ),
                        **base.load_panel_metrics(root),
                        "directions": None,
                    }
    return catalog


def panel_directions(panel: dict[str, Any]) -> dict[str, np.ndarray]:
    if panel["directions"] is None:
        panel["directions"] = base.load_directions(panel["root"])
    return panel["directions"]


def item_internal_analysis(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    rows = []
    confirmed_by_item = {}
    keys = sorted({
        key[:3] for key in catalog
        if (key[0], key[1], key[2], "discovery") in catalog
        and (key[0], key[1], key[2], "confirmation") in catalog
    })
    for model, family, item_id in keys:
        discovery = catalog[(model, family, item_id, "discovery")]
        confirmation = catalog[(model, family, item_id, "confirmation")]
        cosine = base.direction_cosine(
            panel_directions(discovery),
            panel_directions(confirmation),
        )
        confirmed = (
            discovery["candidate"]
            & confirmation["candidate"]
            & (cosine >= DISCOVERY_CONFIRMATION)
        )
        confirmed_by_item[(model, family, item_id)] = confirmed
        rows.append({
            "schema_version": "phase1020_item_internal_summary.v1",
            "phase": protocol.PHASE,
            "model": model,
            "family": family,
            "item_id": item_id,
            "subgroup": discovery["summary"]["subgroup"],
            "confirmed_event_role_count": int(confirmed.sum()),
            "discovery_candidate_count": int(
                discovery["candidate"].sum()
            ),
            "confirmation_candidate_count": int(
                confirmation["candidate"].sum()
            ),
            "median_discovery_confirmation_cosine": (
                base.finite_median(cosine[confirmed])
            ),
            "median_discovery_magnitude": (
                base.finite_median(discovery["magnitude"][confirmed])
            ),
            "median_confirmation_magnitude": (
                base.finite_median(confirmation["magnitude"][confirmed])
            ),
            "claim_limit": (
                "A confirmed event is a repeated branch response, not a "
                "causal path or a complete language mechanism."
            ),
        })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "item_internal_summary.jsonl", rows
    )
    return rows, confirmed_by_item


def normalized(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True)
    result = np.zeros_like(array, dtype=np.float32)
    np.divide(array, norm, out=result, where=norm > EPSILON)
    return result


def consistency(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    squared = np.einsum(
        "...d,...d->...",
        total.astype(np.float64, copy=False),
        total.astype(np.float64, copy=False),
    )
    result = np.full(count.shape, np.nan, dtype=np.float32)
    valid = count >= 2
    result[valid] = (
        (squared[valid] - count[valid])
        / (count[valid] * (count[valid] - 1.0))
    ).astype(np.float32)
    return result


def aggregate_panels(
    panels: list[dict[str, Any]],
    confirmed_masks: list[np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not panels:
        raise RuntimeError("cannot aggregate an empty panel list")
    first = panel_directions(panels[0])
    whole_sum = np.zeros_like(first["whole"], dtype=np.float32)
    head_sum = np.zeros_like(first["head"], dtype=np.float32)
    whole_count = np.zeros(first["whole"].shape[:-1], dtype=np.int32)
    head_count = np.zeros(first["head"].shape[:-1], dtype=np.int32)
    whole_events = first["whole"].shape[1]
    for panel, mask in zip(panels, confirmed_masks):
        directions = panel_directions(panel)
        whole_mask = mask[:, :whole_events]
        head_mask = mask[:, whole_events:]
        whole_sum += directions["whole"] * whole_mask[..., None]
        head_sum += directions["head"] * head_mask[..., None]
        whole_count += whole_mask.astype(np.int32)
        head_count += head_mask.astype(np.int32)
    combined_count = np.concatenate((whole_count, head_count), axis=1)
    combined_consistency = np.concatenate(
        (
            consistency(whole_sum, whole_count),
            consistency(head_sum, head_count),
        ),
        axis=1,
    )
    return {
        "whole": normalized(whole_sum),
        "head": normalized(head_sum),
    }, combined_count, combined_consistency


def combine_split_directions(
    discovery: dict[str, np.ndarray],
    confirmation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        key: normalized(discovery[key] + confirmation[key])
        for key in ("whole", "head")
    }


def masked_median(
    values: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    return base.finite_median(values[mask])


def translation_internal_analysis(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]],
    confirmed_by_item: dict[tuple[str, str, str], np.ndarray],
    subgroup_behavior: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows = []
    comparison_rows = []
    attribution_rows = []
    for model in protocol.MODELS:
        events_path = (
            protocol.OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        if not events_path.exists():
            continue
        aggregates = {}
        candidates = {}
        for profile in protocol.TRANSLATION_PROFILES:
            for direction in protocol.TRANSLATION_DIRECTIONS:
                item_ids = [
                    item_id
                    for item_id, spec in protocol.TRANSLATION_ITEMS.items()
                    if spec["profile"] == profile
                    and spec["direction"] == direction
                    and spec["scan_eligible"]
                    and (
                        model,
                        "translation_mode",
                        item_id,
                        "discovery",
                    ) in catalog
                ]
                if not item_ids:
                    continue
                split_aggregates = {}
                split_counts = {}
                split_consistency = {}
                for split in protocol.SPLITS:
                    panels = [
                        catalog[
                            (model, "translation_mode", item_id, split)
                        ]
                        for item_id in item_ids
                    ]
                    masks = [
                        confirmed_by_item[
                            (model, "translation_mode", item_id)
                        ]
                        for item_id in item_ids
                    ]
                    (
                        split_aggregates[split],
                        split_counts[split],
                        split_consistency[split],
                    ) = aggregate_panels(panels, masks)
                split_cosine = base.direction_cosine(
                    split_aggregates["discovery"],
                    split_aggregates["confirmation"],
                )
                prevalence = np.minimum(
                    split_counts["discovery"],
                    split_counts["confirmation"],
                ) / len(item_ids)
                stable = (
                    (prevalence >= PREVALENCE)
                    & (
                        np.minimum(
                            split_consistency["discovery"],
                            split_consistency["confirmation"],
                        ) >= CROSS_CONCEPT
                    )
                    & (split_cosine >= DISCOVERY_CONFIRMATION)
                )
                key = (profile, direction)
                aggregates[key] = combine_split_directions(
                    split_aggregates["discovery"],
                    split_aggregates["confirmation"],
                )
                candidates[key] = stable
                group_rows.append({
                    "schema_version": (
                        "phase1020_translation_internal_group.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model,
                    "profile": profile,
                    "direction": direction,
                    "concept_count": len(item_ids),
                    "stable_event_role_count": int(stable.sum()),
                    "median_prevalence": masked_median(
                        prevalence, stable
                    ),
                    "median_cross_concept_consistency": masked_median(
                        np.minimum(
                            split_consistency["discovery"],
                            split_consistency["confirmation"],
                        ),
                        stable,
                    ),
                    "median_discovery_confirmation_cosine": masked_median(
                        split_cosine, stable
                    ),
                    "stable_event_role_count_by_role": {
                        role: int(stable[role_index].sum())
                        for role_index, role in enumerate(
                            protocol.CAPTURE_ROLES
                        )
                    },
                    "cross_concept_consistency_by_role": {
                        role: masked_median(
                            np.minimum(
                                split_consistency["discovery"][role_index],
                                split_consistency["confirmation"][role_index],
                            ),
                            stable[role_index],
                        )
                        for role_index, role in enumerate(
                            protocol.CAPTURE_ROLES
                        )
                    },
                    "discovery_confirmation_cosine_by_role": {
                        role: masked_median(
                            split_cosine[role_index],
                            stable[role_index],
                        )
                        for role_index, role in enumerate(
                            protocol.CAPTURE_ROLES
                        )
                    },
                    "behavior_candidate_accuracy": subgroup_behavior[
                        (model, profile)
                    ]["candidate_accuracy"],
                })

        comparison_lookup = {}
        for profile in protocol.TRANSLATION_PROFILES:
            left_key = (profile, "en_zh")
            right_key = (profile, "zh_en")
            if left_key not in aggregates or right_key not in aggregates:
                continue
            mask = candidates[left_key] & candidates[right_key]
            cosine = base.direction_cosine(
                aggregates[left_key], aggregates[right_key]
            )
            value = masked_median(cosine, mask)
            comparison_lookup[f"cross_direction:{profile}"] = value
            comparison_rows.append({
                "schema_version": (
                    "phase1020_translation_direction_comparison.v1"
                ),
                "phase": protocol.PHASE,
                "model": model,
                "comparison": "cross_direction",
                "left": f"{profile}:en_zh",
                "right": f"{profile}:zh_en",
                "shared_candidate_count": int(mask.sum()),
                "median_cosine": value,
                "median_cosine_by_role": {
                    role: masked_median(
                        cosine[role_index],
                        mask[role_index],
                    )
                    for role_index, role in enumerate(
                        protocol.CAPTURE_ROLES
                    )
                },
                "shared_candidate_count_by_role": {
                    role: int(mask[role_index].sum())
                    for role_index, role in enumerate(
                        protocol.CAPTURE_ROLES
                    )
                },
            })
        for direction in protocol.TRANSLATION_DIRECTIONS:
            full_key = ("full", direction)
            if full_key not in aggregates:
                continue
            for profile in protocol.TRANSLATION_PROFILES:
                other_key = (profile, direction)
                if profile == "full" or other_key not in aggregates:
                    continue
                mask = candidates[full_key] & candidates[other_key]
                cosine = base.direction_cosine(
                    aggregates[full_key], aggregates[other_key]
                )
                value = masked_median(cosine, mask)
                comparison_lookup[
                    f"full_profile:{direction}:{profile}"
                ] = value
                comparison_rows.append({
                    "schema_version": (
                        "phase1020_translation_direction_comparison.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model,
                    "comparison": "full_profile_alignment",
                    "left": f"full:{direction}",
                    "right": f"{profile}:{direction}",
                    "shared_candidate_count": int(mask.sum()),
                    "median_cosine": value,
                    "median_cosine_by_role": {
                        role: masked_median(
                            cosine[role_index],
                            mask[role_index],
                        )
                        for role_index, role in enumerate(
                            protocol.CAPTURE_ROLES
                        )
                    },
                    "shared_candidate_count_by_role": {
                        role: int(mask[role_index].sum())
                        for role_index, role in enumerate(
                            protocol.CAPTURE_ROLES
                        )
                    },
                })

        def median_lookup(prefix: str, profiles: tuple[str, ...]) -> float | None:
            values = [
                comparison_lookup.get(f"{prefix}:{direction}:{profile}")
                for direction in protocol.TRANSLATION_DIRECTIONS
                for profile in profiles
            ]
            return base.finite_median([
                value for value in values if value is not None
            ])

        full_operation = median_lookup(
            "full_profile", ("operation_only",)
        )
        full_language = median_lookup(
            "full_profile", ("language_only",)
        )
        full_relation = median_lookup(
            "full_profile", ("relation_only",)
        )
        full_negative = median_lookup(
            "full_profile", ("irrelevant_control",)
        )
        cross_full = comparison_lookup.get("cross_direction:full")
        cross_operation = comparison_lookup.get(
            "cross_direction:operation_only"
        )
        cross_relation = comparison_lookup.get(
            "cross_direction:relation_only"
        )
        required = (
            full_operation,
            full_relation,
            full_negative,
            cross_full,
            cross_operation,
            cross_relation,
        )
        profile_gain = (
            min(full_operation, full_relation) - full_negative
            if all(value is not None for value in required[:3])
            else None
        )
        transform_candidate = bool(
            all(value is not None for value in required)
            and subgroup_behavior[(model, "full")][
                "candidate_accuracy"
            ] >= ACCURACY_GATE
            and subgroup_behavior[(model, "operation_only")][
                "candidate_accuracy"
            ] >= ACCURACY_GATE
            and subgroup_behavior[(model, "relation_only")][
                "candidate_accuracy"
            ] >= ACCURACY_GATE
            and min(cross_full, cross_operation, cross_relation) >= 0.15
            and profile_gain >= 0.15
        )
        finite_task = [
            value
            for value in (
                full_operation,
                full_language,
                full_relation,
            )
            if value is not None
        ]
        if (
            full_negative is not None
            and finite_task
            and full_negative >= max(finite_task) - 0.05
        ):
            interpretation = "generic_instruction_response_candidate"
        elif transform_candidate:
            interpretation = "cross_direction_transform_response_candidate"
        elif (
            full_language is not None
            and full_operation is not None
            and full_relation is not None
            and full_language >= max(full_operation, full_relation) + 0.10
        ):
            interpretation = "language_or_script_cue_dominant"
        else:
            interpretation = "mixed_conditioned_response"
        attribution_rows.append({
            "schema_version": "phase1020_translation_attribution.v1",
            "phase": protocol.PHASE,
            "model": model,
            "behavior": {
                profile: subgroup_behavior[(model, profile)]
                for profile in protocol.TRANSLATION_PROFILES
            },
            "full_operation_alignment": full_operation,
            "full_language_alignment": full_language,
            "full_relation_alignment": full_relation,
            "full_negative_alignment": full_negative,
            "cross_direction_full": cross_full,
            "cross_direction_operation_only": cross_operation,
            "cross_direction_relation_only": cross_relation,
            "operation_relation_over_negative_gain": profile_gain,
            "transform_response_candidate": transform_candidate,
            "interpretation": interpretation,
            "claim_limit": (
                "Even a passing transform-response gate describes a stable "
                "candidate field; it is not yet a translation mechanism."
            ),
        })
        del aggregates, candidates
    protocol.write_jsonl(
        ANALYSIS_ROOT / "translation_internal_groups.jsonl", group_rows
    )
    protocol.write_jsonl(
        ANALYSIS_ROOT / "translation_direction_comparisons.jsonl",
        comparison_rows,
    )
    protocol.write_jsonl(
        ANALYSIS_ROOT / "translation_attribution.jsonl",
        attribution_rows,
    )
    return group_rows, comparison_rows, attribution_rows


def family_recurrence(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]],
    confirmed_by_item: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for model in protocol.MODELS:
        events_path = (
            protocol.OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        if not events_path.exists():
            continue
        events = protocol.read_jsonl(events_path)
        for family in protocol.FAMILIES:
            item_ids = sorted({
                key[2]
                for key in confirmed_by_item
                if key[0] == model and key[1] == family
            })
            if not item_ids:
                continue
            counts = np.stack([
                confirmed_by_item[(model, family, item_id)]
                for item_id in item_ids
            ]).sum(axis=0)
            minimum = max(2, math.ceil(len(item_ids) / 2))
            majority = counts >= minimum
            for role_index, event_index in np.argwhere(majority):
                event = events[int(event_index)]
                rows.append({
                    "schema_version": (
                        "phase1020_repeated_physical_event.v1"
                    ),
                    "phase": protocol.PHASE,
                    "model": model,
                    "family": family,
                    "role": protocol.CAPTURE_ROLES[int(role_index)],
                    "event_id": event["event_id"],
                    "component": event["component"],
                    "depth": int(event["depth"]),
                    "relative_depth": float(event["relative_depth"]),
                    "head": event["head"],
                    "item_count": len(item_ids),
                    "repeat_count": int(
                        counts[role_index, event_index]
                    ),
                    "prevalence": float(
                        counts[role_index, event_index] / len(item_ids)
                    ),
                })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "repeated_physical_events.jsonl", rows
    )
    return rows


def recurrence_distribution(
    repeated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in repeated_rows:
        relative_depth = float(row["relative_depth"])
        if relative_depth <= 0:
            band = "embedding"
        elif relative_depth <= 1.0 / 3.0:
            band = "early"
        elif relative_depth <= 2.0 / 3.0:
            band = "middle"
        else:
            band = "late"
        grouped[
            (
                row["model"],
                row["family"],
                row["role"],
                row["component"],
                band,
            )
        ].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        model, family, role, component, band = key
        result.append({
            "schema_version": (
                "phase1020_repeated_event_distribution.v1"
            ),
            "phase": protocol.PHASE,
            "model": model,
            "family": family,
            "role": role,
            "component": component,
            "depth_band": band,
            "event_count": len(rows),
            "median_prevalence": base.finite_median([
                row["prevalence"] for row in rows
            ]),
            "maximum_prevalence": max(
                row["prevalence"] for row in rows
            ),
        })
    protocol.write_jsonl(
        ANALYSIS_ROOT / "repeated_event_distribution.jsonl", result
    )
    return result


def taotie_rows(
    catalog: dict[tuple[str, str, str, str], dict[str, Any]],
    confirmed_by_item: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    item_id = "taotie_xiezhi"
    for model in protocol.MODELS:
        key = (model, "rare_knowledge", item_id)
        if key not in confirmed_by_item:
            continue
        events = protocol.read_jsonl(
            protocol.OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        discovery = catalog[(*key, "discovery")]
        confirmation = catalog[(*key, "confirmation")]
        cosine = base.direction_cosine(
            panel_directions(discovery),
            panel_directions(confirmation),
        )
        confirmed = confirmed_by_item[key]
        for role_index, event_index in np.argwhere(confirmed):
            event = events[int(event_index)]
            rows.append({
                "schema_version": "phase1020_taotie_relative_event.v1",
                "phase": protocol.PHASE,
                "model": model,
                "pair": "饕餮_vs_獬豸",
                "role": protocol.CAPTURE_ROLES[int(role_index)],
                "event_id": event["event_id"],
                "component": event["component"],
                "depth": int(event["depth"]),
                "relative_depth": float(event["relative_depth"]),
                "head": event["head"],
                "discovery_confirmation_cosine": float(
                    cosine[role_index, event_index]
                ),
                "discovery_consistency": float(
                    discovery["consistency"][role_index, event_index]
                ),
                "confirmation_consistency": float(
                    confirmation["consistency"][role_index, event_index]
                ),
                "discovery_magnitude": float(
                    discovery["magnitude"][role_index, event_index]
                ),
                "confirmation_magnitude": float(
                    confirmation["magnitude"][role_index, event_index]
                ),
            })
    rows.sort(
        key=lambda row: (
            row["discovery_confirmation_cosine"],
            min(
                row["discovery_consistency"],
                row["confirmation_consistency"],
            ),
        ),
        reverse=True,
    )
    rows = rows[:150]
    protocol.write_jsonl(
        ANALYSIS_ROOT / "taotie_relative_events.jsonl", rows
    )
    return rows


def finalize() -> dict[str, Any]:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    behavior_rows, family_behavior, subgroup_behavior = behavior_analysis()
    catalog = panel_catalog()
    item_rows, confirmed_by_item = item_internal_analysis(catalog)
    (
        translation_groups,
        translation_comparisons,
        attribution_rows,
    ) = translation_internal_analysis(
        catalog, confirmed_by_item, subgroup_behavior
    )
    repeated_rows = family_recurrence(catalog, confirmed_by_item)
    distribution_rows = recurrence_distribution(repeated_rows)
    taotie = taotie_rows(catalog, confirmed_by_item)
    qualifying_models = [
        row["model"]
        for row in attribution_rows
        if row["transform_response_candidate"]
    ]
    automatic = {
        "schema_version": "phase1020_automatic_continuation.v1",
        "phase": protocol.PHASE,
        "qualifying_models": qualifying_models,
        "start_targeted_validation": (
            len(qualifying_models) >= 2
        ),
        "decision": (
            "start a held-out targeted validation of the repeated "
            "cross-direction transform candidate"
            if len(qualifying_models) >= 2
            else "do not start causal work; retain the descriptive atlas "
            "and redesign the unresolved cue factor"
        ),
        "minimum_model_count": 2,
    }
    protocol.write_json(
        ANALYSIS_ROOT / "automatic_continuation.json", automatic
    )
    scan_gate = protocol.read_json(ANALYSIS_ROOT / "scan_gate.json")
    model_summaries = {
        model: protocol.read_json(
            protocol.OUT_ROOT / "formal_scan" / model / "summary.json"
        )
        for model in protocol.MODELS
    }
    claim_ledger = {
        "schema_version": "phase1020_claim_ledger.v1",
        "phase": protocol.PHASE,
        "supported": [
            "Behavior qualification was completed before component scanning.",
            "Discovery and confirmation renderings are exactly text-disjoint.",
            "Translation was measured in both English-Chinese directions.",
            "Full, operation-only, language-only, relation-only, and irrelevant-control prompts were measured separately.",
            "Rare terms were tested by meaning-reversing term substitution without contextual answer clues.",
        ],
        "not_supported_without_later_work": [
            "Any repeated response is a complete language mechanism.",
            "A rare-term pair direction is the full meaning of either word.",
            "A translation profile isolates a perfectly independent semantic factor.",
            "A stable cross-direction field is causally necessary or sufficient.",
            "One equation describes rare words, punctuation, translation, and contrast.",
        ],
        "formula_status": (
            "All Phase1020 equations are post-observation measurement "
            "definitions; no language law was fitted."
        ),
    }
    protocol.write_json(
        ANALYSIS_ROOT / "claim_ledger.json", claim_ledger
    )
    summary = {
        "schema_version": "phase1020_analysis_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "counts": {
            "behavior_group_rows": len(behavior_rows),
            "item_internal_rows": len(item_rows),
            "translation_group_rows": len(translation_groups),
            "translation_comparison_rows": len(
                translation_comparisons
            ),
            "translation_attribution_rows": len(attribution_rows),
            "repeated_physical_event_rows": len(repeated_rows),
            "repeated_event_distribution_rows": len(distribution_rows),
            "taotie_relative_event_rows": len(taotie),
        },
        "scan_gate": scan_gate,
        "family_behavior": {
            model: {
                family: family_behavior[(model, family)]
                for family in protocol.FAMILIES
            }
            for model in protocol.MODELS
        },
        "translation_attribution": attribution_rows,
        "automatic_continuation": automatic,
        "model_scan_summaries": model_summaries,
        "claim_ledger": claim_ledger,
    }
    protocol.write_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    finalize()
