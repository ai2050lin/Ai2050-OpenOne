#!/usr/bin/env python3
"""Finalize the Phase1018 descriptive atlas without causal overclaiming."""

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

from phase1018_language_pattern_protocol import (
    CAPTURE_ROLES,
    FAMILIES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


ANALYSIS_ROOT = OUT_ROOT / "analysis"
SPLITS = ("discovery", "confirmation")
PRIMARY_DIRECTION = 0.45
PRIMARY_SURFACE = 0.40
PRIMARY_MAGNITUDE = 1e-4
GAP_GATE = 0.15
ACCURACY_GATE = 0.70
EPSILON = 1e-12


def finite_median(values: np.ndarray | list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return float(np.median(array))


def finite_mean(values: np.ndarray | list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return float(np.mean(array))


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    numerator = np.einsum("...d,...d->...", a64, b64)
    denominator = np.sqrt(
        np.einsum("...d,...d->...", a64, a64)
        * np.einsum("...d,...d->...", b64, b64)
    )
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = denominator > EPSILON
    result[valid] = (numerator[valid] / denominator[valid]).astype(
        np.float32
    )
    return result


def load_directions(panel_root: Path) -> dict[str, np.ndarray]:
    with np.load(panel_root / "directions.npz") as data:
        return {
            "whole": data["whole_direction"].astype(np.float32),
            "head": data["head_direction"].astype(np.float32),
        }


def direction_cosine(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        (
            row_cosine(a["whole"], b["whole"]),
            row_cosine(a["head"], b["head"]),
        ),
        axis=1,
    )


def load_panel_metrics(
    panel_root: Path,
    direction_threshold: float = PRIMARY_DIRECTION,
    surface_threshold: float = PRIMARY_SURFACE,
) -> dict[str, Any]:
    with np.load(panel_root / "response_scalars.npz") as response:
        contrast_names = response["contrast_names"].tolist()
        d_index = contrast_names.index("D")
        magnitude = np.nanmedian(
            response["normalized_magnitude"][:, d_index],
            axis=0,
        ).astype(np.float32)
    with np.load(panel_root / "direction_metrics.npz") as direction:
        consistency = np.concatenate(
            (
                direction["whole_consistency"],
                direction["head_consistency"],
            ),
            axis=1,
        )
        surface = np.concatenate(
            (
                direction["whole_surface_alignment"],
                direction["head_surface_alignment"],
            ),
            axis=1,
        )
    candidate = (
        (magnitude >= PRIMARY_MAGNITUDE)
        & (consistency >= direction_threshold)
        & (surface >= surface_threshold)
    )
    return {
        "magnitude": magnitude,
        "consistency": consistency,
        "surface": surface,
        "candidate": candidate,
    }


def behavior_tables() -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    family = {}
    item_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for model in MODELS:
        summary = read_json(
            OUT_ROOT / "behavior" / model / "formal.summary.json"
        )
        for family_name, values in summary["by_family"].items():
            family[(model, family_name)] = values
        for row in read_jsonl(
            OUT_ROOT / "behavior" / model / "formal.jsonl"
        ):
            item_rows[(model, row["family"], row["item_id"])].append(row)
    item = {}
    for key, rows in item_rows.items():
        by_branch = {
            branch: [
                row["candidate_hit"]
                for row in rows
                if row["state"].startswith(branch)
            ]
            for branch in ("b0", "b1")
        }
        by_surface = {
            surface: [
                row["candidate_hit"]
                for row in rows
                if row["state"].endswith(surface)
            ]
            for surface in ("l0", "l1")
        }
        item[key] = {
            "case_count": len(rows),
            "candidate_accuracy": float(np.mean([
                row["candidate_hit"] for row in rows
            ])),
            "first_token_accuracy": float(np.mean([
                row["first_token_hit"] for row in rows
            ])),
            "median_candidate_margin": float(np.median([
                row["candidate_margin"] for row in rows
            ])),
            "branch_accuracy": {
                branch: float(np.mean(values))
                for branch, values in by_branch.items()
            },
            "surface_accuracy": {
                surface: float(np.mean(values))
                for surface, values in by_surface.items()
            },
        }
    return family, item


def split_overlap_tables() -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], bool],
]:
    rows = []
    family_validity = {}
    for model in MODELS:
        selection = read_json(
            OUT_ROOT / "behavior" / model / "selection.json"
        )
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for mode in set(selection["selected_by_family"].values()):
            by_mode[mode] = read_jsonl(
                OUT_ROOT
                / "protocol"
                / f"cases.{model}.{mode}.jsonl"
            )
        for family in FAMILIES:
            mode = selection["selected_by_family"][family]
            family_cases = [
                row for row in by_mode[mode] if row["family"] == family
            ]
            item_ids = sorted({row["item_id"] for row in family_cases})
            item_overlaps = []
            for item_id in item_ids:
                prompts = {
                    split: {
                        row["raw_prompt"]
                        for row in family_cases
                        if row["item_id"] == item_id
                        and row["split"] == split
                    }
                    for split in SPLITS
                }
                overlap = prompts["discovery"] & prompts["confirmation"]
                denominator = min(
                    len(prompts["discovery"]),
                    len(prompts["confirmation"]),
                )
                ratio = len(overlap) / denominator if denominator else 1.0
                item_overlaps.append(ratio)
                rows.append({
                    "schema_version": (
                        "phase1018_split_overlap_audit.v1"
                    ),
                    "phase": PHASE,
                    "model": model,
                    "family": family,
                    "item_id": item_id,
                    "discovery_unique_prompt_count": len(
                        prompts["discovery"]
                    ),
                    "confirmation_unique_prompt_count": len(
                        prompts["confirmation"]
                    ),
                    "exact_overlap_count": len(overlap),
                    "exact_overlap_ratio": ratio,
                    "independent_confirmation": len(overlap) == 0,
                })
            family_validity[(model, family)] = all(
                ratio == 0.0 for ratio in item_overlaps
            )
    return rows, family_validity


def panel_catalog() -> dict[
    tuple[str, str, str, str], dict[str, Any]
]:
    catalog = {}
    for model in MODELS:
        model_root = OUT_ROOT / "formal_scan" / model
        for family in FAMILIES:
            family_root = model_root / family
            for item_root in sorted(family_root.iterdir()):
                for split in SPLITS:
                    panel_root = item_root / split
                    summary = read_json(panel_root / "summary.json")
                    key = (model, family, item_root.name, split)
                    catalog[key] = {
                        "root": panel_root,
                        "summary": summary,
                        **load_panel_metrics(panel_root),
                    }
    return catalog


def depth_band(relative_depth: float) -> str:
    if relative_depth <= 0:
        return "embedding"
    if relative_depth <= 1.0 / 3.0:
        return "early"
    if relative_depth <= 2.0 / 3.0:
        return "middle"
    return "late"


def event_signature(event: dict[str, Any], role: str) -> str:
    return "|".join(
        (
            str(event["component"]),
            role,
            depth_band(float(event["relative_depth"])),
        )
    )


def top_event_rows(
    *,
    model: str,
    family: str,
    item_id: str,
    events: list[dict[str, Any]],
    confirmed: np.ndarray,
    specific: np.ndarray,
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    matched: np.ndarray,
    mismatched: np.ndarray,
    gap: np.ndarray,
    limit: int = 40,
) -> list[dict[str, Any]]:
    rows = []
    for role_index, event_index in np.argwhere(confirmed):
        event = events[int(event_index)]
        rows.append({
            "schema_version": "phase1018_ranked_event.v1",
            "phase": PHASE,
            "model": model,
            "family": family,
            "item_id": item_id,
            "role": CAPTURE_ROLES[int(role_index)],
            "event_id": event["event_id"],
            "component": event["component"],
            "depth": event["depth"],
            "relative_depth": event["relative_depth"],
            "head": event["head"],
            "specific_gap_ge_0_15": bool(
                specific[int(role_index), int(event_index)]
            ),
            "discovery_magnitude": float(
                discovery["magnitude"][role_index, event_index]
            ),
            "confirmation_magnitude": float(
                confirmation["magnitude"][role_index, event_index]
            ),
            "discovery_consistency": float(
                discovery["consistency"][role_index, event_index]
            ),
            "confirmation_consistency": float(
                confirmation["consistency"][role_index, event_index]
            ),
            "discovery_surface_alignment": float(
                discovery["surface"][role_index, event_index]
            ),
            "confirmation_surface_alignment": float(
                confirmation["surface"][role_index, event_index]
            ),
            "matched_cosine": (
                float(matched[role_index, event_index])
                if np.isfinite(matched[role_index, event_index])
                else None
            ),
            "mismatched_median_cosine": (
                float(mismatched[role_index, event_index])
                if np.isfinite(mismatched[role_index, event_index])
                else None
            ),
            "specificity_gap": (
                float(gap[role_index, event_index])
                if np.isfinite(gap[role_index, event_index])
                else None
            ),
        })
    rows.sort(
        key=lambda row: (
            row["specificity_gap"] is not None,
            row["specificity_gap"]
            if row["specificity_gap"] is not None
            else -math.inf,
            min(
                row["discovery_consistency"],
                row["confirmation_consistency"],
            ),
            min(
                row["discovery_surface_alignment"],
                row["confirmation_surface_alignment"],
            ),
        ),
        reverse=True,
    )
    return rows[:limit]


def finalize() -> dict[str, Any]:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    behavior_family, behavior_item = behavior_tables()
    split_overlap_rows, independent_confirmation = (
        split_overlap_tables()
    )
    catalog = panel_catalog()
    item_rows = []
    family_rows = []
    shared_rows = []
    taotie_rows = []
    threshold_rows = []
    model_summaries = {
        model: read_json(
            OUT_ROOT / "formal_scan" / model / "summary.json"
        )
        for model in MODELS
    }
    family_gap_by_model: dict[tuple[str, str], float | None] = {}
    family_confirmed_by_model: dict[tuple[str, str], int] = {}
    majority_signatures: dict[
        tuple[str, str], set[str]
    ] = defaultdict(set)

    for model in MODELS:
        events = read_jsonl(
            OUT_ROOT / "formal_scan" / model / "events.jsonl"
        )
        for family in FAMILIES:
            item_ids = sorted({
                key[2]
                for key in catalog
                if key[0] == model and key[1] == family
            })
            panel_summaries = [
                catalog[(model, family, item_id, "discovery")]["summary"]
                for item_id in item_ids
            ]
            subgroup_by_item = {
                row["item_id"]: row["subgroup"] for row in panel_summaries
            }
            groups: dict[str, list[str]] = defaultdict(list)
            for item_id in item_ids:
                groups[subgroup_by_item[item_id]].append(item_id)

            confirmation_directions = {
                item_id: load_directions(
                    catalog[
                        (model, family, item_id, "confirmation")
                    ]["root"]
                )
                for item_id in item_ids
            }
            item_details = {}
            all_confirmed_gaps = []
            all_matched = []
            all_mismatched = []

            for item_id in item_ids:
                discovery = catalog[
                    (model, family, item_id, "discovery")
                ]
                confirmation = catalog[
                    (model, family, item_id, "confirmation")
                ]
                confirmed = (
                    discovery["candidate"] & confirmation["candidate"]
                )
                discovery_direction = load_directions(discovery["root"])
                matched = direction_cosine(
                    discovery_direction,
                    confirmation_directions[item_id],
                )
                alternatives = [
                    other
                    for other in groups[subgroup_by_item[item_id]]
                    if other != item_id
                ]
                mismatch_stack = np.stack([
                    direction_cosine(
                        discovery_direction,
                        confirmation_directions[other],
                    )
                    for other in alternatives
                ])
                mismatched = np.nanmedian(
                    mismatch_stack,
                    axis=0,
                ).astype(np.float32)
                gap = matched - mismatched
                specific = confirmed & np.isfinite(gap) & (gap >= GAP_GATE)
                confirmed_gap_values = gap[confirmed]
                all_confirmed_gaps.extend(
                    confirmed_gap_values[np.isfinite(
                        confirmed_gap_values
                    )].tolist()
                )
                all_matched.extend(
                    matched[confirmed & np.isfinite(matched)].tolist()
                )
                all_mismatched.extend(
                    mismatched[
                        confirmed & np.isfinite(mismatched)
                    ].tolist()
                )
                per_role = {}
                for role_index, role in enumerate(CAPTURE_ROLES):
                    role_confirmed = confirmed[role_index]
                    per_role[role] = {
                        "confirmed_count": int(role_confirmed.sum()),
                        "specific_count": int(
                            specific[role_index].sum()
                        ),
                        "median_matched_cosine": finite_median(
                            matched[role_index][role_confirmed]
                        ),
                        "median_mismatched_cosine": finite_median(
                            mismatched[role_index][role_confirmed]
                        ),
                        "median_specificity_gap": finite_median(
                            gap[role_index][role_confirmed]
                        ),
                    }
                first_residual = {}
                for role_index, role in enumerate(CAPTURE_ROLES):
                    depths = [
                        int(events[event_index]["depth"])
                        for event_index in np.flatnonzero(
                            confirmed[role_index]
                        )
                        if events[event_index]["component"] == "residual"
                    ]
                    first_residual[role] = min(depths) if depths else None

                row = {
                    "schema_version": "phase1018_item_summary.v1",
                    "phase": PHASE,
                    "protocol_revision": PROTOCOL_REVISION,
                    "model": model,
                    "family": family,
                    "subgroup": subgroup_by_item[item_id],
                    "item_id": item_id,
                    "behavior": behavior_item[(model, family, item_id)],
                    "discovery_candidate_count": int(
                        discovery["candidate"].sum()
                    ),
                    "confirmation_candidate_count": int(
                        confirmation["candidate"].sum()
                    ),
                    "confirmed_event_role_count": int(confirmed.sum()),
                    "specific_gap_ge_0_15_count": int(specific.sum()),
                    "specific_fraction_of_confirmed": (
                        float(specific.sum() / confirmed.sum())
                        if confirmed.any()
                        else None
                    ),
                    "median_matched_cosine": finite_median(
                        matched[confirmed]
                    ),
                    "median_mismatched_cosine": finite_median(
                        mismatched[confirmed]
                    ),
                    "median_specificity_gap": finite_median(
                        gap[confirmed]
                    ),
                    "first_confirmed_residual_depth": first_residual,
                    "by_role": per_role,
                    "claim_status": (
                        "contextual_response_only"
                        if behavior_item[
                            (model, family, item_id)
                        ]["candidate_accuracy"] < ACCURACY_GATE
                        else "behavior_supported_descriptive_response"
                    ),
                }
                item_rows.append(row)
                item_details[item_id] = {
                    "confirmed": confirmed,
                    "specific": specific,
                    "discovery": discovery,
                    "confirmation": confirmation,
                    "matched": matched,
                    "mismatched": mismatched,
                    "gap": gap,
                }
                if item_id == "taotie":
                    taotie_rows.extend(top_event_rows(
                        model=model,
                        family=family,
                        item_id=item_id,
                        events=events,
                        confirmed=confirmed,
                        specific=specific,
                        discovery=discovery,
                        confirmation=confirmation,
                        matched=matched,
                        mismatched=mismatched,
                        gap=gap,
                    ))

            for subgroup, subgroup_items in groups.items():
                subgroup_count = len(subgroup_items)
                majority_minimum = max(2, math.ceil(subgroup_count / 2))
                confirmed_count = np.stack([
                    item_details[item_id]["confirmed"]
                    for item_id in subgroup_items
                ]).sum(axis=0)
                specific_count = np.stack([
                    item_details[item_id]["specific"]
                    for item_id in subgroup_items
                ]).sum(axis=0)
                majority = confirmed_count >= majority_minimum
                ubiquitous = confirmed_count == subgroup_count
                for role_index, event_index in np.argwhere(majority):
                    event = events[int(event_index)]
                    member_items = [
                        item_id
                        for item_id in subgroup_items
                        if item_details[item_id]["confirmed"][
                            role_index, event_index
                        ]
                    ]
                    gaps = [
                        item_details[item_id]["gap"][
                            role_index, event_index
                        ]
                        for item_id in member_items
                    ]
                    matches = [
                        item_details[item_id]["matched"][
                            role_index, event_index
                        ]
                        for item_id in member_items
                    ]
                    shared_rows.append({
                        "schema_version": (
                            "phase1018_shared_physical_event.v1"
                        ),
                        "phase": PHASE,
                        "model": model,
                        "family": family,
                        "subgroup": subgroup,
                        "role": CAPTURE_ROLES[int(role_index)],
                        "event_id": event["event_id"],
                        "component": event["component"],
                        "depth": event["depth"],
                        "relative_depth": event["relative_depth"],
                        "depth_band": depth_band(
                            float(event["relative_depth"])
                        ),
                        "head": event["head"],
                        "item_count": int(
                            confirmed_count[role_index, event_index]
                        ),
                        "subgroup_item_count": subgroup_count,
                        "majority_minimum": majority_minimum,
                        "ubiquitous": bool(
                            ubiquitous[role_index, event_index]
                        ),
                        "specific_item_count": int(
                            specific_count[role_index, event_index]
                        ),
                        "member_items": member_items,
                        "median_matched_cosine": finite_median(matches),
                        "median_specificity_gap": finite_median(gaps),
                    })
                    majority_signatures[(model, family)].add(
                        event_signature(
                            event, CAPTURE_ROLES[int(role_index)]
                        )
                    )

            for direction_threshold in (0.30, 0.45, 0.60):
                for surface_threshold in (0.20, 0.40, 0.60):
                    confirmed_total = 0
                    item_with_confirmed = 0
                    for item_id in item_ids:
                        discovery = load_panel_metrics(
                            catalog[
                                (model, family, item_id, "discovery")
                            ]["root"],
                            direction_threshold,
                            surface_threshold,
                        )
                        confirmation = load_panel_metrics(
                            catalog[
                                (model, family, item_id, "confirmation")
                            ]["root"],
                            direction_threshold,
                            surface_threshold,
                        )
                        confirmed = (
                            discovery["candidate"]
                            & confirmation["candidate"]
                        )
                        confirmed_total += int(confirmed.sum())
                        item_with_confirmed += int(confirmed.any())
                    threshold_rows.append({
                        "schema_version": (
                            "phase1018_threshold_sensitivity.v1"
                        ),
                        "phase": PHASE,
                        "model": model,
                        "family": family,
                        "direction_consistency_threshold": (
                            direction_threshold
                        ),
                        "surface_alignment_threshold": surface_threshold,
                        "minimum_normalized_magnitude": (
                            PRIMARY_MAGNITUDE
                        ),
                        "confirmed_event_role_count": confirmed_total,
                        "item_with_confirmed_count": item_with_confirmed,
                        "item_count": len(item_ids),
                    })

            family_gap = finite_median(all_confirmed_gaps)
            confirmed_total = int(sum(
                details["confirmed"].sum()
                for details in item_details.values()
            ))
            family_gap_by_model[(model, family)] = family_gap
            family_confirmed_by_model[(model, family)] = confirmed_total
            component_role_counts: dict[
                tuple[str, str, str], dict[str, int]
            ] = defaultdict(lambda: {
                "confirmed": 0,
                "specific": 0,
            })
            for details in item_details.values():
                for role_index, event_index in np.argwhere(
                    details["confirmed"]
                ):
                    event = events[int(event_index)]
                    key = (
                        event["component"],
                        CAPTURE_ROLES[int(role_index)],
                        depth_band(float(event["relative_depth"])),
                    )
                    component_role_counts[key]["confirmed"] += 1
                    component_role_counts[key]["specific"] += int(
                        details["specific"][role_index, event_index]
                    )
            role_summary = {}
            for role_index, role in enumerate(CAPTURE_ROLES):
                role_confirmed = [
                    int(details["confirmed"][role_index].sum())
                    for details in item_details.values()
                ]
                role_specific = [
                    int(details["specific"][role_index].sum())
                    for details in item_details.values()
                ]
                role_matches = []
                role_mismatches = []
                role_gaps = []
                for details in item_details.values():
                    mask = details["confirmed"][role_index]
                    role_matches.extend(
                        details["matched"][role_index][
                            mask & np.isfinite(
                                details["matched"][role_index]
                            )
                        ].tolist()
                    )
                    role_mismatches.extend(
                        details["mismatched"][role_index][
                            mask & np.isfinite(
                                details["mismatched"][role_index]
                            )
                        ].tolist()
                    )
                    role_gaps.extend(
                        details["gap"][role_index][
                            mask & np.isfinite(details["gap"][role_index])
                        ].tolist()
                    )
                role_summary[role] = {
                    "confirmed_event_role_count": sum(role_confirmed),
                    "specific_gap_ge_0_15_count": sum(role_specific),
                    "median_matched_cosine": finite_median(role_matches),
                    "median_mismatched_cosine": finite_median(
                        role_mismatches
                    ),
                    "median_specificity_gap": finite_median(role_gaps),
                }
            family_rows.append({
                "schema_version": "phase1018_family_model_summary.v1",
                "phase": PHASE,
                "protocol_revision": PROTOCOL_REVISION,
                "model": model,
                "family": family,
                "item_count": len(item_ids),
                "subgroups": {
                    subgroup: len(values)
                    for subgroup, values in groups.items()
                },
                "behavior": behavior_family[(model, family)],
                "independent_confirmation": independent_confirmation[
                    (model, family)
                ],
                "confirmed_event_role_count": confirmed_total,
                "median_matched_cosine": finite_median(all_matched),
                "median_mismatched_cosine": finite_median(
                    all_mismatched
                ),
                "median_specificity_gap": family_gap,
                "gap_ge_0_15_fraction": (
                    float(np.mean(
                        np.asarray(all_confirmed_gaps) >= GAP_GATE
                    ))
                    if all_confirmed_gaps
                    else None
                ),
                "majority_physical_event_count": sum(
                    row["model"] == model
                    and row["family"] == family
                    for row in shared_rows
                ),
                "by_role": role_summary,
                "component_role_depth_counts": [
                    {
                        "component": key[0],
                        "role": key[1],
                        "depth_band": key[2],
                        **counts,
                    }
                    for key, counts in sorted(
                        component_role_counts.items()
                    )
                ],
                "claim_boundary": (
                    "descriptive_only_no_causal_closure"
                ),
            })
            del confirmation_directions, item_details

    cross_model_rows = []
    for family in FAMILIES:
        signatures = {
            model: majority_signatures[(model, family)]
            for model in MODELS
        }
        all_signatures = set().union(*signatures.values())
        for signature in sorted(all_signatures):
            models = [
                model for model in MODELS
                if signature in signatures[model]
            ]
            if len(models) >= 2:
                component, role, band = signature.split("|")
                cross_model_rows.append({
                    "schema_version": (
                        "phase1018_cross_model_band_signature.v1"
                    ),
                    "phase": PHASE,
                    "family": family,
                    "component": component,
                    "role": role,
                    "depth_band": band,
                    "model_count": len(models),
                    "models": models,
                    "claim_boundary": (
                        "coarse physical recurrence, not homologous neurons"
                    ),
                })

    continuation_rows = []
    for family in FAMILIES:
        behavior_models = [
            model for model in MODELS
            if behavior_family[(model, family)]["candidate_accuracy"]
            >= ACCURACY_GATE
        ]
        gap_models = [
            model for model in MODELS
            if family_gap_by_model[(model, family)] is not None
            and family_gap_by_model[(model, family)] >= GAP_GATE
        ]
        repeat_models = [
            model for model in MODELS
            if family_confirmed_by_model[(model, family)] > 0
            and independent_confirmation[(model, family)]
        ]
        start_causal = (
            len(behavior_models) >= 2
            and len(gap_models) >= 2
            and len(repeat_models) >= 2
        )
        continuation_rows.append({
            "family": family,
            "behavior_gate_models": behavior_models,
            "direction_gap_gate_models": gap_models,
            "confirmation_repeat_models": repeat_models,
            "start_targeted_causal_test": start_causal,
            "decision": (
                "start preregistered targeted causal test"
                if start_causal
                else "stop at descriptive atlas"
            ),
        })
    automatic = {
        "schema_version": "phase1018_automatic_continuation.v1",
        "phase": PHASE,
        "gates": {
            "candidate_accuracy": ACCURACY_GATE,
            "minimum_model_count": 2,
            "matched_minus_mismatched_direction_gap": GAP_GATE,
            "requires_discovery_confirmation_physical_repeat": True,
        },
        "by_family": continuation_rows,
        "any_targeted_causal_test_started": any(
            row["start_targeted_causal_test"]
            for row in continuation_rows
        ),
    }
    claim_ledger = {
        "schema_version": "phase1018_claim_ledger.v1",
        "phase": PHASE,
        "supported": [
            "Rare semantics produces independently held-out internal response "
            "candidates under the frozen descriptive thresholds.",
            "Physical recurrence, within-item direction recurrence, and "
            "cross-item direction reuse are separately measurable.",
            "Multi-token rare carriers and multi-token translation candidates "
            "can be audited without reducing them to the first token.",
        ],
        "invalidated_or_unconfirmed": [
            "Punctuation, translation, and contrast discovery-confirmation "
            "pairs reused exact prompt texts in protocol revision 3; their "
            "matched cosine of 1 is replay, not held-out replication.",
        ],
        "not_supported_without_later_causal_work": [
            "A repeated response event is necessary or sufficient for output.",
            "A shared physical event carries the same direction across words.",
            "Rare-word contextual response is complete lexical knowledge.",
            "The four families implement one universal language formula.",
            "Coarse cross-model depth recurrence identifies homologous neurons.",
        ],
        "formula_status": (
            "All Phase1018 equations are measurement definitions derived "
            "after protocol construction, not a proposed language law."
        ),
    }
    summary = {
        "schema_version": "phase1018_analysis_summary.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(MODELS),
        "families": list(FAMILIES),
        "model_scan_summaries": model_summaries,
        "counts": {
            "item_summary_rows": len(item_rows),
            "family_model_summary_rows": len(family_rows),
            "split_overlap_audit_rows": len(split_overlap_rows),
            "shared_physical_event_rows": len(shared_rows),
            "cross_model_band_rows": len(cross_model_rows),
            "taotie_ranked_event_rows": len(taotie_rows),
            "threshold_sensitivity_rows": len(threshold_rows),
        },
        "primary_descriptive_thresholds": {
            "direction_consistency": PRIMARY_DIRECTION,
            "surface_alignment": PRIMARY_SURFACE,
            "minimum_normalized_magnitude": PRIMARY_MAGNITUDE,
        },
        "automatic_continuation": automatic,
        "claim_ledger": claim_ledger,
    }
    write_jsonl(ANALYSIS_ROOT / "item_summary.jsonl", item_rows)
    write_jsonl(
        ANALYSIS_ROOT / "family_model_summary.jsonl", family_rows
    )
    write_jsonl(
        ANALYSIS_ROOT / "split_overlap_audit.jsonl", split_overlap_rows
    )
    write_jsonl(
        ANALYSIS_ROOT / "shared_physical_events.jsonl", shared_rows
    )
    write_jsonl(
        ANALYSIS_ROOT / "cross_model_band_signatures.jsonl",
        cross_model_rows,
    )
    write_jsonl(
        ANALYSIS_ROOT / "taotie_ranked_events.jsonl", taotie_rows
    )
    write_jsonl(
        ANALYSIS_ROOT / "threshold_sensitivity.jsonl", threshold_rows
    )
    write_json(ANALYSIS_ROOT / "automatic_continuation.json", automatic)
    write_json(ANALYSIS_ROOT / "claim_ledger.json", claim_ledger)
    write_json(ANALYSIS_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    finalize()
