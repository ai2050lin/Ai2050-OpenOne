#!/usr/bin/env python3
"""Aggregate Phase1015 without turning measurement thresholds into theory."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1015_query_surface_chain_atlas"
)
PHASE1014_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
MODELS = ("qwen3", "glm4", "deepseek7b")
TARGET_OPERATIONS = ("F", "Q")
MATCHED_CONTROLS = {
    "F": ("E", "N"),
    "Q": ("L",),
}
DIRECTION_THRESHOLD = 0.50
ORIENTATION_GAIN_THRESHOLD = 0.30
PREVALENCE_THRESHOLD = 0.70
EPSILON = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def finite_mean(values: list[float | None]) -> float | None:
    selected = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return finite(np.mean(selected)) if selected else None


def pairwise_consistency(vectors: list[np.ndarray]) -> float | None:
    normalized = []
    for value in vectors:
        norm = float(np.linalg.norm(value))
        if norm > EPSILON:
            normalized.append(
                value.astype(np.float64, copy=False) / norm
            )
    count = len(normalized)
    if count < 2:
        return None
    total = np.sum(normalized, axis=0)
    return finite(
        (float(np.dot(total, total)) - count)
        / (count * (count - 1))
    )


def model_panels(model: str) -> list[dict[str, Any]]:
    result = []
    model_root = SOURCE_ROOT / "scan" / model
    for family_root in sorted(
        path for path in model_root.iterdir()
        if path.is_dir()
    ):
        for surface_root in sorted(
            path for path in family_root.iterdir()
            if path.is_dir() and path.name.startswith("surface_")
        ):
            summary = read_json(surface_root / "summary.json")
            responses = np.load(surface_root / "response_scalars.npz")
            directions = np.load(
                surface_root / "direction_consistency.npz"
            )
            answer_directions = np.load(
                surface_root / "answer_head_direction_sums.npz"
            )
            normalized = responses[
                "normalized_magnitude"
            ].astype(np.float32, copy=False)
            operation_names = [
                str(value) for value in responses["operation_names"]
            ]
            role_names = [
                str(value) for value in responses["role_names"]
            ]
            whole_count = int(summary["whole_event_count"])
            head_percentiles = {}
            for operation in TARGET_OPERATIONS:
                target = normalized[
                    :,
                    operation_names.index(operation),
                    :,
                    whole_count:,
                ].astype(np.float64)
                matched = np.maximum.reduce([
                    normalized[
                        :,
                        operation_names.index(control),
                        :,
                        whole_count:,
                    ].astype(np.float64)
                    for control in MATCHED_CONTROLS[operation]
                ])
                median_delta = np.median(target - matched, axis=0)
                percentiles = np.empty_like(
                    median_delta, dtype=np.float32
                )
                for role_index in range(median_delta.shape[0]):
                    values = median_delta[role_index]
                    ordered = np.sort(values)
                    empirical_cdf = np.searchsorted(
                        ordered,
                        values,
                        side="right",
                    )
                    percentiles[role_index] = (
                        empirical_cdf / max(len(values), 1)
                    ).astype(np.float32)
                head_percentiles[operation] = percentiles
            result.append({
                "root": surface_root,
                "summary": summary,
                "units": read_jsonl(surface_root / "units.jsonl"),
                "normalized": normalized,
                "operation_names": operation_names,
                "role_names": role_names,
                "head_concentration_percentile": head_percentiles,
                "whole_direction": directions["whole"].astype(
                    np.float32, copy=False
                ),
                "head_direction": directions["head"].astype(
                    np.float32, copy=False
                ),
                "answer_head_sum": answer_directions[
                    "canonical_sum"
                ].astype(np.float32, copy=False),
                "answer_head_count": answer_directions["count"].astype(
                    np.int32, copy=False
                ),
            })
            responses.close()
            directions.close()
            answer_directions.close()
    return result


def cell_metrics(
    *,
    panel: dict[str, Any],
    operation: str,
    role_index: int,
    event_index: int,
    whole_count: int,
) -> dict[str, Any]:
    operation_index = panel["operation_names"].index(operation)
    target = panel["normalized"][
        :,
        operation_index,
        role_index,
        event_index,
    ].astype(np.float64)
    matched = np.maximum.reduce([
        panel["normalized"][
            :,
            panel["operation_names"].index(control),
            role_index,
            event_index,
        ].astype(np.float64)
        for control in MATCHED_CONTROLS[operation]
    ])
    envelope = np.maximum.reduce([
        panel["normalized"][
            :,
            panel["operation_names"].index(control),
            role_index,
            event_index,
        ].astype(np.float64)
        for control in ("E", "N", "L")
    ])
    target_index = TARGET_OPERATIONS.index(operation)
    if event_index < whole_count:
        raw = panel["whole_direction"][
            0, target_index, role_index, event_index
        ]
        canonical = panel["whole_direction"][
            1, target_index, role_index, event_index
        ]
    else:
        local = event_index - whole_count
        raw = panel["head_direction"][
            0, target_index, role_index, local
        ]
        canonical = panel["head_direction"][
            1, target_index, role_index, local
        ]
    raw_value = finite(raw)
    canonical_value = finite(canonical)
    gain = (
        canonical_value - raw_value
        if canonical_value is not None and raw_value is not None
        else None
    )
    prevalence = finite(np.mean(target > matched))
    envelope_prevalence = finite(np.mean(target > envelope))
    median_delta = finite(np.median(target - matched))
    direction_pass = bool(
        canonical_value is not None
        and canonical_value >= DIRECTION_THRESHOLD
        and gain is not None
        and gain >= ORIENTATION_GAIN_THRESHOLD
    )
    specificity_pass = bool(
        prevalence is not None
        and prevalence >= PREVALENCE_THRESHOLD
        and median_delta is not None
        and median_delta > 0
    )
    concentration_percentile = (
        finite(panel["head_concentration_percentile"][operation][
            role_index, event_index - whole_count
        ])
        if event_index >= whole_count else None
    )
    return {
        "raw_direction_consistency": raw_value,
        "canonical_direction_consistency": canonical_value,
        "orientation_gain": gain,
        "matched_control_prevalence": prevalence,
        "full_control_prevalence": envelope_prevalence,
        "matched_control_median_delta": median_delta,
        "target_median": finite(np.median(target)),
        "matched_control_median": finite(np.median(matched)),
        "direction_pass": direction_pass,
        "specificity_pass": specificity_pass,
        "member_pass": bool(direction_pass and specificity_pass),
        "head_concentration_percentile": concentration_percentile,
        "concentrated_member_pass": bool(
            direction_pass
            and specificity_pass
            and concentration_percentile is not None
            and concentration_percentile >= 0.90
        ),
    }


def aggregate_profile(
    cells: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    selected = [row for row in cells if row["split"] == split]
    passed = [row for row in selected if row["member_pass"]]
    concentrated = [
        row for row in selected if row["concentrated_member_pass"]
    ]
    directional = [row for row in selected if row["direction_pass"]]
    specific = [row for row in selected if row["specificity_pass"]]
    natural_surfaces = {0, 1} if split == "discovery" else {3, 4}
    balanced_surface = 2 if split == "discovery" else 5
    return {
        "panel_count": len(selected),
        "member_panel_count": len(passed),
        "concentrated_member_panel_count": len(concentrated),
        "direction_panel_count": len(directional),
        "specificity_panel_count": len(specific),
        "families": sorted({row["family"] for row in passed}),
        "surfaces": sorted({row["query_surface"] for row in passed}),
        "natural_member_panel_count": sum(
            row["query_surface"] in natural_surfaces for row in passed
        ),
        "balanced_member_panel_count": sum(
            row["query_surface"] == balanced_surface for row in passed
        ),
        "concentrated_families": sorted({
            row["family"] for row in concentrated
        }),
        "concentrated_surfaces": sorted({
            row["query_surface"] for row in concentrated
        }),
        "natural_concentrated_panel_count": sum(
            row["query_surface"] in natural_surfaces
            for row in concentrated
        ),
        "balanced_concentrated_panel_count": sum(
            row["query_surface"] == balanced_surface
            for row in concentrated
        ),
        "mean_canonical_consistency": finite_mean([
            row["canonical_direction_consistency"]
            for row in selected
        ]),
        "mean_orientation_gain": finite_mean([
            row["orientation_gain"]
            for row in selected
        ]),
        "mean_matched_control_prevalence": finite(np.mean([
            row["matched_control_prevalence"] for row in selected
        ])),
        "median_matched_control_delta": finite(np.median([
            row["matched_control_median_delta"] for row in selected
        ])),
    }


def answer_direction_stats(
    panels: list[dict[str, Any]],
    *,
    event_index: int,
    whole_count: int,
    operation: str,
    passing_keys: set[tuple[str, int]],
    split: str,
) -> dict[str, Any]:
    if event_index < whole_count:
        return {
            "cross_surface_within_family": None,
            "cross_family_all_panels": None,
            "vector_panel_count": 0,
        }
    local = event_index - whole_count
    target_index = TARGET_OPERATIONS.index(operation)
    vectors = {}
    for panel in panels:
        summary = panel["summary"]
        if summary["split"] != split:
            continue
        key = (summary["family"], int(summary["query_surface"]))
        if key not in passing_keys:
            continue
        count = int(panel["answer_head_count"][target_index, local])
        if count <= 0:
            continue
        vectors[key] = (
            panel["answer_head_sum"][target_index, local] / count
        )
    within_family = []
    for family in sorted({key[0] for key in vectors}):
        value = pairwise_consistency([
            vector for (name, _), vector in vectors.items()
            if name == family
        ])
        if value is not None:
            within_family.append(value)
    return {
        "cross_surface_within_family": (
            finite(np.mean(within_family))
            if within_family else None
        ),
        "cross_family_all_panels": pairwise_consistency(
            list(vectors.values())
        ),
        "vector_panel_count": len(vectors),
        "family_with_multiple_surface_vector_count": len(within_family),
    }


def build_model_profiles(model: str) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    panels = model_panels(model)
    model_summary = read_json(SOURCE_ROOT / "scan" / model / "summary.json")
    events = read_jsonl(SOURCE_ROOT / "scan" / model / "events.jsonl")
    whole_count = int(model_summary["whole_event_count"])
    role_names = panels[0]["role_names"]
    profiles = []
    answer_candidates = []
    for event in events:
        event_index = int(event["event_index"])
        for operation in TARGET_OPERATIONS:
            for role_index, role in enumerate(role_names):
                cells = []
                for panel in panels:
                    metrics = cell_metrics(
                        panel=panel,
                        operation=operation,
                        role_index=role_index,
                        event_index=event_index,
                        whole_count=whole_count,
                    )
                    summary = panel["summary"]
                    metrics.update({
                        "family": summary["family"],
                        "query_surface": int(
                            summary["query_surface"]
                        ),
                        "split": summary["split"],
                        "balanced_query_inventory": bool(
                            summary["balanced_query_inventory"]
                        ),
                    })
                    cells.append(metrics)
                discovery = aggregate_profile(
                    cells, split="discovery"
                )
                confirmation = aggregate_profile(
                    cells, split="confirmation"
                )
                profile = {
                    "schema_version": (
                        "phase1015_event_role_profile.v1"
                    ),
                    "phase": 1015,
                    "model": model,
                    "event_id": event["event_id"],
                    "event_index": event_index,
                    "component": event["component"],
                    "depth": int(event["depth"]),
                    "relative_depth": float(event["relative_depth"]),
                    "head": event["head"],
                    "operation": operation,
                    "receiver_role": role,
                    "splits": {
                        "discovery": discovery,
                        "confirmation": confirmation,
                    },
                    "claim": (
                        "repeated role-conditioned response; not "
                        "transport, mediation, necessity, or sufficiency"
                    ),
                }
                profiles.append(profile)
                if (
                    role == "answer_boundary"
                    and event["component"]
                    == "attention_head_pre_o_proj"
                ):
                    recurrent = bool(
                        discovery["member_panel_count"] >= 4
                        and len(discovery["families"]) >= 2
                        and len(discovery["surfaces"]) >= 2
                    )
                    surface_diversified = bool(
                        recurrent
                        and discovery[
                            "natural_member_panel_count"
                        ] >= 1
                        and discovery[
                            "balanced_member_panel_count"
                        ] >= 1
                    )
                    concentrated_core = bool(
                        discovery[
                            "concentrated_member_panel_count"
                        ] >= 3
                        and len(discovery[
                            "concentrated_families"
                        ]) >= 2
                        and len(discovery[
                            "concentrated_surfaces"
                        ]) >= 2
                        and discovery[
                            "natural_concentrated_panel_count"
                        ] >= 1
                        and discovery[
                            "balanced_concentrated_panel_count"
                        ] >= 1
                    )
                    passing_keys = {
                        (row["family"], row["query_surface"])
                        for row in cells
                        if row["member_pass"]
                    }
                    direction_stats = {
                        split: answer_direction_stats(
                            panels,
                            event_index=event_index,
                            whole_count=whole_count,
                            operation=operation,
                            passing_keys={
                                key for key in passing_keys
                                if next(
                                    panel["summary"]["split"]
                                    for panel in panels
                                    if (
                                        panel["summary"]["family"],
                                        int(panel["summary"][
                                            "query_surface"
                                        ]),
                                    ) == key
                                ) == split
                            },
                            split=split,
                        )
                        for split in ("discovery", "confirmation")
                    }
                    answer_candidates.append({
                        **profile,
                        "recurrent_discovery_member": recurrent,
                        "surface_diversified_discovery_member": (
                            surface_diversified
                        ),
                        "concentrated_discovery_core": (
                            concentrated_core
                        ),
                        "confirmation_recurrent": bool(
                            confirmation["member_panel_count"] >= 4
                            and len(confirmation["families"]) >= 2
                            and len(confirmation["surfaces"]) >= 2
                        ),
                        "confirmation_balanced_recurrence": bool(
                            confirmation[
                                "balanced_member_panel_count"
                            ] >= 1
                        ),
                        "confirmation_concentrated_core": bool(
                            confirmation[
                                "concentrated_member_panel_count"
                            ] >= 3
                            and len(confirmation[
                                "concentrated_families"
                            ]) >= 2
                            and len(confirmation[
                                "concentrated_surfaces"
                            ]) >= 2
                            and confirmation[
                                "natural_concentrated_panel_count"
                            ] >= 1
                            and confirmation[
                                "balanced_concentrated_panel_count"
                            ] >= 1
                        ),
                        "direction_family": direction_stats,
                    })

    candidate_lookup = {
        (row["event_id"], row["operation"]): row
        for row in answer_candidates
    }
    trajectories = []
    for candidate in answer_candidates:
        if not candidate["concentrated_discovery_core"]:
            continue
        role_profiles = {
            row["receiver_role"]: row
            for row in profiles
            if row["event_id"] == candidate["event_id"]
            and row["operation"] == candidate["operation"]
        }
        trajectories.append({
            "schema_version": "phase1015_role_trajectory.v1",
            "phase": 1015,
            "model": model,
            "event_id": candidate["event_id"],
            "operation": candidate["operation"],
            "depth": candidate["depth"],
            "head": candidate["head"],
            "role_member_panel_counts": {
                split: {
                    role: role_profiles[role]["splits"][split][
                        "member_panel_count"
                    ]
                    for role in role_names
                }
                for split in ("discovery", "confirmation")
            },
            "role_direction_panel_counts": {
                split: {
                    role: role_profiles[role]["splits"][split][
                        "direction_panel_count"
                    ]
                    for role in role_names
                }
                for split in ("discovery", "confirmation")
            },
            "interpretation_limit": (
                "ordered role recurrence is a trajectory signature, "
                "not a directed transport edge"
            ),
        })

    phase1014_rows = read_jsonl(
        PHASE1014_ROOT
        / "analysis"
        / "control_specific_shared_events.jsonl"
    )
    phase1014_rows = [
        row for row in phase1014_rows if row["model"] == model
    ]
    phase1014_rechecks = []
    for old in phase1014_rows:
        key = (old["event_id"], old["operation"])
        current = candidate_lookup.get(key)
        phase1014_rechecks.append({
            "event_id": old["event_id"],
            "operation": old["operation"],
            "present_in_phase1015": current is not None,
            "recurrent_discovery_member": (
                current["recurrent_discovery_member"]
                if current else False
            ),
            "surface_diversified_discovery_member": (
                current["surface_diversified_discovery_member"]
                if current else False
            ),
            "concentrated_discovery_core": (
                current["concentrated_discovery_core"]
                if current else False
            ),
            "confirmation_recurrent": (
                current["confirmation_recurrent"] if current else False
            ),
            "confirmation_concentrated_core": (
                current["confirmation_concentrated_core"]
                if current else False
            ),
        })

    behavior_rows = [
        row
        for panel in panels
        for row in panel["units"]
    ]
    behavior_by_surface = {}
    for surface in range(6):
        selected = [
            row for row in behavior_rows
            if int(row["query_surface"]) == surface
        ]
        behavior_by_surface[str(surface)] = {
            "n_units": len(selected),
            "mean_state_panel_hit_rate": finite(np.mean([
                hit
                for row in selected
                for hit in row["singleton_state_hits"].values()
            ])),
            "base_q_pair_hit_rate": finite(np.mean([
                row["singleton_state_hits"]["base"]
                and row["singleton_state_hits"]["Q"]
                for row in selected
            ])),
        }

    selected_candidates = [
        row for row in answer_candidates
        if row["recurrent_discovery_member"]
    ]
    lexical_candidates = [
        row for row in selected_candidates
        if row["surface_diversified_discovery_member"]
    ]
    summary = {
        "schema_version": "phase1015_model_analysis.v1",
        "phase": 1015,
        "model": model,
        "event_role_profile_count": len(profiles),
        "answer_head_candidate_count": len(answer_candidates),
        "recurrent_answer_head_count": len(selected_candidates),
        "surface_diversified_discovery_head_count": len(
            lexical_candidates
        ),
        "concentrated_discovery_core_count": sum(
            row["concentrated_discovery_core"]
            for row in answer_candidates
        ),
        "confirmation_concentrated_core_count": sum(
            row["concentrated_discovery_core"]
            and row["confirmation_concentrated_core"]
            for row in answer_candidates
        ),
        "confirmation_recurrent_head_count": sum(
            row["confirmation_recurrent"]
            for row in selected_candidates
        ),
        "confirmation_balanced_head_count": sum(
            row["confirmation_balanced_recurrence"]
            for row in selected_candidates
        ),
        "by_operation": {
            operation: {
                "recurrent": sum(
                    row["operation"] == operation
                    for row in selected_candidates
                ),
                "surface_diversified_discovery": sum(
                    row["operation"] == operation
                    for row in lexical_candidates
                ),
                "confirmation_recurrent": sum(
                    row["operation"] == operation
                    and row["confirmation_recurrent"]
                    for row in selected_candidates
                ),
            }
            for operation in TARGET_OPERATIONS
        },
        "phase1014_recheck": {
            "frozen_count": len(phase1014_rechecks),
            "recurrent_count": sum(
                row["recurrent_discovery_member"]
                for row in phase1014_rechecks
            ),
            "surface_diversified_count": sum(
                row["surface_diversified_discovery_member"]
                for row in phase1014_rechecks
            ),
            "concentrated_core_count": sum(
                row["concentrated_discovery_core"]
                for row in phase1014_rechecks
            ),
            "confirmation_recurrent_count": sum(
                row["confirmation_recurrent"]
                for row in phase1014_rechecks
            ),
            "confirmation_concentrated_core_count": sum(
                row["confirmation_concentrated_core"]
                for row in phase1014_rechecks
            ),
            "events": phase1014_rechecks,
        },
        "behavior_by_surface": behavior_by_surface,
        "identity_maximum": model_summary["identity_maximum"],
        "q_causal_prefix_maximum": model_summary[
            "q_causal_prefix_maximum"
        ],
    }
    return summary, profiles, trajectories


def threshold_sensitivity(model: str) -> list[dict[str, Any]]:
    panels = model_panels(model)
    summary = read_json(SOURCE_ROOT / "scan" / model / "summary.json")
    events = read_jsonl(SOURCE_ROOT / "scan" / model / "events.jsonl")
    whole_count = int(summary["whole_event_count"])
    answer_role = panels[0]["role_names"].index("answer_boundary")
    head_events = [
        event for event in events
        if event["component"] == "attention_head_pre_o_proj"
    ]
    result = []
    for direction_threshold in (0.30, 0.40, 0.50, 0.60):
        for prevalence_threshold in (0.50, 0.60, 0.70, 0.80):
            counts = {operation: 0 for operation in TARGET_OPERATIONS}
            core_counts = {
                operation: 0 for operation in TARGET_OPERATIONS
            }
            for operation in TARGET_OPERATIONS:
                for event in head_events:
                    passed = []
                    concentrated = []
                    for panel in panels:
                        if panel["summary"]["split"] != "discovery":
                            continue
                        cell = cell_metrics(
                            panel=panel,
                            operation=operation,
                            role_index=answer_role,
                            event_index=int(event["event_index"]),
                            whole_count=whole_count,
                        )
                        custom_pass = bool(
                            cell[
                                "canonical_direction_consistency"
                            ] is not None
                            and cell[
                                "canonical_direction_consistency"
                            ] >= direction_threshold
                            and cell["orientation_gain"] is not None
                            and cell["orientation_gain"] >= 0.20
                            and cell[
                                "matched_control_prevalence"
                            ] >= prevalence_threshold
                            and cell[
                                "matched_control_median_delta"
                            ] > 0
                        )
                        if custom_pass:
                            key = (
                                panel["summary"]["family"],
                                int(panel["summary"][
                                    "query_surface"
                                ]),
                            )
                            passed.append(key)
                            if (
                                cell[
                                    "head_concentration_percentile"
                                ] is not None
                                and cell[
                                    "head_concentration_percentile"
                                ] >= 0.90
                            ):
                                concentrated.append(key)
                    if (
                        len(passed) >= 4
                        and len({value[0] for value in passed}) >= 2
                        and len({value[1] for value in passed}) >= 2
                    ):
                        counts[operation] += 1
                    if (
                        len(concentrated) >= 3
                        and len({
                            value[0] for value in concentrated
                        }) >= 2
                        and len({
                            value[1] for value in concentrated
                        }) >= 2
                        and any(
                            value[1] in {0, 1}
                            for value in concentrated
                        )
                        and any(
                            value[1] == 2 for value in concentrated
                        )
                    ):
                        core_counts[operation] += 1
            result.append({
                "model": model,
                "direction_threshold": direction_threshold,
                "orientation_gain_threshold": 0.20,
                "prevalence_threshold": prevalence_threshold,
                "recurrent_F_count": counts["F"],
                "recurrent_Q_count": counts["Q"],
                "concentrated_percentile_threshold": 0.90,
                "concentrated_F_core_count": core_counts["F"],
                "concentrated_Q_core_count": core_counts["Q"],
            })
    return result


def concentration_sensitivity(model: str) -> list[dict[str, Any]]:
    """Vary descriptive concentration rulers without changing member tests."""
    panels = model_panels(model)
    summary = read_json(SOURCE_ROOT / "scan" / model / "summary.json")
    events = read_jsonl(SOURCE_ROOT / "scan" / model / "events.jsonl")
    whole_count = int(summary["whole_event_count"])
    answer_role = panels[0]["role_names"].index("answer_boundary")
    head_events = [
        event for event in events
        if event["component"] == "attention_head_pre_o_proj"
    ]
    cached: dict[tuple[str, int], list[tuple[str, int, float]]] = {}
    for operation in TARGET_OPERATIONS:
        for event in head_events:
            cells = []
            for panel in panels:
                if panel["summary"]["split"] != "discovery":
                    continue
                cell = cell_metrics(
                    panel=panel,
                    operation=operation,
                    role_index=answer_role,
                    event_index=int(event["event_index"]),
                    whole_count=whole_count,
                )
                percentile = cell["head_concentration_percentile"]
                if cell["member_pass"] and percentile is not None:
                    cells.append((
                        panel["summary"]["family"],
                        int(panel["summary"]["query_surface"]),
                        float(percentile),
                    ))
            cached[(operation, int(event["event_index"]))] = cells

    result = []
    for percentile_threshold in (0.80, 0.90, 0.95):
        for minimum_panels in (2, 3, 4):
            counts = {operation: 0 for operation in TARGET_OPERATIONS}
            for operation in TARGET_OPERATIONS:
                for event in head_events:
                    selected = [
                        value
                        for value in cached[(
                            operation,
                            int(event["event_index"]),
                        )]
                        if value[2] >= percentile_threshold
                    ]
                    if (
                        len(selected) >= minimum_panels
                        and len({value[0] for value in selected}) >= 2
                        and len({value[1] for value in selected}) >= 2
                        and any(value[1] in {0, 1} for value in selected)
                        and any(value[1] == 2 for value in selected)
                    ):
                        counts[operation] += 1
            result.append({
                "model": model,
                "member_direction_threshold": DIRECTION_THRESHOLD,
                "member_orientation_gain_threshold": (
                    ORIENTATION_GAIN_THRESHOLD
                ),
                "member_prevalence_threshold": PREVALENCE_THRESHOLD,
                "concentrated_percentile_threshold": (
                    percentile_threshold
                ),
                "minimum_concentrated_panels": minimum_panels,
                "concentrated_F_core_count": counts["F"],
                "concentrated_Q_core_count": counts["Q"],
            })
    return result


def depth_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    depths = sorted(float(row["relative_depth"]) for row in rows)
    return {
        "count": len(rows),
        "relative_depth_min": finite(min(depths)) if depths else None,
        "relative_depth_median": (
            finite(np.median(depths)) if depths else None
        ),
        "relative_depth_max": finite(max(depths)) if depths else None,
        "depths": [int(row["depth"]) for row in rows],
    }


def whole_component_role_summary(
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    components = ("residual", "attention_output", "mlp_output")
    roles = sorted({row["receiver_role"] for row in profiles})
    for component in components:
        result[component] = {}
        for operation in TARGET_OPERATIONS:
            result[component][operation] = {}
            for role in roles:
                selected = [
                    row for row in profiles
                    if row["component"] == component
                    and row["operation"] == operation
                    and row["receiver_role"] == role
                ]
                discovery_recurrent = [
                    row for row in selected
                    if (
                        row["splits"]["discovery"][
                            "member_panel_count"
                        ] >= 4
                        and len(row["splits"]["discovery"][
                            "families"
                        ]) >= 2
                        and len(row["splits"]["discovery"][
                            "surfaces"
                        ]) >= 2
                    )
                ]
                surface_diversified = [
                    row for row in discovery_recurrent
                    if (
                        row["splits"]["discovery"][
                            "natural_member_panel_count"
                        ] >= 1
                        and row["splits"]["discovery"][
                            "balanced_member_panel_count"
                        ] >= 1
                    )
                ]
                independently_confirmed = [
                    row for row in discovery_recurrent
                    if (
                        row["splits"]["confirmation"][
                            "member_panel_count"
                        ] >= 4
                        and len(row["splits"]["confirmation"][
                            "families"
                        ]) >= 2
                        and len(row["splits"]["confirmation"][
                            "surfaces"
                        ]) >= 2
                    )
                ]
                result[component][operation][role] = {
                    "discovery_recurrent": depth_shape(
                        discovery_recurrent
                    ),
                    "surface_diversified": depth_shape(
                        surface_diversified
                    ),
                    "independently_confirmed": depth_shape(
                        independently_confirmed
                    ),
                }
    return result


def behavior_stratified_core_profiles(
    *,
    model: str,
    panels: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    answer_role = panels[0]["role_names"].index("answer_boundary")
    result = []
    for candidate in candidates:
        if not candidate["concentrated_discovery_core"]:
            continue
        operation = candidate["operation"]
        event_index = int(candidate["event_index"])
        pair_hit_deltas = []
        pair_miss_deltas = []
        for panel in panels:
            if panel["summary"]["split"] != "confirmation":
                continue
            operation_names = panel["operation_names"]
            target = panel["normalized"][
                :,
                operation_names.index(operation),
                answer_role,
                event_index,
            ].astype(np.float64)
            matched = np.maximum.reduce([
                panel["normalized"][
                    :,
                    operation_names.index(control),
                    answer_role,
                    event_index,
                ].astype(np.float64)
                for control in MATCHED_CONTROLS[operation]
            ])
            deltas = target - matched
            for unit, delta in zip(panel["units"], deltas):
                hits = unit["singleton_state_hits"]
                if bool(hits["base"] and hits[operation]):
                    pair_hit_deltas.append(float(delta))
                else:
                    pair_miss_deltas.append(float(delta))
        hit_median = (
            finite(np.median(pair_hit_deltas))
            if pair_hit_deltas else None
        )
        miss_median = (
            finite(np.median(pair_miss_deltas))
            if pair_miss_deltas else None
        )
        result.append({
            "schema_version": (
                "phase1015_behavior_stratified_core_profile.v1"
            ),
            "phase": 1015,
            "model": model,
            "event_id": candidate["event_id"],
            "operation": operation,
            "depth": candidate["depth"],
            "head": candidate["head"],
            "confirmation_concentrated_core": candidate[
                "confirmation_concentrated_core"
            ],
            "candidate_panel_pair_hit_count": len(pair_hit_deltas),
            "candidate_panel_pair_miss_count": len(pair_miss_deltas),
            "pair_hit_median_matched_control_delta": hit_median,
            "pair_miss_median_matched_control_delta": miss_median,
            "pair_hit_minus_miss_median_delta": (
                hit_median - miss_median
                if hit_median is not None and miss_median is not None
                else None
            ),
            "pair_hit_positive_delta_rate": (
                finite(np.mean(np.asarray(pair_hit_deltas) > 0))
                if pair_hit_deltas else None
            ),
            "pair_miss_positive_delta_rate": (
                finite(np.mean(np.asarray(pair_miss_deltas) > 0))
                if pair_miss_deltas else None
            ),
            "interpretation_limit": (
                "candidate-panel correctness stratification is "
                "post-selection descriptive evidence, not natural "
                "generation accuracy or a causal test"
            ),
        })
    return result


def behavior_stratified_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for operation in TARGET_OPERATIONS:
        result[operation] = {}
        operation_rows = [
            row for row in rows if row["operation"] == operation
        ]
        for label, selected in (
            ("discovery_cores", operation_rows),
            (
                "independently_confirmed_cores",
                [
                    row for row in operation_rows
                    if row["confirmation_concentrated_core"]
                ],
            ),
        ):
            eligible = [
                row for row in selected
                if row["pair_hit_minus_miss_median_delta"] is not None
            ]
            result[operation][label] = {
                "core_count": len(selected),
                "eligible_core_count": len(eligible),
                "median_pair_hit_minus_miss_delta": (
                    finite(np.median([
                        row["pair_hit_minus_miss_median_delta"]
                        for row in eligible
                    ]))
                    if eligible else None
                ),
                "positive_contrast_core_fraction": (
                    finite(np.mean([
                        row["pair_hit_minus_miss_median_delta"] > 0
                        for row in eligible
                    ]))
                    if eligible else None
                ),
                "median_pair_hit_delta": (
                    finite(np.median([
                        row[
                            "pair_hit_median_matched_control_delta"
                        ]
                        for row in eligible
                    ]))
                    if eligible else None
                ),
                "median_pair_miss_delta": (
                    finite(np.median([
                        row[
                            "pair_miss_median_matched_control_delta"
                        ]
                        for row in eligible
                    ]))
                    if eligible else None
                ),
            }
    return result


def main() -> None:
    protocol = read_json(SOURCE_ROOT / "protocol" / "protocol.json")
    output_root = SOURCE_ROOT / "analysis"
    model_summaries = {}
    all_answer_profiles = []
    all_candidates = []
    all_cores = []
    all_trajectories = []
    all_behavior_stratified = []
    all_sensitivity = []
    all_concentration_sensitivity = []
    for model in MODELS:
        summary, profiles, trajectories = build_model_profiles(model)
        write_jsonl(
            output_root / model / "event_role_profiles.jsonl",
            profiles,
        )
        candidates = [
            row for row in profiles
            if row["receiver_role"] == "answer_boundary"
            and row["component"] == "attention_head_pre_o_proj"
        ]
        del candidates
        write_jsonl(
            output_root / model / "role_trajectories.jsonl",
            trajectories,
        )
        # Reconstruct compact answer rows from the trajectory source.
        model_profiles = read_jsonl(
            output_root / model / "event_role_profiles.jsonl"
        )
        summary["whole_component_role_recurrence"] = (
            whole_component_role_summary(model_profiles)
        )
        answer_profiles = [
            row for row in model_profiles
            if row["receiver_role"] == "answer_boundary"
            and row["component"] == "attention_head_pre_o_proj"
        ]
        # The complete candidate fields are generated a second time only
        # for the compact file to keep profile rows generic.
        panels = model_panels(model)
        model_scan = read_json(
            SOURCE_ROOT / "scan" / model / "summary.json"
        )
        whole_count = int(model_scan["whole_event_count"])
        role_index = panels[0]["role_names"].index("answer_boundary")
        compact_candidates = []
        for profile in answer_profiles:
            discovery = profile["splits"]["discovery"]
            confirmation = profile["splits"]["confirmation"]
            recurrent = bool(
                discovery["member_panel_count"] >= 4
                and len(discovery["families"]) >= 2
                and len(discovery["surfaces"]) >= 2
            )
            lexical = bool(
                recurrent
                and discovery["natural_member_panel_count"] >= 1
                and discovery["balanced_member_panel_count"] >= 1
            )
            concentrated_core = bool(
                discovery["concentrated_member_panel_count"] >= 3
                and len(discovery["concentrated_families"]) >= 2
                and len(discovery["concentrated_surfaces"]) >= 2
                and discovery[
                    "natural_concentrated_panel_count"
                ] >= 1
                and discovery[
                    "balanced_concentrated_panel_count"
                ] >= 1
            )
            passing_by_split = defaultdict(set)
            for panel in panels:
                cell = cell_metrics(
                    panel=panel,
                    operation=profile["operation"],
                    role_index=role_index,
                    event_index=profile["event_index"],
                    whole_count=whole_count,
                )
                if cell["member_pass"]:
                    passing_by_split[
                        panel["summary"]["split"]
                    ].add((
                        panel["summary"]["family"],
                        int(panel["summary"]["query_surface"]),
                    ))
            direction_family = {
                split: answer_direction_stats(
                    panels,
                    event_index=profile["event_index"],
                    whole_count=whole_count,
                    operation=profile["operation"],
                    passing_keys=passing_by_split[split],
                    split=split,
                )
                for split in ("discovery", "confirmation")
            }
            compact_candidates.append({
                **profile,
                "recurrent_discovery_member": recurrent,
                "surface_diversified_discovery_member": lexical,
                "concentrated_discovery_core": concentrated_core,
                "confirmation_recurrent": bool(
                    confirmation["member_panel_count"] >= 4
                    and len(confirmation["families"]) >= 2
                    and len(confirmation["surfaces"]) >= 2
                ),
                "confirmation_balanced_recurrence": bool(
                    confirmation["balanced_member_panel_count"] >= 1
                ),
                "confirmation_concentrated_core": bool(
                    confirmation[
                        "concentrated_member_panel_count"
                    ] >= 3
                    and len(confirmation[
                        "concentrated_families"
                    ]) >= 2
                    and len(confirmation[
                        "concentrated_surfaces"
                    ]) >= 2
                    and confirmation[
                        "natural_concentrated_panel_count"
                    ] >= 1
                    and confirmation[
                        "balanced_concentrated_panel_count"
                    ] >= 1
                ),
                "direction_family": direction_family,
                "phase1014_frozen": any(
                    old["event_id"] == profile["event_id"]
                    and old["operation"] == profile["operation"]
                    for old in read_jsonl(
                        PHASE1014_ROOT
                        / "analysis"
                        / "control_specific_shared_events.jsonl"
                    )
                    if old["model"] == model
                ),
            })
        behavior_stratified = behavior_stratified_core_profiles(
            model=model,
            panels=panels,
            candidates=compact_candidates,
        )
        summary["behavior_stratified_confirmation"] = (
            behavior_stratified_summary(behavior_stratified)
        )
        write_jsonl(
            output_root / model / "answer_head_candidates.jsonl",
            compact_candidates,
        )
        write_jsonl(
            output_root
            / model
            / "behavior_stratified_core_profiles.jsonl",
            behavior_stratified,
        )
        write_json(output_root / model / "summary.json", summary)
        model_summaries[model] = summary
        all_answer_profiles.extend(compact_candidates)
        all_candidates.extend([
            row for row in compact_candidates
            if row["recurrent_discovery_member"]
        ])
        all_cores.extend([
            row for row in compact_candidates
            if row["concentrated_discovery_core"]
        ])
        all_trajectories.extend(trajectories)
        all_behavior_stratified.extend(behavior_stratified)
        all_sensitivity.extend(threshold_sensitivity(model))
        all_concentration_sensitivity.extend(
            concentration_sensitivity(model)
        )

    by_model_operation = {}
    for model in MODELS:
        by_model_operation[model] = {}
        for operation in TARGET_OPERATIONS:
            answer_rows = [
                row for row in all_answer_profiles
                if row["model"] == model
                and row["operation"] == operation
            ]
            rows = [
                row for row in all_candidates
                if row["model"] == model
                and row["operation"] == operation
            ]
            core_rows = [
                row for row in answer_rows
                if row["concentrated_discovery_core"]
            ]
            confirmed_core_rows = [
                row for row in core_rows
                if row["confirmation_concentrated_core"]
            ]
            by_model_operation[model][operation] = {
                "recurrent_count": len(rows),
                "surface_diversified_count": sum(
                    row["surface_diversified_discovery_member"]
                    for row in rows
                ),
                "confirmation_recurrent_count": sum(
                    row["confirmation_recurrent"] for row in rows
                ),
                "confirmation_balanced_count": sum(
                    row["confirmation_balanced_recurrence"]
                    for row in rows
                ),
                "concentrated_core_count": sum(
                    row["concentrated_discovery_core"]
                    for row in answer_rows
                ),
                "confirmation_concentrated_core_count": sum(
                    row["concentrated_discovery_core"]
                    and row["confirmation_concentrated_core"]
                    for row in answer_rows
                ),
                "cross_surface_direction_median": (
                    finite(np.median([
                        row["direction_family"]["confirmation"][
                            "cross_surface_within_family"
                        ]
                        for row in rows
                        if row["direction_family"]["confirmation"][
                            "cross_surface_within_family"
                        ] is not None
                    ]))
                    if any(
                        row["direction_family"]["confirmation"][
                            "cross_surface_within_family"
                        ] is not None
                        for row in rows
                    )
                    else None
                ),
                "core_confirmation_cross_surface_direction_median": (
                    finite(np.median([
                        row["direction_family"]["confirmation"][
                            "cross_surface_within_family"
                        ]
                        for row in confirmed_core_rows
                        if row["direction_family"]["confirmation"][
                            "cross_surface_within_family"
                        ] is not None
                    ]))
                    if any(
                        row["direction_family"]["confirmation"][
                            "cross_surface_within_family"
                        ] is not None
                        for row in confirmed_core_rows
                    )
                    else None
                ),
                "core_confirmation_cross_family_direction_median": (
                    finite(np.median([
                        row["direction_family"]["confirmation"][
                            "cross_family_all_panels"
                        ]
                        for row in confirmed_core_rows
                        if row["direction_family"]["confirmation"][
                            "cross_family_all_panels"
                        ] is not None
                    ]))
                    if any(
                        row["direction_family"]["confirmation"][
                            "cross_family_all_panels"
                        ] is not None
                        for row in confirmed_core_rows
                    )
                    else None
                ),
            }

    instrument_comparison = {}
    for namespace in ("scan_smoke", "scan_smoke_bf16"):
        path = SOURCE_ROOT / namespace / "qwen3" / "summary.json"
        if path.exists():
            row = read_json(path)
            instrument_comparison[namespace] = {
                "loaded_8bit": row["model_info"]["loaded_8bit"],
                "identity_maximum": row["identity_maximum"],
                "q_causal_prefix_maximum": row[
                    "q_causal_prefix_maximum"
                ],
            }
    trajectory_summary = {}
    if all_trajectories:
        role_names = list(
            all_trajectories[0]["role_member_panel_counts"][
                "discovery"
            ]
        )
        for model in MODELS:
            trajectory_summary[model] = {}
            for operation in TARGET_OPERATIONS:
                rows = [
                    row for row in all_trajectories
                    if row["model"] == model
                    and row["operation"] == operation
                ]
                trajectory_summary[model][operation] = {
                    "core_count": len(rows),
                    "depth_counts": {
                        str(depth): sum(
                            int(row["depth"]) == depth for row in rows
                        )
                        for depth in sorted({
                            int(row["depth"]) for row in rows
                        })
                    },
                    "median_role_member_panel_counts": {
                        split: {
                            role: (
                                finite(np.median([
                                    row[
                                        "role_member_panel_counts"
                                    ][split][role]
                                    for row in rows
                                ]))
                                if rows else None
                            )
                            for role in role_names
                        }
                        for split in ("discovery", "confirmation")
                    },
                }
    result = {
        "schema_version": "phase1015_analysis_summary.v1",
        "phase": 1015,
        "protocol_digest": protocol["preregistration_digest"],
        "main_scan_precision": "BF16",
        "model_summaries": model_summaries,
        "recurrent_answer_head_count": len(all_candidates),
        "surface_diversified_discovery_head_count": sum(
            row["surface_diversified_discovery_member"]
            for row in all_candidates
        ),
        "concentrated_discovery_core_count": len(all_cores),
        "confirmation_concentrated_core_count": sum(
            row["confirmation_concentrated_core"]
            for row in all_cores
        ),
        "confirmation_recurrent_head_count": sum(
            row["confirmation_recurrent"] for row in all_candidates
        ),
        "confirmation_balanced_head_count": sum(
            row["confirmation_balanced_recurrence"]
            for row in all_candidates
        ),
        "by_model_operation": by_model_operation,
        "concentrated_core_trajectory_summary": trajectory_summary,
        "behavior_stratified_confirmation": {
            model: model_summaries[model][
                "behavior_stratified_confirmation"
            ]
            for model in MODELS
        },
        "instrument_precision_comparison": instrument_comparison,
        "threshold_sensitivity": all_sensitivity,
        "concentration_sensitivity": all_concentration_sensitivity,
        "interpretation": {
            "membership_axis": (
                "whether the same physical component repeatedly responds "
                "across disjoint query surfaces"
            ),
            "direction_axis": (
                "whether its within-component direction is fixed or "
                "surface-conditioned"
            ),
            "trajectory_axis": (
                "which ordered token roles repeatedly respond; this is "
                "not yet a transport edge"
            ),
        },
        "claim_limits": protocol["preregistered_claim_limits"],
    }
    write_jsonl(output_root / "recurrent_answer_heads.jsonl", all_candidates)
    write_jsonl(output_root / "concentrated_answer_head_cores.jsonl", all_cores)
    write_jsonl(output_root / "role_trajectories.jsonl", all_trajectories)
    write_jsonl(
        output_root / "behavior_stratified_core_profiles.jsonl",
        all_behavior_stratified,
    )
    write_json(output_root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
