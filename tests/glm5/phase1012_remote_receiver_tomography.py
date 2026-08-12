#!/usr/bin/env python3
"""Re-map Phase1011 at an unchanged receiver instead of edited token sites.

The assistant answer-boundary token is identical at residual depth zero.
Responses appearing there after layer computation are therefore remote
context responses. They remain descriptive until independently intervened.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1011_native_semantic_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUT_ROOT as PHASE1011_ROOT,
    OUTPUT_MODES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE = 1012
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1012_remote_receiver_tomography"
)
OP_INDEX = {name: index for index, name in enumerate(ANALYSIS_OPERATIONS)}
SPLIT_INDEX = {"discovery": 0, "confirmation": 1}
DIRECTION_AXES = ("semantic_panel", "natural_rollout")
TARGET_OPERATIONS = ("F", "Q", "FQ", "X")
SURFACE_CONTROL_OPERATIONS = ("E", "O", "N", "S")
MIN_N = 8
MIN_POOLS = 2
MIN_TEMPLATES = 2
DISCOVERY_DIRECTION = 0.50
DISCOVERY_PREVALENCE = 0.70
SENSITIVITY_DIRECTIONS = (0.30, 0.50, 0.70)
SENSITIVITY_PREVALENCES = (0.60, 0.70, 0.80)


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def coverage(
    units: list[dict[str, Any]],
    mask: np.ndarray,
) -> dict[str, Any]:
    rows = [row for index, row in enumerate(units) if mask[index]]
    return {
        "n": len(rows),
        "name_pools": sorted({
            int(row["name_pool"]) for row in rows
        }),
        "templates": sorted({
            int(row["template"]) for row in rows
        }),
    }


def evidence_pass(
    row: dict[str, Any],
    *,
    split: str,
    direction_threshold: float,
    prevalence_threshold: float,
) -> bool:
    values = row["splits"][split]
    return bool(
        values["coverage"]["n"] >= MIN_N
        and len(values["coverage"]["name_pools"]) >= MIN_POOLS
        and len(values["coverage"]["templates"]) >= MIN_TEMPLATES
        and values["direction_consistency"] is not None
        and values["direction_consistency"] >= direction_threshold
        and values["control_envelope_prevalence"] is not None
        and values["control_envelope_prevalence"]
        >= prevalence_threshold
        and values["control_envelope_delta"] is not None
        and values["control_envelope_delta"] > 0
    )


def panel_rows(
    *,
    model: str,
    family: str,
    output_mode: str,
) -> list[dict[str, Any]]:
    panel_root = (
        PHASE1011_ROOT
        / "scan"
        / model
        / family
        / output_mode
    )
    events = read_jsonl(panel_root / "events.jsonl")
    units = read_jsonl(panel_root / "units.jsonl")
    scalar = np.load(panel_root / "response_scalars.npz")
    direction = np.load(panel_root / "direction_consistency.npz")
    values = scalar["normalized_magnitude"]
    directions = direction["direction_consistency"]
    direction_counts = direction["direction_count"]
    split_masks = {
        split: np.asarray(
            [row["split"] == split for row in units],
            dtype=np.bool_,
        )
        for split in SPLIT_INDEX
    }
    receiver_events = [
        event for event in events
        if event["stage"] == "prompt"
        and event["role_class"] == "answer_boundary"
        and not (
            event["component"] == "residual"
            and int(event["depth"]) == 0
        )
    ]
    rows = []
    for axis_index, axis in enumerate(DIRECTION_AXES):
        qualified = scalar[f"{axis}_qualified"]
        for operation in TARGET_OPERATIONS:
            operation_index = OP_INDEX[operation]
            for event in receiver_events:
                event_index = int(event["event_index"])
                split_rows = {}
                for split, split_index in SPLIT_INDEX.items():
                    mask = (
                        qualified[:, operation_index]
                        & split_masks[split]
                    )
                    target = values[
                        mask, operation_index, event_index
                    ]
                    controls = values[
                        mask, :, event_index
                    ][:, [
                        OP_INDEX[name]
                        for name in SURFACE_CONTROL_OPERATIONS
                    ]]
                    envelope = (
                        np.nanmax(controls, axis=1)
                        if target.size
                        else np.asarray([], dtype=np.float32)
                    )
                    split_rows[split] = {
                        "coverage": coverage(units, mask),
                        "target_median": (
                            None if not target.size
                            else finite(np.nanmedian(target))
                        ),
                        "surface_control_envelope_median": (
                            None if not envelope.size
                            else finite(np.nanmedian(envelope))
                        ),
                        "control_envelope_delta": (
                            None if not target.size
                            else finite(np.nanmedian(target - envelope))
                        ),
                        "control_envelope_prevalence": (
                            None if not target.size
                            else finite(np.nanmean(target > envelope))
                        ),
                        "direction_consistency": finite(
                            directions[
                                axis_index,
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        ),
                        "direction_count": int(
                            direction_counts[
                                axis_index,
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        ),
                    }
                row = {
                    "schema_version": (
                        "phase1012_remote_receiver_event.v1"
                    ),
                    "phase": PHASE,
                    "source_phase": 1011,
                    "model": model,
                    "family": family,
                    "output_mode": output_mode,
                    "qualification_axis": axis,
                    "operation": operation,
                    "event_id": event["event_id"],
                    "component": event["component"],
                    "depth": int(event["depth"]),
                    "relative_depth": float(event["relative_depth"]),
                    "receiver_role": "answer_boundary",
                    "receiver_input_delta_at_depth0": 0.0,
                    "surface_control_operations": list(
                        SURFACE_CONTROL_OPERATIONS
                    ),
                    "splits": split_rows,
                    "discovery_pass": False,
                    "confirmation_pass": False,
                    "claim": "remote_context_response_only",
                }
                row["discovery_pass"] = evidence_pass(
                    row,
                    split="discovery",
                    direction_threshold=DISCOVERY_DIRECTION,
                    prevalence_threshold=DISCOVERY_PREVALENCE,
                )
                row["confirmation_pass"] = evidence_pass(
                    row,
                    split="confirmation",
                    direction_threshold=DISCOVERY_DIRECTION,
                    prevalence_threshold=DISCOVERY_PREVALENCE,
                )
                rows.append(row)
    scalar.close()
    direction.close()
    return rows


def curve_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["model"],
            row["family"],
            row["output_mode"],
            row["qualification_axis"],
            row["operation"],
            row["component"],
        )].append(row)
    result = []
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda row: int(row["depth"]))
        peaks = {}
        for split in SPLIT_INDEX:
            eligible = [
                row for row in members
                if row["splits"][split][
                    "control_envelope_delta"
                ] is not None
            ]
            peak = (
                None
                if not eligible
                else max(
                    eligible,
                    key=lambda row: row["splits"][split][
                        "control_envelope_delta"
                    ],
                )
            )
            peaks[split] = (
                None if peak is None
                else {
                    "depth": int(peak["depth"]),
                    "relative_depth": float(
                        peak["relative_depth"]
                    ),
                    "control_envelope_delta": peak["splits"][
                        split
                    ]["control_envelope_delta"],
                    "control_envelope_prevalence": peak["splits"][
                        split
                    ]["control_envelope_prevalence"],
                    "direction_consistency": peak["splits"][
                        split
                    ]["direction_consistency"],
                }
            )
        discovery_depths = [
            int(row["depth"]) for row in members
            if row["discovery_pass"]
        ]
        confirmation_depths = [
            int(row["depth"]) for row in members
            if row["confirmation_pass"]
        ]
        both_depths = sorted(
            set(discovery_depths) & set(confirmation_depths)
        )
        stable_peak = bool(
            peaks["discovery"] is not None
            and peaks["confirmation"] is not None
            and abs(
                peaks["discovery"]["relative_depth"]
                - peaks["confirmation"]["relative_depth"]
            ) <= 0.15
        )
        model, family, output_mode, axis, operation, component = key
        result.append({
            "schema_version": "phase1012_remote_receiver_curve.v1",
            "phase": PHASE,
            "model": model,
            "family": family,
            "output_mode": output_mode,
            "qualification_axis": axis,
            "operation": operation,
            "component": component,
            "depth_count": len(members),
            "discovery_pass_depths": discovery_depths,
            "confirmation_pass_depths": confirmation_depths,
            "both_split_pass_depths": both_depths,
            "both_split_pass_count": len(both_depths),
            "peaks": peaks,
            "peak_relative_depth_stable_within_0_15": stable_peak,
            "claim": "remote_response_curve_only",
        })
    return result


def discovery_selection(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze broad regions using discovery recurrence only."""
    grouped: dict[tuple, dict[tuple, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not row["discovery_pass"]:
            continue
        physical = (
            row["model"],
            row["operation"],
            row["component"],
            int(row["depth"]),
        )
        panel = (row["family"], row["output_mode"])
        entry = grouped[physical].setdefault(
            panel,
            {"axes": set(), "rows": []},
        )
        entry["axes"].add(row["qualification_axis"])
        entry["rows"].append(row)
    selections = []
    for model in MODELS:
        for operation in TARGET_OPERATIONS:
            for component in (
                "residual",
                "attention_output",
                "mlp_output",
            ):
                candidates = []
                for physical, panel_map in grouped.items():
                    if physical[:3] != (
                        model, operation, component
                    ):
                        continue
                    both_axis_panels = sum(
                        len(value["axes"]) == 2
                        for value in panel_map.values()
                    )
                    candidates.append({
                        "physical": physical,
                        "panel_count": len(panel_map),
                        "both_axis_panel_count": both_axis_panels,
                        "families": sorted({
                            panel[0] for panel in panel_map
                        }),
                        "output_modes": sorted({
                            panel[1] for panel in panel_map
                        }),
                    })
                if not candidates:
                    continue
                maximum_panel_count = max(
                    row["panel_count"] for row in candidates
                )
                recurrent = [
                    row for row in candidates
                    if row["panel_count"] == maximum_panel_count
                ]
                maximum_both_axis = max(
                    row["both_axis_panel_count"]
                    for row in recurrent
                )
                recurrent = [
                    row for row in recurrent
                    if row["both_axis_panel_count"]
                    == maximum_both_axis
                ]
                recurrent.sort(key=lambda row: row["physical"][3])
                for rank, candidate in enumerate(recurrent[:3], 1):
                    depth = int(candidate["physical"][3])
                    confirmation_rows = [
                        row for row in rows
                        if row["model"] == model
                        and row["operation"] == operation
                        and row["component"] == component
                        and int(row["depth"]) == depth
                    ]
                    selections.append({
                        "schema_version": (
                            "phase1012_discovery_frozen_region.v1"
                        ),
                        "phase": PHASE,
                        "model": model,
                        "operation": operation,
                        "component": component,
                        "depth": depth,
                        "selection_rank_among_ties": rank,
                        "selection_used_confirmation": False,
                        "discovery_panel_count": candidate[
                            "panel_count"
                        ],
                        "discovery_both_axis_panel_count": candidate[
                            "both_axis_panel_count"
                        ],
                        "discovery_families": candidate["families"],
                        "discovery_output_modes": candidate[
                            "output_modes"
                        ],
                        "confirmation_panel_count": len({
                            (row["family"], row["output_mode"])
                            for row in confirmation_rows
                            if row["confirmation_pass"]
                        }),
                        "confirmation_both_axis_panel_count": sum(
                            len({
                                row["qualification_axis"]
                                for row in confirmation_rows
                                if row["family"] == family
                                and row["output_mode"] == output_mode
                                and row["confirmation_pass"]
                            }) == 2
                            for family in FAMILIES
                            for output_mode in OUTPUT_MODES
                        ),
                        "claim": (
                            "broad_receiver_region_for_future_refinement"
                        ),
                    })
    return {
        "schema_version": "phase1012_discovery_selection.v1",
        "phase": PHASE,
        "selection_used_confirmation": False,
        "selection_rule": (
            "maximum number of discovery panels passing separate "
            "direction, prevalence, coverage, and positive-envelope axes; "
            "then maximum panels passing both behavior qualifications"
        ),
        "operational_thresholds_not_theory": {
            "direction": DISCOVERY_DIRECTION,
            "control_envelope_prevalence": DISCOVERY_PREVALENCE,
            "minimum_n": MIN_N,
            "minimum_name_pools": MIN_POOLS,
            "minimum_templates": MIN_TEMPLATES,
        },
        "selections": selections,
    }


def main() -> None:
    source_summary = read_json(
        PHASE1011_ROOT / "final" / "summary.json"
    )
    rows = []
    for model in MODELS:
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                rows.extend(panel_rows(
                    model=model,
                    family=family,
                    output_mode=output_mode,
                ))
    curves = curve_summaries(rows)
    selection = discovery_selection(rows)
    sensitivity = []
    for direction in SENSITIVITY_DIRECTIONS:
        for prevalence in SENSITIVITY_PREVALENCES:
            sensitivity.append({
                "direction_threshold": direction,
                "prevalence_threshold": prevalence,
                "discovery_event_count": int(sum(
                    evidence_pass(
                        row,
                        split="discovery",
                        direction_threshold=direction,
                        prevalence_threshold=prevalence,
                    )
                    for row in rows
                )),
                "confirmation_event_count": int(sum(
                    evidence_pass(
                        row,
                        split="confirmation",
                        direction_threshold=direction,
                        prevalence_threshold=prevalence,
                    )
                    for row in rows
                )),
                "both_split_event_count": int(sum(
                    evidence_pass(
                        row,
                        split="discovery",
                        direction_threshold=direction,
                        prevalence_threshold=prevalence,
                    )
                    and evidence_pass(
                        row,
                        split="confirmation",
                        direction_threshold=direction,
                        prevalence_threshold=prevalence,
                    )
                    for row in rows
                )),
            })
    both_split = [
        row for row in rows
        if row["discovery_pass"] and row["confirmation_pass"]
    ]
    by_model = {
        model: {
            "event_count": int(sum(
                row["model"] == model for row in rows
            )),
            "both_split_pass_count": int(sum(
                row["model"] == model for row in both_split
            )),
            "both_split_by_operation": dict(sorted(Counter(
                row["operation"]
                for row in both_split
                if row["model"] == model
            ).items())),
            "both_split_by_component": dict(sorted(Counter(
                row["component"]
                for row in both_split
                if row["model"] == model
            ).items())),
        }
        for model in MODELS
    }
    summary = {
        "schema_version": "phase1012_remote_receiver_summary.v1",
        "phase": PHASE,
        "source_phase": 1011,
        "source_protocol_digest": source_summary["protocol_digest"],
        "principle": (
            "exclude directly edited token sites and teacher-forced answer "
            "states; map contextual response emerging at an unchanged "
            "answer-boundary receiver"
        ),
        "receiver_input_delta_at_residual_depth0": 0.0,
        "surface_control_envelope": list(
            SURFACE_CONTROL_OPERATIONS
        ),
        "event_count": len(rows),
        "both_split_pass_count": len(both_split),
        "curve_count": len(curves),
        "curves_with_any_both_split_depth": int(sum(
            row["both_split_pass_count"] > 0 for row in curves
        )),
        "curves_with_stable_peak_relative_depth": int(sum(
            row["peak_relative_depth_stable_within_0_15"]
            for row in curves
        )),
        "discovery_frozen_region_count": len(
            selection["selections"]
        ),
        "by_model": by_model,
        "sensitivity": sensitivity,
        "claim_limits": [
            "remote response is not transport or causal mediation",
            "the answer boundary aggregates all preceding context",
            "the surface-control envelope is conservative but not a "
            "perfect lexical match",
            "operational thresholds select follow-up regions and are not "
            "language equations",
        ],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / "receiver_events.jsonl", rows)
    write_jsonl(OUT_ROOT / "receiver_curves.jsonl", curves)
    write_json(OUT_ROOT / "discovery_selection.json", selection)
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
