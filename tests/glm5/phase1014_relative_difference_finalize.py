#!/usr/bin/env python3
"""Summarize repeated relative-difference responses without causal claims."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1014_relative_difference_atlas"
)
MODELS = ("qwen3", "glm4", "deepseek7b")
FAMILIES = (
    "comparison",
    "negation",
    "semantic_role",
    "attribute_binding",
    "spatial_relation",
)
OUTPUT_MODES = ("entity", "property", "binary")
OPERATIONS = ("F", "Q", "FQ", "E", "O", "N", "L", "I", "X")
OP_INDEX = {name: index for index, name in enumerate(OPERATIONS)}
TARGET_OPERATIONS = ("F", "Q")
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_OPERATIONS)}
SPLITS = ("discovery", "confirmation")
SPLIT_INDEX = {name: index for index, name in enumerate(SPLITS)}
DIRECTION_MODES = ("raw", "canonical")
DIRECTION_MODE_INDEX = {
    name: index for index, name in enumerate(DIRECTION_MODES)
}
DIRECTION_AXES = (
    "all_units",
    "singleton_panel",
    "natural_rollout",
)
DIRECTION_AXIS_INDEX = {
    name: index for index, name in enumerate(DIRECTION_AXES)
}
MATCHED_CONTROLS = {
    "F": ("E", "N"),
    "Q": ("L",),
}
CONTROL_OPERATIONS = ("E", "O", "N", "L")
PRIMARY_DIRECTION_THRESHOLD = 0.50
PRIMARY_ORIENTATION_GAIN = 0.30
PRIMARY_CONTROL_PREVALENCE = 0.70
PRIMARY_RECURRENT_PANELS = 4
PRIMARY_CROSS_PANEL_CONSISTENCY = 0.30
PRIMARY_SPECIFICITY_PANELS = 4
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


def pairwise_consistency(vectors: list[np.ndarray]) -> float | None:
    normalized = []
    for value in vectors:
        norm = float(np.linalg.norm(value))
        if norm > EPSILON:
            normalized.append(value.astype(np.float64, copy=False) / norm)
    count = len(normalized)
    if count < 2:
        return None
    summed = np.sum(normalized, axis=0)
    return finite(
        (float(np.dot(summed, summed)) - count)
        / (count * (count - 1))
    )


def profile_pass(
    canonical: float | None,
    raw: float | None,
    *,
    direction_threshold: float = PRIMARY_DIRECTION_THRESHOLD,
    orientation_gain: float = PRIMARY_ORIENTATION_GAIN,
) -> bool:
    return bool(
        canonical is not None
        and raw is not None
        and canonical >= direction_threshold
        and canonical - raw >= orientation_gain
    )


def specificity_pass(
    prevalence: float | None,
    median_delta: float | None,
    *,
    prevalence_threshold: float = PRIMARY_CONTROL_PREVALENCE,
) -> bool:
    return bool(
        prevalence is not None
        and prevalence >= prevalence_threshold
        and median_delta is not None
        and median_delta > 0
    )


def panel_mean_directions(
    path: Path,
    event_count: int,
) -> np.ndarray:
    bundle = np.load(path)
    whole = bundle["whole"]
    heads = bundle["head"]
    whole_count = bundle["whole_count"]
    head_count = bundle["head_count"]
    result = np.empty(
        (
            len(TARGET_OPERATIONS),
            len(SPLITS),
            event_count,
        ),
        dtype=object,
    )
    for operation_index in range(len(TARGET_OPERATIONS)):
        for split_index in range(len(SPLITS)):
            event_index = 0
            for vectors, counts in (
                (
                    whole[operation_index, split_index],
                    whole_count[operation_index, split_index],
                ),
                (
                    heads[operation_index, split_index],
                    head_count[operation_index, split_index],
                ),
            ):
                for local_index in range(vectors.shape[0]):
                    count = int(counts[local_index])
                    result[
                        operation_index,
                        split_index,
                        event_index,
                    ] = (
                        vectors[local_index].astype(
                            np.float32, copy=False
                        ) / count
                        if count > 0
                        else None
                    )
                    event_index += 1
            if event_index != event_count:
                raise RuntimeError(
                    f"event direction count drift: {event_index} "
                    f"!= {event_count}"
                )
    bundle.close()
    return result


def panel_metrics(
    *,
    scan_root: Path,
    analysis_root: Path,
    model: str,
    family: str,
    output_mode: str,
    event_count: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    panel_root = scan_root / model / family / output_mode
    units = read_jsonl(panel_root / "units.jsonl")
    scalar = np.load(panel_root / "response_scalars.npz")
    direction = np.load(panel_root / "direction_consistency.npz")
    values = scalar["normalized_magnitude"]
    consistency = direction["direction_consistency"]
    counts = direction["direction_count"]
    metrics = {
        "canonical": np.full(
            (len(DIRECTION_AXES), 2, 2, event_count),
            np.nan,
            dtype=np.float32,
        ),
        "raw": np.full(
            (len(DIRECTION_AXES), 2, 2, event_count),
            np.nan,
            dtype=np.float32,
        ),
        "direction_count": np.zeros(
            (len(DIRECTION_AXES), 2, 2, event_count),
            dtype=np.int32,
        ),
        "target_median": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "matched_control_median": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "matched_delta_median": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "matched_prevalence": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "control_envelope_median": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "control_envelope_delta_median": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "control_envelope_prevalence": np.full(
            (2, 2, event_count), np.nan, dtype=np.float32
        ),
        "fq_median": np.full(
            (2, event_count), np.nan, dtype=np.float32
        ),
        "interaction_median": np.full(
            (2, event_count), np.nan, dtype=np.float32
        ),
        "lexical_control_median": np.full(
            (2, event_count), np.nan, dtype=np.float32
        ),
        "unit_count": np.zeros(2, dtype=np.int32),
    }
    for split, split_index in SPLIT_INDEX.items():
        mask = np.asarray(
            [row["split"] == split for row in units],
            dtype=np.bool_,
        )
        metrics["unit_count"][split_index] = int(np.sum(mask))
        split_values = values[mask]
        if not split_values.shape[0]:
            continue
        metrics["fq_median"][split_index] = np.median(
            split_values[:, OP_INDEX["FQ"], :], axis=0
        )
        metrics["interaction_median"][split_index] = np.median(
            split_values[:, OP_INDEX["X"], :], axis=0
        )
        metrics["lexical_control_median"][split_index] = np.median(
            split_values[:, OP_INDEX["L"], :], axis=0
        )
        for operation, operation_index in TARGET_INDEX.items():
            target = split_values[:, OP_INDEX[operation], :]
            matched = np.max(
                split_values[
                    :,
                    [
                        OP_INDEX[name]
                        for name in MATCHED_CONTROLS[operation]
                    ],
                    :,
                ],
                axis=1,
            )
            envelope = np.max(
                split_values[
                    :,
                    [OP_INDEX[name] for name in CONTROL_OPERATIONS],
                    :,
                ],
                axis=1,
            )
            metrics["target_median"][
                operation_index, split_index
            ] = np.median(target, axis=0)
            metrics["matched_control_median"][
                operation_index, split_index
            ] = np.median(matched, axis=0)
            metrics["matched_delta_median"][
                operation_index, split_index
            ] = np.median(target - matched, axis=0)
            metrics["matched_prevalence"][
                operation_index, split_index
            ] = np.mean(target > matched, axis=0)
            metrics["control_envelope_median"][
                operation_index, split_index
            ] = np.median(envelope, axis=0)
            metrics["control_envelope_delta_median"][
                operation_index, split_index
            ] = np.median(target - envelope, axis=0)
            metrics["control_envelope_prevalence"][
                operation_index, split_index
            ] = np.mean(target > envelope, axis=0)
            for axis, axis_index in DIRECTION_AXIS_INDEX.items():
                metrics["raw"][
                    axis_index, operation_index, split_index
                ] = consistency[
                    DIRECTION_MODE_INDEX["raw"],
                    axis_index,
                    operation_index,
                    split_index,
                ]
                metrics["canonical"][
                    axis_index, operation_index, split_index
                ] = consistency[
                    DIRECTION_MODE_INDEX["canonical"],
                    axis_index,
                    operation_index,
                    split_index,
                ]
                metrics["direction_count"][
                    axis_index, operation_index, split_index
                ] = counts[
                    axis_index,
                    operation_index,
                    split_index,
                ]
    metric_root = (
        analysis_root / "panel_metrics" / model / family / output_mode
    )
    metric_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metric_root / "metrics.npz", **metrics)
    directions = panel_mean_directions(
        panel_root / "canonical_direction_sums.npz",
        event_count,
    )
    scalar.close()
    direction.close()
    return metrics, directions


def historical_heads() -> set[tuple[str, str, int, int]]:
    path = (
        ROOT
        / "tests"
        / "glm5"
        / "result"
        / "phase1013_head_response_morphology"
        / "discovery_frozen_heads.jsonl"
    )
    if not path.exists():
        return set()
    return {
        (
            str(row["model"]),
            str(row["operation"]),
            int(row["depth"]),
            int(row["head"]),
        )
        for row in read_jsonl(path)
    }


def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["component"],
        int(event["depth"]),
        event.get("head"),
    )


def recurrent_pass(row: dict[str, Any]) -> bool:
    discovery = row["splits"]["discovery"]
    return bool(
        discovery["directional_panel_count"]
        >= PRIMARY_RECURRENT_PANELS
        and len(discovery["families"]) >= 2
        and len(discovery["output_modes"]) >= 2
    )


def finalize(
    scan_namespace: str = "scan",
    models: tuple[str, ...] = MODELS,
) -> dict[str, Any]:
    scan_root = OUT_ROOT / scan_namespace
    analysis_root = OUT_ROOT / (
        "analysis" if scan_namespace == "scan"
        else f"analysis_{scan_namespace}"
    )
    history = historical_heads()
    all_event_profiles = []
    model_summaries = {}
    sensitivity_cache: dict[
        tuple[str, str, str], dict[str, np.ndarray]
    ] = {}

    for model in models:
        model_scan_summary = read_json(scan_root / model / "summary.json")
        events = read_jsonl(scan_root / model / "events.jsonl")
        event_count = len(events)
        panel_data: dict[
            tuple[str, str],
            tuple[dict[str, np.ndarray], np.ndarray],
        ] = {}
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel_data[(family, output_mode)] = panel_metrics(
                    scan_root=scan_root,
                    analysis_root=analysis_root,
                    model=model,
                    family=family,
                    output_mode=output_mode,
                    event_count=event_count,
                )
                sensitivity_cache[
                    (model, family, output_mode)
                ] = panel_data[(family, output_mode)][0]

        for operation, operation_index in TARGET_INDEX.items():
            for event_index, event in enumerate(events):
                split_rows = {}
                axis_rows = {}
                for axis, axis_index in DIRECTION_AXIS_INDEX.items():
                    axis_rows[axis] = {}
                    for split, split_index in SPLIT_INDEX.items():
                        local_passes = []
                        for panel, (metrics, _) in panel_data.items():
                            canonical = finite(
                                metrics["canonical"][
                                    axis_index,
                                    operation_index,
                                    split_index,
                                    event_index,
                                ]
                            )
                            raw = finite(
                                metrics["raw"][
                                    axis_index,
                                    operation_index,
                                    split_index,
                                    event_index,
                                ]
                            )
                            if profile_pass(canonical, raw):
                                local_passes.append(panel)
                        axis_rows[axis][split] = {
                            "directional_panel_count": len(local_passes),
                            "families": sorted({
                                family for family, _ in local_passes
                            }),
                            "output_modes": sorted({
                                output for _, output in local_passes
                            }),
                        }

                for split, split_index in SPLIT_INDEX.items():
                    directional_panels = []
                    specific_panels = []
                    control_envelope_panels = []
                    panel_vectors = []
                    canonical_values = []
                    raw_values = []
                    gains = []
                    matched_prevalences = []
                    matched_deltas = []
                    fq_values = []
                    interaction_values = []
                    lexical_values = []
                    for panel, (metrics, directions) in panel_data.items():
                        canonical = finite(
                            metrics["canonical"][
                                DIRECTION_AXIS_INDEX["all_units"],
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        )
                        raw = finite(
                            metrics["raw"][
                                DIRECTION_AXIS_INDEX["all_units"],
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        )
                        prevalence = finite(
                            metrics["matched_prevalence"][
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        )
                        delta = finite(
                            metrics["matched_delta_median"][
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        )
                        envelope_prevalence = finite(
                            metrics["control_envelope_prevalence"][
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        )
                        envelope_delta = finite(
                            metrics["control_envelope_delta_median"][
                                operation_index,
                                split_index,
                                event_index,
                            ]
                        )
                        if canonical is not None:
                            canonical_values.append(canonical)
                        if raw is not None:
                            raw_values.append(raw)
                        if canonical is not None and raw is not None:
                            gains.append(canonical - raw)
                        if prevalence is not None:
                            matched_prevalences.append(prevalence)
                        if delta is not None:
                            matched_deltas.append(delta)
                        fq_values.append(float(
                            metrics["fq_median"][
                                split_index, event_index
                            ]
                        ))
                        interaction_values.append(float(
                            metrics["interaction_median"][
                                split_index, event_index
                            ]
                        ))
                        lexical_values.append(float(
                            metrics["lexical_control_median"][
                                split_index, event_index
                            ]
                        ))
                        if profile_pass(canonical, raw):
                            directional_panels.append(panel)
                            vector = directions[
                                operation_index,
                                split_index,
                                event_index,
                            ]
                            if vector is not None:
                                panel_vectors.append(vector)
                        if specificity_pass(prevalence, delta):
                            specific_panels.append(panel)
                        if specificity_pass(
                            envelope_prevalence,
                            envelope_delta,
                        ):
                            control_envelope_panels.append(panel)
                    split_rows[split] = {
                        "directional_panel_count": len(
                            directional_panels
                        ),
                        "specificity_panel_count": len(specific_panels),
                        "full_control_envelope_panel_count": len(
                            control_envelope_panels
                        ),
                        "families": sorted({
                            family for family, _ in directional_panels
                        }),
                        "output_modes": sorted({
                            output for _, output in directional_panels
                        }),
                        "mean_canonical_consistency": (
                            finite(np.mean(canonical_values))
                            if canonical_values else None
                        ),
                        "mean_raw_consistency": (
                            finite(np.mean(raw_values))
                            if raw_values else None
                        ),
                        "mean_orientation_gain": (
                            finite(np.mean(gains)) if gains else None
                        ),
                        "mean_matched_control_prevalence": (
                            finite(np.mean(matched_prevalences))
                            if matched_prevalences else None
                        ),
                        "median_matched_control_delta": (
                            finite(np.median(matched_deltas))
                            if matched_deltas else None
                        ),
                        "cross_panel_direction_consistency": (
                            pairwise_consistency(panel_vectors)
                        ),
                        "median_fq_answer_invariant_response": finite(
                            np.median(fq_values)
                        ),
                        "median_factor_interaction_response": finite(
                            np.median(interaction_values)
                        ),
                        "median_lexical_control_response": finite(
                            np.median(lexical_values)
                        ),
                    }

                historical = (
                    model,
                    operation,
                    int(event["depth"]),
                    int(event["head"]),
                ) in history if event.get("head") is not None else False
                profile = {
                    "schema_version": (
                        "phase1014_relative_event_profile.v1"
                    ),
                    "phase": 1014,
                    "model": model,
                    "operation": operation,
                    "event_index": event_index,
                    "event_id": event["event_id"],
                    "component": event["component"],
                    "depth": int(event["depth"]),
                    "relative_depth": float(event["relative_depth"]),
                    "head": event.get("head"),
                    "splits": split_rows,
                    "qualification_axes": axis_rows,
                    "phase1013_coordinate": historical,
                    "recurrent_discovery_candidate": False,
                    "shared_direction_discovery_candidate": False,
                    "control_specific_shared_candidate": False,
                    "claim": (
                        "counterbalanced relative-response recurrence; "
                        "not storage, transport, necessity, or sufficiency"
                    ),
                }
                profile["recurrent_discovery_candidate"] = (
                    recurrent_pass(profile)
                )
                cross_panel = split_rows["discovery"][
                    "cross_panel_direction_consistency"
                ]
                profile["shared_direction_discovery_candidate"] = bool(
                    profile["recurrent_discovery_candidate"]
                    and cross_panel is not None
                    and cross_panel >= PRIMARY_CROSS_PANEL_CONSISTENCY
                )
                profile["control_specific_shared_candidate"] = bool(
                    profile["shared_direction_discovery_candidate"]
                    and split_rows["discovery"][
                        "specificity_panel_count"
                    ] >= PRIMARY_SPECIFICITY_PANELS
                )
                all_event_profiles.append(profile)

        model_profiles = [
            row for row in all_event_profiles if row["model"] == model
        ]
        recurrent = [
            row for row in model_profiles
            if row["recurrent_discovery_candidate"]
        ]
        shared_direction = [
            row for row in recurrent
            if row["shared_direction_discovery_candidate"]
        ]
        control_specific_shared = [
            row for row in shared_direction
            if row["control_specific_shared_candidate"]
        ]
        model_summaries[model] = {
            "formal_scan": bool(model_scan_summary["formal_scan"]),
            "unit_count": int(model_scan_summary["unit_count"]),
            "singleton_forward_count": int(
                model_scan_summary["singleton_forward_count"]
            ),
            "event_count": event_count,
            "identity_maximum": float(
                model_scan_summary["identity_maximum"]
            ),
            "event_operation_profile_count": len(model_profiles),
            "recurrent_candidate_count": len(recurrent),
            "shared_direction_candidate_count": len(
                shared_direction
            ),
            "control_specific_shared_candidate_count": len(
                control_specific_shared
            ),
            "recurrent_confirming_any_panel": int(sum(
                row["splits"]["confirmation"][
                    "directional_panel_count"
                ] > 0
                for row in recurrent
            )),
            "recurrent_confirming_four_panels": int(sum(
                row["splits"]["confirmation"][
                    "directional_panel_count"
                ] >= PRIMARY_RECURRENT_PANELS
                for row in recurrent
            )),
        }

    recurrent_events = [
        row for row in all_event_profiles
        if row["recurrent_discovery_candidate"]
    ]
    recurrent_events.sort(
        key=lambda row: (
            row["model"],
            row["operation"],
            -row["splits"]["confirmation"][
                "directional_panel_count"
            ],
            -row["splits"]["discovery"]["directional_panel_count"],
            row["event_id"],
        )
    )
    shared_direction_events = [
        row for row in recurrent_events
        if row["shared_direction_discovery_candidate"]
    ]
    control_specific_shared_events = [
        row for row in shared_direction_events
        if row["control_specific_shared_candidate"]
    ]

    depth_roles: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in control_specific_shared_events:
        depth_bin = min(9, int(float(row["relative_depth"]) * 10))
        depth_roles[
            (row["operation"], row["component"], depth_bin)
        ].add(row["model"])
    cross_model_role_depth = [
        {
            "operation": operation,
            "component": component,
            "relative_depth_bin": [
                depth_bin / 10,
                (depth_bin + 1) / 10,
            ],
            "models": sorted(models),
            "model_count": len(models),
            "claim": (
                "model-local role/depth recurrence only; physical "
                "coordinates and vector bases are not aligned"
            ),
        }
        for (
            operation,
            component,
            depth_bin,
        ), models in sorted(depth_roles.items())
        if len(models) >= 2
    ]

    sensitivity = []
    for direction_threshold in (0.30, 0.50, 0.70):
        for orientation_gain in (0.10, 0.30, 0.50):
            for required_panels in (2, 4, 6):
                counts_by_model = {}
                for model in models:
                    count = 0
                    model_rows = [
                        row for row in all_event_profiles
                        if row["model"] == model
                    ]
                    for row in model_rows:
                        local_count = 0
                        for family in FAMILIES:
                            for output_mode in OUTPUT_MODES:
                                metrics = sensitivity_cache[
                                    (model, family, output_mode)
                                ]
                                event_index = int(row["event_index"])
                                operation_index = TARGET_INDEX[
                                    row["operation"]
                                ]
                                canonical = finite(
                                    metrics["canonical"][
                                        DIRECTION_AXIS_INDEX[
                                            "all_units"
                                        ],
                                        operation_index,
                                        SPLIT_INDEX["discovery"],
                                        event_index,
                                    ]
                                )
                                raw = finite(
                                    metrics["raw"][
                                        DIRECTION_AXIS_INDEX[
                                            "all_units"
                                        ],
                                        operation_index,
                                        SPLIT_INDEX["discovery"],
                                        event_index,
                                    ]
                                )
                                if profile_pass(
                                    canonical,
                                    raw,
                                    direction_threshold=(
                                        direction_threshold
                                    ),
                                    orientation_gain=orientation_gain,
                                ):
                                    local_count += 1
                        if local_count >= required_panels:
                            count += 1
                    counts_by_model[model] = count
                sensitivity.append({
                    "scope": (
                        "local direction recurrence before "
                        "family/output/control/cross-panel gates"
                    ),
                    "direction_threshold": direction_threshold,
                    "orientation_gain": orientation_gain,
                    "required_panel_count": required_panels,
                    "counts_by_model": counts_by_model,
                })

    component_summary = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in recurrent_events:
        grouped[
            (row["model"], row["operation"], row["component"])
        ].append(row)
    for key, rows in sorted(grouped.items()):
        model, operation, component = key
        component_summary.append({
            "model": model,
            "operation": operation,
            "component": component,
            "candidate_count": len(rows),
            "shared_direction_candidate_count": int(sum(
                row["shared_direction_discovery_candidate"]
                for row in rows
            )),
            "control_specific_shared_candidate_count": int(sum(
                row["control_specific_shared_candidate"]
                for row in rows
            )),
            "depths": sorted({int(row["depth"]) for row in rows}),
            "maximum_discovery_panel_count": max(
                row["splits"]["discovery"][
                    "directional_panel_count"
                ]
                for row in rows
            ),
            "maximum_confirmation_panel_count": max(
                row["splits"]["confirmation"][
                    "directional_panel_count"
                ]
                for row in rows
            ),
        })

    analysis_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        analysis_root / "event_profiles.jsonl",
        all_event_profiles,
    )
    write_jsonl(
        analysis_root / "recurrent_events.jsonl",
        recurrent_events,
    )
    write_jsonl(
        analysis_root / "shared_direction_events.jsonl",
        shared_direction_events,
    )
    write_jsonl(
        analysis_root / "control_specific_shared_events.jsonl",
        control_specific_shared_events,
    )
    write_json(
        analysis_root / "cross_model_role_depth.json",
        cross_model_role_depth,
    )
    summary = {
        "schema_version": (
            "phase1014_relative_difference_analysis.v1"
        ),
        "phase": 1014,
        "scan_namespace": scan_namespace,
        "protocol_digest": read_json(
            OUT_ROOT / "protocol" / "protocol.json"
        )["preregistration_digest"],
        "models": model_summaries,
        "recurrent_event_count": len(recurrent_events),
        "shared_direction_event_count": len(
            shared_direction_events
        ),
        "control_specific_shared_event_count": len(
            control_specific_shared_events
        ),
        "component_summary": component_summary,
        "cross_model_role_depth_recurrence": cross_model_role_depth,
        "threshold_sensitivity": sensitivity,
        "operational_candidate_thresholds_not_theory": {
            "canonical_direction_consistency": (
                PRIMARY_DIRECTION_THRESHOLD
            ),
            "canonical_minus_raw_orientation_gain": (
                PRIMARY_ORIENTATION_GAIN
            ),
            "matched_control_prevalence": (
                PRIMARY_CONTROL_PREVALENCE
            ),
            "recurrent_panel_count": PRIMARY_RECURRENT_PANELS,
            "cross_panel_direction_consistency": (
                PRIMARY_CROSS_PANEL_CONSISTENCY
            ),
            "specificity_panel_count": PRIMARY_SPECIFICITY_PANELS,
            "minimum_family_count": 2,
            "minimum_output_mode_count": 2,
        },
        "selection_contract": {
            "discovery_selects": True,
            "confirmation_never_selects": True,
            "all_units_axis_is_primary": True,
            "behavior_qualified_axes_are_supplemental": True,
            "phase1013_coordinates_are_post_hoc_only": True,
            "weighted_mechanism_score_used": False,
        },
        "measurement_limits": [
            "canonical direction recurrence is not a decoded variable",
            "response magnitude is not causal contribution",
            "same physical coordinate is not a transport edge",
            "relative-depth overlap is not cross-model isomorphism",
            "8-bit observations require independent precision audit",
        ],
    }
    write_json(analysis_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-namespace", default="scan")
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="comma-separated model subset for instrumentation smoke",
    )
    args = parser.parse_args()
    models = tuple(
        value.strip()
        for value in args.models.split(",")
        if value.strip()
    )
    invalid = sorted(set(models) - set(MODELS))
    if invalid:
        raise SystemExit(f"unknown models: {invalid}")
    finalize(args.scan_namespace, models)


if __name__ == "__main__":
    main()
