#!/usr/bin/env python3
"""Measure Phase1010 response fields without assigning mechanism labels.

The scanner retains per-unit scalar responses and qualified aggregate
directions. Raw hidden states and component tensors are never persisted.
Comparisons are made between within-output operation responses, not by
subtracting prompts that request different output vocabularies.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1006_blind_source_and_behavior import eos_token_ids
from phase1008_global_response_atlas_scan import (
    StateCapture,
    direction_consistency,
)
from phase1009_crossfamily_response_scan import (
    EPSILON,
    OP_INDEX,
    SPLIT_INDEX,
    STATE_ORDER,
    capture_stage,
    operation_deltas,
    operation_scales,
    output_metrics,
    stage_case,
    unit_qualification,
)
from phase1010_output_type_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    NATURAL_STATES,
    OUT_ROOT,
    OUTPUT_TYPES,
    PAIR_OPERATIONS,
    PHASE,
    ROLE_CLASSES,
    TIME_STAGES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def event_definitions(
    family: str,
    output_type: str,
    n_layers: int,
) -> tuple[list[dict[str, Any]], dict[tuple, int]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple, int] = {}
    for stage in TIME_STAGES:
        roles = (
            tuple(ROLE_CLASSES[family])
            if stage == "prompt"
            else ("decision_boundary",)
        )
        component_depths = (
            ("residual", range(0, n_layers + 1)),
            ("attention_output", range(1, n_layers + 1)),
            ("mlp_output", range(1, n_layers + 1)),
        )
        for component, depths in component_depths:
            for depth in depths:
                for role in roles:
                    key = (stage, component, int(depth), role)
                    event_index = len(rows)
                    lookup[key] = event_index
                    rows.append({
                        "schema_version": "phase1010_event.v1",
                        "phase": PHASE,
                        "family": family,
                        "output_type": output_type,
                        "event_index": event_index,
                        "event_id": (
                            f"{output_type}.{family}.{stage}.{component}."
                            f"d{int(depth):02d}.{role}"
                        ),
                        "stage": stage,
                        "component": component,
                        "depth": int(depth),
                        "relative_depth": float(depth / max(n_layers, 1)),
                        "role": role,
                        "role_class": (
                            ROLE_CLASSES[family][role]
                            if stage == "prompt"
                            else "decision_boundary"
                        ),
                        "edge_claim_allowed_from_scan": "co_response_only",
                    })
    return rows, lookup


def peak_direction_centroids(
    *,
    events: list[dict[str, Any]],
    normalized_magnitude: np.ndarray,
    semantic_qualified: np.ndarray,
    direction_sum: np.ndarray,
    direction_count: np.ndarray,
    model_name: str,
    family: str,
    output_type: str,
    d_model: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Keep only aggregate directions at data-selected trajectory peaks."""
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for event in events:
        groups[
            (event["stage"], event["component"], event["role"])
        ].append(int(event["event_index"]))
    for indices in groups.values():
        indices.sort(key=lambda index: int(events[index]["depth"]))

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for operation in ANALYSIS_OPERATIONS:
        operation_index = OP_INDEX[operation]
        qualified = semantic_qualified[:, operation_index]
        if int(np.sum(qualified)) == 0:
            continue
        for (stage, component, role), indices in sorted(groups.items()):
            trajectory = normalized_magnitude[
                :, operation_index, indices
            ]
            profile = np.nanmedian(trajectory[qualified], axis=0)
            if not np.any(np.isfinite(profile)):
                continue
            peak_offset = int(np.nanargmax(profile))
            peak_event_index = int(indices[peak_offset])
            peak_event = events[peak_event_index]
            for split, split_index in SPLIT_INDEX.items():
                count = int(direction_count[
                    operation_index, split_index, peak_event_index
                ])
                if count < 2:
                    continue
                mean_direction = (
                    direction_sum[
                        operation_index, split_index, peak_event_index
                    ].astype(np.float64, copy=False)
                    / count
                )
                concentration = float(np.linalg.norm(mean_direction))
                if not np.isfinite(concentration) or concentration <= EPSILON:
                    continue
                centroid = (mean_direction / concentration).astype(
                    np.float16
                )
                centroid_index = len(vectors)
                vectors.append(centroid)
                rows.append({
                    "schema_version": "phase1010_peak_direction.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "family": family,
                    "output_type": output_type,
                    "operation": operation,
                    "split": split,
                    "stage": stage,
                    "component": component,
                    "role": role,
                    "role_class": peak_event["role_class"],
                    "peak_event_index": peak_event_index,
                    "peak_depth": int(peak_event["depth"]),
                    "peak_relative_depth": float(
                        peak_event["relative_depth"]
                    ),
                    "peak_normalized_magnitude": float(
                        profile[peak_offset]
                    ),
                    "direction_count": count,
                    "direction_concentration": concentration,
                    "centroid_index": centroid_index,
                    "interpretation_limit": (
                        "aggregate direction at a response peak; similarity "
                        "does not establish transport or causal identity"
                    ),
                })
    if vectors:
        array = np.stack(vectors)
    else:
        array = np.empty((0, d_model), dtype=np.float16)
    return rows, array


def scan_panel(
    *,
    model,
    info,
    device,
    capture: StateCapture,
    effective_eos: set[int],
    model_name: str,
    family: str,
    output_type: str,
    units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    qualification_by_key: dict[tuple[str, str], dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    events, event_lookup = event_definitions(
        family,
        output_type,
        int(info.n_layers),
    )
    unit_count = len(units)
    operation_count = len(ANALYSIS_OPERATIONS)
    event_count = len(events)
    raw_magnitude = np.full(
        (unit_count, operation_count, event_count),
        np.nan,
        dtype=np.float32,
    )
    normalized_magnitude = np.full_like(raw_magnitude, np.nan)
    semantic_qualified = np.zeros(
        (unit_count, operation_count),
        dtype=np.bool_,
    )
    strict_qualified = np.zeros_like(semantic_qualified)
    rollout_qualified = np.zeros_like(semantic_qualified)
    direction_sum = np.zeros(
        (
            operation_count,
            len(SPLIT_INDEX),
            event_count,
            int(info.d_model),
        ),
        dtype=np.float32,
    )
    direction_count = np.zeros(
        (operation_count, len(SPLIT_INDEX), event_count),
        dtype=np.int32,
    )
    unit_rows: list[dict[str, Any]] = []
    all_output_measurements: dict[
        str, dict[str, dict[str, Any]]
    ] = {}
    started = time.time()

    for unit_index, unit in enumerate(units):
        unit_semantic, unit_strict, unit_rollout = unit_qualification(
            unit,
            qualification_by_key,
        )
        semantic_qualified[unit_index] = unit_semantic
        strict_qualified[unit_index] = unit_strict
        rollout_qualified[unit_index] = unit_rollout
        base = case_by_id[unit["case_ids"]["base"]]
        state_cases = [
            case_by_id[unit["case_ids"][state]]
            for state in NATURAL_STATES
        ] + [dict(base)]
        split_index = SPLIT_INDEX[unit["split"]]
        unit_rows.append({
            "schema_version": "phase1010_scan_unit.v1",
            "phase": PHASE,
            "model": model_name,
            "family": family,
            "output_type": output_type,
            "unit_index": unit_index,
            "unit_id": unit["unit_id"],
            "source_unit_id": unit["source_unit_id"],
            "split": unit["split"],
            "template": int(unit["template"]),
            "name_pool": int(unit["name_pool"]),
            "world_index": int(unit["world_index"]),
            "semantic_qualified": {
                operation: bool(unit_semantic[OP_INDEX[operation]])
                for operation in ANALYSIS_OPERATIONS
            },
            "strict_qualified": {
                operation: bool(unit_strict[OP_INDEX[operation]])
                for operation in ANALYSIS_OPERATIONS
            },
            "rollout_qualified": {
                operation: bool(unit_rollout[OP_INDEX[operation]])
                for operation in ANALYSIS_OPERATIONS
            },
        })
        all_output_measurements[unit["unit_id"]] = {
            state: {} for state in STATE_ORDER
        }
        for stage in TIME_STAGES:
            staged = [stage_case(case, stage) for case in state_cases]
            role_names = (
                list(ROLE_CLASSES[family])
                if stage == "prompt"
                else ["decision_boundary"]
            )
            captured, logits = capture_stage(
                model=model,
                capture=capture,
                device=device,
                staged=staged,
                role_names=role_names,
            )
            stage_outputs = output_metrics(
                logits,
                staged,
                stage,
                effective_eos,
            )
            for state_index, state in enumerate(STATE_ORDER):
                all_output_measurements[
                    unit["unit_id"]
                ][state][stage] = stage_outputs[state_index]
            for (component, depth), values in captured.items():
                deltas = operation_deltas(values)
                scales = operation_scales(values)
                for role_index, role in enumerate(role_names):
                    event_index = event_lookup[
                        (stage, component, int(depth), role)
                    ]
                    for operation in ANALYSIS_OPERATIONS:
                        operation_index = OP_INDEX[operation]
                        delta = deltas[operation][role_index].float()
                        raw = torch.linalg.vector_norm(delta)
                        scale = scales[operation][role_index].float()
                        normalized = raw / torch.clamp(
                            scale,
                            min=EPSILON,
                        )
                        raw_value = float(raw.item())
                        raw_magnitude[
                            unit_index, operation_index, event_index
                        ] = raw_value
                        normalized_magnitude[
                            unit_index, operation_index, event_index
                        ] = float(normalized.item())
                        if (
                            raw_value > EPSILON
                            and unit_semantic[operation_index]
                        ):
                            direction = (
                                delta / torch.clamp(raw, min=EPSILON)
                            ).numpy()
                            direction_sum[
                                operation_index,
                                split_index,
                                event_index,
                            ] += direction.astype(
                                np.float32,
                                copy=False,
                            )
                            direction_count[
                                operation_index,
                                split_index,
                                event_index,
                            ] += 1
            del captured, logits
        if (unit_index + 1) % 4 == 0 or unit_index + 1 == unit_count:
            print(
                f"[scan] {model_name}/{family}/{output_type} "
                f"{unit_index + 1}/{unit_count} units",
                flush=True,
            )

    output_rows: list[dict[str, Any]] = []
    for unit in units:
        measurements = all_output_measurements[unit["unit_id"]]
        base = measurements["base"]
        for operation in PAIR_OPERATIONS:
            variant_state = "base" if operation == "I" else operation
            variant = measurements[variant_state]
            base_probability = base["semantic0"][
                "fixed_base_probability"
            ]
            variant_probability = variant["semantic0"][
                "fixed_base_probability"
            ]
            output_rows.append({
                "schema_version": "phase1010_output_pair.v1",
                "phase": PHASE,
                "model": model_name,
                "family": family,
                "output_type": output_type,
                "unit_id": unit["unit_id"],
                "split": unit["split"],
                "template": int(unit["template"]),
                "name_pool": int(unit["name_pool"]),
                "world_index": int(unit["world_index"]),
                "operation": operation,
                "expected_output_relation": (
                    "changes" if operation in ("F", "Q")
                    else "same_as_base"
                ),
                "base_fixed_choice_margin": base["semantic0"][
                    "fixed_base_margin"
                ],
                "variant_fixed_choice_margin": variant["semantic0"][
                    "fixed_base_margin"
                ],
                "delta_fixed_choice_margin": (
                    variant["semantic0"]["fixed_base_margin"]
                    - base["semantic0"]["fixed_base_margin"]
                ),
                "fixed_panel_probability_l1": float(
                    2.0 * abs(variant_probability - base_probability)
                ),
                "base_correct_margin": base["semantic0"][
                    "correct_margin"
                ],
                "variant_correct_margin": variant["semantic0"][
                    "correct_margin"
                ],
                "base_done_vs_eos_margin": base["function0"][
                    "done_vs_eos_margin"
                ],
                "variant_done_vs_eos_margin": variant["function0"][
                    "done_vs_eos_margin"
                ],
                "base_eos_margin": base["termination"]["eos_margin"],
                "variant_eos_margin": variant["termination"][
                    "eos_margin"
                ],
            })

    panel_root = output_root / output_type / family
    panel_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_root / "response_scalars.npz",
        raw_magnitude=raw_magnitude,
        normalized_magnitude=normalized_magnitude,
        semantic_qualified=semantic_qualified,
        strict_qualified=strict_qualified,
        rollout_qualified=rollout_qualified,
    )
    consistency = direction_consistency(
        direction_sum,
        direction_count,
    )
    np.savez_compressed(
        panel_root / "direction_consistency.npz",
        direction_consistency=consistency,
        direction_count=direction_count,
    )
    centroid_rows, centroids = peak_direction_centroids(
        events=events,
        normalized_magnitude=normalized_magnitude,
        semantic_qualified=semantic_qualified,
        direction_sum=direction_sum,
        direction_count=direction_count,
        model_name=model_name,
        family=family,
        output_type=output_type,
        d_model=int(info.d_model),
    )
    np.savez_compressed(
        panel_root / "peak_direction_centroids.npz",
        centroids=centroids,
    )
    write_jsonl(panel_root / "peak_direction_metadata.jsonl", centroid_rows)
    write_jsonl(panel_root / "events.jsonl", events)
    write_jsonl(panel_root / "units.jsonl", unit_rows)
    write_jsonl(panel_root / "output_pairs.jsonl", output_rows)
    identity = normalized_magnitude[:, OP_INDEX["I"], :]
    summary = {
        "schema_version": "phase1010_panel_scan_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "family": family,
        "output_type": output_type,
        "unit_count": unit_count,
        "event_count": event_count,
        "operation_count": operation_count,
        "scalar_measurement_count": int(
            unit_count * operation_count * event_count
        ),
        "peak_direction_centroid_count": len(centroid_rows),
        "raw_hidden_tensors_persisted": 0,
        "semantic_qualified_pair_counts": {
            operation: int(np.sum(
                semantic_qualified[:, OP_INDEX[operation]]
            ))
            for operation in ANALYSIS_OPERATIONS
        },
        "strict_qualified_pair_counts": {
            operation: int(np.sum(
                strict_qualified[:, OP_INDEX[operation]]
            ))
            for operation in ANALYSIS_OPERATIONS
        },
        "rollout_qualified_pair_counts": {
            operation: int(np.sum(
                rollout_qualified[:, OP_INDEX[operation]]
            ))
            for operation in ANALYSIS_OPERATIONS
        },
        "identity_normalized_floor": {
            "maximum": float(np.nanmax(identity)),
            "mean": float(np.nanmean(identity)),
            "nonzero_count": int(np.sum(identity > EPSILON)),
        },
        "direction_policy": "semantic-qualified pairs only",
        "edge_claim_allowed": "co_response_only",
        "elapsed_seconds": time.time() - started,
    }
    write_json(panel_root / "summary.json", summary)
    del direction_sum, direction_count, consistency, centroids
    return summary


def run_model(
    model_name: str,
    *,
    scope: str,
    limit_units_per_panel: int | None,
) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    behavior = read_json(
        OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    if behavior["protocol_digest"] != protocol["preregistration_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "units.jsonl"
    )
    case_by_id = {case["record_id"]: case for case in cases}
    qualifications = read_jsonl(
        OUT_ROOT / "behavior" / model_name / "pair_qualification.jsonl"
    )
    qualification_by_key = {
        (row["unit_id"], row["operation"]): row
        for row in qualifications
    }
    output_root = (
        OUT_ROOT
        / ("scan" if scope == "formal" else "scan_smoke")
        / model_name
    )
    started = time.time()
    model = tokenizer = device = capture = None
    summaries: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        effective_eos = eos_token_ids(model, tokenizer, model_name)
        capture = StateCapture(model, layers)
        capture.register()
        for output_type in OUTPUT_TYPES:
            for family in FAMILIES:
                panel_units = [
                    unit
                    for unit in units
                    if unit["family"] == family
                    and unit["output_type"] == output_type
                ]
                if limit_units_per_panel is not None:
                    panel_units = panel_units[:limit_units_per_panel]
                summaries.append(scan_panel(
                    model=model,
                    info=info,
                    device=device,
                    capture=capture,
                    effective_eos=effective_eos,
                    model_name=model_name,
                    family=family,
                    output_type=output_type,
                    units=panel_units,
                    case_by_id=case_by_id,
                    qualification_by_key=qualification_by_key,
                    output_root=output_root,
                ))
        summary = {
            "schema_version": "phase1010_scan_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "scope": scope,
            "protocol_digest": protocol["preregistration_digest"],
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "model_class": info.model_class,
                "loaded_8bit": True,
            },
            "panel_summaries": summaries,
            "unit_count": int(sum(row["unit_count"] for row in summaries)),
            "event_count_sum": int(
                sum(row["event_count"] for row in summaries)
            ),
            "scalar_measurement_count": int(sum(
                row["scalar_measurement_count"] for row in summaries
            )),
            "peak_direction_centroid_count": int(sum(
                row["peak_direction_centroid_count"] for row in summaries
            )),
            "raw_hidden_tensors_persisted": 0,
            "elapsed_seconds": time.time() - started,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--scope",
        choices=("smoke", "formal"),
        default="formal",
    )
    parser.add_argument("--limit-units-per-panel", type=int)
    args = parser.parse_args()
    limit = args.limit_units_per_panel
    if args.scope == "smoke" and limit is None:
        limit = 1
    run_model(
        args.model,
        scope=args.scope,
        limit_units_per_panel=limit,
    )


if __name__ == "__main__":
    main()
