#!/usr/bin/env python3
"""Finalize Phase 1001 source-path discovery after the full causal scan.

The exhaustive scan and joint prefixes are already persisted. The original
95%-of-best engineering rule selects all 132 routes and is therefore retained
only as a high-fidelity upper bound. This script uses the thresholds declared
before the scan to test the smallest absolute-sufficient prefixes in natural
generation, freezes the first prefix that also passes the natural gate, and
then runs its wrong-O and cross-pair controls.
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

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1000_factorial_binding_protocol import COLORS, MODEL
from phase1000_scpg_discovery import (
    batches_by_template,
    capture_residuals,
    read_jsonl,
    source_patch_spec,
    write_rows,
)
from phase1001_attention_head_discovery import (
    RESULT_ROOT,
    SOURCE_DEPTH,
    generate_with_patches,
    read_json,
    selected_phase1000_inputs,
    write_json,
)
from phase1001_attention_source_path_decomposition import (
    HEAD_DISCOVERY_ROOT,
    JOINT_ROUTE_SIZES,
    PATH_THRESHOLDS,
    ROLE_NAMES,
    SOURCE_ROOT,
    capture_physical_attention,
    combined_head_patches,
    component_map_for_batch,
    control_rows_for_batch,
    event_from_id,
    make_routes,
    rank_routes,
    summarize_controls,
)


DISCOVERY_ROOT = SOURCE_ROOT / "discovery"


def high_fidelity_size(joint_summary: dict[str, dict[str, Any]]) -> int:
    best = max(
        min(
            item["median_mediation_fraction"],
            item["mean_sufficiency_transfer"],
        )
        for item in joint_summary.values()
    )
    threshold = 0.95 * best
    return min(
        int(size)
        for size, item in joint_summary.items()
        if min(
            item["median_mediation_fraction"],
            item["mean_sufficiency_transfer"],
        )
        >= threshold
    )


def candidate_sizes(
    joint_summary: dict[str, dict[str, Any]],
) -> list[int]:
    return sorted(
        int(size)
        for size, item in joint_summary.items()
        if item["median_mediation_fraction"]
        >= PATH_THRESHOLDS["joint_median_mediation"]
        and item["mean_sufficiency_transfer"]
        >= PATH_THRESHOLDS["joint_mean_sufficiency"]
    )


def natural_selection_rows_for_batch(
    model,
    layers,
    tokenizer,
    device,
    batch: list[dict[str, Any]],
    ranked_routes: list[dict[str, Any]],
    sizes: list[int],
    components,
    source_patch,
    target_heads,
    do_heads,
    effective_eos,
    budget: int,
) -> list[dict[str, Any]]:
    target_cases = [item["target"] for item in batch]
    conditions: list[tuple[str, int | None, list[dict[str, Any]]]] = [
        ("source_do", None, [])
    ]
    for size in sizes:
        conditions.append(
            (
                "source_plus_route_restore",
                size,
                combined_head_patches(
                    ranked_routes,
                    size,
                    components,
                    target_heads,
                    do_heads,
                    "restore",
                ),
            )
        )
    full_head_restore = []
    seen = set()
    for route in ranked_routes:
        event_id = route["event_id"]
        if event_id in seen:
            continue
        seen.add(event_id)
        event = event_from_id(event_id)
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        full_head_restore.append(
            {
                "event": event,
                "vectors": target_heads[layer_number][
                    :, head_index, :
                ],
            }
        )
    conditions.append(
        ("source_plus_full_frozen_head_restore", None, full_head_restore)
    )

    rows = []
    for condition, size, patches in conditions:
        generated = generate_with_patches(
            model,
            layers,
            tokenizer,
            device,
            target_cases,
            source_patch,
            patches,
            {},
            effective_eos,
            budget,
        )
        for index, item in enumerate(batch):
            result = generated[index]
            rows.append(
                {
                    "schema_version": (
                        "phase1001_source_path_natural_selection.v1"
                    ),
                    "phase": 1001,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "condition": condition,
                    "joint_size": size,
                    "prediction": result["prediction"],
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "flipped_to_source": (
                        result["prediction"] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        result["prediction"] == item["target"]["gold"]
                    ),
                    "eos_seen": result["eos_seen"],
                    "exact_short": result["exact_short"],
                    "generated_text": result["text"],
                }
            )
    return rows


def summarize_natural_selection(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        groups[(row["condition"], row["joint_size"])].append(row)
    result = {}
    for (condition, size), values in groups.items():
        key = condition if size is None else f"{condition}/k{size}"
        result[key] = {
            "condition": condition,
            "joint_size": size,
            "n": len(values),
            "flip_rate": float(
                np.mean([row["flipped_to_source"] for row in values])
            ),
            "target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "eos_rate": float(
                np.mean([row["eos_seen"] for row in values])
            ),
            "exact_short_rate": float(
                np.mean([row["exact_short"] for row in values])
            ),
        }
    return result


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 source finalize requires CUDA")
    observation_summary = read_json(
        DISCOVERY_ROOT / "observation_summary.json"
    )
    causal_summary = read_json(DISCOVERY_ROOT / "causal_summary.json")
    joint_summary = read_json(DISCOVERY_ROOT / "joint_summary.json")
    audit_rows = read_jsonl(DISCOVERY_ROOT / "instrument_audit_rows.jsonl")
    ranked_metrics = rank_routes(observation_summary, causal_summary)

    frozen_heads = read_json(HEAD_DISCOVERY_ROOT / "frozen_spec.json")
    events = [
        event_from_id(event_id)
        for event_id in frozen_heads["frozen_joint_event_ids"]
    ]
    route_lookup = {
        route["route_id"]: route for route in make_routes(events)
    }
    ranked_routes = [
        {**route_lookup[item["route_id"]], **item}
        for item in ranked_metrics
    ]
    sizes = candidate_sizes(joint_summary)
    if not sizes:
        raise RuntimeError("no source-route prefix meets absolute gates")
    fidelity_size = high_fidelity_size(joint_summary)

    protocol, _, selected_pairs, directional, _ = (
        selected_phase1000_inputs("formal")
    )
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        effective_eos = eos_ids(model, tokenizer)

        natural_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [item["source"] for item in batch]
            target_cases = [item["target"] for item in batch]
            _, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                _,
                target_values,
                target_weights,
                target_heads,
            ) = capture_physical_attention(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
            )
            _, do_values, do_weights, do_heads = (
                capture_physical_attention(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
            )
            components, _ = component_map_for_batch(
                batch,
                events,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
            )
            natural_rows.extend(
                natural_selection_rows_for_batch(
                    model,
                    layers,
                    tokenizer,
                    device,
                    batch,
                    ranked_routes,
                    sizes,
                    components,
                    source_patch,
                    target_heads,
                    do_heads,
                    effective_eos,
                    natural_budget,
                )
            )
            del (
                source_residuals,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
                components,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[source-natural-selection] "
                    f"{batch_number}/{len(batches)} batches",
                    flush=True,
                )
        natural_summary = summarize_natural_selection(natural_rows)
        write_rows(
            DISCOVERY_ROOT / "natural_selection_rows.jsonl",
            natural_rows,
        )
        write_json(
            DISCOVERY_ROOT / "natural_selection_summary.json",
            natural_summary,
        )
        passing_sizes = [
            size
            for size in sizes
            if natural_summary[
                f"source_plus_route_restore/k{size}"
            ]["target_rate"]
            >= PATH_THRESHOLDS["joint_natural_restoration"]
        ]
        if not passing_sizes:
            frozen_size = sizes[0]
            natural_gate = False
        else:
            frozen_size = min(passing_sizes)
            natural_gate = True

        selected_routes = ranked_routes[:frozen_size]
        control_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [item["source"] for item in batch]
            target_cases = [item["target"] for item in batch]
            source_logits, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                target_logits,
                target_values,
                target_weights,
                target_heads,
            ) = capture_physical_attention(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
            )
            _, do_values, do_weights, do_heads = (
                capture_physical_attention(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
            )
            components, _ = component_map_for_batch(
                batch,
                events,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
            )
            control_rows.extend(
                control_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    selected_routes,
                    components,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    target_heads,
                )
            )
            del (
                source_logits,
                source_residuals,
                target_logits,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
                components,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[source-frozen-controls] "
                    f"{batch_number}/{len(batches)} batches",
                    flush=True,
                )
        control_summary = summarize_controls(control_rows)
        write_rows(DISCOVERY_ROOT / "control_rows.jsonl", control_rows)
        write_json(
            DISCOVERY_ROOT / "control_summary.json", control_summary
        )

        stable_single_routes = []
        for route in selected_routes:
            route_key = route["route_id"]
            causal = causal_summary[route_key]
            control = control_summary[route_key]
            if (
                causal["median_mediation_fraction"]
                >= PATH_THRESHOLDS["single_median_mediation"]
                and causal["mean_sufficiency_transfer"]
                >= PATH_THRESHOLDS["single_mean_sufficiency"]
                and (
                    causal["mean_sufficiency_transfer"]
                    - control["mean_wrong_o_transfer"]
                )
                >= PATH_THRESHOLDS["single_location_excess"]
                and (
                    causal["mean_sufficiency_transfer"]
                    - control["mean_cross_pair_null_transfer"]
                )
                >= PATH_THRESHOLDS["single_location_excess"]
            ):
                stable_single_routes.append(route_key)

        audit_metrics = {
            "max_target_head_reconstruction_error": max(
                row["target_max_abs_head_reconstruction_error"]
                for row in audit_rows
            ),
            "max_do_head_reconstruction_error": max(
                row["do_max_abs_head_reconstruction_error"]
                for row in audit_rows
            ),
            "max_role_delta_reconstruction_error": max(
                row["max_abs_role_delta_reconstruction_error"]
                for row in audit_rows
            ),
            "max_qkv_identity_error": max(
                row["max_abs_qkv_identity_error"] for row in audit_rows
            ),
        }
        frozen_joint = joint_summary[str(frozen_size)]
        gate_checks = {
            "head_reconstruction": (
                max(
                    audit_metrics[
                        "max_target_head_reconstruction_error"
                    ],
                    audit_metrics["max_do_head_reconstruction_error"],
                )
                <= PATH_THRESHOLDS["max_head_reconstruction_error"]
            ),
            "role_partition_reconstruction": (
                audit_metrics["max_role_delta_reconstruction_error"]
                <= PATH_THRESHOLDS[
                    "max_role_delta_reconstruction_error"
                ]
            ),
            "qkv_algebra_identity": (
                audit_metrics["max_qkv_identity_error"]
                <= PATH_THRESHOLDS["max_qkv_identity_error"]
            ),
            "stable_single_route": bool(stable_single_routes),
            "frozen_joint_mediation": (
                frozen_joint["median_mediation_fraction"]
                >= PATH_THRESHOLDS["joint_median_mediation"]
            ),
            "frozen_joint_sufficiency": (
                frozen_joint["mean_sufficiency_transfer"]
                >= PATH_THRESHOLDS["joint_mean_sufficiency"]
            ),
            "source_do_natural": (
                natural_summary["source_do"]["flip_rate"]
                >= PATH_THRESHOLDS["source_do_natural_flip"]
            ),
            "frozen_joint_natural": natural_gate,
        }
        frozen_spec = {
            "schema_version": "phase1001_frozen_source_path_spec.v1",
            "phase": 1001,
            "model": MODEL,
            "ranked_route_ids": [
                route["route_id"] for route in ranked_routes
            ],
            "frozen_joint_size": frozen_size,
            "frozen_joint_route_ids": [
                route["route_id"] for route in selected_routes
            ],
            "high_fidelity_95pct_size": fidelity_size,
            "candidate_sizes_before_natural": sizes,
            "selection_partition": "validation",
            "selection_uses_holdout": False,
            "frozen_before_holdout": True,
        }
        summary = {
            "schema_version": (
                "phase1001_source_path_discovery_summary.v1"
            ),
            "phase": 1001,
            "model": MODEL,
            "stage": "discovery",
            "partition": "validation",
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "source_depth": SOURCE_DEPTH,
            "frozen_head_event_ids": [
                event["event_id"] for event in events
            ],
            "source_roles": list(ROLE_NAMES),
            "route_count_observed": len(observation_summary),
            "route_count_causally_tested": len(causal_summary),
            "ranked_routes": ranked_metrics,
            "joint_summary": joint_summary,
            "candidate_sizes_before_natural": sizes,
            "high_fidelity_95pct_size": fidelity_size,
            "high_fidelity_interpretation": (
                "all-route algebraic upper bound, not compact localization"
            ),
            "frozen_joint_size": frozen_size,
            "frozen_joint_route_ids": frozen_spec[
                "frozen_joint_route_ids"
            ],
            "natural_selection_summary": natural_summary,
            "control_summary": control_summary,
            "stable_single_routes": stable_single_routes,
            "audit_metrics": audit_metrics,
            "thresholds": PATH_THRESHOLDS,
            "gate_checks": gate_checks,
            "source_path_gate_pass": all(gate_checks.values()),
            "qkv_causal_decomposition_open": all(gate_checks.values()),
            "selection_uses_current_partition": True,
            "selection_uses_holdout": False,
            "method_audit": {
                "initial_exhaustive_scan_completed": True,
                "initial_process_interrupted_after_joint_files": True,
                "causal_or_joint_rows_recomputed": False,
                "original_95pct_rule_retained_as_upper_bound": True,
                "operational_freeze_uses_predeclared_absolute_and_natural_gates": True,
            },
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds_finalize": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(DISCOVERY_ROOT / "frozen_spec.json", frozen_spec)
        write_json(DISCOVERY_ROOT / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["source_path_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "candidate_sizes": summary[
                    "candidate_sizes_before_natural"
                ],
                "high_fidelity_95pct_size": summary[
                    "high_fidelity_95pct_size"
                ],
                "frozen_joint_size": summary["frozen_joint_size"],
                "stable_single_routes": summary[
                    "stable_single_routes"
                ],
                "natural": summary["natural_selection_summary"],
                "elapsed_seconds": summary[
                    "elapsed_seconds_finalize"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
