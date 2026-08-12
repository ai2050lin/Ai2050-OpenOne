#!/usr/bin/env python3
"""Phase 1001 causal QK/V/interaction decomposition of frozen source paths.

For each frozen source-role -> attention-head path, the exact change is split
by the algebraic identity

    a_do v_do - a_t v_t
      = (a_do-a_t) v_t
      + a_t (v_do-v_t)
      + (a_do-a_t)(v_do-v_t).

The three terms are named QK routing, V content, and interaction. They are
patched at the real 128-dimensional head output before o_proj. The identity
is a measurement instrument, not a proposed language theory.
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
    prediction_colors,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1001_attention_head_discovery import (
    SOURCE_DEPTH,
    forward_with_patches,
    generate_with_patches,
    read_json,
    write_json,
)
from phase1001_attention_source_path_decomposition import (
    HEAD_DISCOVERY_ROOT,
    PATH_THRESHOLDS,
    SOURCE_ROOT,
    capture_physical_attention,
    component_map_for_batch,
    event_from_id,
    make_routes,
    selected_inputs,
)


QKV_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "qkv_causal_decomposition"
)
BASE_COMPONENTS = ("qk", "value", "interaction")
INDIVIDUAL_COMPONENTS = ("qk", "value", "interaction", "all")
GROUP_COMPONENTS = (
    "qk",
    "value",
    "interaction",
    "qk_plus_value",
    "qk_plus_interaction",
    "value_plus_interaction",
    "all",
)
QKV_THRESHOLDS = {
    "max_identity_error": 1e-4,
    "source_do_natural_flip": 0.90,
    "all_group_median_mediation": 0.30,
    "all_group_mean_sufficiency": 0.30,
    "all_group_natural_restoration": 0.50,
}


def component_keys(name: str) -> tuple[str, ...]:
    mapping = {
        "qk": ("qk",),
        "value": ("value",),
        "interaction": ("interaction",),
        "qk_plus_value": ("qk", "value"),
        "qk_plus_interaction": ("qk", "interaction"),
        "value_plus_interaction": ("value", "interaction"),
        "all": ("qk", "value", "interaction"),
    }
    return mapping[name]


def component_vector(values: dict[str, torch.Tensor], name: str):
    return torch.stack(
        [values[key] for key in component_keys(name)], dim=0
    ).sum(dim=0)


def individual_rows_for_batch(
    model,
    layers,
    device,
    batch,
    routes,
    components,
    candidate_ids,
    source_logits,
    target_logits,
    do_logits,
    source_patch,
    target_heads,
    do_heads,
):
    target_cases = [item["target"] for item in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    rows = []
    for route in routes:
        event = route["event"]
        event_id = event["event_id"]
        role = route["source_role"]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        values = components[event_id][role]
        for name in INDIVIDUAL_COMPONENTS:
            vector = component_vector(values, name)
            restore = (
                do_heads[layer_number][:, head_index, :].float()
                - vector
            )
            sufficiency = (
                target_heads[layer_number][:, head_index, :].float()
                + vector
            )
            restored_logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                head_patches=[{"event": event, "vectors": restore}],
            )
            sufficiency_logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                head_patches=[
                    {"event": event, "vectors": sufficiency}
                ],
            )
            restored_margin = semantic_margin(restored_logits, batch)
            sufficiency_margin = semantic_margin(
                sufficiency_logits, batch
            )
            restored_prediction = prediction_colors(restored_logits)
            sufficiency_prediction = prediction_colors(
                sufficiency_logits
            )
            for index, item in enumerate(batch):
                do_effect = float(
                    do_margin[index] - target_margin[index]
                )
                natural_effect = float(
                    source_margin[index] - target_margin[index]
                )
                rows.append(
                    {
                        "schema_version": (
                            "phase1001_qkv_individual_causal.v1"
                        ),
                        "phase": 1001,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "template": item["target"]["template"],
                        "route_id": route["route_id"],
                        "event_id": event_id,
                        "source_role": role,
                        "component": name,
                        "mediation_fraction": float(
                            (
                                do_margin[index]
                                - restored_margin[index]
                            )
                            / max(abs(do_effect), 1e-8)
                        ),
                        "sufficiency_transfer": float(
                            (
                                sufficiency_margin[index]
                                - target_margin[index]
                            )
                            / max(abs(natural_effect), 1e-8)
                        ),
                        "restored_to_target": (
                            restored_prediction[index]
                            == item["target"]["gold"]
                        ),
                        "sufficiency_flipped": (
                            sufficiency_prediction[index]
                            == item["source"]["gold"]
                        ),
                    }
                )
            del restored_logits, sufficiency_logits
    return rows


def group_patches(
    routes,
    components,
    target_heads,
    do_heads,
    component: str,
    mode: str,
):
    grouped = defaultdict(list)
    for route in routes:
        grouped[route["event_id"]].append(route)
    patches = []
    for event_id, event_routes in grouped.items():
        event = event_routes[0]["event"]
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        vector = torch.stack(
            [
                component_vector(
                    components[event_id][route["source_role"]],
                    component,
                )
                for route in event_routes
            ],
            dim=0,
        ).sum(dim=0)
        if mode == "restore":
            replacement = (
                do_heads[layer_number][:, head_index, :].float()
                - vector
            )
        elif mode == "sufficiency":
            replacement = (
                target_heads[layer_number][:, head_index, :].float()
                + vector
            )
        else:
            raise ValueError(mode)
        patches.append({"event": event, "vectors": replacement})
    return patches


def group_rows_for_batch(
    model,
    layers,
    device,
    batch,
    routes,
    components,
    candidate_ids,
    source_logits,
    target_logits,
    do_logits,
    source_patch,
    target_heads,
    do_heads,
):
    target_cases = [item["target"] for item in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    rows = []
    for name in GROUP_COMPONENTS:
        restore_patches = group_patches(
            routes,
            components,
            target_heads,
            do_heads,
            name,
            "restore",
        )
        sufficiency_patches = group_patches(
            routes,
            components,
            target_heads,
            do_heads,
            name,
            "sufficiency",
        )
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=restore_patches,
        )
        sufficiency_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=sufficiency_patches,
        )
        restored_margin = semantic_margin(restored_logits, batch)
        sufficiency_margin = semantic_margin(sufficiency_logits, batch)
        restored_prediction = prediction_colors(restored_logits)
        sufficiency_prediction = prediction_colors(sufficiency_logits)
        for index, item in enumerate(batch):
            do_effect = float(do_margin[index] - target_margin[index])
            natural_effect = float(
                source_margin[index] - target_margin[index]
            )
            rows.append(
                {
                    "schema_version": (
                        "phase1001_qkv_group_causal.v1"
                    ),
                    "phase": 1001,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "template": item["target"]["template"],
                    "component": name,
                    "route_count": len(routes),
                    "mediation_fraction": float(
                        (do_margin[index] - restored_margin[index])
                        / max(abs(do_effect), 1e-8)
                    ),
                    "sufficiency_transfer": float(
                        (
                            sufficiency_margin[index]
                            - target_margin[index]
                        )
                        / max(abs(natural_effect), 1e-8)
                    ),
                    "restored_to_target": (
                        restored_prediction[index]
                        == item["target"]["gold"]
                    ),
                    "sufficiency_flipped": (
                        sufficiency_prediction[index]
                        == item["source"]["gold"]
                    ),
                }
            )
        del restored_logits, sufficiency_logits
    return rows


def natural_rows_for_batch(
    model,
    layers,
    tokenizer,
    device,
    batch,
    routes,
    components,
    source_patch,
    target_heads,
    do_heads,
    effective_eos,
    budget,
):
    target_cases = [item["target"] for item in batch]
    conditions = {"source_do": []}
    for name in GROUP_COMPONENTS:
        conditions[f"source_minus_{name}"] = group_patches(
            routes,
            components,
            target_heads,
            do_heads,
            name,
            "restore",
        )
    rows = []
    for condition, patches in conditions.items():
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
                        "phase1001_qkv_group_natural.v1"
                    ),
                    "phase": 1001,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "condition": condition,
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


def summarize_causal(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_key)].append(row)
    result = {}
    for key, values in groups.items():
        label = "/".join(str(value) for value in key)
        result[label] = {
            **{name: value for name, value in zip(group_key, key)},
            "n": len(values),
            "median_mediation_fraction": float(
                np.median(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "mean_mediation_fraction": float(
                np.mean(
                    [row["mediation_fraction"] for row in values]
                )
            ),
            "positive_mediation_rate": float(
                np.mean(
                    [row["mediation_fraction"] > 0 for row in values]
                )
            ),
            "mean_sufficiency_transfer": float(
                np.mean(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "median_sufficiency_transfer": float(
                np.median(
                    [row["sufficiency_transfer"] for row in values]
                )
            ),
            "positive_sufficiency_rate": float(
                np.mean(
                    [row["sufficiency_transfer"] > 0 for row in values]
                )
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "template_median_mediation": {
                str(template): float(
                    np.median(
                        [
                            row["mediation_fraction"]
                            for row in values
                            if int(row["template"]) == template
                        ]
                    )
                )
                for template in range(4)
            },
        }
    return result


def summarize_natural(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
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
        for condition, values in groups.items()
    }


def run(stage: str, batch_size: int, natural_budget: int):
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 QKV decomposition requires CUDA")
    protocol, selected_pairs, directional, _ = selected_inputs(stage)
    output_root = QKV_ROOT / stage
    output_root.mkdir(parents=True, exist_ok=True)
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    source_spec = read_json(SOURCE_ROOT / "discovery" / "frozen_spec.json")
    frozen_ids = list(source_spec["frozen_joint_route_ids"])
    frozen_heads = read_json(HEAD_DISCOVERY_ROOT / "frozen_spec.json")
    events = [
        event_from_id(event_id)
        for event_id in frozen_heads["frozen_joint_event_ids"]
    ]
    route_lookup = {
        route["route_id"]: route for route in make_routes(events)
    }
    routes = [route_lookup[item] for item in frozen_ids]

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

        individual_rows = []
        group_rows = []
        natural_rows = []
        identity_errors = []
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
            do_logits, do_values, do_weights, do_heads = (
                capture_physical_attention(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                )
            )
            components, audit = component_map_for_batch(
                batch,
                events,
                target_values,
                target_weights,
                target_heads,
                do_values,
                do_weights,
                do_heads,
            )
            identity_errors.extend(
                row["max_abs_qkv_identity_error"] for row in audit
            )
            individual_rows.extend(
                individual_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    routes,
                    components,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            group_rows.extend(
                group_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    routes,
                    components,
                    candidate_ids,
                    source_logits,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            natural_rows.extend(
                natural_rows_for_batch(
                    model,
                    layers,
                    tokenizer,
                    device,
                    batch,
                    routes,
                    components,
                    source_patch,
                    target_heads,
                    do_heads,
                    effective_eos,
                    natural_budget,
                )
            )
            del (
                source_logits,
                source_residuals,
                target_logits,
                target_values,
                target_weights,
                target_heads,
                do_logits,
                do_values,
                do_weights,
                do_heads,
                components,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[qkv-{stage}] {batch_number}/{len(batches)} batches",
                    flush=True,
                )

        individual_summary = summarize_causal(
            individual_rows, ("route_id", "component")
        )
        group_summary = summarize_causal(group_rows, ("component",))
        natural_summary = summarize_natural(natural_rows)
        write_rows(output_root / "individual_rows.jsonl", individual_rows)
        write_rows(output_root / "group_rows.jsonl", group_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
        write_json(
            output_root / "individual_summary.json",
            individual_summary,
        )
        write_json(output_root / "group_summary.json", group_summary)
        write_json(output_root / "natural_summary.json", natural_summary)

        all_group = group_summary["all"]
        gate_checks = {
            "qkv_identity": max(identity_errors)
            <= QKV_THRESHOLDS["max_identity_error"],
            "source_do_natural": natural_summary["source_do"][
                "flip_rate"
            ]
            >= QKV_THRESHOLDS["source_do_natural_flip"],
            "all_group_mediation": all_group[
                "median_mediation_fraction"
            ]
            >= QKV_THRESHOLDS["all_group_median_mediation"],
            "all_group_sufficiency": all_group[
                "mean_sufficiency_transfer"
            ]
            >= QKV_THRESHOLDS["all_group_mean_sufficiency"],
            "all_group_natural": natural_summary["source_minus_all"][
                "target_rate"
            ]
            >= QKV_THRESHOLDS["all_group_natural_restoration"],
        }
        component_order = sorted(
            BASE_COMPONENTS,
            key=lambda name: (
                -group_summary[name]["median_mediation_fraction"],
                -group_summary[name]["mean_sufficiency_transfer"],
            ),
        )
        summary = {
            "schema_version": (
                f"phase1001_qkv_{stage}_summary.v1"
            ),
            "phase": 1001,
            "model": MODEL,
            "stage": stage,
            "partition": directional[0]["partition"],
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "frozen_route_count": len(routes),
            "frozen_route_ids": frozen_ids,
            "algebraic_components": {
                "qk": "(a_do-a_target) * v_target",
                "value": "a_target * (v_do-v_target)",
                "interaction": (
                    "(a_do-a_target) * (v_do-v_target)"
                ),
                "all": "qk + value + interaction",
            },
            "max_qkv_identity_error": max(identity_errors),
            "individual_summary": individual_summary,
            "group_summary": group_summary,
            "natural_summary": natural_summary,
            "base_component_order_by_group_mediation": component_order,
            "thresholds": QKV_THRESHOLDS,
            "gate_checks": gate_checks,
            "qkv_decomposition_gate_pass": all(gate_checks.values()),
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("discovery", "confirmation"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(
        args.stage, args.batch_size, args.natural_max_new_tokens
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "passed": summary["qkv_decomposition_gate_pass"],
                "gate_checks": summary["gate_checks"],
                "component_order": summary[
                    "base_component_order_by_group_mediation"
                ],
                "group_summary": summary["group_summary"],
                "natural_summary": summary["natural_summary"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
