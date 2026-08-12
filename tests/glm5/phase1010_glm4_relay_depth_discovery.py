#!/usr/bin/env python3
"""Trace when semantic differences enter fixed trailing relay positions.

Full residual states at the discovery-confirmed response-map/instruction
positions are patched at pre-registered depths. The scan reports the complete
depth profile before selecting a depth for BF16 confirmation.
"""
from __future__ import annotations

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

from model_utils import get_layers, load_model, release_model
from phase1009_crossfamily_heldout_causal_replication import (
    candidate_margin,
    finite_fraction,
)
from phase1009_crossfamily_response_scan import case_tensors, stage_case
from phase1010_glm4_source_role_discovery import (
    FAMILIES,
    MODEL,
    OPERATION,
    OUTPUT_TYPE,
    PHASE1008_ROOT,
    source_partition,
)
from phase1010_output_type_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEPTHS = (0, 4, 8, 12, 16, 20, 24, 28)
TARGET_LAYER = 30
EPSILON = 1e-8


def relay_positions(
    originals: list[dict[str, Any]],
    staged: list[dict[str, Any]],
    device,
) -> torch.Tensor:
    rows = [
        source_partition(original, stage)["response_map_instruction"]
        for original, stage in zip(originals, staged)
    ]
    widths = {len(row) for row in rows}
    if widths != {68}:
        raise RuntimeError(f"relay width drift {widths}")
    return torch.tensor(rows, dtype=torch.long, device=device)


def residual_value(output) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise RuntimeError("unexpected residual hook output")
    return value


def replace_output(output, value: torch.Tensor):
    if isinstance(output, tuple):
        return (value,) + tuple(output[1:])
    return value


def natural_capture(
    *,
    model,
    layers,
    cases: list[dict[str, Any]],
    originals: list[dict[str, Any]],
    device,
    selected_heads: list[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[int, torch.Tensor],
]:
    input_ids, attention_mask = case_tensors(cases, device)
    relays = relay_positions(originals, cases, device)
    decision = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in cases
        ],
        dtype=torch.long,
        device=device,
    )
    head_count = int(model.config.num_attention_heads)
    head_dim = int(model.config.head_dim)
    residuals: dict[int, torch.Tensor] = {}
    heads: list[torch.Tensor] = []
    counts = defaultdict(int)

    def make_residual_hook(depth: int):
        def hook(module, args, output):
            value = residual_value(output)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            residuals[depth] = value[
                batch,
                relays.to(value.device),
                :,
            ].detach()
            counts[f"residual:{depth}"] += 1
        return hook

    def head_hook(module, args):
        value = args[0]
        batch = torch.arange(value.shape[0], device=value.device)
        selected = value[
            batch,
            decision.to(value.device),
            :,
        ].reshape(value.shape[0], head_count, head_dim)
        heads.append(selected[:, selected_heads, :].detach())
        counts["heads"] += 1

    handles = [
        model.get_input_embeddings().register_forward_hook(
            make_residual_hook(0)
        ),
        layers[TARGET_LAYER - 1].self_attn.o_proj.register_forward_pre_hook(
            head_hook
        ),
    ]
    for depth in DEPTHS:
        if depth == 0:
            continue
        handles.append(
            layers[depth - 1].register_forward_hook(
                make_residual_hook(depth)
            )
        )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        expected = {f"residual:{depth}" for depth in DEPTHS}
        if any(counts[key] != 1 for key in expected):
            raise RuntimeError(f"residual capture drift {dict(counts)}")
        if counts["heads"] != 1 or len(heads) != 1:
            raise RuntimeError("head capture drift")
        return output.logits[:, -1, :].detach(), heads[0], residuals
    finally:
        for handle in reversed(handles):
            handle.remove()
        del input_ids, attention_mask, relays, decision


def patched_forward(
    *,
    model,
    layers,
    cases: list[dict[str, Any]],
    originals: list[dict[str, Any]],
    device,
    selected_heads: list[int],
    depth: int,
    replacement: torch.Tensor,
    relay_offsets: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = case_tensors(cases, device)
    relays = relay_positions(originals, cases, device)
    if relay_offsets is not None:
        offset_index = torch.tensor(
            relay_offsets,
            dtype=torch.long,
            device=relays.device,
        )
        relays = relays.index_select(1, offset_index)
    decision = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in cases
        ],
        dtype=torch.long,
        device=device,
    )
    head_count = int(model.config.num_attention_heads)
    head_dim = int(model.config.head_dim)
    counts = defaultdict(int)
    heads: list[torch.Tensor] = []

    def patch_hook(module, args, output):
        value = residual_value(output)
        patched = value.clone()
        batch = torch.arange(value.shape[0], device=value.device)[:, None]
        patched[
            batch,
            relays.to(value.device),
            :,
        ] = replacement.to(device=value.device, dtype=value.dtype)
        counts["patch"] += 1
        return replace_output(output, patched)

    def head_hook(module, args):
        value = args[0]
        batch = torch.arange(value.shape[0], device=value.device)
        selected = value[
            batch,
            decision.to(value.device),
            :,
        ].reshape(value.shape[0], head_count, head_dim)
        heads.append(selected[:, selected_heads, :].detach())
        counts["heads"] += 1

    source_module = (
        model.get_input_embeddings()
        if depth == 0
        else layers[depth - 1]
    )
    handles = [
        source_module.register_forward_hook(patch_hook),
        layers[TARGET_LAYER - 1].self_attn.o_proj.register_forward_pre_hook(
            head_hook
        ),
    ]
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if counts["patch"] != 1 or counts["heads"] != 1:
            raise RuntimeError(f"relay patch count drift {dict(counts)}")
        return output.logits[:, -1, :].detach(), heads[0]
    finally:
        for handle in reversed(handles):
            handle.remove()
        del input_ids, attention_mask, relays, decision


def projection_fraction(
    change: torch.Tensor,
    natural: torch.Tensor,
) -> torch.Tensor:
    flat_change = change.float().reshape(change.shape[0], -1)
    flat_natural = natural.float().reshape(natural.shape[0], -1)
    return torch.sum(flat_change * flat_natural, dim=-1) / torch.clamp(
        torch.sum(flat_natural * flat_natural, dim=-1),
        min=EPSILON,
    )


def run_group(
    *,
    model,
    layers,
    device,
    selected_heads: list[int],
    items: list[dict[str, Any]],
    depths: tuple[int, ...] = DEPTHS,
) -> list[dict[str, Any]]:
    base_original = [item["base"] for item in items]
    variant_original = [item["variant"] for item in items]
    base_cases = [stage_case(case, "semantic0") for case in base_original]
    variant_cases = [
        stage_case(case, "semantic0") for case in variant_original
    ]
    base_logits, base_heads, base_residuals = natural_capture(
        model=model,
        layers=layers,
        cases=base_cases,
        originals=base_original,
        device=device,
        selected_heads=selected_heads,
    )
    variant_logits, variant_heads, variant_residuals = natural_capture(
        model=model,
        layers=layers,
        cases=variant_cases,
        originals=variant_original,
        device=device,
        selected_heads=selected_heads,
    )
    base_margin = candidate_margin(
        base_logits,
        base_cases,
        variant_cases,
    )
    variant_margin = candidate_margin(
        variant_logits,
        base_cases,
        variant_cases,
    )
    natural_effect = variant_margin - base_margin
    natural_head_delta = variant_heads - base_heads
    natural_head_norm = torch.linalg.vector_norm(
        natural_head_delta.float().reshape(len(items), -1),
        dim=-1,
    )
    rows = []
    for depth in depths:
        suff_logits, suff_heads = patched_forward(
            model=model,
            layers=layers,
            cases=base_cases,
            originals=base_original,
            device=device,
            selected_heads=selected_heads,
            depth=depth,
            replacement=variant_residuals[depth],
        )
        restore_logits, restore_heads = patched_forward(
            model=model,
            layers=layers,
            cases=variant_cases,
            originals=variant_original,
            device=device,
            selected_heads=selected_heads,
            depth=depth,
            replacement=base_residuals[depth],
        )
        relay_delta = (
            variant_residuals[depth] - base_residuals[depth]
        )
        shuffled_delta = torch.roll(relay_delta, shifts=1, dims=0)
        shuffled_logits, shuffled_heads = patched_forward(
            model=model,
            layers=layers,
            cases=base_cases,
            originals=base_original,
            device=device,
            selected_heads=selected_heads,
            depth=depth,
            replacement=base_residuals[depth] + shuffled_delta,
        )
        noop_logits, _ = patched_forward(
            model=model,
            layers=layers,
            cases=base_cases,
            originals=base_original,
            device=device,
            selected_heads=selected_heads,
            depth=depth,
            replacement=base_residuals[depth],
        )
        suff_margin = candidate_margin(
            suff_logits,
            base_cases,
            variant_cases,
        )
        restore_margin = candidate_margin(
            restore_logits,
            base_cases,
            variant_cases,
        )
        shuffled_margin = candidate_margin(
            shuffled_logits,
            base_cases,
            variant_cases,
        )
        suff_fraction = finite_fraction(
            suff_margin - base_margin,
            natural_effect,
        )
        restore_fraction = finite_fraction(
            variant_margin - restore_margin,
            natural_effect,
        )
        shuffled_fraction = finite_fraction(
            shuffled_margin - base_margin,
            natural_effect,
        )
        suff_head_projection = projection_fraction(
            suff_heads - base_heads,
            natural_head_delta,
        )
        restore_head_projection = projection_fraction(
            variant_heads - restore_heads,
            natural_head_delta,
        )
        shuffled_head_projection = projection_fraction(
            shuffled_heads - base_heads,
            natural_head_delta,
        )
        relay_delta_norm = torch.linalg.vector_norm(
            relay_delta.float().reshape(len(items), -1),
            dim=-1,
        )
        noop_error = torch.max(
            torch.abs(noop_logits - base_logits),
            dim=-1,
        ).values
        for index, item in enumerate(items):
            rows.append({
                "schema_version": "phase1010_relay_depth_unit.v1",
                "phase": PHASE,
                "model": MODEL,
                "split": item["unit"]["split"],
                "precision": (
                    "bfloat16"
                    if not getattr(model, "is_loaded_in_8bit", False)
                    else "8bit"
                ),
                "family": item["unit"]["family"],
                "output_type": OUTPUT_TYPE,
                "operation": OPERATION,
                "unit_id": item["unit"]["unit_id"],
                "template": int(item["unit"]["template"]),
                "name_pool": int(item["unit"]["name_pool"]),
                "depth": depth,
                "relative_depth": depth / len(layers),
                "relay_position_count": 68,
                "relay_delta_norm": float(
                    relay_delta_norm[index].item()
                ),
                "natural_head_delta_norm": float(
                    natural_head_norm[index].item()
                ),
                "sufficiency_fraction": float(
                    suff_fraction[index].item()
                ),
                "restore_fraction": float(
                    restore_fraction[index].item()
                ),
                "shuffled_sufficiency_fraction": float(
                    shuffled_fraction[index].item()
                ),
                "sufficiency_head_projection": float(
                    suff_head_projection[index].item()
                ),
                "restore_head_projection": float(
                    restore_head_projection[index].item()
                ),
                "shuffled_head_projection": float(
                    shuffled_head_projection[index].item()
                ),
                "noop_max_logit_error": float(noop_error[index].item()),
            })
        del suff_logits, restore_logits, shuffled_logits, noop_logits
    return rows


def cell_summaries(
    rows: list[dict[str, Any]],
    *,
    precision: str,
    split: str,
    depths: tuple[int, ...],
) -> list[dict[str, Any]]:
    result = []
    for family in FAMILIES:
        for depth in depths:
            selected = [
                row
                for row in rows
                if row["family"] == family and row["depth"] == depth
            ]

            def median(name: str) -> float:
                return float(np.median([row[name] for row in selected]))

            result.append({
                "schema_version": "phase1010_relay_depth_cell.v1",
                "phase": PHASE,
                "model": MODEL,
                "split": split,
                "precision": precision,
                "family": family,
                "output_type": OUTPUT_TYPE,
                "operation": OPERATION,
                "depth": depth,
                "relative_depth": depth / 40,
                "n": len(selected),
                "median_relay_delta_norm": median("relay_delta_norm"),
                "median_sufficiency_fraction": median(
                    "sufficiency_fraction"
                ),
                "median_restore_fraction": median("restore_fraction"),
                "median_shuffled_sufficiency_fraction": median(
                    "shuffled_sufficiency_fraction"
                ),
                "median_sufficiency_head_projection": median(
                    "sufficiency_head_projection"
                ),
                "median_restore_head_projection": median(
                    "restore_head_projection"
                ),
                "median_shuffled_head_projection": median(
                    "shuffled_head_projection"
                ),
                "maximum_noop_logit_error": max(
                    row["noop_max_logit_error"] for row in selected
                ),
                "claim_limit": (
                    "broad 68-position relay-field patch; not a minimal "
                    "source circuit or proof of first write time"
                ),
            })
    return result


def main() -> None:
    source_confirmation = read_json(
        OUT_ROOT
        / "source_role_mapping"
        / "bf16_confirmation"
        / "summary.json"
    )
    if not source_confirmation["response_map_instruction_repeats"]:
        raise RuntimeError("source relay did not authorize depth tracing")
    selection_bundle = read_json(
        PHASE1008_ROOT
        / "refinement_final"
        / MODEL
        / "causal_selection.json"
    )
    selection = {
        row["operation"]: row
        for row in selection_bundle["selections"]
    }["B"]
    selected_heads = [int(value) for value in selection["selected_heads"]]
    cases = read_jsonl(
        OUT_ROOT / "protocol" / MODEL / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / MODEL / "units.jsonl"
    )
    qualifications = read_jsonl(
        OUT_ROOT / "behavior" / MODEL / "pair_qualification.jsonl"
    )
    qualification = {
        (row["unit_id"], row["operation"]): row
        for row in qualifications
    }
    case_by_id = {case["record_id"]: case for case in cases}
    output_root = OUT_ROOT / "relay_depth_mapping" / "discovery"
    started = time.time()
    model = tokenizer = device = None
    all_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(MODEL, use_8bit=True)
        layers = get_layers(model)
        for family in FAMILIES:
            items = []
            for unit in units:
                if (
                    unit["family"] != family
                    or unit["output_type"] != OUTPUT_TYPE
                    or unit["split"] != "discovery"
                ):
                    continue
                if not qualification[
                    (unit["unit_id"], OPERATION)
                ]["semantic_pair_qualified"]:
                    continue
                items.append({
                    "unit": unit,
                    "base": case_by_id[unit["case_ids"]["base"]],
                    "variant": case_by_id[
                        unit["case_ids"][OPERATION]
                    ],
                })
            grouped: dict[
                tuple[int, int, int],
                list[dict[str, Any]],
            ] = defaultdict(list)
            for item in items:
                base = stage_case(item["base"], "semantic0")
                variant = stage_case(item["variant"], "semantic0")
                grouped[(
                    int(item["unit"]["template"]),
                    len(base["input_ids"]),
                    len(variant["input_ids"]),
                )].append(item)
            for group in grouped.values():
                all_rows.extend(run_group(
                    model=model,
                    layers=layers,
                    device=device,
                    selected_heads=selected_heads,
                    items=group,
                ))
            print(
                f"[relay-depth] {family} n={len(items)} "
                f"groups={len(grouped)}",
                flush=True,
            )
        summaries = cell_summaries(
            all_rows,
            precision="8bit",
            split="discovery",
            depths=DEPTHS,
        )
        depth_scores = []
        for depth in DEPTHS:
            cells = [row for row in summaries if row["depth"] == depth]
            family_scores = [
                min(
                    row["median_sufficiency_fraction"],
                    row["median_restore_fraction"],
                )
                - max(
                    row["median_shuffled_sufficiency_fraction"],
                    0.0,
                )
                for row in cells
            ]
            depth_scores.append({
                "depth": depth,
                "cross_family_min_output_excess": float(
                    min(family_scores)
                ),
                "cross_family_median_output_excess": float(
                    np.median(family_scores)
                ),
                "family_values": {
                    row["family"]: {
                        "sufficiency": row[
                            "median_sufficiency_fraction"
                        ],
                        "restore": row["median_restore_fraction"],
                        "shuffled": row[
                            "median_shuffled_sufficiency_fraction"
                        ],
                        "head_sufficiency_projection": row[
                            "median_sufficiency_head_projection"
                        ],
                        "head_restore_projection": row[
                            "median_restore_head_projection"
                        ],
                    }
                    for row in cells
                },
            })
        strongest = max(
            depth_scores,
            key=lambda row: (
                row["cross_family_min_output_excess"],
                row["cross_family_median_output_excess"],
                row["depth"],
            ),
        )
        strongest_depth = int(strongest["depth"])
        confirmation_depths = sorted({
            0,
            strongest_depth,
            max(0, strongest_depth - 4),
        })
        selection_result = {
            "schema_version": "phase1010_relay_depth_selection.v1",
            "phase": PHASE,
            "model": MODEL,
            "discovery_split_only": True,
            "depths_measured": list(DEPTHS),
            "depth_scores": depth_scores,
            "strongest_depth": strongest_depth,
            "confirmation_depths": confirmation_depths,
            "selection_rule": (
                "measure every frozen depth, then maximize the minimum "
                "across families of min(median output sufficiency, median "
                "restoration) minus positive shuffled sufficiency"
            ),
            "formula_status": (
                "depth-selection instrument only; not a mechanism formula"
            ),
        }
        summary = {
            "schema_version": "phase1010_relay_depth_discovery.v1",
            "phase": PHASE,
            "model": MODEL,
            "precision": "8bit",
            "split": "discovery",
            "layer_target": TARGET_LAYER,
            "selected_heads": selected_heads,
            "relay_position_count": 68,
            "depths": list(DEPTHS),
            "unit_depth_row_count": len(all_rows),
            "cell_depth_count": len(summaries),
            "strongest_depth": strongest_depth,
            "confirmation_depths": confirmation_depths,
            "maximum_noop_logit_error": max(
                row["noop_max_logit_error"] for row in all_rows
            ),
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "broad relay-field depth profile; exact token/layer write "
                "circuit remains unresolved"
            ),
        }
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(output_root / "cell_summaries.jsonl", summaries)
        write_json(output_root / "selection.json", selection_result)
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
