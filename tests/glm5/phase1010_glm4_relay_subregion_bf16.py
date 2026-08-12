#!/usr/bin/env python3
"""BF16 localization of the confirmed 68-token relay field at L24."""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1009_crossfamily_heldout_causal_replication import (
    candidate_margin,
    finite_fraction,
)
from phase1009_crossfamily_response_scan import stage_case
from phase1010_glm4_bf16_causal_audit import (
    BF16_BATCH_SIZE,
    balanced_select,
    chunks,
    load_glm4_bf16,
)
from phase1010_glm4_relay_depth_discovery import (
    FAMILIES,
    MODEL,
    OPERATION,
    OUTPUT_TYPE,
    PHASE1008_ROOT,
    natural_capture,
    patched_forward,
    projection_fraction,
)
from phase1010_output_type_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEPTH = 24
CONFIRMATION_N = 8
SUBREGIONS = {
    "response_map_declaration": tuple(range(0, 27)),
    "semantic_bridge_instruction": tuple(range(27, 46)),
    "output_protocol_instruction": tuple(range(46, 68)),
    "all_relay_positions": tuple(range(0, 68)),
}


def run_batch(
    *,
    model,
    layers,
    device,
    selected_heads,
    items,
):
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
    rows = []
    full_base = base_residuals[DEPTH]
    full_variant = variant_residuals[DEPTH]
    for subregion, offsets in SUBREGIONS.items():
        index = torch.tensor(
            offsets,
            dtype=torch.long,
            device=full_base.device,
        )
        base_part = full_base.index_select(1, index)
        variant_part = full_variant.index_select(1, index)
        delta_part = variant_part - base_part
        shuffled_delta = torch.roll(delta_part, shifts=1, dims=0)
        suff_logits, suff_heads = patched_forward(
            model=model,
            layers=layers,
            cases=base_cases,
            originals=base_original,
            device=device,
            selected_heads=selected_heads,
            depth=DEPTH,
            replacement=variant_part,
            relay_offsets=offsets,
        )
        restore_logits, restore_heads = patched_forward(
            model=model,
            layers=layers,
            cases=variant_cases,
            originals=variant_original,
            device=device,
            selected_heads=selected_heads,
            depth=DEPTH,
            replacement=base_part,
            relay_offsets=offsets,
        )
        shuffled_logits, shuffled_heads = patched_forward(
            model=model,
            layers=layers,
            cases=base_cases,
            originals=base_original,
            device=device,
            selected_heads=selected_heads,
            depth=DEPTH,
            replacement=base_part + shuffled_delta,
            relay_offsets=offsets,
        )
        noop_logits, _ = patched_forward(
            model=model,
            layers=layers,
            cases=base_cases,
            originals=base_original,
            device=device,
            selected_heads=selected_heads,
            depth=DEPTH,
            replacement=base_part,
            relay_offsets=offsets,
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
        suff_head = projection_fraction(
            suff_heads - base_heads,
            natural_head_delta,
        )
        restore_head = projection_fraction(
            variant_heads - restore_heads,
            natural_head_delta,
        )
        shuffled_head = projection_fraction(
            shuffled_heads - base_heads,
            natural_head_delta,
        )
        delta_norm = torch.linalg.vector_norm(
            delta_part.float().reshape(len(items), -1),
            dim=-1,
        )
        full_norm = torch.linalg.vector_norm(
            (full_variant - full_base).float().reshape(len(items), -1),
            dim=-1,
        )
        noop_error = torch.max(
            torch.abs(noop_logits - base_logits),
            dim=-1,
        ).values
        for item_index, item in enumerate(items):
            rows.append({
                "schema_version": "phase1010_relay_subregion_unit.v1",
                "phase": PHASE,
                "model": MODEL,
                "precision": "bfloat16",
                "split": "confirmation",
                "family": item["unit"]["family"],
                "output_type": OUTPUT_TYPE,
                "operation": OPERATION,
                "unit_id": item["unit"]["unit_id"],
                "depth": DEPTH,
                "subregion": subregion,
                "position_count": len(offsets),
                "subregion_delta_norm": float(
                    delta_norm[item_index].item()
                ),
                "full_relay_delta_norm": float(
                    full_norm[item_index].item()
                ),
                "subregion_to_full_delta_norm_ratio": float(
                    delta_norm[item_index].item()
                    / max(full_norm[item_index].item(), 1e-8)
                ),
                "sufficiency_fraction": float(
                    suff_fraction[item_index].item()
                ),
                "restore_fraction": float(
                    restore_fraction[item_index].item()
                ),
                "shuffled_sufficiency_fraction": float(
                    shuffled_fraction[item_index].item()
                ),
                "sufficiency_head_projection": float(
                    suff_head[item_index].item()
                ),
                "restore_head_projection": float(
                    restore_head[item_index].item()
                ),
                "shuffled_head_projection": float(
                    shuffled_head[item_index].item()
                ),
                "noop_max_logit_error": float(
                    noop_error[item_index].item()
                ),
            })
    return rows


def main() -> None:
    depth_confirmation = read_json(
        OUT_ROOT
        / "relay_depth_mapping"
        / "bf16_confirmation"
        / "summary.json"
    )
    if DEPTH not in depth_confirmation["confirmed_depths"]:
        raise RuntimeError("L24 was not BF16-confirmed")
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
    output_root = OUT_ROOT / "relay_subregion_mapping" / "bf16"
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
    try:
        model, tokenizer, device = load_glm4_bf16()
        layers = get_layers(model)
        for family in FAMILIES:
            candidates = []
            for unit in units:
                if (
                    unit["family"] != family
                    or unit["output_type"] != OUTPUT_TYPE
                    or unit["split"] != "confirmation"
                ):
                    continue
                if not qualification[
                    (unit["unit_id"], OPERATION)
                ]["semantic_pair_qualified"]:
                    continue
                candidates.append({
                    "unit": unit,
                    "base": case_by_id[unit["case_ids"]["base"]],
                    "variant": case_by_id[
                        unit["case_ids"][OPERATION]
                    ],
                })
            items = balanced_select(candidates, CONFIRMATION_N)
            grouped = defaultdict(list)
            for item in items:
                base = stage_case(item["base"], "semantic0")
                variant = stage_case(item["variant"], "semantic0")
                grouped[(
                    int(item["unit"]["template"]),
                    len(base["input_ids"]),
                    len(variant["input_ids"]),
                )].append(item)
            for group in grouped.values():
                for batch in chunks(group, BF16_BATCH_SIZE):
                    all_rows.extend(run_batch(
                        model=model,
                        layers=layers,
                        device=device,
                        selected_heads=selected_heads,
                        items=batch,
                    ))
            print(
                f"[relay-subregion] {family} n={len(items)}",
                flush=True,
            )
        summaries = []
        for family in FAMILIES:
            for subregion, offsets in SUBREGIONS.items():
                selected = [
                    row
                    for row in all_rows
                    if row["family"] == family
                    and row["subregion"] == subregion
                ]

                def median(field):
                    return float(np.median([
                        row[field] for row in selected
                    ]))

                summaries.append({
                    "schema_version": (
                        "phase1010_relay_subregion_cell.v1"
                    ),
                    "phase": PHASE,
                    "model": MODEL,
                    "precision": "bfloat16",
                    "split": "confirmation",
                    "family": family,
                    "output_type": OUTPUT_TYPE,
                    "operation": OPERATION,
                    "depth": DEPTH,
                    "subregion": subregion,
                    "position_count": len(offsets),
                    "n": len(selected),
                    "median_subregion_to_full_delta_norm_ratio": median(
                        "subregion_to_full_delta_norm_ratio"
                    ),
                    "median_sufficiency_fraction": median(
                        "sufficiency_fraction"
                    ),
                    "median_restore_fraction": median(
                        "restore_fraction"
                    ),
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
                        row["noop_max_logit_error"]
                        for row in selected
                    ),
                })
        region_support = {
            subregion: {
                family: next(
                    row
                    for row in summaries
                    if row["family"] == family
                    and row["subregion"] == subregion
                )
                for family in FAMILIES
            }
            for subregion in SUBREGIONS
        }
        result = {
            "schema_version": "phase1010_relay_subregion_bf16.v1",
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "split": "confirmation",
            "depth": DEPTH,
            "subregions_frozen_from_prompt_construction": {
                key: [min(value), max(value)]
                for key, value in SUBREGIONS.items()
            },
            "cell_summaries": summaries,
            "region_support": region_support,
            "maximum_noop_logit_error": max(
                row["noop_max_logit_error"] for row in all_rows
            ),
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "coarse fixed text subregions; individual relay tokens and "
                "the QK/V split remain unresolved"
            ),
        }
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(output_root / "cell_summaries.jsonl", summaries)
        write_json(output_root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            del model
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
