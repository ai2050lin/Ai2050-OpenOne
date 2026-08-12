#!/usr/bin/env python3
"""BF16-confirm Phase1011 GLM4 native-task frozen-head positives."""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1010_glm4_bf16_causal_audit import (
    balanced_select,
    chunks,
    load_glm4_bf16,
)
from phase1011_frozen_head_local_validation import (
    PHASE1008_ROOT,
    QUALIFICATION_AXES,
    SOURCE_OPERATION,
    run_batch,
    summarize_subset,
)
from phase1011_native_semantic_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODEL = "glm4"
CELL_N = 8
BATCH_SIZE = 2


def main() -> None:
    eight_bit = read_json(
        OUT_ROOT / "causal_frozen_heads" / MODEL / "summary.json"
    )
    positive_keys = sorted({
        (
            row["family"],
            row["output_mode"],
            row["operation"],
        )
        for row in eight_bit["cell_summaries"]
        if row["descriptive_directional_replication_flag"]
    })
    if not positive_keys:
        raise RuntimeError("no 8-bit GLM4 positive cell authorized")
    selection_bundle = read_json(
        PHASE1008_ROOT
        / "refinement_final"
        / MODEL
        / "causal_selection.json"
    )
    selections = {
        selection["operation"]: selection
        for selection in selection_bundle["selections"]
    }
    units = read_jsonl(
        OUT_ROOT / "protocol" / MODEL / "units.jsonl"
    )
    cases = read_jsonl(
        OUT_ROOT / "protocol" / MODEL / "cases.jsonl"
    )
    qualification_rows = read_jsonl(
        OUT_ROOT / "behavior" / MODEL / "pair_qualification.jsonl"
    )
    qualification = {
        (row["unit_id"], row["operation"]): row
        for row in qualification_rows
    }
    case_by_id = {case["record_id"]: case for case in cases}
    output_root = OUT_ROOT / "precision_bf16" / MODEL
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
    summaries = []
    try:
        model, tokenizer, device = load_glm4_bf16()
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        device_map = {
            str(key): str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        }
        print(
            f"[phase1011-bf16] input_device={device} "
            f"mapped_components={len(device_map)}",
            flush=True,
        )
        for family, output_mode, operation in positive_keys:
            positive_axes = {
                row["qualification_axis"]
                for row in eight_bit["cell_summaries"]
                if row["family"] == family
                and row["output_mode"] == output_mode
                and row["operation"] == operation
                and row["descriptive_directional_replication_flag"]
            }
            candidates = []
            for unit in units:
                if (
                    unit["family"] != family
                    or unit["output_mode"] != output_mode
                    or unit["split"] != "confirmation"
                ):
                    continue
                pair = qualification[(unit["unit_id"], operation)]
                if not any(
                    pair[QUALIFICATION_AXES[axis]]
                    for axis in positive_axes
                ):
                    continue
                candidates.append({
                    "unit": unit,
                    "operation": operation,
                    "qualification": pair,
                    "base": case_by_id[
                        unit["case_ids"]["base"]
                    ],
                    "variant": case_by_id[
                        unit["case_ids"][operation]
                    ],
                })
            items = balanced_select(candidates, CELL_N)
            if len(items) < CELL_N:
                raise RuntimeError(
                    f"BF16 cell underfilled: {family}/"
                    f"{output_mode}/{operation} n={len(items)}"
                )
            grouped: dict[
                tuple[int, int, int],
                list[dict[str, Any]],
            ] = defaultdict(list)
            for item in items:
                grouped[(
                    int(item["unit"]["template"]),
                    len(item["base"]["input_ids"]),
                    len(item["variant"]["input_ids"]),
                )].append(item)
            cell_rows = []
            selection = selections[SOURCE_OPERATION[operation]]
            layer = layers[int(selection["layer"]) - 1]
            for group in grouped.values():
                for batch_items in chunks(group, BATCH_SIZE):
                    cell_rows.extend(run_batch(
                        model=model,
                        layer=layer,
                        device=device,
                        head_count=head_count,
                        selection=selection,
                        items=batch_items,
                    ))
            for row in cell_rows:
                row["schema_version"] = (
                    "phase1011_glm4_bf16_native_unit.v1"
                )
                row["precision"] = "bfloat16"
                row["bf16_entry_reason"] = (
                    "positive_8bit_native_cell"
                )
            all_rows.extend(cell_rows)
            for axis in QUALIFICATION_AXES:
                cell_summary = summarize_subset(
                    model_name=MODEL,
                    family=family,
                    output_mode=output_mode,
                    operation=operation,
                    qualification_axis=axis,
                    rows=cell_rows,
                )
                cell_summary["schema_version"] = (
                    "phase1011_glm4_bf16_native_cell.v1"
                )
                cell_summary["precision"] = "bfloat16"
                cell_summary["eight_bit_positive_axis"] = (
                    axis in positive_axes
                )
                matching_8bit = next(
                    row
                    for row in eight_bit["cell_summaries"]
                    if row["family"] == family
                    and row["output_mode"] == output_mode
                    and row["operation"] == operation
                    and row["qualification_axis"] == axis
                )
                cell_summary["eight_bit_flag"] = matching_8bit[
                    "descriptive_directional_replication_flag"
                ]
                cell_summary["bf16_confirms_eight_bit_flag"] = bool(
                    matching_8bit[
                        "descriptive_directional_replication_flag"
                    ]
                    and cell_summary[
                        "descriptive_directional_replication_flag"
                    ]
                )
                summaries.append(cell_summary)
            print(
                f"[phase1011-bf16] {family}/{output_mode}/"
                f"{operation} n={len(cell_rows)}",
                flush=True,
            )
        maximum_noop = max(
            row["noop_max_logit_error"] for row in all_rows
        )
        if maximum_noop > 1e-5:
            raise RuntimeError(f"BF16 no-op failed: {maximum_noop}")
        entered = [
            row for row in summaries if row["eight_bit_positive_axis"]
        ]
        confirmed = [
            row for row in entered
            if row["bf16_confirms_eight_bit_flag"]
        ]
        result = {
            "schema_version": (
                "phase1011_glm4_bf16_native_summary.v1"
            ),
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "entry_frozen_from_8bit_before_bf16": True,
            "entered_task_cells": [
                {
                    "family": family,
                    "output_mode": output_mode,
                    "operation": operation,
                }
                for family, output_mode, operation in positive_keys
            ],
            "sample_n_per_task_cell": CELL_N,
            "batch_size": BATCH_SIZE,
            "entered_positive_axis_count": len(entered),
            "bf16_confirmed_positive_axis_count": len(confirmed),
            "bf16_confirmation_rate": (
                len(confirmed) / max(len(entered), 1)
            ),
            "maximum_noop_logit_error": maximum_noop,
            "no_op_audit_pass": True,
            "device_map": device_map,
            "cell_summaries": summaries,
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "precision confirmation of selected local interventions; "
                "n=8 per task cell is not a stable population estimate "
                "and does not establish natural-rollout causality"
            ),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(
            output_root / "cell_summaries.jsonl", summaries
        )
        write_json(output_root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        del model, tokenizer, device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
