#!/usr/bin/env python3
"""BF16 confirmation of discovery-frozen relay write depths."""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
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
    cell_summaries,
    run_group,
)
from phase1010_output_type_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


CONFIRMATION_N = 8


def main() -> None:
    discovery_root = OUT_ROOT / "relay_depth_mapping" / "discovery"
    discovery = read_json(discovery_root / "summary.json")
    selection_result = read_json(discovery_root / "selection.json")
    depths = tuple(int(value) for value in selection_result[
        "confirmation_depths"
    ])
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
    output_root = (
        OUT_ROOT / "relay_depth_mapping" / "bf16_confirmation"
    )
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
    try:
        model, tokenizer, device = load_glm4_bf16()
        layers = get_layers(model)
        device_map = {
            str(key): str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        }
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
            if len(items) < CONFIRMATION_N:
                raise RuntimeError(
                    f"{family}: BF16 depth confirmation underfilled"
                )
            grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(
                list
            )
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
                    all_rows.extend(run_group(
                        model=model,
                        layers=layers,
                        device=device,
                        selected_heads=selected_heads,
                        items=batch,
                        depths=depths,
                    ))
            print(
                f"[relay-depth-bf16] {family} n={len(items)} "
                f"depths={list(depths)}",
                flush=True,
            )
        summaries = cell_summaries(
            all_rows,
            precision="bfloat16",
            split="confirmation",
            depths=depths,
        )
        for row in all_rows:
            row["schema_version"] = (
                "phase1010_relay_depth_bf16_unit.v1"
            )
            row["precision"] = "bfloat16"
        for row in summaries:
            row["schema_version"] = (
                "phase1010_relay_depth_bf16_cell.v1"
            )
        by_key = {
            (row["family"], int(row["depth"])): row
            for row in summaries
        }
        l0_negative_control_pass = all(
            abs(by_key[(family, 0)][
                "median_sufficiency_fraction"
            ]) <= 1e-8
            and abs(by_key[(family, 0)][
                "median_restore_fraction"
            ]) <= 1e-8
            and abs(by_key[(family, 0)][
                "median_shuffled_sufficiency_fraction"
            ]) <= 1e-8
            for family in FAMILIES
        )
        confirmed_depths = []
        for depth in depths:
            if depth == 0:
                continue
            if all(
                by_key[(family, depth)][
                    "median_sufficiency_fraction"
                ] > 0
                and by_key[(family, depth)][
                    "median_restore_fraction"
                ] > 0
                and by_key[(family, depth)][
                    "median_sufficiency_fraction"
                ]
                > by_key[(family, depth)][
                    "median_shuffled_sufficiency_fraction"
                ]
                for family in FAMILIES
            ):
                confirmed_depths.append(depth)
        result = {
            "schema_version": (
                "phase1010_relay_depth_bf16_confirmation.v1"
            ),
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "split": "confirmation",
            "source_discovery_summary": discovery,
            "selected_heads": selected_heads,
            "relay_position_count": 68,
            "confirmation_depths": list(depths),
            "confirmation_n_per_family": CONFIRMATION_N,
            "cell_summaries": summaries,
            "l0_negative_control_pass": l0_negative_control_pass,
            "confirmed_depths": confirmed_depths,
            "maximum_noop_logit_error": max(
                row["noop_max_logit_error"] for row in all_rows
            ),
            "device_map": device_map,
            "elapsed_seconds": time.time() - started,
            "strongest_supported_statement": (
                "The fixed trailing relay field contains a transferable "
                "code-label F state by the confirmed sampled depth(s), and "
                "that state drives both the frozen L30 head vector and the "
                "output margin in two held-out families."
            ),
            "automatic_next_step": (
                "localize_relay_tokens_and_separate_value_content_from_"
                "attention_routing"
                if confirmed_depths
                else "stop_relay_depth_route"
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
