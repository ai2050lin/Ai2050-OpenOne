#!/usr/bin/env python3
"""BF16 precision audit for Phase1010 GLM4 frozen-head causal cells.

All adequately powered person cells are audited. Non-person cells enter only
when the independent 8-bit screen was positive. Cell entry and sample order
are therefore frozen before BF16 outcomes are observed.
"""
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

from model_utils import MODEL_CONFIGS, get_layers
from phase1009_crossfamily_heldout_causal_replication import (
    SOURCE_OPERATION,
    run_batch,
    summarize,
)
from phase1009_crossfamily_response_scan import stage_case
from phase1010_output_type_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODEL = "glm4"
BF16_CELL_N = 8
BF16_BATCH_SIZE = 2
PHASE1008_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
)


def chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_glm4_bf16():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = MODEL_CONFIGS[MODEL]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_memory = {0: "11GiB", "cpu": "18GiB"}
    print(
        "[bf16] Loading glm4 with auto CPU/GPU placement "
        f"max_memory={max_memory}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    return model, tokenizer, device


def balanced_select(
    items: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(
        items,
        key=lambda row: (
            int(row["unit"]["template"]),
            int(row["unit"]["name_pool"]),
            int(row["unit"]["world_index"]),
            row["unit"]["unit_id"],
        ),
    ):
        groups[(
            int(item["unit"]["template"]),
            int(item["unit"]["name_pool"]),
        )].append(item)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    offset = 0
    while len(selected) < count:
        progressed = False
        for key in keys:
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        offset += 1
    return selected


def main() -> None:
    screen = read_json(
        OUT_ROOT / "causal_screen" / MODEL / "summary.json"
    )
    entered_cells = []
    for cell in screen["cell_summaries"]:
        if not cell["adequately_powered"]:
            continue
        if cell["output_type"] == "person":
            reason = "all_adequately_powered_person_cells"
        elif cell["localized_directional_contribution"]:
            reason = "positive_nonperson_8bit_cell"
        else:
            continue
        entered_cells.append({
            "family": cell["family"],
            "output_type": cell["output_type"],
            "operation": cell["operation"],
            "entry_reason": reason,
            "eight_bit_cell": cell,
        })
    if not entered_cells:
        raise RuntimeError("no BF16 audit cells were authorized")

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
    qualifications = read_jsonl(
        OUT_ROOT / "behavior" / MODEL / "pair_qualification.jsonl"
    )
    qualification = {
        (row["unit_id"], row["operation"]): row
        for row in qualifications
    }
    case_by_id = {case["record_id"]: case for case in cases}
    output_root = OUT_ROOT / "precision_audit" / "glm4_bf16"
    started = time.time()
    model = tokenizer = device = None
    all_rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_glm4_bf16()
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        device_map = {
            str(key): str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        }
        print(
            f"[bf16] input_device={device}; "
            f"mapped_components={len(device_map)}",
            flush=True,
        )
        for entered in entered_cells:
            family = entered["family"]
            output_type = entered["output_type"]
            operation = entered["operation"]
            selection = selections[SOURCE_OPERATION[operation]]
            layer = layers[int(selection["layer"]) - 1]
            candidates = []
            for unit in units:
                if (
                    unit["family"] != family
                    or unit["output_type"] != output_type
                    or unit["split"] != "confirmation"
                ):
                    continue
                if not qualification[
                    (unit["unit_id"], operation)
                ]["semantic_pair_qualified"]:
                    continue
                candidates.append({
                    "unit": unit,
                    "base": case_by_id[unit["case_ids"]["base"]],
                    "variant": case_by_id[
                        unit["case_ids"][operation]
                    ],
                })
            items = balanced_select(candidates, BF16_CELL_N)
            if len(items) < BF16_CELL_N:
                raise RuntimeError(
                    f"BF16 authorized cell underfilled after freeze: "
                    f"{family}/{output_type}/{operation} n={len(items)}"
                )
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
            cell_rows = []
            for group in grouped.values():
                for batch_items in chunks(group, BF16_BATCH_SIZE):
                    cell_rows.extend(run_batch(
                        model=model,
                        layer=layer,
                        device=device,
                        head_count=head_count,
                        selection=selection,
                        family=family,
                        operation=operation,
                        items=batch_items,
                    ))
            for row in cell_rows:
                row["schema_version"] = (
                    "phase1010_glm4_bf16_causal_unit.v1"
                )
                row["phase"] = PHASE
                row["precision"] = "bfloat16"
                row["output_type"] = output_type
                row["bf16_entry_reason"] = entered["entry_reason"]
            summary = summarize(
                MODEL,
                family,
                operation,
                cell_rows,
            )
            summary["schema_version"] = (
                "phase1010_glm4_bf16_causal_cell.v1"
            )
            summary["phase"] = PHASE
            summary["precision"] = "bfloat16"
            summary["output_type"] = output_type
            summary["bf16_entry_reason"] = entered["entry_reason"]
            summary["eight_bit_positive"] = bool(
                entered["eight_bit_cell"][
                    "localized_directional_contribution"
                ]
            )
            summary["bf16_gate_agrees_with_8bit"] = bool(
                summary["localized_directional_contribution"]
                == summary["eight_bit_positive"]
            )
            summary["sample_policy"] = (
                "eight deterministic balanced confirmation units selected "
                "without BF16 outcome access"
            )
            all_rows.extend(cell_rows)
            cell_summaries.append(summary)
            print(
                f"[bf16] {family}/{output_type}/{operation} "
                f"n={len(cell_rows)} positive="
                f"{summary['localized_directional_contribution']} "
                f"8bit={summary['eight_bit_positive']}",
                flush=True,
            )

        no_op_pass = all(
            row["maximum_noop_logit_error"] <= 1e-5
            for row in cell_summaries
        )
        if not no_op_pass:
            raise RuntimeError("GLM4 BF16 no-op audit failed")
        positive_nonperson = [
            row
            for row in cell_summaries
            if row["output_type"] != "person"
            and row["localized_directional_contribution"]
        ]
        support: dict[str, set[str]] = defaultdict(set)
        for row in positive_nonperson:
            support[
                f"{row['output_type']}:{row['operation']}"
            ].add(row["family"])
        source_mapping_authorized = any(
            len(families) >= 2 for families in support.values()
        )
        result = {
            "schema_version": "phase1010_glm4_bf16_causal_audit.v1",
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "load_mode": "device_map_auto_cpu_gpu",
            "max_memory": {"cuda:0": "11GiB", "cpu": "18GiB"},
            "input_device": str(device),
            "device_map": device_map,
            "entry_policy": (
                "all adequately powered person cells plus only 8-bit "
                "positive non-person cells"
            ),
            "entered_cell_count": len(entered_cells),
            "cell_summaries": cell_summaries,
            "no_op_audit_pass": no_op_pass,
            "positive_nonperson_cell_ids": [
                (
                    f"{row['family']}:{row['output_type']}:"
                    f"{row['operation']}"
                )
                for row in positive_nonperson
            ],
            "positive_nonperson_family_support": {
                key: sorted(value) for key, value in support.items()
            },
            "upstream_source_mapping_authorized": (
                source_mapping_authorized
            ),
            "authorization_rule": (
                "the same non-person output type and operation retains "
                "the frozen-head local contribution in at least two "
                "held-out families under BF16"
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(
            output_root / "cell_summaries.jsonl",
            cell_summaries,
        )
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
