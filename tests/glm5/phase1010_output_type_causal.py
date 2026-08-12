#!/usr/bin/env python3
"""Audit Phase1008-frozen head sets across Phase1010 output types.

Selection is independent of all Phase1010 responses. Cells with fewer than
eight semantically qualified confirmation pairs are reported as underpowered
and are not used to support output-general claims.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1009_crossfamily_heldout_causal_replication import (
    SOURCE_OPERATION,
    run_batch,
    summarize,
)
from phase1009_crossfamily_response_scan import stage_case
from phase1010_output_type_protocol import (
    FAMILIES,
    OUT_ROOT,
    OUTPUT_TYPES,
    PHASE,
    canonical,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODELS = ("qwen3", "glm4")
OPERATIONS = ("F", "Q")
MIN_CELL_N = 8
PHASE1008_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def run_model(model_name: str) -> dict[str, Any]:
    protocol = read_json(OUT_ROOT / "protocol" / "protocol.json")
    selection_bundle = read_json(
        PHASE1008_ROOT
        / "refinement_final"
        / model_name
        / "causal_selection.json"
    )
    selections = {
        selection["operation"]: selection
        for selection in selection_bundle["selections"]
    }
    if not all(
        selection["selection_pass"]
        and not selection["confirmation_data_used_for_selection"]
        for selection in selections.values()
    ):
        raise RuntimeError(f"{model_name}: frozen selection gate failed")
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "units.jsonl"
    )
    qualification_rows = read_jsonl(
        OUT_ROOT / "behavior" / model_name / "pair_qualification.jsonl"
    )
    qualification = {
        (row["unit_id"], row["operation"]): row
        for row in qualification_rows
    }
    case_by_id = {case["record_id"]: case for case in cases}
    output_root = OUT_ROOT / "causal_screen" / model_name
    started = time.time()
    model = tokenizer = device = None
    all_rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        for output_type in OUTPUT_TYPES:
            for family in FAMILIES:
                for operation in OPERATIONS:
                    selection = selections[SOURCE_OPERATION[operation]]
                    layer = layers[int(selection["layer"]) - 1]
                    items = []
                    for unit in units:
                        if (
                            unit["output_type"] != output_type
                            or unit["family"] != family
                            or unit["split"] != "confirmation"
                        ):
                            continue
                        if not qualification[
                            (unit["unit_id"], operation)
                        ]["semantic_pair_qualified"]:
                            continue
                        items.append({
                            "unit": unit,
                            "base": case_by_id[
                                unit["case_ids"]["base"]
                            ],
                            "variant": case_by_id[
                                unit["case_ids"][operation]
                            ],
                        })
                    if len(items) < MIN_CELL_N:
                        cell_summaries.append({
                            "schema_version": (
                                "phase1010_output_causal_cell.v1"
                            ),
                            "phase": PHASE,
                            "model": model_name,
                            "family": family,
                            "output_type": output_type,
                            "operation": operation,
                            "source_phase": 1008,
                            "source_operation": SOURCE_OPERATION[operation],
                            "selection_independent_of_phase1010": True,
                            "evaluation_split": "confirmation",
                            "n": len(items),
                            "adequately_powered": False,
                            "localized_directional_contribution": False,
                            "status": "underpowered_not_run",
                            "interpretation_limit": (
                                "insufficient behavior-qualified pairs; "
                                "absence of evidence is not a negative "
                                "mechanism result"
                            ),
                        })
                        print(
                            f"[causal-screen] {model_name}/{family}/"
                            f"{output_type}/{operation} underpowered "
                            f"n={len(items)}",
                            flush=True,
                        )
                        continue
                    grouped: dict[
                        tuple[int, int, int],
                        list[dict[str, Any]],
                    ] = defaultdict(list)
                    for item in items:
                        base = stage_case(item["base"], "semantic0")
                        variant = stage_case(
                            item["variant"],
                            "semantic0",
                        )
                        grouped[(
                            int(item["unit"]["template"]),
                            len(base["input_ids"]),
                            len(variant["input_ids"]),
                        )].append(item)
                    cell_rows = []
                    for batch_items in grouped.values():
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
                            "phase1010_output_causal_unit.v1"
                        )
                        row["phase"] = PHASE
                        row["output_type"] = output_type
                        row["selection_independent_of_phase1010"] = True
                    summary = summarize(
                        model_name,
                        family,
                        operation,
                        cell_rows,
                    )
                    summary["schema_version"] = (
                        "phase1010_output_causal_cell.v1"
                    )
                    summary["phase"] = PHASE
                    summary["output_type"] = output_type
                    summary["selection_independent_of_phase1010"] = True
                    summary["adequately_powered"] = True
                    summary["status"] = "measured"
                    cell_summaries.append(summary)
                    all_rows.extend(cell_rows)
                    print(
                        f"[causal-screen] {model_name}/{family}/"
                        f"{output_type}/{operation} n={len(cell_rows)} "
                        f"positive="
                        f"{summary['localized_directional_contribution']}",
                        flush=True,
                    )

        measured = [
            row
            for row in cell_summaries
            if row["adequately_powered"]
        ]
        no_op_pass = all(
            row["maximum_noop_logit_error"] <= 1e-5
            for row in measured
        )
        if not no_op_pass:
            raise RuntimeError(f"{model_name}: no-op audit failed")
        positive = [
            row
            for row in measured
            if row["localized_directional_contribution"]
        ]
        output_operation_support = {}
        for output_type in OUTPUT_TYPES:
            for operation in OPERATIONS:
                cells = [
                    row
                    for row in positive
                    if row["output_type"] == output_type
                    and row["operation"] == operation
                ]
                output_operation_support[
                    f"{output_type}:{operation}"
                ] = sorted({row["family"] for row in cells})
        output_general_operations = []
        person_specific_operations = []
        for operation in OPERATIONS:
            nonperson_support = {
                output_type: output_operation_support[
                    f"{output_type}:{operation}"
                ]
                for output_type in OUTPUT_TYPES
                if output_type != "person"
            }
            if any(
                len(families) >= 2
                for families in nonperson_support.values()
            ):
                output_general_operations.append(operation)
            person_families = output_operation_support[
                f"person:{operation}"
            ]
            adequately_powered_nonperson = [
                row
                for row in measured
                if row["operation"] == operation
                and row["output_type"] != "person"
            ]
            if (
                len(person_families) >= 2
                and adequately_powered_nonperson
                and not any(
                    row["localized_directional_contribution"]
                    for row in adequately_powered_nonperson
                )
            ):
                person_specific_operations.append(operation)
        result = {
            "schema_version": "phase1010_output_causal_model.v1",
            "phase": PHASE,
            "model": model_name,
            "protocol_digest": protocol["preregistration_digest"],
            "source_phase1008_selection_digest": digest(
                selection_bundle
            ),
            "selection_used_phase1010_data": False,
            "evaluation_split": "phase1010_confirmation",
            "cell_count": len(cell_summaries),
            "measured_cell_count": len(measured),
            "underpowered_cell_count": (
                len(cell_summaries) - len(measured)
            ),
            "unit_operation_count": len(all_rows),
            "cell_summaries": cell_summaries,
            "positive_cell_count": len(positive),
            "positive_cell_ids": [
                (
                    f"{row['family']}:{row['output_type']}:"
                    f"{row['operation']}"
                )
                for row in positive
            ],
            "output_operation_family_support": output_operation_support,
            "output_general_operations_by_frozen_gate": (
                output_general_operations
            ),
            "person_specific_operations_by_frozen_gate": (
                person_specific_operations
            ),
            "no_op_audit_pass": no_op_pass,
            "causal_scope": (
                "Phase1008-frozen local head contribution at one layer; "
                "not necessity, sufficiency, path identity, or closure"
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
        return result
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
