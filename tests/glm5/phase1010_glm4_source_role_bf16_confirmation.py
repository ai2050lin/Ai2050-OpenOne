#!/usr/bin/env python3
"""BF16 confirmation of discovery-frozen GLM4 source roles."""
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

from model_utils import get_layers
from phase1009_crossfamily_response_scan import stage_case
from phase1010_glm4_bf16_causal_audit import (
    BF16_BATCH_SIZE,
    balanced_select,
    chunks,
    load_glm4_bf16,
)
from phase1010_glm4_source_role_discovery import (
    FAMILIES,
    MODEL,
    OPERATION,
    OUTPUT_TYPE,
    PHASE1008_ROOT,
    run_group,
    summarize_cell,
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
    discovery_root = OUT_ROOT / "source_role_mapping" / "discovery"
    discovery = read_json(discovery_root / "summary.json")
    role_selection = read_json(discovery_root / "selection.json")
    confirmation_roles = tuple(role_selection["confirmation_roles"])
    if not confirmation_roles:
        raise RuntimeError("no discovery-frozen roles for confirmation")
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
    if selected_heads != role_selection["selected_heads"]:
        raise RuntimeError("source-role head selection drift")
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
    output_root = OUT_ROOT / "source_role_mapping" / "bf16_confirmation"
    started = time.time()
    model = tokenizer = device = None
    all_rows: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    reconstruction_errors = []
    try:
        model, tokenizer, device = load_glm4_bf16()
        layers = get_layers(model)
        layer = layers[int(selection["layer"]) - 1]
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
                    f"{family}: confirmation underfilled n={len(items)}"
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
            family_rows = []
            family_errors = []
            for group in grouped.values():
                for batch in chunks(group, BF16_BATCH_SIZE):
                    rows, error = run_group(
                        model=model,
                        layer=layer,
                        device=device,
                        selected_heads=selected_heads,
                        items=batch,
                        roles=confirmation_roles,
                    )
                    for row in rows:
                        row["schema_version"] = (
                            "phase1010_source_role_bf16_unit.v1"
                        )
                        row["split"] = "confirmation"
                        row["precision"] = "bfloat16"
                    family_rows.extend(rows)
                    family_errors.append(error)
            family_cells = summarize_cell(
                family,
                family_rows,
                max(family_errors),
                roles=confirmation_roles,
                split="confirmation",
            )
            for row in family_cells:
                row["schema_version"] = (
                    "phase1010_source_role_bf16_cell.v1"
                )
                row["precision"] = "bfloat16"
            all_rows.extend(family_rows)
            all_cells.extend(family_cells)
            reconstruction_errors.extend(family_errors)
            print(
                f"[source-bf16] {family} n={len(items)} roles="
                f"{len(confirmation_roles)} reconstruction="
                f"{max(family_errors):.6g}",
                flush=True,
            )

        by_key = {
            (row["family"], row["source_role"]): row
            for row in all_cells
        }
        selected_role_support = {}
        for role in role_selection["selected_atomic_roles"]:
            selected_role_support[role] = {
                family: {
                    "sufficiency": by_key[
                        (family, role)
                    ]["median_sufficiency_fraction"],
                    "restore": by_key[
                        (family, role)
                    ]["median_restore_fraction"],
                    "shuffled": by_key[
                        (family, role)
                    ]["median_shuffled_sufficiency_fraction"],
                    "delta_norm_ratio": by_key[
                        (family, role)
                    ]["median_role_to_full_delta_norm_ratio"],
                }
                for family in FAMILIES
            }
        response_role = "response_map_instruction"
        response_repeats = bool(
            response_role in role_selection["selected_atomic_roles"]
            and all(
                by_key[(family, response_role)][
                    "median_sufficiency_fraction"
                ] > 0
                and by_key[(family, response_role)][
                    "median_restore_fraction"
                ] > 0
                and by_key[(family, response_role)][
                    "median_sufficiency_minus_shuffled"
                ] > 0
                for family in FAMILIES
            )
        )
        norm_comparisons = {
            family: {
                "response_map_instruction": by_key[
                    (family, response_role)
                ]["median_role_to_full_delta_norm_ratio"],
                "task_semantics": by_key[
                    (family, "task_semantics")
                ]["median_role_to_full_delta_norm_ratio"],
            }
            for family in FAMILIES
        }
        summary = {
            "schema_version": (
                "phase1010_source_role_bf16_confirmation.v1"
            ),
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "split": "confirmation",
            "source_discovery_summary": discovery,
            "layer": int(selection["layer"]),
            "selected_heads": selected_heads,
            "output_type": OUTPUT_TYPE,
            "operation": OPERATION,
            "families": list(FAMILIES),
            "confirmation_n_per_family": CONFIRMATION_N,
            "confirmation_roles": list(confirmation_roles),
            "selected_role_support": selected_role_support,
            "response_map_instruction_repeats": response_repeats,
            "role_delta_norm_comparison": norm_comparisons,
            "maximum_attention_reconstruction_error": max(
                reconstruction_errors
            ),
            "maximum_noop_logit_error": max(
                row["noop_max_logit_error"] for row in all_rows
            ),
            "device_map": device_map,
            "elapsed_seconds": time.time() - started,
            "strongest_supported_statement": (
                "For code-label F interventions in two held-out families, "
                "the discovery-frozen L30 head group receives most of its "
                "changed direct attention contribution from fixed trailing "
                "response-map/instruction positions whose contextual states "
                "have changed. This is a local relay observation, not the "
                "origin of the language rule."
            ),
            "automatic_next_step": (
                "trace_when_and_how_semantic_information_is_written_into_"
                "the_fixed_trailing_relay_positions"
                if response_repeats
                else "do_not_continue_upstream_from_this_role"
            ),
        }
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(output_root / "cell_summaries.jsonl", all_cells)
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            del model
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
