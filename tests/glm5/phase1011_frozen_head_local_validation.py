#!/usr/bin/env python3
"""Test Phase1008-frozen head sets on Phase1011 native tasks.

Selection is independent of Phase1011. The intervention reads a controlled
candidate-token margin at the natural assistant boundary; natural rollout
qualification is retained as a separate sample axis.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from phase1008_heldout_head_causal_sampling import (
    capture_heads_and_logits,
    forward_with_head_patch,
)
from phase1009_crossfamily_response_protocol import canonical
from phase1011_native_semantic_protocol import (
    FAMILIES,
    OUT_ROOT,
    OUTPUT_MODES,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODELS = ("qwen3", "glm4")
OPERATIONS = ("F", "Q")
SOURCE_OPERATION = {"F": "B", "Q": "Q"}
QUALIFICATION_AXES = {
    "semantic_panel": "semantic_pair_qualified",
    "natural_rollout": "rollout_pair_qualified",
    "strict_rollout": "strict_rollout_pair_qualified",
}
PHASE1008_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
)
EPSILON = 1e-8
MIN_CELL_N = 8


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stage_case(case: dict[str, Any]) -> dict[str, Any]:
    row = dict(case)
    row["scan_role_positions"] = {
        "decision_boundary": len(case["input_ids"]) - 1
    }
    return row


def answer_id(case: dict[str, Any]) -> int:
    return int(case["answer_token_ids"][0])


def candidate_margin(
    logits: torch.Tensor,
    base_cases: list[dict[str, Any]],
    variant_cases: list[dict[str, Any]],
) -> torch.Tensor:
    base_ids = torch.tensor(
        [answer_id(case) for case in base_cases],
        dtype=torch.long,
        device=logits.device,
    )
    variant_ids = torch.tensor(
        [answer_id(case) for case in variant_cases],
        dtype=torch.long,
        device=logits.device,
    )
    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, variant_ids] - logits[batch, base_ids]


def signed_fraction(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    result = torch.full_like(numerator, torch.nan)
    valid = torch.abs(denominator) > EPSILON
    result[valid] = numerator[valid] / denominator[valid]
    return result


def run_batch(
    *,
    model,
    layer,
    device,
    head_count: int,
    selection: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_cases = [stage_case(item["base"]) for item in items]
    variant_cases = [stage_case(item["variant"]) for item in items]
    base_logits, base_heads = capture_heads_and_logits(
        model, layer, base_cases, device, head_count
    )
    variant_logits, variant_heads = capture_heads_and_logits(
        model, layer, variant_cases, device, head_count
    )
    selected = [int(value) for value in selection["selected_heads"]]
    controls = [int(value) for value in selection["control_heads"]]
    all_heads = list(range(head_count))
    selected_sufficiency_logits = forward_with_head_patch(
        model, layer, base_cases, device, variant_heads, selected
    )
    selected_restore_logits = forward_with_head_patch(
        model, layer, variant_cases, device, base_heads, selected
    )
    control_sufficiency_logits = forward_with_head_patch(
        model, layer, base_cases, device, variant_heads, controls
    )
    control_restore_logits = forward_with_head_patch(
        model, layer, variant_cases, device, base_heads, controls
    )
    all_sufficiency_logits = forward_with_head_patch(
        model, layer, base_cases, device, variant_heads, all_heads
    )
    all_restore_logits = forward_with_head_patch(
        model, layer, variant_cases, device, base_heads, all_heads
    )
    wrong_heads = torch.roll(variant_heads, shifts=1, dims=0)
    wrong_sufficiency_logits = forward_with_head_patch(
        model, layer, base_cases, device, wrong_heads, selected
    )
    noop_logits = forward_with_head_patch(
        model, layer, base_cases, device, base_heads, selected
    )
    margins = {
        "base": candidate_margin(
            base_logits, base_cases, variant_cases
        ),
        "variant": candidate_margin(
            variant_logits, base_cases, variant_cases
        ),
        "selected_sufficiency": candidate_margin(
            selected_sufficiency_logits, base_cases, variant_cases
        ),
        "selected_restore": candidate_margin(
            selected_restore_logits, base_cases, variant_cases
        ),
        "control_sufficiency": candidate_margin(
            control_sufficiency_logits, base_cases, variant_cases
        ),
        "control_restore": candidate_margin(
            control_restore_logits, base_cases, variant_cases
        ),
        "all_sufficiency": candidate_margin(
            all_sufficiency_logits, base_cases, variant_cases
        ),
        "all_restore": candidate_margin(
            all_restore_logits, base_cases, variant_cases
        ),
        "wrong_sufficiency": candidate_margin(
            wrong_sufficiency_logits, base_cases, variant_cases
        ),
        "noop": candidate_margin(
            noop_logits, base_cases, variant_cases
        ),
    }
    natural_effect = margins["variant"] - margins["base"]
    fractions = {
        "selected_sufficiency": signed_fraction(
            margins["selected_sufficiency"] - margins["base"],
            natural_effect,
        ),
        "selected_restore": signed_fraction(
            margins["variant"] - margins["selected_restore"],
            natural_effect,
        ),
        "control_sufficiency": signed_fraction(
            margins["control_sufficiency"] - margins["base"],
            natural_effect,
        ),
        "control_restore": signed_fraction(
            margins["variant"] - margins["control_restore"],
            natural_effect,
        ),
        "all_sufficiency": signed_fraction(
            margins["all_sufficiency"] - margins["base"],
            natural_effect,
        ),
        "all_restore": signed_fraction(
            margins["variant"] - margins["all_restore"],
            natural_effect,
        ),
        "wrong_sufficiency": signed_fraction(
            margins["wrong_sufficiency"] - margins["base"],
            natural_effect,
        ),
    }
    noop_error = torch.max(
        torch.abs(noop_logits - base_logits), dim=-1
    ).values
    rows = []
    for index, item in enumerate(items):
        qualification = item["qualification"]
        row = {
            "schema_version": "phase1011_frozen_head_unit.v1",
            "phase": PHASE,
            "model": selection["model"],
            "family": item["unit"]["family"],
            "output_mode": item["unit"]["output_mode"],
            "operation": item["operation"],
            "source_phase": 1008,
            "source_operation": selection["operation"],
            "selection_split": selection["selection_split"],
            "evaluation_split": "phase1011_confirmation",
            "selection_used_phase1011_data": False,
            "unit_id": item["unit"]["unit_id"],
            "template": int(item["unit"]["template"]),
            "name_pool": int(item["unit"]["name_pool"]),
            "world_index": int(item["unit"]["world_index"]),
            "layer": int(selection["layer"]),
            "selected_heads": selected,
            "control_heads": controls,
            "semantic_panel_qualified": bool(
                qualification["semantic_pair_qualified"]
            ),
            "natural_rollout_qualified": bool(
                qualification["rollout_pair_qualified"]
            ),
            "strict_rollout_qualified": bool(
                qualification["strict_rollout_pair_qualified"]
            ),
            "base_margin": float(margins["base"][index].item()),
            "variant_margin": float(margins["variant"][index].item()),
            "natural_effect": float(natural_effect[index].item()),
            "noop_max_logit_error": float(noop_error[index].item()),
        }
        for name in (
            "selected_sufficiency",
            "selected_restore",
            "control_sufficiency",
            "control_restore",
            "all_sufficiency",
            "all_restore",
            "wrong_sufficiency",
        ):
            row[f"{name}_margin"] = float(margins[name][index].item())
            value = float(fractions[name][index].item())
            row[f"{name}_fraction"] = (
                value if math.isfinite(value) else None
            )
        row["selected_sufficiency_flip"] = bool(
            margins["selected_sufficiency"][index] > 0
        )
        row["selected_restore_flip"] = bool(
            margins["selected_restore"][index] < 0
        )
        rows.append(row)
    del (
        base_logits,
        variant_logits,
        base_heads,
        variant_heads,
        selected_sufficiency_logits,
        selected_restore_logits,
        control_sufficiency_logits,
        control_restore_logits,
        all_sufficiency_logits,
        all_restore_logits,
        wrong_sufficiency_logits,
        noop_logits,
    )
    return rows


def summarize_subset(
    *,
    model_name: str,
    family: str,
    output_mode: str,
    operation: str,
    qualification_axis: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    field = f"{qualification_axis}_qualified"
    subset = [row for row in rows if row[field]]

    def finite_values(name: str) -> np.ndarray:
        return np.asarray(
            [
                float(row[name])
                for row in subset
                if row[name] is not None
                and math.isfinite(float(row[name]))
            ],
            dtype=np.float64,
        )

    def median(name: str) -> float | None:
        values = finite_values(name)
        return None if not values.size else float(np.median(values))

    def paired_delta(left: str, right: str) -> np.ndarray:
        result = []
        for row in subset:
            if row[left] is None or row[right] is None:
                continue
            left_value = float(row[left])
            right_value = float(row[right])
            if math.isfinite(left_value) and math.isfinite(right_value):
                result.append(left_value - right_value)
        return np.asarray(result, dtype=np.float64)

    suff_control = paired_delta(
        "selected_sufficiency_fraction",
        "control_sufficiency_fraction",
    )
    restore_control = paired_delta(
        "selected_restore_fraction",
        "control_restore_fraction",
    )
    suff_wrong = paired_delta(
        "selected_sufficiency_fraction",
        "wrong_sufficiency_fraction",
    )
    sufficient_n = len(subset) >= MIN_CELL_N
    median_suff = median("selected_sufficiency_fraction")
    median_restore = median("selected_restore_fraction")
    paired_suff = (
        None if not suff_control.size
        else float(np.median(suff_control))
    )
    paired_restore = (
        None if not restore_control.size
        else float(np.median(restore_control))
    )
    directional_flag = bool(
        sufficient_n
        and median_suff is not None
        and median_restore is not None
        and paired_suff is not None
        and paired_restore is not None
        and median_suff > 0
        and median_restore > 0
        and paired_suff > 0.05
        and paired_restore > 0.05
    )
    return {
        "schema_version": "phase1011_frozen_head_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "family": family,
        "output_mode": output_mode,
        "operation": operation,
        "qualification_axis": qualification_axis,
        "n": len(subset),
        "minimum_n_met": sufficient_n,
        "median_natural_effect": median("natural_effect"),
        "median_selected_sufficiency_fraction": median_suff,
        "median_selected_restore_fraction": median_restore,
        "median_control_sufficiency_fraction": median(
            "control_sufficiency_fraction"
        ),
        "median_control_restore_fraction": median(
            "control_restore_fraction"
        ),
        "median_all_sufficiency_fraction": median(
            "all_sufficiency_fraction"
        ),
        "median_all_restore_fraction": median(
            "all_restore_fraction"
        ),
        "median_wrong_sufficiency_fraction": median(
            "wrong_sufficiency_fraction"
        ),
        "median_paired_selected_minus_control_sufficiency": (
            paired_suff
        ),
        "median_paired_selected_minus_control_restore": paired_restore,
        "median_paired_selected_minus_wrong_sufficiency": (
            None if not suff_wrong.size
            else float(np.median(suff_wrong))
        ),
        "selected_sufficiency_flip_rate": (
            None if not subset
            else float(np.mean([
                row["selected_sufficiency_flip"] for row in subset
            ]))
        ),
        "selected_restore_flip_rate": (
            None if not subset
            else float(np.mean([
                row["selected_restore_flip"] for row in subset
            ]))
        ),
        "maximum_noop_logit_error": (
            None if not subset
            else float(max(
                row["noop_max_logit_error"] for row in subset
            ))
        ),
        "descriptive_directional_replication_flag": directional_flag,
        "flag_status": (
            "descriptive_gate_not_mechanism_or_closure"
        ),
    }


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
    output_root = OUT_ROOT / "causal_frozen_heads" / model_name
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                for operation in OPERATIONS:
                    selection = selections[SOURCE_OPERATION[operation]]
                    layer = layers[int(selection["layer"]) - 1]
                    items = []
                    for unit in units:
                        if (
                            unit["family"] != family
                            or unit["output_mode"] != output_mode
                            or unit["split"] != "confirmation"
                        ):
                            continue
                        pair_qualification = qualification[
                            (unit["unit_id"], operation)
                        ]
                        if not any(
                            pair_qualification[field]
                            for field in QUALIFICATION_AXES.values()
                        ):
                            continue
                        items.append({
                            "unit": unit,
                            "operation": operation,
                            "qualification": pair_qualification,
                            "base": case_by_id[
                                unit["case_ids"]["base"]
                            ],
                            "variant": case_by_id[
                                unit["case_ids"][operation]
                            ],
                        })
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
                    for batch_items in grouped.values():
                        cell_rows.extend(run_batch(
                            model=model,
                            layer=layer,
                            device=device,
                            head_count=head_count,
                            selection=selection,
                            items=batch_items,
                        ))
                    all_rows.extend(cell_rows)
                    print(
                        f"[frozen-head] {model_name}/{family}/"
                        f"{output_mode}/{operation} n={len(cell_rows)}",
                        flush=True,
                    )
        summaries = []
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                for operation in OPERATIONS:
                    cell_rows = [
                        row for row in all_rows
                        if row["family"] == family
                        and row["output_mode"] == output_mode
                        and row["operation"] == operation
                    ]
                    for axis in QUALIFICATION_AXES:
                        summaries.append(summarize_subset(
                            model_name=model_name,
                            family=family,
                            output_mode=output_mode,
                            operation=operation,
                            qualification_axis=axis,
                            rows=cell_rows,
                        ))
        no_op_maximum = max(
            row["noop_max_logit_error"] for row in all_rows
        )
        if no_op_maximum > 1e-5:
            raise RuntimeError(
                f"{model_name}: no-op audit failed {no_op_maximum}"
            )
        positive = [
            row for row in summaries
            if row["descriptive_directional_replication_flag"]
        ]
        result = {
            "schema_version": "phase1011_frozen_head_model.v1",
            "phase": PHASE,
            "model": model_name,
            "protocol_digest": protocol["preregistration_digest"],
            "source_phase1008_selection_digest": digest(selection_bundle),
            "selection_used_phase1011_data": False,
            "selection_surface": (
                "Phase1008 explicit-map tasks; evaluated here at native "
                "assistant boundary without an explicit response map"
            ),
            "unit_operation_count": len(all_rows),
            "cell_count": len(summaries),
            "qualified_cell_count": int(sum(
                row["minimum_n_met"] for row in summaries
            )),
            "descriptive_positive_cell_count": len(positive),
            "descriptive_positive_cells": [{
                key: row[key]
                for key in (
                    "family",
                    "output_mode",
                    "operation",
                    "qualification_axis",
                    "n",
                )
            } for row in positive],
            "maximum_noop_logit_error": no_op_maximum,
            "no_op_audit_pass": True,
            "cell_summaries": summaries,
            "elapsed_seconds": time.time() - started,
            "causal_scope": (
                "independent local directional intervention on a "
                "controlled token margin; no natural-rollout intervention, "
                "shared mechanism, necessity, sufficiency, or closure claim"
            ),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(
            output_root / "cell_summaries.jsonl", summaries
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
