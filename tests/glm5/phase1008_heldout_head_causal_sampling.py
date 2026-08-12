#!/usr/bin/env python3
"""Held-out causal sampling of Phase1008 discovery-frozen head groups."""
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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1008_global_response_atlas_protocol import (
    OUT_ROOT,
    PHASE,
    canonical,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1008_global_response_atlas_scan import (
    case_tensors,
    semantic_answer_ids,
    stage_case,
)


MODELS = ("qwen3", "glm4")
EPSILON = 1e-8


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def capture_heads_and_logits(
    model,
    layer,
    staged_cases: list[dict[str, Any]],
    device,
    head_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in staged_cases
        ],
        dtype=torch.long,
        device=device,
    )
    input_ids, attention = case_tensors(staged_cases, device)
    captured: list[torch.Tensor] = []
    count = [0]

    def hook(module, args):
        value = args[0]
        batch = torch.arange(value.shape[0], device=value.device)
        selected = value[batch, positions.to(value.device), :]
        if selected.shape[-1] % head_count:
            raise RuntimeError("attention width/head count drift")
        captured.append(
            selected.reshape(
                selected.shape[0],
                head_count,
                selected.shape[-1] // head_count,
            ).detach()
        )
        count[0] += 1

    handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if count[0] != 1 or len(captured) != 1:
            raise RuntimeError(f"capture count drift: {count[0]}")
        logits = output.logits[:, -1, :].detach()
        return logits, captured[0]
    finally:
        handle.remove()
        del input_ids, attention, positions


def forward_with_head_patch(
    model,
    layer,
    staged_cases: list[dict[str, Any]],
    device,
    source_heads: torch.Tensor,
    head_indices: list[int],
) -> torch.Tensor:
    positions = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in staged_cases
        ],
        dtype=torch.long,
        device=device,
    )
    input_ids, attention = case_tensors(staged_cases, device)
    count = [0]

    def hook(module, args):
        value = args[0]
        patched = value.clone()
        batch = torch.arange(value.shape[0], device=value.device)
        selected = patched[batch, positions.to(value.device), :]
        reshaped = selected.reshape(
            selected.shape[0],
            source_heads.shape[1],
            source_heads.shape[2],
        ).clone()
        reshaped[:, head_indices, :] = source_heads[
            :, head_indices, :
        ].to(device=value.device, dtype=value.dtype)
        patched[batch, positions.to(value.device), :] = reshaped.reshape(
            selected.shape
        )
        count[0] += 1
        return (patched,) + tuple(args[1:])

    handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        if count[0] != 1:
            raise RuntimeError(f"patch count drift: {count[0]}")
        return output.logits[:, -1, :].detach()
    finally:
        handle.remove()
        del input_ids, attention, positions


def candidate_margin(
    logits: torch.Tensor,
    base_cases: list[dict[str, Any]],
    variant_cases: list[dict[str, Any]],
) -> torch.Tensor:
    base_ids = torch.tensor(
        [semantic_answer_ids(case)[0] for case in base_cases],
        dtype=torch.long,
        device=logits.device,
    )
    variant_ids = torch.tensor(
        [semantic_answer_ids(case)[0] for case in variant_cases],
        dtype=torch.long,
        device=logits.device,
    )
    batch = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch, variant_ids] - logits[batch, base_ids]


def finite_fraction(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return numerator / torch.clamp(denominator, min=EPSILON)


def run_batch(
    *,
    model,
    layer,
    device,
    head_count: int,
    selection: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_cases = [stage_case(item["base"], "semantic0") for item in items]
    variant_cases = [
        stage_case(item["variant"], "semantic0") for item in items
    ]
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
    noop_logits = forward_with_head_patch(
        model, layer, base_cases, device, base_heads, selected
    )
    wrong_heads = torch.roll(variant_heads, shifts=1, dims=0)
    wrong_sufficiency_logits = forward_with_head_patch(
        model, layer, base_cases, device, wrong_heads, selected
    )
    margins = {
        "base": candidate_margin(base_logits, base_cases, variant_cases),
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
        "noop": candidate_margin(noop_logits, base_cases, variant_cases),
    }
    natural_effect = margins["variant"] - margins["base"]
    fractions = {
        "selected_sufficiency": finite_fraction(
            margins["selected_sufficiency"] - margins["base"],
            natural_effect,
        ),
        "selected_restore": finite_fraction(
            margins["variant"] - margins["selected_restore"],
            natural_effect,
        ),
        "control_sufficiency": finite_fraction(
            margins["control_sufficiency"] - margins["base"],
            natural_effect,
        ),
        "control_restore": finite_fraction(
            margins["variant"] - margins["control_restore"],
            natural_effect,
        ),
        "all_sufficiency": finite_fraction(
            margins["all_sufficiency"] - margins["base"],
            natural_effect,
        ),
        "all_restore": finite_fraction(
            margins["variant"] - margins["all_restore"],
            natural_effect,
        ),
        "wrong_sufficiency": finite_fraction(
            margins["wrong_sufficiency"] - margins["base"],
            natural_effect,
        ),
    }
    noop_logit_error = torch.max(
        torch.abs(noop_logits - base_logits), dim=-1
    ).values
    rows = []
    for index, item in enumerate(items):
        row = {
            "schema_version": "phase1008_heldout_head_causal_unit.v1",
            "phase": PHASE,
            "model": selection["model"],
            "operation": selection["operation"],
            "unit_id": item["unit"]["unit_id"],
            "split": item["unit"]["split"],
            "template": item["unit"]["template"],
            "name_pool": item["unit"]["name_pool"],
            "layer": int(selection["layer"]),
            "selected_heads": selected,
            "control_heads": controls,
            "base_margin": float(margins["base"][index].item()),
            "variant_margin": float(margins["variant"][index].item()),
            "natural_effect": float(natural_effect[index].item()),
            "noop_max_logit_error": float(
                noop_logit_error[index].item()
            ),
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
            row[f"{name}_fraction"] = float(
                fractions[name][index].item()
            )
        row["selected_sufficiency_flip"] = bool(
            margins["selected_sufficiency"][index] > 0
        )
        row["selected_restore_flip"] = bool(
            margins["selected_restore"][index] < 0
        )
        row["all_sufficiency_flip"] = bool(
            margins["all_sufficiency"][index] > 0
        )
        row["all_restore_flip"] = bool(
            margins["all_restore"][index] < 0
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
        noop_logits,
        wrong_sufficiency_logits,
    )
    return rows


def summarize_rows(
    model_name: str,
    operation: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def median(field: str) -> float:
        return float(np.median([row[field] for row in rows]))

    def mean_bool(field: str) -> float:
        return float(np.mean([row[field] for row in rows]))

    result = {
        "schema_version": "phase1008_heldout_head_causal_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "operation": operation,
        "n": len(rows),
        "median_natural_effect": median("natural_effect"),
        "median_selected_sufficiency_fraction": median(
            "selected_sufficiency_fraction"
        ),
        "median_selected_restore_fraction": median(
            "selected_restore_fraction"
        ),
        "median_control_sufficiency_fraction": median(
            "control_sufficiency_fraction"
        ),
        "median_control_restore_fraction": median(
            "control_restore_fraction"
        ),
        "median_all_sufficiency_fraction": median(
            "all_sufficiency_fraction"
        ),
        "median_all_restore_fraction": median("all_restore_fraction"),
        "median_wrong_sufficiency_fraction": median(
            "wrong_sufficiency_fraction"
        ),
        "selected_minus_control_sufficiency": (
            median("selected_sufficiency_fraction")
            - median("control_sufficiency_fraction")
        ),
        "selected_minus_control_restore": (
            median("selected_restore_fraction")
            - median("control_restore_fraction")
        ),
        "selected_minus_wrong_sufficiency": (
            median("selected_sufficiency_fraction")
            - median("wrong_sufficiency_fraction")
        ),
        "selected_sufficiency_flip_rate": mean_bool(
            "selected_sufficiency_flip"
        ),
        "selected_restore_flip_rate": mean_bool(
            "selected_restore_flip"
        ),
        "all_sufficiency_flip_rate": mean_bool(
            "all_sufficiency_flip"
        ),
        "all_restore_flip_rate": mean_bool("all_restore_flip"),
        "maximum_noop_logit_error": max(
            row["noop_max_logit_error"] for row in rows
        ),
    }
    result["localized_directional_contribution"] = bool(
        result["median_selected_sufficiency_fraction"] > 0
        and result["median_selected_restore_fraction"] > 0
        and result["selected_minus_control_sufficiency"] > 0.05
        and result["selected_minus_control_restore"] > 0.05
    )
    result["interpretation_limit"] = (
        "A positive result establishes only a held-out local causal "
        "contribution at one layer and position. It does not establish a "
        "complete path, necessity, sufficiency, or a closed mechanism."
    )
    return result


def run_model(model_name: str) -> dict[str, Any]:
    atlas_summary = read_json(
        OUT_ROOT / "refinement_final" / "summary.json"
    )
    if atlas_summary["automatic_next_action"] != (
        "heldout_head_causal_sampling_warranted"
    ):
        raise RuntimeError("refinement atlas did not authorize causal sample")
    selection_bundle = read_json(
        OUT_ROOT / "refinement_final" / model_name
        / "causal_selection.json"
    )
    selections = selection_bundle["selections"]
    if not all(selection["selection_pass"] for selection in selections):
        raise RuntimeError(f"{model_name}: causal selection gate failed")
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "refinement_scan" / model_name / "units.jsonl"
    )
    case_by_id = {case["record_id"]: case for case in cases}
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    output_root = OUT_ROOT / "causal_sample" / model_name
    started = time.time()
    model = tokenizer = device = None
    all_rows: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        for selection in selections:
            operation = selection["operation"]
            layer = layers[int(selection["layer"]) - 1]
            items = []
            for unit in units:
                if unit["split"] != "confirmation":
                    continue
                if not unit["semantic_qualified"][operation]:
                    continue
                unit_id = unit["unit_id"]
                items.append({
                    "unit": unit,
                    "base": case_by_id[f"{unit_id}.base"],
                    "variant": case_by_id[f"{unit_id}.{operation}"],
                })
            grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = (
                defaultdict(list)
            )
            for item in items:
                base_staged = stage_case(item["base"], "semantic0")
                variant_staged = stage_case(item["variant"], "semantic0")
                grouped[(
                    int(item["unit"]["template"]),
                    len(base_staged["input_ids"]),
                    len(variant_staged["input_ids"]),
                )].append(item)
            operation_rows = []
            for batch_items in grouped.values():
                operation_rows.extend(run_batch(
                    model=model,
                    layer=layer,
                    device=device,
                    head_count=head_count,
                    selection=selection,
                    items=batch_items,
                ))
            if len(operation_rows) < 8:
                raise RuntimeError(
                    f"{model_name}/{operation}: underfilled confirmation"
                )
            all_rows.extend(operation_rows)
            print(
                f"[causal-sample] {model_name} {operation} "
                f"n={len(operation_rows)}",
                flush=True,
            )
        summaries = [
            summarize_rows(
                model_name,
                operation,
                [
                    row for row in all_rows
                    if row["operation"] == operation
                ],
            )
            for operation in ("B", "Q")
        ]
        no_op_pass = all(
            summary["maximum_noop_logit_error"] <= 1e-5
            for summary in summaries
        )
        if not no_op_pass:
            raise RuntimeError(f"{model_name}: no-op audit failed")
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(output_root / "operation_summaries.jsonl", summaries)
        result = {
            "schema_version": "phase1008_heldout_head_causal_model.v1",
            "phase": PHASE,
            "model": model_name,
            "selection_digest": digest(selection_bundle),
            "selection_used_confirmation_data": False,
            "evaluation_split": "confirmation",
            "unit_operation_count": len(all_rows),
            "operation_summaries": summaries,
            "no_op_audit_pass": no_op_pass,
            "causal_scope": (
                "sampled local contribution only; no path or closure claim"
            ),
            "elapsed_seconds": time.time() - started,
        }
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
