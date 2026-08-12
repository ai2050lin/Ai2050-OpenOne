#!/usr/bin/env python3
"""Replicate Phase1008-frozen head sets across Phase1009 language families."""
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
from phase1008_heldout_head_causal_sampling import (
    capture_heads_and_logits,
    forward_with_head_patch,
)
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    OUT_ROOT,
    PHASE,
    canonical,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from phase1009_crossfamily_response_scan import stage_case


MODELS = ("qwen3", "glm4")
OPERATIONS = ("F", "Q")
SOURCE_OPERATION = {"F": "B", "Q": "Q"}
EPSILON = 1e-8
PHASE1008_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def semantic_id(case: dict[str, Any]) -> int:
    return int(case["answer_token_ids"][int(case["semantic_step"])])


def candidate_margin(
    logits: torch.Tensor,
    base_cases: list[dict[str, Any]],
    variant_cases: list[dict[str, Any]],
) -> torch.Tensor:
    base_ids = torch.tensor(
        [semantic_id(case) for case in base_cases],
        dtype=torch.long,
        device=logits.device,
    )
    variant_ids = torch.tensor(
        [semantic_id(case) for case in variant_cases],
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
    family: str,
    operation: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_cases = [stage_case(item["base"], "semantic0") for item in items]
    variant_cases = [
        stage_case(item["variant"], "semantic0") for item in items
    ]
    base_logits, base_heads = capture_heads_and_logits(
        model,
        layer,
        base_cases,
        device,
        head_count,
    )
    variant_logits, variant_heads = capture_heads_and_logits(
        model,
        layer,
        variant_cases,
        device,
        head_count,
    )
    selected = [int(value) for value in selection["selected_heads"]]
    controls = [int(value) for value in selection["control_heads"]]
    all_heads = list(range(head_count))
    selected_sufficiency_logits = forward_with_head_patch(
        model,
        layer,
        base_cases,
        device,
        variant_heads,
        selected,
    )
    selected_restore_logits = forward_with_head_patch(
        model,
        layer,
        variant_cases,
        device,
        base_heads,
        selected,
    )
    control_sufficiency_logits = forward_with_head_patch(
        model,
        layer,
        base_cases,
        device,
        variant_heads,
        controls,
    )
    control_restore_logits = forward_with_head_patch(
        model,
        layer,
        variant_cases,
        device,
        base_heads,
        controls,
    )
    all_sufficiency_logits = forward_with_head_patch(
        model,
        layer,
        base_cases,
        device,
        variant_heads,
        all_heads,
    )
    all_restore_logits = forward_with_head_patch(
        model,
        layer,
        variant_cases,
        device,
        base_heads,
        all_heads,
    )
    noop_logits = forward_with_head_patch(
        model,
        layer,
        base_cases,
        device,
        base_heads,
        selected,
    )
    wrong_heads = torch.roll(variant_heads, shifts=1, dims=0)
    wrong_sufficiency_logits = forward_with_head_patch(
        model,
        layer,
        base_cases,
        device,
        wrong_heads,
        selected,
    )
    margins = {
        "base": candidate_margin(base_logits, base_cases, variant_cases),
        "variant": candidate_margin(
            variant_logits,
            base_cases,
            variant_cases,
        ),
        "selected_sufficiency": candidate_margin(
            selected_sufficiency_logits,
            base_cases,
            variant_cases,
        ),
        "selected_restore": candidate_margin(
            selected_restore_logits,
            base_cases,
            variant_cases,
        ),
        "control_sufficiency": candidate_margin(
            control_sufficiency_logits,
            base_cases,
            variant_cases,
        ),
        "control_restore": candidate_margin(
            control_restore_logits,
            base_cases,
            variant_cases,
        ),
        "all_sufficiency": candidate_margin(
            all_sufficiency_logits,
            base_cases,
            variant_cases,
        ),
        "all_restore": candidate_margin(
            all_restore_logits,
            base_cases,
            variant_cases,
        ),
        "wrong_sufficiency": candidate_margin(
            wrong_sufficiency_logits,
            base_cases,
            variant_cases,
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
    noop_error = torch.max(
        torch.abs(noop_logits - base_logits),
        dim=-1,
    ).values
    rows = []
    for index, item in enumerate(items):
        row = {
            "schema_version": "phase1009_crossfamily_causal_unit.v1",
            "phase": PHASE,
            "model": selection["model"],
            "family": family,
            "operation": operation,
            "source_phase": 1008,
            "source_operation": selection["operation"],
            "selection_split": selection["selection_split"],
            "evaluation_split": "confirmation",
            "unit_id": item["unit"]["unit_id"],
            "template": int(item["unit"]["template"]),
            "name_pool": int(item["unit"]["name_pool"]),
            "layer": int(selection["layer"]),
            "selected_heads": selected,
            "control_heads": controls,
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
            row[f"{name}_fraction"] = float(
                fractions[name][index].item()
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


def summarize(
    model_name: str,
    family: str,
    operation: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def median(field: str) -> float:
        return float(np.median([row[field] for row in rows]))

    def mean_bool(field: str) -> float:
        return float(np.mean([row[field] for row in rows]))

    selected_control_sufficiency = np.array([
        row["selected_sufficiency_fraction"]
        - row["control_sufficiency_fraction"]
        for row in rows
    ])
    selected_control_restore = np.array([
        row["selected_restore_fraction"]
        - row["control_restore_fraction"]
        for row in rows
    ])
    selected_wrong_sufficiency = np.array([
        row["selected_sufficiency_fraction"]
        - row["wrong_sufficiency_fraction"]
        for row in rows
    ])
    result = {
        "schema_version": "phase1009_crossfamily_causal_cell.v1",
        "phase": PHASE,
        "model": model_name,
        "family": family,
        "operation": operation,
        "source_phase": 1008,
        "source_operation": SOURCE_OPERATION[operation],
        "selection_independent_of_phase1009": True,
        "evaluation_split": "confirmation",
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
        "median_all_restore_fraction": median(
            "all_restore_fraction"
        ),
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
        "median_paired_selected_minus_control_sufficiency": float(
            np.median(selected_control_sufficiency)
        ),
        "median_paired_selected_minus_control_restore": float(
            np.median(selected_control_restore)
        ),
        "median_paired_selected_minus_wrong_sufficiency": float(
            np.median(selected_wrong_sufficiency)
        ),
        "paired_selected_above_control_sufficiency_rate": float(
            np.mean(selected_control_sufficiency > 0)
        ),
        "paired_selected_above_control_restore_rate": float(
            np.mean(selected_control_restore > 0)
        ),
        "paired_selected_above_wrong_sufficiency_rate": float(
            np.mean(selected_wrong_sufficiency > 0)
        ),
        "selected_sufficiency_flip_rate": mean_bool(
            "selected_sufficiency_flip"
        ),
        "selected_restore_flip_rate": mean_bool(
            "selected_restore_flip"
        ),
        "maximum_noop_logit_error": max(
            row["noop_max_logit_error"] for row in rows
        ),
    }
    result["localized_directional_contribution"] = bool(
        result["median_selected_sufficiency_fraction"] > 0
        and result["median_selected_restore_fraction"] > 0
        and result[
            "median_paired_selected_minus_control_sufficiency"
        ] > 0.05
        and result[
            "median_paired_selected_minus_control_restore"
        ] > 0.05
    )
    result["interpretation_limit"] = (
        "A positive cell means a Phase1008-frozen head set has a held-out "
        "local directional contribution in another language family. It "
        "does not prove a shared complete mechanism, necessity, transport, "
        "or natural-rollout closure."
    )
    return result


def run_model(model_name: str) -> dict[str, Any]:
    atlas = read_json(OUT_ROOT / "final" / "summary.json")
    if not atlas["automatic_next_step_rule"]["eligible"]:
        raise RuntimeError("Phase1009 atlas did not authorize causal sampling")
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
    output_root = OUT_ROOT / "causal_replication" / model_name
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        for family in FAMILIES:
            for operation in OPERATIONS:
                selection = selections[SOURCE_OPERATION[operation]]
                layer = layers[int(selection["layer"]) - 1]
                items = []
                for unit in units:
                    if (
                        unit["family"] != family
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
                if len(cell_rows) < 8:
                    raise RuntimeError(
                        f"{model_name}/{family}/{operation}: "
                        f"underfilled confirmation n={len(cell_rows)}"
                    )
                all_rows.extend(cell_rows)
                print(
                    f"[causal-replication] {model_name}/{family}/"
                    f"{operation} n={len(cell_rows)}",
                    flush=True,
                )
        summaries = [
            summarize(
                model_name,
                family,
                operation,
                [
                    row for row in all_rows
                    if row["family"] == family
                    and row["operation"] == operation
                ],
            )
            for family in FAMILIES
            for operation in OPERATIONS
        ]
        no_op_pass = all(
            row["maximum_noop_logit_error"] <= 1e-5
            for row in summaries
        )
        if not no_op_pass:
            raise RuntimeError(f"{model_name}: no-op audit failed")
        positive_cells = [
            row for row in summaries
            if row["localized_directional_contribution"]
        ]
        positive_families = sorted({
            row["family"] for row in positive_cells
        })
        result = {
            "schema_version": "phase1009_crossfamily_causal_model.v1",
            "phase": PHASE,
            "model": model_name,
            "source_phase1008_selection_digest": digest(selection_bundle),
            "selection_used_phase1009_data": False,
            "evaluation_split": "phase1009_confirmation",
            "unit_operation_count": len(all_rows),
            "cell_summaries": summaries,
            "positive_cell_count": len(positive_cells),
            "positive_families": positive_families,
            "cross_family_local_replication": bool(
                len(positive_families) >= 2
            ),
            "no_op_audit_pass": no_op_pass,
            "causal_scope": (
                "independent local directional replication only; no shared "
                "path, mechanism, necessity, sufficiency, or closure claim"
            ),
            "elapsed_seconds": time.time() - started,
        }
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(output_root / "cell_summaries.jsonl", summaries)
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
