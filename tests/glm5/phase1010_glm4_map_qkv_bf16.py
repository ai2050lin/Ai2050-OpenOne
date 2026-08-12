#!/usr/bin/env python3
"""BF16 QK/V/interaction decomposition of the confirmed map relay."""
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
from phase1009_crossfamily_response_scan import case_tensors, stage_case
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
    forward_with_selected_replacement,
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


TARGET_LAYER = 30
MAP_OFFSETS = tuple(range(0, 27))
COMPONENTS = ("qk_routing", "value_content", "interaction", "all")
CONFIRMATION_N = 8
EPSILON = 1e-8


def capture_attention_detail(
    *,
    model,
    layer,
    cases,
    originals,
    device,
    selected_heads,
):
    input_ids, attention_mask = case_tensors(cases, device)
    decision = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in cases
        ],
        dtype=torch.long,
        device=device,
    )
    head_count = int(model.config.num_attention_heads)
    kv_count = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    captured = {}
    counts = defaultdict(int)

    def value_hook(module, args, output):
        captured["values"] = output.detach().reshape(
            output.shape[0],
            output.shape[1],
            kv_count,
            head_dim,
        )
        counts["values"] += 1

    def head_hook(module, args):
        value = args[0]
        batch = torch.arange(value.shape[0], device=value.device)
        captured["heads"] = value[
            batch,
            decision.to(value.device),
            :,
        ].detach().reshape(value.shape[0], head_count, head_dim)
        counts["heads"] += 1

    def attention_hook(module, args, output):
        if (
            not isinstance(output, tuple)
            or len(output) < 2
            or output[1] is None
        ):
            raise RuntimeError("attention weights unavailable")
        weights = output[1]
        batch = torch.arange(weights.shape[0], device=weights.device)
        captured["weights"] = weights[
            batch,
            :,
            decision.to(weights.device),
            :,
        ].detach()
        counts["weights"] += 1

    handles = [
        layer.self_attn.v_proj.register_forward_hook(value_hook),
        layer.self_attn.o_proj.register_forward_pre_hook(head_hook),
        layer.self_attn.register_forward_hook(attention_hook),
    ]
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        if any(counts[key] != 1 for key in ("values", "heads", "weights")):
            raise RuntimeError(f"QKV capture drift {dict(counts)}")
        values = captured["values"].repeat_interleave(
            head_count // kv_count,
            dim=2,
        )[:, :, selected_heads, :]
        weights = captured["weights"][:, selected_heads, :]
        heads = captured["heads"][:, selected_heads, :]
        map_positions = []
        for original, staged in zip(originals, cases):
            relay = source_partition(
                original,
                staged,
            )["response_map_instruction"]
            map_positions.append([relay[offset] for offset in MAP_OFFSETS])
        return (
            output.logits[:, -1, :].detach(),
            heads,
            weights,
            values,
            map_positions,
        )
    finally:
        for handle in reversed(handles):
            handle.remove()
        del input_ids, attention_mask, decision


def component_vectors(
    *,
    base_cases,
    variant_cases,
    base_weights,
    variant_weights,
    base_values,
    variant_values,
    base_positions,
    variant_positions,
):
    result = {
        component: torch.zeros(
            base_values.shape[0],
            base_values.shape[2],
            base_values.shape[3],
            dtype=torch.float32,
            device=base_values.device,
        )
        for component in COMPONENTS
    }
    identity_errors = []
    attention_mass_rows = []
    for index in range(len(base_cases)):
        bpos = torch.tensor(
            base_positions[index],
            dtype=torch.long,
            device=base_weights.device,
        )
        vpos = torch.tensor(
            variant_positions[index],
            dtype=torch.long,
            device=variant_weights.device,
        )
        base_tokens = torch.tensor(
            [base_cases[index]["input_ids"][position] for position in base_positions[index]],
            dtype=torch.long,
        )
        variant_tokens = torch.tensor(
            [
                variant_cases[index]["input_ids"][position]
                for position in variant_positions[index]
            ],
            dtype=torch.long,
        )
        if not torch.equal(base_tokens, variant_tokens):
            raise RuntimeError("map token surface drift within F pair")
        a0 = base_weights[index].index_select(1, bpos).float()
        a1 = variant_weights[index].index_select(1, vpos).float()
        v0 = base_values[index].index_select(0, bpos).permute(
            1, 0, 2
        ).float()
        v1 = variant_values[index].index_select(0, vpos).permute(
            1, 0, 2
        ).float()
        delta_a = a1 - a0
        delta_v = v1 - v0
        qk = (delta_a[:, :, None] * v0).sum(dim=1)
        value = (a0[:, :, None] * delta_v).sum(dim=1)
        interaction = (delta_a[:, :, None] * delta_v).sum(dim=1)
        all_value = qk + value + interaction
        direct = (
            a1[:, :, None] * v1 - a0[:, :, None] * v0
        ).sum(dim=1)
        identity_errors.append(float(
            torch.max(torch.abs(all_value - direct)).item()
        ))
        result["qk_routing"][index] = qk.to(result["qk_routing"].device)
        result["value_content"][index] = value.to(
            result["value_content"].device
        )
        result["interaction"][index] = interaction.to(
            result["interaction"].device
        )
        result["all"][index] = all_value.to(result["all"].device)
        attention_mass_rows.append({
            "base": a0.sum(dim=1),
            "variant": a1.sum(dim=1),
        })
    return result, max(identity_errors), attention_mass_rows


def run_batch(
    *,
    model,
    layer,
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
    (
        base_logits,
        base_heads,
        base_weights,
        base_values,
        base_positions,
    ) = capture_attention_detail(
        model=model,
        layer=layer,
        cases=base_cases,
        originals=base_original,
        device=device,
        selected_heads=selected_heads,
    )
    (
        variant_logits,
        variant_heads,
        variant_weights,
        variant_values,
        variant_positions,
    ) = capture_attention_detail(
        model=model,
        layer=layer,
        cases=variant_cases,
        originals=variant_original,
        device=device,
        selected_heads=selected_heads,
    )
    vectors, identity_error, attention_mass = component_vectors(
        base_cases=base_cases,
        variant_cases=variant_cases,
        base_weights=base_weights,
        variant_weights=variant_weights,
        base_values=base_values,
        variant_values=variant_values,
        base_positions=base_positions,
        variant_positions=variant_positions,
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
    for component in COMPONENTS:
        vector = vectors[component].to(base_heads.device)
        suff_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=base_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=base_heads + vector,
        )
        restore_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=variant_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=variant_heads - vector,
        )
        shuffled_vector = torch.roll(vector, shifts=1, dims=0)
        shuffled_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=base_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=base_heads + shuffled_vector,
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
        vector_norm = torch.linalg.vector_norm(
            vector.float().reshape(len(items), -1),
            dim=-1,
        )
        all_norm = torch.linalg.vector_norm(
            vectors["all"].float().reshape(len(items), -1),
            dim=-1,
        )
        for index, item in enumerate(items):
            rows.append({
                "schema_version": "phase1010_map_qkv_unit.v1",
                "phase": PHASE,
                "model": MODEL,
                "precision": "bfloat16",
                "split": "confirmation",
                "family": item["unit"]["family"],
                "output_type": OUTPUT_TYPE,
                "operation": OPERATION,
                "unit_id": item["unit"]["unit_id"],
                "layer": TARGET_LAYER,
                "selected_heads": selected_heads,
                "source_region": "response_map_declaration",
                "component": component,
                "component_norm": float(vector_norm[index].item()),
                "all_component_norm": float(all_norm[index].item()),
                "component_to_all_norm_ratio": float(
                    vector_norm[index].item()
                    / max(all_norm[index].item(), EPSILON)
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
                "base_attention_mass": [
                    float(value)
                    for value in attention_mass[index]["base"].tolist()
                ],
                "variant_attention_mass": [
                    float(value)
                    for value in attention_mass[index]["variant"].tolist()
                ],
                "qkv_identity_max_error_in_batch": identity_error,
            })
    return rows, identity_error


def main() -> None:
    subregion = read_json(
        OUT_ROOT
        / "relay_subregion_mapping"
        / "bf16"
        / "summary.json"
    )
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
    output_root = OUT_ROOT / "map_qkv_decomposition" / "bf16"
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
    identity_errors = []
    try:
        model, tokenizer, device = load_glm4_bf16()
        layers = get_layers(model)
        layer = layers[TARGET_LAYER - 1]
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
                    rows, error = run_batch(
                        model=model,
                        layer=layer,
                        device=device,
                        selected_heads=selected_heads,
                        items=batch,
                    )
                    all_rows.extend(rows)
                    identity_errors.append(error)
            print(
                f"[map-qkv] {family} n={len(items)}",
                flush=True,
            )
        summaries = []
        for family in FAMILIES:
            for component in COMPONENTS:
                selected = [
                    row
                    for row in all_rows
                    if row["family"] == family
                    and row["component"] == component
                ]

                def median(field):
                    return float(np.median([
                        row[field] for row in selected
                    ]))

                summaries.append({
                    "schema_version": "phase1010_map_qkv_cell.v1",
                    "phase": PHASE,
                    "model": MODEL,
                    "precision": "bfloat16",
                    "split": "confirmation",
                    "family": family,
                    "output_type": OUTPUT_TYPE,
                    "operation": OPERATION,
                    "layer": TARGET_LAYER,
                    "selected_heads": selected_heads,
                    "source_region": "response_map_declaration",
                    "component": component,
                    "n": len(selected),
                    "median_component_to_all_norm_ratio": median(
                        "component_to_all_norm_ratio"
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
                    "median_attention_mass_by_head_base": [
                        float(np.median([
                            row["base_attention_mass"][head]
                            for row in selected
                        ]))
                        for head in range(len(selected_heads))
                    ],
                    "median_attention_mass_by_head_variant": [
                        float(np.median([
                            row["variant_attention_mass"][head]
                            for row in selected
                        ]))
                        for head in range(len(selected_heads))
                    ],
                })
        result = {
            "schema_version": "phase1010_map_qkv_bf16.v1",
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "split": "confirmation",
            "source_subregion_summary": subregion,
            "layer": TARGET_LAYER,
            "selected_heads": selected_heads,
            "source_region": "response_map_declaration",
            "component_identity": (
                "a1*v1-a0*v0=(a1-a0)*v0+a0*(v1-v0)+"
                "(a1-a0)*(v1-v0)"
            ),
            "identity_is_measurement_not_language_formula": True,
            "cell_summaries": summaries,
            "maximum_qkv_identity_error": max(identity_errors),
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "local L30 attention decomposition; it does not explain "
                "how earlier layers computed the map-token value states"
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
