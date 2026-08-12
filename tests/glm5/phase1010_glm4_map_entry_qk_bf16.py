#!/usr/bin/env python3
"""Localize confirmed QK routing to semantic response-map entries."""
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
from phase1010_glm4_map_qkv_bf16 import (
    CONFIRMATION_N,
    MAP_OFFSETS,
    TARGET_LAYER,
    capture_attention_detail,
)
from phase1010_glm4_source_role_discovery import (
    FAMILIES,
    MODEL,
    OPERATION,
    OUTPUT_TYPE,
    PHASE1008_ROOT,
    forward_with_selected_replacement,
)
from phase1010_output_type_protocol import (
    OUT_ROOT,
    PHASE,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


ENTRY_GROUPS = (
    "base_answer_entry",
    "variant_answer_entry",
    "answer_pair_entries",
    "other_map_entries",
    "map_header_and_separators",
    "all_map_entries",
)
EPSILON = 1e-8


def find_subsequence(values: list[int], pattern: list[int]) -> int:
    matches = [
        start
        for start in range(len(values) - len(pattern) + 1)
        if values[start : start + len(pattern)] == pattern
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"map entry token match drift pattern={pattern} matches={matches}"
        )
    return matches[0]


def entry_offset_groups(case, tokenizer) -> dict[str, list[int]]:
    relay_start = int(case["shared_semantic_prefix_length"])
    map_ids = [
        int(value)
        for value in case["input_ids"][
            relay_start : relay_start + len(MAP_OFFSETS)
        ]
    ]
    entries = {}
    occupied = set()
    for entity in case["candidate_entities"]:
        label = case["response_mapping"][entity]
        pattern = [
            int(value)
            for value in tokenizer.encode(
                f" {entity}={label}",
                add_special_tokens=False,
            )
        ]
        start = find_subsequence(map_ids, pattern)
        offsets = list(range(start, start + len(pattern)))
        entries[entity] = offsets
        occupied.update(offsets)
    base_entity = case["gold_entity"]
    remaining = sorted(set(MAP_OFFSETS) - occupied)
    return {
        "entry_by_entity": entries,
        "base_entity": base_entity,
        "other_surface": remaining,
    }


def semantic_groups(base_case, variant_case, tokenizer):
    base = entry_offset_groups(base_case, tokenizer)
    variant = entry_offset_groups(variant_case, tokenizer)
    if base["entry_by_entity"] != variant["entry_by_entity"]:
        raise RuntimeError("response map entry layout changed within F pair")
    base_entity = base_case["gold_entity"]
    variant_entity = variant_case["gold_entity"]
    if base_entity == variant_entity:
        raise RuntimeError("F did not change answer entity")
    entries = base["entry_by_entity"]
    pair = sorted(set(entries[base_entity]) | set(entries[variant_entity]))
    other_entries = sorted({
        offset
        for entity, offsets in entries.items()
        if entity not in (base_entity, variant_entity)
        for offset in offsets
    })
    return {
        "base_answer_entry": entries[base_entity],
        "variant_answer_entry": entries[variant_entity],
        "answer_pair_entries": pair,
        "other_map_entries": other_entries,
        "map_header_and_separators": base["other_surface"],
        "all_map_entries": list(MAP_OFFSETS),
    }


def run_batch(
    *,
    model,
    layer,
    tokenizer,
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
    vectors = {
        group: torch.zeros_like(base_heads, dtype=torch.float32)
        for group in ENTRY_GROUPS
    }
    attention_delta_mass = {
        group: torch.zeros(
            len(items),
            len(selected_heads),
            dtype=torch.float32,
            device=base_heads.device,
        )
        for group in ENTRY_GROUPS
    }
    position_counts = {group: [] for group in ENTRY_GROUPS}
    for index, (base_case, variant_case) in enumerate(
        zip(base_original, variant_original)
    ):
        groups = semantic_groups(base_case, variant_case, tokenizer)
        for group, offsets in groups.items():
            bpos = torch.tensor(
                [base_positions[index][offset] for offset in offsets],
                dtype=torch.long,
                device=base_weights.device,
            )
            vpos = torch.tensor(
                [variant_positions[index][offset] for offset in offsets],
                dtype=torch.long,
                device=variant_weights.device,
            )
            a0 = base_weights[index].index_select(1, bpos).float()
            a1 = variant_weights[index].index_select(1, vpos).float()
            v0 = base_values[index].index_select(0, bpos).permute(
                1, 0, 2
            ).float()
            qk = ((a1 - a0)[:, :, None] * v0).sum(dim=1)
            vectors[group][index] = qk.to(vectors[group].device)
            attention_delta_mass[group][index] = (
                a1 - a0
            ).sum(dim=1).to(attention_delta_mass[group].device)
            position_counts[group].append(len(offsets))
    rows = []
    all_norm = torch.linalg.vector_norm(
        vectors["all_map_entries"].reshape(len(items), -1),
        dim=-1,
    )
    for group in ENTRY_GROUPS:
        vector = vectors[group]
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
        shuffled = torch.roll(vector, shifts=1, dims=0)
        shuffled_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=base_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=base_heads + shuffled,
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
        norm = torch.linalg.vector_norm(
            vector.reshape(len(items), -1),
            dim=-1,
        )
        for item_index, item in enumerate(items):
            rows.append({
                "schema_version": "phase1010_map_entry_qk_unit.v1",
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
                "component": "qk_routing",
                "entry_group": group,
                "position_count": position_counts[group][item_index],
                "component_norm": float(norm[item_index].item()),
                "all_map_qk_norm": float(
                    all_norm[item_index].item()
                ),
                "component_to_all_map_qk_norm_ratio": float(
                    norm[item_index].item()
                    / max(all_norm[item_index].item(), EPSILON)
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
                "attention_delta_mass_by_head": [
                    float(value)
                    for value in attention_delta_mass[group][
                        item_index
                    ].tolist()
                ],
            })
    return rows


def main() -> None:
    qkv = read_json(
        OUT_ROOT
        / "map_qkv_decomposition"
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
    output_root = OUT_ROOT / "map_entry_qk" / "bf16"
    started = time.time()
    model = tokenizer = device = None
    all_rows = []
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
                    all_rows.extend(run_batch(
                        model=model,
                        layer=layer,
                        tokenizer=tokenizer,
                        device=device,
                        selected_heads=selected_heads,
                        items=batch,
                    ))
            print(
                f"[map-entry-qk] {family} n={len(items)}",
                flush=True,
            )
        summaries = []
        for family in FAMILIES:
            for group in ENTRY_GROUPS:
                selected = [
                    row
                    for row in all_rows
                    if row["family"] == family
                    and row["entry_group"] == group
                ]

                def median(field):
                    return float(np.median([
                        row[field] for row in selected
                    ]))

                summaries.append({
                    "schema_version": "phase1010_map_entry_qk_cell.v1",
                    "phase": PHASE,
                    "model": MODEL,
                    "precision": "bfloat16",
                    "split": "confirmation",
                    "family": family,
                    "output_type": OUTPUT_TYPE,
                    "operation": OPERATION,
                    "layer": TARGET_LAYER,
                    "component": "qk_routing",
                    "entry_group": group,
                    "n": len(selected),
                    "median_position_count": median("position_count"),
                    "median_component_to_all_map_qk_norm_ratio": median(
                        "component_to_all_map_qk_norm_ratio"
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
                    "median_attention_delta_mass_by_head": [
                        float(np.median([
                            row["attention_delta_mass_by_head"][head]
                            for row in selected
                        ]))
                        for head in range(len(selected_heads))
                    ],
                })
        result = {
            "schema_version": "phase1010_map_entry_qk_bf16.v1",
            "phase": PHASE,
            "model": MODEL,
            "precision": "bfloat16",
            "split": "confirmation",
            "source_qkv_summary": qkv,
            "layer": TARGET_LAYER,
            "selected_heads": selected_heads,
            "component": "qk_routing",
            "semantic_entry_groups": list(ENTRY_GROUPS),
            "cell_summaries": summaries,
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "semantic map-entry localization inside a synthetic "
                "explicit mapping protocol; natural-language generality "
                "and earlier rule computation remain unresolved"
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
