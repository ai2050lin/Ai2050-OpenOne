#!/usr/bin/env python3
"""Discover which source-role contributions reach frozen GLM4 L30 heads.

The decomposition uses the transformer's exact attention computation as a
measurement identity. It does not name any role a language mechanism. Every
atomic source partition is measured before a small role set is frozen for
disjoint BF16 confirmation.
"""
from __future__ import annotations

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
from phase1009_crossfamily_heldout_causal_replication import (
    candidate_margin,
    finite_fraction,
)
from phase1009_crossfamily_response_scan import case_tensors, stage_case
from phase1010_output_type_protocol import (
    OUT_ROOT,
    PHASE,
    canonical,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


MODEL = "glm4"
OUTPUT_TYPE = "code"
OPERATION = "F"
FAMILIES = ("negation", "semantic_role")
MIN_DISCOVERY_N = 8
ATOMIC_ROLES = (
    "focal_source",
    "focal_bridge",
    "focal_operator",
    "query_anchor",
    "query_operator",
    "nuisance",
    "semantic_context_other",
    "assistant_boundary",
    "response_map_instruction",
    "answer_prefix",
    "decision_boundary",
)
AGGREGATE_ROLES = {
    "task_semantics": (
        "focal_source",
        "focal_bridge",
        "focal_operator",
        "query_anchor",
        "query_operator",
        "semantic_context_other",
    ),
    "prompt_semantics_all": (
        "focal_source",
        "focal_bridge",
        "focal_operator",
        "query_anchor",
        "query_operator",
        "nuisance",
        "semantic_context_other",
    ),
    "response_surface_all": (
        "assistant_boundary",
        "response_map_instruction",
        "answer_prefix",
        "decision_boundary",
    ),
    "all_sources": ATOMIC_ROLES,
}
ALL_ROLES = ATOMIC_ROLES + tuple(AGGREGATE_ROLES)
PHASE1008_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
)
EPSILON = 1e-8


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def source_partition(
    case: dict[str, Any],
    staged: dict[str, Any],
) -> dict[str, list[int]]:
    total_length = len(staged["input_ids"])
    prompt_length = len(case["input_ids"])
    shared_prefix = int(case["shared_semantic_prefix_length"])
    roles = {role: [] for role in ATOMIC_ROLES}
    occupied: set[int] = set()
    for role, value in case["role_positions"].items():
        position = int(value)
        if role == "answer_boundary":
            target = "assistant_boundary"
        else:
            target = str(case["role_classes"][role])
        if target not in roles:
            raise RuntimeError(f"unknown role class {target}")
        roles[target].append(position)
        occupied.add(position)
    for position in range(shared_prefix):
        if position not in occupied:
            roles["semantic_context_other"].append(position)
            occupied.add(position)
    for position in range(shared_prefix, prompt_length):
        if position not in occupied:
            roles["response_map_instruction"].append(position)
            occupied.add(position)
    for position in range(prompt_length, total_length - 1):
        if position not in occupied:
            roles["answer_prefix"].append(position)
            occupied.add(position)
    decision = total_length - 1
    if decision in occupied:
        raise RuntimeError("decision boundary partition overlap")
    roles["decision_boundary"].append(decision)
    occupied.add(decision)
    expected = set(range(total_length))
    if occupied != expected:
        raise RuntimeError(
            f"source partition coverage drift missing="
            f"{sorted(expected - occupied)[:8]} extra="
            f"{sorted(occupied - expected)[:8]}"
        )
    return roles


def capture_selected_attention(
    *,
    model,
    layer,
    cases: list[dict[str, Any]],
    original_cases: list[dict[str, Any]],
    device,
    selected_heads: list[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    float,
]:
    input_ids, attention_mask = case_tensors(cases, device)
    positions = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in cases
        ],
        dtype=torch.long,
        device=device,
    )
    captured: dict[str, torch.Tensor] = {}
    counts = defaultdict(int)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    if num_heads % num_kv_heads:
        raise RuntimeError("GQA head ratio drift")

    def value_hook(module, args, output):
        captured["values"] = output.detach().reshape(
            output.shape[0],
            output.shape[1],
            num_kv_heads,
            head_dim,
        )
        counts["values"] += 1

    def head_hook(module, args):
        value = args[0]
        batch = torch.arange(value.shape[0], device=value.device)
        captured["heads"] = value[
            batch,
            positions.to(value.device),
            :,
        ].detach().reshape(value.shape[0], num_heads, head_dim)
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
            positions.to(weights.device),
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
            raise RuntimeError(f"attention capture count drift {dict(counts)}")
        logits = output.logits[:, -1, :].detach()
        values = captured["values"].repeat_interleave(
            num_heads // num_kv_heads,
            dim=2,
        )
        weights = captured["weights"]
        heads = captured["heads"]
        contributions = {
            role: torch.zeros(
                len(cases),
                len(selected_heads),
                head_dim,
                dtype=torch.float32,
                device=heads.device,
            )
            for role in ATOMIC_ROLES
        }
        for index, (original, staged) in enumerate(
            zip(original_cases, cases)
        ):
            partition = source_partition(original, staged)
            for role, source_positions in partition.items():
                source_index = torch.tensor(
                    source_positions,
                    dtype=torch.long,
                    device=weights.device,
                )
                attention = weights[
                    index,
                    selected_heads,
                    :,
                ].index_select(1, source_index)
                selected_values = values[
                    index,
                    :,
                    selected_heads,
                    :,
                ].index_select(0, source_index)
                selected_values = selected_values.permute(1, 0, 2)
                contributions[role][index] = (
                    attention[:, :, None].float()
                    * selected_values.float()
                ).sum(dim=1).to(heads.device)
        rebuilt = torch.stack(
            list(contributions.values()),
            dim=0,
        ).sum(dim=0)
        selected_physical = heads[:, selected_heads, :].float()
        reconstruction_error = float(
            torch.max(torch.abs(rebuilt - selected_physical)).item()
        )
        for name, members in AGGREGATE_ROLES.items():
            contributions[name] = torch.stack(
                [contributions[member] for member in members],
                dim=0,
            ).sum(dim=0)
        return logits, heads, contributions, reconstruction_error
    finally:
        for handle in reversed(handles):
            handle.remove()
        del input_ids, attention_mask, positions


def forward_with_selected_replacement(
    *,
    model,
    layer,
    cases: list[dict[str, Any]],
    device,
    selected_heads: list[int],
    replacement: torch.Tensor,
) -> torch.Tensor:
    input_ids, attention_mask = case_tensors(cases, device)
    positions = torch.tensor(
        [
            int(case["scan_role_positions"]["decision_boundary"])
            for case in cases
        ],
        dtype=torch.long,
        device=device,
    )
    head_count = int(model.config.num_attention_heads)
    head_dim = int(model.config.head_dim)
    count = [0]

    def hook(module, args):
        value = args[0]
        patched = value.clone()
        batch = torch.arange(value.shape[0], device=value.device)
        selected = patched[
            batch,
            positions.to(value.device),
            :,
        ].reshape(value.shape[0], head_count, head_dim)
        selected = selected.clone()
        selected[:, selected_heads, :] = replacement.to(
            device=value.device,
            dtype=value.dtype,
        )
        patched[
            batch,
            positions.to(value.device),
            :,
        ] = selected.reshape(value.shape[0], -1)
        count[0] += 1
        return (patched,) + tuple(args[1:])

    handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        if count[0] != 1:
            raise RuntimeError(f"replacement hook count drift {count[0]}")
        return output.logits[:, -1, :].detach()
    finally:
        handle.remove()
        del input_ids, attention_mask, positions


def run_group(
    *,
    model,
    layer,
    device,
    selected_heads: list[int],
    items: list[dict[str, Any]],
    roles: tuple[str, ...] = ALL_ROLES,
) -> tuple[list[dict[str, Any]], float]:
    base_original = [item["base"] for item in items]
    variant_original = [item["variant"] for item in items]
    base_cases = [stage_case(case, "semantic0") for case in base_original]
    variant_cases = [
        stage_case(case, "semantic0") for case in variant_original
    ]
    (
        base_logits,
        base_heads,
        base_contributions,
        base_error,
    ) = capture_selected_attention(
        model=model,
        layer=layer,
        cases=base_cases,
        original_cases=base_original,
        device=device,
        selected_heads=selected_heads,
    )
    (
        variant_logits,
        variant_heads,
        variant_contributions,
        variant_error,
    ) = capture_selected_attention(
        model=model,
        layer=layer,
        cases=variant_cases,
        original_cases=variant_original,
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
    selected_base = base_heads[:, selected_heads, :].float()
    selected_variant = variant_heads[:, selected_heads, :].float()
    full_delta = selected_variant - selected_base
    full_delta_norm = torch.linalg.vector_norm(
        full_delta.reshape(len(items), -1),
        dim=-1,
    )
    no_op_logits = forward_with_selected_replacement(
        model=model,
        layer=layer,
        cases=base_cases,
        device=device,
        selected_heads=selected_heads,
        replacement=selected_base,
    )
    no_op_error = torch.max(
        torch.abs(no_op_logits - base_logits),
        dim=-1,
    ).values
    rows = []
    for role in roles:
        role_delta = (
            variant_contributions[role] - base_contributions[role]
        )
        role_delta_norm = torch.linalg.vector_norm(
            role_delta.reshape(len(items), -1),
            dim=-1,
        )
        sufficiency_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=base_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=selected_base + role_delta,
        )
        restore_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=variant_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=selected_variant - role_delta,
        )
        shuffled_delta = torch.roll(role_delta, shifts=1, dims=0)
        shuffled_logits = forward_with_selected_replacement(
            model=model,
            layer=layer,
            cases=base_cases,
            device=device,
            selected_heads=selected_heads,
            replacement=selected_base + shuffled_delta,
        )
        sufficiency_margin = candidate_margin(
            sufficiency_logits,
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
        sufficiency_fraction = finite_fraction(
            sufficiency_margin - base_margin,
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
        directions = role_delta.reshape(len(items), -1)
        directions = directions / torch.clamp(
            torch.linalg.vector_norm(directions, dim=-1, keepdim=True),
            min=EPSILON,
        )
        direction_sum = directions.float().sum(dim=0)
        n = len(items)
        direction_consistency = (
            None
            if n < 2
            else float(
                (
                    torch.dot(direction_sum, direction_sum) - n
                ).item()
                / (n * (n - 1))
            )
        )
        for index, item in enumerate(items):
            rows.append({
                "schema_version": (
                    "phase1010_source_role_discovery_unit.v1"
                ),
                "phase": PHASE,
                "model": MODEL,
                "split": "discovery",
                "family": item["unit"]["family"],
                "output_type": OUTPUT_TYPE,
                "operation": OPERATION,
                "unit_id": item["unit"]["unit_id"],
                "template": int(item["unit"]["template"]),
                "name_pool": int(item["unit"]["name_pool"]),
                "source_role": role,
                "role_kind": (
                    "atomic"
                    if role in ATOMIC_ROLES
                    else "predefined_aggregate"
                ),
                "base_margin": float(base_margin[index].item()),
                "variant_margin": float(variant_margin[index].item()),
                "natural_effect": float(natural_effect[index].item()),
                "role_delta_norm": float(role_delta_norm[index].item()),
                "full_selected_head_delta_norm": float(
                    full_delta_norm[index].item()
                ),
                "role_to_full_delta_norm_ratio": float(
                    role_delta_norm[index].item()
                    / max(full_delta_norm[index].item(), EPSILON)
                ),
                "sufficiency_fraction": float(
                    sufficiency_fraction[index].item()
                ),
                "restore_fraction": float(
                    restore_fraction[index].item()
                ),
                "shuffled_sufficiency_fraction": float(
                    shuffled_fraction[index].item()
                ),
                "direction_consistency_in_batch": direction_consistency,
                "noop_max_logit_error": float(no_op_error[index].item()),
            })
        del sufficiency_logits, restore_logits, shuffled_logits
    return rows, max(base_error, variant_error)


def summarize_cell(
    family: str,
    rows: list[dict[str, Any]],
    reconstruction_error: float,
    *,
    roles: tuple[str, ...] = ALL_ROLES,
    split: str = "discovery",
) -> list[dict[str, Any]]:
    result = []
    for role in roles:
        selected = [row for row in rows if row["source_role"] == role]

        def median(name: str) -> float:
            return float(np.median([row[name] for row in selected]))

        suff = median("sufficiency_fraction")
        restore = median("restore_fraction")
        shuffled = median("shuffled_sufficiency_fraction")
        result.append({
            "schema_version": "phase1010_source_role_discovery_cell.v1",
            "phase": PHASE,
            "model": MODEL,
            "split": split,
            "family": family,
            "output_type": OUTPUT_TYPE,
            "operation": OPERATION,
            "source_role": role,
            "role_kind": (
                "atomic"
                if role in ATOMIC_ROLES
                else "predefined_aggregate"
            ),
            "n": len(selected),
            "median_role_delta_norm": median("role_delta_norm"),
            "median_role_to_full_delta_norm_ratio": median(
                "role_to_full_delta_norm_ratio"
            ),
            "median_sufficiency_fraction": suff,
            "median_restore_fraction": restore,
            "median_shuffled_sufficiency_fraction": shuffled,
            "median_sufficiency_minus_shuffled": float(suff - shuffled),
            "maximum_attention_reconstruction_error": (
                reconstruction_error
            ),
            "maximum_noop_logit_error": max(
                row["noop_max_logit_error"] for row in selected
            ),
            "claim_limit": (
                "source-role contribution to a frozen local head group; "
                "not an upstream language rule or complete path"
            ),
        })
    return result


def main() -> None:
    precision = read_json(
        OUT_ROOT
        / "precision_audit"
        / "glm4_bf16"
        / "summary.json"
    )
    if not precision["upstream_source_mapping_authorized"]:
        raise RuntimeError("BF16 gate did not authorize source mapping")
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
    output_root = OUT_ROOT / "source_role_mapping" / "discovery"
    started = time.time()
    model = tokenizer = device = None
    all_rows: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    reconstruction_errors = []
    try:
        model, tokenizer, device = load_model(MODEL, use_8bit=True)
        layers = get_layers(model)
        layer = layers[int(selection["layer"]) - 1]
        for family in FAMILIES:
            items = []
            for unit in units:
                if (
                    unit["family"] != family
                    or unit["output_type"] != OUTPUT_TYPE
                    or unit["split"] != "discovery"
                ):
                    continue
                if not qualification[
                    (unit["unit_id"], OPERATION)
                ]["semantic_pair_qualified"]:
                    continue
                items.append({
                    "unit": unit,
                    "base": case_by_id[unit["case_ids"]["base"]],
                    "variant": case_by_id[
                        unit["case_ids"][OPERATION]
                    ],
                })
            if len(items) < MIN_DISCOVERY_N:
                raise RuntimeError(
                    f"{family}: underpowered discovery n={len(items)}"
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
                rows, error = run_group(
                    model=model,
                    layer=layer,
                    device=device,
                    selected_heads=selected_heads,
                    items=group,
                )
                family_rows.extend(rows)
                family_errors.append(error)
            family_cells = summarize_cell(
                family,
                family_rows,
                max(family_errors),
            )
            all_rows.extend(family_rows)
            all_cells.extend(family_cells)
            reconstruction_errors.extend(family_errors)
            print(
                f"[source-discovery] {family} n={len(items)} "
                f"groups={len(grouped)} reconstruction="
                f"{max(family_errors):.6g}",
                flush=True,
            )

        atomic_scores = []
        for role in ATOMIC_ROLES:
            family_cells = [
                row
                for row in all_cells
                if row["source_role"] == role
            ]
            family_directional = [
                min(
                    row["median_sufficiency_fraction"],
                    row["median_restore_fraction"],
                )
                - max(
                    row["median_shuffled_sufficiency_fraction"],
                    0.0,
                )
                for row in family_cells
            ]
            atomic_scores.append({
                "source_role": role,
                "cross_family_min_directional_excess": float(
                    min(family_directional)
                ),
                "cross_family_median_directional_excess": float(
                    np.median(family_directional)
                ),
                "family_values": {
                    row["family"]: {
                        "sufficiency": row[
                            "median_sufficiency_fraction"
                        ],
                        "restore": row["median_restore_fraction"],
                        "shuffled": row[
                            "median_shuffled_sufficiency_fraction"
                        ],
                    }
                    for row in family_cells
                },
            })
        ranked = sorted(
            atomic_scores,
            key=lambda row: (
                row["cross_family_min_directional_excess"],
                row["cross_family_median_directional_excess"],
                row["source_role"],
            ),
            reverse=True,
        )
        selected_atomic_roles = [
            row["source_role"]
            for row in ranked
            if row["cross_family_min_directional_excess"] > 0
        ][:3]
        confirmation_roles = selected_atomic_roles + [
            "task_semantics",
            "response_surface_all",
            "all_sources",
        ]
        selection_result = {
            "schema_version": "phase1010_source_role_selection.v1",
            "phase": PHASE,
            "model": MODEL,
            "source_phase": 1008,
            "layer": int(selection["layer"]),
            "selected_heads": selected_heads,
            "output_type": OUTPUT_TYPE,
            "operation": OPERATION,
            "discovery_families": list(FAMILIES),
            "discovery_split_only": True,
            "atomic_role_scores": ranked,
            "selected_atomic_roles": selected_atomic_roles,
            "fixed_aggregate_controls": [
                "task_semantics",
                "response_surface_all",
                "all_sources",
            ],
            "confirmation_roles": confirmation_roles,
            "selection_rule": (
                "rank every atomic role by the minimum across families of "
                "min(median sufficiency, median restoration) minus the "
                "positive shuffled baseline; keep at most three roles "
                "with positive excess"
            ),
            "formula_status": (
                "selection instrument only, not a language mechanism formula"
            ),
        }
        summary = {
            "schema_version": "phase1010_source_role_discovery.v1",
            "phase": PHASE,
            "model": MODEL,
            "precision": "8bit",
            "prerequisite_bf16_digest": digest(precision),
            "source_selection_digest": digest(selection_bundle),
            "layer": int(selection["layer"]),
            "selected_heads": selected_heads,
            "output_type": OUTPUT_TYPE,
            "operation": OPERATION,
            "family_count": len(FAMILIES),
            "unit_role_row_count": len(all_rows),
            "cell_role_count": len(all_cells),
            "atomic_role_count": len(ATOMIC_ROLES),
            "aggregate_role_count": len(AGGREGATE_ROLES),
            "maximum_attention_reconstruction_error": max(
                reconstruction_errors
            ),
            "maximum_noop_logit_error": max(
                row["noop_max_logit_error"] for row in all_rows
            ),
            "selected_atomic_roles": selected_atomic_roles,
            "confirmation_roles": confirmation_roles,
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "maps direct source-role contributions into the frozen L30 "
                "head group; it does not locate earlier-layer computation"
            ),
        }
        write_jsonl(output_root / "units.jsonl", all_rows)
        write_jsonl(output_root / "cell_summaries.jsonl", all_cells)
        write_json(output_root / "selection.json", selection_result)
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
