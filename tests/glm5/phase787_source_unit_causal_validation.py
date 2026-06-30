#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight  # noqa: E402
from phase781_surface_form_candidate_causal_patch import make_row  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402
from phase785_positive_negative_subspace_split import (  # noqa: E402
    component_keys_for_routes,
    observe_logits,
    parse_budgets,
    parse_csv,
    route_vectors,
    selected_dims_for_mode,
    selected_score_sums,
    signed_channel_entries,
)
from phase786_head_mlp_source_audit import (  # noqa: E402
    capture_answer_outputs_and_sources,
    component_selected_score_sum,
    enrich_selected_rows_with_target_id,
    infer_num_heads,
)


OUT_ROOT = Path("results/glm5_phase787_source_unit_causal_validation")
RESULT_ROOT = Path("tests/result/phase787_source_unit_causal_validation")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def safe_rate(values: list[Any]) -> float | None:
    vals = [bool(v) for v in values]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def subspace_specs(modes: list[str], budgets: list[int]) -> list[tuple[str, str, int | None]]:
    specs = []
    for mode in modes:
        if mode in {"all_positive", "all_negative", "all"}:
            specs.append((mode, mode, None))
        else:
            for budget in budgets:
                specs.append((mode, f"{mode}_{budget}", budget))
    return specs


def unit_rank_score(mode: str, signed: float, positive: float, negative_abs: float, abs_sum: float) -> float:
    if mode in {"positive", "all_positive"}:
        return float(positive)
    if mode in {"negative", "all_negative"}:
        return float(negative_abs)
    return float(abs_sum)


def attention_head_scores(
    model,
    key: tuple[str, int],
    selected_dims: torch.Tensor,
    base_state: dict[str, Any],
    donor_state: dict[str, Any],
    readout_direction: torch.Tensor,
    mode: str,
) -> list[dict[str, Any]]:
    layer_idx = key[1]
    attn = get_layers(model)[layer_idx].self_attn
    pre_base = base_state["attn_o_inputs"].get(key)
    pre_donor = donor_state["attn_o_inputs"].get(key)
    if pre_base is None or pre_donor is None or not hasattr(attn, "o_proj"):
        return []
    weight = attn.o_proj.weight.detach().float().cpu()
    n_heads = infer_num_heads(model, attn)
    if not n_heads:
        return []
    in_features = int(weight.shape[1])
    if in_features % n_heads != 0:
        return []
    head_dim = in_features // n_heads
    dims = selected_dims.long()
    delta_pre = (pre_donor - pre_base).float()
    selected_readout = readout_direction[dims].float()
    rows = []
    for head_id in range(n_heads):
        start = head_id * head_dim
        end = start + head_dim
        projected = torch.matmul(weight[dims, start:end], delta_pre[start:end])
        per_dim = projected * selected_readout
        signed = float(per_dim.sum().item())
        positive = float(torch.clamp(per_dim, min=0.0).sum().item())
        negative_abs = float(torch.clamp(-per_dim, min=0.0).sum().item())
        abs_sum = float(per_dim.abs().sum().item())
        rows.append(
            {
                "unit_id": int(head_id),
                "unit_score": unit_rank_score(mode, signed, positive, negative_abs, abs_sum),
                "source_signed_score": signed,
                "source_positive_score": positive,
                "source_negative_abs_score": negative_abs,
                "source_abs_score": abs_sum,
                "head_dim": head_dim,
                "num_units": n_heads,
            }
        )
    return sorted(rows, key=lambda r: (r["unit_score"], r["source_abs_score"]), reverse=True)


def mlp_channel_scores(
    model,
    key: tuple[str, int],
    selected_dims: torch.Tensor,
    base_state: dict[str, Any],
    donor_state: dict[str, Any],
    readout_direction: torch.Tensor,
    mode: str,
) -> list[dict[str, Any]]:
    layer_idx = key[1]
    layer = get_layers(model)[layer_idx]
    pre_base = base_state["mlp_down_inputs"].get(key)
    pre_donor = donor_state["mlp_down_inputs"].get(key)
    if pre_base is None or pre_donor is None or not hasattr(layer.mlp, "down_proj"):
        return []
    weight = layer.mlp.down_proj.weight.detach().float().cpu()
    dims = selected_dims.long()
    delta_pre = (pre_donor - pre_base).float()
    coeff = torch.matmul(weight[dims, :].T, readout_direction[dims].float())
    channel_scores = delta_pre * coeff
    rows = []
    for channel_id, value in enumerate(channel_scores.tolist()):
        signed = float(value)
        positive = max(signed, 0.0)
        negative_abs = max(-signed, 0.0)
        abs_sum = abs(signed)
        rows.append(
            {
                "unit_id": int(channel_id),
                "unit_score": unit_rank_score(mode, signed, positive, negative_abs, abs_sum),
                "source_signed_score": signed,
                "source_positive_score": positive,
                "source_negative_abs_score": negative_abs,
                "source_abs_score": abs_sum,
                "num_units": int(channel_scores.numel()),
            }
        )
    return sorted(rows, key=lambda r: (r["unit_score"], r["source_abs_score"]), reverse=True)


def select_unit_set(
    scores: list[dict[str, Any]],
    size: int,
    selection_kind: str,
    seed_parts: tuple[Any, ...],
) -> list[dict[str, Any]]:
    if not scores:
        return []
    size = min(int(size), len(scores))
    if selection_kind == "top":
        return scores[:size]
    top_ids = {int(r["unit_id"]) for r in scores[:size]}
    pool = [r for r in scores if int(r["unit_id"]) not in top_ids]
    if len(pool) < size:
        pool = [r for r in scores if int(r["unit_id"]) not in set()]
    rng = random.Random(stable_seed(*seed_parts, selection_kind, size))
    picked = list(pool)
    rng.shuffle(picked)
    return picked[:size]


def source_set_meta(
    source_unit_kind: str,
    units: list[dict[str, Any]],
    component_scores: dict[str, float],
    selection_kind: str,
    set_size: int,
) -> dict[str, Any]:
    signed = sum(float(u.get("source_signed_score") or 0.0) for u in units)
    positive = sum(float(u.get("source_positive_score") or 0.0) for u in units)
    negative_abs = sum(float(u.get("source_negative_abs_score") or 0.0) for u in units)
    abs_sum = sum(float(u.get("source_abs_score") or 0.0) for u in units)
    return {
        "source_unit_kind": source_unit_kind,
        "source_selection_kind": selection_kind,
        "source_set_size": int(set_size),
        "source_unit_ids": [int(u["unit_id"]) for u in units],
        "source_set_signed_score": signed,
        "source_set_positive_score": positive,
        "source_set_negative_abs_score": negative_abs,
        "source_set_abs_score": abs_sum,
        **component_scores,
    }


def component_shortlist(
    items: list[dict[str, Any]],
    max_per_kind: int,
) -> list[dict[str, Any]]:
    by_kind: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_kind[(item["source_unit_kind"], item["source_selection_kind"])].append(item)
    out = []
    for vals in by_kind.values():
        vals.sort(key=lambda r: abs(float(r.get("component_selected_abs_score_sum") or 0.0)), reverse=True)
        out.extend(vals[: max_per_kind])
    return out


def install_source_replacement(
    model,
    tokenizer,
    device,
    prompt: str,
    replacements: list[dict[str, Any]],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    handles = []
    for repl in replacements:
        kind = repl["component_kind"]
        layer_idx = int(repl["layer"])
        key = (kind, layer_idx)
        layer = layers[layer_idx]
        if repl["source_unit_kind"] == "attention_head_set":
            source_vec = source_state["attn_o_inputs"].get(key)
            if source_vec is None:
                continue
            attn = layer.self_attn
            n_heads = infer_num_heads(model, attn)
            if not n_heads:
                continue
            in_features = int(source_vec.numel())
            if in_features % n_heads != 0:
                continue
            head_dim = in_features // n_heads
            head_ids = [int(x) for x in repl["source_unit_ids"]]

            def o_pre_hook(_module, inputs, source_vec=source_vec, head_ids=head_ids, head_dim=head_dim):
                if not inputs or not torch.is_tensor(inputs[0]):
                    return inputs
                patched = inputs[0].clone()
                src = source_vec.to(device=patched.device, dtype=patched.dtype)
                for head_id in head_ids:
                    start = head_id * head_dim
                    end = start + head_dim
                    patched[0, answer_pos, start:end] = src[start:end]
                return (patched, *inputs[1:])

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(o_pre_hook))
        elif repl["source_unit_kind"] == "mlp_channel_set":
            source_vec = source_state["mlp_down_inputs"].get(key)
            if source_vec is None or not hasattr(layer.mlp, "down_proj"):
                continue
            channel_ids = torch.tensor([int(x) for x in repl["source_unit_ids"]], dtype=torch.long)

            def down_pre_hook(_module, inputs, source_vec=source_vec, channel_ids=channel_ids):
                if not inputs or not torch.is_tensor(inputs[0]):
                    return inputs
                patched = inputs[0].clone()
                idx = channel_ids.to(device=patched.device)
                src = source_vec.to(device=patched.device, dtype=patched.dtype)
                patched[0, answer_pos, idx] = src[idx]
                return (patched, *inputs[1:])

            handles.append(layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook))
    try:
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device=device),
                return_dict=True,
                use_cache=False,
            )
        logits = out.logits[0, -1].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return {"ids": ids, "answer_pos": answer_pos, "logits": logits}


def make_source_row(
    model_name: str,
    route: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    intervention_kind: str,
    obs: dict[str, Any],
    reference_obs: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    row = make_row(model_name, route, case, prompt_variant, intervention_kind, obs, reference_obs)
    row.update(
        {
            "row_kind": "phase787_source_unit_causal_validation",
            "route_id": route["route_id"],
            "route_size": route["route_size"],
            "component_labels": route["component_labels"],
            "components": route["components"],
            "position_scope": "answer_site",
            **meta,
        }
    )
    return row


def route_source_candidates(
    model,
    route: dict[str, Any],
    base_state: dict[str, Any],
    donor_state: dict[str, Any],
    readout_direction: torch.Tensor,
    mode: str,
    budget_label: str,
    budget: int | None,
    args: argparse.Namespace,
    seed_parts: tuple[Any, ...],
) -> list[dict[str, Any]]:
    base_vecs = route_vectors(base_state, route)
    donor_vecs = route_vectors(donor_state, route)
    if base_vecs is None or donor_vecs is None:
        return []
    entries, _route_meta = signed_channel_entries(route, base_vecs, donor_vecs, readout_direction)
    selected = selected_dims_for_mode(mode, budget, route, base_vecs, entries, seed_parts)
    if not selected:
        return []
    total_scores = selected_score_sums(selected, entries)
    candidates = []
    for key, dims in selected.items():
        comp_scores = component_selected_score_sum(entries, key, dims)
        if key[0] == "attn":
            scores = attention_head_scores(model, key, dims, base_state, donor_state, readout_direction, mode)
            size = int(args.attn_source_set_size)
            source_kind = "attention_head_set"
        elif key[0] == "mlp":
            scores = mlp_channel_scores(model, key, dims, base_state, donor_state, readout_direction, mode)
            size = int(args.mlp_source_set_size)
            source_kind = "mlp_channel_set"
        else:
            continue
        if not scores:
            continue
        for selection_kind in ("top", "random"):
            units = select_unit_set(scores, size, selection_kind, (*seed_parts, key[0], key[1], mode))
            if not units:
                continue
            meta = {
                "component_kind": key[0],
                "layer": key[1],
                "source_component_label": f"{key[0]}:L{key[1]}",
                "subspace_mode": mode,
                "budget_label": budget_label,
                "budget_requested": budget,
                "selected_output_dim_count": int(dims.numel()),
                "total_selected_dim_count": sum(int(v.numel()) for v in selected.values()),
                **source_set_meta(source_kind, units, comp_scores, selection_kind, size),
                **total_scores,
            }
            candidates.append(meta)
    return component_shortlist(candidates, args.max_components_per_kind)


def audit_case_route(
    model,
    tokenizer,
    device,
    unembed: torch.Tensor,
    args: argparse.Namespace,
    case: dict[str, Any],
    source_row: dict[str, Any],
    route: dict[str, Any],
    component_keys: set[tuple[str, int]],
    specs: list[tuple[str, str, int | None]],
) -> list[dict[str, Any]]:
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    target_id = int(source_row.get("target_token_id") or source_row.get("target_id") or 0)
    if not target_id:
        raise ValueError(f"source row lacks target_token_id: {source_row.keys()}")
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_variant = route["compare_variant"]
    donor_prompt = surface_prompt_for_variant(case, donor_variant)
    base_state = capture_answer_outputs_and_sources(model, tokenizer, device, baseline_prompt, component_keys)
    donor_state = capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    base_obs = observe_logits(tokenizer, base_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    donor_obs = observe_logits(tokenizer, donor_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    readout_direction = unembed[target_id].float() - unembed[case_variant_token_id].float()
    rows = []
    seed_parts = (args.model, args.round_name, case["case_id"], route["route_id"])
    for mode, budget_label, budget in specs:
        candidates = route_source_candidates(model, route, base_state, donor_state, readout_direction, mode, budget_label, budget, args, seed_parts)
        for meta in candidates:
            replacement = dict(meta)
            source_state = donor_state
            prompt_state = install_source_replacement(model, tokenizer, device, baseline_prompt, [replacement], source_state)
            obs = observe_logits(tokenizer, prompt_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
            rows.append(
                make_source_row(
                    args.model,
                    route,
                    case,
                    COMPARE_BASELINE,
                    "patch_baseline_from_donor_source_units",
                    obs,
                    base_obs,
                    meta,
                )
            )
            if meta["source_selection_kind"] == "top":
                source_state = base_state
                replace_state = install_source_replacement(model, tokenizer, device, donor_prompt, [replacement], source_state)
                replace_obs = observe_logits(tokenizer, replace_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
                rows.append(
                    make_source_row(
                        args.model,
                        route,
                        case,
                        donor_variant,
                        "replace_donor_source_units_with_baseline",
                        replace_obs,
                        donor_obs,
                        meta,
                    )
                )
    return rows


def group_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in key_fields)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "n": len(items),
                "case_n": len({r["case_id"] for r in items}),
                "components": dict(Counter(str(r.get("source_component_label")) for r in items)),
                "strict_open_rate": safe_rate([r.get("target_top1") for r in items]),
                "semantic_equiv_open_rate": safe_rate([r.get("semantic_equiv_open") for r in items]),
                "pool_top1_rate": safe_rate([r.get("pool_target_top1") for r in items]),
                "strict_gain_rate_vs_reference": safe_rate([r.get("strict_gain_vs_reference") for r in items if "strict_gain_vs_reference" in r]),
                "strict_loss_rate_vs_reference": safe_rate([r.get("strict_loss_vs_reference") for r in items if "strict_loss_vs_reference" in r]),
                "semantic_loss_rate_vs_reference": safe_rate([r.get("semantic_loss_vs_reference") for r in items if "semantic_loss_vs_reference" in r]),
                "pool_loss_rate_vs_reference": safe_rate([r.get("pool_loss_vs_reference") for r in items if "pool_loss_vs_reference" in r]),
                "mean_delta_margin_vs_reference": safe_mean([r.get("delta_margin_vs_reference") for r in items]),
                "mean_margin_target_vs_case_variant": safe_mean([r.get("margin_target_vs_case_variant") for r in items]),
                "mean_target_rank": safe_mean([r.get("target_rank") for r in items]),
                "mean_source_set_signed_score": safe_mean([r.get("source_set_signed_score") for r in items]),
                "mean_source_set_positive_score": safe_mean([r.get("source_set_positive_score") for r in items]),
                "mean_source_set_negative_abs_score": safe_mean([r.get("source_set_negative_abs_score") for r in items]),
                "mean_source_set_abs_score": safe_mean([r.get("source_set_abs_score") for r in items]),
                "mean_selected_output_dim_count": safe_mean([r.get("selected_output_dim_count") for r in items]),
                "mean_total_selected_dim_count": safe_mean([r.get("total_selected_dim_count") for r in items]),
                "top1_classes": dict(Counter(str(r.get("top1_competitor_class")) for r in items)),
            }
        )
        payload["sufficiency_score"] = (
            (payload["strict_gain_rate_vs_reference"] or 0.0)
            * max(payload["mean_delta_margin_vs_reference"] or 0.0, 0.0)
            if payload.get("intervention_kind") == "patch_baseline_from_donor_source_units"
            else 0.0
        )
        payload["negative_effect_score"] = (
            (1.0 - (payload["strict_gain_rate_vs_reference"] or 0.0))
            * max(-(payload["mean_delta_margin_vs_reference"] or 0.0), 0.0)
            if payload.get("intervention_kind") == "patch_baseline_from_donor_source_units"
            else 0.0
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("model") or "",
            r.get("source_unit_kind") or "",
            r.get("subspace_mode") or "",
            r.get("intervention_kind") or "",
            r.get("source_selection_kind") or "",
        )
    )
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], specs: list[tuple[str, str, int | None]]) -> dict[str, Any]:
    by_intervention = group_rows(
        rows,
        [
            "model",
            "source_unit_kind",
            "subspace_mode",
            "budget_label",
            "source_selection_kind",
            "source_set_size",
            "intervention_kind",
        ],
    )
    by_component = group_rows(
        rows,
        [
            "model",
            "route_id",
            "source_component_label",
            "source_unit_kind",
            "subspace_mode",
            "budget_label",
            "source_selection_kind",
            "source_set_size",
            "intervention_kind",
        ],
    )
    top_sufficiency = sorted(
        [r for r in by_component if r.get("intervention_kind") == "patch_baseline_from_donor_source_units"],
        key=lambda r: (r.get("sufficiency_score") or 0.0, r.get("mean_delta_margin_vs_reference") or 0.0),
        reverse=True,
    )
    top_negative = sorted(
        [
            r
            for r in by_component
            if r.get("intervention_kind") == "patch_baseline_from_donor_source_units"
            and r.get("subspace_mode") in {"negative", "all_negative"}
        ],
        key=lambda r: (r.get("mean_delta_margin_vs_reference") or 0.0, r.get("strict_gain_rate_vs_reference") or 0.0),
    )
    top_replacement_losses = sorted(
        [r for r in by_component if r.get("intervention_kind") == "replace_donor_source_units_with_baseline"],
        key=lambda r: (r.get("mean_delta_margin_vs_reference") or 0.0, r.get("strict_loss_rate_vs_reference") or 0.0),
    )
    return {
        "phase": 787,
        "title": "Source Unit Causal Validation for Signed Subspace",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase780_round": args.source_phase780_round,
        "source_phase786_round": args.source_phase786_round,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in specs],
        "method_note": (
            "Patch top or random source units from donor into baseline at answer-site source tensors. "
            "Attention units are o_proj input head slices; MLP units are down_proj input activation channels. "
            "Top source units are ranked by Phase786-style signed readout contribution."
        ),
        "by_intervention": by_intervention,
        "by_component": by_component,
        "top_sufficiency_components": top_sufficiency[:40],
        "top_negative_components": top_negative[:40],
        "top_replacement_losses": top_replacement_losses[:40],
        "strict_interpretation": (
            "This phase is a first causal validation of architecture source units. "
            "It validates answer-site source-unit intervention only; it still does not prove full Q/K/V or cross-position semantic fibers."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = select_surface_cases(args.model, args)
    routes = select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    component_keys = component_keys_for_routes(routes)
    specs = subspace_specs(parse_csv(args.subspace_modes), parse_budgets(args.budgets))
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)} specs={len(specs)}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys, specs))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: source-unit causal validation {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, args, attn_impl, routes, specs)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase787_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase787_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_intervention": summary["by_intervention"],
                "top_sufficiency_components": summary["top_sufficiency_components"][:8],
                "top_negative_components": summary["top_negative_components"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 787 Source Unit Causal Validation ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: patch donor source units into baseline, with random controls.",
        "- Attention source units are o_proj input head slices.",
        "- MLP source units are down_proj input activation channels.",
        "",
        "## Cross-Model Intervention Summary",
        "",
        "| model | source | subspace | selection | intervention | cases | strict gain | strict loss | delta margin | source signed | top1 classes |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("by_intervention", []):
            lines.append(
                f"| {model_name} | `{row['source_unit_kind']}` | `{row['subspace_mode']}` | `{row['source_selection_kind']}{row['source_set_size']}` | "
                f"`{row['intervention_kind']}` | {row['case_n']} | {fmt(row['strict_gain_rate_vs_reference'])} | "
                f"{fmt(row['strict_loss_rate_vs_reference'])} | {fmt(row['mean_delta_margin_vs_reference'])} | "
                f"{fmt(row['mean_source_set_signed_score'])} | `{json.dumps(row['top1_classes'], ensure_ascii=False)}` |"
            )
    lines += [
        "",
        "## Top Sufficiency Components",
        "",
        "| model | route | component | source | subspace | selection | cases | strict gain | delta margin | source signed |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in (data.get("top_sufficiency_components") or [])[:20]:
            selection_label = f"{row.get('source_selection_kind')}{row.get('source_set_size', '')}"
            lines.append(
                f"| {model_name} | `{row['route_id']}` | `{row['source_component_label']}` | `{row['source_unit_kind']}` | "
                f"`{row['subspace_mode']}` | `{selection_label}` | {row['case_n']} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['mean_delta_margin_vs_reference'])} | {fmt(row['mean_source_set_signed_score'])} |"
            )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "- This validates answer-site source-unit effects, not full Q/K/V path or cross-position semantic fibers.",
        "- Random controls are matched by unit count but not by activation norm.",
        "- MLP channel sets are activation channels, closer to neuron-level than residual channels but still not biological neurons.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase787_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 787,
        "title": "Source Unit Causal Validation for Signed Subspace",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase787_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase787_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase780_round": args.source_phase780_round, "models": {}}
    specs = subspace_specs(parse_csv(args.subspace_modes), parse_budgets(args.budgets))
    for model_name in MODELS:
        args.model = model_name
        selected = select_surface_cases(model_name, args)
        routes = select_routes(model_name, args)
        if args.max_routes and len(routes) > args.max_routes:
            routes = routes[: args.max_routes]
        payload["models"][model_name] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in specs],
            "routes": routes,
            "attn_source_set_size": args.attn_source_set_size,
            "mlp_source_set_size": args.mlp_source_set_size,
            "max_components_per_kind": args.max_components_per_kind,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-phase786-round", default="main")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--subspace-modes", default="positive,negative")
    parser.add_argument("--budgets", default="1024")
    parser.add_argument("--attn-source-set-size", type=int, default=8)
    parser.add_argument("--mlp-source-set-size", type=int, default=32)
    parser.add_argument("--max-components-per-kind", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.summarize_only:
        write_cross_summary(args.round_name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --dry-run or --summarize-only")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
