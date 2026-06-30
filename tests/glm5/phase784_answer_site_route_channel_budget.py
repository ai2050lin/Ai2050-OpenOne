#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
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
from phase780_surface_form_component_localization import (  # noqa: E402
    COMPARE_BASELINE,
    lm_head_weight,
    observation_for,
    tensor_from_output,
)
from phase781_surface_form_candidate_causal_patch import make_row  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase784_answer_site_route_channel_budget")
RESULT_ROOT = Path("tests/result/phase784_answer_site_route_channel_budget")


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


def parse_budgets(text: str) -> list[int | str]:
    out: list[int | str] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "all":
            out.append("all")
        else:
            val = int(item)
            if val > 0:
                out.append(val)
    seen = set()
    clean: list[int | str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        clean.append(item)
    return clean


def component_keys_for_routes(routes: list[dict[str, Any]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for route in routes:
        for comp in route.get("components") or []:
            out.add((str(comp["component_kind"]), int(comp["layer"])))
    return out


def capture_selected_answer_components_for_prompt(
    model,
    tokenizer,
    device,
    prompt: str,
    component_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    captured: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    for kind, layer_idx in sorted(component_keys):
        layer = layers[layer_idx]
        module = getattr(layer, "self_attn" if kind == "attn" else "mlp")

        def hook(_module, _inputs, output, key=(kind, layer_idx)):
            tensor = tensor_from_output(output)
            if tensor is not None:
                captured[key] = tensor[0, answer_pos].detach().float().cpu()

        handles.append(module.register_forward_hook(hook))
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
    return {"ids": ids, "answer_pos": answer_pos, "logits": logits, "components": captured}


def observe_logits(
    tokenizer,
    logits: torch.Tensor,
    case: dict[str, Any],
    prompt_variant: str,
    source_row: dict[str, Any],
    case_variant_token_id: int,
    top_k: int,
) -> dict[str, Any]:
    obs, _top = observation_for(tokenizer, logits, case, prompt_variant, source_row, case_variant_token_id, top_k)
    return obs


def route_vectors(state: dict[str, Any], route: dict[str, Any]) -> dict[tuple[str, int], torch.Tensor] | None:
    out: dict[tuple[str, int], torch.Tensor] = {}
    for comp in route["components"]:
        key = (str(comp["component_kind"]), int(comp["layer"]))
        vec = state["components"].get(key)
        if vec is None:
            return None
        out[key] = vec
    return out


def ranked_channel_entries(
    route: dict[str, Any],
    base_vecs: dict[tuple[str, int], torch.Tensor],
    donor_vecs: dict[tuple[str, int], torch.Tensor],
    readout_direction: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    total_dims = 0
    total_positive = 0.0
    total_abs = 0.0
    per_component_positive: dict[str, float] = {}
    hidden_dim = None
    for comp in route["components"]:
        key = (str(comp["component_kind"]), int(comp["layer"]))
        base = base_vecs[key].float()
        donor = donor_vecs[key].float()
        delta = donor - base
        if hidden_dim is None:
            hidden_dim = int(delta.numel())
        scores = delta * readout_direction
        total_dims += int(scores.numel())
        pos = torch.clamp(scores, min=0.0)
        abs_scores = scores.abs()
        total_positive += float(pos.sum().item())
        total_abs += float(abs_scores.sum().item())
        comp_label = f"{key[0]}:L{key[1]}"
        per_component_positive[comp_label] = float(pos.sum().item())
        positive_idx = torch.nonzero(pos > 0, as_tuple=False).flatten()
        for idx in positive_idx.tolist():
            entries.append(
                {
                    "key": key,
                    "dim": int(idx),
                    "score": float(pos[int(idx)].item()),
                    "signed_score": float(scores[int(idx)].item()),
                    "abs_score": float(abs_scores[int(idx)].item()),
                }
            )
    entries.sort(key=lambda row: row["score"], reverse=True)
    meta = {
        "total_dims": total_dims,
        "hidden_dim": hidden_dim,
        "positive_dim_count": len(entries),
        "total_positive_score": total_positive,
        "total_abs_score": total_abs,
        "per_component_positive_score": per_component_positive,
    }
    return entries, meta


def selected_dims_for_budget(
    budget: int | str,
    route: dict[str, Any],
    base_vecs: dict[tuple[str, int], torch.Tensor],
    entries: list[dict[str, Any]],
) -> dict[tuple[str, int], torch.Tensor]:
    selected: dict[tuple[str, int], list[int]] = defaultdict(list)
    if budget == "all":
        for comp in route["components"]:
            key = (str(comp["component_kind"]), int(comp["layer"]))
            selected[key] = list(range(int(base_vecs[key].numel())))
    else:
        for item in entries[: int(budget)]:
            selected[item["key"]].append(int(item["dim"]))
    return {key: torch.tensor(sorted(set(dims)), dtype=torch.long) for key, dims in selected.items() if dims}


def selected_score_sum(selected: dict[tuple[str, int], torch.Tensor], entries: list[dict[str, Any]]) -> float:
    pairs = {(key, int(dim)) for key, dims in selected.items() for dim in dims.tolist()}
    return sum(float(item["score"]) for item in entries if (item["key"], int(item["dim"])) in pairs)


def component_dim_counts(selected: dict[tuple[str, int], torch.Tensor]) -> dict[str, int]:
    return {f"{key[0]}:L{key[1]}": int(dims.numel()) for key, dims in sorted(selected.items(), key=lambda x: (x[0][0], x[0][1]))}


def run_with_dim_route_replacement(
    model,
    tokenizer,
    device,
    prompt: str,
    source_vecs: dict[tuple[str, int], torch.Tensor],
    selected_dims: dict[tuple[str, int], torch.Tensor],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    handles = []
    for (kind, layer_idx), dims in selected_dims.items():
        layer = layers[layer_idx]
        module = getattr(layer, "self_attn" if kind == "attn" else "mlp")
        source_vec = source_vecs[(kind, layer_idx)]

        def hook(_module, _inputs, output, dims=dims, source_vec=source_vec):
            tensor = tensor_from_output(output)
            if tensor is None:
                return output
            patched = tensor.clone()
            idx = dims.to(device=patched.device)
            value = source_vec.to(device=patched.device, dtype=patched.dtype)
            patched[0, answer_pos, idx] = value[idx]
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handles.append(module.register_forward_hook(hook))
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


def make_budget_row(
    model_name: str,
    route: dict[str, Any],
    case: dict[str, Any],
    prompt_variant: str,
    intervention_kind: str,
    obs: dict[str, Any],
    reference_obs: dict[str, Any] | None,
    budget_label: str,
    budget_requested: int | None,
    selected_dims: dict[tuple[str, int], torch.Tensor] | None,
    route_meta: dict[str, Any],
    selected_positive_score: float | None,
) -> dict[str, Any]:
    row = make_row(model_name, route, case, prompt_variant, intervention_kind, obs, reference_obs)
    selected_dims = selected_dims or {}
    actual_dim_count = sum(int(v.numel()) for v in selected_dims.values())
    total_dims = int(route_meta.get("total_dims") or 0)
    total_positive_score = float(route_meta.get("total_positive_score") or 0.0)
    row.update(
        {
            "row_kind": "phase784_answer_site_channel_budget_observation",
            "route_id": route["route_id"],
            "route_size": route["route_size"],
            "component_labels": route["component_labels"],
            "components": route["components"],
            "position_scope": "answer_site",
            "budget_label": budget_label,
            "budget_requested": budget_requested,
            "actual_dim_count": actual_dim_count,
            "budget_fraction": (actual_dim_count / total_dims) if total_dims else None,
            "selected_positive_score": selected_positive_score,
            "positive_score_coverage": (
                (float(selected_positive_score) / total_positive_score)
                if selected_positive_score is not None and total_positive_score > 0
                else None
            ),
            "total_route_dims": total_dims,
            "hidden_dim": route_meta.get("hidden_dim"),
            "positive_dim_count": route_meta.get("positive_dim_count"),
            "total_positive_score": total_positive_score,
            "total_abs_score": route_meta.get("total_abs_score"),
            "component_dim_counts": component_dim_counts(selected_dims),
            "per_component_positive_score": route_meta.get("per_component_positive_score"),
        }
    )
    return row


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
    budgets: list[int | str],
) -> list[dict[str, Any]]:
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_variant = route["compare_variant"]
    donor_prompt = surface_prompt_for_variant(case, donor_variant)

    base_state = capture_selected_answer_components_for_prompt(model, tokenizer, device, baseline_prompt, component_keys)
    donor_state = capture_selected_answer_components_for_prompt(model, tokenizer, device, donor_prompt, component_keys)
    base_obs = observe_logits(tokenizer, base_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
    donor_obs = observe_logits(tokenizer, donor_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
    base_vecs = route_vectors(base_state, route)
    donor_vecs = route_vectors(donor_state, route)
    if base_vecs is None or donor_vecs is None:
        return []

    target_id = int(base_obs["target_token_id"])
    direction = unembed[target_id].float() - unembed[case_variant_token_id].float()
    entries, route_meta = ranked_channel_entries(route, base_vecs, donor_vecs, direction)

    rows = [
        make_budget_row(args.model, route, case, COMPARE_BASELINE, "normal_baseline", base_obs, None, "reference", None, {}, route_meta, None),
        make_budget_row(args.model, route, case, donor_variant, "normal_donor", donor_obs, None, "reference", None, {}, route_meta, None),
    ]

    for budget in budgets:
        selected = selected_dims_for_budget(budget, route, base_vecs, entries)
        if not selected:
            continue
        score_sum = selected_score_sum(selected, entries)
        label = str(budget)
        requested = None if budget == "all" else int(budget)

        patch_state = run_with_dim_route_replacement(model, tokenizer, device, baseline_prompt, donor_vecs, selected)
        patch_obs = observe_logits(tokenizer, patch_state["logits"], case, COMPARE_BASELINE, source_row, case_variant_token_id, args.top_k)
        rows.append(
            make_budget_row(
                args.model,
                route,
                case,
                COMPARE_BASELINE,
                "patch_baseline_from_donor_channel_budget",
                patch_obs,
                base_obs,
                label,
                requested,
                selected,
                route_meta,
                score_sum,
            )
        )

        replace_state = run_with_dim_route_replacement(model, tokenizer, device, donor_prompt, base_vecs, selected)
        replace_obs = observe_logits(tokenizer, replace_state["logits"], case, donor_variant, source_row, case_variant_token_id, args.top_k)
        rows.append(
            make_budget_row(
                args.model,
                route,
                case,
                donor_variant,
                "replace_donor_channel_budget_with_baseline",
                replace_obs,
                donor_obs,
                label,
                requested,
                selected,
                route_meta,
                score_sum,
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
                "component_labels": items[0].get("component_labels"),
                "strict_open_rate": safe_rate([r.get("target_top1") for r in items]),
                "semantic_equiv_open_rate": safe_rate([r.get("semantic_equiv_open") for r in items]),
                "pool_top1_rate": safe_rate([r.get("pool_target_top1") for r in items]),
                "mean_margin_target_vs_case_variant": safe_mean([r.get("margin_target_vs_case_variant") for r in items]),
                "mean_target_rank": safe_mean([r.get("target_rank") for r in items]),
                "mean_delta_margin_vs_reference": safe_mean([r.get("delta_margin_vs_reference") for r in items]),
                "strict_gain_rate_vs_reference": safe_rate([r.get("strict_gain_vs_reference") for r in items if "strict_gain_vs_reference" in r]),
                "strict_loss_rate_vs_reference": safe_rate([r.get("strict_loss_vs_reference") for r in items if "strict_loss_vs_reference" in r]),
                "semantic_loss_rate_vs_reference": safe_rate([r.get("semantic_loss_vs_reference") for r in items if "semantic_loss_vs_reference" in r]),
                "pool_loss_rate_vs_reference": safe_rate([r.get("pool_loss_vs_reference") for r in items if "pool_loss_vs_reference" in r]),
                "mean_actual_dim_count": safe_mean([r.get("actual_dim_count") for r in items]),
                "mean_budget_fraction": safe_mean([r.get("budget_fraction") for r in items]),
                "mean_positive_score_coverage": safe_mean([r.get("positive_score_coverage") for r in items]),
                "mean_positive_dim_count": safe_mean([r.get("positive_dim_count") for r in items]),
                "mean_total_route_dims": safe_mean([r.get("total_route_dims") for r in items]),
                "top1_classes": dict(Counter(str(r.get("top1_competitor_class")) for r in items)),
            }
        )
        payload["sufficiency_score"] = (
            (payload["mean_delta_margin_vs_reference"] or 0.0) * (payload["strict_gain_rate_vs_reference"] or 0.0)
            if payload.get("intervention_kind") == "patch_baseline_from_donor_channel_budget"
            else 0.0
        )
        payload["necessity_score"] = (
            (-(payload["mean_delta_margin_vs_reference"] or 0.0))
            * ((payload["strict_loss_rate_vs_reference"] or 0.0) + 0.5 * (payload["semantic_loss_rate_vs_reference"] or 0.0))
            if payload.get("intervention_kind") == "replace_donor_channel_budget_with_baseline"
            else 0.0
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("model") or "",
            r.get("route_id") or "",
            r.get("intervention_kind") or "",
            float(r.get("mean_actual_dim_count") or 0.0),
        )
    )
    return out


def add_full_budget_ratio(by_budget: list[dict[str, Any]]) -> None:
    full: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in by_budget:
        if row.get("intervention_kind") == "patch_baseline_from_donor_channel_budget" and row.get("budget_label") == "all":
            key = (row.get("model"), row.get("route_id"), row.get("compare_variant"), row.get("route_size"))
            full[key] = row
    for row in by_budget:
        if row.get("intervention_kind") != "patch_baseline_from_donor_channel_budget":
            continue
        key = (row.get("model"), row.get("route_id"), row.get("compare_variant"), row.get("route_size"))
        ref = full.get(key)
        if not ref:
            continue
        full_gain = ref.get("strict_gain_rate_vs_reference") or 0.0
        full_margin = ref.get("mean_delta_margin_vs_reference") or 0.0
        row["strict_gain_fraction_of_full"] = (
            (row.get("strict_gain_rate_vs_reference") or 0.0) / full_gain if full_gain else None
        )
        row["margin_gain_fraction_of_full"] = (
            (row.get("mean_delta_margin_vs_reference") or 0.0) / full_margin if full_margin else None
        )


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], budgets: list[int | str]) -> dict[str, Any]:
    by_budget = group_rows(
        rows,
        ["model", "route_id", "compare_variant", "route_size", "budget_label", "intervention_kind"],
    )
    add_full_budget_ratio(by_budget)
    top_suff = sorted(
        [r for r in by_budget if r.get("intervention_kind") == "patch_baseline_from_donor_channel_budget"],
        key=lambda r: (
            r.get("sufficiency_score") or 0.0,
            r.get("strict_gain_rate_vs_reference") or 0.0,
            r.get("mean_delta_margin_vs_reference") or 0.0,
        ),
        reverse=True,
    )
    top_low_budget = sorted(
        [
            r
            for r in by_budget
            if r.get("intervention_kind") == "patch_baseline_from_donor_channel_budget"
            and r.get("budget_label") != "all"
            and (r.get("strict_gain_rate_vs_reference") or 0.0) > 0.0
        ],
        key=lambda r: (
            -(r.get("mean_actual_dim_count") or 0.0),
            r.get("strict_gain_rate_vs_reference") or 0.0,
            r.get("mean_delta_margin_vs_reference") or 0.0,
        ),
        reverse=True,
    )
    top_nec = sorted(
        [r for r in by_budget if r.get("intervention_kind") == "replace_donor_channel_budget_with_baseline"],
        key=lambda r: (r.get("necessity_score") or 0.0, -(r.get("mean_delta_margin_vs_reference") or 0.0)),
        reverse=True,
    )
    return {
        "phase": 784,
        "title": "Answer-Site Route Channel Budget",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase780_round": args.source_phase780_round,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "budgets": [str(x) for x in budgets],
        "routes": routes,
        "method_note": (
            "At answer_site, rank route component dimensions by positive direct contribution to target-vs-source-token "
            "readout direction, then patch only the selected dimensions. This is channel/subspace budget evidence, "
            "not biological neuron proof."
        ),
        "by_budget_intervention": by_budget,
        "top_sufficiency_budgets": top_suff,
        "top_low_budget_successes": top_low_budget,
        "top_necessity_budgets": top_nec,
        "strict_interpretation": (
            "If small budgets recover most of full-route strict gain, the answer-site route has sparse/channel-like support. "
            "If only all-dim patch works, the route is distributed at current block-output granularity."
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
    budgets = parse_budgets(args.budgets)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)} budgets={budgets}")
    cmap = case_map_for(args)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys, budgets))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: answer-site channel budget {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_rows(rows, args, attn_impl, routes, budgets)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase784_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase784_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_routes": summary["n_routes"],
                "top_sufficiency_budgets": summary["top_sufficiency_budgets"][:8],
                "top_low_budget_successes": summary["top_low_budget_successes"][:8],
                "top_necessity_budgets": summary["top_necessity_budgets"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 784 Answer-Site Route Channel Budget ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Test: answer-site route channel/subspace budget.",
        "- Models are run sequentially; bf16, quantization off; attention implementation prefers flash/sdpa/eager.",
        "- Strict interpretation: block-output dimension budget, not final head/neuron atlas.",
        "",
        "## Routes",
        "",
        "| model | route | compare | size | components |",
        "|---|---|---|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for route in data.get("routes") or []:
            labels = ", ".join(route.get("component_labels") or [])
            lines.append(f"| {model_name} | `{route['route_id']}` | `{route['compare_variant']}` | {route['route_size']} | `{labels}` |")

    lines += [
        "",
        "## Budget Intervention Summary",
        "",
        "| model | route | budget | intervention | cases | dims | frac | score cover | strict gain | delta margin | gain/full | margin/full | top1 classes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("by_budget_intervention") or []:
            if row.get("intervention_kind") not in {
                "patch_baseline_from_donor_channel_budget",
                "replace_donor_channel_budget_with_baseline",
            }:
                continue
            lines.append(
                f"| {model_name} | `{row['route_id']}` | `{row['budget_label']}` | `{row['intervention_kind']}` | {row['case_n']} | "
                f"{fmt(row['mean_actual_dim_count'])} | {fmt(row['mean_budget_fraction'])} | {fmt(row['mean_positive_score_coverage'])} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['mean_delta_margin_vs_reference'])} | "
                f"{fmt(row.get('strict_gain_fraction_of_full'))} | {fmt(row.get('margin_gain_fraction_of_full'))} | "
                f"`{json.dumps(row['top1_classes'], ensure_ascii=False)}` |"
            )

    lines += [
        "",
        "## Low-Budget Successes",
        "",
        "| model | route | budget | dims | frac | score cover | strict gain | delta margin |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in (data.get("top_low_budget_successes") or [])[:12]:
            lines.append(
                f"| {model_name} | `{row['route_id']}` | `{row['budget_label']}` | {fmt(row['mean_actual_dim_count'])} | "
                f"{fmt(row['mean_budget_fraction'])} | {fmt(row['mean_positive_score_coverage'])} | "
                f"{fmt(row['strict_gain_rate_vs_reference'])} | {fmt(row['mean_delta_margin_vs_reference'])} |"
            )

    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- `all` should approximate the Phase 782 full answer-site route patch.",
        "- Small-budget success means the readout-side route has sparse/channel-like support under the current ranking rule.",
        "- Small-budget failure means the route is distributed or the ranking rule is incomplete.",
        "- This does not yet identify attention heads or biological neurons.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase784_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 784,
        "title": "Answer-Site Route Channel Budget",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase784_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase784_cross_model_summary.md", payload)
    print(json.dumps({"round": round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = {"round": args.round_name, "source_phase780_round": args.source_phase780_round, "models": {}}
    for model_name in MODELS:
        args.model = model_name
        selected = select_surface_cases(model_name, args)
        routes = select_routes(model_name, args)
        if args.max_routes and len(routes) > args.max_routes:
            routes = routes[: args.max_routes]
        payload["models"][model_name] = {
            "selected_cases": len(selected),
            "domains": dict(Counter(r.get("domain") for r in selected)),
            "budgets": [str(x) for x in parse_budgets(args.budgets)],
            "routes": routes,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--budgets", default="32,128,512,all")
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
