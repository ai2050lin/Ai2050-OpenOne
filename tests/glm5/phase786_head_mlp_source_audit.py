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
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight, tensor_from_output  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402
from phase785_positive_negative_subspace_split import (  # noqa: E402
    component_keys_for_routes,
    parse_budgets,
    parse_csv,
    route_vectors,
    selected_dims_for_mode,
    selected_score_sums,
    signed_channel_entries,
)


OUT_ROOT = Path("results/glm5_phase786_head_mlp_source_audit")
RESULT_ROOT = Path("tests/result/phase786_head_mlp_source_audit")


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


def top_fraction(scores: torch.Tensor, k: int) -> float | None:
    if scores.numel() == 0:
        return None
    vals = scores.detach().float().abs()
    denom = float(vals.sum().item())
    if denom <= 0:
        return None
    kk = min(int(k), int(vals.numel()))
    return float(torch.topk(vals, kk).values.sum().item() / denom)


def infer_num_heads(model, attn) -> int | None:
    for name in ("num_heads", "num_attention_heads", "n_heads"):
        val = getattr(attn, name, None)
        if val:
            return int(val)
    cfg = getattr(model, "config", None)
    for name in ("num_attention_heads", "n_head", "num_heads"):
        val = getattr(cfg, name, None)
        if val:
            return int(val)
    return None


def source_hook_keys_for_routes(routes: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return component_keys_for_routes(routes)


def capture_answer_outputs_and_sources(
    model,
    tokenizer,
    device,
    prompt: str,
    component_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_pos = len(ids) - 1
    layers = get_layers(model)
    outputs: dict[tuple[str, int], torch.Tensor] = {}
    attn_o_inputs: dict[tuple[str, int], torch.Tensor] = {}
    mlp_down_inputs: dict[tuple[str, int], torch.Tensor] = {}
    handles = []
    for kind, layer_idx in sorted(component_keys):
        layer = layers[layer_idx]
        if kind == "attn":
            module = layer.self_attn
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, key=(kind, layer_idx): outputs.__setitem__(
                        key,
                        tensor_from_output(output)[0, answer_pos].detach().float().cpu()
                        if tensor_from_output(output) is not None
                        else torch.empty(0),
                    )
                )
            )

            def o_pre_hook(_module, inputs, key=(kind, layer_idx)):
                if inputs and torch.is_tensor(inputs[0]):
                    attn_o_inputs[key] = inputs[0][0, answer_pos].detach().float().cpu()

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(o_pre_hook))
        elif kind == "mlp":
            module = layer.mlp
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, key=(kind, layer_idx): outputs.__setitem__(
                        key,
                        tensor_from_output(output)[0, answer_pos].detach().float().cpu()
                        if tensor_from_output(output) is not None
                        else torch.empty(0),
                    )
                )
            )
            if hasattr(layer.mlp, "down_proj"):

                def down_pre_hook(_module, inputs, key=(kind, layer_idx)):
                    if inputs and torch.is_tensor(inputs[0]):
                        mlp_down_inputs[key] = inputs[0][0, answer_pos].detach().float().cpu()

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
    outputs = {k: v for k, v in outputs.items() if v.numel() > 0}
    return {
        "ids": ids,
        "answer_pos": answer_pos,
        "logits": logits,
        "components": outputs,
        "attn_o_inputs": attn_o_inputs,
        "mlp_down_inputs": mlp_down_inputs,
    }


def component_selected_score_sum(entries: list[dict[str, Any]], key: tuple[str, int], dims: torch.Tensor) -> dict[str, float]:
    wanted = {(key, int(d)) for d in dims.tolist()}
    signed = 0.0
    pos = 0.0
    neg_abs = 0.0
    abs_sum = 0.0
    for item in entries:
        if (item["key"], int(item["dim"])) not in wanted:
            continue
        signed += float(item["signed_score"])
        pos += float(item["positive_score"])
        neg_abs += float(item["negative_abs_score"])
        abs_sum += float(item["abs_score"])
    return {
        "component_selected_signed_score_sum": signed,
        "component_selected_positive_score_sum": pos,
        "component_selected_negative_abs_score_sum": neg_abs,
        "component_selected_abs_score_sum": abs_sum,
    }


def audit_attention_heads(
    model,
    route: dict[str, Any],
    key: tuple[str, int],
    selected_dims: torch.Tensor,
    base_state: dict[str, Any],
    donor_state: dict[str, Any],
    readout_direction: torch.Tensor,
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    case: dict[str, Any],
    subspace_mode: str,
    budget_label: str,
    budget: int | None,
) -> list[dict[str, Any]]:
    layer_idx = key[1]
    layer = get_layers(model)[layer_idx]
    attn = layer.self_attn
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
    delta_pre = (pre_donor - pre_base).float()
    dims = selected_dims.long()
    component_scores = component_selected_score_sum(entries, key, dims)
    selected_readout = readout_direction[dims].float()
    rows = []
    per_head_scores = []
    for head_id in range(n_heads):
        start = head_id * head_dim
        end = start + head_dim
        projected = torch.matmul(weight[dims, start:end], delta_pre[start:end])
        per_dim_scores = projected * selected_readout
        signed = float(per_dim_scores.sum().item())
        pos = float(torch.clamp(per_dim_scores, min=0.0).sum().item())
        neg_abs = float(torch.clamp(-per_dim_scores, min=0.0).sum().item())
        abs_sum = float(per_dim_scores.abs().sum().item())
        per_head_scores.append(abs_sum)
        rows.append(
            {
                "row_kind": "phase786_source_audit",
                "source_kind": "attention_head_o_proj",
                "model": args.model,
                "round": args.round_name,
                "case_id": case["case_id"],
                "domain": case.get("domain"),
                "relation": case.get("relation"),
                "route_id": route["route_id"],
                "compare_variant": route["compare_variant"],
                "component_kind": "attn",
                "layer": layer_idx,
                "component_label": f"attn:L{layer_idx}",
                "subspace_mode": subspace_mode,
                "budget_label": budget_label,
                "budget_requested": budget,
                "selected_output_dim_count": int(dims.numel()),
                "num_heads": n_heads,
                "head_dim": head_dim,
                "head_id": head_id,
                "source_channel_id": None,
                "source_signed_score": signed,
                "source_positive_score": pos,
                "source_negative_abs_score": neg_abs,
                "source_abs_score": abs_sum,
                **component_scores,
            }
        )
    abs_tensor = torch.tensor(per_head_scores, dtype=torch.float32)
    for row in rows:
        row["head_top1_abs_fraction"] = top_fraction(abs_tensor, 1)
        row["head_top3_abs_fraction"] = top_fraction(abs_tensor, min(3, n_heads))
        row["head_top8_abs_fraction"] = top_fraction(abs_tensor, min(8, n_heads))
    return rows


def audit_mlp_channels(
    model,
    route: dict[str, Any],
    key: tuple[str, int],
    selected_dims: torch.Tensor,
    base_state: dict[str, Any],
    donor_state: dict[str, Any],
    readout_direction: torch.Tensor,
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    case: dict[str, Any],
    subspace_mode: str,
    budget_label: str,
    budget: int | None,
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
    abs_scores = channel_scores.abs()
    component_scores = component_selected_score_sum(entries, key, dims)
    top_n = min(int(args.top_mlp_channels), int(abs_scores.numel()))
    top_vals, top_ids = torch.topk(abs_scores, top_n)
    rows = []
    for rank, (abs_val, channel_id_tensor) in enumerate(zip(top_vals.tolist(), top_ids.tolist()), 1):
        channel_id = int(channel_id_tensor)
        signed = float(channel_scores[channel_id].item())
        rows.append(
            {
                "row_kind": "phase786_source_audit",
                "source_kind": "mlp_down_input_channel",
                "model": args.model,
                "round": args.round_name,
                "case_id": case["case_id"],
                "domain": case.get("domain"),
                "relation": case.get("relation"),
                "route_id": route["route_id"],
                "compare_variant": route["compare_variant"],
                "component_kind": "mlp",
                "layer": layer_idx,
                "component_label": f"mlp:L{layer_idx}",
                "subspace_mode": subspace_mode,
                "budget_label": budget_label,
                "budget_requested": budget,
                "selected_output_dim_count": int(dims.numel()),
                "num_heads": None,
                "head_dim": None,
                "head_id": None,
                "source_channel_rank": rank,
                "source_channel_id": channel_id,
                "source_signed_score": signed,
                "source_positive_score": max(signed, 0.0),
                "source_negative_abs_score": max(-signed, 0.0),
                "source_abs_score": float(abs_val),
                "mlp_intermediate_size": int(abs_scores.numel()),
                "mlp_top1_abs_fraction": top_fraction(abs_scores, 1),
                "mlp_top8_abs_fraction": top_fraction(abs_scores, min(8, int(abs_scores.numel()))),
                "mlp_top32_abs_fraction": top_fraction(abs_scores, min(32, int(abs_scores.numel()))),
                **component_scores,
            }
        )
    return rows


def subspace_specs(modes: list[str], budgets: list[int]) -> list[tuple[str, str, int | None]]:
    specs = []
    for mode in modes:
        if mode in {"all_positive", "all_negative", "all"}:
            specs.append((mode, mode, None))
        else:
            for budget in budgets:
                specs.append((mode, f"{mode}_{budget}", budget))
    return specs


def route_audit_rows(
    model,
    route: dict[str, Any],
    case: dict[str, Any],
    base_state: dict[str, Any],
    donor_state: dict[str, Any],
    unembed: torch.Tensor,
    source_row: dict[str, Any],
    specs: list[tuple[str, str, int | None]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    base_vecs = route_vectors(base_state, route)
    donor_vecs = route_vectors(donor_state, route)
    if base_vecs is None or donor_vecs is None:
        return []
    case_variant_token_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id"))
    # Phase 785 uses the target-vs-case readout direction, so reuse it exactly.
    target_id = int(source_row.get("target_token_id") or source_row.get("target_id") or 0)
    if not target_id:
        # fall back to the answer string's first token id when older rows do not carry target_token_id.
        raise ValueError(f"source row lacks target_token_id: {source_row.keys()}")
    readout_direction = unembed[target_id].float() - unembed[case_variant_token_id].float()
    entries, _route_meta = signed_channel_entries(route, base_vecs, donor_vecs, readout_direction)
    rows: list[dict[str, Any]] = []
    seed_parts = (args.model, args.round_name, case["case_id"], route["route_id"])
    for mode, budget_label, budget in specs:
        selected = selected_dims_for_mode(mode, budget, route, base_vecs, entries, seed_parts)
        if not selected:
            continue
        total_score_sums = selected_score_sums(selected, entries)
        for key, dims in selected.items():
            if key[0] == "attn":
                subrows = audit_attention_heads(
                    model,
                    route,
                    key,
                    dims,
                    base_state,
                    donor_state,
                    readout_direction,
                    entries,
                    args,
                    case,
                    mode,
                    budget_label,
                    budget,
                )
            elif key[0] == "mlp":
                subrows = audit_mlp_channels(
                    model,
                    route,
                    key,
                    dims,
                    base_state,
                    donor_state,
                    readout_direction,
                    entries,
                    args,
                    case,
                    mode,
                    budget_label,
                    budget,
                )
            else:
                subrows = []
            for row in subrows:
                row.update(total_score_sums)
                row["total_selected_dim_count"] = sum(int(v.numel()) for v in selected.values())
            rows.extend(subrows)
    return rows


def select_target_token_id_from_row(source_row: dict[str, Any]) -> int | None:
    for key in ("target_token_id", "correct_token_id", "target_id"):
        val = source_row.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def enrich_selected_rows_with_target_id(tokenizer, selected: list[dict[str, Any]], cmap: dict[str, dict[str, Any]]) -> None:
    for row in selected:
        if select_target_token_id_from_row(row) is not None:
            continue
        case = cmap[row["case_id"]]
        target = str(case.get("target") or case.get("target_value") or case.get("answer") or "")
        toks = tokenizer.encode(target, add_special_tokens=False)
        if toks:
            row["target_token_id"] = int(toks[0])


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = select_surface_cases(args.model, args)
    routes = select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    component_keys = source_hook_keys_for_routes(routes)
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
            baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
            base_state = capture_answer_outputs_and_sources(model, tokenizer, device, baseline_prompt, component_keys)
            donor_states: dict[str, dict[str, Any]] = {}
            for route in routes:
                donor_variant = route["compare_variant"]
                if donor_variant not in donor_states:
                    donor_prompt = surface_prompt_for_variant(case, donor_variant)
                    donor_states[donor_variant] = capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
                rows.extend(route_audit_rows(model, route, case, base_state, donor_states[donor_variant], unembed, source_row, specs, args))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: source audit {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_rows(rows, args, attn_impl, routes, specs)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase786_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase786_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "top_attention_heads": summary["top_attention_heads"][:8],
                "top_mlp_channels": summary["top_mlp_channels"][:8],
                "concentration_summary": summary["concentration_summary"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


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
                "mean_selected_output_dim_count": safe_mean([r.get("selected_output_dim_count") for r in items]),
                "mean_source_signed_score": safe_mean([r.get("source_signed_score") for r in items]),
                "mean_source_positive_score": safe_mean([r.get("source_positive_score") for r in items]),
                "mean_source_negative_abs_score": safe_mean([r.get("source_negative_abs_score") for r in items]),
                "mean_source_abs_score": safe_mean([r.get("source_abs_score") for r in items]),
                "mean_component_selected_signed_score_sum": safe_mean([r.get("component_selected_signed_score_sum") for r in items]),
                "mean_component_selected_abs_score_sum": safe_mean([r.get("component_selected_abs_score_sum") for r in items]),
                "mean_head_top1_abs_fraction": safe_mean([r.get("head_top1_abs_fraction") for r in items]),
                "mean_head_top3_abs_fraction": safe_mean([r.get("head_top3_abs_fraction") for r in items]),
                "mean_head_top8_abs_fraction": safe_mean([r.get("head_top8_abs_fraction") for r in items]),
                "mean_mlp_top1_abs_fraction": safe_mean([r.get("mlp_top1_abs_fraction") for r in items]),
                "mean_mlp_top8_abs_fraction": safe_mean([r.get("mlp_top8_abs_fraction") for r in items]),
                "mean_mlp_top32_abs_fraction": safe_mean([r.get("mlp_top32_abs_fraction") for r in items]),
                "source_positive_rate": safe_rate([(r.get("source_signed_score") or 0.0) > 0 for r in items]),
                "domains": dict(Counter(str(r.get("domain")) for r in items)),
            }
        )
        out.append(payload)
    return out


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], specs: list[tuple[str, str, int | None]]) -> dict[str, Any]:
    by_source = group_rows(
        rows,
        ["model", "source_kind", "route_id", "component_label", "subspace_mode", "budget_label"],
    )
    by_attention_head = group_rows(
        [r for r in rows if r.get("source_kind") == "attention_head_o_proj"],
        ["model", "route_id", "component_label", "subspace_mode", "budget_label", "head_id"],
    )
    by_mlp_channel = group_rows(
        [r for r in rows if r.get("source_kind") == "mlp_down_input_channel"],
        ["model", "route_id", "component_label", "subspace_mode", "budget_label", "source_channel_id"],
    )
    top_attention = sorted(
        by_attention_head,
        key=lambda r: (r.get("mean_source_abs_score") or 0.0, abs(r.get("mean_source_signed_score") or 0.0)),
        reverse=True,
    )
    top_mlp = sorted(
        by_mlp_channel,
        key=lambda r: (r.get("mean_source_abs_score") or 0.0, abs(r.get("mean_source_signed_score") or 0.0)),
        reverse=True,
    )
    concentration_summary = group_rows(
        rows,
        ["model", "source_kind", "subspace_mode", "budget_label"],
    )
    return {
        "phase": 786,
        "title": "Head Projection and MLP Activation-Channel Source Audit",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "source_phase780_round": args.source_phase780_round,
        "source_phase785_round": args.source_phase785_round,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "subspace_modes": parse_csv(args.subspace_modes),
        "budgets": parse_budgets(args.budgets),
        "routes": routes,
        "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in specs],
        "method_note": (
            "For attention components, decompose selected output dimensions through o_proj input head slices. "
            "For MLP components, decompose selected output dimensions through down_proj input activation channels. "
            "This is a source attribution audit, not yet a head/neuron causal patch."
        ),
        "by_source": by_source,
        "by_attention_head": by_attention_head,
        "by_mlp_channel": by_mlp_channel,
        "top_attention_heads": top_attention,
        "top_mlp_channels": top_mlp,
        "concentration_summary": concentration_summary,
        "strict_interpretation": (
            "Concentration indicates where Phase785 signed subspaces project through known architecture. "
            "It does not prove natural causal necessity until follow-up head/channel interventions are run."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 786 Head Projection and MLP Activation-Channel Source Audit ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Attention evidence: o_proj input is split by head and projected to selected D+/D- output dimensions.",
        "- MLP evidence: down_proj input activation channels are projected to selected D+/D- output dimensions.",
        "- Strict interpretation: source attribution, not causal ablation yet.",
        "",
        "## Concentration Summary",
        "",
        "| model | source | subspace | budget | n | cases | head top1 | head top3 | head top8 | mlp top1 | mlp top8 | mlp top32 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("concentration_summary", []):
            lines.append(
                f"| {model_name} | `{row['source_kind']}` | `{row['subspace_mode']}` | `{row['budget_label']}` | {row['n']} | {row['case_n']} | "
                f"{fmt(row.get('mean_head_top1_abs_fraction'))} | {fmt(row.get('mean_head_top3_abs_fraction'))} | {fmt(row.get('mean_head_top8_abs_fraction'))} | "
                f"{fmt(row.get('mean_mlp_top1_abs_fraction'))} | {fmt(row.get('mean_mlp_top8_abs_fraction'))} | {fmt(row.get('mean_mlp_top32_abs_fraction'))} |"
            )
    lines += [
        "",
        "## Top Attention Heads",
        "",
        "| model | route | component | subspace | budget | head | cases | signed | abs | positive rate |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in (data.get("top_attention_heads") or [])[:20]:
            lines.append(
                f"| {model_name} | `{row['route_id']}` | `{row['component_label']}` | `{row['subspace_mode']}` | `{row['budget_label']}` | "
                f"{row['head_id']} | {row['case_n']} | {fmt(row['mean_source_signed_score'])} | {fmt(row['mean_source_abs_score'])} | {fmt(row['source_positive_rate'])} |"
            )
    lines += [
        "",
        "## Top MLP Activation Channels",
        "",
        "| model | route | component | subspace | budget | channel | cases | signed | abs | positive rate |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in (data.get("top_mlp_channels") or [])[:20]:
            lines.append(
                f"| {model_name} | `{row['route_id']}` | `{row['component_label']}` | `{row['subspace_mode']}` | `{row['budget_label']}` | "
                f"{row['source_channel_id']} | {row['case_n']} | {fmt(row['mean_source_signed_score'])} | {fmt(row['mean_source_abs_score'])} | {fmt(row['source_positive_rate'])} |"
            )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "- High concentration supports a route from signed residual subspace to architectural source units.",
        "- It remains attribution evidence until head/channel causal patch or ablation confirms necessity.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_summary(round_name: str) -> dict[str, Any]:
    by_model = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase786_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 786,
        "title": "Head Projection and MLP Activation-Channel Source Audit",
        "round": round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / round_name
        write_json(out_dir / "phase786_cross_model_summary.json", payload)
        write_markdown(out_dir / "phase786_cross_model_summary.md", payload)
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
            "source_audit_note": "attention=o_proj head slices; mlp=down_proj input activation channels",
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-phase785-round", default="main")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="category,edible,grows_on_tree")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--subspace-modes", default="positive,negative,all_positive,all_negative")
    parser.add_argument("--budgets", default="1024")
    parser.add_argument("--top-mlp-channels", type=int, default=16)
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
