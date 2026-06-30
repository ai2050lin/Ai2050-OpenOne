#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import get_layers, release_model  # noqa: E402
from phase112_attention_transport_head_mapping_cuda import get_attention_module, get_num_heads, get_o_proj  # noqa: E402
from phase132_source_value_contribution_cuda import get_num_kv_heads, get_v_proj  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase776_readout_bridge_competition_audit import load_model_bf16_prefer_flash  # noqa: E402
from phase778_surface_form_normalization_causal_audit import select_surface_cases, surface_prompt_for_variant  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight  # noqa: E402
from phase782_multi_component_surface_route_patch import select_routes  # noqa: E402
from phase785_positive_negative_subspace_split import parse_budgets, parse_csv  # noqa: E402
from phase786_head_mlp_source_audit import capture_answer_outputs_and_sources, component_keys_for_routes, enrich_selected_rows_with_target_id  # noqa: E402
from phase788_matched_source_unit_causal_fiber_validation import route_source_candidates, subspace_specs  # noqa: E402
from phase791_upstream_qkv_source_token_causal_fiber_trace import source_groups_for_prompt  # noqa: E402


OUT_ROOT = Path("results/glm5_phase793_qkvo_independent_causal_decomposition")
RESULT_ROOT = Path("tests/result/phase793_qkvo_independent_causal_decomposition")

DEFAULT_SOURCE_GROUPS = [
    "candidate_tokens",
    "target_value_tokens",
    "instruction",
    "question",
    "all_pre_answer",
]


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


def get_q_proj(attn: Any) -> Any:
    for name in ["q_proj", "query", "query_proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise TypeError(f"Cannot find q projection for {type(attn).__name__}")


def get_k_proj(attn: Any) -> Any:
    for name in ["k_proj", "key", "key_proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise TypeError(f"Cannot find k projection for {type(attn).__name__}")


def source_groups_for(args: argparse.Namespace) -> list[str]:
    if args.source_groups:
        return [x.strip() for x in args.source_groups.split(",") if x.strip()]
    return DEFAULT_SOURCE_GROUPS[: args.max_source_groups]


def kv_heads_for(head_ids: list[int], num_heads: int, num_kv_heads: int) -> list[int]:
    if num_kv_heads <= 0:
        return []
    if num_heads <= 0:
        return []
    if num_heads % num_kv_heads == 0:
        repeat = max(1, num_heads // num_kv_heads)
        return sorted({min(num_kv_heads - 1, max(0, int(h) // repeat)) for h in head_ids})
    return sorted({min(num_kv_heads - 1, max(0, round(int(h) * num_kv_heads / num_heads))) for h in head_ids})


def zero_projection_hook(head_ids: list[int], num_heads: int, positions: list[int]) -> Callable:
    pos = sorted({int(p) for p in positions if int(p) >= 0})
    heads = sorted({int(h) for h in head_ids if 0 <= int(h) < num_heads})

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any):
        if not torch.is_tensor(output) or not pos or not heads:
            return output
        if output.shape[-1] % num_heads != 0:
            raise RuntimeError(f"projection output dim {output.shape[-1]} not divisible by heads {num_heads}")
        y = output.clone()
        head_dim = y.shape[-1] // num_heads
        yv = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        valid_pos = [p for p in pos if p < yv.shape[1]]
        for p in valid_pos:
            for h in heads:
                yv[:, p, h, :] = 0
        return y

    return hook


def zero_oproj_pre_hook(head_ids: list[int], num_heads: int, positions: list[int]) -> Callable:
    pos = sorted({int(p) for p in positions if int(p) >= 0})
    heads = sorted({int(h) for h in head_ids if 0 <= int(h) < num_heads})

    def hook(_module: Any, inputs: tuple[Any, ...]):
        x = inputs[0]
        if not torch.is_tensor(x) or not pos or not heads:
            return inputs
        if x.shape[-1] % num_heads != 0:
            raise RuntimeError(f"o_proj input dim {x.shape[-1]} not divisible by heads {num_heads}")
        y = x.clone()
        head_dim = y.shape[-1] // num_heads
        yv = y.view(y.shape[0], y.shape[1], num_heads, head_dim)
        valid_pos = [p for p in pos if p < yv.shape[1]]
        for p in valid_pos:
            for h in heads:
                yv[:, p, h, :] = 0
        return (y,) + tuple(inputs[1:])

    return hook


def install_qkvo_zero(
    model,
    layer_idx: int,
    head_ids: list[int],
    source_positions: list[int],
    answer_pos: int,
    op: str,
) -> list[Any]:
    attn = get_attention_module(get_layers(model)[layer_idx])
    num_heads = get_num_heads(model, attn)
    num_kv_heads = get_num_kv_heads(model, attn, num_heads)
    heads = [int(h) for h in head_ids if 0 <= int(h) < num_heads]
    if not heads:
        return []
    if op == "q_answer_zero":
        return [get_q_proj(attn).register_forward_hook(zero_projection_hook(heads, num_heads, [answer_pos]))]
    if op == "k_source_zero":
        kv_heads = kv_heads_for(heads, num_heads, num_kv_heads)
        return [get_k_proj(attn).register_forward_hook(zero_projection_hook(kv_heads, num_kv_heads, source_positions))]
    if op == "v_source_zero":
        kv_heads = kv_heads_for(heads, num_heads, num_kv_heads)
        return [get_v_proj(attn).register_forward_hook(zero_projection_hook(kv_heads, num_kv_heads, source_positions))]
    if op == "o_answer_zero":
        return [get_o_proj(attn).register_forward_pre_hook(zero_oproj_pre_hook(heads, num_heads, [answer_pos]))]
    raise ValueError(op)


def run_logits_with_intervention(
    model,
    device,
    ids: list[int],
    install: Callable[[], list[Any]],
) -> tuple[torch.Tensor, str | None]:
    handles: list[Any] = []
    try:
        handles = install()
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu(), None
    except Exception as exc:
        return torch.empty(0), f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()


def target_ids_from_row(tokenizer, case: dict[str, Any], source_row: dict[str, Any]) -> tuple[int, int]:
    target_id = int(source_row.get("target_token_id") or source_row.get("target_id") or 0)
    contrast_id = int(source_row.get("top1_token_id") or source_row.get("source_top1_token_id") or 0)
    if not target_id:
        target_id = int(tokenizer.encode(str(case["answer"]), add_special_tokens=False)[0])
    if not contrast_id or contrast_id == target_id:
        contrast_id = int(tokenizer.encode(str(case.get("contrast_answer") or ""), add_special_tokens=False)[0])
    return target_id, contrast_id


def make_intervention_row(
    args: argparse.Namespace,
    case: dict[str, Any],
    route: dict[str, Any],
    meta: dict[str, Any],
    op: str,
    source_group: str,
    source_positions: list[int],
    base_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    error: str | None,
) -> dict[str, Any]:
    base_target = logit_diag(base_logits, target_id)
    after_target = logit_diag(after_logits, target_id) if after_logits.numel() else {}
    before_margin = margin(base_logits, target_id, contrast_id)
    after_margin = margin(after_logits, target_id, contrast_id) if after_logits.numel() else float("nan")
    base_rank = int(base_target["target_rank"])
    after_rank = int(after_target.get("target_rank", base_rank)) if after_target else None
    return {
        "row_kind": "phase793_qkvo_independent_causal_decomposition",
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "domain": case.get("domain"),
        "relation": case.get("relation"),
        "object": case.get("object"),
        "target_answer": case.get("answer"),
        "route_id": route["route_id"],
        "compare_variant": route["compare_variant"],
        "component_kind": meta["component_kind"],
        "layer": int(meta["layer"]),
        "source_component_label": meta["source_component_label"],
        "source_selection_kind": meta["source_selection_kind"],
        "source_unit_kind": meta["source_unit_kind"],
        "source_set_size": meta["source_set_size"],
        "source_unit_ids": [int(x) for x in meta["source_unit_ids"]],
        "subspace_mode": meta["subspace_mode"],
        "budget_label": meta["budget_label"],
        "intervention_op": op,
        "source_group": source_group,
        "source_positions_n": len(source_positions),
        "target_token_id": target_id,
        "contrast_token_id": contrast_id,
        "base_target_top1": bool(base_target["target_top1"]),
        "after_target_top1": bool(after_target.get("target_top1", False)) if after_target else None,
        "token_top1_loss": bool(base_target["target_top1"]) and bool(after_target) and not bool(after_target.get("target_top1")),
        "base_target_rank": base_rank,
        "after_target_rank": after_rank,
        "target_rank_delta": (after_rank - base_rank) if after_rank is not None else None,
        "target_rank_worse": (after_rank > base_rank) if after_rank is not None else None,
        "target_logit_drop": float(base_logits[target_id].item() - after_logits[target_id].item()) if after_logits.numel() else None,
        "contrast_logit_gain": float(after_logits[contrast_id].item() - base_logits[contrast_id].item()) if after_logits.numel() else None,
        "margin_drop_target_vs_contrast": float(before_margin - after_margin) if after_logits.numel() else None,
        "intervention_error": error,
        "interpretation_boundary": (
            "q_answer_zero and o_answer_zero are answer-position head interventions without source-group specificity; "
            "k_source_zero and v_source_zero are source-position kv-head interventions. Zero-ablation tests necessity-like effects, not donor-to-recipient replacement."
        ),
    }


def attention_candidates(
    model,
    route: dict[str, Any],
    base_answer_state: dict[str, Any],
    donor_answer_state: dict[str, Any],
    readout_direction: torch.Tensor,
    specs: list[tuple[str, str, int | None]],
    args: argparse.Namespace,
    seed_parts: tuple[Any, ...],
) -> list[dict[str, Any]]:
    out = []
    for mode, budget_label, budget in specs:
        rows = route_source_candidates(model, route, base_answer_state, donor_answer_state, readout_direction, mode, budget_label, budget, args, seed_parts)
        for row in rows:
            if row.get("source_unit_kind") != "attention_head_set":
                continue
            if row.get("source_selection_kind") not in {"top", "matched"}:
                continue
            out.append(row)
    return out


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
    source_groups: list[str],
    ops: list[str],
) -> list[dict[str, Any]]:
    target_id, contrast_id = target_ids_from_row(tokenizer, case, source_row)
    baseline_prompt = surface_prompt_for_variant(case, COMPARE_BASELINE)
    donor_prompt = surface_prompt_for_variant(case, route["compare_variant"])
    base_answer_state = capture_answer_outputs_and_sources(model, tokenizer, device, baseline_prompt, component_keys)
    donor_answer_state = capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    readout_direction = unembed[target_id].float() - unembed[contrast_id].float()
    candidates = attention_candidates(
        model,
        route,
        base_answer_state,
        donor_answer_state,
        readout_direction,
        specs,
        args,
        (args.model, args.round_name, case["case_id"], route["route_id"]),
    )
    if not candidates:
        return []
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    answer_pos = len(donor_ids) - 1
    source_groups_map = source_groups_for_prompt(tokenizer, donor_prompt, case, donor_ids)
    base_logits = donor_answer_state["logits"]
    rows: list[dict[str, Any]] = []
    for meta in candidates:
        layer = int(meta["layer"])
        head_ids = [int(x) for x in meta["source_unit_ids"]]
        for op in ops:
            if op in {"q_answer_zero", "o_answer_zero"}:
                op_groups = [(f"{op}:answer_position", [answer_pos])]
            else:
                op_groups = [(g, [int(p) for p in source_groups_map.get(g, [])]) for g in source_groups]
            for source_group, src_positions in op_groups:
                if not src_positions:
                    continue

                def install(layer=layer, head_ids=head_ids, src_positions=src_positions, answer_pos=answer_pos, op=op):
                    return install_qkvo_zero(model, layer, head_ids, src_positions, answer_pos, op)

                after_logits, error = run_logits_with_intervention(model, device, donor_ids, install)
                rows.append(
                    make_intervention_row(
                        args,
                        case,
                        route,
                        meta,
                        op,
                        source_group,
                        src_positions,
                        base_logits,
                        after_logits,
                        target_id,
                        contrast_id,
                        error,
                    )
                )
    return rows


def group_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v["case_id"] for v in vals}),
                "component_counts": dict(Counter(v["source_component_label"] for v in vals)),
                "error_rate": safe_rate([v.get("intervention_error") for v in vals]),
                "mean_margin_drop": safe_mean([v.get("margin_drop_target_vs_contrast") for v in vals]),
                "mean_target_logit_drop": safe_mean([v.get("target_logit_drop") for v in vals]),
                "mean_contrast_logit_gain": safe_mean([v.get("contrast_logit_gain") for v in vals]),
                "token_top1_loss_rate": safe_rate([v.get("token_top1_loss") for v in vals]),
                "target_rank_worse_rate": safe_rate([v.get("target_rank_worse") for v in vals]),
                "mean_target_rank_delta": safe_mean([v.get("target_rank_delta") for v in vals]),
                "base_target_top1_rate": safe_rate([v.get("base_target_top1") for v in vals]),
                "after_target_top1_rate": safe_rate([v.get("after_target_top1") for v in vals if v.get("after_target_top1") is not None]),
            }
        )
        payload["effect_score"] = max(payload["mean_margin_drop"] or 0.0, 0.0) * (1.0 + max(payload["target_rank_worse_rate"] or 0.0, 0.0))
        out.append(payload)
    out.sort(key=lambda r: (r.get("effect_score") or 0.0, r.get("mean_margin_drop") or 0.0), reverse=True)
    return out


def matched_comparisons(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ["model", "intervention_op", "subspace_mode", "budget_label", "source_set_size", "source_group"]
    idx: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    for row in grouped:
        key = tuple(row.get(k) for k in key_fields)
        idx[(key, str(row.get("source_selection_kind")))] = row
    out = []
    for key in sorted({k for k, _sel in idx}):
        top = idx.get((key, "top"))
        matched = idx.get((key, "matched"))
        if not top or not matched:
            continue
        payload = {field: value for field, value in zip(key_fields, key)}
        payload.update(
            {
                "top_n": top.get("n"),
                "matched_n": matched.get("n"),
                "top_mean_margin_drop": top.get("mean_margin_drop"),
                "matched_mean_margin_drop": matched.get("mean_margin_drop"),
                "top_minus_matched_margin_drop": (top.get("mean_margin_drop") or 0.0) - (matched.get("mean_margin_drop") or 0.0),
                "top_token_top1_loss_rate": top.get("token_top1_loss_rate"),
                "matched_token_top1_loss_rate": matched.get("token_top1_loss_rate"),
                "top_minus_matched_top1_loss_rate": (top.get("token_top1_loss_rate") or 0.0) - (matched.get("token_top1_loss_rate") or 0.0),
                "top_rank_worse_rate": top.get("target_rank_worse_rate"),
                "matched_rank_worse_rate": matched.get("target_rank_worse_rate"),
                "top_minus_matched_rank_worse_rate": (top.get("target_rank_worse_rate") or 0.0) - (matched.get("target_rank_worse_rate") or 0.0),
            }
        )
        payload["matched_specificity_score"] = max(payload["top_minus_matched_margin_drop"], 0.0) * (
            1.0 + max(payload["top_minus_matched_rank_worse_rate"], 0.0)
        )
        out.append(payload)
    out.sort(key=lambda r: (r.get("matched_specificity_score") or 0.0, r.get("top_minus_matched_margin_drop") or 0.0), reverse=True)
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], source_groups: list[str], ops: list[str]) -> dict[str, Any]:
    by_op = group_rows(rows, ["model", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "intervention_op", "source_group"])
    by_component = group_rows(rows, ["model", "source_component_label", "source_selection_kind", "intervention_op", "source_group"])
    comparisons = matched_comparisons(by_op)
    return {
        "phase": 793,
        "title": "Q/K/V/O Independent Causal Decomposition and Closure Gate",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "intervention_ops": ops,
        "source_groups": source_groups,
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "by_op": by_op,
        "by_component": by_component,
        "matched_comparisons": comparisons,
        "top_op_effects": by_op[:40],
        "top_matched_specificity": comparisons[:40],
        "strict_boundary": (
            "This phase independently zero-ablates q_proj/k_proj/v_proj/o_proj paths for Phase 788/791 attention source units. "
            "It tests necessity-like effects and token closure gate, not donor-to-recipient replacement or full generation closure."
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
    specs = subspace_specs(parse_csv(args.subspace_modes), parse_budgets(args.budgets))
    source_groups = source_groups_for(args)
    ops = parse_csv(args.intervention_ops)
    log(f"{args.model}/{args.round_name}: selected cases={len(selected)} routes={len(routes)} specs={len(specs)} ops={ops} source_groups={source_groups}")
    cmap = case_map_for(args)
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "domains": dict(Counter(cmap[row["case_id"]]["domain"] for row in selected if row["case_id"] in cmap)),
            "routes": routes,
            "source_groups": source_groups,
            "intervention_ops": ops,
            "intervention_specs": [{"mode": m, "budget_label": label, "budget": b} for m, label, b in specs],
        }
    component_keys = component_keys_for_routes(routes)
    model, tokenizer, device, attn_impl = load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    try:
        enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys, specs, source_groups, ops))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: qkvo independent decomposition {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl, routes, source_groups, ops)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase793_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase793_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "top_matched_specificity": summary["top_matched_specificity"][:8],
                "top_op_effects": summary["top_op_effects"][:8],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return summary


def build_atlas(payload: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(node_id: str, node_type: str, **attrs: Any) -> None:
        nodes[node_id] = {**nodes.get(node_id, {}), "id": node_id, "type": node_type, **attrs}

    task = "phase793:qkvo_independent_causal_decomposition"
    node(task, "task", label="Phase 793 Q/K/V/O independent decomposition")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for row in summary.get("top_matched_specificity", [])[:20]:
            op_node = f"{model_name}:op:{row['intervention_op']}"
            source_node = f"{model_name}:source:{row['source_group']}"
            node(op_node, "qkvo_intervention", label=row["intervention_op"])
            node(source_node, "source_group", label=row["source_group"])
            edges.append(
                {
                    "id": f"{model_name}:{row['intervention_op']}:{row['source_group']}:{row['subspace_mode']}",
                    "source": model_node,
                    "target": op_node,
                    "type": "has_independent_path_effect",
                    "weight": row.get("matched_specificity_score"),
                    "metrics": row,
                }
            )
            edges.append(
                {
                    "id": f"{model_name}:{row['intervention_op']}->{row['source_group']}:{row['subspace_mode']}",
                    "source": op_node,
                    "target": source_node,
                    "type": "source_conditioned" if "source_zero" in row["intervention_op"] else "answer_position_only",
                    "weight": row.get("top_minus_matched_margin_drop"),
                }
            )
    return {
        "schema_version": "atlas_graph_v1",
        "phase": 793,
        "graph": {"nodes": list(nodes.values()), "edges": edges},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 793 Q/K/V/O Independent Causal Decomposition ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Intervention: independent zero-ablation of q_proj/k_proj/v_proj/o_proj paths.",
        "- Q/O are answer-position head interventions; K/V are source-position kv-head interventions.",
        "- This tests necessity-like path effects and token closure gate, not full generation closure.",
        "",
        "## Top Minus Matched Specificity",
        "",
        "| model | op | subspace | source group | top drop | matched drop | drop gap | top1 loss gap | rank worse gap |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_matched_specificity", [])[:20]:
            lines.append(
                f"| {model_name} | `{row['intervention_op']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{fmt(row['top_mean_margin_drop'])} | {fmt(row['matched_mean_margin_drop'])} | "
                f"{fmt(row['top_minus_matched_margin_drop'])} | {fmt(row['top_minus_matched_top1_loss_rate'])} | "
                f"{fmt(row['top_minus_matched_rank_worse_rate'])} |"
            )
    lines += [
        "",
        "## Top Operation Effects",
        "",
        "| model | selection | op | subspace | source group | cases | margin drop | target drop | rank worse | top1 loss |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_op_effects", [])[:24]:
            lines.append(
                f"| {model_name} | `{row['source_selection_kind']}` | `{row['intervention_op']}` | `{row['subspace_mode']}` | `{row['source_group']}` | "
                f"{row['case_n']} | {fmt(row['mean_margin_drop'])} | {fmt(row['mean_target_logit_drop'])} | "
                f"{fmt(row['target_rank_worse_rate'])} | {fmt(row['token_top1_loss_rate'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase793_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 793,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    for root in (OUT_ROOT / round_name, RESULT_ROOT / round_name):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "phase793_cross_model_summary.json", payload)
        write_json(root / "phase793_atlas_graph.json", build_atlas(payload))
        write_markdown(root / "phase793_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="smoke")
    parser.add_argument("--source-phase776-round", default="confirm")
    parser.add_argument("--source-phase780-round", default="confirm")
    parser.add_argument("--source-prompt-variants", default="without_candidate_list,constrained_free_prompt,with_candidate_list")
    parser.add_argument("--relations", default="")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--route-sizes", default="6")
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--min-candidate-score", type=float, default=0.0)
    parser.add_argument("--route-compare-variants", default="with_candidate_list,lowercase_short_value")
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--budgets", default="1024")
    parser.add_argument("--subspace-modes", default="positive")
    parser.add_argument("--attn-source-set-size", type=int, default=4)
    parser.add_argument("--mlp-source-set-size", type=int, default=8)
    parser.add_argument("--max-components-per-kind", type=int, default=1)
    parser.add_argument("--max-source-groups", type=int, default=4)
    parser.add_argument("--source-groups", default="")
    parser.add_argument("--intervention-ops", default="q_answer_zero,k_source_zero,v_source_zero,o_answer_zero")
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_only:
        payload = summarize_round(args.round_name)
        print(json.dumps({"round": args.round_name, "status": payload["status"], "models": payload["models"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only")
    result = run_model(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
