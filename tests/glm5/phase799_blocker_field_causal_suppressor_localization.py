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
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

import phase796_global_competitor_token_identity_audit as p796  # noqa: E402
import phase798_full_vocab_blocker_identity_anchor as p798  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402


RESULT_ROOT = Path("tests/result/phase799_blocker_field_causal_suppressor_localization")
OUT_ROOT = RESULT_ROOT


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
    vals = [bool(v) for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    denom = float(weights.sum().item())
    if denom <= 0:
        return float(values.mean().item())
    return float((values * weights).sum().item() / denom)


def class_suppression_stats(
    args: argparse.Namespace,
    rec_vals: torch.Tensor,
    after_vals: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    rec_blocker_ids: torch.Tensor,
    after_target_logit: float,
) -> dict[str, Any]:
    if rec_blocker_ids.numel() == 0:
        return {"class_counts": {}, "class_mean_suppression": {}, "class_remaining_rate": {}, "class_top_suppression": {}}
    rec_target_logit = float(rec_vals[int(target_id)].item())
    rec_gaps = rec_vals[rec_blocker_ids] - rec_target_logit
    order = torch.argsort(rec_gaps, descending=True)
    limit = max(0, int(args.max_baseline_blocker_classify))
    if limit > 0:
        order = order[:limit]
    counters: Counter[str] = Counter()
    suppressions: dict[str, list[float]] = defaultdict(list)
    remaining: dict[str, list[bool]] = defaultdict(list)
    top_supp: dict[str, dict[str, Any]] = {}
    for idx_tensor in order.tolist():
        tid = int(rec_blocker_ids[int(idx_tensor)].item())
        text = p798.cached_token_text(args, tid)
        cls = p796.classify_competitor(
            args._tokenizer,
            tid,
            text,
            int(target_id),
            int(contrast_id),
            prompt_ids,
            prompt_text,
            candidate_ids,
            case_values,
        )
        suppression = float(rec_vals[tid].item() - after_vals[tid].item())
        counters[cls] += 1
        suppressions[cls].append(suppression)
        remaining[cls].append(bool(float(after_vals[tid].item()) > after_target_logit))
        if cls not in top_supp or suppression > float(top_supp[cls]["suppression"]):
            top_supp[cls] = {
                "token_id": tid,
                "token_text": text,
                "suppression": suppression,
                "after_still_above_target": bool(float(after_vals[tid].item()) > after_target_logit),
            }
    return {
        "class_counts": dict(counters),
        "class_mean_suppression": {cls: safe_mean(vals) for cls, vals in suppressions.items()},
        "class_remaining_rate": {cls: safe_rate(vals) for cls, vals in remaining.items()},
        "class_top_suppression": top_supp,
    }


def blocker_field_response(
    args: argparse.Namespace,
    recipient_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
    target_answer: str,
) -> dict[str, Any]:
    rec_vals = recipient_logits.detach().float().cpu()
    after_vals = after_logits.detach().float().cpu()
    rec_target = float(rec_vals[int(target_id)].item())
    after_target = float(after_vals[int(target_id)].item())
    rec_mask = rec_vals > rec_target
    rec_mask[int(target_id)] = False
    after_mask = after_vals > after_target
    after_mask[int(target_id)] = False
    rec_ids = torch.nonzero(rec_mask, as_tuple=False).flatten()
    after_ids = torch.nonzero(after_mask, as_tuple=False).flatten()

    target_gain = after_target - rec_target
    if rec_ids.numel():
        rec_blocker_delta = after_vals[rec_ids] - rec_vals[rec_ids]
        suppression = -rec_blocker_delta
        rec_gaps = rec_vals[rec_ids] - rec_target
        mean_suppression = float(suppression.mean().item())
        gap_weighted_suppression = weighted_mean(suppression, rec_gaps.clamp_min(0))
        target_relative_lift = float((target_gain + suppression).mean().item())
        positive_suppression_rate = float((suppression > 0).float().mean().item())
        strong_suppression_rate = float((suppression > float(args.strong_suppression_threshold)).float().mean().item())
    else:
        mean_suppression = None
        gap_weighted_suppression = None
        target_relative_lift = None
        positive_suppression_rate = None
        strong_suppression_rate = None

    new_mask = after_mask & ~rec_mask
    resolved_mask = rec_mask & ~after_mask
    persistent_mask = rec_mask & after_mask
    new_count = int(new_mask.sum().item())
    resolved_count = int(resolved_mask.sum().item())
    persistent_count = int(persistent_mask.sum().item())
    rec_count = int(rec_ids.numel())
    after_count = int(after_ids.numel())
    new_rate = new_count / max(after_count, 1)
    resolved_rate = resolved_count / max(rec_count, 1)

    recipient_full = p798.full_vocab_snapshot(
        args,
        recipient_logits,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        target_answer,
    )
    after_full = p798.full_vocab_snapshot(
        args,
        after_logits,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        target_answer,
    )
    rec_anchor_gap = recipient_full.get("surface_target_variant_max_gap")
    after_anchor_gap = after_full.get("surface_target_variant_max_gap")
    rec_anchor_gap0 = float(rec_anchor_gap) if rec_anchor_gap is not None else 0.0
    after_anchor_gap0 = float(after_anchor_gap) if after_anchor_gap is not None else 0.0
    anchor_gap_improvement = rec_anchor_gap0 - after_anchor_gap0

    class_stats = class_suppression_stats(
        args,
        rec_vals,
        after_vals,
        target_id,
        contrast_id,
        prompt_ids,
        prompt_text,
        candidate_ids,
        case_values,
        rec_ids,
        after_target,
    )
    base_score = (
        float(args.alpha_target) * target_gain
        + float(args.beta_anchor) * anchor_gap_improvement
        + float(args.gamma_suppress) * (mean_suppression or 0.0)
        - float(args.lambda_new_blocker) * new_rate
    )
    pressure_score = base_score / (1.0 + math.log1p(max(after_count, 0)))
    return {
        "baseline_full_blocker_count": rec_count,
        "after_full_blocker_count": after_count,
        "full_blocker_count_delta": after_count - rec_count,
        "new_blocker_count": new_count,
        "new_blocker_rate": new_rate,
        "resolved_baseline_blocker_count": resolved_count,
        "resolved_baseline_blocker_rate": resolved_rate,
        "persistent_baseline_blocker_count": persistent_count,
        "target_logit_gain": target_gain,
        "baseline_blocker_mean_suppression": mean_suppression,
        "baseline_blocker_gap_weighted_suppression": gap_weighted_suppression,
        "baseline_blocker_target_relative_lift": target_relative_lift,
        "baseline_blocker_positive_suppression_rate": positive_suppression_rate,
        "baseline_blocker_strong_suppression_rate": strong_suppression_rate,
        "recipient_identity_anchor_fragmented_full_vocab": recipient_full.get("identity_anchor_fragmented_full_vocab"),
        "after_identity_anchor_fragmented_full_vocab": after_full.get("identity_anchor_fragmented_full_vocab"),
        "recipient_surface_target_variant_count_above": recipient_full.get("surface_target_variant_count_above"),
        "after_surface_target_variant_count_above": after_full.get("surface_target_variant_count_above"),
        "recipient_surface_target_variant_max_gap": rec_anchor_gap,
        "after_surface_target_variant_max_gap": after_anchor_gap,
        "identity_anchor_gap_improvement": anchor_gap_improvement,
        "after_surface_target_identity_rank": after_full.get("surface_target_identity_rank"),
        "after_full_above_class_counts": after_full.get("full_above_class_counts"),
        "after_full_rank_window": after_full.get("full_rank_window"),
        "baseline_blocker_class_counts": class_stats["class_counts"],
        "baseline_blocker_class_mean_suppression": class_stats["class_mean_suppression"],
        "baseline_blocker_class_remaining_rate": class_stats["class_remaining_rate"],
        "baseline_blocker_class_top_suppression": class_stats["class_top_suppression"],
        "closure_fiber_score": base_score,
        "closure_fiber_pressure_score": pressure_score,
    }


def enhanced_make_audit_row(*call_args: Any, **call_kwargs: Any) -> dict[str, Any]:
    row = p796._phase799_original_make_audit_row(*call_args, **call_kwargs)
    (
        args,
        case,
        _route,
        _route_components,
        _meta,
        _ladder_id,
        _source_group,
        _paired_count,
        _recipient_variant,
        _donor_variant,
        recipient_logits,
        _donor_logits,
        after_logits,
        target_id,
        contrast_id,
        recipient_prompt,
        _donor_prompt,
        recipient_ids,
        _donor_ids,
        recipient_candidate_ids,
        _donor_candidate_ids,
        case_values,
        _error,
    ) = call_args
    response = blocker_field_response(
        args,
        recipient_logits,
        after_logits,
        target_id,
        contrast_id,
        recipient_ids,
        recipient_prompt,
        recipient_candidate_ids,
        case_values,
        str(case.get("answer", "")),
    )
    row.update(
        {
            "row_kind": "phase799_blocker_field_causal_suppressor_localization",
            "phase799_boundary": (
                "This row scores whether a candidate causal fiber suppresses the baseline full-vocabulary blocker field. "
                "It is a localization candidate score, not proof that one unit is a complete suppressor."
            ),
            **response,
        }
    )
    return row


def merge_counter_dicts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get(key) or {})
    return dict(counter)


def mean_nested_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for cls, value in (row.get(key) or {}).items():
            vals[str(cls)].append(value)
    return {cls: safe_mean(items) for cls, items in vals.items()}


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
                "mean_target_logit_gain": safe_mean([v.get("target_logit_gain") for v in vals]),
                "mean_delta_global_margin_vs_recipient": safe_mean([v.get("delta_global_margin_vs_recipient") for v in vals]),
                "mean_baseline_full_blocker_count": safe_mean([v.get("baseline_full_blocker_count") for v in vals]),
                "mean_after_full_blocker_count": safe_mean([v.get("after_full_blocker_count") for v in vals]),
                "mean_full_blocker_count_delta": safe_mean([v.get("full_blocker_count_delta") for v in vals]),
                "mean_new_blocker_rate": safe_mean([v.get("new_blocker_rate") for v in vals]),
                "mean_resolved_baseline_blocker_rate": safe_mean([v.get("resolved_baseline_blocker_rate") for v in vals]),
                "mean_baseline_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_gap_weighted_suppression": safe_mean([v.get("baseline_blocker_gap_weighted_suppression") for v in vals]),
                "mean_target_relative_lift": safe_mean([v.get("baseline_blocker_target_relative_lift") for v in vals]),
                "mean_positive_suppression_rate": safe_mean([v.get("baseline_blocker_positive_suppression_rate") for v in vals]),
                "mean_strong_suppression_rate": safe_mean([v.get("baseline_blocker_strong_suppression_rate") for v in vals]),
                "mean_identity_anchor_gap_improvement": safe_mean([v.get("identity_anchor_gap_improvement") for v in vals]),
                "mean_after_surface_target_identity_rank": safe_mean([v.get("after_surface_target_identity_rank") for v in vals]),
                "identity_anchor_fragmented_after_rate": safe_rate([v.get("after_identity_anchor_fragmented_full_vocab") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "mean_closure_fiber_score": safe_mean([v.get("closure_fiber_score") for v in vals]),
                "mean_closure_fiber_pressure_score": safe_mean([v.get("closure_fiber_pressure_score") for v in vals]),
                "baseline_blocker_class_counts": merge_counter_dicts(vals, "baseline_blocker_class_counts"),
                "after_full_above_class_counts": merge_counter_dicts(vals, "after_full_above_class_counts"),
                "baseline_blocker_class_mean_suppression": mean_nested_metric(vals, "baseline_blocker_class_mean_suppression"),
                "baseline_blocker_class_remaining_rate": mean_nested_metric(vals, "baseline_blocker_class_remaining_rate"),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("token_closure_gain_rate") or 0.0,
            r.get("mean_closure_fiber_pressure_score") or -999.0,
            r.get("mean_baseline_blocker_suppression") or -999.0,
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]], ladders: list[str], source_groups: list[str]) -> dict[str, Any]:
    by_model = group_rows(rows, ["model"])
    by_ladder = group_rows(rows, ["model", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "ladder_id", "source_group"])
    by_component = group_rows(rows, ["model", "source_component_label", "source_selection_kind", "subspace_mode", "budget_label", "source_set_size", "ladder_id", "source_group"])
    by_case = group_rows(rows, ["model", "case_id", "ladder_id", "source_group"])
    return {
        "phase": 799,
        "title": "Blocker-Field Causal Suppressor Localization",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_routes": len(routes),
        "routes": routes,
        "ladders": ladders,
        "source_groups": source_groups,
        "score_weights": {
            "alpha_target": args.alpha_target,
            "beta_anchor": args.beta_anchor,
            "gamma_suppress": args.gamma_suppress,
            "lambda_new_blocker": args.lambda_new_blocker,
        },
        "by_model": by_model,
        "by_ladder": by_ladder,
        "by_component": by_component,
        "by_case": by_case,
        "top_suppressor_candidates": by_component[:80],
        "top_ladder_candidates": by_ladder[:80],
        "strict_boundary": (
            "This phase localizes candidate suppressor fibers by scoring response of the baseline blocker field. "
            "Positive scores are candidates, not proof of a complete neural suppressor or token closure mechanism."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = OUT_ROOT / args.round_name
    result_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    selected = p796.select_surface_cases(args.model, args)
    routes = p796.select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    specs = p796.subspace_specs(p796.parse_csv(args.subspace_modes), p796.parse_budgets(args.budgets))
    source_groups = p796.source_groups_for(args)
    ladders = p796.parse_csv(args.ladders) or p796.DEFAULT_LADDERS
    route_allowed_kinds = set(p796.parse_csv(args.route_component_kinds) or ["attn", "mlp"])
    log(
        f"{args.model}/{args.round_name}: cases={len(selected)} routes={len(routes)} specs={len(specs)} "
        f"ladders={ladders} groups={source_groups}"
    )
    cmap = p796.case_map_for(args)
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "source_groups": source_groups,
            "ladders": ladders,
        }
    component_keys = p796.component_keys_for_routes(routes)
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    setattr(args, "_tokenizer", tokenizer)
    setattr(args, "_token_text_cache", {})
    if not hasattr(p796, "_phase799_original_make_audit_row"):
        p796._phase799_original_make_audit_row = p796.make_audit_row
    p796.make_audit_row = enhanced_make_audit_row
    try:
        p796.enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = p796.lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(
                    p796.audit_case_route(
                        model,
                        tokenizer,
                        device,
                        unembed,
                        args,
                        case,
                        source_row,
                        route,
                        component_keys,
                        specs,
                        ladders,
                        source_groups,
                        route_allowed_kinds,
                    )
                )
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: blocker-field suppressor localization {ci}/{len(selected)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        if hasattr(args, "_tokenizer"):
            delattr(args, "_tokenizer")
        if hasattr(args, "_token_text_cache"):
            delattr(args, "_token_text_cache")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args, attn_impl, routes, ladders, source_groups)
    for root in (out_dir, result_dir):
        write_jsonl(root / f"phase799_{args.model}_rows.jsonl", rows)
        write_json(root / f"phase799_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_model": summary["by_model"][:2],
                "top_suppressor_candidates": summary["top_suppressor_candidates"][:5],
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

    task = "phase799:blocker_field_suppressor_localization"
    node(task, "task", label="Phase 799 blocker-field suppressor localization")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for idx, row in enumerate(summary.get("top_suppressor_candidates", [])[:30]):
            comp = row.get("source_component_label") or f"candidate:{idx}"
            comp_node = f"{model_name}:suppressor_candidate:{comp}:{row.get('ladder_id')}:{row.get('source_group')}"
            node(comp_node, "suppressor_candidate", label=comp, metrics=row)
            edges.append(
                {
                    "id": f"{model_name}:candidate:{idx}",
                    "source": model_node,
                    "target": comp_node,
                    "type": "candidate_suppresses_blocker_field",
                    "weight": row.get("mean_closure_fiber_pressure_score"),
                    "metrics": row,
                }
            )
    return {"schema_version": "atlas_graph_v1", "phase": 799, "graph": {"nodes": list(nodes.values()), "edges": edges}}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 799 Blocker-Field Causal Suppressor Localization ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: scores candidate fibers by target gain, identity-anchor improvement, baseline blocker suppression, and new-blocker penalty.",
        "- This phase gives suppressor candidates, not final token closure.",
        "",
        "## By Model",
        "",
        "| model | rows | cases | target gain | blocker suppression | target-relative lift | new blocker rate | resolved rate | anchor gap | token gain | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        row = (data.get("by_model") or [{}])[0]
        lines.append(
            f"| {model_name} | {row.get('n')} | {row.get('case_n')} | {fmt(row.get('mean_target_logit_gain'))} | "
            f"{fmt(row.get('mean_baseline_blocker_suppression'))} | {fmt(row.get('mean_target_relative_lift'))} | "
            f"{fmt(row.get('mean_new_blocker_rate'))} | {fmt(row.get('mean_resolved_baseline_blocker_rate'))} | "
            f"{fmt(row.get('mean_identity_anchor_gap_improvement'))} | {fmt(row.get('token_closure_gain_rate'))} | "
            f"{fmt(row.get('mean_closure_fiber_pressure_score'))} |"
        )
    lines += [
        "",
        "## Top Suppressor Candidates",
        "",
        "| model | component | selection | ladder | source group | rows | target gain | blocker suppression | target-relative lift | new rate | resolved rate | anchor gap | score |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_suppressor_candidates", [])[:24]:
            lines.append(
                f"| {model_name} | `{row.get('source_component_label')}` | `{row.get('source_selection_kind')}` | `{row.get('ladder_id')}` | `{row.get('source_group')}` | "
                f"{row.get('n')} | {fmt(row.get('mean_target_logit_gain'))} | {fmt(row.get('mean_baseline_blocker_suppression'))} | "
                f"{fmt(row.get('mean_target_relative_lift'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
                f"{fmt(row.get('mean_resolved_baseline_blocker_rate'))} | {fmt(row.get('mean_identity_anchor_gap_improvement'))} | "
                f"{fmt(row.get('mean_closure_fiber_pressure_score'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = OUT_ROOT / round_name / f"phase799_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 799,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    for root in (OUT_ROOT / round_name, RESULT_ROOT / round_name):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "phase799_cross_model_summary.json", payload)
        write_json(root / "phase799_atlas_graph.json", build_atlas(payload))
        write_markdown(root / "phase799_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p796.build_parser()
    parser.add_argument("--full-rank-window", type=int, default=128)
    parser.add_argument("--max-full-above-classify", type=int, default=40000)
    parser.add_argument("--max-surface-variants-saved", type=int, default=32)
    parser.add_argument("--max-baseline-blocker-classify", type=int, default=40000)
    parser.add_argument("--strong-suppression-threshold", type=float, default=0.5)
    parser.add_argument("--alpha-target", type=float, default=1.0)
    parser.add_argument("--beta-anchor", type=float, default=1.0)
    parser.add_argument("--gamma-suppress", type=float, default=1.0)
    parser.add_argument("--lambda-new-blocker", type=float, default=1.0)
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
