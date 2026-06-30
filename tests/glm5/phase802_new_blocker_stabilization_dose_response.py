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
import phase799_blocker_field_causal_suppressor_localization as p799  # noqa: E402
import phase801_target_neutral_suppressor_causal_test as p801  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase802_new_blocker_stabilization_dose_response")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def safe_mean(values: list[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def safe_rate(values: list[Any]) -> float | None:
    vals = [bool(v) for v in values if v is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def parse_alpha_grid(text: str) -> list[float]:
    vals: list[float] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals or [0.0, 0.5, 1.0]


def alpha_label(alpha: float) -> str:
    text = f"{alpha:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def project_delta_alpha(delta: torch.Tensor, target_direction: torch.Tensor, alpha: float) -> tuple[torch.Tensor, dict[str, Any]]:
    direction = target_direction.float()
    delta_f = delta.float()
    denom = float(torch.dot(direction, direction).item())
    if denom <= 1e-9:
        parallel = torch.zeros_like(delta_f)
    else:
        coeff = float(torch.dot(delta_f, direction).item() / denom)
        parallel = coeff * direction
    neutral = delta_f - parallel
    projected = neutral + float(alpha) * parallel
    direct_before = float(torch.dot(delta_f, direction).item())
    direct_after = float(torch.dot(projected.float(), direction).item())
    return projected, {
        "direct_target_component_before": direct_before,
        "direct_target_component_after": direct_after,
        "direct_target_component_removed": direct_before - direct_after,
        "delta_norm": float(delta_f.norm().item()),
        "parallel_norm": float(parallel.norm().item()),
        "neutral_norm": float(neutral.norm().item()),
        "projected_norm": float(projected.float().norm().item()),
        "target_direction_alpha": float(alpha),
    }


def projected_component_state_alpha(
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    route_components: list[dict[str, Any]],
    target_direction: torch.Tensor,
    alpha: float,
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    projected: dict[tuple[str, int], torch.Tensor] = {}
    metrics: dict[str, list[float]] = defaultdict(list)
    for comp in route_components:
        key = (str(comp["component_kind"]), int(comp["layer"]))
        rec_vec = recipient_state.get("components", {}).get(key)
        donor_vec = donor_state.get("components", {}).get(key)
        if rec_vec is None or donor_vec is None:
            continue
        delta = donor_vec.float() - rec_vec.float()
        delta_projected, meta = project_delta_alpha(delta, target_direction, alpha)
        projected[key] = (rec_vec.float() + delta_projected).detach().cpu()
        for k, v in meta.items():
            metrics[k].append(v)
    summary = {f"mean_{k}": safe_mean(vs) for k, vs in metrics.items()}
    summary["target_direction_alpha"] = float(alpha)
    return projected, summary


def classify_new_blockers(
    args: argparse.Namespace,
    recipient_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
) -> dict[str, Any]:
    rec_vals = recipient_logits.detach().float().cpu()
    after_vals = after_logits.detach().float().cpu()
    rec_target = float(rec_vals[int(target_id)].item())
    after_target = float(after_vals[int(target_id)].item())
    rec_mask = rec_vals > rec_target
    rec_mask[int(target_id)] = False
    after_mask = after_vals > after_target
    after_mask[int(target_id)] = False
    new_ids = torch.nonzero(after_mask & ~rec_mask, as_tuple=False).flatten()
    if new_ids.numel() == 0:
        return {
            "new_blocker_class_counts": {},
            "new_blocker_class_mean_after_gap": {},
            "new_blocker_class_mean_logit_delta": {},
            "new_blocker_class_top_examples": {},
        }
    after_gaps = after_vals[new_ids] - after_target
    order = torch.argsort(after_gaps, descending=True)
    limit = max(0, int(args.max_new_blocker_classify))
    if limit > 0:
        order = order[:limit]
    counts: Counter[str] = Counter()
    gaps: dict[str, list[float]] = defaultdict(list)
    deltas: dict[str, list[float]] = defaultdict(list)
    examples: dict[str, dict[str, Any]] = {}
    for idx_tensor in order.tolist():
        tid = int(new_ids[int(idx_tensor)].item())
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
        gap = float(after_vals[tid].item() - after_target)
        delta = float(after_vals[tid].item() - rec_vals[tid].item())
        counts[cls] += 1
        gaps[cls].append(gap)
        deltas[cls].append(delta)
        if cls not in examples or gap > float(examples[cls]["after_gap"]):
            examples[cls] = {
                "token_id": tid,
                "token_text": text,
                "after_gap": gap,
                "logit_delta": delta,
            }
    return {
        "new_blocker_class_counts": dict(counts),
        "new_blocker_class_mean_after_gap": {cls: safe_mean(vals) for cls, vals in gaps.items()},
        "new_blocker_class_mean_logit_delta": {cls: safe_mean(vals) for cls, vals in deltas.items()},
        "new_blocker_class_top_examples": examples,
    }


def label_phase802(row: dict[str, Any], args: argparse.Namespace) -> str:
    tg = safe_float(row.get("target_logit_gain")) or 0.0
    bs = safe_float(row.get("baseline_blocker_mean_suppression")) or 0.0
    new_rate = safe_float(row.get("new_blocker_rate")) or 0.0
    anchor = safe_float(row.get("identity_anchor_gap_improvement")) or 0.0
    if row.get("token_closure_gain"):
        return "token_closure"
    if bs > args.min_old_suppression and new_rate <= args.max_stable_new_rate and anchor >= args.min_anchor_improvement:
        return "old_suppress_new_stable_anchor_ok"
    if bs > args.min_old_suppression and new_rate <= args.max_stable_new_rate:
        return "old_suppress_new_stable_anchor_weak"
    if bs > args.min_old_suppression and new_rate > args.max_stable_new_rate:
        return "old_suppress_new_unstable"
    if tg > args.target_boost_threshold and bs <= 0:
        return "threshold_shift_without_suppression"
    return "weak_or_mixed"


def add_phase802_metrics(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    tg = safe_float(row.get("target_logit_gain")) or 0.0
    bs = safe_float(row.get("baseline_blocker_mean_suppression")) or 0.0
    resolved = safe_float(row.get("resolved_baseline_blocker_rate")) or 0.0
    new_rate = safe_float(row.get("new_blocker_rate")) or 0.0
    anchor = safe_float(row.get("identity_anchor_gap_improvement")) or 0.0
    target_penalty = 1.0 + max(abs(tg) - float(args.target_gain_budget), 0.0)
    anchor_factor = 1.0 + max(anchor, 0.0) / (1.0 + abs(anchor))
    stable_score = max(bs, 0.0) * resolved * max(1.0 - new_rate, 0.0)
    row["phase802_old_suppress_new_stable_score"] = stable_score
    row["phase802_output_field_closure_score"] = stable_score * anchor_factor / target_penalty
    row["phase802_label"] = label_phase802(row, args)
    row["phase802_boundary"] = (
        "Alpha dose response tests whether adding back a controlled target-readout component stabilizes new blockers "
        "while preserving old-blocker suppression. It is not a neuron-level mechanism proof."
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
) -> list[dict[str, Any]]:
    target_id, contrast_id = p796.target_ids_from_row(tokenizer, case, source_row)
    recipient_variant = args.recipient_variant
    donor_variant = route["compare_variant"]
    recipient_prompt = p796.surface_prompt_for_variant(case, recipient_variant)
    donor_prompt = p796.surface_prompt_for_variant(case, donor_variant)
    recipient_ids = tokenizer.encode(recipient_prompt, add_special_tokens=False)
    donor_ids = tokenizer.encode(donor_prompt, add_special_tokens=False)
    recipient_answer_pos = len(recipient_ids) - 1
    recipient_groups = p796.source_groups_for_prompt(tokenizer, recipient_prompt, case, recipient_ids)
    donor_groups = p796.source_groups_for_prompt(tokenizer, donor_prompt, case, donor_ids)
    recipient_candidate_ids = p796.candidate_position_ids(tokenizer, recipient_ids, recipient_groups)
    donor_candidate_ids = p796.candidate_position_ids(tokenizer, donor_ids, donor_groups)
    case_vals = p796.value_strings(case)
    recipient_state = p796.capture_answer_outputs_and_sources(model, tokenizer, device, recipient_prompt, component_keys)
    donor_state = p796.capture_answer_outputs_and_sources(model, tokenizer, device, donor_prompt, component_keys)
    route_components = selected_route_components(route, set(p796.parse_csv(args.route_component_kinds) or ["attn", "mlp"]), args.max_route_components)
    target_direction = unembed[int(target_id)].float().cpu()
    rows: list[dict[str, Any]] = []
    for alpha in parse_alpha_grid(args.alpha_grid):
        projected, projection_metrics = projected_component_state_alpha(
            recipient_state,
            donor_state,
            route_components,
            target_direction,
            alpha,
        )
        if not projected:
            after_logits = torch.empty(0)
            error = "no_projected_components"
        else:
            after_logits, error = p801.run_logits_with_projected_route(model, device, recipient_ids, projected, recipient_answer_pos)
        row = p801.make_row(
            args,
            case,
            route,
            route_components,
            f"alpha_{alpha_label(alpha)}",
            projection_metrics,
            recipient_variant,
            donor_variant,
            recipient_state["logits"],
            donor_state["logits"],
            after_logits,
            target_id,
            contrast_id,
            recipient_prompt,
            donor_prompt,
            recipient_ids,
            donor_ids,
            recipient_candidate_ids,
            donor_candidate_ids,
            case_vals,
            error,
        )
        row["row_kind"] = "phase802_new_blocker_stabilization_dose_response"
        row["target_direction_alpha"] = float(alpha)
        if after_logits.numel():
            row.update(
                classify_new_blockers(
                    args,
                    recipient_state["logits"],
                    after_logits,
                    target_id,
                    contrast_id,
                    recipient_ids,
                    recipient_prompt,
                    recipient_candidate_ids,
                    case_vals,
                )
            )
        rows.append(add_phase802_metrics(row, args))
    return rows


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
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_target_logit_gain": safe_mean([v.get("target_logit_gain") for v in vals]),
                "mean_direct_target_component_after": safe_mean([v.get("mean_direct_target_component_after") for v in vals]),
                "mean_baseline_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_resolved_baseline_blocker_rate": safe_mean([v.get("resolved_baseline_blocker_rate") for v in vals]),
                "mean_new_blocker_rate": safe_mean([v.get("new_blocker_rate") for v in vals]),
                "mean_new_blocker_count": safe_mean([v.get("new_blocker_count") for v in vals]),
                "mean_full_blocker_count_delta": safe_mean([v.get("full_blocker_count_delta") for v in vals]),
                "mean_identity_anchor_gap_improvement": safe_mean([v.get("identity_anchor_gap_improvement") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "mean_old_suppress_new_stable_score": safe_mean([v.get("phase802_old_suppress_new_stable_score") for v in vals]),
                "mean_output_field_closure_score": safe_mean([v.get("phase802_output_field_closure_score") for v in vals]),
                "label_counts": dict(Counter(v.get("phase802_label") for v in vals)),
                "baseline_blocker_class_counts": merge_counter_dicts(vals, "baseline_blocker_class_counts"),
                "after_full_above_class_counts": merge_counter_dicts(vals, "after_full_above_class_counts"),
                "new_blocker_class_counts": merge_counter_dicts(vals, "new_blocker_class_counts"),
                "new_blocker_class_mean_after_gap": mean_nested_metric(vals, "new_blocker_class_mean_after_gap"),
                "new_blocker_class_mean_logit_delta": mean_nested_metric(vals, "new_blocker_class_mean_logit_delta"),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("mean_output_field_closure_score") or -999.0,
            -(r.get("mean_new_blocker_rate") or 999.0),
            r.get("mean_baseline_blocker_suppression") or -999.0,
        ),
        reverse=True,
    )
    return out


def best_alpha_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    fields = ["model", "case_id", "route_id", "route_component_signature", "compare_variant"]
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        by_alpha = {float(v.get("target_direction_alpha") or 0.0): v for v in vals}
        alpha0 = by_alpha.get(0.0)
        alpha1 = by_alpha.get(1.0)
        best = max(vals, key=lambda v: safe_float(v.get("phase802_output_field_closure_score")) or -999.0)
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "best_alpha": best.get("target_direction_alpha"),
                "best_target_gain": best.get("target_logit_gain"),
                "best_old_suppression": best.get("baseline_blocker_mean_suppression"),
                "best_new_blocker_rate": best.get("new_blocker_rate"),
                "best_resolved_rate": best.get("resolved_baseline_blocker_rate"),
                "best_anchor_improvement": best.get("identity_anchor_gap_improvement"),
                "best_score": best.get("phase802_output_field_closure_score"),
                "best_label": best.get("phase802_label"),
                "best_token_closure_gain": best.get("token_closure_gain"),
                "alpha0_new_blocker_rate": alpha0.get("new_blocker_rate") if alpha0 else None,
                "alpha0_old_suppression": alpha0.get("baseline_blocker_mean_suppression") if alpha0 else None,
                "alpha1_new_blocker_rate": alpha1.get("new_blocker_rate") if alpha1 else None,
                "alpha1_old_suppression": alpha1.get("baseline_blocker_mean_suppression") if alpha1 else None,
            }
        )
        a0_new = safe_float(payload["alpha0_new_blocker_rate"])
        best_new = safe_float(payload["best_new_blocker_rate"])
        a0_supp = safe_float(payload["alpha0_old_suppression"])
        best_supp = safe_float(payload["best_old_suppression"])
        payload["new_blocker_reduction_vs_alpha0"] = (a0_new - best_new) if a0_new is not None and best_new is not None else None
        payload["old_suppression_retention_vs_alpha0"] = (best_supp / a0_supp) if a0_supp and abs(a0_supp) > 1e-9 and best_supp is not None else None
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("best_token_closure_gain"),
            r.get("best_score") or -999.0,
            r.get("new_blocker_reduction_vs_alpha0") or -999.0,
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": 802,
        "title": "New-Blocker Stabilization Dose Response",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows}),
        "n_routes": len(routes),
        "alpha_grid": parse_alpha_grid(args.alpha_grid),
        "routes": routes,
        "by_model": group_rows(rows, ["model"]),
        "by_alpha": group_rows(rows, ["model", "target_direction_alpha"]),
        "by_route_alpha": group_rows(rows, ["model", "route_component_signature", "target_direction_alpha"])[:120],
        "best_alpha_triplets": best_alpha_rows(rows, args)[:120],
        "strict_boundary": (
            "This phase probes alpha dose response from target-neutral to raw route patch. "
            "It tests output-field stabilization and new-blocker control, not neuron-level closure."
        ),
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = p796.select_surface_cases(args.model, args)
    routes = p796.select_routes(args.model, args)
    if args.max_routes and len(routes) > args.max_routes:
        routes = routes[: args.max_routes]
    component_keys = p796.component_keys_for_routes(routes)
    cmap = case_map_for(args)
    log(
        f"{args.model}/{args.round_name}: cases={len(selected)} routes={len(routes)} "
        f"alpha_grid={parse_alpha_grid(args.alpha_grid)}"
    )
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "alpha_grid": parse_alpha_grid(args.alpha_grid),
        }
    model, tokenizer, device, attn_impl = p796.load_model_bf16_prefer_flash(args.model, args.attn_implementations)
    setattr(args, "_tokenizer", tokenizer)
    setattr(args, "_token_text_cache", {})
    try:
        p796.enrich_selected_rows_with_target_id(tokenizer, selected, cmap)
        unembed = lm_head_weight(model)
        rows: list[dict[str, Any]] = []
        for ci, source_row in enumerate(selected, 1):
            case = cmap[source_row["case_id"]]
            for route in routes:
                rows.extend(audit_case_route(model, tokenizer, device, unembed, args, case, source_row, route, component_keys))
            if ci % args.log_every == 0 or ci == len(selected):
                log(f"{args.model}: new-blocker dose response {ci}/{len(selected)} cases; rows={len(rows)}")
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
    summary = summarize(rows, args, attn_impl, routes)
    write_jsonl(out_dir / f"phase802_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase802_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_alpha": summary["by_alpha"],
                "best_alpha_triplets": summary["best_alpha_triplets"][:5],
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

    task = "phase802:new_blocker_stabilization_dose_response"
    node(task, "task", label="Phase 802 new-blocker stabilization dose response")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for row in summary.get("by_alpha", []):
            alpha_node = f"{model_name}:alpha:{row.get('target_direction_alpha')}"
            node(alpha_node, "alpha_dose", label=f"alpha={row.get('target_direction_alpha')}", metrics=row)
            edges.append(
                {
                    "id": f"{model_name}:alpha:{row.get('target_direction_alpha')}",
                    "source": model_node,
                    "target": alpha_node,
                    "type": "target_direction_dose_response",
                    "weight": row.get("mean_output_field_closure_score"),
                    "metrics": row,
                }
            )
    return {"schema_version": "atlas_graph_v1", "phase": 802, "graph": {"nodes": list(nodes.values()), "edges": edges}}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 802 New-Blocker Stabilization Dose Response ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: alpha=0 is target-neutral, alpha=1 is raw route patch, alpha>1 over-injects the direct target direction.",
        "- This phase tests whether adding controlled target-readout dose reduces new blockers while preserving old-blocker suppression.",
        "",
        "## By Alpha",
        "",
        "| model | alpha | rows | cases | target gain | old suppression | resolved | new rate | anchor | closure score | token gain | labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in sorted(data.get("by_alpha", []), key=lambda r: safe_float(r.get("target_direction_alpha")) or 0.0):
            lines.append(
                f"| {model_name} | {fmt(row.get('target_direction_alpha'))} | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_target_logit_gain'))} | {fmt(row.get('mean_baseline_blocker_suppression'))} | "
                f"{fmt(row.get('mean_resolved_baseline_blocker_rate'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
                f"{fmt(row.get('mean_identity_anchor_gap_improvement'))} | {fmt(row.get('mean_output_field_closure_score'))} | "
                f"{fmt(row.get('token_closure_gain_rate'))} | `{json.dumps(row.get('label_counts') or {}, ensure_ascii=False, sort_keys=True)}` |"
            )
    lines += [
        "",
        "## Best Alpha Triplets",
        "",
        "| model | case | route | best alpha | target gain | old suppress | new rate | anchor | score | label | new reduction vs a0 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("best_alpha_triplets", [])[:20]:
            lines.append(
                f"| {model_name} | `{row.get('case_id')}` | `{row.get('route_component_signature')}` | "
                f"{fmt(row.get('best_alpha'))} | {fmt(row.get('best_target_gain'))} | "
                f"{fmt(row.get('best_old_suppression'))} | {fmt(row.get('best_new_blocker_rate'))} | "
                f"{fmt(row.get('best_anchor_improvement'))} | {fmt(row.get('best_score'))} | "
                f"`{row.get('best_label')}` | {fmt(row.get('new_blocker_reduction_vs_alpha0'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = RESULT_ROOT / round_name / f"phase802_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 802,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    root = RESULT_ROOT / round_name
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "phase802_cross_model_summary.json", payload)
    write_json(root / "phase802_atlas_graph.json", build_atlas(payload))
    write_markdown(root / "phase802_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p801.build_parser()
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--target-gain-budget", type=float, default=1.0)
    parser.add_argument("--min-old-suppression", type=float, default=0.25)
    parser.add_argument("--max-stable-new-rate", type=float, default=0.10)
    parser.add_argument("--min-anchor-improvement", type=float, default=0.0)
    parser.add_argument("--target-boost-threshold", type=float, default=1.0)
    parser.add_argument("--max-new-blocker-classify", type=int, default=40000)
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
