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
import phase802_new_blocker_stabilization_dose_response as p802  # noqa: E402
from model_utils import release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import lm_head_weight  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase803_semantic_new_blocker_source_localization")


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


def semantic_new_blocker_ids(
    args: argparse.Namespace,
    recipient_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    prompt_ids: list[int],
    prompt_text: str,
    candidate_ids: set[int],
    case_values: set[str],
) -> list[dict[str, Any]]:
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
        return []
    rows: list[dict[str, Any]] = []
    for tid in new_ids.tolist():
        token_id = int(tid)
        text = p798.cached_token_text(args, token_id)
        cls = p796.classify_competitor(
            args._tokenizer,
            token_id,
            text,
            int(target_id),
            int(contrast_id),
            prompt_ids,
            prompt_text,
            candidate_ids,
            case_values,
        )
        if cls != "semantic_or_lexical_competitor":
            continue
        rows.append(
            {
                "token_id": token_id,
                "token_text": text,
                "after_gap_vs_target": float(after_vals[token_id].item() - after_target),
                "logit_delta_vs_recipient": float(after_vals[token_id].item() - rec_vals[token_id].item()),
            }
        )
    rows.sort(key=lambda r: r["after_gap_vs_target"], reverse=True)
    return rows[: max(1, int(args.max_semantic_new_blockers))]


def matched_semantic_metrics(
    recipient_logits: torch.Tensor,
    alpha0_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    semantic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not semantic_rows:
        return {
            "matched_semantic_count": 0,
            "matched_semantic_mean_logit_delta_vs_recipient": None,
            "matched_semantic_mean_logit_delta_vs_alpha0": None,
            "matched_semantic_true_suppression_vs_alpha0": None,
            "matched_semantic_mean_gap_vs_target": None,
            "matched_semantic_still_above_target_rate": None,
            "matched_semantic_top_examples": [],
        }
    rec_vals = recipient_logits.detach().float().cpu()
    a0_vals = alpha0_logits.detach().float().cpu()
    after_vals = after_logits.detach().float().cpu()
    target_val = float(after_vals[int(target_id)].item())
    deltas_rec: list[float] = []
    deltas_a0: list[float] = []
    suppress_vs_a0: list[float] = []
    gaps: list[float] = []
    still: list[bool] = []
    examples: list[dict[str, Any]] = []
    for item in semantic_rows:
        tid = int(item["token_id"])
        delta_rec = float(after_vals[tid].item() - rec_vals[tid].item())
        delta_a0 = float(after_vals[tid].item() - a0_vals[tid].item())
        gap = float(after_vals[tid].item() - target_val)
        deltas_rec.append(delta_rec)
        deltas_a0.append(delta_a0)
        suppress_vs_a0.append(-delta_a0)
        gaps.append(gap)
        still.append(gap > 0)
        examples.append(
            {
                "token_id": tid,
                "token_text": item.get("token_text"),
                "gap_vs_target": gap,
                "delta_vs_recipient": delta_rec,
                "delta_vs_alpha0": delta_a0,
                "suppression_vs_alpha0": -delta_a0,
            }
        )
    examples.sort(key=lambda r: r["gap_vs_target"], reverse=True)
    return {
        "matched_semantic_count": len(semantic_rows),
        "matched_semantic_mean_logit_delta_vs_recipient": safe_mean(deltas_rec),
        "matched_semantic_mean_logit_delta_vs_alpha0": safe_mean(deltas_a0),
        "matched_semantic_true_suppression_vs_alpha0": safe_mean(suppress_vs_a0),
        "matched_semantic_mean_gap_vs_target": safe_mean(gaps),
        "matched_semantic_still_above_target_rate": safe_rate(still),
        "matched_semantic_top_examples": examples[:10],
    }


def label_trace(row: dict[str, Any], args: argparse.Namespace) -> str:
    true_supp = safe_float(row.get("matched_semantic_true_suppression_vs_alpha0")) or 0.0
    still_rate = safe_float(row.get("matched_semantic_still_above_target_rate")) or 0.0
    target_gain_delta = safe_float(row.get("target_gain_delta_vs_alpha0")) or 0.0
    gap = safe_float(row.get("matched_semantic_mean_gap_vs_target"))
    old_supp = safe_float(row.get("baseline_blocker_mean_suppression")) or 0.0
    if row.get("token_closure_gain"):
        return "token_closure"
    if true_supp > args.min_true_semantic_suppression and still_rate <= args.max_semantic_still_rate and old_supp > args.min_old_suppression:
        return "true_semantic_new_blocker_suppression_candidate"
    if target_gain_delta > args.min_threshold_gain and (gap is not None and gap < 0) and true_supp <= args.min_true_semantic_suppression:
        return "threshold_cover_not_true_suppression"
    if still_rate <= args.max_semantic_still_rate and old_supp > args.min_old_suppression:
        return "semantic_blockers_below_target_no_logit_suppression"
    if still_rate > args.max_semantic_still_rate:
        return "semantic_new_blockers_persist"
    return "weak_or_mixed"


def component_release_rows(
    args: argparse.Namespace,
    model,
    device,
    recipient_ids: list[int],
    recipient_answer_pos: int,
    recipient_logits: torch.Tensor,
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    route_components: list[dict[str, Any]],
    target_direction: torch.Tensor,
    target_id: int,
    contrast_id: int,
    recipient_prompt: str,
    recipient_candidate_ids: set[int],
    case_values: set[str],
    full_alpha0_semantic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_ids = {int(x["token_id"]) for x in full_alpha0_semantic}
    rows: list[dict[str, Any]] = []
    for comp in route_components:
        projected, projection_metrics = p802.projected_component_state_alpha(
            recipient_state,
            donor_state,
            [comp],
            target_direction,
            float(args.component_source_alpha),
        )
        if not projected:
            continue
        after_logits, error = p801.run_logits_with_projected_route(model, device, recipient_ids, projected, recipient_answer_pos)
        if error or not after_logits.numel():
            rows.append(
                {
                    "row_kind": "phase803_component_source",
                    "component_label": f"{comp['component_kind']}:L{int(comp['layer'])}",
                    "component_kind": comp["component_kind"],
                    "layer": int(comp["layer"]),
                    "component_error": error,
                }
            )
            continue
        semantic = semantic_new_blocker_ids(
            args,
            recipient_logits,
            after_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_values,
        )
        sem_ids = {int(x["token_id"]) for x in semantic}
        overlap = len(full_ids & sem_ids)
        union = len(full_ids | sem_ids)
        response = p799.blocker_field_response(
            args,
            recipient_logits,
            after_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_values,
            "",
        )
        rows.append(
            {
                "row_kind": "phase803_component_source",
                "component_label": f"{comp['component_kind']}:L{int(comp['layer'])}",
                "component_kind": comp["component_kind"],
                "layer": int(comp["layer"]),
                "component_alpha": float(args.component_source_alpha),
                "component_semantic_new_count": len(semantic),
                "component_overlap_with_full_alpha0_count": overlap,
                "component_overlap_with_full_alpha0_jaccard": overlap / max(union, 1),
                "component_semantic_mean_gap": safe_mean([x.get("after_gap_vs_target") for x in semantic]),
                "component_semantic_mean_logit_delta": safe_mean([x.get("logit_delta_vs_recipient") for x in semantic]),
                "component_semantic_release_score": len(semantic) * max(safe_mean([x.get("after_gap_vs_target") for x in semantic]) or 0.0, 0.0),
                "component_top_semantic_new_blockers": semantic[:10],
                **projection_metrics,
                **response,
            }
        )
    rows.sort(
        key=lambda r: (
            r.get("component_overlap_with_full_alpha0_count") or 0,
            r.get("component_semantic_release_score") or 0.0,
        ),
        reverse=True,
    )
    return rows


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
    alpha_logits: dict[float, torch.Tensor] = {}
    alpha_rows: list[dict[str, Any]] = []
    for alpha in p802.parse_alpha_grid(args.alpha_grid):
        projected, projection_metrics = p802.projected_component_state_alpha(
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
        if after_logits.numel():
            alpha_logits[float(alpha)] = after_logits
        row = p801.make_row(
            args,
            case,
            route,
            route_components,
            f"alpha_{p802.alpha_label(alpha)}",
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
        row["row_kind"] = "phase803_semantic_new_blocker_trace"
        row["target_direction_alpha"] = float(alpha)
        row.update(p802.classify_new_blockers(args, recipient_state["logits"], after_logits, target_id, contrast_id, recipient_ids, recipient_prompt, recipient_candidate_ids, case_vals) if after_logits.numel() else {})
        alpha_rows.append(row)

    alpha0_logits = alpha_logits.get(0.0)
    if alpha0_logits is None:
        return alpha_rows
    full_alpha0_semantic = semantic_new_blocker_ids(
        args,
        recipient_state["logits"],
        alpha0_logits,
        target_id,
        contrast_id,
        recipient_ids,
        recipient_prompt,
        recipient_candidate_ids,
        case_vals,
    )
    alpha0_target_gain = None
    for row in alpha_rows:
        if float(row.get("target_direction_alpha") or 0.0) == 0.0:
            alpha0_target_gain = safe_float(row.get("target_logit_gain")) or 0.0
            break
    traced_rows: list[dict[str, Any]] = []
    for row in alpha_rows:
        alpha = float(row.get("target_direction_alpha") or 0.0)
        after_logits = alpha_logits.get(alpha)
        if after_logits is None:
            traced_rows.append(row)
            continue
        metrics = matched_semantic_metrics(
            recipient_state["logits"],
            alpha0_logits,
            after_logits,
            target_id,
            full_alpha0_semantic,
        )
        row.update(metrics)
        row["alpha0_semantic_new_blocker_count"] = len(full_alpha0_semantic)
        row["target_gain_delta_vs_alpha0"] = (safe_float(row.get("target_logit_gain")) or 0.0) - (alpha0_target_gain or 0.0)
        row["phase803_true_semantic_suppression_score"] = max(
            safe_float(row.get("matched_semantic_true_suppression_vs_alpha0")) or 0.0,
            0.0,
        ) * max(1.0 - (safe_float(row.get("matched_semantic_still_above_target_rate")) or 0.0), 0.0)
        row["phase803_label"] = label_trace(row, args)
        row["phase803_boundary"] = (
            "This row tracks the same semantic new blockers released at alpha=0 across target-direction doses. "
            "A lower new-blocker rate alone is treated as threshold cover unless matched blocker logits decrease."
        )
        traced_rows.append(row)

    if args.component_source:
        source_rows = component_release_rows(
            args,
            model,
            device,
            recipient_ids,
            recipient_answer_pos,
            recipient_state["logits"],
            recipient_state,
            donor_state,
            route_components,
            target_direction,
            target_id,
            contrast_id,
            recipient_prompt,
            recipient_candidate_ids,
            case_vals,
            full_alpha0_semantic,
        )
        for src in source_rows:
            src.update(
                {
                    "model": args.model,
                    "round": args.round_name,
                    "case_id": case["case_id"],
                    "domain": case.get("domain"),
                    "relation": case.get("relation"),
                    "object": case.get("object"),
                    "target_answer": case.get("answer"),
                    "route_id": route["route_id"],
                    "compare_variant": route["compare_variant"],
                    "route_component_signature": p801.route_signature(route_components),
                    "alpha0_semantic_new_blocker_count": len(full_alpha0_semantic),
                }
            )
        traced_rows.extend(source_rows)
    return traced_rows


def merge_counter_dicts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get(key) or {})
    return dict(counter)


def group_trace_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    trace_rows = [r for r in rows if r.get("row_kind") == "phase803_semantic_new_blocker_trace"]
    for row in trace_rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_target_logit_gain": safe_mean([v.get("target_logit_gain") for v in vals]),
                "mean_target_gain_delta_vs_alpha0": safe_mean([v.get("target_gain_delta_vs_alpha0") for v in vals]),
                "mean_old_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_new_blocker_rate": safe_mean([v.get("new_blocker_rate") for v in vals]),
                "mean_alpha0_semantic_new_count": safe_mean([v.get("alpha0_semantic_new_blocker_count") for v in vals]),
                "mean_matched_semantic_delta_vs_recipient": safe_mean([v.get("matched_semantic_mean_logit_delta_vs_recipient") for v in vals]),
                "mean_matched_semantic_delta_vs_alpha0": safe_mean([v.get("matched_semantic_mean_logit_delta_vs_alpha0") for v in vals]),
                "mean_true_semantic_suppression_vs_alpha0": safe_mean([v.get("matched_semantic_true_suppression_vs_alpha0") for v in vals]),
                "mean_matched_semantic_gap_vs_target": safe_mean([v.get("matched_semantic_mean_gap_vs_target") for v in vals]),
                "mean_matched_semantic_still_above_target_rate": safe_mean([v.get("matched_semantic_still_above_target_rate") for v in vals]),
                "mean_true_semantic_suppression_score": safe_mean([v.get("phase803_true_semantic_suppression_score") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "label_counts": dict(Counter(v.get("phase803_label") for v in vals)),
                "new_blocker_class_counts": merge_counter_dicts(vals, "new_blocker_class_counts"),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("mean_true_semantic_suppression_score") or -999.0,
            -(r.get("mean_matched_semantic_still_above_target_rate") or 999.0),
        ),
        reverse=True,
    )
    return out


def group_component_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    source_rows = [r for r in rows if r.get("row_kind") == "phase803_component_source"]
    for row in source_rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(vals),
                "case_n": len({v.get("case_id") for v in vals}),
                "mean_component_semantic_new_count": safe_mean([v.get("component_semantic_new_count") for v in vals]),
                "mean_overlap_with_full_alpha0_count": safe_mean([v.get("component_overlap_with_full_alpha0_count") for v in vals]),
                "mean_overlap_with_full_alpha0_jaccard": safe_mean([v.get("component_overlap_with_full_alpha0_jaccard") for v in vals]),
                "mean_component_semantic_gap": safe_mean([v.get("component_semantic_mean_gap") for v in vals]),
                "mean_component_semantic_logit_delta": safe_mean([v.get("component_semantic_mean_logit_delta") for v in vals]),
                "mean_component_semantic_release_score": safe_mean([v.get("component_semantic_release_score") for v in vals]),
                "mean_old_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_new_blocker_rate": safe_mean([v.get("new_blocker_rate") for v in vals]),
            }
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("mean_overlap_with_full_alpha0_count") or 0.0,
            r.get("mean_component_semantic_release_score") or 0.0,
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    trace_rows = [r for r in rows if r.get("row_kind") == "phase803_semantic_new_blocker_trace"]
    source_rows = [r for r in rows if r.get("row_kind") == "phase803_component_source"]
    return {
        "phase": 803,
        "title": "Semantic New-Blocker Source Localization",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_trace_rows": len(trace_rows),
        "n_component_rows": len(source_rows),
        "n_cases": len({r.get("case_id") for r in trace_rows}),
        "n_routes": len(routes),
        "alpha_grid": p802.parse_alpha_grid(args.alpha_grid),
        "routes": routes,
        "by_alpha": group_trace_rows(rows, ["model", "target_direction_alpha"]),
        "by_route_alpha": group_trace_rows(rows, ["model", "route_component_signature", "target_direction_alpha"])[:120],
        "top_component_sources": group_component_rows(rows, ["model", "component_label", "component_kind", "layer"])[:80],
        "by_component_route": group_component_rows(rows, ["model", "route_component_signature", "component_label"])[:120],
        "strict_boundary": (
            "This phase matches semantic new blockers released at alpha=0 and tracks their logits across alpha. "
            "It distinguishes threshold cover from true semantic new-blocker logit suppression, but it still uses "
            "answer-position hidden-state route patching rather than neuron-level causal graph closure."
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
        f"alpha_grid={p802.parse_alpha_grid(args.alpha_grid)} component_source={args.component_source}"
    )
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "alpha_grid": p802.parse_alpha_grid(args.alpha_grid),
            "component_source": args.component_source,
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
                log(f"{args.model}: semantic new-blocker source localization {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase803_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase803_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_alpha": summary["by_alpha"],
                "top_component_sources": summary["top_component_sources"][:8],
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

    task = "phase803:semantic_new_blocker_source_localization"
    node(task, "task", label="Phase 803 semantic new-blocker source localization")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for row in summary.get("top_component_sources", [])[:30]:
            comp = f"{model_name}:semantic_source:{row.get('component_label')}"
            node(comp, "component_source", label=row.get("component_label"), metrics=row)
            edges.append(
                {
                    "id": f"{model_name}:source:{row.get('component_label')}:{len(edges)}",
                    "source": model_node,
                    "target": comp,
                    "type": "semantic_new_blocker_source_candidate",
                    "weight": row.get("mean_component_semantic_release_score"),
                    "metrics": row,
                }
            )
    return {"schema_version": "atlas_graph_v1", "phase": 803, "graph": {"nodes": list(nodes.values()), "edges": edges}}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 803 Semantic New-Blocker Source Localization ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: tracks the same semantic new blockers released at alpha=0 across target-direction doses.",
        "- A lower new-blocker rate is not counted as true suppression unless matched semantic blocker logits drop.",
        "",
        "## Matched Semantic New Blockers By Alpha",
        "",
        "| model | alpha | rows | cases | target gain | target gain vs a0 | old suppress | new rate | matched delta vs a0 | true suppress vs a0 | still above | label counts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in sorted(data.get("by_alpha", []), key=lambda r: safe_float(r.get("target_direction_alpha")) or 0.0):
            lines.append(
                f"| {model_name} | {fmt(row.get('target_direction_alpha'))} | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_target_logit_gain'))} | {fmt(row.get('mean_target_gain_delta_vs_alpha0'))} | "
                f"{fmt(row.get('mean_old_blocker_suppression'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
                f"{fmt(row.get('mean_matched_semantic_delta_vs_alpha0'))} | {fmt(row.get('mean_true_semantic_suppression_vs_alpha0'))} | "
                f"{fmt(row.get('mean_matched_semantic_still_above_target_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False, sort_keys=True)}` |"
            )
    lines += [
        "",
        "## Top Component Sources",
        "",
        "| model | component | rows | cases | semantic new | overlap | jaccard | gap | logit delta | release score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_component_sources", [])[:20]:
            lines.append(
                f"| {model_name} | `{row.get('component_label')}` | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_component_semantic_new_count'))} | {fmt(row.get('mean_overlap_with_full_alpha0_count'))} | "
                f"{fmt(row.get('mean_overlap_with_full_alpha0_jaccard'))} | {fmt(row.get('mean_component_semantic_gap'))} | "
                f"{fmt(row.get('mean_component_semantic_logit_delta'))} | {fmt(row.get('mean_component_semantic_release_score'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = RESULT_ROOT / round_name / f"phase803_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 803,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    root = RESULT_ROOT / round_name
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "phase803_cross_model_summary.json", payload)
    write_json(root / "phase803_atlas_graph.json", build_atlas(payload))
    write_markdown(root / "phase803_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p802.build_parser()
    parser.add_argument("--max-semantic-new-blockers", type=int, default=64)
    parser.add_argument("--min-true-semantic-suppression", type=float, default=0.25)
    parser.add_argument("--max-semantic-still-rate", type=float, default=0.20)
    parser.add_argument("--min-threshold-gain", type=float, default=0.75)
    parser.add_argument("--component-source", action="store_true")
    parser.add_argument("--component-source-alpha", type=float, default=0.0)
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
