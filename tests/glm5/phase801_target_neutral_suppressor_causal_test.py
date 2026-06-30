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
import phase799_blocker_field_causal_suppressor_localization as p799  # noqa: E402
from model_utils import get_layers, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, write_json, write_jsonl  # noqa: E402
from phase735_source_restricted_writer_validation import MODELS  # noqa: E402
from phase771_matched_causal_intervention_reliability_test import case_map_for, margin  # noqa: E402
from phase773_instruction_source_disentanglement import fmt  # noqa: E402
from phase780_surface_form_component_localization import COMPARE_BASELINE, lm_head_weight, tensor_from_output  # noqa: E402
from phase795_multi_component_causal_fiber_closure import selected_route_components  # noqa: E402


RESULT_ROOT = Path("tests/result/phase801_target_neutral_suppressor_causal_test")


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


def route_signature(route_components: list[dict[str, Any]]) -> str:
    return "+".join(f"{c['component_kind']}:L{int(c['layer'])}" for c in route_components)


def project_delta(delta: torch.Tensor, target_direction: torch.Tensor, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
    direction = target_direction.float()
    delta_f = delta.float()
    denom = float(torch.dot(direction, direction).item())
    if denom <= 1e-9:
        parallel = torch.zeros_like(delta_f)
    else:
        coeff = float(torch.dot(delta_f, direction).item() / denom)
        parallel = coeff * direction
    neutral = delta_f - parallel
    if mode == "raw":
        projected = delta_f
    elif mode == "target_neutral":
        projected = neutral
    elif mode == "target_only":
        projected = parallel
    else:
        raise ValueError(f"unknown projection mode: {mode}")
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
    }


def projected_component_state(
    recipient_state: dict[str, Any],
    donor_state: dict[str, Any],
    route_components: list[dict[str, Any]],
    target_direction: torch.Tensor,
    projection_mode: str,
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
        delta_projected, meta = project_delta(delta, target_direction, projection_mode)
        projected[key] = (rec_vec.float() + delta_projected).detach().cpu()
        for k, v in meta.items():
            metrics[k].append(v)
    return projected, {
        f"mean_{k}": safe_mean(vs) for k, vs in metrics.items()
    }


def install_projected_route(
    model,
    projected: dict[tuple[str, int], torch.Tensor],
    recipient_answer_pos: int,
) -> list[Any]:
    handles: list[Any] = []
    layers = get_layers(model)
    for key, vec in projected.items():
        kind, layer_idx = key
        layer = layers[int(layer_idx)]
        module = getattr(layer, "self_attn" if kind == "attn" else "mlp", None)
        if module is None:
            continue

        def hook(_module, _inputs, output, vec=vec, answer_pos=recipient_answer_pos):
            tensor = tensor_from_output(output)
            if tensor is None:
                return output
            patched = tensor.clone()
            patched[0, answer_pos] = vec.to(device=patched.device, dtype=patched.dtype)
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        handles.append(module.register_forward_hook(hook))
    return handles


def run_logits_with_projected_route(
    model,
    device,
    ids: list[int],
    projected: dict[tuple[str, int], torch.Tensor],
    recipient_answer_pos: int,
) -> tuple[torch.Tensor, str | None]:
    handles: list[Any] = []
    try:
        handles = install_projected_route(model, projected, recipient_answer_pos)
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu(), None
    except Exception as exc:
        return torch.empty(0), f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()


def make_row(
    args: argparse.Namespace,
    case: dict[str, Any],
    route: dict[str, Any],
    route_components: list[dict[str, Any]],
    projection_mode: str,
    projection_metrics: dict[str, Any],
    recipient_variant: str,
    donor_variant: str,
    recipient_logits: torch.Tensor,
    donor_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target_id: int,
    contrast_id: int,
    recipient_prompt: str,
    donor_prompt: str,
    recipient_ids: list[int],
    donor_ids: list[int],
    recipient_candidate_ids: set[int],
    donor_candidate_ids: set[int],
    case_values: set[str],
    error: str | None,
) -> dict[str, Any]:
    rec = p796.topk_snapshot(
        args._tokenizer,
        recipient_logits,
        target_id,
        contrast_id,
        recipient_ids,
        recipient_prompt,
        recipient_candidate_ids,
        case_values,
        args.top_k,
    )
    donor = p796.topk_snapshot(
        args._tokenizer,
        donor_logits,
        target_id,
        contrast_id,
        donor_ids,
        donor_prompt,
        donor_candidate_ids,
        case_values,
        args.top_k,
    )
    after = (
        p796.topk_snapshot(
            args._tokenizer,
            after_logits,
            target_id,
            contrast_id,
            recipient_ids,
            recipient_prompt,
            recipient_candidate_ids,
            case_values,
            args.top_k,
        )
        if after_logits.numel()
        else None
    )
    target_gain = float(after["target_logit"] - rec["target_logit"]) if after else None
    global_delta = (
        after["global_margin_target_vs_top_non_target"] - rec["global_margin_target_vs_top_non_target"]
        if after
        else None
    )
    response = (
        p799.blocker_field_response(
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
        if after_logits.numel()
        else {}
    )
    route_labels = [f"{c['component_kind']}:L{int(c['layer'])}" for c in route_components]
    tg = safe_float(response.get("target_logit_gain"))
    bs = safe_float(response.get("baseline_blocker_mean_suppression"))
    resolved = safe_float(response.get("resolved_baseline_blocker_rate")) or 0.0
    new_rate = safe_float(response.get("new_blocker_rate")) or 0.0
    tolerance = float(args.target_neutral_tolerance)
    neutral_gate = max(1.0 - abs(tg or 0.0) / max(tolerance, 1e-6), 0.0)
    target_neutral_score = max(bs or 0.0, 0.0) * resolved * max(1.0 - new_rate, 0.0) * neutral_gate
    label = "not_evaluable"
    if after_logits.numel():
        if projection_mode == "target_neutral" and abs(tg or 0.0) <= tolerance and (bs or 0.0) > args.min_neutral_suppression and new_rate <= args.max_neutral_new_rate:
            label = "target_neutral_suppressor_evidence"
        elif projection_mode == "target_neutral" and (bs or 0.0) > args.min_neutral_suppression:
            label = "neutral_suppression_with_residual_target_gain"
        elif projection_mode == "target_only" and (bs or 0.0) <= 0 and (tg or 0.0) > 0:
            label = "target_only_threshold_shift"
        elif projection_mode == "raw" and (bs or 0.0) > args.min_neutral_suppression and (tg or 0.0) > 0:
            label = "raw_suppressor_like"
        else:
            label = "weak_or_mixed"
    return {
        "row_kind": "phase801_target_neutral_suppressor_causal_test",
        "model": args.model,
        "round": args.round_name,
        "case_id": case["case_id"],
        "domain": case.get("domain"),
        "relation": case.get("relation"),
        "object": case.get("object"),
        "target_answer": case.get("answer"),
        "contrast_answer": case.get("contrast_answer"),
        "route_id": route["route_id"],
        "compare_variant": route["compare_variant"],
        "recipient_variant": recipient_variant,
        "donor_variant": donor_variant,
        "projection_mode": projection_mode,
        "route_component_labels": route_labels,
        "route_component_signature": route_signature(route_components),
        "route_component_count": len(route_labels),
        "target_token_id": int(target_id),
        "contrast_token_id": int(contrast_id),
        "recipient_target_rank": rec["target_rank"],
        "donor_target_rank": donor["target_rank"],
        "after_target_rank": after["target_rank"] if after else None,
        "recipient_target_top1": rec["target_top1"],
        "donor_target_top1": donor["target_top1"],
        "after_target_top1": after["target_top1"] if after else None,
        "token_closure_gain": bool(after and after["target_top1"] and not rec["target_top1"]) if after else None,
        "recipient_global_margin": rec["global_margin_target_vs_top_non_target"],
        "donor_global_margin": donor["global_margin_target_vs_top_non_target"],
        "after_global_margin": after["global_margin_target_vs_top_non_target"] if after else None,
        "delta_global_margin_vs_recipient": global_delta,
        "recipient_margin_target_vs_contrast": float(margin(recipient_logits, target_id, contrast_id)),
        "donor_margin_target_vs_contrast": float(margin(donor_logits, target_id, contrast_id)),
        "after_margin_target_vs_contrast": float(margin(after_logits, target_id, contrast_id)) if after_logits.numel() else None,
        "target_logit_gain_vs_recipient": target_gain,
        "target_neutral_suppressor_score": target_neutral_score,
        "phase801_label": label,
        "intervention_error": error,
        "phase801_boundary": (
            "This phase removes the direct target-readout component from hidden-state route deltas. "
            "A positive target-neutral score is evidence against a pure target-booster explanation, "
            "but it is not yet neuron-level suppressor proof."
        ),
        **projection_metrics,
        **response,
    }


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
    for projection_mode in p796.parse_csv(args.projection_modes):
        projected, projection_metrics = projected_component_state(
            recipient_state,
            donor_state,
            route_components,
            target_direction,
            projection_mode,
        )
        if not projected:
            after_logits = torch.empty(0)
            error = "no_projected_components"
        else:
            after_logits, error = run_logits_with_projected_route(model, device, recipient_ids, projected, recipient_answer_pos)
        rows.append(
            make_row(
                args,
                case,
                route,
                route_components,
                projection_mode,
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
        )
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
                "mean_direct_target_component_before": safe_mean([v.get("mean_direct_target_component_before") for v in vals]),
                "mean_direct_target_component_after": safe_mean([v.get("mean_direct_target_component_after") for v in vals]),
                "mean_direct_target_component_removed": safe_mean([v.get("mean_direct_target_component_removed") for v in vals]),
                "mean_baseline_blocker_suppression": safe_mean([v.get("baseline_blocker_mean_suppression") for v in vals]),
                "mean_gap_weighted_suppression": safe_mean([v.get("baseline_blocker_gap_weighted_suppression") for v in vals]),
                "mean_target_relative_lift": safe_mean([v.get("baseline_blocker_target_relative_lift") for v in vals]),
                "mean_positive_suppression_rate": safe_mean([v.get("baseline_blocker_positive_suppression_rate") for v in vals]),
                "mean_baseline_full_blocker_count": safe_mean([v.get("baseline_full_blocker_count") for v in vals]),
                "mean_after_full_blocker_count": safe_mean([v.get("after_full_blocker_count") for v in vals]),
                "mean_full_blocker_count_delta": safe_mean([v.get("full_blocker_count_delta") for v in vals]),
                "mean_resolved_baseline_blocker_rate": safe_mean([v.get("resolved_baseline_blocker_rate") for v in vals]),
                "mean_new_blocker_rate": safe_mean([v.get("new_blocker_rate") for v in vals]),
                "mean_identity_anchor_gap_improvement": safe_mean([v.get("identity_anchor_gap_improvement") for v in vals]),
                "token_closure_gain_rate": safe_rate([v.get("token_closure_gain") for v in vals]),
                "mean_target_neutral_suppressor_score": safe_mean([v.get("target_neutral_suppressor_score") for v in vals]),
                "label_counts": dict(Counter(v.get("phase801_label") for v in vals)),
                "baseline_blocker_class_counts": merge_counter_dicts(vals, "baseline_blocker_class_counts"),
                "baseline_blocker_class_mean_suppression": mean_nested_metric(vals, "baseline_blocker_class_mean_suppression"),
            }
        )
        bs = safe_float(payload["mean_baseline_blocker_suppression"]) or 0.0
        rr = safe_float(payload["mean_resolved_baseline_blocker_rate"]) or 0.0
        nr = safe_float(payload["mean_new_blocker_rate"]) or 0.0
        tg = abs(safe_float(payload["mean_target_logit_gain"]) or 0.0)
        payload["group_target_neutral_score"] = max(bs, 0.0) * rr * max(1.0 - nr, 0.0) / (1.0 + tg)
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("projection_mode") == "target_neutral",
            r.get("group_target_neutral_score") or -999.0,
            r.get("mean_baseline_blocker_suppression") or -999.0,
        ),
        reverse=True,
    )
    return out


def triplet_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    fields = ["model", "case_id", "route_id", "route_component_signature", "compare_variant"]
    for row in rows:
        key = tuple(row.get(f) for f in fields)
        groups[key][str(row.get("projection_mode"))] = row
    out = []
    for key, by_mode in groups.items():
        raw = by_mode.get("raw")
        neutral = by_mode.get("target_neutral")
        target_only = by_mode.get("target_only")
        if not raw or not neutral:
            continue
        raw_supp = safe_float(raw.get("baseline_blocker_mean_suppression")) or 0.0
        neutral_supp = safe_float(neutral.get("baseline_blocker_mean_suppression")) or 0.0
        raw_tg = safe_float(raw.get("target_logit_gain")) or 0.0
        neutral_tg = safe_float(neutral.get("target_logit_gain")) or 0.0
        target_only_tg = safe_float(target_only.get("target_logit_gain")) if target_only else None
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "raw_target_gain": raw_tg,
                "neutral_target_gain": neutral_tg,
                "target_only_target_gain": target_only_tg,
                "raw_blocker_suppression": raw_supp,
                "neutral_blocker_suppression": neutral_supp,
                "target_only_blocker_suppression": target_only.get("baseline_blocker_mean_suppression") if target_only else None,
                "neutral_suppression_retention": (neutral_supp / raw_supp) if abs(raw_supp) > 1e-9 else None,
                "target_gain_removed_by_projection": raw_tg - neutral_tg,
                "neutral_new_blocker_rate": neutral.get("new_blocker_rate"),
                "neutral_resolved_rate": neutral.get("resolved_baseline_blocker_rate"),
                "neutral_target_neutral_suppressor_score": neutral.get("target_neutral_suppressor_score"),
                "neutral_label": neutral.get("phase801_label"),
                "raw_label": raw.get("phase801_label"),
            }
        )
        payload["target_neutral_pass"] = bool(
            abs(neutral_tg) <= float(args.target_neutral_tolerance)
            and neutral_supp > float(args.min_neutral_suppression)
            and (safe_float(neutral.get("new_blocker_rate")) or 0.0) <= float(args.max_neutral_new_rate)
        )
        out.append(payload)
    out.sort(
        key=lambda r: (
            r.get("target_neutral_pass"),
            r.get("neutral_target_neutral_suppressor_score") or -999.0,
            r.get("neutral_blocker_suppression") or -999.0,
        ),
        reverse=True,
    )
    return out


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, attn_impl: str, routes: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = group_rows(rows, ["model"])
    by_projection = group_rows(rows, ["model", "projection_mode"])
    by_route_projection = group_rows(rows, ["model", "projection_mode", "route_component_signature"])
    triplets = triplet_rows(rows, args)
    return {
        "phase": 801,
        "title": "Target-Neutral Suppressor Causal Test",
        "model": args.model,
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attn_implementation": attn_impl,
        "dtype": "bfloat16",
        "quantization": "off",
        "n_rows": len(rows),
        "n_cases": len({r.get("case_id") for r in rows}),
        "n_routes": len(routes),
        "projection_modes": p796.parse_csv(args.projection_modes),
        "target_neutral_tolerance": args.target_neutral_tolerance,
        "routes": routes,
        "by_model": by_model,
        "by_projection": by_projection,
        "by_route_projection": by_route_projection[:80],
        "top_target_neutral_triplets": triplets[:80],
        "strict_boundary": (
            "This phase subtracts direct target-readout direction from hidden-state route deltas. "
            "It tests whether blocker suppression survives target-direction removal. "
            "It does not yet test Q/K/V/O internal spaces or neuron-level mechanisms."
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
        f"projection_modes={p796.parse_csv(args.projection_modes)}"
    )
    if args.dry_run:
        return {
            "model": args.model,
            "round": args.round_name,
            "selected_cases": len(selected),
            "routes": routes,
            "projection_modes": p796.parse_csv(args.projection_modes),
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
                log(f"{args.model}: target-neutral suppressor test {ci}/{len(selected)} cases; rows={len(rows)}")
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
    write_jsonl(out_dir / f"phase801_{args.model}_rows.jsonl", rows)
    write_json(out_dir / f"phase801_{args.model}_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "round": args.round_name,
                "attn": attn_impl,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "by_projection": summary["by_projection"],
                "top_target_neutral_triplets": summary["top_target_neutral_triplets"][:5],
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

    task = "phase801:target_neutral_suppressor_causal_test"
    node(task, "task", label="Phase 801 target-neutral suppressor causal test")
    for model_name, summary in payload.get("by_model", {}).items():
        model_node = f"model:{model_name}"
        node(model_node, "model", label=model_name)
        edges.append({"id": f"{task}->{model_node}", "source": task, "target": model_node, "type": "tested_model"})
        for row in summary.get("by_route_projection", [])[:30]:
            cand = f"{model_name}:{row.get('projection_mode')}:{row.get('route_component_signature')}"
            node(cand, "projection_route", label=row.get("route_component_signature"), metrics=row)
            edges.append(
                {
                    "id": f"{model_name}:projection:{row.get('projection_mode')}:{len(edges)}",
                    "source": model_node,
                    "target": cand,
                    "type": "target_neutral_projection_response",
                    "weight": row.get("group_target_neutral_score"),
                    "metrics": row,
                }
            )
    return {"schema_version": "atlas_graph_v1", "phase": 801, "graph": {"nodes": list(nodes.values()), "edges": edges}}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 801 Target-Neutral Suppressor Causal Test ({payload['round']})",
        "",
        f"- Status: `{payload['status']}`",
        "- Boundary: hidden-state route deltas are decomposed into raw, target-neutral, and target-only components.",
        "- This tests whether blocker suppression survives removal of the direct target-readout direction.",
        "",
        "## By Projection",
        "",
        "| model | projection | rows | cases | target gain | blocker suppression | resolved | new rate | neutral score | token gain | labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("by_projection", []):
            lines.append(
                f"| {model_name} | `{row.get('projection_mode')}` | {row.get('n')} | {row.get('case_n')} | "
                f"{fmt(row.get('mean_target_logit_gain'))} | {fmt(row.get('mean_baseline_blocker_suppression'))} | "
                f"{fmt(row.get('mean_resolved_baseline_blocker_rate'))} | {fmt(row.get('mean_new_blocker_rate'))} | "
                f"{fmt(row.get('group_target_neutral_score'))} | {fmt(row.get('token_closure_gain_rate'))} | "
                f"`{json.dumps(row.get('label_counts') or {}, ensure_ascii=False, sort_keys=True)}` |"
            )
    lines += [
        "",
        "## Top Target-Neutral Triplets",
        "",
        "| model | case | route | raw target | neutral target | raw suppress | neutral suppress | neutral new | neutral score | pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in MODELS:
        data = payload["by_model"].get(model_name)
        if not data:
            continue
        for row in data.get("top_target_neutral_triplets", [])[:20]:
            lines.append(
                f"| {model_name} | `{row.get('case_id')}` | `{row.get('route_component_signature')}` | "
                f"{fmt(row.get('raw_target_gain'))} | {fmt(row.get('neutral_target_gain'))} | "
                f"{fmt(row.get('raw_blocker_suppression'))} | {fmt(row.get('neutral_blocker_suppression'))} | "
                f"{fmt(row.get('neutral_new_blocker_rate'))} | {fmt(row.get('neutral_target_neutral_suppressor_score'))} | "
                f"{row.get('target_neutral_pass')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_round(round_name: str) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model_name in MODELS:
        path = RESULT_ROOT / round_name / f"phase801_{model_name}_summary.json"
        if path.exists():
            by_model[model_name] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "phase": 801,
        "round": round_name,
        "status": "complete" if len(by_model) == len(MODELS) else "partial",
        "models": list(by_model),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_model": by_model,
    }
    root = RESULT_ROOT / round_name
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "phase801_cross_model_summary.json", payload)
    write_json(root / "phase801_atlas_graph.json", build_atlas(payload))
    write_markdown(root / "phase801_cross_model_summary.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = p796.build_parser()
    parser.add_argument("--projection-modes", default="raw,target_neutral,target_only")
    parser.add_argument("--target-neutral-tolerance", type=float, default=0.75)
    parser.add_argument("--min-neutral-suppression", type=float, default=0.25)
    parser.add_argument("--max-neutral-new-rate", type=float, default=0.10)
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
